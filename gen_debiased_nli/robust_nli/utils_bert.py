import sys

import torch
from torch import nn

sys.path.append("../")

from torch.nn import CrossEntropyLoss, MSELoss
# from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
from transformers import BertPreTrainedModel, BertModel

from .losses import FocalLoss, POELoss, RUBILoss
from .utils_glue import get_word_similarity_new, get_length_features
from .mutils import grad_mul_const


class BertDebiasForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, config):
        super(BertDebiasForSequenceClassification, self).__init__(config)
        self.num_labels = config.NUM_LABELS
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.NUM_LABELS)
        # self.apply(self.init_weights)
        self.init_weights()

        self.config = config
        self.hypothesis_only = self.get_bool_value(config, "hypothesis_only")
        self.gamma_focal = config.gamma_focal if hasattr(config, "gamma_focal") else 2
        self.ensemble_training = self.get_bool_value(config, "ensemble_training")
        self.poe_alpha = config.poe_alpha if hasattr(config, 'poe_alpha') else 1

        # Sets the rubi parameters.
        self.similarity = self.get_list_value(config, "similarity")
        self.rubi = self.get_bool_value(config, 'rubi')
        self.hans = self.get_bool_value(config, 'hans')
        self.hans_features = self.get_bool_value(config, 'hans_features')
        self.focal_loss = self.get_bool_value(config, 'focal_loss')
        self.length_features = self.get_list_value(config, "length_features")
        self.hans_only = self.get_bool_value(config, 'hans_only')
        self.aggregate_ensemble = self.get_str_value(config, 'aggregate_ensemble')
        self.poe_loss = self.get_bool_value(config, 'poe_loss')
        self.weighted_bias_only = self.get_bool_value(config, "weighted_bias_only")

        num_labels_bias_only = self.config.NUM_LABELS
        if self.rubi or self.hypothesis_only or self.focal_loss or self.poe_loss or self.hans_only:
            if self.hans:
                num_features = 4 + len(self.similarity)

                if self.hans_features:
                    num_features += len(self.length_features)

                if not config.nonlinear_h_classifier:
                    self.h_classifier1 = nn.Linear(num_features, num_labels_bias_only)
                else:
                    self.h_classifier1 = nn.Sequential(
                        nn.Linear(num_features, num_features),
                        nn.Tanh(),
                        nn.Linear(num_features, num_features),
                        nn.Tanh(),
                        nn.Linear(num_features, num_labels_bias_only))

                if self.ensemble_training:
                    self.h_classifier1_second = self.get_classifier(config, config.nonlinear_h_classifier,
                                                                    num_labels_bias_only)
            else:
                # Loads the classifiers from the pretrained model.
                self.h_classifier1 = self.get_classifier(config, config.nonlinear_h_classifier, num_labels_bias_only)

            self.lambda_h = config.lambda_h

    def get_bool_value(self, config, attribute):
        return True if hasattr(config, attribute) and eval('config.' + attribute) else False

    def get_str_value(self, config, attribute):
        return eval('config.' + attribute) if hasattr(config, attribute) else ""

    def get_list_value(self, config, attribute):
        return eval('config.' + attribute) if hasattr(config, attribute) else []

    def get_classifier(self, config, nonlinear, num_labels):
        if nonlinear == "deep":
            classifier = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, int(config.hidden_size / 2)),
                nn.Tanh(),
                nn.Linear(int(config.hidden_size / 2), int(config.hidden_size / 4)),
                nn.Tanh(),
                nn.Linear(int(config.hidden_size / 4), num_labels),
            )
        else:
            classifier = nn.Linear(config.hidden_size, num_labels)
        return classifier

    def set_ensemble_training(self, ensemble_training):
        self.ensemble_training = ensemble_training

    def set_hans(self, hans):
        self.hans = hans

    def set_rubi(self, rubi):
        self.rubi = rubi

    def set_poe_loss(self, poe_loss):
        self.poe_loss = poe_loss

    def set_focal_loss(self, focal_loss):
        self.focal_loss = focal_loss

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, h_ids=None,
                h_attention_mask=None, p_ids=None, p_attention_mask=None, have_overlap=None,
                overlap_rate=None, subsequence=None, constituent=None, binary_labels=None):

        if self.hypothesis_only:
            outputs = self.bert(h_ids, token_type_ids=None, attention_mask=h_attention_mask)
            pooled_h = outputs[1]
            pooled_h_g = self.dropout(pooled_h)
            logits = self.h_classifier1(pooled_h_g)
            outputs = (logits,) + outputs[2:]
        elif not self.hans_only:
            outputs = self.bert(input_ids, position_ids=position_ids, \
                                token_type_ids=token_type_ids, \
                                attention_mask=attention_mask, head_mask=head_mask)
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            # add hidden states and attention if they are here
            outputs = (logits,) + outputs[2:]

        if self.hans:  # if both are correct.
            h_outputs = self.bert(h_ids, token_type_ids=None, attention_mask=h_attention_mask)

            if self.ensemble_training:  # also computes the h-only results.
                pooled_h_second = h_outputs[1]
                h_embd_second = grad_mul_const(pooled_h_second, 0.0)
                pooled_h_g_second = self.dropout(h_embd_second)
                h_logits_second = self.h_classifier1_second(pooled_h_g_second)
                h_outputs_second = (h_logits_second,) + h_outputs[2:]

            h_matrix = h_outputs[0]
            h_matrix = grad_mul_const(h_matrix, 0.0)
            h_matrix = self.dropout(h_matrix)

            p_outputs = self.bert(p_ids, token_type_ids=None, attention_mask=p_attention_mask)
            p_matrix = p_outputs[0]
            p_matrix = grad_mul_const(p_matrix, 0.0)
            p_matrix = self.dropout(p_matrix)

            # compute similarity features.
            if self.hans_features:
                simialrity_score = get_word_similarity_new(h_matrix, p_matrix, self.similarity, \
                                                           h_attention_mask, p_attention_mask)

            # this is the default case.
            hans_h_inputs = torch.cat((simialrity_score, \
                                       have_overlap.view(-1, 1), overlap_rate.view(-1, 1), subsequence.view(-1, 1),
                                       constituent.view(-1, 1)), 1)

            if self.hans_features and len(self.length_features) != 0:
                length_features = get_length_features(p_attention_mask, h_attention_mask, self.length_features)
                hans_h_inputs = torch.cat((hans_h_inputs, length_features), 1)

            h_logits = self.h_classifier1(hans_h_inputs)
            h_outputs = (h_logits,) + h_outputs[2:]

            if self.hans_only:
                logits = h_logits
                # overwrite outputs.
                outputs = h_outputs


        elif self.focal_loss or self.poe_loss or self.rubi:
            h_outputs = self.bert(h_ids, token_type_ids=None, attention_mask=h_attention_mask)
            pooled_h = h_outputs[1]
            h_embd = grad_mul_const(pooled_h, 0.0)
            pooled_h_g = self.dropout(h_embd)
            h_logits = self.h_classifier1(pooled_h_g)
            h_outputs = (h_logits,) + h_outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if self.focal_loss:
                    loss_fct = FocalLoss(gamma=self.gamma_focal, \
                                         ensemble_training=self.ensemble_training,
                                         aggregate_ensemble=self.aggregate_ensemble)
                elif self.poe_loss:
                    loss_fct = POELoss(ensemble_training=self.ensemble_training, poe_alpha=self.poe_alpha)
                elif self.rubi:
                    loss_fct = RUBILoss(num_labels=self.num_labels)
                elif self.hans_only:
                    if self.weighted_bias_only and self.hans:
                        weights = torch.tensor([0.5, 1.0, 0.5]).cuda()
                        loss_fct = CrossEntropyLoss(weight=weights)
                else:
                    loss_fct = CrossEntropyLoss()

                if self.rubi or self.focal_loss or self.poe_loss:
                    if self.ensemble_training:
                        model_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1), \
                                              h_logits.view(-1, self.num_labels),
                                              h_logits_second.view(-1, self.num_labels))
                    else:
                        model_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1), \
                                              h_logits.view(-1, self.num_labels))

                    if self.weighted_bias_only and self.hans:
                        weights = torch.tensor([0.5, 1.0, 0.5]).cuda()
                        h_loss_fct = CrossEntropyLoss(weight=weights)
                        if self.ensemble_training:
                            h_loss_fct_second = CrossEntropyLoss()
                    else:
                        h_loss_fct = CrossEntropyLoss()

                    h_loss = h_loss_fct(h_logits.view(-1, self.num_labels), labels.view(-1))
                    if self.ensemble_training:
                        h_loss += h_loss_fct_second(h_logits_second.view(-1, self.num_labels), labels.view(-1))

                    loss = model_loss + self.lambda_h * h_loss
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        all_outputs = {}
        all_outputs["bert"] = outputs
        if self.rubi or self.focal_loss or self.poe_loss:
            all_outputs["h"] = h_outputs
        if self.ensemble_training:
            all_outputs["h_second"] = h_outputs_second
        return all_outputs  # (loss), logits, (hidden_states), (attentions)
