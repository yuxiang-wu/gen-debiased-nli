# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open
from os.path import join

import jsonlines
import numpy as np
import torch
import torch.nn.functional as f
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

from .heuristics_utils import have_lexical_overlap, is_subsequence, is_constituent

logger = logging.getLogger(__name__)


def dot_product_matrix_attention(matrix_1, matrix_2):
    return matrix_1.bmm(matrix_2.transpose(2, 1))


def get_emb(tokens, word_vec):
    matrix = np.zeros((len(tokens), 300))
    for i, p in enumerate(tokens):
        matrix[i, :] = word_vec[p]
    return matrix


def get_word_similarity_new(prem_matrix, hyp_matrix, scores, h_mask, p_mask):
    # normalize the token embeddings.
    # [8, 64, 768]
    prem_matrix = f.normalize(prem_matrix, p=2, dim=2)
    hyp_matrix = f.normalize(hyp_matrix, p=2, dim=2)

    prem_matrix = prem_matrix * p_mask.view(prem_matrix.shape[0], prem_matrix.shape[1], 1).float()
    hyp_matrix = hyp_matrix * h_mask.view(hyp_matrix.shape[0], hyp_matrix.shape[1], 1).float()

    similarity_matrix = hyp_matrix.bmm(prem_matrix.transpose(2, 1))  # batch_size*seqlen(h)*seqlen(p)
    similarity = torch.max(similarity_matrix, 2)[0]  # batch_size*seqlen => hsize

    sim_score = []
    if "min" in scores or "second_min" in scores:
        # compute the min and second min in the similarities.
        similarity_replace = similarity.clone()
        # all the similarity values are smaller than 1 so 10 is a good number
        # so that the masked elements are not selected during the top minimum computations.
        similarity_replace[h_mask == 0] = 10
        y, i = torch.topk(similarity_replace, k=2, dim=1, largest=False, sorted=True)
        if "min" in scores:
            sim_score.append(y[:, 0].view(-1, 1))
        if "second_min" in scores:
            sim_score.append(y[:, 1].view(-1, 1))
    if "mean" in scores:
        h_lens = torch.sum(h_mask, 1)
        # note that to account for zero values, we have to consider the length not
        # getting mean.
        sum_similarity = torch.sum(similarity, 1)
        mean_similarity = sum_similarity / h_lens.float()
        sim_score.append(mean_similarity.view(-1, 1))
    if "max" in scores:
        max_similarity = torch.max(similarity, 1)[0]
        sim_score.append(max_similarity.view(-1, 1))

    similarity_score = torch.cat(sim_score, 1)
    return similarity_score


def get_length_features(p_mask, h_mask, length_features):
    features = []
    p_lengths = torch.sum(p_mask, dim=1)
    h_lengths = torch.sum(h_mask, dim=1)
    if "log-len-diff" in length_features:
        features.append((torch.log(torch.max((p_lengths - h_lengths), torch.ones_like(p_lengths)).float())).view(-1, 1))
    if "len-diff" in length_features:
        features.append((p_lengths - h_lengths).float().view(-1, 1))
    return torch.cat(features, 1)


def get_hans_features(premise, hypothesis, parse):
    constituent = is_constituent(premise, hypothesis, parse)
    subsequence = is_subsequence(premise, hypothesis)
    lexical_overlap, overlap_rate = have_lexical_overlap(premise, hypothesis)
    return constituent, subsequence, lexical_overlap, overlap_rate


def get_hans_features_new(premise, hypothesis, parse, tokenizer):
    premise_tokens = tokenizer.tokenize(premise)
    hyp_tokens = tokenizer.tokenize(hypothesis)
    premise_tokens = [p.lower() for p in premise_tokens]
    hyp_tokens = [h.lower() for h in hyp_tokens]
    premise_tokens = " ".join(premise_tokens)
    hyp_tokens = " ".join(hyp_tokens)
    constituent = is_constituent(premise_tokens, hyp_tokens, parse)
    subsequence = is_subsequence(premise_tokens, hyp_tokens)
    lexical_overlap, overlap_rate = have_lexical_overlap(premise_tokens, hyp_tokens, get_hans_new_features=True)
    return constituent, subsequence, lexical_overlap, overlap_rate


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, parse=None, binary_label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.parse = parse
        self.binary_label = binary_label


class RUBIInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, h_ids, input_mask_h):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.h_ids = h_ids
        self.input_mask_h = input_mask_h


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class HansInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, h_ids, input_mask_h,
                 p_ids, input_mask_p, have_overlap, overlap_rate, subsequence, constituent, binary_label=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.h_ids = h_ids
        self.input_mask_h = input_mask_h
        self.p_ids = p_ids
        self.input_mask_p = input_mask_p
        self.have_overlap = have_overlap
        self.overlap_rate = overlap_rate
        self.subsequence = subsequence
        self.constituent = constituent
        self.binary_label = binary_label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_jsonl(cls, filepath):
        """ Reads the jsonl file path. """
        lines = []
        with jsonlines.open(filepath) as f:
            for line in f:
                lines.append(line)
        return lines


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def __init__(self, hans=False):
        # It joins the other two label to one label.
        self.num_classes = 3
        self.hans = hans

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_dev_labels(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "dev_matched.tsv"))
        labels = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            label = line[-1]
            labels.append(label)
        return np.array(labels)

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            text_a = line[8]
            text_b = line[9]
            label = line[-1]

            if self.hans:
                parse = line[6]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, parse=parse))
            else:
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")

    def get_dev_labels(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv"))
        labels = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            label = line[-1]
            labels.append(label)
        return np.array(labels)


class SnliProcessor(DataProcessor):
    """Processor for the SNLI data set (GLUE version)."""

    def __init__(self):
        self.num_classes = 3

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_validation_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def get_dev_labels(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "test.tsv"))
        labels = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            label = line[-1]
            labels.append(label)
        return np.array(labels)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class NliProcessor(DataProcessor):
    """Processor for the dataset of the format of SNLI
    (InferSent version), could be 2 or 3 classes."""

    # We use get_labels() class to convert the labels to indices,
    # later during the transfer it will be problematic if the labels
    # are not the same order as the SNLI/MNLI so we return the whole
    # 3 labels, but for getting the actual number of classes, we use
    # self.num_classes.

    def __init__(self, data_dir):
        # We assume there is a training file there and we read labels from there.
        labels = [line.rstrip() for line in open(join(data_dir, 'labels.train'))]
        self.labels = list(set(labels))
        labels = ["contradiction", "entailment", "neutral"]
        ordered_labels = []
        for l in labels:
            if l in self.labels:
                ordered_labels.append(l)
        self.labels = ordered_labels
        self.num_classes = len(self.labels)

    def get_dev_labels(self, data_dir):
        labels = [line.rstrip() for line in open(join(data_dir, 'labels.test'))]
        return np.array(labels)

    def get_validation_dev_examples(self, data_dir):
        return self._create_examples(data_dir, "dev")

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, data_dir, set_type):
        """Creates examples for the training and dev sets."""
        s1s = [line.rstrip() for line in open(join(data_dir, 's1.' + set_type))]
        s2s = [line.rstrip() for line in open(join(data_dir, 's2.' + set_type))]
        labels = [line.rstrip() for line in open(join(data_dir, 'labels.' + set_type))]

        examples = []
        for (i, line) in enumerate(s1s):
            guid = "%s-%s" % (set_type, i)
            text_a = s1s[i]
            text_b = s2s[i]
            label = labels[i]
            # In case of hidden labels, changes it with entailment.
            if label == "hidden":
                label = "entailment"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class FEVERProcessor(DataProcessor):
    """Processor for the processed FEVER dataset."""

    def __init__(self):
        self.num_classes = 3

    def read_jsonl(self, filepath):
        """ Reads the jsonl file path. """
        lines = []
        with jsonlines.open(filepath) as f:
            for line in f:
                lines.append(line)
        return lines

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples( \
            self.read_jsonl(join(data_dir, "nli.train.jsonl")), \
            "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples( \
            self.read_jsonl(join(data_dir, "nli.dev.jsonl")), \
            "dev")

    def get_labels(self):
        """See base class."""
        return ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

    def _create_examples(self, items, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, item) in enumerate(items):
            guid = "%s-%s" % (set_type, i)
            # Claim has artifacts so this needs to be text_b.
            text_a = items[i]["claim"]
            text_b = items[i]["evidence"] if "evidence" in items[i] else items[i]["evidence_sentence"]
            label = items[i]["gold_label"] if "gold_label" in items[i] else items[i]["label"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class HansProcessor(DataProcessor):
    """Processor for the processed Hans dataset."""

    def __init__(self, hans=False):
        self.num_classes = 2
        self.hans = hans  # this is added only to test hans-only classifier on HANS dataset.

    def read_jsonl(self, filepath):
        """ Reads the jsonl file path. """
        lines = []
        with jsonlines.open(filepath) as f:
            for line in f:
                lines.append(line)
        return lines

    def get_train_examples(self, data_dir):
        """See base class."""
        pass

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples( \
            self._read_tsv(join(data_dir, "heuristics_evaluation_set.txt")), \
            "dev")

    def get_dev_labels(self, data_dir):
        items = self._read_tsv(os.path.join(data_dir, "heuristics_evaluation_set.txt"))
        labels = []
        for (i, item) in enumerate(items):
            if i == 0:
                continue
            label = items[i][0]
            labels.append(label)
        return np.array(labels)

    def get_labels(self):
        """See base class."""
        return ["non-entailment", "entailment"]

    def _create_examples(self, items, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, item) in enumerate(items):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            # Claim has artifacts so this needs to be text_b.
            text_a = items[i][5]
            text_b = items[i][6]
            label = items[i][0]

            if self.hans:
                parse = items[i][3]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, parse=parse))
            else:
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True, rubi=False, rubi_text="b",
                                 hans=False, hans_features=False):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        rubi: In case of having this option, it also adds on the hypothesis only examples
        to the dataset created.
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        if rubi:
            tokens_h = tokenizer.tokenize(example.text_b if rubi_text == "b" else example.text_a)
            half_max_seq_length = int(max_seq_length / 2)
            if len(tokens_h) > (half_max_seq_length - 2):
                tokens_h = tokens_h[:(half_max_seq_length - 2)]
            tokens_h = ["[CLS]"] + tokens_h + ["[SEP]"]
            h_ids = tokenizer.convert_tokens_to_ids(tokens_h)
            input_mask_h = [1] * len(h_ids)
            padding_h = [0] * (half_max_seq_length - len(h_ids))
            h_ids += padding_h
            input_mask_h += padding_h
            assert len(h_ids) == half_max_seq_length
            assert len(input_mask_h) == half_max_seq_length

            if hans:  # this is only for rubi, so only compute this for p
                def get_ids_mask(text, max_seq_length):
                    tokens_h = tokenizer.tokenize(text)
                    half_max_seq_length = int(max_seq_length / 2)
                    if len(tokens_h) > (half_max_seq_length - 2):
                        tokens_h = tokens_h[:(half_max_seq_length - 2)]
                    tokens_h = ["[CLS]"] + tokens_h + ["[SEP]"]
                    h_ids = tokenizer.convert_tokens_to_ids(tokens_h)
                    input_mask_h = [1] * len(h_ids)
                    padding_h = [0] * (half_max_seq_length - len(h_ids))
                    h_ids += padding_h
                    input_mask_h += padding_h
                    assert len(h_ids) == half_max_seq_length
                    assert len(input_mask_h) == half_max_seq_length
                    return h_ids, input_mask_h

                p_ids, input_mask_p = get_ids_mask(example.text_a if rubi_text == "b" else example.text_b,
                                                   max_seq_length)
                if hans_features:
                    have_overlap, constituent, subsequence, overlap_rate = get_hans_features_new(example.text_a,
                                                                                                 example.text_b,
                                                                                                 example.parse,
                                                                                                 tokenizer)

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        if rubi:
            if hans:
                features.append(
                    HansInputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=label_id,
                                      h_ids=h_ids,
                                      input_mask_h=input_mask_h,
                                      p_ids=p_ids,
                                      input_mask_p=input_mask_p,
                                      have_overlap=have_overlap,
                                      overlap_rate=overlap_rate,
                                      subsequence=subsequence,
                                      constituent=constituent,
                                      ))
            else:
                features.append(
                    RUBIInputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=label_id,
                                      h_ids=h_ids,
                                      input_mask_h=input_mask_h))
        else:
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def per_class_accuracy(preds, labels):
    unique_labels = np.unique(labels)
    results = {}
    for l in unique_labels:
        indices = (l == labels)
        acc = (preds[indices] == labels[indices]).mean()
        results["acc_" + str(int(l))] = acc
    acc = (preds == labels).mean()
    results["acc"] = acc
    return results


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "snli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "nli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "fever":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "hans":
        return per_class_accuracy(preds, labels)
    else:
        raise KeyError(task_name)


processors = {
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "snli": SnliProcessor,
    "nli": NliProcessor,
    "fever": FEVERProcessor,
    "hans": HansProcessor,
}

output_modes = {
    "mnli": "classification",
    "mnli-mm": "classification",
    "snli": "classification",
    "nli": "classification",
    "fever": "classification",
    "hans": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "mnli": 3,
    "mnli-mm": 3,
    "snli": 3,
    "fever": 3,
    "hans": 2,
}
