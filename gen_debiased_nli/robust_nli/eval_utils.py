import argparse
import glob
import logging
import os
from os.path import join

import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from tqdm import tqdm
from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer,
                          XLMConfig, XLMForSequenceClassification,
                          XLMTokenizer, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer)

from .utils_bert import BertDebiasForSequenceClassification
from .utils_glue import (compute_metrics, convert_examples_to_features,
                        processors)

MODEL_CLASSES = {
    'bert': (BertConfig, BertDebiasForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
}
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig)),
                 ())

task_to_data_dir = {
    "snli": "../../data/datasets/SNLI/original",
    "mnli": "../../data/datasets/MNLI",
    "mnli-mm": "../../data/datasets/MNLI",
    "addonerte": "../../data/datasets/AddOneRTE",
    "dpr": "../../data/datasets/DPR/",
    "sprl": "../../data/datasets/SPRL/",
    "fnplus": "../../data/datasets/FNPLUS/",
    "joci": "../../data/datasets/JOCI/",
    "mpe": "../../data/datasets/MPE/",
    "scitail": "../../data/datasets/SciTail/",
    "sick": "../../data/datasets/SICK/",
    "glue": "../../data/datasets/GLUEDiagnostic/",
    "QQP": "../../data/datasets/QQP/",
    "snlihard": "../../data/datasets/SNLIHard/",
    "MNLIMatchedHard": "../../data/datasets/MNLIMatchedHard/",
    "MNLIMismatchedHard": "../../data/datasets/MNLIMismatchedHard/",
    "mnlimatched": "../../data/datasets/MNLIMatched/",
    "mnlimismatched": "../../data/datasets/MNLIMismatched/",
    "fever": "../../data/datasets/FEVER/",
    "fever-symmetric-generated": "../../data/datasets/FEVER-symmetric-generated/",
    "MNLIMatchedHardWithHardTest": "../../data/datasets/MNLIMatchedHardWithHardTest/",
    "MNLIMismatchedHardWithHardTest": "../../data/datasets/MNLIMismatchedHardWithHardTest/",
    "MNLITrueMatched": "../../data/datasets/MNLITrueMatched",
    "MNLITrueMismatched": "../../data/datasets/MNLITrueMismatched",
    "HANS": "../../data/datasets/HANS",
    "HANS-const": "../../data/datasets/HANS/constituent",
    "HANS-lex": "../../data/datasets/HANS/lexical_overlap",
    "HANS-sub": "../../data/datasets/HANS/subsequence",
}

# All of these tasks use the NliProcessor # I added snli to this one as well.
nli_task_names = ["addonerte", "dpr", "sprl", "fnplus", "joci", "mpe", "scitail", "sick", "glue", "QQP", \
                  "snlihard", "mnlimatched", "mnlimismatched", "MNLIMatchedHardWithHardTest", \
                  "MNLIMismatchedHardWithHardTest", "MNLITrueMismatched", "MNLITrueMatched", "MNLIMatchedHard",
                  "MNLIMismatchedHard"]
actual_task_names = ["snli", "mnli", "mnli-mm"]
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.CRITICAL)


def get_parser():
    parser = argparse.ArgumentParser()
    # RUBI parameters, this is deactivated by default.
    parser.add_argument("--ensemble_training", action="store_true", help="Train the h-only and hans bias-only together\
        on MNLI.")
    parser.add_argument("--poe_alpha", default=1.0, type=float, help="alpha for poe method.")
    parser.add_argument("--aggregate_ensemble", choices=["mean"], default="mean",
                        help="When using ensemble training with focal loss, one can combine  the\
                             two bias only predictions with mean.")
    parser.add_argument("--hans_only", action="store_true")
    parser.add_argument("--weighted_bias_only", action="store_true", help="If specified bias-only\
                           model's loss is weighted. Only impacts hans.")
    parser.add_argument("--gamma_focal", type=float, default=2.0)
    parser.add_argument("--similarity", type=str, nargs="+", default=[], choices=["max", "mean", "min", "second_min"])
    parser.add_argument("--hans", action="store_true", help="If selected trains the bias-only with hans features.")
    parser.add_argument("--length_features", type=str, nargs="+", default=[], help="options are len-diff, log-len-diff")
    parser.add_argument("--hans_features", action="store_true",
                        help="If selected, computes the features for the hans experiment")
    parser.add_argument("--rubi_text", choices=["a", "b"], default="b")
    parser.add_argument("--poe_loss", action="store_true", help="Uses the product of the expert loss.")
    parser.add_argument("--focal_loss", action="store_true", help="Uses the focal loss for classification,\
           where instead of the probabilities of the objects, we use the h only probabilities")
    parser.add_argument("--lambda_h", default=1.0, type=float)
    parser.add_argument("--rubi", action="store_true", help="If specified use rubi network.")
    parser.add_argument("--hypothesis_only", action="store_true")
    parser.add_argument("--nonlinear_h_classifier", choices=["deep", None], default=None)
    parser.add_argument("--save_labels_file", type=str, default=None, \
                        help="If specified, saves the labels.")
    parser.add_argument("--output_label_format", type=str, default="kaggle", choices=["kaggle", "numpy"],
                        help="the format of saving the labels.")
    # Bert parameters.
    parser.add_argument("--outputfile", type=str, default=None, help="If specified, saves the results.")
    parser.add_argument("--binerize_eval", action="store_true",
                        help="If specified, it binerize the dataset. During eval")
    parser.add_argument("--use_cached_dataset", action="store_true", help="If specified will use the cached dataset")
    parser.add_argument("--model_type", default=None, type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str,  # required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--eval_task_names", nargs='+', type=str, default=[], \
                        help="list of the tasks to evaluate on them.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=100000,  # this was 10000  # 50
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=100000,  # this was 10000 # 50
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    return parser


# writes the labels in the kaggle format.
def write_in_kaggle_format(args, label_ids, gold_labels, save_labels_file, eval_task):
    # make a dictionary from the labels.
    labels_map = {}
    i = 0
    for label in gold_labels:
        labels_map[i] = label
        i = i + 1

    ids_file = join(task_to_data_dir[eval_task], "ids.test")
    ids = [line.strip('\n') for line in open(ids_file)]

    with open(save_labels_file, 'w') as f:
        f.write("pairID,gold_label\n")
        for i, l in enumerate(label_ids):
            label = labels_map[l]
            f.write("{0},{1}\n".format(ids[i], label))


def write_in_numpy_format(args, preds, save_labels_file):
    np.save(save_labels_file, preds)


def binarize_preds(preds):
    # maps the third label (neutral one) to first, which is contradiction.
    preds[preds == 2] = 0
    return preds


def load_and_cache_examples(args, task, tokenizer, evaluate=False, dev_evaluate=False):
    data_dir = task_to_data_dir[task]

    if task.startswith("fever"):
        processor = processors["fever"]()
    elif task in nli_task_names:
        processor = processors["nli"](data_dir)
    elif task in ["mnli"]:
        processor = processors["mnli"](hans=args.hans)
    elif task == "mnli-mm":
        processor = processors["mnli-mm"](hans=args.hans)
    elif task.startswith("HANS"):
        processor = processors["hans"](hans=args.hans)
    else:
        processor = processors[task]()

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    print("File is: ", cached_features_file)

    if False:  # os.path.exists(cached_features_file) and args.use_cached_dataset:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if dev_evaluate:  # and task in nli_task_names:
            examples = processor.get_validation_dev_examples(data_dir)
        else:
            examples = processor.get_dev_examples(data_dir) if evaluate else \
                processor.get_train_examples(data_dir)

        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, "classification",
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                rubi=args.rubi or args.hypothesis_only or args.focal_loss or args.poe_loss or args.hans_only,
                                                rubi_text=args.rubi_text,
                                                hans=(args.hans and not evaluate) or args.hans_only, \
                                                hans_features=args.hans_features)

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    if (args.hans and not evaluate) or args.hans_only:
        all_h_ids = torch.tensor([f.h_ids for f in features], dtype=torch.long)
        all_h_masks = torch.tensor([f.input_mask_h for f in features], dtype=torch.long)
        all_p_ids = torch.tensor([f.p_ids for f in features], dtype=torch.long)
        all_p_masks = torch.tensor([f.input_mask_p for f in features], dtype=torch.long)
        all_have_overlap = torch.tensor([f.have_overlap for f in features], dtype=torch.float)
        all_overlap_rate = torch.tensor([f.overlap_rate for f in features], dtype=torch.float)
        all_subsequence = torch.tensor([f.subsequence for f in features], dtype=torch.float)
        all_constituent = torch.tensor([f.constituent for f in features], dtype=torch.float)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, \
                                all_h_ids, all_h_masks, all_p_ids, all_p_masks, all_have_overlap, all_overlap_rate, \
                                all_subsequence, all_constituent)

    elif args.rubi or args.hypothesis_only or args.focal_loss or args.poe_loss:
        # Hypothesis representations.
        all_h_ids = torch.tensor([f.h_ids for f in features], dtype=torch.long)
        all_h_masks = torch.tensor([f.input_mask_h for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, \
                                all_h_ids, all_h_masks)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return dataset, processor.get_labels(), processor.num_classes


def get_batch_emebddings(model, args, input_ids, token_type_ids=None, attention_mask=None,
                         position_ids=None, head_mask=None, h_ids=None, h_attention_mask=None, labels=None):
    if args.hypothesis_only:
        outputs = model.bert(h_ids, token_type_ids=None, attention_mask=h_attention_mask)
        pooled_output = outputs[1]
    else:
        outputs = model.bert(input_ids, position_ids=position_ids, \
                             token_type_ids=token_type_ids, \
                             attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]
    return pooled_output


def get_embeddings(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    if "mnli" in args.task_name and "mnli-mm" not in args.task_name:
        args.eval_task_names.append("mnli-mm")

    results = {}
    for eval_task in args.eval_task_names:
        # eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, eval_labels, num_classes = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        print(eval_dataset)
        args.eval_batch_size = args.per_gpu_eval_batch_size
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        embeddings = []

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                if args.hypothesis_only or args.focal_loss or args.poe_loss:
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                              # XLM don't use segment_ids
                              'labels': batch[3],
                              'h_ids': batch[4],
                              'h_attention_mask': batch[5]}
                else:
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                              # XLM don't use segment_ids
                              'labels': batch[3]}

                embedding = get_batch_emebddings(model, args, **inputs)
                embeddings.append(embedding)

            results[eval_task] = torch.cat(embeddings, dim=0)

    return results


def evaluate(args, model, tokenizer, prefix="", dev_evaluate=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    if "mnli" in args.task_name and "mnli-mm" not in args.task_name:
        args.eval_task_names.append("mnli-mm")

    results = {}
    for eval_task in args.eval_task_names:
        # eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, eval_labels, num_classes = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True, \
                                                                         dev_evaluate=dev_evaluate)

        print("num_classes ", num_classes, "eval_labels ", eval_labels)

        print(eval_dataset)
        args.eval_batch_size = args.per_gpu_eval_batch_size
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                if args.hypothesis_only or args.focal_loss or args.poe_loss:
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                              # XLM don't use segment_ids
                              'labels': batch[3],
                              'h_ids': batch[4],
                              'h_attention_mask': batch[5]}
                elif args.hans_only:
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                              # XLM don't use segment_ids
                              'labels': batch[3],
                              'h_ids': batch[4],
                              'h_attention_mask': batch[5],
                              'p_ids': batch[6],
                              'p_attention_mask': batch[7],
                              'have_overlap': batch[8],
                              'overlap_rate': batch[9],
                              'subsequence': batch[10],
                              'constituent': batch[11]
                              }
                else:
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                              # XLM don't use segment_ids
                              'labels': batch[3]}

                outputs = model(**inputs)["bert"]
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps

        max_preds = np.argmax(preds, axis=1)

        # convert 1,2 labels to 1 in case of binary dataset.
        if num_classes == 2 and args.binerize_eval:
            max_preds = binarize_preds(max_preds)
            out_label_ids = binarize_preds(out_label_ids)

        if eval_task in nli_task_names:
            eval_task_metric = "nli"
        elif eval_task.startswith("fever"):
            eval_task_metric = "fever"
        elif eval_task.startswith("HANS"):
            eval_task_metric = "hans"
        else:
            eval_task_metric = eval_task

        result = compute_metrics(eval_task_metric, max_preds, out_label_ids)

        if args.save_labels_file is not None:
            save_labels_file = args.save_labels_file + "_" + eval_task
            if args.output_label_format == "kaggle":
                write_in_kaggle_format(args, max_preds, eval_labels, save_labels_file, eval_task)
            elif args.output_label_format == "numpy":
                write_in_numpy_format(args, preds, save_labels_file)

        results[eval_task] = result["acc"]
        if eval_task.startswith("HANS"):
            results[eval_task + "_not-entailment"] = result["acc_0"]
            results[eval_task + "_entailment"] = result["acc_1"]
        print("results is ", result, " eval_task ", eval_task)

    return results, preds


def do_evaluate(args, output_dir, tokenizer, model, config, return_embeddings=False, dev_evaluate=False):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(output_dir, do_lower_case=args.do_lower_case)
    checkpoints = [output_dir]
    results = []
    preds_list = []
    if args.eval_all_checkpoints:
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    logger.info("Evaluate the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
        global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
        model = model_class.from_pretrained(checkpoint)
        model.set_rubi(False)
        model.set_ensemble_training(False)
        if args.hans:
            model.set_hans(False)
            model.set_focal_loss(False)
            model.set_poe_loss(False)
        if args.hans_only:
            model.set_hans(True)

        model.to(args.device)
        if return_embeddings:
            result = get_embeddings(args, model, tokenizer)
        else:
            result, preds = evaluate(args, model, tokenizer, prefix=global_step, dev_evaluate=dev_evaluate)
            preds_list.append(preds)
        results.append(result)

    if return_embeddings:
        return results
    else:
        return results, preds_list
