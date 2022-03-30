# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""  Train a NLI model on the concatenation of two datasets and evaluate on in-domain and out-of-domain datasets. """
import argparse
import logging
import math
import os
import random
from functools import partial

import datasets
import numpy as np
import torch
import transformers
# from accelerate import Accelerator
from datasets import load_metric
from nltk.tokenize.treebank import TreebankWordTokenizer
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    BertConfig,
    BertTokenizer,
    PretrainedConfig,
    get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version

from gen_debiased_nli.data.nli import load_all_nli_eval_datasets, eval_nli_predictions, get_nli_dataset
from gen_debiased_nli.robust_nli.utils_bert import BertDebiasForSequenceClassification
from gen_debiased_nli.utils import save_model, truncate_dataset, load_jsonl, convert_to_dataset

# from gen_debiased_nli.data.nli.robust_nli import eval_robust_nli

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")

# Initialise wandb
try:
    import wandb

    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn(
            "W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except (ImportError, AttributeError):
    _has_wandb = False


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--exp_name", type=str, default=None, required=True, help="Experiment name.")

    parser.add_argument("--train_data", type=str, default=None, help="Name or path of the training dataset.")
    parser.add_argument("--dev_data", type=str, default=None, help="Name or path of the dev set.")
    parser.add_argument("--num_samples", type=int, default=math.inf, help="Number of samples from the first dataset.")
    parser.add_argument("--data_dir", type=str, default="data", help="path to the data directory.")
    parser.add_argument("--validation_split_percentage", default=0.1,
                        help="The percentage of the train set used as validation set in case there's no validation split")

    parser.add_argument("--max_length", type=int, default=128,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded if `--pad_to_max_lengh` is passed.")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--early_stop_patience", type=int, default=3,
                        help="Early stop if the validation performance does not improve for this number of epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="The scheduler type to use.",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                 "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--do_train", action="store_true", help="Whether to train the model on the train set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")

    # # POE
    # parser.add_argument("--bias_model_path", type=str, required=True,
    #                     help="Path to pretrained model or model identifier from huggingface.co/models.")
    #
    # parser.add_argument("--lambda_h", default=1.0, type=float, help="Loss coefficient for the bias-only model")
    #
    # parser.add_argument("--poe_loss", action="store_true", help="Uses the product of the expert loss.")
    # parser.add_argument("--poe_alpha", default=1.0, type=float, help="alpha for poe method.")
    #
    # parser.add_argument("--focal_loss", action="store_true", help="Uses the focal loss for classification,\
    #        where instead of the probabilities of the objects, we use the h only probabilities")
    # parser.add_argument("--gamma_focal", type=float, default=2.0)
    #
    # parser.add_argument("--rubi", action="store_true", help="If specified use rubi network.")
    #

    # Arguments imported from Karimi's code
    parser.add_argument("--ensemble_training", action="store_true", help="Train the h-only and hans bias-only together\
        on MNLI.")  # RUBI parameters, this is deactivated by default.
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

    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


# Labels
label_list = ['entailment', 'neutral', 'contradiction']
num_labels = len(label_list)


def get_dataset_splits(args):
    def _del_columns(dataset):
        dataset = dataset.remove_columns(
            list(filter(lambda n: n not in ["premise", "hypothesis", "label"], dataset.column_names))
        )
        return dataset

    assert args.train_data is not None
    if os.path.exists(args.train_data):
        train = convert_to_dataset(load_jsonl(args.train_data))
    else:
        try:
            train, _, _ = get_nli_dataset(args.data_dir, args.train_data)
        except ValueError:
            train = None

    if args.dev_data is None:
        assert train is not None
        data_dict = train.train_test_split(test_size=args.validation_split_percentage, seed=args.seed)
        train, dev = data_dict["train"], data_dict["test"]
    elif os.path.exists(args.dev_data):
        dev = convert_to_dataset(load_jsonl(args.dev_data))
    else:
        try:
            _, dev, _ = get_nli_dataset(args.data_dir, args.dev_data)
        except ValueError:
            dev = None

    return _del_columns(train), _del_columns(dev), None


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if _has_wandb:
        wandb.init(project="ADG2", name=args.exp_name, dir=args.output_dir, config=vars(args))

    device = "cuda"

    # Load pretrained model and tokenizer
    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task="snli")
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)

    # Adds rubi parameters here.
    config.NUM_LABELS = num_labels
    config.rubi = args.rubi
    config.hans = args.hans
    config.nonlinear_h_classifier = args.nonlinear_h_classifier
    config.hypothesis_only = args.hypothesis_only
    config.lambda_h = args.lambda_h
    config.focal_loss = args.focal_loss
    config.poe_loss = args.poe_loss
    config.similarity = args.similarity
    config.gamma_focal = args.gamma_focal
    config.weighted_bias_only = args.weighted_bias_only
    config.length_features = args.length_features
    config.hans_features = args.hans_features
    config.hans_only = args.hans_only
    config.ensemble_training = args.ensemble_training
    config.aggregate_ensemble = args.aggregate_ensemble
    config.poe_alpha = args.poe_alpha

    model = BertDebiasForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    model.to(device)

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    # # Load the bias-only model (essentially a hypothesis-only model)
    # bias_config = AutoConfig.from_pretrained(args.bias_model_path, num_labels=num_labels)
    # bias_tokenizer = AutoTokenizer.from_pretrained(args.bias_model_path, use_fast=not args.use_slow_tokenizer)
    # bias_model = AutoModelForSequenceClassification.from_pretrained(
    #     args.bias_model_path,
    #     from_tf=bool(".ckpt" in args.bias_model_path),
    #     config=bias_config,
    # )
    # bias_model.to(device)

    # Preprocessing the datasets
    sentence1_key, sentence2_key = "premise", "hypothesis"

    # PennTreebank tokenizer
    ptb_tokenizer = TreebankWordTokenizer()

    def _ptb_tokenize(s):
        return " ".join(ptb_tokenizer.tokenize(s))

    def preprocess_function(examples, is_test=False, max_length=None):
        # Tokenize the texts
        texts = ([_ptb_tokenize(ex[sentence1_key]) for ex in examples],
                 [_ptb_tokenize(ex[sentence2_key]) for ex in examples])
        h_texts = ([_ptb_tokenize(ex[sentence2_key]) for ex in examples],)  # hypothesis-only input

        inputs = tokenizer(*texts, padding=True,
                           max_length=args.max_length if max_length is None else max_length,
                           truncation=True, return_tensors="pt")
        h_inputs = tokenizer(*h_texts, padding=True,
                             max_length=args.max_length if max_length is None else max_length,
                             truncation=True, return_tensors="pt")

        # output_dict = {"inputs": inputs, "h_inputs": h_inputs}
        output_dict = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"],
                       "token_type_ids": inputs["token_type_ids"],
                       "h_ids": h_inputs["input_ids"], "h_attention_mask": h_inputs["attention_mask"]}

        if "label" in examples[0]:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                output_dict["labels"] = torch.tensor([label_to_id[ex["label"]] for ex in examples], dtype=torch.long)
            else:
                # In all cases, rename the column to labels because the model will expect that.
                output_dict["labels"] = torch.tensor([ex["label"] for ex in examples], dtype=torch.long)

        if is_test:
            output_dict["examples"] = examples  # add the original examples for evaluation purpose

        return output_dict

    if args.do_train:
        # Get the datasets
        train_dataset, eval_dataset, _ = get_dataset_splits(args)

        train_dataset = truncate_dataset(train_dataset, args.num_samples)
        train_dataset = train_dataset.filter(lambda ex: 0 <= ex["label"] < num_labels)

        assert eval_dataset is not None
        eval_dataset = eval_dataset.filter(lambda ex: 0 <= ex["label"] < num_labels)

        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        # DataLoaders creation:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=preprocess_function, batch_size=args.per_device_train_batch_size
        )
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=preprocess_function, batch_size=args.per_device_eval_batch_size
        )

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # Prepare everything with our `accelerator`.
        # model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        #     model, optimizer, train_dataloader, eval_dataloader
        # )
        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        # Get the metric function
        metric = load_metric("accuracy")

        # Train!
        total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps))
        completed_steps = 0

        best_accuracy, patience = float(0.), args.early_stop_patience

        # if args.focal_loss:
        #     loss_fct = FocalLoss(gamma=args.gamma_focal)
        # elif args.poe_loss:
        #     loss_fct = POELoss(poe_alpha=args.poe_alpha)
        # elif args.rubi:
        #     loss_fct = RUBILoss(num_labels=num_labels)
        # else:
        #     raise ValueError("Debiasing loss is not specified.")
        # h_loss_fct = CrossEntropyLoss()

        for epoch in range(args.num_train_epochs):
            model.train()
            # bias_model.train()
            for step, batch in enumerate(train_dataloader):
                if step == 0 and epoch == 0:
                    print(batch)  # debug

                new_batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**new_batch)
                loss = outputs["bert"][0]

                # h_inputs = batch.pop("h_inputs")
                # new_h_inputs = {k: v.to(device) for k, v in h_inputs.items()}
                # h_outputs = bias_model(**new_h_inputs)

                # logits = outputs.logits
                # h_logits = h_outputs.logits
                # labels = batch.pop("labels").to(device)
                #
                # model_loss = loss_fct(logits.view(-1, num_labels), labels.view(-1), h_logits.view(-1, num_labels))
                # h_loss = h_loss_fct(h_logits.view(-1, num_labels), labels.view(-1))
                # loss = model_loss + args.lambda_h * h_loss

                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                # accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                    if _has_wandb:
                        wandb.log({"loss": loss * args.gradient_accumulation_steps, "step": completed_steps,
                                   "lr": lr_scheduler.get_last_lr()})

                if completed_steps >= args.max_train_steps:
                    break

            model.eval()
            # bias_model.eval()
            for step, batch in enumerate(eval_dataloader):
                new_batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**new_batch)["bert"]
                tmp_eval_loss, logits = outputs[:2]

                # inputs = batch.pop("inputs")
                # new_inputs = {k: v.to(device) for k, v in inputs.items()}
                # outputs = model(**new_inputs)  # only uses the unbiased model

                # preds = outputs.logits.argmax(dim=-1)
                preds = logits.argmax(dim=-1)
                labels = batch["labels"].to(device)

                metric.add_batch(predictions=preds, references=labels)
            eval_metric = metric.compute()

            logger.info(f"epoch {epoch} eval - {eval_metric}")
            if _has_wandb:
                log_metric = {f"eval_{k}": v for k, v in eval_metric.items()}
                log_metric["epoch"] = epoch
                wandb.log(log_metric)

            eval_metric_value = eval_metric["accuracy"]
            if eval_metric_value > best_accuracy:
                best_accuracy = eval_metric_value
                patience = args.early_stop_patience
                if args.output_dir is not None:
                    save_model(model, os.path.join(args.output_dir, "best_ckpt"), tokenizer=tokenizer)
            else:
                patience -= 1
                if patience <= 0:
                    break

            if args.output_dir is not None:
                save_model(model, os.path.join(args.output_dir, "latest_ckpt"), tokenizer=tokenizer)

    # Run predictions on the in-domain and out-of-domain data
    if args.do_predict:
        # Note: test on all NLI validation sets
        all_dev_datasets = [(name, f"{name}_val", data) for name, data in load_all_nli_eval_datasets(args.data_dir).items()]
        all_test_datasets = [(name, f"{name}_test", data) for name, data in
                             load_all_nli_eval_datasets(args.data_dir, return_test=True).items()]
        all_eval_datasets = all_dev_datasets + all_test_datasets

        best_ckpt_dir = os.path.join(args.output_dir, "best_ckpt")
        logger.info(f"Loading best checkpoint from {best_ckpt_dir}")
        best_model = BertDebiasForSequenceClassification.from_pretrained(best_ckpt_dir, from_tf=False, config=config)
        best_model.to(device)  # use single GPU rather than accelerator, because we need to pop examples

        for data_name, data_split_name, dataset in all_eval_datasets:
            if dataset is None:
                continue

            dataset = dataset.filter(lambda ex: 0 <= ex["label"] < num_labels)
            if "anli" in data_name:
                dataloader = DataLoader(dataset, collate_fn=partial(preprocess_function, is_test=True, max_length=512),
                                        batch_size=args.per_device_eval_batch_size)
            else:
                dataloader = DataLoader(dataset, collate_fn=partial(preprocess_function, is_test=True),
                                        batch_size=args.per_device_eval_batch_size)

            best_model.eval()

            all_examples = []
            all_preds, all_labels, all_probs = None, None, None
            for step, batch in enumerate(dataloader):
                examples = batch.pop("examples")
                all_examples.extend(examples)

                new_batch = {k: v.to(device) for k, v in batch.items()}
                outputs = best_model(**new_batch)["bert"]
                _, logits = outputs[:2]

                # inputs = batch.pop("inputs")
                # examples = batch.pop("examples")
                # all_examples.extend(examples)
                #
                # new_inputs = {k: v.to(device) for k, v in inputs.items()}
                # outputs = best_model(**new_inputs)

                preds = logits.argmax(dim=-1).detach().cpu().numpy()
                labels = batch["labels"].detach().cpu().numpy()
                probs = logits.softmax(-1).detach().cpu().numpy()

                all_preds = np.append(all_preds, preds) if all_preds is not None else preds
                all_labels = np.append(all_labels, labels) if all_labels is not None else labels
                all_probs = np.concatenate([all_probs, probs], axis=0) if all_probs is not None else probs

            results = eval_nli_predictions(all_preds, all_labels, probs=all_probs, examples=all_examples,
                                           dataset=data_name)

            logger.info(f"{data_split_name} accuracy: {results}")
            if _has_wandb:
                log_metric = {f"{data_split_name}_{k}": v for k, v in results.items()}
                wandb.log(log_metric)

            if args.output_dir is not None:
                with open(os.path.join(args.output_dir, "results"), "a") as fo:
                    fo.write(f"{data_split_name}: {results}\n")


if __name__ == "__main__":
    main()
