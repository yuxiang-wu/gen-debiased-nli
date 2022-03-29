"""
Loads the data from this paper [An Empirical Study on Model-agnostic Debiasing Strategies for Robust Natural Language Inference](https://arxiv.org/abs/2010.03777)
"""

import codecs
import json
import random
from collections import defaultdict
from typing import List, Dict

import numpy as np

from ...utils import simple_accuracy

label2id = {"entailment": 0, "neutral": 1, "contradiction": 2,
            "non-entailment": 1, "non-contradiction": 1}  # non-x are reduced to the neutral class
label_list = ["entailment", "neutral", "contradiction"]

Tri_dataset = ["IS_CS", "LI_LI", "PI_CD", "PI_SP", "ST_LM", "ST_NE", "ST_SE", "ST_WO"]
Ent_bin_dataset = ["IS_SD"]
Con_bin_dataset = ["LI_TS"]


def load_debias_nli(data_dir):
    """v2: random split each subset into two halves (instead of random split the entire dataset like v1)"""
    fsrc = codecs.open(f"{data_dir}/nli_debiasing_datasets/robust_nli.txt", "r",
                       encoding="utf-8").read().strip().split("\n")
    data_splits = defaultdict(list)
    for src in fsrc:
        example = json.loads(src)
        split, glabel_str = example['split'], example['label']

        label_id = label2id[glabel_str]
        # glabel_list = [glabel_str]
        # if split in Ent_bin_dataset:
        #     if glabel_str == "non-entailment":
        #         glabel_list = ["neutral", "contradiction"]
        # elif split in Con_bin_dataset:
        #     if glabel_str == "non-contradiction":
        #         glabel_list = ["entailment", "neutral"]

        new_example = {"premise": example["prem"], "hypothesis": example["hypo"], "label": label_id,
                       "glabel_str": glabel_str, "split": split}
        data_splits[split].append(new_example)

    random.seed(42)

    dev, test = [], []
    for split, subset in data_splits.items():
        random.shuffle(subset)
        for i, data in enumerate(subset):
            if i % 2 == 0:
                dev.append(data)
            test.append(data)

    return None, dev, test


def eval_debias_nli(predictions: np.ndarray, orig_examples: List[Dict]) -> Dict:
    pred_list = predictions.tolist()
    assert len(pred_list) == len(orig_examples), "prediction and examples length mismatch"

    split_preds, split_labels = defaultdict(list), defaultdict(list)
    for pred, example in zip(pred_list, orig_examples):
        split, label = example['split'], example['label']
        split_preds[split].append(pred)
        split_labels[split].append(label)

    # Convert to np.ndarray
    for split, preds in split_preds.items():
        split_preds[split] = np.array(preds)
    for split, labels in split_labels.items():
        split_labels[split] = np.array(labels)

    all_dataset = Tri_dataset + Ent_bin_dataset + Con_bin_dataset
    all_results = {}
    for split in all_dataset:
        preds = split_preds[split]
        labels = split_labels[split]

        if split in Ent_bin_dataset:
            preds[preds == 2] = 1
        elif split in Con_bin_dataset:
            preds[preds == 0] = 1

        acc = simple_accuracy(preds, labels)
        print("{} : {}".format(split, acc))
        all_results[split] = acc

    st_mean = np.mean([all_results["ST_LM"], all_results["ST_NE"], all_results["ST_SE"], all_results["ST_WO"]])
    all_results["ST_avg"] = st_mean

    avg = np.mean([st_mean, all_results["IS_CS"], all_results["LI_LI"], all_results["PI_CD"],
                   all_results["PI_SP"], all_results["IS_SD"], all_results["LI_TS"]])

    all_results["AVG"] = avg
    return all_results
