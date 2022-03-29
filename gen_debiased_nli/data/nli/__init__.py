"""
Load various NLI datasets for evaluation. The datasets are mainly from works related to these repositories:
- https://github.com/joestacey/robust-nli/blob/main/data.py
- https://github.com/azpoliak/hypothesis-only-NLI
- https://github.com/tyliupku/nli-debiasing-datasets
- https://github.com/tommccoy1/hans/
"""

import os
from typing import List, Dict
import numpy as np

from .add_one_rte import load_add_one_rte
from .anli import load_anli_r1, load_anli_r2, load_anli_r3, load_anli_all
from .debias_nli import load_debias_nli, eval_debias_nli
from .glue_diagnostic import load_glue_diagnostic
from .hans import load_hans
from .joci import load_joci
from .mnli import load_mnli, load_mnli_matched, load_mnli_mismatched
from .mnli_hard import load_mnli_matched_hard, load_mnli_mismatched_hard
from .mpe import load_mpe
from .qqp import load_qqp
from .recast_white import load_dpr, load_sprl, load_fnplus
from .scitail import load_scitail
from .sick import load_sick
from .snli import load_snli
from .snli_hard import load_snli_hard
from ...utils import simple_accuracy, convert_to_dataset

LABEL_LIST = ["entailment", "neutral", "contradiction"]

COMMON_NLI = [("snli", load_snli), ("mnli", load_mnli)]

HYPOTHESIS_ONLY_NLI = [("add1", load_add_one_rte), ("scitail", load_scitail),
                       ("sick", load_sick), ("mpe", load_mpe), ("joci", load_joci),
                       ("snli-hard", load_snli_hard), ("dpr", load_dpr), ("sprl", load_sprl),
                       ("fnplus", load_fnplus), ("glue", load_glue_diagnostic),
                       ("mnli-match", load_mnli_matched), ("mnli-mismatch", load_mnli_mismatched),
                       ("qqp", load_qqp)]
HARDER_NLI = [("hans", load_hans), ("anli-r1", load_anli_r1), ("anli-r2", load_anli_r2),
              ("anli-r3", load_anli_r3), ("anli-all", load_anli_all), ("debias-nli", load_debias_nli),
              ("mnli-matched-hard", load_mnli_matched_hard), ("mnli-mismatched-hard", load_mnli_mismatched_hard)]

ALL_NLI = COMMON_NLI + HYPOTHESIS_ONLY_NLI + HARDER_NLI
ALL_NLI_NAMES = [data[0] for data in ALL_NLI]
eps = 1e-9

BINARY_CLS_DS = ["add1", "dpr", "sprl", "fnplus", "scitail", "qqp", "hans"]


def get_nli_dataset(data_dir, name):
    assert os.path.isdir(data_dir)

    all_nli_dict = {n: f for n, f in ALL_NLI}
    if name in all_nli_dict:
        load_f = all_nli_dict[name]
        train_data, dev_data, test_data = load_f(data_dir)
    else:
        raise ValueError(f"Cannot find {name} in the NLI datasets.")

    return convert_to_dataset(train_data), convert_to_dataset(dev_data), convert_to_dataset(test_data)


def load_all_nli_datasets(data_dir):
    assert os.path.isdir(data_dir)
    all_train_data, all_dev_data, all_test_data = {}, {}, {}

    for name, load_f in ALL_NLI:  # data folders are in data root directory
        train_data, dev_data, test_data = load_f(data_dir)
        all_train_data[name] = convert_to_dataset(train_data)
        all_dev_data[name] = convert_to_dataset(dev_data)
        all_test_data[name] = convert_to_dataset(test_data)

    return all_train_data, all_dev_data, all_test_data


def load_all_nli_eval_datasets(data_dir, return_test=False):
    """Load all the out-of-domain evaluation datasets."""
    assert os.path.isdir(data_dir)
    _, dev_data_dict, test_data_dict = load_all_nli_datasets(data_dir)

    if not return_test:
        return dev_data_dict
    else:
        return test_data_dict


def binarize_preds(preds):
    # maps the 2 label (contradiction) to 1, which is neutral.
    preds[preds == 2] = 1
    return preds


def binarize_probs(probs):
    # sum the second (neutral) and third (contradiction) column
    new_probs = np.zeros((probs.shape[0], 2), dtype=np.float)
    new_probs[:, 0] = probs[:, 0]
    new_probs[:, 1] = probs[:, 1] + probs[:, 2]
    return new_probs


def eval_nli_predictions(predictions: np.ndarray, labels: np.ndarray, probs: np.ndarray = None,
                         examples: List[Dict] = None, dataset=None) -> Dict:
    if dataset == "debias-nli":
        return eval_debias_nli(predictions, examples)

    if dataset in BINARY_CLS_DS:
        if probs is None:
            predictions = binarize_preds(predictions)
            labels = binarize_preds(labels)
        else:
            assert probs.ndim == 2
            bin_probs = binarize_probs(probs)
            predictions = bin_probs.argmax(-1)
            labels = binarize_preds(labels)

    return {"acc": simple_accuracy(predictions, labels)}
