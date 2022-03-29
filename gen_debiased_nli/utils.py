import json
import math
import os
from typing import List, Dict

from accelerate import Accelerator
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

task_to_datasets = {
    "sst2": ("glue", "sst2"),
    "snli": ("snli", None),
    "imdb": ("imdb", None),
    "cola": ("glue", "cola"),
}

task_to_keys = {
    "sst2": ("sentence", None),
    "snli": ("premise", "hypothesis"),
    "imdb": ("text", None),
    "cola": ("sentence", None),
}

task_to_metrics = {
    "sst2": ("glue", "sst2"),
    "snli": ("accuracy",),
    "imdb": ("accuracy",),
    "cola": ("glue", "cola"),
}


# def init_wandb():
#     """Initialise wandb"""
#     try:
#         import wandb
#
#         wandb.ensure_configured()
#         if wandb.api.api_key is None:
#             has_wandb = False
#             wandb.termwarn(
#                 "W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
#         else:
#             has_wandb = False if os.getenv("WANDB_DISABLED") else True
#     except (ImportError, AttributeError):
#         has_wandb = False
#
#     return has_wandb

def shuffle_dataset(dataset: Dataset, seed=None):
    shuffled_dataset = dataset.shuffle(seed=seed) if seed is not None else dataset
    return shuffled_dataset


def split_dataset(dataset: Dataset, percentage, seed=None):
    shuffled_dataset = shuffle_dataset(dataset, seed)

    size = len(dataset)
    assert 0. < percentage < 1., f"Invalide percentage {percentage}"
    split1 = shuffled_dataset.select(range(0, math.floor(size * percentage)))
    split2 = shuffled_dataset.select(range(math.floor(size * percentage), size))

    return split1, split2


def del_columns(dataset):
    dataset.remove_columns_(
        list(filter(lambda n: n not in ["premise", "hypothesis", "label"], dataset.column_names))
    )
    return dataset


def truncate_dataset(dataset: Dataset, max_num_examples: int):
    return dataset.select(range(min(len(dataset), max_num_examples)))


def save_model(model: PreTrainedModel, save_dir, accelerator: Accelerator = None,
               tokenizer: PreTrainedTokenizer = None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if accelerator is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_dir, save_function=accelerator.save)
    else:
        model.save_pretrained(save_dir)

    if tokenizer is not None:
        if accelerator is None:
            tokenizer.save_pretrained(save_dir)
        elif accelerator.is_local_main_process:
            tokenizer.save_pretrained(save_dir, save_function=accelerator.save)


def load_jsonl(fn):
    all_data = []
    with open(fn, "r") as f:
        for line in f.readlines():
            all_data.append(json.loads(line))
    return all_data


def write_jsonl(fn, all_data):
    with open(fn, "w") as f:
        for data in all_data:
            f.write(json.dumps(data) + "\n")


def apply_dropout(m):
    for each_module in m.children():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.train()
    return m


def enable_dropout(m):
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.train()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def convert_to_dataset(data: List[Dict]):
    if isinstance(data, Dataset) or data is None:
        return data

    all_keys = list(data[0].keys())
    data_dict = {k: [d[k] if k in d else None for d in data] for k in all_keys}
    dataset = Dataset.from_dict(data_dict)
    return dataset
