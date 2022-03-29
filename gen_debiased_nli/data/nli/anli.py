from datasets import load_dataset, concatenate_datasets

from ...utils import del_columns


def load_anli_r1(data_dir=None):
    raw_dataset = load_dataset("anli")
    train_set = del_columns(raw_dataset["train_r1"])
    val_set = del_columns(raw_dataset["dev_r1"])
    test_set = del_columns(raw_dataset["test_r1"])

    return train_set, val_set, test_set


def load_anli_r2(data_dir=None):
    raw_dataset = load_dataset("anli")
    train_set = del_columns(raw_dataset["train_r2"])
    val_set = del_columns(raw_dataset["dev_r2"])
    test_set = del_columns(raw_dataset["test_r2"])

    return train_set, val_set, test_set


def load_anli_r3(data_dir=None):
    raw_dataset = load_dataset("anli")
    train_set = del_columns(raw_dataset["train_r3"])
    val_set = del_columns(raw_dataset["dev_r3"])
    test_set = del_columns(raw_dataset["test_r3"])

    return train_set, val_set, test_set


def load_anli_all(data_dir=None):
    raw_dataset = load_dataset("anli")
    train_set = concatenate_datasets([del_columns(raw_dataset[f"train_r{i}"]) for i in [1, 2, 3]])
    val_set = concatenate_datasets([del_columns(raw_dataset[f"dev_r{i}"]) for i in [1, 2, 3]])
    test_set = concatenate_datasets([del_columns(raw_dataset[f"test_r{i}"]) for i in [1, 2, 3]])

    return train_set, val_set, test_set
