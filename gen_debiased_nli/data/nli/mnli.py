from datasets import load_dataset

from ...utils import del_columns

seed = 42


def load_mnli(data_dir=None):
    raw_dataset = load_dataset("multi_nli")

    train_set = del_columns(raw_dataset["train"])
    train_test_dict = train_set.train_test_split(test_size=10000, seed=seed)
    train_set, dev_set = train_test_dict["train"], train_test_dict["test"]

    test_set = del_columns(raw_dataset["validation_matched"])  # use matched set as test set
    # val_mismatched_set = del_columns(raw_dataset["validation_mismatched"])
    return train_set, dev_set, test_set


def load_mnli_matched(data_dir=None):
    raw_dataset = load_dataset("multi_nli")
    test_set = del_columns(raw_dataset["validation_matched"])  # use matched set as test set
    return None, None, test_set


def load_mnli_mismatched(data_dir=None):
    raw_dataset = load_dataset("multi_nli")
    test_set = del_columns(raw_dataset["validation_mismatched"])

    return None, None, test_set
