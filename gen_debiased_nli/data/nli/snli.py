from datasets import load_dataset


def load_snli(data_dir=None):
    raw_dataset = load_dataset("snli")

    train_set = raw_dataset["train"]
    val_set = raw_dataset["validation"]
    test_set = raw_dataset["test"]

    return train_set, val_set, test_set
