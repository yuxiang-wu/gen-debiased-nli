import csv

from ...utils import convert_to_dataset

LABEL2ID = {"entailment": 0, "non-entailment": 1}
LABEL_LIST = ["entailment", "non-entailment"]
NUM_LABELS = len(LABEL_LIST)
ID_MAP = {0: 0, 1: 1, 2: 1}  # mapping 3-class to 2-class ids


def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for i, line in enumerate(reader):
            if i == 0: continue  # skip the first line
            lines.append(line)
        return lines


def load_hans(data_dir):
    train, dev, test = [], [], []

    train_lines = read_tsv(f"{data_dir}/HANS/heuristics_train_set.txt")
    for line in train_lines:
        label_id, prem, hyp = LABEL2ID[line[0]], line[5], line[6]
        train.append({"premise": prem, "hypothesis": hyp, "label": label_id})

    train_data_dict = convert_to_dataset(train).train_test_split(test_size=0.1, seed=42)
    train = train_data_dict["train"]
    dev = train_data_dict["test"]

    eval_lines = read_tsv(f"{data_dir}/HANS/heuristics_evaluation_set.txt")
    for line in eval_lines:
        label_id, prem, hyp = LABEL2ID[line[0]], line[5], line[6]
        test.append({"premise": prem, "hypothesis": hyp, "label": label_id})
    test = convert_to_dataset(test)

    return train, dev, test


def load_hans_entail(data_dir):
    test = []

    eval_lines = read_tsv(f"{data_dir}/HANS/heuristics_evaluation_set.txt")
    for line in eval_lines:
        label_id, prem, hyp = LABEL2ID[line[0]], line[5], line[6]
        test.append({"premise": prem, "hypothesis": hyp, "label": label_id})
    test = convert_to_dataset(test)

    return None, None, test
