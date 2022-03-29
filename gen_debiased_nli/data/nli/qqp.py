import csv
import os

LABEL2ID = {"entailment": 0, "neutral": 1}
LABEL_LIST = ["entailment", "neutral"]
NUM_LABELS = len(LABEL_LIST)
ID_MAP = {0: 0, 1: 1, 2: 1}  # mapping 3-class to 2-class ids

LABEL_DICT = {'1': "entailment", '0': "neutral"}


def _load_file(datapath):
    data = []
    with open(datapath) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            sentence1 = row[1]
            sentence2 = row[2]
            label = LABEL_DICT[row[0]]
            label_id = LABEL2ID[label]
            data.append({"premise": sentence1, "hypothesis": sentence2, "label": label_id})

    return data


def load_qqp(data_dir):
    train = _load_file(os.path.join(data_dir, 'QQP', 'train.tsv'))
    dev = _load_file(os.path.join(data_dir, 'QQP', 'dev.tsv'))
    test = _load_file(os.path.join(data_dir, 'QQP', 'test.tsv'))

    return train, dev, test
