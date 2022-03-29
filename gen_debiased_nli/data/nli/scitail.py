import json_lines

LABEL2ID = {"entailment": 0, "neutral": 1}
LABEL_LIST = ["entailment", "neutral"]
NUM_LABELS = len(LABEL_LIST)
ID_MAP = {0: 0, 1: 1, 2: 1}  # mapping 3-class to 2-class ids


def load_scitail(data_dir):
    train, dev, test = [], [], []
    scitail_data = {"train": train, "dev": dev, "test": test}

    for split in ["train", "dev", "test"]:
        with open(f"{data_dir}/SciTail/snli_format/scitail_1.0_{split}.txt", 'rb') as f:
            for item in json_lines.reader(f):
                prem, hypo, glabel_str = item['sentence1'], item['sentence2'], item['gold_label']
                label_id = LABEL2ID[glabel_str]
                scitail_data[split].append({"premise": prem, "hypothesis": hypo, "label": label_id})

    return train, dev, test
