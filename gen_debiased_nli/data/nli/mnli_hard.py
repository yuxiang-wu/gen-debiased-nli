import os

LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}


def _load_files(data_dir):
    sentences2 = []
    sentences1 = []
    label_ids = []

    with open(os.path.join(data_dir, 'labels.test'), 'r') as f:
        for label in f.readlines():
            label_ids.append(LABEL2ID[label.strip()])

    with open(os.path.join(data_dir, 's1.test'), 'r') as f:
        for s1 in f.readlines():
            sentences1.append(s1.strip())

    with open(os.path.join(data_dir, 's2.test'), 'r') as f:
        for s2 in f.readlines():
            sentences2.append(s2.strip())

    data = []
    for s1, s2, label in zip(sentences1, sentences2, label_ids):
        data.append({"premise": s1, "hypothesis": s2, "label": label})

    return data


def load_mnli_matched_hard(data_dir):
    test_data = _load_files(os.path.join(data_dir, "MNLIMatchedHardWithHardTest"))
    return None, None, test_data


def load_mnli_mismatched_hard(data_dir):
    test_data = _load_files(os.path.join(data_dir, "MNLIMismatchedHardWithHardTest"))
    return None, None, test_data
