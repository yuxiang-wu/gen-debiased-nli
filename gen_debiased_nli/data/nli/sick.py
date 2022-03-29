import os

import pandas as pd

LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}


def _load_file(datapath):
    label_dict = {"NEUTRAL": "neutral", "CONTRADICTION": "contradiction",
                  "ENTAILMENT": "entailment"}
    df = pd.read_csv(datapath, sep="\t")
    sentences1 = df['sentence_A'].tolist()
    sentences2 = df['sentence_B'].tolist()
    labels = df['entailment_judgment'].tolist()
    labels = [label_dict[label] for label in labels]

    data = []
    for s1, s2, label in zip(sentences1, sentences2, labels):
        label_id = LABEL2ID[label]
        data.append({"premise": s1, "hypothesis": s2, "label": label_id})
    return data


def load_sick(data_dir):
    train = _load_file(os.path.join(data_dir, 'SICK/SICK_train.txt'))
    dev = _load_file(os.path.join(data_dir, 'SICK/SICK_trial.txt'))
    test = _load_file(os.path.join(data_dir, 'SICK/SICK_test_annotated.txt'))

    return train, dev, test
