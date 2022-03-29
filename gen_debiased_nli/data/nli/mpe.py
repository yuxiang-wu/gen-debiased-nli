import os

import numpy as np
import pandas as pd

LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}


def _load_file(datapath):
    df = pd.read_csv(datapath, sep="\t")
    premise1 = df['premise1'].tolist()
    premise2 = df['premise2'].tolist()
    premise3 = df['premise3'].tolist()
    premise4 = df['premise4'].tolist()

    premise1 = [s.split('/')[1] for s in premise1]
    premise2 = [s.split('/')[1] for s in premise2]
    premise3 = [s.split('/')[1] for s in premise3]
    premise4 = [s.split('/')[1] for s in premise4]

    sentences1 = [" ".join([s1, s2, s3, s4]) for s1, s2, s3, s4 in zip(premise1, premise2, premise3, premise4)]
    sentences2 = df['hypothesis'].tolist()
    labels = df['gold_label'].tolist()

    indices = [i for i, x in enumerate(labels) if x is not np.nan]
    sentences1 = np.array(sentences1)[indices]
    sentences2 = np.array(sentences2)[indices]
    labels = np.array(labels)[indices]

    data = []
    for s1, s2, label in zip(sentences1, sentences2, labels):
        label_id = LABEL2ID[label]
        data.append({"premise": s1, "hypothesis": s2, "label": label_id})
    return data


def load_mpe(data_dir):
    train = _load_file(os.path.join(data_dir, 'MPE/mpe_train.txt'))
    dev = _load_file(os.path.join(data_dir, 'MPE/mpe_dev.txt'))
    test = _load_file(os.path.join(data_dir, 'MPE/mpe_test.txt'))

    return train, dev, test
