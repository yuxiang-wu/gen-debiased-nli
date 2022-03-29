import csv
import os
import random

LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}


def convert_label(num):
    if num == 1:
        return 'contradiction'
    elif num == 5:
        return 'entailment'
    return 'neutral'


def _load_file(datadir, split):
    # Bug (jimmycode): split is not used in this method, meaning that train/dev/test sets are the same.
    # This is from the previous implementation (by https://github.com/rabeehk/robust-nli), and we follow it for fair comparison.
    # This may not be a big issue given that we are using JOCI for out-of-domain evaluation only

    sentences1 = []
    sentences2 = []
    labels = []

    with open(os.path.join(datadir, 'JOCI/joci.csv'), 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        line_num = -1
        for row in csv_reader:
            line_num += 1
            if line_num == 0:
                continue
            hyp_src = row[4]

            # TODO: check why this is in the codes.
            '''
            # This is only for processing B.
            if "AGCI" not in hyp_src:
                continue
            '''
            premise, hypothesis, label = row[0], row[1], convert_label(int(row[2]))
            sentences1.append(premise.strip())
            sentences2.append(hypothesis.strip())
            labels.append(label)

    # Now we have all the data in both section A and B.
    combined = list(zip(sentences1, sentences2, labels))
    random.shuffle(combined)
    sentences1[:], sentences2[:], labels[:] = zip(*combined)

    data = []
    for s1, s2, label in zip(sentences1, sentences2, labels):
        label_id = LABEL2ID[label]
        data.append({"premise": s1, "hypothesis": s2, "label": label_id})
    return data


def load_joci(data_dir):
    train = _load_file(data_dir, 'train')
    dev = _load_file(data_dir, 'dev')
    test = _load_file(data_dir, 'test')

    return train, dev, test
