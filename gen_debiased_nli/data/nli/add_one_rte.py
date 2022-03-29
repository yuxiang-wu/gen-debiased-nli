import os

LABEL2ID = {"entailment": 0, "contradiction": 1}
LABEL_LIST = ["entailment", "contradiction"]
NUM_LABELS = len(LABEL_LIST)
ID_MAP = {0: 0, 1: 1, 2: 1}  # mapping 3-class to 2-class ids


def _convert_label(score, is_test):
    """ Converts not_entailed to contradiction, since we convert
    contradiction and neutral to one label, it does not matter to
    which label we convert the not_entailed labels.
    """
    score = float(score)
    if is_test:
        if score <= 3:
            return "contradiction"
        elif score >= 4:
            return "entailment"
        return  # those between 3 and 4 are ignored

    if score < 3.5:
        return "contradiction"
    return "entailment"


def _load_file(datadir, type):
    data = []
    line_count = -1
    for line in open(os.path.join(datadir, "AddOneRTE/addone-entailment/splits/data.%s" % (type))):
        line_count += 1
        line = line.split("\t")
        assert (len(line) == 7)  # "add one rte %s file has a bad line" % (f))
        label = _convert_label(line[0], type == "test")
        if not label:
            continue
        label_id = LABEL2ID[label]
        hypothesis = line[-1].replace("<b><u>", "").replace("</u></b>", "").strip()
        premise = line[-2].replace("<b><u>", "").replace("</u></b>", "").strip()

        data.append({"premise": premise, "hypothesis": hypothesis, "label": label_id})

    return data


def load_add_one_rte(data_dir):
    train = _load_file(data_dir, 'train')
    dev = _load_file(data_dir, 'dev')
    test = _load_file(data_dir, 'test')

    return train, dev, test
