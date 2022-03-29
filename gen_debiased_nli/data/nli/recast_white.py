import os

label2id = {"entailment": 0, "contradiction": 1}
label_list = list(label2id.keys())
num_labels = len(label_list)
id_map = {0: 0, 1: 1, 2: 1}  # mapping 3-class to 2-class ids


def _process_file(datapath):
    data = {}
    for type in ['train', 'dev', 'test']:
        data[type] = {}
        data[type]['s1'] = []
        data[type]['s2'] = []
        data[type]['labels'] = []

    orig_sent, hyp_sent, data_split, src, label = None, None, None, None, None
    for line in open(datapath):
        if line.startswith("entailed: "):
            label = "entailment"
            if "not-entailed" in line:
                label = "contradiction"
        elif line.startswith("text: "):
            orig_sent = " ".join(line.split("text: ")[1:]).strip()
        elif line.startswith("hypothesis: "):
            hyp_sent = " ".join(line.split("hypothesis: ")[1:]).strip()
        elif line.startswith("partof: "):
            data_split = line.split("partof: ")[-1].strip()
        elif line.startswith("provenance: "):
            src = line.split("provenance: ")[-1].strip()
        elif not line.strip():
            assert orig_sent != None
            assert hyp_sent != None
            assert data_split != None
            assert src != None
            assert label != None
            data[data_split]['labels'].append(label)
            data[data_split]['s1'].append(orig_sent)
            data[data_split]['s2'].append(hyp_sent)

            orig_sent, hyp_sent, data_split, src, label = None, None, None, None, None

    data_dict = {"train": [], "dev": [], "test": []}
    for split in ["train", "dev", "test"]:
        assert (len(data[split]['labels']) == len(data[split]['s1']) == len(data[split]['s2']))
        for s1, s2, label in zip(data[split]['s1'], data[split]['s2'], data[split]['labels']):
            label_id = label2id[label]
            data_dict[split].append({"premise": s1, "hypothesis": s2, "label": label_id})

    return data_dict


def load_dpr(data_dir):
    data = _process_file(os.path.join(data_dir, "rte/dpr_data.txt"))
    return data["train"], data["dev"], data["test"]


def load_sprl(data_dir):
    data = _process_file(os.path.join(data_dir, "rte/sprl_data.txt"))
    return data["train"], data["dev"], data["test"]


def load_fnplus(data_dir):
    data = _process_file(os.path.join(data_dir, "rte/fnplus_data.txt"))
    return data["train"], data["dev"], data["test"]
