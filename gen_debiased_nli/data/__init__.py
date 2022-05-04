import os

#from .fact_check import ALL_FC_NAMES, get_fc_dataset
from .nli import ALL_NLI_NAMES, get_nli_dataset


def get_dataset(data_dir, name):
    assert os.path.isdir(data_dir)

    if name in ALL_NLI_NAMES:
        return get_nli_dataset(data_dir, name)
    #elif name in ALL_FC_NAMES:
    #    return get_fc_dataset(data_dir, name)
    else:
        raise ValueError(f"Cannot find {name} in the datasets.")
