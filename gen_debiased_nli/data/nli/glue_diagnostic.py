import os
import pandas as pd
import numpy as np

LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}


def load_glue_diagnostic(data_dir):
    datapath = os.path.join(data_dir, "GLUEDiagnostic", "diagnostic-full.tsv")
    df = pd.read_csv(datapath, sep='\t')
    labels = df['Label'].values.tolist()

    # Filters all nan labels.
    indices = [i for i, x in enumerate(labels) if x is not np.nan]
    data = {}
    data['premise'] = np.array(df['Premise'].values.tolist())[indices]
    data['hypothesis'] = np.array(df['Hypothesis'].values.tolist())[indices]
    data['label'] = np.array(df['Label'].values.tolist())[indices]
    assert (len(data['premise']) == len(data['hypothesis']) == len(data['label']))

    test_data = [{"premise": p, "hypothesis": h, "label": LABEL2ID[l]}
                 for p, h, l in zip(data['premise'], data['hypothesis'], data['label'])]
    return None, None, test_data
