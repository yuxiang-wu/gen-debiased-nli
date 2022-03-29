import json_lines

LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}


def load_snli_hard(data_dir):
    test_data = []
    with open(f'{data_dir}/SNLIHard/snli_1.0_test_hard.jsonl', 'r') as f:
        for item in json_lines.reader(f):
            label_id = LABEL2ID[item['gold_label']]
            test_data.append(
                {"premise": item['sentence1'], "hypothesis": item['sentence2'], "label": label_id}
            )

    return None, None, test_data
