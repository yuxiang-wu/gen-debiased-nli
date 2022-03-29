# Generating Data to Mitigate Spurious Correlations in Natural Language Inference Datasets (GD-NLI)

This repository hosts the data and code of the
paper [Generating Data to Mitigate Spurious Correlations in Natural Language Inference Datasets](https://arxiv.org/abs/2203.12942)
published at [ACL 2022](https://www.2022.aclweb.org/).

![diagram](https://i.imgur.com/evRIqlo.png)

<br>

* [Download our Generated Debiased NLI (GD-NLI) Datasets](#download-our-generated-debiased-nli-gd-nli-datasets)
* [Code and Models](#code-and-models)
    * [Installation and setup](#installation-and-setup)
    * [Download data](#download-data)
    * [Training with our datasets](#training-with-our-datasets)
    * [Training PoE models with our datasets](#training-poe-models-with-our-datasets)
* [Citing](#citing)

## Download our Generated Debiased NLI (GD-NLI) Datasets

Our synthetic debiased NLI datasets (GD-NLI) can be downloaded using the following links. All data files follow jsonline format. Each line contains a json object with the standard NLI fields `premise`, `hypothesis`, `label`, and a metadata field `type` that indicates whether the sample comes from the original dataset or is generated.

| Dataset    |      #Samples | Link                                                                                                      |
| ---------- | ---------:| --------|
| All data   |         - | [zip](https://storage.googleapis.com/allennlp-public-data/gen-debiased-nli/gen-debiased-nli-datasets.zip) |
| SNLI Z-Aug | 1,142,475 | [jsonl](https://storage.googleapis.com/allennlp-public-data/gen-debiased-nli/snli_z-aug.jsonl)|
| SNLI Seq-Z |   933,085 | [jsonl](https://storage.googleapis.com/allennlp-public-data/gen-debiased-nli/snli_seq-z.jsonl)|
| SNLI Par-Z |   927,906 | [jsonl](https://storage.googleapis.com/allennlp-public-data/gen-debiased-nli/snli_par-z.jsonl)|
| MNLI Z-Aug |   744,326 | [jsonl](https://storage.googleapis.com/allennlp-public-data/gen-debiased-nli/mnli_z-aug.jsonl)|
| MNLI Seq-Z |   740,811 | [jsonl](https://storage.googleapis.com/allennlp-public-data/gen-debiased-nli/mnli_seq-z.jsonl)|
| MNLI Par-Z |   744,200 | [jsonl](https://storage.googleapis.com/allennlp-public-data/gen-debiased-nli/mnli_par-z.jsonl)|

## Code and Models

### Installation and setup

```bash
conda create --name gdnli python=3.7 && conda activate gdnli
git clone https://github.com/jimmycode/gen-debiased-nli
cd gen-debiased-nli
. scripts/init.sh
```

### Download data

You can download GD-NLI datasets with the links provided above, or run the following script:

```bash
. scripts/get_data.sh
```

We use SNLI-hard, MNLI-hard, HANS, and an adversarial attack suite to evaluate our models. We also provide a script to download these evaluation datasets.

```bash
. scripts/get_eval_data.sh
```

### Training with our datasets

### Training PoE models with our datasets


## Citing

Please use the following bibtex to cite our work:

```
@inproceedings{gen-debiased-nli-2022,
    title = "Generating Data to Mitigate Spurious Correlations in Natural Language Inference Datasets",
    author = "Wu, Yuxiang  and
      Gardner, Matt  and
      Stenetorp, Pontus  and
      Dasigi, Pradeep",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics",
    month = may,
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```

