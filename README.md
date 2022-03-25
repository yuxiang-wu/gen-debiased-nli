# Generating Data to Mitigate Spurious Correlations in Natural Language Inference Datasets

This repository hosts the data and code of the
paper [Generating Data to Mitigate Spurious Correlations in Natural Language Inference Datasets](https://arxiv.org/abs/2203.12942)
published at [ACL 2022](https://www.2022.aclweb.org/).

![diagram](https://i.imgur.com/evRIqlo.png)

## Abstract

Natural language processing models often exploit spurious correlations between task-independent features and labels in
datasets to perform well only within the distributions they are trained on, while not generalising to different task
distributions. We propose to tackle this problem by generating a debiased version of a dataset, which can then be used
to train a debiased, off-the-shelf model, by simply replacing its training data. Our approach consists of 1) a method
for training data generators to generate high-quality, label-consistent data samples; and 2) a filtering mechanism for
removing data points that contribute to spurious correlations, measured in terms of z-statistics. We generate debiased
versions of the SNLI and MNLI datasets, and we evaluate on a large suite of debiased, out-of-distribution, and
adversarial test sets. Results show that models trained on our debiased datasets generalise better than those trained on
the original datasets in all settings. On the majority of the datasets, our method outperforms or performs comparably to
previous state-of-the-art debiasing strategies, and when combined with an orthogonal technique, product-of-experts, it
improves further and outperforms previous best results of SNLI-hard and MNLI-hard.

## Download Our Generated Debiased NLI Datasets


| Dataset    |      Size | link                                                                                                      |
| ---------- | ---------:| --------|
| All data   |         - | [zip](https://storage.googleapis.com/allennlp-public-data/gen-debiased-nli/gen-debiased-nli-datasets.zip) |
| SNLI Z-Aug | 1,142,475 | [jsonl](https://storage.googleapis.com/allennlp-public-data/gen-debiased-nli/snli_z-aug.jsonl)|
| SNLI Seq-Z |   933,085 | [jsonl](https://storage.googleapis.com/allennlp-public-data/gen-debiased-nli/snli_seq-z.jsonl)|
| SNLI Par-Z |   927,906 | [jsonl](https://storage.googleapis.com/allennlp-public-data/gen-debiased-nli/snli_par-z.jsonl)|
| MNLI Z-Aug |   744,326 | [jsonl](https://storage.googleapis.com/allennlp-public-data/gen-debiased-nli/mnli_z-aug.jsonl)|
| MNLI Seq-Z |   740,811 | [jsonl](https://storage.googleapis.com/allennlp-public-data/gen-debiased-nli/mnli_seq-z.jsonl)|
| MNLI Par-Z |   744,200 | [jsonl](https://storage.googleapis.com/allennlp-public-data/gen-debiased-nli/mnli_par-z.jsonl)|


## Citation

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

