#!/bin/bash

dataset_dir="data"
mkdir -p $dataset_dir

echo "Downloading and Processing SNLI-hard set"
mkdir $dataset_dir/snli_hard
wget --directory-prefix=$dataset_dir/snli_hard  https://nlp.stanford.edu/projects/snli/snli_1.0_test_hard.jsonl
mv $dataset_dir/snli_hard $dataset_dir/SNLIHard

# Process the HANS dataset.
mkdir $dataset_dir/HANS
wget -O  $dataset_dir/HANS/heuristics_evaluation_set.txt \
    "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"
wget -O  $dataset_dir/HANS/heuristics_train_set.txt \
    "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_train_set.txt"

# Dataset from [An Empirical Study on Model-agnostic Debiasing Strategies for Robust Natural Language Inference](https://github.com/tyliupku/nli-debiasing-datasets)
mkdir -p $dataset_dir/nli_debiasing_datasets
wget --directory-prefix=$dataset_dir/nli_debiasing_datasets https://raw.githubusercontent.com/tyliupku/nli-debiasing-datasets/master/robust_nli.txt.zip
unzip $dataset_dir/nli_debiasing_datasets/robust_nli.txt.zip -d $dataset_dir/nli_debiasing_datasets/
rm $dataset_dir/nli_debiasing_datasets/robust_nli.txt.zip

# MNLI-hard from [End-to-End Bias Mitigation by Modelling Biases in Corpora](https://www.aclweb.org/anthology/2020.acl-main.769)

## Downloads the MNLIMatchedHard devlopment set.
mkdir $dataset_dir/MNLIMatchedHardWithHardTest
wget -O $dataset_dir/MNLIMatchedHardWithHardTest/MNLIMatchedHardWithHardTest.zip  "https://www.dropbox.com/s/3aktzl4bhmqti9n/MNLIMatchedHardWithHardTest.zip"
unzip -j $dataset_dir/MNLIMatchedHardWithHardTest/MNLIMatchedHardWithHardTest.zip -d $dataset_dir/MNLIMatchedHardWithHardTest
rm $dataset_dir/MNLIMatchedHardWithHardTest/MNLIMatchedHardWithHardTest.zip

## Downloads the MNLIMismatchedHard development set.
mkdir -p $dataset_dir/MNLIMismatchedHardWithHardTest
wget -O $dataset_dir/MNLIMismatchedHardWithHardTest/MNLIMismatchedHardWithHardTest.zip  "https://www.dropbox.com/s/bidxvrd8s2msyan/MNLIMismatchedHardWithHardTest.zip"
unzip -j $dataset_dir/MNLIMismatchedHardWithHardTest/MNLIMismatchedHardWithHardTest.zip -d $dataset_dir/MNLIMismatchedHardWithHardTest
rm $dataset_dir/MNLIMismatchedHardWithHardTest/MNLIMismatchedHardWithHardTest.zip
