#!/bin/bash

pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.8.2
pip install -r requirements.txt
pip install -e .