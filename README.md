# Multiturn 

This is the code for Kaggle Red Teaming Challenge

``` bash

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

source miniconda/bin/activate

git clone git@github.com:martinjingyu/MultiJailbreak.git

cd MultiJailbreak

conda create -n Mul python=3.12

conda activate Mul

pip install -r requirements.txt

pip install -U "huggingface_hub[cli]"

hf auth login

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

git config --global credential.helper store

python utils/download_data.py

```