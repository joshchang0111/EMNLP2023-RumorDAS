# Beyond Detection: A Defend-and-Summarize Strategy for Robust and Interpretable Rumor Analysis on Social Media

![](https://github.com/joshchang0111/EMNLP2023-RumorDAS/blob/master/das_overview.png)

## Introduction
Code for the EMNLP 2023 paper "Beyond Detection: A Defend-and-Summarize Strategy for Robust and Interpretable Rumor Analysis on Social Media" by Yi-Ting Chang, Yun-Zhu Song, Yi-Syuan Chen, and Hong-Han Shuai.

## Environmental Setup
```
ipdb
tqdm
wandb -> Logging
emoji
numpy
numba
scipy
sklearn
wordcloud
matplotlib

** NLP Tools **
nltk
wordcloud
rouge-score -> For text generation
tweet-preprocessor

** PyTorch **
torch==1.11.0+cu102
torch-cluster==1.6.0
torch-scatter==2.0.9
torch-sparse==0.6.15
torch-geometric==2.2.0
kmeans-pytorch -> Remember to install from source

** Hugging Face **
transformers==4.18.0.dev0
datasets==2.0.0
evaluate -> For perplexity evaluation
```
### Install pytorch as follows.
```
$ pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install torch==1.11.0+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
$ pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
```
**Note**: ([Reference Solution](https://stackoverflow.com/questions/70008715/pytorch-and-torch-scatter-were-compiled-with-different-cuda-versions-on-google-c))
For torch-scatter/torch-cluster/torch-sparse, you should first obtain the reference url by your desired pytorch & cuda version according to your computer. Next, you need to specify the latest version of torch-scatter provided by the link (which is 2.0.8) when installing through pip.

### Install kmeans_pytorch
```
$ git clone https://github.com/subhadarship/kmeans_pytorch
$ cd kmeans_pytorch
$ pip install --editable .
$ pip install numba
```

### Install Hugging Face Transformers **from source**.
```
pip install git+https://github.com/huggingface/transformers
```

## Dataset

## Run the Codes

### Train BiTGN
### Train BiTGN + ARG
### Train Response Extractor (AutoEncoder)
### Train Response Abstractor (SSRA)

## Citation