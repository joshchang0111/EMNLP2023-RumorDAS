# Beyond Detection: A Defend-and-Summarize Strategy for Robust and Interpretable Rumor Analysis on Social Media
![](https://img.shields.io/badge/Python-3.8-blue) ![](https://img.shields.io/badge/Pytorch-1.11.0-orange)

![](https://github.com/joshchang0111/EMNLP2023-RumorDAS/blob/master/das_overview.png)
[Paper] [Datasets]

## Introduction
Original PyTorch implementation for the EMNLP 2023 paper "Beyond Detection: A Defend-and-Summarize Strategy for Robust and Interpretable Rumor Analysis on Social Media" by Yi-Ting Chang, Yun-Zhu Song, Yi-Syuan Chen, and Hong-Han Shuai.

This project is organized in the following structure.
```
|__ src
    |__ main.py   -> organize main flow for the codes
    |__ data      -> data preprocessing/build datasets
    |__ models    -> different model classes
    |__ others
    |__ pipelines -> different trainers
    |__ scripts   -> scripts for train/test of each task
|__ dataset       -> put the processed datasets here
    |__ re2019
    |__ twitter15
    |__ twitter16
|__ {$output_dir} -> store the experimental results with the `exp_id` in each script as the folder name
    |__ re2019
    |__ twitter15
    |__ twitter16
```

## Environmental Setup
This code is developed under **Ubuntu 20.04.3 LTS** and **Python 3.8.10**. Run the script `build_env.sh` first to install necessary packages through pip.

### Install PyTorch as follows.
```bash
$ pip install torch==1.11.0+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
$ pip install torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
```
**[[Reference Solution for Installing PyTorch Geometric](https://stackoverflow.com/questions/70008715/pytorch-and-torch-scatter-were-compiled-with-different-cuda-versions-on-google-c)]**: For installing torch-scatter/torch-cluster/torch-sparse, you should first obtain the **reference url** by your desired *PyTorch* and *CUDA* version according to your computer. Next, you need to specify the latest version of torch-scatter provided by the link (which is 2.0.8 in this case) when installing through pip.

### Install kmeans_pytorch from source.
```bash
$ git clone https://github.com/subhadarship/kmeans_pytorch
$ cd kmeans_pytorch
$ pip install --editable .
$ pip install numba
```

## Dataset
The datasets should be placed at the folder `dataset` on the same layer as `src`. Each dataset should contain several files organized in the following structure.
```
dataset
|__ {DATASET_NAME_0}
    |__ data.csv          -> all data information
    |__ data_tree_ids.csv -> for `build_cluster.sh`
    |__ graph.pth         -> cache for graph data, created after training the detection model once.
    |__ split_{$FOLD_N}
        |__ train.csv
        |__ test.csv
        |__ cluster_summary/train -> store the cluster information for training SSRA.
            |__ kmeans-{$N_CLUSTERS}.csv
            |__ ...
    |__ ...
|__ ...
```
The file `data.csv` consists of 8 columns of data as follows.
|   Column   | Description |
|------------|-------------|
|source_id   |tweet id of the source tweet for each conversation thread|
|tweet_id    |tweet id for each tweet|
|parent_idx  |index of each tweet's parent node, set to `None` if the tweet is source|
|self_idx    |index of each tweet in a conversation thread, arranged in chronological order|
|num_parent  |number of parent nodes in each conversation thread|
|max_seq_len |maximal sequence length for each conversation thread|
|text        |textual content for each tweet|
|veracity    |veracity label for the source post of each conversation thread|

Our processed datasets are available at [Datasets].

## Run the Codes
We provide the training and evaluation scripts for each component of our framework in the folder `src/scripts`. Notice that each script requires an output root directory (`$output_dir`) and an experiment name (`--exp_name`). After executing each script, the experimental results including the model checkpoints will be automatically stored in the following structure:
```
{$output_dir}
|__ {$DATASET_NAME_0}
    |__ {$EXP_NAME_0}
    |__ {$EXP_NAME_2}
    |__ ...
|__ {$DATASET_NAME_1}
    |__ {$EXP_NAME_1}
|__ ...
```

### 1. Train BiTGN (RoBERTa)
Train the model.
```bash
$ sh scripts/detection/train.sh
```
Evaluate trained models.
```bash
$ sh scripts/detection/eval.sh
```

### 2. Train BiTGN (BART) + ARG
#### 2.1 Adversarial Training Stage 1
Train the detector along with the generator.
```bash
$ sh scripts/attack/stage1/train.sh
```
Evaluate the trained detector.
```bash
$ sh scripts/attack/stage1/eval.sh
```
#### 2.2 Adversarial Training Stage 2
Train the generator to attack the detector while fixing the detector. Note that this training stage should be executed after stage 1 finished, or at least one checkpoint from stage 1 exists.
```bash
$ sh scripts/attack/stage2/train.sh
```

### 3. Train Response Extractor (AutoEncoder)
This stage obtains the embedding from the pre-trained detector, please make sure you have at least one checkpoint from previous step before you run the following script.
```bash
$ sh scripts/summarizer/filter/train.sh
```

### 4. Train Response Abstractor (SSRA)
#### 4.1 Build Clusters
In order to train the **S**elf-**S**upervised **R**esponse **A**bstractor (SSRA) with $k$-means settings, you need to build the clusters information from the dataset first by running the following script. Note that this step also requires a checkpoint from step 2.2, so make sure the settings is correct.
```bash
$ sh scripts/summarizer/build_cluster.sh
```
#### 4.2 Start Training
To train the response abstractor with $k$-means settings, check that you have already built clusters information as documented in the dataset description .
```bash
$ sh scripts/summarizer/ssra-kmeans/train.sh
```
Evaluate the trained abstractors.
```bash
$ sh scripts/summarizer/ssra-kmeans/eval.sh
```

### 5. Evaluate the BiTGN with DAS
Evaluate DAS with the following hyper-parameters:
- extract ratio $\rho$ in the range $\{0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 0.9\}$
- number of clusters $k$ in the range $\{1, 2, 3, 4, 5\}$
```bash
$ sh scripts/attack/stage2/eval.sh
```
The script performs the following evaluation for each fold and hyper-parameters set:
1. Evaluate stage-2 detector *without* adversarial attack.
2. Evaluate stage-2 detector *under* adversarial attack *without* summarizer.
3. Evaluate stage-2 detector *under* adversarial attack *with* summarizer.

## Citation
Coming soon.