# Beyond Detection: A Defend-and-Summarize Strategy for Robust and Interpretable Rumor Analysis on Social Media

![](https://github.com/joshchang0111/EMNLP2023-RumorDAS/blob/master/das_overview.png)

## Introduction
Code for the EMNLP 2023 paper "Beyond Detection: A Defend-and-Summarize Strategy for Robust and Interpretable Rumor Analysis on Social Media" by Yi-Ting Chang, Yun-Zhu Song, Yi-Syuan Chen, and Hong-Han Shuai.

## Environmental Setup
This code is developed under Ubuntu 20.04.3 LTS and Python 3.8.10. Run the script `build_env.sh` first to install necessary packages through pip.

### Install PyTorch as follows.
```
$ pip install torch==1.11.0+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
$ pip install torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
```
**[[Reference Solution for Installing PyTorch Geometric](https://stackoverflow.com/questions/70008715/pytorch-and-torch-scatter-were-compiled-with-different-cuda-versions-on-google-c)]**: For installing torch-scatter/torch-cluster/torch-sparse, you should first obtain the **reference url** by your desired *PyTorch* and *CUDA* version according to your computer. Next, you need to specify the latest version of torch-scatter provided by the link (which is 2.0.8 in this case) when installing through pip.

### Install kmeans_pytorch from source.
```
$ git clone https://github.com/subhadarship/kmeans_pytorch
$ cd kmeans_pytorch
$ pip install --editable .
$ pip install numba
```

## Dataset
The datasets should be placed at the folder `dataset` on the same layer as `src`. Each dataset should contain a main csv file `data.csv` and 5 folders in the format `split_{$FOLD_N}`. consisting of 8 columns (source_id, tweet_id, parent_idx, self_idx, num_parent, max_seq_len, text, veracity).

Our processed datasets are available at [...].

## Run the Codes
We provide the training and evaluation scripts for each component of our framework in the folder `src/scripts`. Notice that each script requires an output root directory (`$output_dir`) and an experiment name (`--exp_name`). After executing each script, the experimental results including the model checkpoints will be automatically stored in the following structure:
```
|__ {$output_dir}
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
```
$ sh scripts/detection/train.sh
```
Evaluate trained models.
```
$ sh scripts/detection/eval.sh
```

### 2. Train BiTGN (BART) + ARG
#### 2.1 Adversarial Training Stage 1
Train the detector along with the generator.
```
$ sh scripts/attack/stage1/train.sh
```
Evaluate the trained detector.
```
$ sh scripts/attack/stage1/eval.sh
```
#### 2.2 Adversarial Training Stage 2
Train the generator to attack the detector while fixing the detector. Note that this training stage should be executed after stage 1 finished, or at least one checkpoint from stage 1 exists.
```
$ sh scripts/attack/stage2/train.sh
```

### 3. Train Response Extractor (AutoEncoder)
This stage obtains the embedding from the pre-trained detector, please make sure you have at least one checkpoint from previous step before you run the following script.
```
$ sh scripts/summarizer/filter/train.sh
```

### 4. Build Clusters
In order to train the **S**elf-**S**upervised **R**esponse **A**bstractor (SSRA) with $k$-means settings, you need to build the clusters information from the dataset first by running the following script. Note that this step also requires a checkpoint from step 2.2, so make sure the settings is correct.
```
$ sh scripts/summarizer/build_cluster.sh
```

### 5. Train Response Abstractor (SSRA)
To train the response abstractor with $k$-means settings, check that you have already built clusters information as documented in the dataset description .
```
$ sh scripts/summarizer/ssra-kmeans/train.sh
```
Evaluate the trained abstractors.
```
$ sh scripts/summarizer/ssra-kmeans/eval.sh
```

## Citation