# Beyond Detection: A Defend-and-Summarize Strategy for Robust and Interpretable Rumor Analysis on Social Media

![](https://github.com/joshchang0111/EMNLP2023-RumorDAS/blob/master/das_overview.png)

## Introduction
Code for the EMNLP 2023 paper "Beyond Detection: A Defend-and-Summarize Strategy for Robust and Interpretable Rumor Analysis on Social Media" by Yi-Ting Chang, Yun-Zhu Song, Yi-Syuan Chen, and Hong-Han Shuai.

## Environmental Setup
Run the script `build_env.sh` first to install necessary packages through pip.

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
```

## Dataset
The datasets should be placed at the folder `dataset` on the same layer as `src`. Each dataset should contain a main csv file `data.csv` and 5 folders in the format `split_{$FOLD_N}`. consisting of 8 columns (source_id, tweet_id, parent_idx, self_idx, num_parent, max_seq_len, text, veracity).

### Build Clusters
In order to train the **S**elf-**S**upervised **R**esponse **A**bstractor (SSRA) with $k$-means settings, you need to build the clusters information from the dataset first by running the following command.
```
python main.py \
    --task_type build_cluster_summary \
    --model_name_or_path facebook/bart-base \
    --cluster_type kmeans \
    --cluster_mode train \
    --num_clusters $num_clusters \
    --dataset_name $dataset \
    --fold $i \
    --per_device_train_batch_size $batch_size
```
More details can be found at `src/scripts/summarizer/build_cluster.sh`.

Our processed datasets are available at [...].

## Run the Codes
We provide the training & evaluation scripts for each component of our framework in the folder `src/scripts`. Notice that each script requires defining an experiment name (`--exp_name`), and a folder with that name will be automatically created after running the command, any experimental results will be stored in this folder, including model checkpoints.

### 1. Train BiTGN (RoBERTa)
```
python main.py \
    --task_type train_detector \
    --model_name_or_path roberta-base \
    --td_gcn \
    --bu_gcn \
    --dataset_name $dataset \
    --train_file train.csv \
    --validation_file test.csv \
    --fold $i \
    --do_train \
    --per_device_train_batch_size $batch_size \
    --learning_rate 2e-5 \
    --num_train_epochs 10 \
    --exp_name $exp_name \
    --output_dir $output_dir
```
Evaluation command can be found in `src/scripts/detection/eval.sh`.

### 2. Train BiTGN (BART) + ARG
#### 2.1 Adversarial Training Stage 1
```
python main.py \
    --task_type train_adv_stage1 \
    --model_name_or_path facebook/bart-base \
    --td_gcn \
    --bu_gcn \
    --dataset_name $dataset \
    --train_file train.csv \
    --validation_file test.csv \
    --fold $i \
    --do_train \
    --per_device_train_batch_size $batch_size \
    --learning_rate 2e-5 \
    --num_train_epochs 10 \
    --exp_name $exp_name \
    --output_dir $output_dir \
```
#### 2.2 Adversarial Training Stage 2
```
python main.py \
    --task_type train_adv_stage2 \
    --model_name_or_path facebook/bart-base \
    --td_gcn \
    --bu_gcn \
    --dataset_name $dataset \
    --train_file train.csv \
    --validation_file test.csv \
    --fold $i \
    --do_train \
    --per_device_train_batch_size $batch_size \
    --learning_rate 2e-5 \
    --num_train_epochs 10 \
    --exp_name $exp_name \
    --output_dir "$output_dir"
```

### 3. Train Response Extractor (AutoEncoder)
```
python main.py \
    --task_type train_filter \
    --model_name_or_path facebook/bart-base \
    --filter_layer_enc $n_layer \
    --filter_layer_dec $n_layer \
    --dataset_name $dataset \
    --train_file train.csv \
    --validation_file test.csv \
    --fold $i \
    --do_train \
    --per_device_train_batch_size $batch_size \
    --learning_rate $lr \
    --num_train_epochs 50 \
    --exp_name filter_$n_layer \
    --output_dir $output_dir
```

### 4. Train Response Abstractor (SSRA)
To train the response abstractor with $k$-means settings, check that you already build clusters as documented in the dataset description.
```
python main.py \
    --task_type ssra_kmeans \
    --model_name_or_path lidiya/bart-base-samsum \
    --cluster_type kmeans \
    --cluster_mode train \
    --num_clusters $num_clusters \
    --dataset_name $dataset \
    --train_file train.csv \
    --validation_file test.csv \
    --fold $i \
    --do_train \
    --per_device_train_batch_size $batch_size \
    --learning_rate $lr \
    --num_train_epochs 10 \
    --exp_name ssra_kmeans_$num_clusters \
    --output_dir $output_dir
```
Evaluation command can be found in `src/scripts/ssra-kmeans/eval.sh`.

## Citation