#!bin/sh

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="RumorDAS"
export WANDB_DIR=... ## need to be defined
output_dir=/mnt/1T/projects/RumorDAS       ## need to be defined
batch_size=256
lr=4e-5

#############################################
## Transformer AutoEncoder Response Filter ##
#############################################
for dataset in re2019 twitter15 twitter16
do
	for i in $(seq 0 4)
	do
		for n_layer in 2 4 6
		do
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
		done
	done
done
