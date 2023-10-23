#!bin/sh

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="RumorDAS"
export WANDB_DIR=... ## need to be defined
output_dir=...       ## need to be defined
batch_size=8
exp_name=bi-tgn-roberta

for dataset in re2019 twitter15 twitter16
do
	for i in $(seq 0 4)
	do
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
	done
done
