#!bin/sh

export WANDB_PROJECT="RumorDAS"
export WANDB_DIR=... ## need to be defined
output_dir=/mnt/1T/projects/RumorDAS       ## need to be defined
batch_size=8

for dataset in re2019 twitter15 twitter16
do
	for i in $(seq 0 4)
	do
		for num_clusters in $(seq 1 5)
		do
			## Build cluster summary pairs by kmeans for training
			python main.py \
				--task_type build_cluster_summary \
				--model_name_or_path facebook/bart-base \
				--cluster_type kmeans \
				--cluster_mode train \
				--num_clusters $num_clusters \
				--dataset_name $dataset \
				--fold $i \
				--per_device_train_batch_size $batch_size \
				--output_dir $output_dir

			## Build cluster summary pairs by kmeans for testing
			#python main.py \
			#	--task_type build_cluster_summary \
			#	--model_name_or_path facebook/bart-base \
			#	--cluster_type kmeans \
			#	--cluster_mode test \
			#	--num_clusters $num_clusters \
			#	--dataset_name $dataset \
			#	--fold $i \
			#	--per_device_train_batch_size $batch_size
		done
	done
done