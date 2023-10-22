#!bin/sh

export WANDB_PROJECT="RumorV2"

if [ $(hostname) = "josh-System-Product-Name" ]; then
	export WANDB_DIR="/mnt/hdd1/projects/RumorV2"
	output_dir="/mnt/hdd1/projects/RumorV2/results"
	batch_size=8
elif [ $(hostname) = "ED716" ]; then
	export CUDA_VISIBLE_DEVICES=1
	export WANDB_DIR="/mnt/1T/projects/RumorV2"
	output_dir="/mnt/1T/projects/RumorV2/results"
	batch_size=1
else
	export CUDA_VISIBLE_DEVICES=1
	export WANDB_DIR="/nfs/home/joshchang/projects/RumorV2"
	output_dir="/nfs/home/joshchang/projects/RumorV2/results"
	batch_size=16
fi

for dataset in semeval2019 twitter15 twitter16
do
	if [ $dataset = "PHEME" ]; then
		## Event-wise cross validation
		folds=$(seq 0 8)
	else
		folds=$(seq 0 4)
	fi

	for i in $folds
	do
		if [ "$i" = "comp" ]
		then
			## For semeval2019 fold [comp]
			eval_file=dev.csv
			test_file=test.csv
		else
			## For 5-fold
			eval_file=test.csv
			test_file=test.csv
		fi

		for num_clusters in $(seq 1 5)
		do
			## Build cluster summary pairs by kmeans for training
			python main.py \
				--task_type build_cluster_summary \
				--model_name_or_path facebook/bart-base \
				--cluster_type kmeans \
				--cluster_mode train \
				--num_clusters "$num_clusters" \
				--dataset_name "$dataset" \
				--fold "$i" \
				--per_device_train_batch_size "$batch_size"

			## Build cluster summary pairs by kmeans for testing
			#python main.py \
			#	--task_type build_cluster_summary \
			#	--model_name_or_path facebook/bart-base \
			#	--cluster_type kmeans \
			#	--cluster_mode test \
			#	--num_clusters "$num_clusters" \
			#	--dataset_name "$dataset" \
			#	--fold "$i" \
			#	--per_device_train_batch_size "$batch_size"
		done

		## Build cluster summary pairs based on topics
		#python main.py \
		#	--task_type build_cluster_summary \
		#	--model_name_or_path facebook/bart-base \
		#	--cluster_type topics \
		#	--dataset_name "$dataset" \
		#	--fold "$i" \
		#	--per_device_train_batch_size "$batch_size" \

	done
done