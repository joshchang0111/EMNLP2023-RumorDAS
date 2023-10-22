#!bin/sh

export WANDB_PROJECT="RumorV2"

if [ $(hostname) = "ED716" ]; then
	export CUDA_VISIBLE_DEVICES=0
	export WANDB_DIR="/mnt/1T/projects/RumorV2"
	output_dir="/mnt/1T/projects/RumorV2/results"
	batch_size=128
	lr=2e-5
elif [ $(hostname) = "esc4000-g4" ]; then
	export CUDA_VISIBLE_DEVICES=0
	export WANDB_DIR="/nfs/home/joshchang/projects/RumorV2"
	output_dir="/nfs/home/joshchang/projects/RumorV2/results"
	batch_size=4
elif [ $(hostname) = "basic-1" ]; then
	export CUDA_VISIBLE_DEVICES=0
	export WANDB_DIR="/nfs/home/joshchang/projects/RumorV2"
	output_dir="/nfs/home/joshchang/projects/RumorV2/results"
	batch_size=4
elif [ $(hostname) = "basic-4" ]; then
	export CUDA_VISIBLE_DEVICES=2
	export WANDB_DIR="/nfs/home/joshchang/projects/RumorV2"
	output_dir="/nfs/home/joshchang/projects/RumorV2/results"
	batch_size=256
	lr=4e-5
fi

#############################################
## Transformer AutoEncoder Response Filter ##
#############################################
for dataset in twitter16
do
	if [ $dataset = "PHEME" ]; then
		## Event-wise cross validation
		folds=$(seq 0 8)
	else
		folds=$(seq 0 4)
		#folds=$(seq 0 0)
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

		## layers=2+2
		#python main.py \
		#	--task_type train_filter \
		#	--model_name_or_path facebook/bart-base \
		#	--filter_layer_enc 2 \
		#	--filter_layer_dec 2 \
		#	--dataset_name "$dataset" \
		#	--train_file train.csv \
		#	--validation_file "$eval_file" \
		#	--fold "$i" \
		#	--do_train \
		#	--per_device_train_batch_size "$batch_size" \
		#	--learning_rate 2e-5 \
		#	--num_train_epochs 50 \
		#	--exp_name filter \
		#	--output_dir "$output_dir"

		for n_layer in 2 4 6
		do
			python main.py \
				--task_type train_filter \
				--model_name_or_path facebook/bart-base \
				--filter_layer_enc $n_layer \
				--filter_layer_dec $n_layer \
				--dataset_name $dataset \
				--train_file train.csv \
				--validation_file $eval_file \
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
