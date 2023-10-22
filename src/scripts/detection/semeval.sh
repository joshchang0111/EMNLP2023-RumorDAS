#!bin/sh

export WANDB_PROJECT="RumorV2"

if [ $(hostname) = "ED716" ]; then
	export CUDA_VISIBLE_DEVICES=1
	export WANDB_DIR="/mnt/1T/projects/RumorV2"
	output_dir="/mnt/1T/projects/RumorV2/results"
	gacl_path="/mnt/1T/projects/GACL"
	batch_size=8
elif [ $(hostname) = "esc4000-g4" ]; then
	export CUDA_VISIBLE_DEVICES=0
	export WANDB_DIR="/nfs/home/joshchang/projects/RumorV2"
	output_dir="/nfs/home/joshchang/projects/RumorV2/results"
	batch_size=4
elif [ $(hostname) = "basic-1" ]; then
	export CUDA_VISIBLE_DEVICES=0
	export WANDB_DIR="/nfs/home/joshchang/projects/RumorV2"
	output_dir="/nfs/home/joshchang/projects/RumorV2/results"
	gacl_path="/nfs/home/joshchang/projects/GACL"
	batch_size=16
elif [ $(hostname) = "basic-4" ]; then
	export CUDA_VISIBLE_DEVICES=7
	export WANDB_DIR="/nfs/home/joshchang/projects/RumorV2"
	output_dir="/nfs/home/joshchang/projects/RumorV2/results"
	gacl_path="/nfs/home/joshchang/projects/GACL"
	batch_size=16
fi

for dataset in semeval2019
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

		###########
		## Train ##
		###########
		## bi-tgn-roberta
		#python main.py \
		#	--task_type train_detector \
		#	--model_name_or_path roberta-base \
		#	--td_gcn \
		#	--bu_gcn \
		#	--dataset_name "$dataset" \
		#	--train_file train.csv \
		#	--validation_file "$eval_file" \
		#	--fold "$i" \
		#	--do_train \
		#	--per_device_train_batch_size "$batch_size" \
		#	--learning_rate 2e-5 \
		#	--num_train_epochs 10 \
		#	--exp_name bi-tgn-roberta/lr2e-5 \
		#	--output_dir "$output_dir"
		#	#--save_strategy no \

		##############
		## Evaluate ##
		##############
		#python main.py \
		#	--task_type train_detector \
		#	--model_name_or_path roberta-base \
		#	--td_gcn \
		#	--bu_gcn \
		#	--dataset_name "$dataset" \
		#	--train_file train.csv \
		#	--validation_file "$test_file" \
		#	--fold "$i" \
		#	--do_eval \
		#	--exp_name bi-tgn-roberta/lr2e-5 \
		#	--output_dir $output_dir

		########################
		## Obtain Predictions ##
		########################
		python main.py \
			--task_type train_detector \
			--model_name_or_path roberta-base \
			--td_gcn \
			--bu_gcn \
			--dataset_name "$dataset" \
			--train_file train.csv \
			--validation_file "$test_file" \
			--fold "$i" \
			--do_predict \
			--exp_name bi-tgn-roberta/lr2e-5 \
			--output_dir $output_dir
	done
done
