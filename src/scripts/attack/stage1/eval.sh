#!bin/sh

export WANDB_PROJECT="RumorV2"

if [ $(hostname) = "josh-System-Product-Name" ]; then
	export WANDB_DIR="/mnt/hdd1/projects/RumorV2"
	output_dir="/mnt/hdd1/projects/RumorV2/results"
elif [ $(hostname) = "ED716" ]; then
	export CUDA_VISIBLE_DEVICES=1
	export WANDB_DIR="/mnt/1T/projects/RumorV2"
	output_dir="/mnt/1T/projects/RumorV2/results"
	batch_size=8
elif [ $(hostname) = "yisyuan-PC2" ]; then
	export CUDA_VISIBLE_DEVICES=1
	export WANDB_DIR="/home/joshchang/project/RumorV2"
	output_dir="/home/joshchang/project/RumorV2/results"
elif [ $(hostname) = "esc4000-g4" ]; then
	export CUDA_VISIBLE_DEVICES=0
	export WANDB_DIR="/nfs/home/joshchang/projects/RumorV2"
	output_dir="/nfs/home/joshchang/projects/RumorV2/results"
	batch_size=4
elif [ $(hostname) = "basic-4" ]; then
	export CUDA_VISIBLE_DEVICES=0
	export WANDB_DIR="/nfs/home/joshchang/projects/RumorV2"
	output_dir="/nfs/home/joshchang/projects/RumorV2/results"
	batch_size=16
fi

##############################################
## Adv. Stage 1: train detector & generator ##
##############################################
## Evaluate stage-1 detector ##

#for dataset in PHEME
for dataset in twitter16
#for dataset in twitter15
#for dataset in semeval2019 twitter15 twitter16
#for dataset in semeval2019
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

		## ==========================================
		## Commands below are controlled by `--td_gcn` and `--bu_gcn`

		## transformer
		#python main.py \
		#	--task_type train_adv_stage1 \
		#	--model_name_or_path facebook/bart-base \
		#	--dataset_name "$dataset" \
		#	--train_file train.csv \
		#	--validation_file "$eval_file" \
		#	--fold "$i" \
		#	--do_eval \
		#	--exp_name transformer/adv-stage1 \
		#	--output_dir "$output_dir"

		## bi-tgn
		python main.py \
			--task_type train_adv_stage1 \
			--model_name_or_path facebook/bart-base \
			--td_gcn \
			--bu_gcn \
			--dataset_name "$dataset" \
			--train_file train.csv \
			--validation_file "$eval_file" \
			--fold "$i" \
			--do_eval \
			--exp_name bi-tgn/adv-stage1 \
			--output_dir "$output_dir"

		## bi-wetgn
		#python main.py \
		#	--task_type train_adv_stage1 \
		#	--model_name_or_path facebook/bart-base \
		#	--td_gcn \
		#	--bu_gcn \
		#	--edge_filter \
		#	--dataset_name "$dataset" \
		#	--train_file train.csv \
		#	--validation_file "$eval_file" \
		#	--fold "$i" \
		#	--do_eval \
		#	--exp_name bi-tgn/adv-stage1 \
		#	--output_dir "$output_dir"
	done
done
