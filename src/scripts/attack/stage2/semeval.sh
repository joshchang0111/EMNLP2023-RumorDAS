#!bin/sh

export WANDB_PROJECT="RumorV2"

if [ $(hostname) = "josh-System-Product-Name" ]; then
	export WANDB_DIR="/mnt/hdd1/projects/RumorV2"
	output_dir="/mnt/hdd1/projects/RumorV2/results"
	batch_size=4
elif [ $(hostname) = "ED716" ]; then
	export CUDA_VISIBLE_DEVICES=1
	export WANDB_DIR="/mnt/1T/projects/RumorV2"
	output_dir="/mnt/1T/projects/RumorV2/results"
	gacl_path="/mnt/1T/projects/GACL"
	batch_size=8
elif [ $(hostname) = "yisyuan-PC2" ]; then
	export CUDA_VISIBLE_DEVICES=1
	export WANDB_DIR="/home/joshchang/project/RumorV2"
	output_dir="/home/joshchang/project/RumorV2/results"
	batch_size=4
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
	export CUDA_VISIBLE_DEVICES=4
	export WANDB_DIR="/nfs/home/joshchang/projects/RumorV2"
	output_dir="/nfs/home/joshchang/projects/RumorV2/results"
	gacl_path="/nfs/home/joshchang/projects/GACL"
	batch_size=16
fi

#########################################################
## Adv. Stage 2: train generator while fixing detector ##
#########################################################

## NOTE: 
## - twitter16-fold2: learning rate = 2e-5

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
		
		## ==========================================
		## Commands below are controlled by `--td_gcn` and `--bu_gcn`

		## transformer
		#python main.py \
		#	--task_type train_adv_stage2 \
		#	--model_name_or_path facebook/bart-base \
		#	--dataset_name "$dataset" \
		#	--train_file train.csv \
		#	--validation_file "$eval_file" \
		#	--fold "$i" \
		#	--do_train \
		#	--do_eval \
		#	--per_device_train_batch_size "$batch_size" \
		#	--learning_rate 2e-5 \
		#	--num_train_epochs 10 \
		#	--exp_name transformer/adv-stage2 \
		#	--output_dir "$output_dir" \
		#	--report_to wandb

		## bi-tgn
		python main.py \
			--task_type train_adv_stage2 \
			--model_name_or_path facebook/bart-base \
			--td_gcn \
			--bu_gcn \
			--dataset_name "$dataset" \
			--train_file train.csv \
			--validation_file "$eval_file" \
			--fold "$i" \
			--do_train \
			--per_device_train_batch_size "$batch_size" \
			--learning_rate 2e-5 \
			--num_train_epochs 10 \
			--exp_name bi-tgn/adv-stage2 \
			--output_dir "$output_dir" \
			--report_to wandb
	done
done

