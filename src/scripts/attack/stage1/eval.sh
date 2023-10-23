#!bin/sh

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="RumorDAS"
export WANDB_DIR=... ## need to be defined
output_dir=...       ## need to be defined
batch_size=8
exp_name=bi-tgn/adv-stage1

##############################################
## Adv. Stage 1: train detector & generator ##
##############################################
## Evaluate stage-1 detector ##

for dataset in re2019 twitter15 twitter16
do
	for i in $(seq 0 4)
	do
		## ==========================================================
		## Commands below are controlled by `--td_gcn` and `--bu_gcn`

		python main.py \
			--task_type train_adv_stage1 \
			--model_name_or_path facebook/bart-base \
			--td_gcn \
			--bu_gcn \
			--dataset_name $dataset \
			--train_file train.csv \
			--validation_file test.csv \
			--fold $i \
			--do_eval \
			--exp_name $exp_name \
			--output_dir $output_dir
	done
done
