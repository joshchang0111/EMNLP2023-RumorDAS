#!bin/sh

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="RumorDAS"
export WANDB_DIR=... ## need to be defined
output_dir=/mnt/1T/projects/RumorDAS       ## need to be defined
batch_size=8
exp_name=bi-tgn/adv-stage2

#########################################################
## Adv. Stage 2: train generator while fixing detector ##
#########################################################

for dataset in re2019 twitter15 twitter16
do
	for i in $(seq 0 4)
	do	
		## ==========================================================
		## Commands below are controlled by `--td_gcn` and `--bu_gcn`

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
			--output_dir "$output_dir" \
			--report_to wandb
	done
done

