#!bin/sh

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="RumorDAS"
export WANDB_DIR=... ## need to be defined
output_dir=...       ## need to be defined
batch_size=16
exp_name=bi-tgn-roberta

for dataset in re2019 twitter15 twitter16
do
	for i in $(seq 0 4)
	do
		##############
		## Evaluate ##
		##############
		python main.py \
			--task_type train_detector \
			--model_name_or_path roberta-base \
			--td_gcn \
			--bu_gcn \
			--dataset_name $dataset \
			--train_file train.csv \
			--validation_file test.csv \
			--fold $i \
			--do_eval \
			--exp_name $exp_name \
			--output_dir $output_dir

		########################
		## Obtain Predictions ##
		########################
		#python main.py \
		#	--task_type train_detector \
		#	--model_name_or_path roberta-base \
		#	--td_gcn \
		#	--bu_gcn \
		#	--dataset_name "$dataset" \
		#	--train_file train.csv \
		#	--validation_file test.csv \
		#	--fold "$i" \
		#	--do_predict \
		#	--exp_name bi-tgn-roberta/lr2e-5 \
		#	--output_dir $output_dir
	done
done
