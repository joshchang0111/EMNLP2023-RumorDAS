#!bin/sh

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="RumorDAS"
export WANDB_DIR=... ## need to be defined
output_dir=/mnt/1T/projects/RumorDAS       ## need to be defined
batch_size=8
exp_name=bi-tgn/adv-stage2

#######################################
## Evaluate BiTGN w/ DAS, w/o Attack ##
#######################################
for extract_ratio in 0.05 0.1 0.15 0.2 0.25 0.5 0.75 0.9
do
	for dataset in re2019 twitter15 twitter16
	do
		for num_clusters in $(seq 1 5)
		do
			for i in $(seq 0 4)
			do
				## Defensive Response Extractor (DRE) - Cluster Only
				#python main.py \
				#	--task_type train_adv_stage2 \
				#	--attack_type untargeted \
				#	--model_name_or_path facebook/bart-base \
				#	--td_gcn \
				#	--bu_gcn \
				#	--num_clusters $num_clusters \
				#	--extractor_name_or_path kmeans \
				#	--dataset_name $dataset \
				#	--train_file train.csv \
				#	--validation_file $test_file \
				#	--fold $i \
				#	--do_eval \
				#	--exp_name bi-tgn/adv-stage2 \
				#	--output_dir $output_dir

				## Self-Supervised Response Abstractor (SSRA) Only
				#python main.py \
				#	--task_type train_adv_stage2 \
				#	--attack_type untargeted \
				#	--model_name_or_path facebook/bart-base \
				#	--td_gcn \
				#	--bu_gcn \
				#	--num_clusters $num_clusters \
				#	--extractor_name_or_path kmeans \
				#	--abstractor_name_or_path ssra_kmeans_$num_clusters \
				#	--summarizer_output_type ssra_only \
				#	--dataset_name $dataset \
				#	--train_file train.csv \
				#	--validation_file $test_file \
				#	--fold $i \
				#	--do_eval \
				#	--per_device_eval_batch_size $batch_size \
				#	--exp_name bi-tgn/adv-stage2 \
				#	--output_dir $output_dir

				## Defensive Response Extractor (DRE) - Filter Only
				#python main.py \
				#	--task_type train_adv_stage2 \
				#	--attack_type untargeted \
				#	--model_name_or_path facebook/bart-base \
				#	--td_gcn \
				#	--bu_gcn \
				#	--num_clusters $num_clusters \
				#	--filter_layer_enc 4 \
				#	--filter_layer_dec 4 \
				#	--extractor_name_or_path filter,kmeans \
				#	--filter_ratio $extract_ratio \
				#	--dataset_name $dataset \
				#	--train_file train.csv \
				#	--validation_file $test_file \
				#	--fold $i \
				#	--do_eval \
				#	--exp_name bi-tgn/adv-stage2 \
				#	--output_dir $output_dir

				## ====================================

				## Defend-And-Summarize (DAS) Framework
				python main.py \
					--task_type train_adv_stage2 \
					--attack_type untargeted \
					--model_name_or_path facebook/bart-base \
					--td_gcn \
					--bu_gcn \
					--num_clusters $num_clusters \
					--filter_layer_enc 4 \
					--filter_layer_dec 4 \
					--extractor_name_or_path filter,kmeans \
					--filter_ratio $extract_ratio \
					--abstractor_name_or_path ssra_kmeans_$num_clusters \
					--dataset_name $dataset \
					--train_file train.csv \
					--validation_file test.csv \
					--fold $i \
					--do_eval \
					--exp_name bi-tgn/adv-stage2 \
					--output_dir $output_dir

				## w/o summarizer
				#python main.py \
				#	--task_type train_adv_stage2 \
				#	--attack_type untargeted \
				#	--model_name_or_path facebook/bart-base \
				#	--td_gcn \
				#	--bu_gcn \
				#	--dataset_name $dataset \
				#	--train_file train.csv \
				#	--validation_file $test_file \
				#	--fold $i \
				#	--do_eval \
				#	--exp_name bi-tgn/adv-stage2 \
				#	--output_dir $output_dir
			done
		done
	done
done