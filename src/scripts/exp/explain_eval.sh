## Experiments: Framing Effect ##
## 	- to observe the framing effect of the original datasets, 
##	  whether there exists such responses in the original data

if [ $(hostname) = "ED716" ]; then
	export CUDA_VISIBLE_DEVICES=0
	export WANDB_DIR="/mnt/1T/projects/RumorV2"
	output_dir="/mnt/1T/projects/RumorV2/results"
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
	batch_size=16
elif [ $(hostname) = "basic-4" ]; then
	export CUDA_VISIBLE_DEVICES=7
	export WANDB_DIR="/nfs/home/joshchang/projects/RumorV2"
	output_dir="/nfs/home/joshchang/projects/RumorV2/results"
	batch_size=16
fi

eval_file=test.csv

#######################################
## Evaluate BiTGN w/ DAS, w/o Attack ##
#######################################
#for extract_ratio in 0.05 0.1 0.15 0.2 0.25 0.5 0.75 0.9
#for extract_ratio in 0.1 0.25 0.5 0.9
#for extract_ratio in 0.05 0.15 0.2 0.75
for extract_ratio in 1
do
	for dataset in semeval2019 twitter15 twitter16
	#for dataset in twitter15
	do
		folds=$(seq 0 4)

		for num_clusters in $(seq 1 5)
		#for num_clusters in $(seq 3 3)
		do
			for i in $folds
			do
				## For 5-fold
				eval_file=test.csv
				test_file=test.csv
				
				## Defensive Response Extractor (DRE) - Cluster Only
				#python exp.py \
				#	--explain \
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
				python exp.py \
					--explain \
					--task_type train_adv_stage2 \
					--attack_type untargeted \
					--model_name_or_path facebook/bart-base \
					--td_gcn \
					--bu_gcn \
					--num_clusters $num_clusters \
					--extractor_name_or_path kmeans \
					--abstractor_name_or_path ssra_kmeans_$num_clusters \
					--summarizer_output_type ssra_only \
					--dataset_name $dataset \
					--train_file train.csv \
					--validation_file $test_file \
					--fold $i \
					--do_eval \
					--per_device_eval_batch_size $batch_size \
					--exp_name bi-tgn/adv-stage2 \
					--output_dir $output_dir

				## Defensive Response Extractor (DRE) - Filter Only
				#python exp.py \
				#	--explain \
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
				#python exp.py \
				#	--explain \
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
				#	--abstractor_name_or_path ssra_kmeans_$num_clusters \
				#	--dataset_name $dataset \
				#	--train_file train.csv \
				#	--validation_file $test_file \
				#	--fold $i \
				#	--do_eval \
				#	--exp_name bi-tgn/adv-stage2 \
				#	--output_dir $output_dir

				## w/o summarizer
				#python exp.py \
				#	--explain \
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