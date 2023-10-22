export WANDB_PROJECT=RumorV2

if [ $(hostname) = "ED716" ]; then
	export CUDA_VISIBLE_DEVICES=1
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
	batch_size=4
elif [ $(hostname) = "basic-4" ]; then
	export CUDA_VISIBLE_DEVICES=0
	export WANDB_DIR="/nfs/home/joshchang/projects/RumorV2"
	output_dir="/nfs/home/joshchang/projects/RumorV2/results"
	batch_size=16
fi

################################################
## Self-Supervised Response Abstractor (SSRA) ##
################################################
##for dataset in semeval2019 twitter15 twitter16
#for dataset in twitter16
#do
#	if [ $dataset = "PHEME" ]; then
#		## Event-wise cross validation
#		folds=$(seq 0 8)
#	else
#		folds=$(seq 0 4)
#	fi
#
#	for i in $folds
#	do
#		if [ "$i" = "comp" ]; then
#			## For semeval2019 fold [comp]
#			eval_file=dev.csv
#			test_file=test.csv
#		else
#			## For 5-fold
#			eval_file=test.csv
#			test_file=test.csv
#		fi
#
#		################
#		## Evaluation ##
#		################
#
#		for num_clusters in $(seq 1 5)
#		do
#			python main.py \
#				--task_type ssra_kmeans \
#				--model_name_or_path lidiya/bart-base-samsum \
#				--cluster_type kmeans \
#				--cluster_mode train \
#				--num_clusters "$num_clusters" \
#				--dataset_name "$dataset" \
#				--train_file train.csv \
#				--validation_file "$eval_file" \
#				--fold "$i" \
#				--do_eval \
#				--exp_name ssra_kmeans_"$num_clusters" \
#				--output_dir "$output_dir"
#		done
#	done
#done

for dataset in semeval2019 twitter15 twitter16
do
	folds=$(seq 0 4)

	for i in $folds
	do
		eval_file=test.csv
		test_file=test.csv

		################
		## Prediction ##
		################

		for num_clusters in $(seq 3 3)
		do
			python main.py \
				--task_type ssra_kmeans \
				--model_name_or_path lidiya/bart-base-samsum \
				--cluster_type kmeans \
				--cluster_mode train \
				--num_clusters "$num_clusters" \
				--dataset_name "$dataset" \
				--train_file train.csv \
				--validation_file "$eval_file" \
				--fold "$i" \
				--do_eval \
				--min_target_length 10 \
				--max_target_length 128 \
				--exp_name ssra_kmeans_"$num_clusters" \
				--output_dir "$output_dir"
		done
	done
done
