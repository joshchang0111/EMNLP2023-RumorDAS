export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="RumorDAS"
export WANDB_DIR=... ## need to be defined
output_dir=...       ## need to be defined
batch_size=8

################################################
## Self-Supervised Response Abstractor (SSRA) ##
################################################
## Evaluation ##
for dataset in re2019 twitter15 twitter16
do
	for i in $(seq 0 4)
	do
		for num_clusters in $(seq 1 5)
		do
			python main.py \
				--task_type ssra_kmeans \
				--model_name_or_path lidiya/bart-base-samsum \
				--cluster_type kmeans \
				--cluster_mode train \
				--num_clusters $num_clusters \
				--dataset_name $dataset \
				--train_file train.csv \
				--validation_file test.csv \
				--fold $i \
				--do_eval \
				--exp_name ssra_kmeans_$num_clusters \
				--output_dir $output_dir
		done
	done
done

## Prediction ##
#for dataset in re2019 twitter15 twitter16
#do
#	for i in $(seq 0 4)
#	do
#		for num_clusters in $(seq 3 3)
#		do
#			python main.py \
#				--task_type ssra_kmeans \
#				--model_name_or_path lidiya/bart-base-samsum \
#				--cluster_type kmeans \
#				--cluster_mode train \
#				--num_clusters $num_clusters \
#				--dataset_name $dataset \
#				--train_file train.csv \
#				--validation_file test.csv \
#				--fold $i \
#				--do_eval \
#				--min_target_length 10 \
#				--max_target_length 128 \
#				--exp_name ssra_kmeans_$num_clusters \
#				--output_dir $output_dir
#		done
#	done
#done
