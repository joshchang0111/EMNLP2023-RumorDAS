export WANDB_PROJECT=RumorV2

if [ $(hostname) = "josh-System-Product-Name" ]; then
	export WANDB_DIR="/mnt/hdd1/projects/RumorV2"
	output_dir="/mnt/hdd1/projects/RumorV2/results"
	batch_size=4
elif [ $(hostname) = "ED716" ]; then
	export CUDA_VISIBLE_DEVICES=0
	export WANDB_DIR="/mnt/1T/projects/RumorV2"
	output_dir="/mnt/1T/projects/RumorV2/results"
	batch_size=4
	lr=2e-5
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
	batch_size=4
elif [ $(hostname) = "basic-4" ]; then
	export CUDA_VISIBLE_DEVICES=6
	export WANDB_DIR="/nfs/home/joshchang/projects/RumorV2"
	output_dir="/nfs/home/joshchang/projects/RumorV2/results"
	batch_size=16
	lr=4e-5
fi

################################################
## Self-Supervised Response Abstractor (SSRA) ##
################################################
for dataset in twitter16
do
	if [ $dataset = "PHEME" ]; then
		## Event-wise cross validation
		folds=$(seq 0 8)
	else
		folds=$(seq 0 4)
	fi

	for i in $folds
	do
		if [ "$i" = "comp" ]; then
			## For semeval2019 fold [comp]
			eval_file=dev.csv
			test_file=test.csv
		else
			## For 5-fold
			eval_file=test.csv
			test_file=test.csv
		fi
		
		#########################
		## LOO (Leave-One-Out) ##
		#########################
		#python main.py \
		#	--task_type ssra_loo \
		#	--model_name_or_path lidiya/bart-base-samsum \
		#	--dataset_name "$dataset" \
		#	--train_file train.csv \
		#	--validation_file test.csv \
		#	--fold "$i" \
		#	--do_train \
		#	--do_eval \
		#	--per_device_train_batch_size "$batch_size" \
		#	--learning_rate 8e-6 \
		#	--num_train_epochs 10 \
		#	--exp_name ssra_loo \
		#	--output_dir "$output_dir"
		#	#--report_to wandb
		#	## For debugging
		#	#--evaluation_strategy steps \
		#	#--eval_steps 10

		###################################
		## KMeans (cluster-summary pair) ##
		###################################
		## Load loo model
		#python main.py \
		#	--task_type ssra_kmeans \
		#	--model_name_or_path ssra_kmeans \
		#	--dataset_name "$dataset" \
		#	--train_file train.csv \
		#	--validation_file test.csv \
		#	--fold "$i" \
		#	--do_train \
		#	--do_eval \
		#	--per_device_train_batch_size "$batch_size" \
		#	--learning_rate 1e-5 \
		#	--num_train_epochs 10 \
		#	--exp_name ssra_kmeans \
		#	--output_dir "$output_dir" \
		#	--report_to wandb

		for num_clusters in $(seq 1 5)
		do
			## Load bart-base-samsum
			python main.py \
				--task_type ssra_kmeans \
				--model_name_or_path lidiya/bart-base-samsum \
				--cluster_type kmeans \
				--cluster_mode train \
				--num_clusters "$num_clusters" \
				--dataset_name "$dataset" \
				--train_file train.csv \
				--validation_file test.csv \
				--fold "$i" \
				--do_train \
				--per_device_train_batch_size "$batch_size" \
				--learning_rate 2e-5 \
				--num_train_epochs 10 \
				--exp_name ssra_kmeans_"$num_clusters" \
				--output_dir "$output_dir" \
				--report_to wandb
		done
	done
done
