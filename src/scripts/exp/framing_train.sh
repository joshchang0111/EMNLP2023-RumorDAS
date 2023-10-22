## Experiments: Framing Effect ##
## 	- to observe the framing effect of the original datasets, 
##	  whether there exists such responses in the original data

if [ $(hostname) = "ED716" ]; then
	export CUDA_VISIBLE_DEVICES=1
	export WANDB_DIR="/mnt/1T/projects/RumorV2"
	output_dir="/mnt/1T/projects/RumorV2/results"
	batch_size=8
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

#for dataset in semeval2019 twitter15 twitter16
for dataset in twitter16
do
	folds=$(seq 1 4)
	for fold in $folds
	do
		## Train bi-tgn-roberta
		#python exp.py \
		#	--framing \
		#	--task_type train_detector \
		#	--model_name_or_path roberta-base \
		#	--td_gcn \
		#	--bu_gcn \
		#	--dataset_name $dataset \
		#	--train_file train.csv \
		#	--validation_file $eval_file \
		#	--fold $fold \
		#	--do_train \
		#	--per_device_train_batch_size "$batch_size" \
		#	--learning_rate 2e-5 \
		#	--num_train_epochs 10 \
		#	--exp_name exp/bi-tgn-roberta \
		#	--output_dir "$output_dir"
		
		## Train bi-tgn-bart
		#python exp.py \
		#	--framing \
		#	--task_type train_detector \
		#	--model_name_or_path facebook/bart-base \
		#	--td_gcn \
		#	--bu_gcn \
		#	--dataset_name $dataset \
		#	--train_file train.csv \
		#	--validation_file $eval_file \
		#	--fold $fold \
		#	--do_train \
		#	--per_device_train_batch_size "$batch_size" \
		#	--learning_rate 2e-5 \
		#	--num_train_epochs 10 \
		#	--exp_name exp/bi-tgn-bart \
		#	--output_dir "$output_dir"

		## Train transformer-bart, max_tree_len set to 16
		python exp.py \
			--framing \
			--task_type train_adv_stage1 \
			--model_name_or_path facebook/bart-base \
			--dataset_name $dataset \
			--train_file train.csv \
			--validation_file $eval_file \
			--fold $fold \
			--do_train \
			--per_device_train_batch_size $batch_size \
			--learning_rate 2e-5 \
			--num_train_epochs 10 \
			--exp_name exp/transformer-bart \
			--output_dir $output_dir
	done
done