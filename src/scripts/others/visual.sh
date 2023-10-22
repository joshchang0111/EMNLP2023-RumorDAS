export WANDB_PROJECT="RumorV2"

if [ $(hostname) = "josh-System-Product-Name" ]; then
	export WANDB_DIR="/mnt/hdd1/projects/RumorV2"
	output_dir="/mnt/hdd1/projects/RumorV2/results"
	batch_size=8
elif [ $(hostname) = "yisyuan-PC2" ]; then
	export CUDA_VISIBLE_DEVICES=1
	export WANDB_DIR="/home/joshchang/project/RumorV2"
	output_dir="/home/joshchang/project/RumorV2/results"
	batch_size=8
else
	export CUDA_VISIBLE_DEVICES=1
	export WANDB_DIR="/nfs/home/joshchang/projects/RumorV2"
	output_dir="/nfs/home/joshchang/projects/RumorV2/results"
	batch_size=16
fi

for dataset in semeval2019
do
	if [ $dataset = "PHEME" ]; then
		## Event-wise cross validation
		folds=$(seq 0 8)
	else
		folds=$(seq 0 4)
		folds=$(seq 0 0)
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

		python others/plot_tsne.py \
			--task_type train_adv_stage2 \
			--attack_type untargeted \
			--model_name_or_path facebook/bart-base \
			--add_gcn \
			--bi_gcn \
			--dataset_name "$dataset" \
			--train_file train.csv \
			--validation_file "$eval_file" \
			--fold "$i" \
			--do_eval \
			--exp_name bi-tgn/adv-stage2 \
			--output_dir "$output_dir"
	done
done