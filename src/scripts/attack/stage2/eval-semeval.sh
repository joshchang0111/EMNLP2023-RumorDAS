#!bin/sh

export WANDB_PROJECT="RumorV2"

if [ $(hostname) = "josh-System-Product-Name" ]; then
	export WANDB_DIR="/mnt/hdd1/projects/RumorV2"
	output_dir="/mnt/hdd1/projects/RumorV2/results"
elif [ $(hostname) = "ED716" ]; then
	export CUDA_VISIBLE_DEVICES=0
	export WANDB_DIR="/mnt/1T/projects/RumorV2"
	output_dir="/mnt/1T/projects/RumorV2/results"
	batch_size=8
elif [ $(hostname) = "yisyuan-PC2" ]; then
	export CUDA_VISIBLE_DEVICES=1
	export WANDB_DIR="/home/joshchang/project/RumorV2"
	output_dir="/home/joshchang/project/RumorV2/results"
elif [ $(hostname) = "esc4000-g4" ]; then
	export CUDA_VISIBLE_DEVICES=0
	export WANDB_DIR="/nfs/home/joshchang/projects/RumorV2"
	output_dir="/nfs/home/joshchang/projects/RumorV2/results"
	batch_size=4
elif [ $(hostname) = "basic-4" ]; then
	export CUDA_VISIBLE_DEVICES=4
	export WANDB_DIR="/nfs/home/joshchang/projects/RumorV2"
	output_dir="/nfs/home/joshchang/projects/RumorV2/results"
	batch_size=16
fi

#########################################################
## Adv. Stage 2: train generator while fixing detector ##
#########################################################
## Evaluate the detector under attack ##

for dataset in semeval2019 twitter16
do
	if [ $dataset = "PHEME" ]; then
		## Event-wise cross validation
		folds=$(seq 0 8)
	else
		folds=$(seq 0 4)
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

		## ======================= ##
		## Evaluate w/o summarizer ##
		## ======================= ##
		#python main.py \
		#	--task_type train_adv_stage2 \
		#	--attack_type untargeted \
		#	--model_name_or_path facebook/bart-base \
		#	--td_gcn \
		#	--bu_gcn \
		#	--dataset_name "$dataset" \
		#	--train_file train.csv \
		#	--validation_file "$test_file" \
		#	--fold "$i" \
		#	--do_eval \
		#	--exp_name bi-tgn/adv-stage2 \
		#	--output_dir "$output_dir"

		## ======================= ##
		## Evaluate w/  summarizer ##
		## ======================= ##
		## RA-BART-base-SAMSum ##
		#python main.py \
		#	--task_type train_adv_stage2 \
		#	--attack_type untargeted \
		#	--model_name_or_path facebook/bart-base \
		#	--td_gcn \
		#	--bu_gcn \
		#	--abstractor_name_or_path lidiya/bart-base-samsum \
		#	--dataset_name "$dataset" \
		#	--train_file train.csv \
		#	--validation_file "$test_file" \
		#	--fold "$i" \
		#	--do_eval \
		#	--exp_name bi-tgn/adv-stage2 \
		#	--output_dir "$output_dir"

		## SSRA-LOO ##
		#python main.py \
		#	--task_type train_adv_stage2 \
		#	--attack_type untargeted \
		#	--model_name_or_path facebook/bart-base \
		#	--td_gcn \
		#	--bu_gcn \
		#	--abstractor_name_or_path ssra_loo \
		#	--dataset_name "$dataset" \
		#	--train_file train.csv \
		#	--validation_file "$test_file" \
		#	--fold "$i" \
		#	--do_eval \
		#	--exp_name bi-tgn/adv-stage2 \
		#	--output_dir "$output_dir"

		## Response Filter (AutoEncoder) ##
		#python main.py \
		#	--task_type train_adv_stage2 \
		#	--attack_type untargeted \
		#	--model_name_or_path facebook/bart-base \
		#	--td_gcn \
		#	--bu_gcn \
		#	--extractor_name_or_path autoencoder \
		#	--dataset_name "$dataset" \
		#	--train_file train.csv \
		#	--validation_file "$test_file" \
		#	--fold "$i" \
		#	--do_eval \
		#	--exp_name bi-tgn/adv-stage2 \
		#	--output_dir "$output_dir"

		## Response Filter + Clustering Extractor + SSRA-KMeans
		#python main.py \
		#	--task_type train_adv_stage2 \
		#	--attack_type untargeted \
		#	--model_name_or_path facebook/bart-base \
		#	--td_gcn \
		#	--bu_gcn \
		#	--extractor_name_or_path autoencoder,kmeans \
		#	--abstractor_name_or_path ssra_stage2 \
		#	--dataset_name "$dataset" \
		#	--train_file train.csv \
		#	--validation_file "$test_file" \
		#	--fold "$i" \
		#	--do_eval \
		#	--exp_name bi-tgn/adv-stage2 \
		#	--output_dir "$output_dir"
	done
done

## CARE (Different number of filter layers)
#for dataset in semeval2019 twitter15 twitter16
#do
#	if [ $dataset = "PHEME" ]; then
#		## Event-wise cross validation
#		folds=$(seq 0 8)
#	else
#		folds=$(seq 0 4)
#	fi
#
#	for filter_layer in 2 4 6
#	do
#		for i in $folds
#		do
#			if [ "$i" = "comp" ]
#			then
#				## For semeval2019 fold [comp]
#				eval_file=dev.csv
#				test_file=test.csv
#			else
#				## For 5-fold
#				eval_file=test.csv
#				test_file=test.csv
#			fi
#
#			## Cluster-Aware Response Extractor (CARE)
#			python main.py \
#				--task_type train_adv_stage2 \
#				--attack_type untargeted \
#				--model_name_or_path facebook/bart-base \
#				--td_gcn \
#				--bu_gcn \
#				--num_clusters 3 \
#				--filter_layer_enc $filter_layer \
#				--filter_layer_dec $filter_layer \
#				--extractor_name_or_path filter,kmeans \
#				--dataset_name $dataset \
#				--train_file train.csv \
#				--validation_file $test_file \
#				--fold $i \
#				--do_eval \
#				--exp_name bi-tgn/adv-stage2 \
#				--output_dir $output_dir
#		done
#	done
#done

## SSRA-KMeans
#for dataset in semeval2019 twitter15 twitter16
#do
#	if [ $dataset = "PHEME" ]; then
#		## Event-wise cross validation
#		folds=$(seq 0 8)
#	else
#		folds=$(seq 0 4)
#	fi
#
#	for num_clusters in $(seq 5 5)
#	do
#		for i in $folds
#		do
#			if [ "$i" = "comp" ]
#			then
#				## For semeval2019 fold [comp]
#				eval_file=dev.csv
#				test_file=test.csv
#			else
#				## For 5-fold
#				eval_file=test.csv
#				test_file=test.csv
#			fi
#		
#			python main.py \
#				--task_type train_adv_stage2 \
#				--attack_type untargeted \
#				--model_name_or_path facebook/bart-base \
#				--td_gcn \
#				--bu_gcn \
#				--extractor_name_or_path kmeans \
#				--abstractor_name_or_path ssra_kmeans_$num_clusters \
#				--num_clusters $num_clusters \
#				--dataset_name $dataset \
#				--train_file train.csv \
#				--validation_file $test_file \
#				--fold $i \
#				--do_eval \
#				--exp_name bi-tgn/adv-stage2 \
#				--output_dir $output_dir
#		done
#	done
#done

#####################################################
## Response Extractor with different extract ratio ##
#####################################################
#for extract_ratio in 0.05 0.1 0.15 0.2 0.25 0.5 0.75 0.9
for extract_ratio in 0.2 0.25 0.5 0.75 0.9
do
	for dataset in semeval2019 twitter15 twitter16
	do
		if [ $dataset = "PHEME" ]; then
			## Event-wise cross validation
			folds=$(seq 0 8)
		else
			folds=$(seq 0 4)
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

			## ==========================================================
			## Commands below are controlled by `--td_gcn` and `--bu_gcn`
			## Response Filter
			#python main.py \
			#	--task_type train_adv_stage2 \
			#	--attack_type untargeted \
			#	--model_name_or_path facebook/bart-base \
			#	--td_gcn \
			#	--bu_gcn \
			#	--extractor_name_or_path filter \
			#	--filter_ratio "$extract_ratio" \
			#	--dataset_name "$dataset" \
			#	--train_file train.csv \
			#	--validation_file "$test_file" \
			#	--fold "$i" \
			#	--do_eval \
			#	--exp_name bi-tgn/adv-stage2 \
			#	--output_dir "$output_dir"

			## Clustering Extractor
			#python main.py \
			#	--task_type train_adv_stage2 \
			#	--attack_type untargeted \
			#	--model_name_or_path facebook/bart-base \
			#	--td_gcn \
			#	--bu_gcn \
			#	--extractor_name_or_path kmeans \
			#	--extract_ratio "$extract_ratio" \
			#	--dataset_name "$dataset" \
			#	--train_file train.csv \
			#	--validation_file "$test_file" \
			#	--fold "$i" \
			#	--do_eval \
			#	--exp_name bi-tgn/adv-stage2 \
			#	--output_dir "$output_dir"

			## bi-tgn + (Response Filter + Clustering Extractor + SSRA)
			#python main.py \
			#	--task_type train_adv_stage2 \
			#	--attack_type untargeted \
			#	--model_name_or_path facebook/bart-base \
			#	--td_gcn \
			#	--bu_gcn \
			#	--extractor_name_or_path filter,kmeans \
			#	--filter_ratio "$extract_ratio" \
			#	--abstractor_name_or_path ssra_stage2_kmeans \
			#	--dataset_name "$dataset" \
			#	--train_file train.csv \
			#	--validation_file "$test_file" \
			#	--fold "$i" \
			#	--do_eval \
			#	--exp_name bi-tgn/adv-stage2 \
			#	--output_dir "$output_dir"

			## Cluster-Aware Response Extractor (CARE)
			#python main.py \
			#	--task_type train_adv_stage2 \
			#	--attack_type untargeted \
			#	--model_name_or_path facebook/bart-base \
			#	--td_gcn \
			#	--bu_gcn \
			#	--num_clusters 3 \
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
		done
	done
done

##################################################################
## CARS with different extract ratio and number of clusters (k) ##
##################################################################
#for extract_ratio in 0.05 0.1 0.15 0.2 0.25 0.5 0.75 0.9
for extract_ratio in 0.5 0.75 0.9
do
	for dataset in semeval2019 twitter16
	do
		if [ $dataset = "PHEME" ]; then
			## Event-wise cross validation
			folds=$(seq 0 8)
		else
			folds=$(seq 0 4)
		fi

		for num_clusters in $(seq 5 5)
		do
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
			
				## Cluster-Aware Response Summarizer (CARS)
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
					--validation_file $test_file \
					--fold $i \
					--do_eval \
					--exp_name bi-tgn/adv-stage2 \
					--output_dir $output_dir
			done
		done
	done
done