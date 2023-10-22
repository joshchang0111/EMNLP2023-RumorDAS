export CUDA_VISIBLE_DEVICES=0

for dataset in semeval2019 twitter15 twitter16
do
	#########################
	## Evaluate Perplexity ##
	#########################
	#for num_clusters in $(seq 1 5)
	#do
	#	python others/evaluate_summary.py \
	#		--eval_ppl \
	#		--model_type ssra_kmeans_"$num_clusters" \
	#		--data_name "$dataset" \
	#		--data_root_V2 ../dataset/processedV2
	#done
	#
	#python others/evaluate_summary.py \
	#	--eval_ppl \
	#	--model_type ra \
	#	--data_name "$dataset" \
	#	--data_root_V2 ../dataset/processedV2
	#
	#python others/evaluate_summary.py \
	#	--eval_ppl \
	#	--model_type ssra_loo \
	#	--data_name "$dataset" \
	#	--data_root_V2 ../dataset/processedV2

	#python others/evaluate_summary.py \
	#	--eval_ppl \
	#	--model_type chatgpt \
	#	--data_name $dataset \
	#	--data_root_V2 ../dataset/processedV2

	####################################
	## Generate data files for factCC ##
	####################################
	#for factCC_format in all_responses response_wise
	for factCC_format in all_responses
	do
		#for num_clusters in $(seq 1 5)
		#do
		#	python others/evaluate_summary.py \
		#		--generate_for_factCC \
		#		--factCC_format "$factCC_format" \
		#		--model_type ssra_kmeans_"$num_clusters" \
		#		--data_name "$dataset" \
		#		--data_root_V2 ../dataset/processedV2
		#done
		#
		#python others/evaluate_summary.py \
		#	--generate_for_factCC \
		#	--factCC_format "$factCC_format" \
		#	--model_type ra \
		#	--data_name "$dataset" \
		#	--data_root_V2 ../dataset/processedV2
		#
		#python others/evaluate_summary.py \
		#	--generate_for_factCC \
		#	--factCC_format "$factCC_format" \
		#	--model_type ssra_loo \
		#	--data_name "$dataset" \
		#	--data_root_V2 ../dataset/processedV2
		
		python others/evaluate_summary.py \
			--generate_for_factCC \
			--factCC_format $factCC_format \
			--model_type chatgpt \
			--data_name $dataset \
			--data_root_V2 ../dataset/processedV2
	done

	#for factCC_format in cluster_wise
	#do
	#	for num_clusters in $(seq 2 5)
	#	do
	#		python others/evaluate_summary.py \
	#			--generate_for_factCC \
	#			--factCC_format "$factCC_format" \
	#			--model_type ssra_kmeans_"$num_clusters" \
	#			--data_name "$dataset" \
	#			--data_root_V2 ../dataset/processedV2
	#	done
	#done
done