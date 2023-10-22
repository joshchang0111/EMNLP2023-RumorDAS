export CUDA_VISIBLE_DEVICES=0
DATASET=semeval2019
FOLD=0
OUTPUT_DIR=/mnt/1T/projects/RumorV2/results

## Model Config.
NUM_CLUSTERS=3
EXTRACT_RATIO=0.25

## Load bi-tgn for demo
#streamlit run demo.py \
#	--server.headless true \
#	-- \
#	--task_type train_adv_stage1 \
#	--model_name_or_path facebook/bart-base \
#	--td_gcn \
#	--bu_gcn \
#	--dataset_name $DATASET \
#	--fold $FOLD \
#	--exp_name bi-tgn/adv-stage1 \
#	--output_dir $OUTPUT_DIR

## Load bi-tgn with DAS for demo
streamlit run demo.py \
	--server.headless true \
	--server.port 8000 \
	-- \
	--task_type train_adv_stage2 \
	--model_name_or_path facebook/bart-base \
	--td_gcn \
	--bu_gcn \
	--num_clusters $NUM_CLUSTERS \
	--filter_layer_enc 4 \
	--filter_layer_dec 4 \
	--extractor_name_or_path filter,kmeans \
	--filter_ratio $EXTRACT_RATIO \
	--abstractor_name_or_path ssra_kmeans_$NUM_CLUSTERS \
	--dataset_name $DATASET \
	--fold $FOLD \
	--exp_name bi-tgn/adv-stage1 \
	--output_dir $OUTPUT_DIR
