#############
## Dataset ##
#############
#python data/preprocess/preprocess.py --txt2csv
#python data/preprocess/preprocess.py --csv4hf
#python data/preprocess/preprocess.py --csv4hf --simple
#python data/preprocess/preprocess.py --csv4hf --simple --fold comp
#python data/preprocess/preprocess.py --processV2_stance
#python data/preprocess/preprocess.py --processV2 --dataset Pheme
#python data/preprocess/preprocess.py --processV2 --dataset twitter15
#python data/preprocess/preprocess.py --processV2 --dataset twitter16
#python data/preprocess/preprocess.py --pheme_event_wise
#python data/preprocess/preprocess.py --processV2_fold --dataset semeval2019
#python data/preprocess/preprocess.py --processV2_fold --dataset semeval2019 --fold comp
#python data/preprocess/preprocess.py --processV2_fold --dataset twitter15
#python data/preprocess/preprocess.py --processV2_fold --dataset twitter16
#python data/preprocess/preprocess.py --process_semeval2019_dev_set
#python data/preprocess/preprocess.py --create_tree_ids_file --dataset semeval2019
#python data/preprocess/preprocess.py --create_tree_ids_file --dataset twitter15
#python data/preprocess/preprocess.py --create_tree_ids_file --dataset twitter16

#python data/postprocess.py --wordcloud
#python data/postprocess.py --get_event
#python data/postprocess.py --get_event_from_pheme
#python data/topic_model.py

## PHEME_veracity: raw -> processed
#python data/preprocess/preprocess_pheme.py --dataset PHEME_veracity --preprocess
#python data/preprocess/preprocess_pheme.py --dataset PHEME --split_5_fold
#python data/preprocess/preprocess_pheme.py --dataset PHEME --split_event_wise

## twitter15, twitter16
#python data/preprocess/preprocess_twitter.py --recover_4_classes --dataset twitter15
#python data/preprocess/preprocess_twitter.py --recover_4_classes --dataset twitter16
#python data/preprocess/preprocess_twitter.py --process_twitter16 --dataset twitter16

## Build graph dataset
#python data/graph_dataset.py --dataset_name semeval2019
#python data/graph_dataset.py --dataset_name twitter15
#python data/graph_dataset.py --dataset_name twitter16
#python data/graph_dataset.py --dataset_name PHEME

## Build topics
#python data/topic_model.py --dataset_name semeval2019
#python data/topic_model.py --dataset_name twitter15
#python data/topic_model.py --dataset_name twitter16


#for dataset in twitter15 twitter16 semeval2019
#do
#	## Statistics
#	#python data/preprocess/statistics.py --dataset $dataset
#
#	## 2023/8/3 - Obtain topics via clustering
#	#python data/cluster_topics.py \
#	#	--split_data_via_cluster \
#	#	--select_number_of_clusters \
#	#	--dataset_name $dataset
#	#python data/cluster_topics.py \
#	#	--split_data_via_cluster \
#	#	--dataset_name $dataset
#done