#python others/plot.py --plot_response_impact --dataset_name semeval2019
#python others/plot.py --plot_response_impact --dataset_name twitter15
#python others/plot.py --plot_response_impact --dataset_name twitter16

#python others/plot.py --plot_extract_ratio --dataset_name twitter15
#python others/plot.py --plot_extract_ratio --dataset_name twitter16
#python others/plot.py --plot_extract_ratio --dataset_name semeval2019

#python others/plot.py --plot_all_extract_ratio
CUDA_VISIBLE_DEVICES=1 python data/preprocess/plot_tsne.py