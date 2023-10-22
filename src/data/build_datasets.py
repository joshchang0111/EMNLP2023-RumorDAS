import ipdb
import math
import random
import logging
import pandas as pd
from datasets import load_dataset

## Self-defined
from .build_datasets_adv import build_datasets_adv
from .build_datasets_filter import build_datasets_filter
from .build_datasets_abstractor import build_datasets_loo_abstractor, build_datasets_clustering_abstractor
from .build_datasets_clustering import build_datasets_clustering

## Call the same logger used in main.py
logger = logging.getLogger("__main__")

def build_datasets(data_args, model_args, training_args, config, tokenizer, model):
	"""Build datasets according to different tasks"""
	
	if training_args.task_type == "train_detector" or \
	   training_args.task_type == "train_adv_stage1" or \
	   training_args.task_type == "train_adv_stage2":
		train_dataset, eval_dataset, test_dataset = build_datasets_adv(
			data_args, model_args, training_args, 
			config, tokenizer, model
		)
	elif training_args.task_type == "train_filter":
		train_dataset, eval_dataset, test_dataset = build_datasets_filter(
			data_args, model_args, training_args, 
			config, tokenizer, model
		)
	elif training_args.task_type == "build_cluster_summary":
		train_dataset, eval_dataset, test_dataset = build_datasets_clustering(
			data_args, model_args, training_args, 
			config, tokenizer, model
		)
	elif training_args.task_type == "ssra_loo":
		train_dataset, eval_dataset, test_dataset = build_datasets_loo_abstractor(
			data_args, model_args, training_args, 
			config, tokenizer, model
		)
	elif training_args.task_type == "ssra_kmeans":
		train_dataset, eval_dataset, test_dataset = build_datasets_clustering_abstractor(
			data_args, model_args, training_args, 
			config, tokenizer, model
		)
	else:
		raise ValueError("training_args.task_type not specified!")		
	return train_dataset, eval_dataset, test_dataset