import ipdb
import math
import random
import logging
import numpy as np
import pandas as pd
import preprocessor as pre ## TweetPreprocessor
from tqdm import tqdm

import torch
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

## Call the same logger used in main.py
logger = logging.getLogger("__main__")

def build_datasets_loo_abstractor(data_args, model_args, training_args, config, tokenizer, model):
	"""Build (leave-one-out) datasets for fine-tuning abstractor"""

	##################
	## Load Dataset ##
	##################
	print("\nLoading dataset...")
	print("[{}]: fold [{}]".format(data_args.dataset_name, data_args.fold))

	## Loading a dataset from my local files.
	if training_args.do_train:
		data_files = {
			"train"     : "{}/{}/split_{}/{}".format(data_args.dataset_root, data_args.dataset_name, data_args.fold, data_args.train_file), 
			"validation": "{}/{}/split_{}/{}".format(data_args.dataset_root, data_args.dataset_name, data_args.fold, data_args.validation_file), 
			"test"      : "{}/{}/split_{}/{}".format(data_args.dataset_root, data_args.dataset_name, data_args.fold, data_args.validation_file)
		}
	elif training_args.do_eval:
		data_files = {
			"test"      : "{}/{}/split_{}/{}".format(data_args.dataset_root, data_args.dataset_name, data_args.fold, data_args.validation_file)
		}
	raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
	
	## Read tweet contents
	dataset_content = pd.read_csv("{}/{}/data.csv".format(data_args.dataset_root, data_args.dataset_name))
	dataset_content["source_id"] = dataset_content["source_id"].astype(str) ## For PHEME, twitter15, twitter16
	dataset_content["tweet_id"]  = dataset_content["tweet_id"].astype(str)
	dataset_content = dataset_content.set_index("tweet_id").T.to_dict() ## Each tweet_id maps to all information
	
	################
	## Preprocess ##
	################
	print("\nProcessing dataset...")

	## Padding strategy
	padding = "max_length" if data_args.pad_to_max_length else False
	data_args.max_tree_length = int(tokenizer.model_max_length / data_args.max_tweet_length) - 1
	
	if training_args.do_train:
		###################################################################
		## Load model for similarity measurement & pseudo label creation ##
		###################################################################
		sim_model_name =  "vinai/bertweet-base"
		print("Loading [{}] for similarity measurement...".format(sim_model_name))
		sim_tokenizer = AutoTokenizer.from_pretrained(sim_model_name)
		sim_model = AutoModel.from_pretrained(sim_model_name).cuda()

	## TODO:
	## - Rewrite preprocess_function
	def preprocess_train(examples):
		"""
		Preprocess function for abstractive response summarizer
		Input:
			- examples: keys = ["source_id", "tweet_ids", "label_veracity"]
		"""
		def parse_trees_from_str(input_trees):
			"""Parse each tree from string of tweet_ids to list"""
			output_trees = [tweet_ids_str.split(",") for tweet_ids_str in input_trees]
			assert len(output_trees) == len(input_trees)
			return output_trees

		def clean_text(raw_responses):
			"""Remove @'s and url's"""
			pre.set_options(pre.OPT.URL, pre.OPT.MENTION)
			clean_responses = [pre.tokenize(response).replace("$MENTION$", "").replace("$URL$", "") for response in raw_responses]
			return clean_responses

		def create_input_pseudo_summary_pair(tree_idx, inputs_for_similarity):
			"""Create pseudo summary label (of one tree) based on `sim_model's` embeddings."""
			start = (tree_idx + 0) * (data_args.max_tree_length + 1)
			for i in range(data_args.max_tree_length + 1 + 1):
				if i == (data_args.max_tree_length + 1):
					break
				if inputs_for_similarity["input_ids"][start + i] == pad_sequence["input_ids"]:
					break

			## Tree with no responses or only 1 response
			if i == 0 or i == 1:
				return [], [], []

			tree_inputs = {
				"input_ids": inputs_for_similarity["input_ids"][start:start + i], 
				"token_type_ids": inputs_for_similarity["token_type_ids"][start:start + i],
				"attention_mask": inputs_for_similarity["attention_mask"][start:start + i]
			}

			## Obtain embeddings
			with torch.no_grad():
				encoder_outputs = sim_model(
					input_ids=torch.LongTensor(tree_inputs["input_ids"]).cuda(),
					attention_mask=torch.LongTensor(tree_inputs["attention_mask"]).cuda(),
					token_type_ids=torch.LongTensor(tree_inputs["token_type_ids"]).cuda(),
					return_dict=True
				)

			## Calculate cosine similarity between each response
			tree_responses_idx, tree_summary_idx, tree_weights = [], [], []
			response_hidden_states = torch.mean(encoder_outputs["last_hidden_state"].cpu(), dim=1) ## average all tokens
			tree_length = response_hidden_states.shape[0]
			for ri in range(tree_length): ## Each response takes turns as summary label
				pseudo_summary = response_hidden_states[ri].view(1, -1)

				weights_i = []
				for rj in range(tree_length):
					if ri != rj:
						sim = F.cosine_similarity(pseudo_summary, response_hidden_states[rj].view(1, -1))
						weights_i.append(sim)

						#print("{} <-> {}: {:.4f}".format(ri, rj, sim.item()))

				weights_i = torch.exp(torch.cat(weights_i))
				weights_i = weights_i / torch.sum(weights_i)
				weights_i = weights_i * weights_i.shape[0]

				indices = list(range(start, start + tree_length))
				indices.remove(start + ri)
				tree_responses_idx.append(indices)
				tree_summary_idx.append(start + ri)
				tree_weights.append(weights_i)

			return tree_responses_idx, tree_summary_idx, tree_weights

		def map_content_by_idx(model_inputs, label_tokens, responses_idx, summary_idx):
			"""Map responses input and pseudo summary by index of model_inputs"""
			input_ids, attn_mask = [], []
			for indices in responses_idx: ## For each data pair
				sequence_ids, sequence_msk = [], []
				for idx in indices: ## For each response in the thread
					sequence_ids.extend(model_inputs["input_ids"][idx])
					sequence_msk.extend(model_inputs["attention_mask"][idx])
				
				## Pad to `max_tree_length`
				for _ in range(data_args.max_tree_length + 1 - len(indices)):
					sequence_ids.extend(pad_sequence["input_ids"])
					sequence_msk.extend(pad_sequence["attention_mask"])

				input_ids.append(sequence_ids)
				attn_mask.append(sequence_msk)
			
			labels = []
			for index in summary_idx: ## For each label
				labels.append(label_tokens["input_ids"][index])

			assert len(input_ids) == len(attn_mask) == len(labels)
			return input_ids, attn_mask, labels

		## ================================================
		trees = parse_trees_from_str(examples["tweet_ids"])
		responses, label_candidates = [], []
		for tree in trees: ## For each tree
			raw_responses = []
			for response_id in tree[1:]: ## Ignore source post
				response = dataset_content[response_id]["text"]
				raw_responses.append(response)

			raw_responses = raw_responses[:data_args.max_tree_length + 1] ## truncate
			raw_responses.extend([""] * (data_args.max_tree_length + 1 - len(raw_responses))) ## padding

			## Create pseudo summary labels
			clean_responses = clean_text(raw_responses)

			responses.extend(raw_responses)
			label_candidates.extend(clean_responses)
		
		## Tokenize for `similarity measurement`
		pad_sequence = sim_tokenizer("", max_length=data_args.max_tweet_length, padding=padding, truncation=True)
		inputs_for_similarity = sim_tokenizer(responses, max_length=data_args.max_tweet_length, padding=padding, truncation=True)

		responses_idx, summary_idx, weights = [], [], []
		for tree_idx in tqdm(range(len(trees)), desc="Creating pseudo summary"):
			tree_responses_idx, tree_summary_idx, tree_weights = create_input_pseudo_summary_pair(tree_idx, inputs_for_similarity)
			responses_idx.extend(tree_responses_idx)
			summary_idx.extend(tree_summary_idx)
			weights.extend(tree_weights)

		## Tokenize by bart tokenizer
		model_inputs = tokenizer(responses, max_length=data_args.max_tweet_length, padding=padding, truncation=True)
		with tokenizer.as_target_tokenizer():
			label_tokens = tokenizer(label_candidates, max_length=data_args.max_tweet_length, padding=padding, truncation=True)

		## Map inputs by `responses_idx`, `summary_idx`, `weights`
		input_ids, attn_mask, labels = map_content_by_idx(model_inputs, label_tokens, responses_idx, summary_idx)
		## Pad each weight to `max_tree_length`
		weights = [torch.cat((w, torch.ones(data_args.max_tree_length + 1 - len(w)))) for w in weights]

		## If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
		## padding in the loss.
		if padding == "max_length" and data_args.ignore_pad_token_for_loss:
			labels = [
				[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels
			]

		## Final assignment
		model_inputs["input_ids"] = input_ids
		model_inputs["attention_mask"] = attn_mask
		model_inputs["labels"] = labels
		model_inputs["redundancy_weights"] = weights
		
		return model_inputs

	def preprocess_test(examples):
		"""Preprocess the data for inference"""
		def parse_trees_from_str(input_trees):
			"""Parse each tree from string of tweet_ids to list"""
			output_trees = [tweet_ids_str.split(",") for tweet_ids_str in input_trees]
			assert len(output_trees) == len(input_trees)
			return output_trees

		def clean_text(raw_responses):
			"""Remove @'s and url's"""
			pre.set_options(pre.OPT.URL, pre.OPT.MENTION)
			clean_responses = [pre.tokenize(response).replace("$MENTION$", "").replace("$URL$", "") for response in raw_responses]
			return clean_responses
		
		def split_3_groups(trees):
			trees_new, source_ids = [], []
			for tree in tqdm(trees, desc="Split 3 groups"):
				src_id = tree[0]
				rids = tree[1:] ## Ignore source post
				rids = np.array(rids)

				## Set number of clusters
				num_clusters = 3
				while (num_clusters >= len(rids)) and (num_clusters != 1):
					num_clusters = math.ceil(num_clusters / 2)
				
				## Split index
				ridx = np.arange(len(rids))
				np.random.shuffle(ridx)
				groups_ridx = np.array_split(ridx, 3)

				for gidx, group_ridx in enumerate(groups_ridx):
					group_ridx.sort()
					subtree = rids[group_ridx].tolist()
					subtree.insert(0, src_id)
					
					trees_new.append(subtree)
					source_ids.append("{}_{}".format(src_id, gidx))
			
			return trees_new, source_ids

		## ===========================================================================
		max_tree_length = int(tokenizer.model_max_length / data_args.max_tweet_length)

		trees = parse_trees_from_str(examples["tweet_ids"])

		source_ids_group = None
		if data_args.split_3_groups:
			trees, source_ids_group = split_3_groups(trees)

		responses, empty_labels = [], [] ## List of strings, including padding nodes
		for tids in trees:
			rids = tids[1:] ## Ignore source post
			rids = rids[:max_tree_length] ## Truncation

			rtxt = [dataset_content[rid]["text"] for rid in rids]
			rtxt.extend([""] * (max_tree_length - len(rids))) ## Padding
			
			responses.extend(rtxt)
			empty_labels.append("")
		
		model_inputs = tokenizer(responses, max_length=data_args.max_tweet_length, padding=padding, truncation=True)
		with tokenizer.as_target_tokenizer():
			labels = tokenizer(empty_labels, max_length=data_args.max_tweet_length, padding=padding, truncation=True)

		input_ids = model_inputs["input_ids"]
		attn_mask = model_inputs["attention_mask"]

		source_ids, new_input_ids, new_attn_mask, new_labels = [], [], [], []
		for tidx, tids in enumerate(trees):
			r_input_ids = input_ids[tidx * max_tree_length:(tidx + 1) * max_tree_length]
			r_attn_mask = attn_mask[tidx * max_tree_length:(tidx + 1) * max_tree_length]

			r_input_ids = sum(r_input_ids, [])
			r_attn_mask = sum(r_attn_mask, [])

			if len(tids) == 1:
				continue

			src_id = tids[0] if source_ids_group is None else source_ids_group[tidx]

			source_ids.append(src_id)
			new_labels.append(labels["input_ids"][tidx])
			new_input_ids.append(r_input_ids)
			new_attn_mask.append(r_attn_mask)

		## If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
		## padding in the loss.
		if padding == "max_length" and data_args.ignore_pad_token_for_loss:
			labels = [
				[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels
			]
		
		## Final assignment
		model_inputs["input_ids"] = new_input_ids
		model_inputs["attention_mask"] = new_attn_mask
		model_inputs["labels"] = new_labels
		model_inputs["source_id"] = source_ids

		return model_inputs

	## ======================================================================
	preprocess_function = {
		"train"     : preprocess_train, 
		"validation": preprocess_train, 
		"test"      : preprocess_test
	}
	with training_args.main_process_first(desc="dataset map pre-processing"):
		## Separately process each sets of raw_datasets
		for key_data in raw_datasets.keys():
			raw_datasets[key_data] = raw_datasets[key_data].map(
				preprocess_function[key_data],
				batched=True,
				load_from_cache_file=not data_args.overwrite_cache,
				remove_columns=raw_datasets[key_data].column_names, ## Enable the function to return more samples than input
				desc="Running tokenizer on {} dataset".format(key_data)
			)

	####################
	## Build each set ##
	####################
	train_dataset, eval_dataset, test_dataset = None, None, None
	## Make train & evaluation dataset
	if training_args.do_train:
		if "train" not in raw_datasets:
			raise ValueError("--do_train requires a train dataset")
		if "validation" not in raw_datasets:
			raise ValueError("--do_train requires a validation dataset")
		train_dataset = raw_datasets["train"]
		eval_dataset = raw_datasets["validation"]
		
		## Shuffle train dataset
		train_dataset = train_dataset.shuffle()

	## Make test dataset
	if training_args.do_eval:
		if "test" not in raw_datasets:
			raise ValueError("--do_eval requires a test dataset")
		test_dataset = raw_datasets["test"]

	return train_dataset, eval_dataset, test_dataset

## ======================================================================================================
def build_datasets_clustering_abstractor(data_args, model_args, training_args, config, tokenizer, model):
	"""Build (cluster-summary) datasets for fine-tuning abstractor"""

	##################
	## Load Dataset ##
	##################
	print("\nLoading dataset...")
	print("[{}]: fold [{}]".format(data_args.dataset_name, data_args.fold))

	## Loading a dataset from my local files.
	data_files = {
		"train"     : "{}/{}/split_{}/{}".format(data_args.dataset_root, data_args.dataset_name, data_args.fold, data_args.train_file), 
		"validation": "{}/{}/split_{}/{}".format(data_args.dataset_root, data_args.dataset_name, data_args.fold, data_args.validation_file), 
		"test"      : "{}/{}/split_{}/{}".format(data_args.dataset_root, data_args.dataset_name, data_args.fold, data_args.validation_file)
	}
	raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)

	## Load cluster-summary pair ids
	assert (model_args.cluster_mode == "train" or model_args.cluster_mode == "test"), "Please specify the cluster_mode"
	assert (model_args.num_clusters is not None), "Please specify the number of clustres desired"
	print("Loading cluster-summary pair...")
	print("Num. clusters: {}".format(model_args.num_clusters))

	cluster_df = pd.read_csv("{}/{}/split_{}/cluster_summary/{}/{}-{}.csv".format(
		data_args.dataset_root, 
		data_args.dataset_name, 
		data_args.fold, 
		model_args.cluster_mode, 
		model_args.cluster_type, 
		model_args.num_clusters
	))
	cluster_df["source_id"] = cluster_df["source_id"].astype(str)

	## Read tweet contents
	dataset_content = pd.read_csv("{}/{}/data.csv".format(data_args.dataset_root, data_args.dataset_name))
	ignored = 0
	for src_id, group in dataset_content.groupby("source_id"):
		if len(group) == 1:
			ignored = ignored + 1
	dataset_content["source_id"] = dataset_content["source_id"].astype(str) ## For PHEME, twitter15, twitter16
	dataset_content["tweet_id"]  = dataset_content["tweet_id"].astype(str)
	dataset_content = dataset_content.set_index("tweet_id").T.to_dict() ## Each tweet_id maps to all information

	################
	## Preprocess ##
	################
	print("\nProcessing dataset...")

	## Padding strategy
	padding = "max_length" if data_args.pad_to_max_length else False
	
	def preprocess_function(examples):
		"""
		Preprocess function for abstractive response summarizer
		Input:
			- examples: keys = ["source_id", "tweet_ids", "label_veracity"]
		"""
		def parse_trees_from_str(input_trees):
			"""Parse each tree from string of tweet_ids to list"""
			output_trees = [tweet_ids_str.split(",") for tweet_ids_str in input_trees]
			assert len(output_trees) == len(input_trees)
			return output_trees

		def clean_text(text):
			"""Remove @'s and url's"""
			pre.set_options(pre.OPT.URL, pre.OPT.MENTION)
			return pre.tokenize(text).replace("$MENTION", "").replace("$URL$", "")

		def build_cluster_summary_pair(src_ids, trees):
			"""Build cluster-summary pair from `cluster_df`."""
			data_pairs = [] ## store each pair's tid
			sid_cids = [] ## store the source id and cluster id corresponding to data_pairs
			text_all = []
			text_tid_map = {} ## store each tweet's index in `text_all`
			for tree_idx, src_id in enumerate(src_ids):
				tree = trees[tree_idx]

				## Filter trees not in cluster_summary file (trees that have 1 or less response)
				if src_id in cluster_df["source_id"].unique():
					tweets_df = cluster_df.loc[cluster_df["source_id"] == src_id]

					if key_data == "test":
						new_df = tweets_df.copy()
						new_df["cluster_id"] = tweets_df["cluster_id"].apply(lambda x: x.split("_")[0])
						tweets_df = new_df

					if model_args.cluster_type == "kmeans":
						for cluster_id, cluster in tweets_df.groupby("cluster_id"):
							if key_data == "test": ## Take all responses as input when inference								
								input_tid = cluster["tweet_id"].unique().tolist()
							else:
								input_tid = cluster[cluster["is_centroid"] == 0]["tweet_id"].tolist()
							label_tid = cluster[cluster["is_centroid"] == 1]["tweet_id"].tolist()
							
							## Filter those clusters with only 1 response (816)
							if len(input_tid) == 0:
								continue
							data_pairs.append([input_tid, label_tid])
							sid_cids.append([src_id, cluster_id])
							
							## Record content_map
							for tid in cluster["tweet_id"]:
								text_tid_map[tid] = len(text_all)
								text_all.append(clean_text(dataset_content[tid]["text"]))
					elif model_args.cluster_type == "topics":
						for row_idx, row in tweets_df.iterrows(): ## Each row represents a cluster
							input_tid = row["tweet_ids"].split(",")
							label_tid = [row["centroid"]] ## To make the behavior unified

							input_tid.remove(label_tid[0])
							if len(input_tid) == 0: ## Filter those clusters with only 1 response
								continue
							data_pairs.append([input_tid, label_tid])
							sid_cids.append([src_id, row["cluster_id"]])

							## Record content_map
							for tid in row["tweet_ids"].split(","):
								text_tid_map[tid] = len(text_all)
								text_all.append(clean_text(dataset_content[tid]["text"]))
			return data_pairs, sid_cids, text_all, text_tid_map

		def map_final_results(model_inputs, model_labels, data_pairs, text_tid_map):
			pad_sequence = tokenizer("", max_length=data_args.max_tweet_length, padding=padding, truncation=True)
			max_tree_length = int(tokenizer.model_max_length / data_args.max_tweet_length)

			all_label_ids = []
			all_input_ids, all_attn_mask = [], []
			for input_tid, label_tid in data_pairs: ## For each pair
				## 1. Process input
				input_tid = input_tid[:max_tree_length] ## Truncation tree

				tree_input_ids, tree_attn_mask = [], []
				for tid in input_tid:
					text_idx = text_tid_map[tid]
					tree_input_ids.extend(model_inputs["input_ids"][text_idx])
					tree_attn_mask.extend(model_inputs["attention_mask"][text_idx])

				## Pad tree
				tree_input_ids.extend(pad_sequence["input_ids"] * (max_tree_length - len(input_tid)))
				tree_attn_mask.extend(pad_sequence["attention_mask"] * (max_tree_length - len(input_tid)))
				
				## 2. Process label
				text_idx = text_tid_map[label_tid[0]]
				tree_label_ids = model_labels["input_ids"][text_idx]

				## 3. Collect final data for training
				all_input_ids.append(tree_input_ids)
				all_attn_mask.append(tree_attn_mask)
				all_label_ids.append(tree_label_ids)
			
			assert len(all_input_ids) == len(all_attn_mask) == len(all_label_ids)
			return all_input_ids, all_attn_mask, all_label_ids

		## ======================================================================
		src_ids, trees = examples["source_id"], examples["tweet_ids"]
		src_ids = [str(src_id) for src_id in src_ids] ## Convert src_id to string
		trees = parse_trees_from_str(examples["tweet_ids"])

		data_pairs, sid_cids, text_all, text_tid_map = build_cluster_summary_pair(src_ids, trees)
		
		model_inputs = tokenizer(text_all, max_length=data_args.max_tweet_length, padding=padding, truncation=True)
		with tokenizer.as_target_tokenizer():
			model_labels = tokenizer(text_all, max_length=data_args.max_tweet_length, padding=padding, truncation=True)

		input_ids, attn_mask, label_ids = map_final_results(model_inputs, model_labels, data_pairs, text_tid_map)

		## Final assignment
		model_inputs["input_ids"] = input_ids
		model_inputs["attention_mask"] = attn_mask
		model_inputs["labels"] = label_ids
		
		## TODO: add source_id, cluster_id
		model_inputs["source_id"] = [src_id for src_id, cluster_id in sid_cids]
		model_inputs["cluster_id"] = [cluster_id for src_id, cluster_id in sid_cids]

		return model_inputs

	## ======================================================================
	with training_args.main_process_first(desc="dataset map pre-processing"):
		## Separately process each sets of raw_datasets
		for key_data in raw_datasets.keys():
			raw_datasets[key_data] = raw_datasets[key_data].map(
				preprocess_function,
				batched=True,
				load_from_cache_file=not data_args.overwrite_cache,
				remove_columns=raw_datasets[key_data].column_names, ## Enable the function to return more samples than input
				desc="Running tokenizer on {} dataset".format(key_data)
			)

	####################
	## Build each set ##
	####################
	train_dataset, eval_dataset, test_dataset = None, None, None
	## Make train & evaluation dataset
	if training_args.do_train:
		if "train" not in raw_datasets:
			raise ValueError("--do_train requires a train dataset")
		if "validation" not in raw_datasets:
			raise ValueError("--do_train requires a validation dataset")
		train_dataset = raw_datasets["train"]
		eval_dataset = raw_datasets["validation"]

		## Shuffle train dataset
		train_dataset = train_dataset.shuffle()

	## Make test dataset
	if training_args.do_eval:
		if "test" not in raw_datasets:
			raise ValueError("--do_eval requires a test dataset")
		test_dataset = raw_datasets["test"]
	
	return train_dataset, eval_dataset, test_dataset