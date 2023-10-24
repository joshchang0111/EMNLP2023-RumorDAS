import ipdb
import json
import random
import logging
import pandas as pd

from torch.utils.data import DataLoader

from datasets import load_dataset

## Call the same logger used in main_summ.py
logger = logging.getLogger("__main__")

def build_datasets_clustering(data_args, model_args, training_args, config, tokenizer, model):
	"""Build datasets for building cluster summary dataset."""
	##################
	## Load Dataset ##
	##################
	print("\nLoading dataset...")
	print("[{}]: fold [{}]".format(data_args.dataset_name, data_args.fold))

	data_files = {"train": "{}/{}/data_tree_ids.csv".format(data_args.dataset_root, data_args.dataset_name)}
	raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)

	## Read tweet contents
	data_df = pd.read_csv("{}/{}/data.csv".format(data_args.dataset_root, data_args.dataset_name))
	data_df["source_id"] = data_df["source_id"].astype(str) ## For PHEME, twitter15, twitter16
	data_df["tweet_id"]  = data_df["tweet_id"].astype(str)
	dataset_content = data_df.set_index("tweet_id").T.to_dict() ## Each tweet_id maps to all information

	################
	## Preprocess ##
	################
	print("\nProcessing dataset...")

	## Padding strategy
	max_target_length = data_args.max_target_length ## For generator
	padding = "max_length" if data_args.pad_to_max_length else False
	
	assert data_args.max_seq_length * data_args.max_tree_length <= tokenizer.model_max_length, \
	"Max length of tree sequence ({}) larger than the max input length for the model ({})!".format(
		data_args.max_seq_length * data_args.max_tree_length, tokenizer.model_max_length
	)
	max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

	def preprocess_function(examples):
		"""
		Input:
			- examples: keys = ["source_id", "tweet_ids"]
		"""
		def parse_trees_from_str(input_trees):
			"""Parse each tree from string of tweet_ids to list"""
			output_trees = [tweet_ids_str.split(",") for tweet_ids_str in input_trees]
			assert len(output_trees) == len(input_trees)
			return output_trees

		def id2text(trees):
			"""
			Convert all tweet ids to corresponding text
			Return
				- tweet_ids: list of tweet id that maps `texts`
				- texts    : list of all tweet texts
			"""
			tweet_ids = [tweet_id for tree in trees for tweet_id in tree]
			texts = [dataset_content[tweet_id]["text"] for tweet_id in tweet_ids]
			return tweet_ids, texts

		def formulate_one_tree(model_inputs, start_idx, end_idx):
			"""Take the whole tree completely without truncating."""
			input_ids = model_inputs["input_ids"][start_idx:end_idx]
			attn_mask = model_inputs["attention_mask"][start_idx:end_idx]
			
			## Padding to make each tree have 300 nodes
			padding = [-1] * data_args.max_tweet_length
			input_ids.extend([padding] * (300 - len(input_ids)))
			attn_mask.extend([padding] * (300 - len(attn_mask)))
			return input_ids, attn_mask

		def extract_topic_words_probs(tree_topic):
			"""
			Input:
				- tree_topic: dict, all topics of a tree
			"""
			top_k_topic_words = 10
			corpus, topic_words, topic_probs = [], [], []
			for topic_id in range(len(tree_topic)):
				topic_words_probs = tree_topic[str(topic_id)][:top_k_topic_words] ## Pick top k representative words
				w = [pair[0] for pair in topic_words_probs]
				p = [pair[1] for pair in topic_words_probs]
				corpus.extend(w)
				topic_words.append(w)
				topic_probs.append(p)
			return topic_words, topic_probs, list(set(corpus))

		def add_special_tokens_and_pad(topic_ids, topic_probs, max_topic_seq_len):
			new_topic_ids, new_topic_probs, topic_msk = [], [], []
			max_topic_seq_len = max_topic_seq_len + 2 ## bos & eos
			for i, tree_topic_ids in enumerate(topic_ids): ## For each tree
				new_ids, new_probs, msks = [], [], []
				for j, ids in enumerate(tree_topic_ids): ## For each topic
					## Add bos & eos token
					ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
					prb = [0] + topic_probs[i][j] + [0]
					#prb = [sum(topic_probs[i][j]) / len(ids)] + topic_probs[i][j] + [sum(topic_probs[i][j])/len(ids)]
					#ipdb.set_trace()
					msk = [1] * len(ids)

					## Padding
					num_padding = max_topic_seq_len - len(ids)
					ids.extend([tokenizer.pad_token_id] * num_padding)
					prb.extend([0] * num_padding)
					msk.extend([0] * num_padding)
					
					new_ids.append(ids)
					new_probs.append(prb)
					msks.append(msk)

				new_topic_ids.append(new_ids)
				new_topic_probs.append(new_probs)
				topic_msk.append(msks)
			return new_topic_ids, new_topic_probs, topic_msk

		## ------------------------------------------------------------------------------------------------------

		## Take all responses into consideration
		src_ids, trees = examples["source_id"], examples["tweet_ids"]
		src_ids = [str(src_id) for src_id in src_ids] ## Convert src_id to strings
		trees = parse_trees_from_str(trees)
		tree_lens = [len(tree) for tree in trees]
		tweet_ids, texts = id2text(trees)

		model_inputs = tokenizer(texts, padding=padding, max_length=data_args.max_tweet_length, truncation=True)

		start_idx = 0
		input_ids = []
		attn_mask = []
		for tree_i in trees:
			input_ids_i, attn_mask_i = formulate_one_tree(model_inputs, start_idx, start_idx + len(tree_i))
			input_ids.append(input_ids_i)
			attn_mask.append(attn_mask_i)
			start_idx = start_idx + len(tree_i)

		## Final assignment
		model_inputs["source_id"] = src_ids
		model_inputs["tweet_ids"] = [",".join(tweet_ids) for tweet_ids in trees] ## list of list (tweet ids of a tree)
		model_inputs["tree_lens"] = tree_lens
		model_inputs["input_ids"] = input_ids
		model_inputs["attention_mask"] = attn_mask

		return model_inputs

	with training_args.main_process_first(desc="dataset map pre-processing"):
		for key_data in raw_datasets.keys(): ## Separately process each sets of raw_datasets
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
	train_dataset = raw_datasets["train"]
	eval_dataset = None
	test_dataset = None

	return train_dataset, eval_dataset, test_dataset