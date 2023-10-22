import os
import ipdb
import emoji
import random
import logging
import warnings
import pandas as pd
from datasets import load_dataset

## Self-defined
from .graph_dataset import GraphDataset

## Call the same logger used in main_summ.py
logger = logging.getLogger("__main__")

def build_datasets_adv(data_args, model_args, training_args, config, tokenizer, model):
	"""Build datasets for adversarial training"""
	##################
	## Load Dataset ##
	##################
	print("\nLoading dataset...")
	print("[{}]: fold [{}]".format(data_args.dataset_name, data_args.fold))

	## Loading a dataset from my local files.
	if data_args.dataset_name == "PHEME":
		## Conduct event-wise cross-validation for PHEME
		data_files = {
			"train"     : "{}/{}/event_{}/{}".format(data_args.dataset_root, data_args.dataset_name, data_args.fold, data_args.train_file), 
			"validation": "{}/{}/event_{}/{}".format(data_args.dataset_root, data_args.dataset_name, data_args.fold, data_args.validation_file), 
			"test"      : "{}/{}/event_{}/{}".format(data_args.dataset_root, data_args.dataset_name, data_args.fold, data_args.validation_file)
		}
	else:
		data_files = {
			"train"     : "{}/{}/split_{}/{}".format(data_args.dataset_root, data_args.dataset_name, data_args.fold, data_args.train_file), 
			"validation": "{}/{}/split_{}/{}".format(data_args.dataset_root, data_args.dataset_name, data_args.fold, data_args.validation_file), 
			"test"      : "{}/{}/split_{}/{}".format(data_args.dataset_root, data_args.dataset_name, data_args.fold, data_args.validation_file)
		}
	raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)

	## Read tweet contents
	data_df = pd.read_csv("{}/{}/data.csv".format(data_args.dataset_root, data_args.dataset_name))
	data_df["source_id"] = data_df["source_id"].astype(str) ## For PHEME, twitter15, twitter16
	data_df["tweet_id"]  = data_df["tweet_id"].astype(str)
	#data_df["text"] = data_df["text"].apply(lambda x: emoji.demojize(x))
	dataset_content = data_df.set_index("tweet_id").T.to_dict() ## Each tweet_id maps to all information

	## Build graph_dataset
	graph_dataset = GraphDataset(data_args, model_args)

	## Labels for detector
	label_list = raw_datasets["train"].unique("label_veracity")
	label_list.sort()  ## sort for determinism, ["false", "non-rumor", "true", "unverified"]
	num_labels = len(label_list)
	data_args.label_list = label_list

	assert num_labels == data_args.num_labels, "num_labels specified in data arguments doesn't match your actual dataset!"

	################
	## Preprocess ##
	################
	print("\nProcessing dataset...")

	## Padding strategy
	max_target_length = data_args.max_target_length ## For generator
	padding = "max_length" if data_args.pad_to_max_length else False

	if model.__class__.__name__ == "RobertaForRumorDetection":
		data_args.max_tree_length = int(tokenizer.model_max_length / data_args.max_tweet_length) ## 16, need not consider generated node
	elif model.__class__.__name__ == "BartForRumorDetectionAndResponseGeneration":
		#if not (training_args.task_type == "train_adv_stage2" and \
		#		training_args.do_eval and model.summarizer is not None):
		#	data_args.max_tree_length = int((tokenizer.model_max_length / data_args.max_tweet_length) / 2)
		#else:
		#	data_args.max_tree_length = int( tokenizer.model_max_length / data_args.max_tweet_length) - 1 ## 31 (32 - 1) for bart-base when evaluating

		if (training_args.task_type == "train_adv_stage2" and training_args.do_eval and not training_args.do_train and model.summarizer is not None) or \
		   (training_args.task_type == "train_adv_stage1" and training_args.do_eval and not training_args.do_train and training_args.framing):
			data_args.max_tree_length = int( tokenizer.model_max_length / data_args.max_tweet_length) - 1 ## 31 (32 - 1) for bart-base when evaluating
		else:
			data_args.max_tree_length = int((tokenizer.model_max_length / data_args.max_tweet_length) / 2)
			#data_args.max_tree_length = int(tokenizer.model_max_length / data_args.max_tweet_length) - 1

	## Some models have set the order of the labels to use, so let's make sure we do use it.
	label_to_id = {v: i for i, v in enumerate(label_list)}
	if label_to_id is not None:
		config.label2id = label_to_id
		config.id2label = {id: label for label, id in config.label2id.items()}

	def preprocess_function(examples):
		"""
		Input:
			- examples: keys = ["source_id", "tweet_ids", "label_veracity"]
		Different preprocessing for different set:
			- train     : w/  tree decomposition -> need to predict each response
			- validation: w/  tree decomposition -> evaluate detector's performance on augmented tree
			- test      : w/o tree decomposition -> test on original tree (for detector's performance, no evaluation on attacker)
		"""
		def parse_trees_from_str(input_trees):
			"""Parse each tree from string of tweet_ids to list"""
			output_trees = [tweet_ids_str.split(",") for tweet_ids_str in input_trees]
			assert len(output_trees) == len(input_trees)
			return output_trees

		def tree_decomposition(input_src_ids, input_trees, input_labels):
			"""
			Truncate and augment by decomposing each tree.
			Sorting by time-order, adding each node forms a subtree.
			"""
			td_edges, td_edges_gen = [], []
			bu_edges, bu_edges_gen = [], []
			output_src_ids, output_trees, output_labels, output_targets, tree_lens = [], [], [], [], []
			for tree_idx in range(len(input_src_ids)): ## For each tree
				## Truncate, adding one more node for trees longer than max_tree_length
				## for generated target
				input_tree = input_trees[tree_idx][:data_args.max_tree_length + 1]
				td_edge_index, bu_edge_index = graph_dataset[input_src_ids[tree_idx]]
				for n in range(1, len(input_tree)): ## decomposition loop: this loop will ignore tree without responses
					output_src_ids.append(input_src_ids[tree_idx])
					output_trees.append(input_tree[:n])
					output_labels.append(input_labels[tree_idx])
					output_targets.append(input_tree[n])
					tree_lens.append(n)
					
					## Edges
					td_edges.append(graph_dataset.pad(graph_dataset.drop_edge(td_edge_index[:, td_edge_index[1] < n], "td")))
					bu_edges.append(graph_dataset.pad(graph_dataset.drop_edge(bu_edge_index[:, bu_edge_index[0] < n], "bu")))
					td_edges_gen.append(td_edge_index[:, td_edge_index[1] == n])
					bu_edges_gen.append(bu_edge_index[:, bu_edge_index[0] == n])
				
			assert len(output_src_ids) == len(output_trees) == len(output_labels) == \
				   len(output_targets) == len(td_edges) == len(td_edges_gen) == len(tree_lens)
			
			return output_src_ids, output_trees, output_labels, output_targets, tree_lens, \
				   td_edges, td_edges_gen, \
				   bu_edges, bu_edges_gen

		def truncate_trees(input_src_ids, input_trees):
			"""Truncate only"""
			output_trees = [input_tree[:data_args.max_tree_length] for input_tree in input_trees]
			tree_lens = [len(tree) for tree in output_trees]
			td_edges = []
			bu_edges = []
			for src_id in input_src_ids:
				td_edge_index, bu_edge_index = graph_dataset[src_id]
				td_edges.append(graph_dataset.pad(td_edge_index[:, td_edge_index[1] < data_args.max_tree_length]))
				bu_edges.append(graph_dataset.pad(bu_edge_index[:, bu_edge_index[0] < data_args.max_tree_length]))
			assert len(input_trees) == len(output_trees) == len(td_edges) == len(bu_edges)
			return output_trees, tree_lens, td_edges, None, bu_edges, None

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

		def formulate_one_tree(tree_idx):
			"""Process input_ids and attention_mask for each tree"""
			tree_input_ids, tree_attn_mask = [], []
			tree_token_type_ids = [] ## For BERT only
			for k, tweet_id in enumerate(trees[tree_idx]): ## For each node in tree
				idx_of_result = tweet_ids.index(tweet_id)
				node_input_ids = model_inputs["input_ids"][idx_of_result]
				node_attn_mask = model_inputs["attention_mask"][idx_of_result]

				## if not source tweet -> modify first token from <s> to </s>
				if tweet_id != src_id:
					node_input_ids[0] = tokenizer.sep_token_id
				
				tree_input_ids.extend(node_input_ids)
				tree_attn_mask.extend(node_attn_mask)

				if "token_type_ids" in model_inputs:
					node_token_type_ids = model_inputs["token_type_ids"][idx_of_result]
					if k % 2 == 1:
						node_token_type_ids = [1] * len(node_token_type_ids)
					tree_token_type_ids.extend(node_token_type_ids)

			## Padding to make each tree the same length!
			max_length = data_args.max_tree_length * data_args.max_tweet_length
			num_pad_token = max_length - len(tree_input_ids)
			tree_input_ids.extend([tokenizer.pad_token_id] * num_pad_token)
			tree_attn_mask.extend([0] * num_pad_token)
			tree_token_type_ids.extend([0] * num_pad_token)

			return tree_input_ids, tree_attn_mask, tree_token_type_ids

		def prepare_target(tree_idx):
			"""Only for train & validation set"""
			idx_of_target = tweet_ids.index(targets[tree_idx])
			target_ids = labels_gen["input_ids"][idx_of_target]
			return target_ids

		def prepare_adv_labels_det(labels):
			if training_args.attack_type == "shift":
				print("Building {} dataset with shifted labels...".format(key_data))
				adv_labels = [(label + 1) % data_args.num_labels for label in labels]
			elif training_args.attack_type == "reverse":
				print("Building {} dataset with reverse labels...".format(key_data))
				adv_labels = []
				for label in labels:
					if label == config.label2id["true"]:
						adv_labels.append(config.label2id["false"])
					elif label == config.label2id["false"]:
						adv_labels.append(config.label2id["true"])
					else:
						adv_labels.append(label)
			elif training_args.attack_type == "untargeted":
				print("untargeted attack: will train the model with `negative loss`")
				adv_labels = labels
			return adv_labels

		## ======================================================================================================
		src_ids, trees, label_veracity = examples["source_id"], examples["tweet_ids"], examples["label_veracity"]
		src_ids = [str(src_id) for src_id in src_ids]
		trees = parse_trees_from_str(trees)
		tweet_ids, texts = id2text(trees)

		if key_data == "train" or key_data == "validation":
			## For train & validation sets: 
			##   - decompose a tree in to several subtrees
			##   - prepare labels for generation
			src_ids, trees, label_veracity, targets, tree_lens, \
			td_edges, td_edges_gen, \
			bu_edges, bu_edges_gen = tree_decomposition(src_ids, trees, label_veracity)

			## Tokenize targets for generator's prediction
			with tokenizer.as_target_tokenizer():
				labels_gen = tokenizer(texts, padding=padding, max_length=data_args.max_tweet_length, truncation=True)
			
			## If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
			## padding in the loss.
			if padding == "max_length" and data_args.ignore_pad_token_for_loss:
				labels_gen["input_ids"] = [
					[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels_gen["input_ids"]
				]
		else:
			## For test set: 
			##   - only need to truncate tree length
			trees, tree_lens, td_edges, td_edges_gen, bu_edges, bu_edges_gen = truncate_trees(src_ids, trees)
		
		## Tokenize the texts
		model_inputs = tokenizer(texts, padding=padding, max_length=data_args.max_tweet_length, truncation=True)
		
		## Formulate inputs from model_inputs, keys = ["input_ids", "attention_mask"] for BART
		## Example: <s>src_tweet</s></s>response_0</s></s>response_1</s>
		input_ids_new, attn_mask_new, token_type_ids_new, output_target_ids = [], [], [], []
		for tree_idx, src_id in enumerate(src_ids): ## For each tree (including subtrees)
			tree_input_ids, tree_attn_mask, tree_token_type_ids = formulate_one_tree(tree_idx)

			input_ids_new.append(tree_input_ids)
			attn_mask_new.append(tree_attn_mask)
			token_type_ids_new.append(tree_token_type_ids)

			if key_data == "train" or key_data == "validation":
				target_ids = prepare_target(tree_idx)
				output_target_ids.append(target_ids)

		## Map labels to IDs
		labels = [label_to_id[l] for l in label_veracity]
		if training_args.task_type == "train_adv_stage2" and key_data != "test":
			labels = prepare_adv_labels_det(labels)

		## Final assignment
		model_inputs["tree_lens"] = tree_lens
		model_inputs["source_id"] = src_ids
		model_inputs["input_ids"] = input_ids_new
		model_inputs["attention_mask"] = attn_mask_new
		model_inputs["labels_det"] = labels ## Detection target
		if key_data == "train" or key_data == "validation":
			model_inputs["labels_gen"] = output_target_ids ## Generation target
		if "token_type_ids" in model_inputs: ## Only for BERT
			model_inputs["token_type_ids"] = token_type_ids_new
		
		if model_args.td_gcn:
			model_inputs["td_edges"] = td_edges
			if td_edges_gen is not None:
				model_inputs["td_edges_gen"] = td_edges_gen
		if model_args.bu_gcn:
			model_inputs["bu_edges"] = bu_edges
			if bu_edges_gen is not None:
				model_inputs["bu_edges_gen"] = bu_edges_gen

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
		#eval_dataset = raw_datasets["validation"]
		eval_dataset = raw_datasets["test"]

		## Shuffle train dataset
		train_dataset = train_dataset.shuffle()

	## Make test dataset
	if training_args.do_eval:
		if "test" not in raw_datasets:
			raise ValueError("--do_eval requires a test dataset")
		test_dataset = raw_datasets["test"]

		if training_args.framing:
			test_dataset = raw_datasets["validation"]

	if data_args.gacl:
		gacl_eids = os.listdir("{}/data/{}".format(data_args.gacl_path, data_args.dataset_name))
		gacl_eids = [eid for eid in gacl_eids if eid[0] != "."]

		train_dataset = train_dataset.filter(lambda example: str(example["source_id"]) in gacl_eids)
		eval_dataset = eval_dataset.filter(lambda example: str(example["source_id"]) in gacl_eids)
		test_dataset = test_dataset.filter(lambda example: str(example["source_id"]) in gacl_eids)

	return train_dataset, eval_dataset, test_dataset