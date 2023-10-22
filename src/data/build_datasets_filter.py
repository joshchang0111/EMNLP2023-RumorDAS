import ipdb
import random
import logging
import pandas as pd

from torch.utils.data import DataLoader

from datasets import load_dataset

## Call the same logger used in main_summ.py
logger = logging.getLogger("__main__")

def build_datasets_filter(data_args, model_args, training_args, config, tokenizer, model):
	"""Build datasets for training autoencoder filter."""
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
	dataset_content = data_df.set_index("tweet_id").T.to_dict() ## Each tweet_id maps to all information

	## Labels for detector
	label_list = raw_datasets["train"].unique("label_veracity")
	label_list.sort() ## sort for determinism
	num_labels = len(label_list)

	assert num_labels == data_args.num_labels, "num_labels specified in data arguments doesn't match your actual dataset!"

	################
	## Preprocess ##
	################
	print("\nProcessing dataset...")

	## Padding strategy
	max_target_length = data_args.max_target_length ## For generator
	padding = "max_length" if data_args.pad_to_max_length else False
	
	## Some models have set the order of the labels to use, so let's make sure we do use it.
	label_to_id = {v: i for i, v in enumerate(label_list)}

	if label_to_id is not None:
		config.label2id = label_to_id
		config.id2label = {id: label for label, id in config.label2id.items()}
	
	assert data_args.max_seq_length * data_args.max_tree_length <= tokenizer.model_max_length, \
	"Max length of tree sequence ({}) larger than the max input length for the model ({})!".format(
		data_args.max_seq_length * data_args.max_tree_length, tokenizer.model_max_length
	)
	max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

	def preprocess_function(examples):
		"""
		Input:
			- examples: keys = ["source_id", "tweet_ids", "label_veracity"]
		Different preprocessing for different set:
			- train     : w/  augmentation -> need to predict each response
			- validation: w/  augmentation -> evaluate detector's performance on augmented tree
			- test      : w/o augmentation -> test on original tree (for detector's performance, no evaluation on attacker)
		"""
		def filter_data_with_target_class(src_ids, trees):
			"""Train the AE to reconstruct a specific class"""
			stance = ["support", "comment", "query", "deny"]
			verity = ["true", "false", "unverified"]

			key = ""
			if model_args.target_class_ext_ae.lower() in stance:
				key = "stance"
			elif model_args.target_class_ext_ae.lower() in verity:
				key = "veracity"
			else: ## `all`
				return src_ids, trees

			print("The model will be trained to reconstruct responses with label `{}`".format(model_args.target_class_ext_ae))

			filter_src_ids, filter_trees = [], []
			for idx in range(len(src_ids)):
				if dataset_content[src_ids[idx]][key] == model_args.target_class_ext_ae.lower():
					filter_src_ids.append(src_ids[idx])
					filter_trees.append(trees[idx])

			return filter_src_ids, filter_trees

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

		## ------------------------------------------------------------------------------------------------------
		src_ids, trees = examples["source_id"], examples["tweet_ids"]
		src_ids = [str(src_id) for src_id in src_ids] ## Convert to strings
		src_ids, trees = filter_data_with_target_class(src_ids, trees)
		trees = parse_trees_from_str(trees)
		tweet_ids, texts = id2text(trees)

		model_inputs = tokenizer(texts, padding=padding, max_length=data_args.max_tweet_length, truncation=True)

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
	## Make train & evaluation dataset
	if training_args.do_train:
		if "train" not in raw_datasets:
			raise ValueError("--do_train requires a train dataset")
		train_dataset = raw_datasets["train"]
		if data_args.max_train_samples is not None:
			train_dataset = train_dataset.select(range(data_args.max_train_samples))

		## Shuffle train dataset
		train_dataset = train_dataset.shuffle()

		if "validation" not in raw_datasets:
			raise ValueError("--do_train requires a validation dataset")
		eval_dataset = raw_datasets["validation"]
		if data_args.max_eval_samples is not None:
			eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

		## Log a few random samples from the training set:
		for index in random.sample(range(len(train_dataset)), 3):
			logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
	else:
		train_dataset = None
		eval_dataset  = None

	## Make test dataset
	if training_args.do_eval:
		if "test" not in raw_datasets:
			raise ValueError("--do_eval requires a test dataset")
		test_dataset = raw_datasets["test"]
		if data_args.max_eval_samples is not None:
			test_dataset = test_dataset.select(range(data_args.max_eval_samples))
	else:
		test_dataset = None

	return train_dataset, eval_dataset, test_dataset