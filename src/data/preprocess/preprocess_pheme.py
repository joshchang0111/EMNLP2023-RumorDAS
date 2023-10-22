import os
import csv
import ipdb
import json
import random
import argparse
import numpy as np
import pandas as pd
import preprocessor as pre

from tqdm import tqdm
from datetime import datetime
from nltk.tokenize import TweetTokenizer

random.seed(10)

def parse_args():
	parser = argparse.ArgumentParser(description="Rumor Detection")

	parser.add_argument("--preprocess"  , action="store_true")
	parser.add_argument("--split_5_fold", action="store_true")
	parser.add_argument("--split_event_wise", action="store_true")

	parser.add_argument("--dataset", type=str, default=None)
	parser.add_argument("--raw_data_root", type=str, default="../dataset/raw")
	parser.add_argument("--V2_data_root" , type=str, default="../dataset/processedV2")
	parser.add_argument("--fold", type=str, default="0,1,2,3,4", help="either use 5-fold data or train/dev/test from rumoureval2019 competition")
	parser.add_argument("--n_fold", type=int, default=5)

	args = parser.parse_args()

	return args

def get_file_or_dirs(path, filter=None):
	file_or_dirs = [file_or_dir for file_or_dir in os.listdir(path) if not file_or_dir.startswith(".")]
	return file_or_dirs

def convert_annotations(annotation, string=True):
	"""
	Convert PHEME rumour annotations into True, False, Unverified.
	This function is provided in `PHEME_veracity` dataset.
	"""
	if "misinformation" in annotation.keys() and "true" in annotation.keys():
		if int(annotation['misinformation']) == 0 and int(annotation["true"]) == 0:
			if string:
				label = "unverified"
			else:
				label = 2
		elif int(annotation["misinformation"]) == 0 and int(annotation["true"]) == 1 :
			if string:
				label = "true"
			else:
				label = 1
		elif int(annotation["misinformation"]) == 1 and int(annotation["true"]) == 0 :
			if string:
				label = "false"
			else:
				label = 0
		elif int(annotation["misinformation"]) == 1 and int(annotation["true"]) == 1:
			#print ("OMG! They both are 1!")
			#print(annotation["misinformation"])
			#print(annotation["true"])
			label = None
	elif "misinformation" in annotation.keys() and "true" not in annotation.keys():
		## all instances have misinfo label but don't have true label
		if int(annotation["misinformation"]) == 0:
			if string:
				label = "unverified"
			else:
				label = 2
		elif int(annotation["misinformation"]) == 1:
			if string:
				label = "false"
			else:
				label = 0
	elif "true" in annotation.keys() and "misinformation" not in annotation.keys():
		#print ("Has true not misinformation")
		label = None
	else:
		#print("No annotations")
		label = None
	return label

def clean_text(line):

	## Remove @, reduce length, handle strip
	tokenizer = TweetTokenizer(strip_handles=False, reduce_len=True)
	line = " ".join(tokenizer.tokenize(line))

	## Remove url, emoji, mention, prserved words, only preserve smiley
	#pre.set_options(pre.OPT.URL, pre.OPT.EMOJI, pre.OPT.MENTION, pre.OPT.RESERVED)
	pre.set_options(pre.OPT.URL, pre.OPT.RESERVED)
	line = pre.tokenize(line)

	## Remove non-sacii 
	line = "".join([i if ord(i) else "" for i in line]) ## remove non-sacii
	return line

def get_paths_of_each_event(args):
	"""Obtain the path of all json files in the format of dictionary."""
	raw_dataset_path = "{}/{}/all-rnr-annotated-threads".format(args.raw_data_root, args.dataset)
	
	print("{:20s}\t{:4s}\t{:4s}\t{:4s}\t{:4s}\t{:4s}\t{:4s}".format("Event", "#src", "#tr", "#fr", "#ur", "#nr", "#rsp"))
	print("-" * 20)

	path_dict, events2remove = {}, []
	total_nt, total_nf, total_nu, total_nn, total_src = 0, 0, 0, 0, 0
	labels = ["non-rumours", "rumours"]
	events = get_file_or_dirs(raw_dataset_path)
	for event in events: ## For each event
		event_key = "-".join(event.split("-")[:-3])
		path_dict[event_key] = {}
		balance_num = []
		nt, nf, nu, nn = 0, 0, 0, 0
		for label in labels: ## For each label
			src_post_dir = "{}/{}/{}".format(raw_dataset_path, event, label)
			src_post_ids = get_file_or_dirs(src_post_dir)

			path_dict[event_key][label] = []
			for src_post_id in src_post_ids: ## For each source post (tree)
				path_dict[event_key][label].append("{}/{}".format(src_post_dir, src_post_id))

				annotation = json.load(open("{}/{}/annotation.json".format(src_post_dir, src_post_id)))
				veracity = convert_annotations(annotation)
				if veracity == "true":
					nt = nt + 1
				elif veracity == "false":
					nf = nf + 1
				elif veracity == "unverified":
					nu = nu + 1
				elif veracity is None:
					nn = nn + 1

			balance_num.append(len(src_post_ids))
			if len(src_post_ids) < 200:
				events2remove.append(event_key)
		
		total_nt = total_nt + nt
		total_nf = total_nf + nf
		total_nu = total_nu + nu
		total_nn = total_nn + nn
		print("{:20s}\t{:4d}\t{:4d}\t{:4d}\t{:4d}\t{:4d}".format(event_key, len(src_post_ids), nt, nf, nu, nn))

		"""
		## Balance labels for each event
		balance_num = balance_num[1]
		for label in labels:
			random.shuffle(path_dict[event_key][label])
			path_dict[event_key][label] = path_dict[event_key][label][:balance_num]
		"""

	total_src = total_nt + total_nf + total_nu + total_nn
	print("-" * 20)
	print("{:20s}\t{:4d}\t{:4d}\t{:4d}\t{:4d}\t{:4d}".format("Total", total_src, total_nt, total_nf, total_nu, total_nn))
	print("-" * 20)

	"""
	## Filter event
	events2remove = list(set(events2remove))
	for event in events2remove:
		del path_dict[event]
	print("\nEvents [{}] are removed...".format(", ".join(events2remove)))
	"""

	return path_dict

def get_structure(structure, parent="None", prefix="", struct_dict=None):
	if isinstance(structure, list):
		return struct_dict
	if struct_dict is None:
		struct_dict = {}
	for key, sub_struct in structure.items():
		print("{}{}".format(prefix, key))
		struct_dict[key] = parent
		struct_dict = get_structure(sub_struct, key, prefix="{}\t".format(prefix), struct_dict=struct_dict)
	return struct_dict

def process_pheme(args):
	"""
	Preprocess PHEME dataset with `veracity` label.
	Output file: data.csv
		- format: source_id,tweet_id,parent_idx,self_idx,num_parent,max_seq_len,text,veracity
	"""
	path_dict = get_paths_of_each_event(args)

	"""
	print("\n***** After balanced & filtered *****")
	print("{:20s}[{:15s}]\t{:4s}\t{:5s}\t{:6s}\t{:11s}\t{:5s}\t{:5s}".format("Event", "Label", "#src", "#true", "#false", "#unverified", "#none", "#resp"))
	print("-" * len("{:20s}[{:15s}]".format("Event", "Label")))
	"""
	print("Take only labels: [true, false, unverified]")
	total_src_post = 0
	total_response = 0
	no_veracity = 0
	
	tree_dict = {}
	for event in path_dict.keys(): ## For each event
		#for label in path_dict[event].keys(): ## For each label
		label = "rumours"
		n_reactions, nt, nf, nu, nn = 0, 0, 0, 0, 0
		for path in path_dict[event][label]: ## For each source post
			src_id = path.split("/")[-1]
			src_path = "{}/source-tweets/{}.json".format(path, src_id)
			src_data = json.load(open(src_path))
	
			## Obtain veracity labels
			annotation = json.load(open("{}/annotation.json".format(path)))
			veracity = convert_annotations(annotation)
			if veracity == "true":
				nt = nt + 1
			elif veracity == "false":
				nf = nf + 1
			elif veracity == "unverified":
				nu = nu + 1
			elif veracity is None:
				nn = nn + 1
				veracity = label
	
			tree_dict[src_id] = {"source": {}, "response": []}
			tree_dict[src_id]["source"] = {
				"source_id": src_id, 
				"tweet_id" : src_id, 
				"parent_idx": "None", 
				"self_idx": 1, 
				"num_parent": "-", 
				"max_seq_len": "-", 
				"text": clean_text(src_data["text"]), 
				"veracity": veracity, 
				"event": event
			}
			
			## Get responses
			reaction_dir = "{}/reactions".format(path)
			reaction_ids = get_file_or_dirs(reaction_dir)
			
			for reaction in reaction_ids: ## For each response
				if reaction.replace(".json", "") == src_id: ## some reactions contain source post
					continue
				response_path = "{}/{}".format(reaction_dir, reaction)
				response_data = json.load(open(response_path))
				tree_dict[src_id]["response"].append(
					{
						"source_id": src_id, 
						"tweet_id" : reaction.replace(".json", ""), 
						"parent_id": response_data["in_reply_to_status_id_str"],# if response_data["in_reply_to_status_id_str"] is not None else src_id, 
						"parent_idx": "-", 
						"num_parent": "-", 
						"max_seq_len": "-",
						"created_at": datetime.strptime(response_data["created_at"], "%a %b %d %H:%M:%S %z %Y"), 
						"text": clean_text(response_data["text"]), 
						"veracity": veracity, 
						"event": event
					}
				)
				n_reactions = n_reactions + 1
				#struct_dict = get_structure(structure)
				#if response_data["in_reply_to_status_id_str"] is None:
				#	structure = json.load(open("{}/structure.json".format(path)))
				#	ipdb.set_trace()
			
			## Sort by `created_at`
			response_self_idx_map = {}
			tree_dict[src_id]["response"].sort(key=lambda x: x["created_at"])
			for response_idx, response in enumerate(tree_dict[src_id]["response"]):
				response["self_idx"] = response_idx + 2
				response_self_idx_map[response["tweet_id"]] = response["self_idx"]

			## Get `parent_idx`
			for response in tree_dict[src_id]["response"]:
				parent_id = response["parent_id"]
				if parent_id == src_id:
					response["parent_idx"] = 1
				elif parent_id not in response_self_idx_map:
					response["parent_idx"] = 1
				else:
					response["parent_idx"] = response_self_idx_map[parent_id]
		
		print("{:20s}\t{:4d}\t{:4d}\t{:4d}\t{:4d}\t{:4d}\t{:4d}".format(event, len(path_dict[event][label]), nt, nf, nu, nn, n_reactions))
		total_src_post = total_src_post + len(path_dict[event][label])
		total_response = total_response + n_reactions
	
	print()
	print("Total src post: {}".format(total_src_post))
	print("Total response: {}".format(total_response))

	ipdb.set_trace()
	## Write
	print("\nWrite `data.csv`...")
	with open("{}/PHEME/data.csv".format(args.V2_data_root), "w") as fcsv:
		writer = csv.writer(fcsv)
		writer.writerow(["source_id", "tweet_id", "parent_idx", "self_idx", "num_parent", "max_seq_len", "text", "veracity", "event"])
		for src_id in tree_dict.keys():
			src_dict = tree_dict[src_id]["source"]
			writer.writerow([src_dict["source_id"], src_dict["tweet_id"], "None", src_dict["self_idx"], "-", "-", src_dict["text"], src_dict["veracity"], src_dict["event"]])
			for response_dict in tree_dict[src_id]["response"]:
				writer.writerow([response_dict["source_id"], response_dict["tweet_id"], response_dict["parent_idx"], response_dict["self_idx"], "-", "-", response_dict["text"], response_dict["veracity"], response_dict["event"]])

def split_5_fold(args):
	"""
	Split 5 fold of V2 dataset from `data.csv`.
	Format:
		train.csv: ["source_id", "tweet_ids", "label_veracity"]
		test.csv : ["source_id", "tweet_ids", "label_veracity"]
	"""
	data_path = "{}/{}/data.csv".format(args.V2_data_root, args.dataset)
	data_df = pd.read_csv(data_path)
	groupby_source = data_df.groupby("source_id")

	rows = []
	for source_id, group in groupby_source:
		tweet_ids = ",".join(group["tweet_id"].astype(str).tolist())
		label = group["veracity"].tolist()[0]
		rows.append([source_id, tweet_ids, label])
	
	output_df = pd.DataFrame(columns=["source_id", "tweet_ids", "label_veracity"], data=rows)
	output_df = output_df.sample(frac=1) ## Shuffle
	groupby_labels = output_df.groupby("label_veracity")

	## Split 5 portions for each label
	folds = []
	for fold_idx in range(args.n_fold):
		portion = []
		for label, group in groupby_labels:
			n_portion = int(len(group) / args.n_fold)
			start_idx = fold_idx * n_portion
			end_idx   = len(group) if fold_idx == args.n_fold - 1 else (fold_idx + 1) * n_portion

			portion.append(group.iloc[start_idx:end_idx])
			print("label: {:15s}, fold_idx: {:4d}, start_idx: {:4d}, end_idx: {:4d}".format(label, fold_idx, start_idx, end_idx))

		folds.append(pd.concat(portion))
	
	## Write `train.csv` and `test.csv` for each split
	for fold_idx in range(args.n_fold):
		## Take turns be test set
		test = folds[fold_idx]

		## Get train set
		train = []
		for i in range(args.n_fold):
			if i != fold_idx:
				train.append(folds[i])
		train = pd.concat(train)

		print("Fold [{}], # train: {:5d}, # test: {:5d}".format(fold_idx, len(train), len(test)))

		## Write file
		split_path = "{}/{}/split_{}".format(args.V2_data_root, args.dataset, fold_idx)
		os.makedirs(split_path, exist_ok=True)
		train.to_csv("{}/train.csv".format(split_path), index=False)
		test.to_csv("{}/test.csv".format(split_path), index=False)

def split_event_wise(args):
	"""
	Split event-wise folds of PHEME V2 dataset from `data.csv`.
	Format:
		train.csv: ["source_id", "tweet_ids", "label_veracity"]
		test.csv : ["source_id", "tweet_ids", "label_veracity"]
	"""
	def write_set_to_file(train_or_test, set_df):
		rows = []
		groupby_source = set_df.groupby("source_id")
		for source_id, group in groupby_source:
			tweet_ids = ",".join(group["tweet_id"].astype(str).tolist())
			label = group["veracity"].tolist()[0]
			rows.append([source_id, tweet_ids, label])

		output_df = pd.DataFrame(columns=["source_id", "tweet_ids", "label_veracity"], data=rows)
		output_df.to_csv("{}/{}/event_{}/{}.csv".format(args.V2_data_root, args.dataset, event_idx, train_or_test), index=False)

	print("Split event wise")

	data_path = "{}/{}/data.csv".format(args.V2_data_root, args.dataset)
	data_df = pd.read_csv(data_path)
	groupby_event = data_df.groupby("event")

	## Each event takes turn to be the testing set
	for event_idx, (event, test_group) in enumerate(groupby_event):
		print("Test set: {}".format(event))
		train_group = data_df.loc[data_df["event"] != event]

		os.makedirs("{}/{}/event_{}".format(args.V2_data_root, args.dataset, event_idx), exist_ok=True)
		write_set_to_file("train", train_group)
		write_set_to_file("test", test_group)
		with open("{}/{}/event_{}/event.txt".format(args.V2_data_root, args.dataset, event_idx), "w") as fw:
			fw.write("{}\n".format(event))

def main(args):
	print("Preprocess {} dataset...\n".format(args.dataset))
	if args.dataset is None:
		raise ValueError("Raw dataset name (--dataset) not specified!")
	
	if args.preprocess:
		if args.dataset == "PHEME_veracity":
			process_pheme(args)
	elif args.split_5_fold:
		split_5_fold(args)
	elif args.split_event_wise:
		split_event_wise(args)

if __name__ == "__main__":
	args = parse_args()
	main(args)