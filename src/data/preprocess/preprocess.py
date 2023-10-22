import os
import csv
import ipdb
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def parse_args():
	parser = argparse.ArgumentParser(description="Rumor Detection")

	## What to do
	parser.add_argument("--txt2csv", action="store_true")
	parser.add_argument("--csv4hf", action="store_true")
	parser.add_argument("--processV2", action="store_true")
	parser.add_argument("--processV2_stance", action="store_true")
	parser.add_argument("--filter_stance_summary", action="store_true")
	parser.add_argument("--pheme_event_wise", action="store_true")
	parser.add_argument("--processV2_fold", action="store_true")
	parser.add_argument("--process_semeval2019_dev_set", action="store_true")
	parser.add_argument("--create_tree_ids_file", action="store_true")

	## Options for writing csv file
	parser.add_argument("--simple", action="store_true", help="No graph structures written.")

	## Others
	parser.add_argument("--dataset", type=str, default="semeval2019", choices=["semeval2019", "Pheme", "twitter15", "twitter16"])
	parser.add_argument("--data_root", type=str, default="../dataset/processed")
	parser.add_argument("--data_root_V2", type=str, default="../dataset/processedV2")
	parser.add_argument("--fold", type=str, default="0,1,2,3,4", help="either use 5-fold data or train/dev/test from rumoureval2019 competition")
	parser.add_argument("--result_path", type=str, default="../result")

	args = parser.parse_args()

	return args

def txt2csv(args):
	csv_file = open("{}/{}/data.csv".format(args.data_root, args.dataset), "w")
	writer = csv.writer(csv_file)
	writer.writerow(["source id", "tweet id", "parent idx", "self idx", "num parent node", "max seq len", "text"])

	for line in tqdm(open("{}/{}/data.text.txt".format(args.data_root, args.dataset)).readlines()):
		line = line.strip().rstrip()
		line = line.replace("<end>", "")
		writer.writerow(line.split("\t"))

def csv4hf(args):
	"""
	Convert original dataset into the format for huggingface example.
	Requirements: 
		data.text.txt need to be combined with train.veracity.txt / test.veracity.txt
		output files should be .csv
	"""
	folds = args.fold.split(",")
	for fold in folds:
		print("===Fold {}===".format(fold))
		## Load label first
		print("Loading label file...")
		label_dict = {"train": {}, "test": {}}
		for data_type, _ in label_dict.items(): ## train & valid
			for line in tqdm(open("{}/{}/split_{}/{}.veracity.txt".format(args.data_root, args.dataset, fold, data_type)).readlines()):
				line = line.strip().rstrip()
				label, src_id = line.split("\t")[0], line.split("\t")[1]
				label_dict[data_type][src_id] = label
		
		## Load tree data
		print("\nLoading tree file...")
		tweet_dict = {}
		for line in tqdm(open("{}/{}/data.text.txt".format(args.data_root, args.dataset)).readlines()):
			line = line.strip().rstrip()
			src_id = line.split("\t")[0]

			if src_id not in tweet_dict:
				tweet_dict[src_id] = []
			tweet_dict[src_id].append(line)

		## Write csv file
		print("\nWriting csv file...")
		for data_type, _ in label_dict.items():
			if args.simple:
				## Write only source_id, text content of whole tree, and veracity label
				with open("{}/{}/split_{}/{}.simple.csv".format(args.data_root, args.dataset, fold, data_type), "w") as csv_file:
					writer = csv.writer(csv_file)
					writer.writerow(["source_id", "text", "label_veracity"])

					## Write every tree, one row for one tree
					for src_id, tweet_lines in tqdm(list(tweet_dict.items())):
						if src_id not in label_dict[data_type]:
							continue
						all_text = []
						for tweet_line in tweet_lines:
							all_text.append(tweet_line.split("\t")[-1])
						text = "[SEP]".join(all_text)
						writer.writerow([src_id, text, label_dict[data_type][src_id]])
			else:
				with open("{}/{}/split_{}/{}.csv".format(args.data_root, args.dataset, fold, data_type), "w") as csv_file:
					writer = csv.writer(csv_file)
					writer.writerow(["source_id", "tweet_id", "parent_idx", "self_idx", "num_parent", "max_seq_len", "text", "label_veracity"])

					## Write every tree
					for src_id, tweet_lines in tqdm(list(tweet_dict.items())):
						if src_id not in label_dict[data_type]:
							continue
						## Write every response node
						for tweet_line in tweet_lines:
							features = tweet_line.split("\t")
							features.append(label_dict[data_type][src_id])
							writer.writerow(features)

def processV2_stance(args):
	"""
	Preprocess another version of semeval2019 dataset.
	Target format:
		- data.text.csv: contains all information (src id, tweet id, ..., text)
		- split_i/train.csv: 
		- split_i/test.csv
	"""
	##############
	## data.csv ##
	##############
	path_in  = "{}/{}".format(args.data_root   , args.dataset)
	path_out = "{}/{}".format(args.data_root_V2, args.dataset)
	os.makedirs(path_out, exist_ok=True)

	## Read content
	print("\nReading text content from {}/{}".format(args.data_root, args.dataset))
	rows = []
	with open("{}/data.text.txt".format(path_in), "r") as infile:
		for line in tqdm(list(infile.readlines())):
			line = line.strip().rstrip()
			line = line.replace("<end>", "")
			cols = line.split("\t")
			rows.append(cols)

	## Read labels (verity & stance)
	print("\nReading labels from {}/{}".format(args.data_root, args.dataset))
	verity_dict, stance_dict = {}, {}
	with open("{}/data.veracity.txt".format(path_in), "r") as infile:
		for line in tqdm(list(infile.readlines())):
			line = line.strip().rstrip()
			verity, src_id = line.split("\t")
			verity_dict[src_id] = verity

	with open("{}/data.stance.txt".format(path_in), "r") as infile:
		for line in tqdm(list(infile.readlines())):
			line = line.strip().rstrip()
			stance, twt_id = line.split("\t")
			stance_dict[twt_id] = stance

	## Write csv
	print("\nWriting data.csv to {}/{}".format(args.data_root_V2, args.dataset))
	headers = ["source_id", "tweet_id", "parent_idx", "self_idx", "num_parent", "max_seq_len", "text", "veracity", "stance"]
	with open("{}/data.csv".format(path_out), "w") as outfile:
		writer = csv.writer(outfile)
		writer.writerow(headers)
		for cols in tqdm(rows):
			src_id, twt_id = cols[0], cols[1]
			cols.extend([verity_dict[src_id], stance_dict[twt_id]])
			writer.writerow(cols)
	'''
	## Way to convert dataframe into dictionary with "tweet_id" as key and other columns as values
	data_df = pd.read_csv("{}/data.csv".format(path_out))
	data_df.head().set_index("tweet_id").T.to_dict()
	ipdb.set_trace()
	'''
	print("\nWriting summ.stance.idx.csv to {}/{}".format(args.data_root_V2, args.dataset))
	with open("{}/summ.stance.idx.csv".format(path_out), "w") as outfile:
		writer = csv.writer(outfile)
		writer.writerow(["source_id", "support_ids", "deny_ids", "query_ids", "comment_ids"])

		data_df = pd.read_csv("{}/data.csv".format(path_out))
		src_id_groups = data_df.groupby("source_id")
		for src_id, reply_df in src_id_groups:
			stance_ids = {"support": "", "deny": "", "query": "", "comment": ""}
			stance_groups = reply_df.groupby("stance")
			for stance, group in stance_groups:
				ids_list = group["tweet_id"].tolist()
				if src_id in ids_list:
					ids_list.remove(src_id)
				ids_str = ",".join(ids_list)
				stance_ids[stance] = ids_str

			writer.writerow([src_id, stance_ids["support"], stance_ids["deny"], stance_ids["query"], stance_ids["comment"]])

	print("\nWriting summ.idx.csv to {}/{}".format(args.data_root_V2, args.dataset))
	with open("{}/summ.idx.csv".format(path_out), "w") as outfile:
		writer = csv.writer(outfile)
		writer.writerow(["source_id", "all_ids"])

		data_df = pd.read_csv("{}/data.csv".format(path_out))
		src_id_groups = data_df.groupby("source_id")
		for src_id, reply_df in src_id_groups:

			ids_list = reply_df["tweet_id"].tolist()
			if str(src_id) in ids_list:
				ids_list.remove(str(src_id))
			ids_str = ",".join(ids_list)

			writer.writerow([src_id, ids_str])

	## Four summary version
	print("\nWriting summ.4.idx.csv to {}/{}".format(args.data_root_V2, args.dataset))
	with open("{}/summ.4.idx.csv".format(path_out), "w") as outfile:
		writer = csv.writer(outfile)
		writer.writerow(["source_id", "all_0_ids", "all_1_ids", "all_2_ids", "all_3_ids"])

		data_df = pd.read_csv("{}/data.csv".format(path_out))
		src_id_groups = data_df.groupby("source_id")
		for src_id, reply_df in src_id_groups:

			ids_list = reply_df["tweet_id"].tolist()
			if str(src_id) in ids_list:
				ids_list.remove(str(src_id))
			
			## Split replies into 4 groups
			ids_4_groups = list(np.array_split(ids_list, 4))

			ids_strs = []
			for ids_group_i in ids_4_groups:
				ids_strs.append(",".join(list(ids_group_i)))
			
			write_cols = [src_id]
			write_cols.extend(ids_strs)
			writer.writerow(write_cols)

def processV2(args):
	"""
	Preprocess another version of datasets (Pheme, twitter15, twitter16).
	Target format:
		- data.text.csv: contains all information (src id, tweet id, ..., text)
		- split_i/train.csv: 
		- split_i/test.csv
	"""
	##############
	## data.csv ##
	##############
	path_in  = "{}/{}".format(args.data_root   , args.dataset)
	path_out = "{}/{}".format(args.data_root_V2, args.dataset)
	os.makedirs(path_out, exist_ok=True)

	## Read content
	print("\nReading text content from {}/{}".format(args.data_root, args.dataset))
	rows = []
	with open("{}/data.text.txt".format(path_in), "r") as infile:
		for line in tqdm(list(infile.readlines())):
			line = line.strip().rstrip()
			line = line.replace("<end>", "")
			cols = line.split("\t")
			rows.append(cols)

	## Read labels (verity & stance)
	print("\nReading labels from {}/{}".format(args.data_root, args.dataset))
	verity_dict = {}
	with open("{}/data.label.txt".format(path_in), "r") as infile:
		for line in tqdm(list(infile.readlines())):
			line = line.strip().rstrip()
			verity, src_id = line.split("\t")[0], line.split("\t")[-1]
			verity_dict[src_id] = verity

	## Write csv
	print("\nWriting data.csv to {}/{}".format(args.data_root_V2, args.dataset))
	headers = ["source_id", "tweet_id", "parent_idx", "self_idx", "num_parent", "max_seq_len", "text", "veracity"]
	with open("{}/data.csv".format(path_out), "w") as outfile:
		writer = csv.writer(outfile)
		writer.writerow(headers)
		for cols in tqdm(rows):
			src_id = cols[0]
			twt_id = "{}_{}".format(src_id, int(cols[2]) - 1) if cols[1] != "None" else src_id
			cols_new = [src_id, twt_id, cols[1], cols[2], cols[3], cols[4], cols[6], verity_dict[src_id]]
			writer.writerow(cols_new)
	"""
	## Way to convert dataframe into dictionary with "tweet_id" as key and other columns as values
	data_df = pd.read_csv("{}/data.csv".format(path_out))
	data_df.head().set_index("tweet_id").T.to_dict()
	ipdb.set_trace()
	"""

	## One summary version
	print("\nWriting summ.idx.csv to {}/{}".format(args.data_root_V2, args.dataset))
	with open("{}/summ.idx.csv".format(path_out), "w") as outfile:
		writer = csv.writer(outfile)
		writer.writerow(["source_id", "all_ids"])

		data_df = pd.read_csv("{}/data.csv".format(path_out))
		src_id_groups = data_df.groupby("source_id")
		for src_id, reply_df in src_id_groups:

			ids_list = reply_df["tweet_id"].tolist()
			if str(src_id) in ids_list:
				ids_list.remove(str(src_id))
			ids_str = ",".join(ids_list)

			writer.writerow([src_id, ids_str])

	## Four summary version
	print("\nWriting summ.4.idx.csv to {}/{}".format(args.data_root_V2, args.dataset))
	with open("{}/summ.4.idx.csv".format(path_out), "w") as outfile:
		writer = csv.writer(outfile)
		writer.writerow(["source_id", "all_0_ids", "all_1_ids", "all_2_ids", "all_3_ids"])

		data_df = pd.read_csv("{}/data.csv".format(path_out))
		src_id_groups = data_df.groupby("source_id")
		for src_id, reply_df in src_id_groups:

			ids_list = reply_df["tweet_id"].tolist()
			if str(src_id) in ids_list:
				ids_list.remove(str(src_id))
			
			## Split replies into 4 groups
			ids_4_groups = list(np.array_split(ids_list, 4))

			ids_strs = []
			for ids_group_i in ids_4_groups:
				ids_strs.append(",".join(list(ids_group_i)))
			
			write_cols = [src_id]
			write_cols.extend(ids_strs)
			writer.writerow(write_cols)

	###################
	## Copy 5 splits ##
	###################
	print("\nWriting 5 fold to {}/{}".format(args.data_root_V2, args.dataset))
	folds = [0, 1, 2, 3, 4]
	for fold in folds:
		path_in_split  = "{}/split_{}".format(path_in , fold)
		path_out_split = "{}/split_{}".format(path_out, fold)
		os.makedirs(path_out_split, exist_ok=True)

		file_types = ["train", "test"]
		for file_type in file_types:
			rows = []
			## Read from split
			with open("{}/{}.label.txt".format(path_in_split, file_type), "r") as infile:
				for line in tqdm(list(infile.readlines()), desc="[Fold {}-{:5s}]".format(fold, file_type)):
					line = line.strip().rstrip()
					cols = line.split("\t")
					cols[0], cols[-1] = cols[-1], cols[0]
					rows.append(cols)

			## Write to data_root_V2	
			headers = ["source_id", "event", "label_veracity"] if args.dataset == "Pheme" else ["source_id", "label_veracity"]
			with open("{}/{}.csv".format(path_out_split, file_type), "w") as outfile:
				writer = csv.writer(outfile)
				writer.writerow(headers)
				for row in rows:
					writer.writerow(row)

def filter_stance_summary(args):
	"""Replace empty stance with empty summary"""
	stances = ["support", "deny", "query", "comment"]

	models = [
		#"t5-small", 
		#"linydub/bart-large-samsum", 
		#"philschmid/bart-base-samsum", 
		#"knkarthick/MEETING-SUMMARY-BART-LARGE-XSUM-SAMSUM-DIALOGSUM", 
		#"google/pegasus-reddit_tifu"
		"facebook/bart-large-xsum"
	]
	for model in models:
		model = model.replace("/", " ")
		content_file = "{}/stance-summarization/{}/generated_predictions.csv".format(args.result_path, model)
		summ_id_file = "../dataset/processedV2/{}/summ.stance.idx.csv".format(args.dataset)

		content_df = pd.read_csv(content_file)
		summ_id_df = pd.read_csv(summ_id_file)

		for src_id in summ_id_df["source_id"].tolist():
			id_row = summ_id_df.loc[summ_id_df["source_id"] == src_id]
			for stance in stances:
				if id_row["{}_ids".format(stance)].isna().item():
					index = content_df.index[content_df["source_id"] == src_id][0]
					content_df["{} summary".format(stance)][index] = ""

		output_dir = "{}/stance-summarization/{}".format(args.result_path, model)
		content_df.to_csv("{}/filtered.csv".format(output_dir), index=False)

		## Write to dataset path
		output_dir = "../dataset/processedV2/{}/stance_summary/{}".format(args.dataset, model)
		os.makedirs(output_dir, exist_ok=True)
		content_df.to_csv("{}/summ.csv".format(output_dir), index=False)

def pheme_event_wise(args):
	"""Create event-wise split for Pheme"""
	path_in  = "{}/Pheme".format(args.data_root)
	path_out = "{}/Pheme_ew".format(args.data_root_V2)

	event_groups = {}
	with open("{}/data.label.txt".format(path_in)) as f_label:
		lines = f_label.readlines()
		for line in lines:
			line = line.strip().rstrip()
			label, event, src_id = line.split("\t")
			
			if event not in event_groups:
				event_groups[event] = []
			event_groups[event].append([src_id, event, label])

	## Display statistics
	for event, group in event_groups.items():
		print("Event: {:20s}, # sourc tweets: {}".format(event, len(group)))

	## Write splits
	for test_event in event_groups.keys(): ## Take turns be test set
		print("split_{}".format(test_event))
		train, test = [], []
		for event, group in event_groups.items():
			if event == test_event:
				test.extend(group)
			else:
				train.extend(group)

		print("    # train: {}".format(len(train)))
		print("    # test : {}".format(len(test)))

		path_out_event = "{}/split_{}".format(path_out, test_event)
		os.makedirs(path_out_event, exist_ok=True)

		with open("{}/train.csv".format(path_out_event), "w") as outfile:
			writer = csv.writer(outfile)
			writer.writerow(["source_id", "event", "label_veracity"])
			for row in train:
				writer.writerow(row)

		with open("{}/test.csv".format(path_out_event), "w") as outfile:
			writer = csv.writer(outfile)
			writer.writerow(["source_id", "event", "label_veracity"])
			for row in test:
				writer.writerow(row)

def processV2_fold(args):
	'''
	Process each fold of V2 dataset.
	Format:
		train_adv.csv: ["source_id", "tweet_ids", "label_veracity"]
		test_adv.csv : ["source_id", "tweet_ids", "label_veracity"]
	'''
	def write_csv(train_or_test, fold):
		data_path = "{}/{}/data.csv".format(args.data_root_V2, args.dataset)
		data_df = pd.read_csv(data_path)

		fold_path = "{}/{}/split_{}".format(args.data_root_V2, args.dataset, fold)
		fold_df = pd.read_csv("{}/{}.csv".format(fold_path, train_or_test))
		src_ids = fold_df["source_id"].tolist()

		rows = [["source_id", "tweet_ids", "label_veracity"]]
		for src_id in src_ids:
			tweets = data_df.loc[data_df["source_id"] == src_id]
			tweet_ids = tweets["tweet_id"].tolist()
			label = list(set(tweets["veracity"].tolist()))[0]
			
			row = [src_id, ",".join(tweet_ids), label]
			rows.append(row)

		outfile = "{}/{}_adv.csv".format(fold_path, train_or_test)
		writer = csv.writer(open(outfile, "w"))
		for row in rows:
			writer.writerow(row)

	for fold in args.fold.split(","):
		print("Writing fold [{}]".format(fold))
		write_csv(train_or_test="train", fold=fold)
		write_csv(train_or_test="test" , fold=fold)

def process_semeval2019_dev_set(args):
	def get_rows(src_ids):
		rows = [["source_id", "tweet_ids", "label_veracity"]]
		for src_id in src_ids:
			tweets = data_df.loc[data_df["source_id"] == src_id]
			tweet_ids = tweets["tweet_id"].tolist()
			label = list(set(tweets["veracity"].tolist()))[0]
			
			row = [src_id, ",".join(tweet_ids), label]
			rows.append(row)
		return rows

	data_path = "{}/{}/data.csv".format(args.data_root_V2, args.dataset)
	data_df = pd.read_csv(data_path)

	dev_path_in = "{}/{}/split_comp/dev.veracity.txt".format(args.data_root, args.dataset)
	src_ids = []
	with open(dev_path_in, "r") as f:
		for line in f.readlines():
			line = line.strip().rstrip()
			src_id = line.split("\t")[1]
			src_ids.append(src_id)

	rows = get_rows(src_ids)

	## Write to file
	outfile = "{}/{}/split_comp/dev.csv".format(args.data_root_V2, args.dataset)
	writer = csv.writer(open(outfile, "w"))
	for row in rows:
		writer.writerow(row)

def create_tree_ids_file(args):
	"""`data.csv` -> `data_tree_ids.csv`, for building cluster summary"""
	data_df = pd.read_csv("{}/{}/data.csv".format(args.data_root_V2, args.dataset))

	with open("{}/{}/data_tree_ids.csv".format(args.data_root_V2, args.dataset), "w") as fw:
		writer = csv.writer(fw)
		writer.writerow(["source_id", "tweet_ids"])

		for src_id, reply_df in data_df.groupby("source_id"):
			tweet_ids_str = ",".join(reply_df["tweet_id"].tolist())
			writer.writerow([src_id, tweet_ids_str])

if __name__ == "__main__":
	args = parse_args()
	if args.txt2csv:
		txt2csv(args)
	elif args.csv4hf:
		csv4hf(args)
	elif args.processV2_stance:
		processV2_stance(args)
	elif args.processV2:
		processV2(args)
	elif args.filter_stance_summary:
		filter_stance_summary(args)
	elif args.pheme_event_wise:
		pheme_event_wise(args)
	elif args.processV2_fold:
		processV2_fold(args)
	elif args.process_semeval2019_dev_set:
		process_semeval2019_dev_set(args)
	elif args.create_tree_ids_file:
		create_tree_ids_file(args)
