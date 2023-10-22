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
	parser.add_argument("--recover_4_classes", action="store_true")
	parser.add_argument("--process_twitter16", action="store_true")

	## Others
	parser.add_argument("--dataset", type=str, default="twitter15", choices=["twitter15", "twitter16"])
	parser.add_argument("--data_root_raw", type=str, default="../dataset/raw")
	parser.add_argument("--data_root_V1", type=str, default="../dataset/processed")
	parser.add_argument("--data_root_V2", type=str, default="../dataset/processedV2")
	parser.add_argument("--fold", type=str, default="0,1,2,3,4", help="either use 5-fold data or train/dev/test from rumoureval2019 competition")

	args = parser.parse_args()

	return args

def recover_4_classes(args):
	"""Recover rumor labels (true, false, unverified) of V2 dataset from raw datasets."""
	raw_labels_dict = {}
	raw_labels_path = "{}/rumor_detection_acl2017/{}/label.txt".format(args.data_root_raw, args.dataset)
	with open(raw_labels_path, "r") as f:
		for line in f.readlines():
			line = line.strip().rstrip()
			label, src_id = line.split(":")
			raw_labels_dict[int(src_id)] = label

	## For `data.csv`
	data_df = pd.read_csv("{}/{}/data.csv".format(args.data_root_V2, args.dataset))
	data_df["veracity"] = data_df.apply(lambda row: raw_labels_dict[row["source_id"]], axis=1)
	data_df.to_csv("{}/{}/data.csv".format(args.data_root_V2, args.dataset))

	## For each fold
	for fold in args.fold.split(","):
		sets = ["train", "test"]
		fold_path = "{}/{}/split_{}".format(args.data_root_V2, args.dataset, fold)
		for train_or_test in sets:
			print("Fold [{}] - {:5s} set".format(fold, train_or_test))
			fold_df = pd.read_csv("{}/{}.csv".format(fold_path, train_or_test))
			fold_df["label_veracity"] = fold_df.apply(lambda row: raw_labels_dict[row["source_id"]], axis=1)
			fold_df.to_csv("{}/{}.csv".format(fold_path, train_or_test), index=False)

def process_twitter16(args):
	def remove_weird_token(text):
		text_new = []
		last_uD = False
		for token in text.split():
			if token.isdigit() and last_uD and len(token) == 2:
				continue
			if token.startswith("uD"):
				last_uD = True
				continue
			last_uD = False
			text_new.append(token)
		return " ".join(text_new).replace("\\", "").strip().rstrip()
	
	data_df = pd.read_csv("{}/{}/data_ori.csv".format(args.data_root_V2, args.dataset))
	
	#text = data_df.iloc[1472]["text"]
	#text_new = []
	#last_uD = False
	#for token in text.split():
	#	if token.isdigit() and last_uD and len(token) == 2:
	#		continue
	#	if token.startswith("uD"):
	#		last_uD = True
	#		continue
	#	last_uD = False
	#	text_new.append(token)
	#text_new = " ".join(text_new).replace("\\", "").strip().rstrip()
	#text_new = remove_weird_token(text)
	tqdm.pandas()
	data_df["text"] = data_df["text"].progress_apply(remove_weird_token)
	ipdb.set_trace()
	data_df.to_csv("{}/{}/data.csv".format(args.data_root_V2, args.dataset), index=False)

if __name__ == "__main__":
	args = parse_args()
	
	print("Preprocessing [{}]".format(args.dataset))
	if args.recover_4_classes:
		recover_4_classes(args)
	elif args.process_twitter16:
		process_twitter16(args)