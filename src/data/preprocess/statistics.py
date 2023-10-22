import os
import csv
import ipdb
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

def parse_args():
	parser = argparse.ArgumentParser(description="Rumor Detection")

	## What to do
	#parser.add_argument("--txt2csv", action="store_true")

	## Others
	parser.add_argument("--dataset", type=str, default="semeval2019", choices=["semeval2019", "Pheme", "twitter15", "twitter16"])
	parser.add_argument("--data_root", type=str, default="../dataset/processed")
	parser.add_argument("--data_root_V2", type=str, default="../dataset/processedV2")
	parser.add_argument("--fold", type=str, default="0,1,2,3,4", help="either use 5-fold data or train/dev/test from rumoureval2019 competition")

	args = parser.parse_args()

	return args

def statistics(args):
	print("\n*** Statistics of {} ***\n".format(args.dataset))

	path_i = "{}/{}/data.csv".format(args.data_root_V2, args.dataset)

	data_df = pd.read_csv(path_i)

	label_set = set(data_df["veracity"])
	src_group = data_df.groupby("source_id")

	n_posts  = len(data_df)
	n_claims = len(src_group.size())
	max_tree_len = np.max(src_group.size())
	min_tree_len = np.min(src_group.size())
	avg_tree_len = np.mean(src_group.size())

	## Count labels
	label_cnt = {}
	for label in label_set:
		label_cnt[label] = 0

	## For each source post (claim)
	for src_id, group in src_group:
		label_cnt[group["veracity"].tolist()[0]] += 1

	print("# claims: {:4d}".format(n_claims))
	print("# posts : {:4d}".format(n_posts))
	for label in label_cnt.keys():
		print("# {:10s}: {:4d}".format(label, label_cnt[label]))

	print("Max. tree len.: {:6d}".format(max_tree_len))
	print("Min. tree len.: {:6d}".format(min_tree_len))
	print("Avg. tree len.: {:6.2f}".format(avg_tree_len))

	## Plot histogram of tree length
	tree_lens = src_group.size().values
	plt.hist(tree_lens, range(min(tree_lens), max(tree_lens) + 10, 10))
	plt.title(args.dataset.capitalize())
	plt.savefig("{}/{}/tree_len.png".format(args.data_root_V2, args.dataset))

	#ipdb.set_trace()

if __name__ == "__main__":
	args = parse_args()

	statistics(args)
