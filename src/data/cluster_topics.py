import os
import ipdb
import json
import emoji
import openai
import shutil
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from wordcloud import WordCloud

import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import RobertaTokenizer, RobertaModel

def parse_args():
	parser = argparse.ArgumentParser(description="Split data via clustering.")

	## Which experiment
	parser.add_argument("--split_data_via_cluster", action="store_true")
	parser.add_argument("--select_number_of_clusters", action="store_true")
	parser.add_argument("--collect_semeval2019_event", action="store_true")

	## Parameters
	parser.add_argument("--n_clusters", type=int, default=3)

	## Others
	parser.add_argument("--dataset_name", type=str, default="twitter15", choices=["semeval2019", "twitter15", "twitter16"])
	parser.add_argument("--dataset_root", type=str, default="../dataset/processedV2")
	parser.add_argument("--fold", type=str, default="0,1,2,3,4", help="either use 5-fold data or train/dev/test from rumoureval2019 competition")
	parser.add_argument("--result_path", type=str, default="/mnt/1T/projects/RumorV2/results")

	args = parser.parse_args()

	return args

def load_data(args):
	def preprocess_txt(txt):
		return emoji.demojize(txt).replace("URL", "").replace("url", "")

	print("\nLoad data...")
	print("Dataset: [{}]".format(args.dataset_name))

	data_df = pd.read_csv("{}/{}/data.csv".format(args.dataset_root, args.dataset_name))
	group_src = data_df.groupby("source_id")
	
	src_ids, src_txts = [], []
	for src_id, group in group_src:
		src_txt = group.iloc[0]["text"]
		src_txt = preprocess_txt(src_txt)
		src_txts.append(src_txt)
		src_ids.append(src_id)
	
	return data_df, src_ids, src_txts

def load_model(device):
	print("\nLoad RoBERTa-Large & tokenizer...")
	tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
	model = RobertaModel.from_pretrained("roberta-large")
	model.to(device)
	return tokenizer, model

def get_roberta_txt_feat(tokenizer, model, text, device):
	"""
	Input format: "<s> {} <\s>".format(text)
	"""
	encoded_input = tokenizer(text, return_tensors="pt")
	encoded_input = encoded_input.to(device)
	outputs = model(**encoded_input)
	return outputs

def split_data_via_cluster(args):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Device: {}".format(device))

	data_df, src_ids, src_txts = load_data(args)
	tokenizer, model = load_model(device)

	print("\nCollect text features of each source post...")
	features = []
	with torch.no_grad():
		for src_txt in tqdm(src_txts):
			outputs = get_roberta_txt_feat(tokenizer, model, src_txt, device)
			feature = outputs["last_hidden_state"][0][0] ## First token of first data in a batch
			feature = feature.cpu().numpy()
			features.append(feature)

	print("\nPerform k-means clustering on features of source posts...")
	if args.select_number_of_clusters:
		for n_clusters in range(2, 11):
			silhouette_scores, cluster_results = [], []
			for _ in tqdm(range(100), desc="{} Clusters".format(n_clusters)):
				kmeans_fit = KMeans(n_clusters=n_clusters, n_init="auto").fit(features)
				silhouette = silhouette_score(features, kmeans_fit.labels_)
				silhouette_scores.append(silhouette)
				cluster_results.append(kmeans_fit)

			print("[{}] Clusters".format(n_clusters))
			silhouette_scores = np.array(silhouette_scores)
			print("Max Silhouette Score: {}".format(np.max(silhouette_scores)))
			print("Avg Silhouette Score: {}".format(np.mean(silhouette_scores)))
	else:
		silhouette_scores, cluster_results = [], []
		n_iters = 1
		for _ in tqdm(range(1), desc="{} Clusters".format(args.n_clusters)):
			kmeans_fit = KMeans(n_clusters=args.n_clusters, n_init="auto").fit(features)
			silhouette = silhouette_score(features, kmeans_fit.labels_)
			silhouette_scores.append(silhouette)
			cluster_results.append(kmeans_fit)

		print("[{}] Clusters".format(args.n_clusters))
		silhouette_scores = np.array(silhouette_scores)
		print("Max Silhouette Score: {}".format(np.max(silhouette_scores)))
		print("Avg Silhouette Score: {}".format(np.mean(silhouette_scores)))

		max_idx = np.argmax(silhouette_scores)
		labels_ = cluster_results[max_idx].labels_

		src_ids, src_txts = np.array(src_ids), np.array(src_txts)

		fw = open("{}/{}/cluster_topics/cluster_ids.txt".format(args.dataset_root, args.dataset_name), "w")

		clusters = {}
		for cid in range(args.n_clusters):
			print("# cluster [{}]: {}".format(cid, (labels_ == cid).sum()))

			clusters[cid] = {}
			clusters[cid]["src_ids"]  = src_ids[labels_ == cid]
			clusters[cid]["src_txts"] = src_txts[labels_ == cid]

			full_txt = " ".join(list(clusters[cid]["src_txts"]))
			wordcloud = WordCloud(width=1000, height=500).generate(full_txt)
			wordcloud.to_file("{}/{}/cluster_topics/cloud_{}.png".format(args.dataset_root, args.dataset_name, cid))
			
			for src_id in clusters[cid]["src_ids"]:
				fw.write("{}: {}\n".format(src_id, cid))

		fw.close()
		ipdb.set_trace()

def collect_semeval2019_event(args):
	data_df = pd.read_csv("{}/semeval2019/data.csv".format(args.dataset_root))
	group_src = data_df.groupby("source_id")

	for src_id, thread_df in group_src:
		src_txt = thread_df.iloc[0]["text"]

def main(args):
	if args.split_data_via_cluster:
		split_data_via_cluster(args)
	elif args.collect_semeval2019_event:
		collect_semeval2019_event(args)

if __name__ == "__main__":
	args = parse_args()
	main(args)
	