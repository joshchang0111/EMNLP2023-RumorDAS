import os
import ipdb
import json
import math
import random
import shutil
import argparse
import itertools
import pandas as pd
from tqdm import tqdm

import transformers
from datasets import load_metric
from evaluate import load

def parse_args():
	parser = argparse.ArgumentParser(description="Rumor Detection")

	## What to do
	parser.add_argument("--eval_ppl", action="store_true")
	parser.add_argument("--generate_for_factCC", action="store_true")

	parser.add_argument("--model_type", type=str, default="kmeans") ## kmeans, loo
	parser.add_argument("--num_clusters", type=int, default=1) ## 1, 2, 3, 4, 5
	parser.add_argument("--factCC_format", type=str, default=None, help="response_wise, all_responses")

	## Others
	parser.add_argument("--data_name", type=str, default="semeval2019", choices=["semeval2019", "Pheme", "twitter15", "twitter16"])
	parser.add_argument("--data_root", type=str, default="../../dataset/processed")
	parser.add_argument("--data_root_V2", type=str, default="../../dataset/processedV2")
	parser.add_argument("--fold", type=str, default="0,1,2,3,4", help="either use 5-fold data or train/dev/test from rumoureval2019 competition")
	parser.add_argument("--result_path", type=str, default="/mnt/1T/projects/RumorV2/results")

	args = parser.parse_args()

	return args

def eval_ppl(args):
	print("Evaluating perplexity of generated summary...")
	print("Model Type: {}".format(args.model_type))

	metric = load("perplexity", module_type="metric")
		
	ppls = []
	for fold in args.fold.split(","):
		print("{} Fold [{}]".format(args.data_name, fold))

		summary_df = pd.read_csv("{}/{}/{}/{}/summary.csv".format(args.result_path, args.data_name, args.model_type, fold))
		#ipdb.set_trace()
		if "kmeans" in args.model_type:
			summaries = []
			for source_id, group in summary_df.groupby("source_id"): ## For each thread
				summary_clusters = []
				for cluster_id, cluster in group.groupby("cluster_id"): ## Create all possible summary combinations
					summary_clusters.append(cluster["summary"].tolist())
				
				summary_thread = list(itertools.product(*summary_clusters))
				summary_thread = [" ".join(summ) for summ in summary_thread]
				summaries.extend(summary_thread)
		else:
			summaries = summary_df["summary"].tolist()
		
		new_summaries = []
		for summary in summaries:
			if not isinstance(summary, str):
				new_summaries.append(".")
			else:
				new_summary = summary.replace("$", "").strip().rstrip()
				new_summary = "." if new_summary == "" else new_summary
				new_summaries.append(summary)
		summaries = new_summaries

		ppl = metric.compute(
			model_id="gpt2", 
			predictions=summaries, 
			add_start_token=True,
			device="cuda"
		)
		ppls.append(ppl)
	
	with open("{}/{}/{}/ppl.txt".format(args.result_path, args.data_name, args.model_type), "w") as fw:
		fw.write("{}\t{}\n".format("Fold", "Perplexity"))
		for fold_idx, ppl in enumerate(ppls):
			fw.write("{:4d}\t{}\n".format(fold_idx, ppl["mean_perplexity"]))

def generate_for_factCC(args):
	print("Generating data files for factCC...")
	print("factCC_format: {}".format(args.factCC_format))
	print("Model Type: {}".format(args.model_type))

	if args.factCC_format == "cluster_wise":
		for fold in args.fold.split(","):
			print("{} Fold [{}]".format(args.data_name, fold))
			dataset_path = "{}/{}/data.csv".format(args.data_root_V2, args.data_name)
			summary_root = "{}/{}/{}/{}".format(args.result_path, args.data_name, args.model_type, fold)
			summary_path = "{}/summary.csv".format(summary_root)
			cluster_path = "{}/{}/split_{}/cluster_summary/train/kmeans-{}.csv".format(args.data_root_V2, args.data_name, fold, args.model_type.split("_")[-1])			

			dataset_df = pd.read_csv(dataset_path)
			summary_df = pd.read_csv(summary_path)
			cluster_df = pd.read_csv(cluster_path)

			new_df = cluster_df.copy()
			new_df["cluster_id"] = cluster_df["cluster_id"].apply(lambda x: x.split("_")[0])
			cluster_df = new_df
			
			if "kmeans" not in args.model_type:
				raise ValueError("Wrong model type specified.")

			source_id, summaries = [], []
			for src_id, group in summary_df.groupby("source_id"): ## For each thread
				summary_clusters = {}
				for cluster_id, cluster in group.groupby("cluster_id"): ## Create all possible summary combinations
					summary_clusters[cluster_id] = cluster["summary"].tolist()[0] ## Only one summary for each cluster
				
				#summary_thread.append(summary_clusters)
				summaries.append(summary_clusters)
				source_id.append(src_id)

			os.makedirs("{}/factCC_{}".format(summary_root, args.factCC_format), exist_ok=True)
			with open("{}/factCC_{}/data-dev.jsonl".format(summary_root, args.factCC_format), "w") as fw:
				for idx in range(len(summaries)):
					summary, src_id = summaries[idx], source_id[idx]
					tree_df = dataset_df[dataset_df["source_id"] == src_id]
					clus_df = cluster_df[cluster_df["source_id"] == src_id]
					resp_df = tree_df[tree_df["tweet_id"] != src_id]

					## For each cluster
					for cluster_id, cluster in clus_df.groupby("cluster_id"):
						summary_cluster_i = summary[int(cluster_id)]
						resp_df_cluster_i = resp_df[resp_df["tweet_id"].isin(cluster["tweet_id"].tolist())] ## Take the responses belong to this cluster as inputs
						
						summary_cluster_i = summary_cluster_i.replace("$", "").strip().rstrip()
						summary_cluster_i = "." if summary_cluster_i == "" else summary_cluster_i

						obj = {}
						obj["label"] = "CORRECT" ## Dummy Label
						obj["id"] = "{}_{}".format(src_id, cluster_id)
						obj["text"] = " ".join(resp_df_cluster_i["text"].tolist())
						obj["claim"] = summary_cluster_i

						fw.write("{}\n".format(json.dumps(obj)))
						#ipdb.set_trace()
					
			#new_summaries = []
			#for summary in summaries:
			#	if not isinstance(summary, str):
			#		new_summaries.append(".")
			#	else:
			#		new_summary = summary.replace("$", "").strip().rstrip()
			#		new_summary = "." if new_summary == "" else new_summary
			#		new_summaries.append(summary)
			#summaries = new_summaries

	else:
		for fold in args.fold.split(","):
			print("{} Fold [{}]".format(args.data_name, fold))
			dataset_path = "{}/{}/data.csv".format(args.data_root_V2, args.data_name)
			summary_root = "{}/{}/{}/{}".format(args.result_path, args.data_name, args.model_type, fold)
			summary_path = "{}/summary.csv".format(summary_root)

			dataset_df = pd.read_csv(dataset_path)
			summary_df = pd.read_csv(summary_path)

			if "kmeans" in args.model_type:
				source_id, summaries = [], []
				for src_id, group in summary_df.groupby("source_id"): ## For each thread
					summary_clusters = []
					for cluster_id, cluster in group.groupby("cluster_id"): ## Create all possible summary combinations
						summary_clusters.append(cluster["summary"].tolist())

					summary_thread = list(itertools.product(*summary_clusters))
					summary_thread = [" ".join(summ) for summ in summary_thread]
					summaries.extend(summary_thread)
					source_id.append(src_id)
			else:
				source_id = summary_df["source_id"].tolist()
				summaries = summary_df["summary"].tolist()

			#print("{} Fold [{}], {}".format(args.data_name, fold, len(summaries)))
			new_summaries = []
			for summary in summaries:
				if not isinstance(summary, str):
					new_summaries.append(".")
				else:
					new_summary = summary.replace("$", "").strip().rstrip()
					new_summary = "." if new_summary == "" else new_summary
					new_summaries.append(summary)
			summaries = new_summaries

			#shutil.rmtree("{}/{}".format(summary_root, args.factCC_format))
			os.makedirs("{}/factCC_{}".format(summary_root, args.factCC_format), exist_ok=True)
			with open("{}/factCC_{}/data-dev.jsonl".format(summary_root, args.factCC_format), "w") as fw:
				for idx in range(len(summaries)):
					summary, src_id = summaries[idx], source_id[idx]
					tree_df = dataset_df[dataset_df["source_id"] == src_id]
					resp_df = tree_df[tree_df["tweet_id"] != src_id]

					if args.factCC_format == "all_responses":
						obj = {}
						obj["label"] = "CORRECT" ## Dummy Label
						obj["id"] = src_id
						obj["text"] = " ".join(resp_df["text"].tolist())
						obj["claim"] = summary

						fw.write("{}\n".format(json.dumps(obj)))

					elif args.factCC_format == "response_wise":
						resp_text = resp_df["text"].tolist()
						for r_txt in resp_text:
							obj = {}
							obj["label"] = "CORRECT" ## Dummy Label
							obj["id"] = src_id
							obj["text"] = r_txt
							obj["claim"] = summary

							fw.write("{}\n".format(json.dumps(obj)))

if __name__ == "__main__":
	args = parse_args()

	if args.eval_ppl:
		eval_ppl(args)
	elif args.generate_for_factCC:
		generate_for_factCC(args)