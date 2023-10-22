import os
import ipdb
import json
import nltk
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from wordcloud import WordCloud
from scipy.stats import bootstrap
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

## self-defined
from metrics import f1_score_3_class

def parse_args():
	parser = argparse.ArgumentParser(description="Rumor Detection")

	## What to do
	parser.add_argument("--wordcloud", action="store_true")
	parser.add_argument("--get_semeval2019_event", action="store_true")
	parser.add_argument("--get_semeval2019_event_from_pheme", action="store_true")
	parser.add_argument("--semeval2019_event_wise_eval", action="store_true")
	parser.add_argument("--bootstrap_accuracy", action="store_true")

	## Others
	parser.add_argument("--dataset_name", type=str, default="semeval2019", choices=["semeval2019", "Pheme", "twitter15", "twitter16"])
	parser.add_argument("--dataset_root", type=str, default="../dataset/processedV2")
	parser.add_argument("--results_root", type=str, default="/mnt/1T/projects/RumorV2/results")
	parser.add_argument("--n_fold", type=int, default=5)
	parser.add_argument("--fold", type=str, default="0,1,2,3,4,comp", help="either use 5-fold data or train/dev/test from rumoureval2019 competition")

	args = parser.parse_args()

	return args

def plot_wordcloud(args):
	print("\nPlot word cloud...")
	#folds = args.fold.split(",")
	folds = ["comp"]

	for fold in folds:
		print("\nProcessing fold {}...".format(fold))
		path_fold = "{}/{}/split_{}".format(args.dataset_root, args.dataset_name, fold)
		path_data = "{}/{}/data.csv".format(args.dataset_root, args.dataset_name)

		train_ids = pd.read_csv("{}/train.csv".format(path_fold))["source_id"].tolist()
		test_ids  = pd.read_csv("{}/test.csv".format(path_fold))["source_id"].tolist()

		train_texts, test_texts = [], []

		data_df = pd.read_csv(path_data)
		print("Getting texts...")
		for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
			text = row["text"].replace("<end>", "").replace("URL", "")
			text = " ".join(nltk.word_tokenize(text))
		
			if row["tweet_id"] in train_ids:
				train_texts.append(text)
			elif row["tweet_id"] in test_ids:
				test_texts.append(text)

		print("Plotting word cloud...")
		train_texts = " ".join(train_texts)
		wordcloud = WordCloud(width=1000, height=500, background_color="white").generate(train_texts)
		wordcloud.to_file("{}/train.cloud.png".format(path_fold))

		with open("{}/train.cloud.top200.txt".format(path_fold), "w") as fw:
			for word, value in wordcloud.words_.items():
				fw.write("{:.4f}\t{}\n".format(value, word))

		test_texts  = " ".join(test_texts)
		wordcloud = WordCloud(width=1000, height=500, background_color="white").generate(test_texts)
		wordcloud.to_file("{}/test.cloud.png".format(path_fold))

		with open("{}/test.cloud.top200.txt".format(path_fold), "w") as fw:
			for word, value in wordcloud.words_.items():
				fw.write("{:.4f}\t{}\n".format(value, word))

def get_semeval2019_event_from_pheme(args, write=True):
	def remove_hidden_files_dirs(dirs):
		dirs = [dir for dir in dirs if not dir.startswith(".") and "README" not in dir]
		return dirs
	
	path_semeval = "{}/semeval2019/data.csv".format(args.dataset_root)

	path_pheme_0 = "{}/../raw/pheme-rumour-scheme-dataset/threads/en".format(args.dataset_root)
	path_pheme_1 = "{}/../raw/PHEME_veracity/all-rnr-annotated-threads".format(args.dataset_root)
	path_pheme_2 = "{}/../raw/pheme-rnr-dataset".format(args.dataset_root)

	pheme_ids   = {"charliehebdo": [], "ferguson": [], "gurlitt": [], "prince-toronto": [], "sydneysiege": [], "ebola-essien": [], "germanwings-crash": [], "ottawashooting": [], "putinmissing": []}
	semeval_ids = {"charliehebdo": [], "ferguson": [], "gurlitt": [], "prince-toronto": [], "sydneysiege": [], "ebola-essien": [], "germanwings-crash": [], "ottawashooting": [], "putinmissing": []}

	print("\n[PHEME]")
	## pheme-rumour-scheme-dataset
	event_dir = os.listdir(path_pheme_0)
	event_dir = remove_hidden_files_dirs(event_dir)
	for dir in event_dir:
		tids = os.listdir("{}/{}".format(path_pheme_0, dir))
		tids = remove_hidden_files_dirs(tids)
		
		pheme_ids[dir].extend(tids)
	
	## PHEME_veracity
	labels = ["non-rumours", "rumours"]
	event_dir = os.listdir(path_pheme_1)
	event_dir = remove_hidden_files_dirs(event_dir)
	for dir in event_dir:
		for label in labels:
			tids = os.listdir("{}/{}/{}".format(path_pheme_1, dir, label))
			tids = remove_hidden_files_dirs(tids)

			event_name = dir.replace("-all-rnr-threads", "")
			pheme_ids[event_name].extend(tids)
			pheme_ids[event_name] = list(set(pheme_ids[event_name]))
		
		#print("{:20s}: {}".format(event_name, len(pheme_ids[event_name])))
	
	## pheme-rnr-dataset
	event_dir = os.listdir(path_pheme_2)
	event_dir = remove_hidden_files_dirs(event_dir)
	for dir in event_dir:
		for label in labels:
			tids = os.listdir("{}/{}/{}".format(path_pheme_2, dir, label))
			tids = remove_hidden_files_dirs(tids)

			pheme_ids[dir].extend(tids)
			pheme_ids[dir] = list(set(pheme_ids[dir]))
		
	for event in pheme_ids:
		print("{:20s}: {}".format(event, len(pheme_ids[event])))

	print("\nRead & gather source tweet ID from [SemEval2019]...")
	src_ids = []
	data_df = pd.read_csv(path_semeval)
	for idx, row in data_df.iterrows():
		if row["source_id"] != row["tweet_id"]:
			continue
		src_ids.append(str(row["source_id"]))
	
	print("\n[SemEval2019]")
	total = 0
	total_ids = []
	for event in pheme_ids.keys():
		for src_id in src_ids: ## Source IDs in RumorEval2019
			if src_id in pheme_ids[event]:
				semeval_ids[event].append(src_id)
		print("{:20s}: {}".format(event, len(semeval_ids[event])))
		total = total + len(semeval_ids[event])
		total_ids.extend(semeval_ids[event])
	print("{:20s}: {}".format("Total", total))
	print("{:20s}: {}".format("Total", len(list(set(total_ids)))))

	if write:
		with open("{}/semeval2019/event_map.json".format(args.dataset_root), "w") as fw:
			fw.write(json.dumps(semeval_ids, indent=4))
	else:
		return semeval_ids

def get_semeval2019_event(args):
	"""Get different events of dataset (For semeval2019)"""
	event_strs = {
		"charliehebdo": ["charliehebdo", "charlie", "hebdo"], 
		"ferguson": ["ferguson"], 
		"gurlitt": ["gurlitt"], 
		"prince-toronto": ["prince-toronto", "prince", "toronto"], 
		"sydneysiege": ["sydneysiege", "sydney", "siege"], 
		"ebola-essien": ["ebola-essien", "ebola", "essien"], 
		"germanwings-crash": ["germanwings-crash", "germanwings", "crash"], 
		"ottawashooting": ["ottawashooting", "ottawa", "shooting"], 
		"putinmissing": ["putinmissing", "putin", "missing"]

	}
	event_cnt = {"charliehebdo": [], "ferguson": [], "gurlitt": [], "prince-toronto": [], "sydneysiege": [], "ebola-essien": [], "germanwings-crash": [], "ottawashooting": [], "putinmissing": []}
	path_in = "{}/{}/data.csv".format(args.dataset_root, args.dataset_name)

	## Gather texts
	print("\nRead & gather source text content...")
	src_ids, src_texts = [], []
	data_df = pd.read_csv(path_in)
	for idx, row in data_df.iterrows():
		if row["source_id"] != row["tweet_id"]:
			continue
		src_ids.append(row["source_id"])
		src_texts.append(row["text"])

	## Count each event
	print("\nCount each event")
	total = 0
	total_txt = []
	for event in event_cnt.keys():

		## Iterate through all source texts
		for text in src_texts:
			for event_str in event_strs[event]:
				if event_str in text.lower():
					event_cnt[event].append(text)
					break

		print("{:20s}: {:3d}".format(event, len(event_cnt[event])))
		total += len(event_cnt[event])
		total_txt.extend(event_cnt[event])

	print(total)
	print(len(list(set(total_txt))))
	#ipdb.set_trace()

def semeval2019_event_wise_eval(args):
	event_id_map = get_semeval2019_event_from_pheme(args, write=False)
	id_event_map = {}
	for event in event_id_map:
		for id_ in event_id_map[event]:
			id_event_map[id_] = event

	## Read 5-Fold Predictions
	path_in = "{}/semeval2019/bi-tgn-roberta/lr2e-5".format(args.results_root)
	event_preds = {"charliehebdo": [], "ferguson": [], "gurlitt": [], "prince-toronto": [], "sydneysiege": [], "ebola-essien": [], "germanwings-crash": [], "ottawashooting": [], "putinmissing": []}
	event_label = {"charliehebdo": [], "ferguson": [], "gurlitt": [], "prince-toronto": [], "sydneysiege": [], "ebola-essien": [], "germanwings-crash": [], "ottawashooting": [], "putinmissing": []}
	total_preds = 0
	for fold_i in range(args.n_fold):
		preds_df = pd.read_csv("{}/{}/predictions.csv".format(path_in, fold_i))
		
		#print("=" * 32)
		for event in event_id_map:
			event_ids = event_id_map[event]
			event_df  = preds_df.loc[preds_df["source_id"].isin(event_ids)]
			
			#print("{:20s}: {:2d} samples".format(event, len(event_df)))
			if len(event_df) == 0:
				continue

			hard_pred = event_df["hard_pred"].values
			gt_label  = event_df["gt_label"].values

			event_preds[event].append(hard_pred)
			event_label[event].append(gt_label)

	strs, path_out = [], "{}/event_wise.txt".format(path_in)
	header = "{:20s}\t{:9s}\t{:6s}\t{:8s}\t{:8s}\t{:6s}\t{:6s}\t{:6s}\t{:10s}\t{:10s}\t{:10s}".format(
		"Event Name", "# Samples", "Acc", "macro-F1", "micro-F1", "F1-0", "F1-1", "F1-2", "# Label-0", "# Label-1", "# Label-2"
	)
	print("=" * 25)
	print(header)
	if not os.path.isfile(path_out):
		open(path_out, "w").write("{}\n".format(header))

	for event in event_preds:
		if len(event_preds[event]) == 0:
			str_ = "{:20s}\t{:<9d}\t{:.4f}\t{:<8.4f}\t{:<8.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(event, 0, 0, 0, 0, 0, 0, 0)
			strs.append(str_)
			print(str_)
			continue

		preds = np.concatenate(event_preds[event])
		label = np.concatenate(event_label[event])

		F1_all = f1_score_3_class(preds, label)
		F1_all = np.array(F1_all)
		
		## Only take classes that exist in `label`
		indices = np.array(list(set(label)))
		F1_filt = F1_all[indices]

		micro_f1 = f1_score(preds, label, average="micro")
		macro_f1 = sum(F1_filt) / len(F1_filt)

		str_ = "{:20s}\t{:<9d}\t{:.4f}\t{:<8.4f}\t{:<8.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:<10d}\t{:<10d}\t{:<10d}".format(
			event, len(label), (preds == label).sum() / len(label), macro_f1, micro_f1, 
			F1_all[0], F1_all[1], F1_all[2], (label == 0).sum(), (label == 1).sum(), (label == 2).sum()
		)
		strs.append(str_)
		print(str_)

	#with open(path_out, "a") as fw:
	#	for str_ in strs:
	#		fw.write("{}\n".format(str_))

def bootstrap_accuracy(args):
	def accuracy(x):
		return x.sum() / len(x)
	
	path_in = "{}/semeval2019/bi-tgn-roberta/lr2e-5".format(args.results_root)
	
	## Read 5-Fold Predictions
	preds, label = [], []
	for fold_i in range(args.n_fold):
		preds_df = pd.read_csv("{}/{}/predictions.csv".format(path_in, fold_i))
		preds.extend(preds_df["hard_pred"].tolist())
		label.extend(preds_df["gt_label"].tolist())
	
	preds = np.array(preds)
	label = np.array(label)
	
	correct = (preds == label) * 1
	correct = (correct, )
	
	bootstrap_ci = bootstrap(correct, accuracy, confidence_level=0.95, random_state=123, method="percentile")

	plt.hist(bootstrap_ci.bootstrap_distribution, bins=25)
	plt.title("Bootstrap Distribution of RumorEval2019")
	plt.tight_layout()
	plt.savefig("bootstrap.png", dpi=300)
	
	ipdb.set_trace()

if __name__ == "__main__":
	args = parse_args()

	if args.wordcloud:
		plot_wordcloud(args)
	elif args.get_semeval2019_event_from_pheme:
		get_semeval2019_event_from_pheme(args)
	elif args.get_semeval2019_event:
		get_semeval2019_event(args)
	elif args.semeval2019_event_wise_eval:
		semeval2019_event_wise_eval(args)
	elif args.bootstrap_accuracy:
		bootstrap_accuracy(args)

