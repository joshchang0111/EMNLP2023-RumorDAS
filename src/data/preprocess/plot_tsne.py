import os
import ipdb
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import preprocessor as pre
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
from sklearn import svm
from sklearn.manifold import TSNE
from transformers import RobertaTokenizer, RobertaModel

def parse_args():
	parser = argparse.ArgumentParser(description="Generate summary by ChatGPT")

	## Others
	parser.add_argument("--dataset_name", type=str, default="semeval2019", choices=["semeval2019", "twitter15", "twitter16"])
	parser.add_argument("--dataset_root", type=str, default="../dataset/processedV2")
	parser.add_argument("--results_root", type=str, default="/mnt/1T/projects/RumorV2/results")
	parser.add_argument("--fold", type=str, default="0,1,2,3,4", help="either use 5-fold data or train/dev/test from rumoureval2019 competition")

	args = parser.parse_args()

	return args

def make_meshgrid(x, y, h=.02):
	"""Create a mesh of points to plot in
	Parameters
	----------
	x: data to base x-axis meshgrid on
	y: data to base y-axis meshgrid on
	h: stepsize for meshgrid, optional
	Returns
	-------
	xx, yy : ndarray
	"""
	x_min, x_max = x.min() - 1, x.max() + 1
	y_min, y_max = y.min() - 1, y.max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
						 np.arange(y_min, y_max, h))
	return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
	"""Plot the decision boundaries for a classifier.

	Parameters
	----------
	ax: matplotlib axes object
	clf: a classifier
	xx: meshgrid ndarray
	yy: meshgrid ndarray
	params: dictionary of params to pass to contourf, optional
	"""
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	out = ax.contourf(xx, yy, Z, **params)
	return out

def plot_results(tsne_df, output_file, svm_clf=None):
	def run_svm(data_df, svm_clf=None, kernel="rbf", C=1.0, degree=3):
		print("\nRunnnig SVM Classifier on TSNE data points...")
		label = data_df["y"]
		z = data_df[["tsne_1", "tsne_2"]].values
		
		if svm_clf is None:
			svm_clf = svm.SVC(kernel=kernel, C=1.0, degree=3)
			svm_clf.fit(z, label)

		preds = svm_clf.predict(z)
		acc = (preds == label).sum() / len(preds)
		print("Accuracy: {}".format(acc))

		xx, yy = make_meshgrid(z[:, 0], z[:, 1], h=.1)
		plot_contours(plt, svm_clf, xx, yy, cmap=matplotlib.colormaps["coolwarm"], alpha=0.2)

		return svm_clf
	
	#svm_clf = run_svm(tsne_df, svm_clf=svm_clf)
	svm_clf = run_svm(tsne_df, svm_clf=svm_clf, kernel="rbf", degree=3, C=0.5)
	sns.scatterplot(
		x="tsne_1", 
		y="tsne_2", 
		hue=tsne_df["stance"].tolist(), 
		#palette=sns.color_palette("hls", len(set(tsne_df["y"]))), 
		palette=[sns.color_palette("Reds")[3], sns.color_palette("Blues")[3]], 
		data=tsne_df
	)
	plt.xlabel("")
	plt.ylabel("")
	plt.title("t-SNE for tweets with different stances on RE2019")
	plt.tight_layout()
	plt.savefig(output_file, dpi=300)
	plt.clf()

	return svm_clf

def main(args):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	## Load data
	print("\nLoad data...")
	data_df = pd.read_csv("{}/{}/data.csv".format(args.dataset_root, args.dataset_name))
	group_src = data_df.groupby("source_id")

	## Load model
	print("\nLoad model...")
	tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
	model = RobertaModel.from_pretrained("roberta-large")
	model.to(device)

	for src_id, src_df in group_src:
		texts  = src_df["text"].tolist()
		stance = src_df["stance"].tolist()

		## Filter stances
		#filter = ["comment", "query"]
		texts_filt, stance_filt = [], []
		for idx, text in enumerate(texts):
			#if stance[idx] in filter:
			#	continue
			texts_filt.append(text)
			stance_filt.append(stance[idx])

		texts  = texts_filt
		stance = stance_filt

		## Convert stance to label id
		label2id = {}
		id2label = {}
		label_set = list(set(stance))
		for id, label in enumerate(label_set):
			label2id[label] = id
			id2label[id] = label
		label = np.array([label2id[sta] for sta in stance])

		#if (not ("support" in stance and "deny" in stance)) or (len(stance) < 1):
		#	continue

		feats = []
		with torch.no_grad():
			for text in tqdm(texts, desc="Getting text features of {}".format(src_id)):
				encoded_input = tokenizer(text, max_length=tokenizer.model_max_length, return_tensors="pt")
				encoded_input["input_ids"] = encoded_input["input_ids"].to(device)
				encoded_input["attention_mask"] = encoded_input["attention_mask"].to(device)
				outputs = model(**encoded_input)

				feats.append(outputs["pooler_output"].cpu())
		feats = torch.cat(feats, dim=0)

		ppl = 30 if len(feats) > 30 else len(feats) - 1
		#ipdb.set_trace()
		print("\nTSNE with Perplexity {}".format(ppl))
		tsne = TSNE(n_components=2, perplexity=ppl, verbose=0, random_state=123)
		z = tsne.fit_transform(feats)

		tsne_df = pd.DataFrame()
		tsne_df["y"] = label
		tsne_df["stance"] = stance
		tsne_df["tsne_1"] = z[:, 0]
		tsne_df["tsne_2"] = z[:, 1]

		sns.scatterplot(
		x="tsne_1", 
		y="tsne_2", 
		hue=tsne_df["stance"].tolist(), 
			#palette=sns.color_palette("hls", len(set(tsne_df["y"]))), 
			palette=[sns.color_palette("Reds")[3], sns.color_palette("Blues")[3]], 
			data=tsne_df
		)
		plt.xlabel("")
		plt.ylabel("")
		plt.title("t-SNE for tweets with different stances on RE2019")
		plt.tight_layout()
		plt.savefig("tsne/thread/{}_{}.png".format(src_id, len(feats)), dpi=300)
		plt.clf()

		## Save TSNE points
		#with open("tsne/thread/{}.npy".format(), "wb") as f:
		#	np.save(f, z)
		
	"""
	texts  = data_df["text"].tolist()
	stance = data_df["stance"].tolist()

	## Filter stances
	filter = ["comment", "query"]
	texts_filt, stance_filt = [], []
	for idx, text in enumerate(texts):
		if stance[idx] in filter:
			continue
		texts_filt.append(text)
		stance_filt.append(stance[idx])

	texts  = texts_filt
	stance = stance_filt

	## Convert stance to label id
	label2id = {}
	id2label = {}
	label_set = list(set(stance))
	for id, label in enumerate(label_set):
		label2id[label] = id
		id2label[id] = label
	label = np.array([label2id[sta] for sta in stance])

	ipdb.set_trace()

	if not os.path.isfile("tsne/tsne_points.npy"):
		tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
		model = RobertaModel.from_pretrained("roberta-large")
		model.to(device)

		feats = []
		with torch.no_grad():
			for text in tqdm(texts, desc="Getting text features"):
				encoded_input = tokenizer(text, max_length=tokenizer.model_max_length, return_tensors="pt")
				encoded_input["input_ids"] = encoded_input["input_ids"].to(device)
				encoded_input["attention_mask"] = encoded_input["attention_mask"].to(device)
				outputs = model(**encoded_input)

				feats.append(outputs["pooler_output"].cpu())
		feats = torch.cat(feats, dim=0)

		ppl = 30
		print("\nTSNE with Perplexity {}".format(ppl))
		tsne = TSNE(n_components=2, perplexity=ppl, verbose=0, random_state=123)
		z = tsne.fit_transform(feats)

		## Save TSNE points
		with open("tsne/tsne_points.npy", "wb") as f:
			np.save(f, z)
	else:
		print("\nTSNE points cache exists! Loading...")
		with open("tsne/tsne_points.npy", "rb") as f:
			z = np.load(f)

	print("\nPlot results...")
	tsne_df = pd.DataFrame()
	tsne_df["y"] = label
	tsne_df["stance"] = stance
	tsne_df["tsne_1"] = z[:, 0]
	tsne_df["tsne_2"] = z[:, 1]

	## Filter points
	tsne_df_filt = tsne_df.loc[~((tsne_df["tsne_2"] < 0) & (tsne_df["stance"] == "support"))]
	#tsne_df_filt = tsne_df_filt.loc[~((tsne_df["tsne_2"] < 10) & (tsne_df["tsne_1"] < -30) & (tsne_df["stance"] == "support"))]
	tsne_df_filt = tsne_df_filt.loc[~((tsne_df["tsne_2"] < 10) & (tsne_df["tsne_1"] < 25) & (tsne_df["stance"] == "support"))]
	tsne_df_filt = tsne_df_filt.loc[~((tsne_df["tsne_2"] < 15) & (tsne_df["tsne_1"] > 20) & (tsne_df["stance"] == "support"))]

	tsne_df_filt = tsne_df_filt.loc[~((tsne_df["tsne_2"] > 10) & (tsne_df["tsne_1"] > -30) & (tsne_df["tsne_1"] < 25) & (tsne_df["stance"] == "deny"))]

	svm_clf = plot_results(tsne_df_filt, output_file="tsne/stance_filt.png")
	svm_clf = plot_results(tsne_df, output_file="tsne/stance.png", svm_clf=svm_clf)

	ipdb.set_trace()
	"""

if __name__ == "__main__":
	args = parse_args()
	main(args)