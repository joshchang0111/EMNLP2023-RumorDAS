import os
import ipdb
import json
import scipy
import shutil
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import krippendorff as kd 

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from tqdm import tqdm
from matplotlib import gridspec
from matplotlib.patches import Patch 
from statsmodels.stats import inter_rater as irr

def parse_args():
	parser = argparse.ArgumentParser(description="Analyze experiment results")

	## Which experiment
	parser.add_argument("--evaluate_A1", action="store_true")
	parser.add_argument("--evaluate_A2", action="store_true")
	parser.add_argument("--evaluate_B1", action="store_true")
	parser.add_argument("--evaluate_B2", action="store_true")
	parser.add_argument("--box_plot_AB", action="store_true")
	parser.add_argument("--box_plot_A", action="store_true")
	parser.add_argument("--bar_plot_B", action="store_true")
	parser.add_argument("--agreement_analysis", action="store_true")

	## Others
	parser.add_argument("--dataset_name", type=str, default="twitter15", choices=["semeval2019", "twitter15", "twitter16"])
	parser.add_argument("--dataset_root", type=str, default="../dataset/processedV2")
	parser.add_argument("--fold", type=str, default="0,1,2,3,4", help="either use 5-fold data or train/dev/test from rumoureval2019 competition")
	parser.add_argument("--result_path", type=str, default="/mnt/1T/projects/RumorV2/results")

	args = parser.parse_args()

	return args

def normalize(scores):
	return [score / 15 for score in scores]

def filter_users(df, users=[], first_only=False):
	email_col = [col for col in df.columns if "Email" in col]
	email_col = email_col[0]
	for user in users:
		user_idx = df.index[df[email_col] == user]
		if first_only:
			user_idx = user_idx[:1]
		df = df.drop(user_idx)
	return df

def read_b_labels():
	questions = []
	files = os.listdir("{}/user_study/B.1".format(args.dataset_root))
	for file in files:
		if file.startswith("."): continue
		if "results" in file: continue
		questions.append(file)
	questions.sort()

	## Get labels
	labels = []
	for question in questions:
		lines = open("{}/user_study/B.1/{}".format(args.dataset_root, question)).readlines()
		label = lines[1].strip().rstrip()
		labels.append(label.lower() == "true")
	labels = np.array(labels)
	return labels

def evaluate_A(args, subtask_id, return_scores=False, return_results=False, verbose=True):
	if verbose:
		print("\nEvaluate A.{}".format(subtask_id))

	def check_trick_question(s1_score, s2_score):
		if subtask_id == 1:
			if s1_score > s2_score:
				return True
		elif subtask_id == 2:
			if s1_score < s2_score:
				return True
		return False

	def eval_single_sample(row, _verbose=False):
		sample_idx = 0
		loo_scores, kmeans_scores = [], []
		for idx in range(0, len(columns), 2):
			s1_score = row[columns[idx]]
			s2_score = row[columns[idx + 1]]
			
			if "10" in columns[idx]:
				if not check_trick_question(s1_score, s2_score) and verbose:
					print("[{:15s}] failed to answer the trick question. s1: {}, s2: {}".format(
						row.iloc[1].split("@")[0][:15], s1_score, s2_score
					))
				continue
			
			loo_score, kmeans_score = None, None
			if mapping_df["summary 1"].iloc[sample_idx] == "LOO":
				loo_score, kmeans_score = s1_score, s2_score
				s1_label, s2_label = "LOO", "kmeans"
			elif mapping_df["summary 1"].iloc[sample_idx] == "kmeans":
				loo_score, kmeans_score = s2_score, s1_score
				s2_label, s1_label = "LOO", "kmeans"
	
			loo_scores.append(loo_score)
			kmeans_scores.append(kmeans_score)

			if _verbose:
				if loo_score == kmeans_score:
					winner = "tie"
				elif loo_score > kmeans_score:
					winner = "LOO"
				else:
					winner = "kmeans"
				print("S1 ({:6s}): {}, S2 ({:6s}): {}, -> {}".format(s1_label, s1_score, s2_label, s2_score, winner))
	
			sample_idx = sample_idx + 1

		if verbose:
			print("[{:15s}] LOO score: {}, kmeans score: {}".format(row.iloc[1].split("@")[0][:15], sum(loo_scores), sum(kmeans_scores)))
		return sum(loo_scores), sum(kmeans_scores)
		
	results_df = pd.read_csv("{}/user_study/A.{}/A{}_results.csv".format(args.dataset_root, subtask_id, subtask_id))
	mapping_df = pd.read_csv("{}/user_study/A_{}_map.csv".format(args.dataset_root, subtask_id), index_col=None)

	if subtask_id == 1:
		results_df = filter_users(results_df, users=["yume16680@gmail.com"], first_only=True)
	elif subtask_id == 2:
		results_df = filter_users(results_df, users=["tongnzong@gmail.com"], first_only=True)
		results_df = filter_users(results_df, users=["minimumeyes@gmail.com"])

	if verbose:
		print("# Samples (Users): {}".format(len(results_df)))

	columns = results_df.columns[2:]
	
	loo_scores, kmeans_scores = [], []
	for human_idx in range(len(results_df)):
		loo_score, kmeans_score = eval_single_sample(results_df.iloc[human_idx])
		loo_scores.append(loo_score)
		kmeans_scores.append(kmeans_score)
	
	if verbose:
		print("=" * 17)
		print("[{:15s}] LOO score: {} ({:.4f}), kmeans score: {} ({:.4f})".format(
			"Total", 
			sum(loo_scores)   , sum(loo_scores) / 15 / len(loo_scores), 
			sum(kmeans_scores), sum(kmeans_scores) / 15 / len(kmeans_scores) 
		))

	if return_scores:
		return loo_scores, kmeans_scores
	
	if return_results:
		return results_df

def evaluate_B(args, subtask_id, return_accs=False, return_results=False, verbose=True):
	if verbose:
		print("\nEvaluate B.{}".format(subtask_id))

	def eval_acc(pred, labels, type):
		n_correct = (pred == labels).sum()
		acc = n_correct / labels.shape[0]
		if verbose:
			print("({}) # correct: {:2d}, Acc.: {:.2f} ".format(type, n_correct, acc), end="")
		return acc
	
	def print_pred(pred):
		for idx, p in enumerate(pred):
			print("{:2d}. {}".format(idx + 1, p))

	results_df = pd.read_csv("{}/user_study/B.{}/B{}_results.csv".format(args.dataset_root, subtask_id, subtask_id))
	results_df = results_df.drop(results_df.columns[0], axis=1) ## time
	results_df = results_df.drop(results_df.columns[-3:], axis=1) ## Payment receive, bank transfer info.

	if verbose:
		print("# Samples (Users): {}".format(len(results_df)))
	
	questions = []
	files = os.listdir("{}/user_study/B.{}".format(args.dataset_root, subtask_id))
	for file in files:
		if file.startswith("."):
			continue
		if "results" in file:
			continue
		questions.append(file)
	questions.sort()

	## Get labels
	labels = []
	for question in questions:
		lines = open("{}/user_study/B.{}/{}".format(args.dataset_root, subtask_id, question)).readlines()
		label = lines[1].strip().rstrip()
		labels.append(label.lower() == "true")
	labels = np.array(labels)

	assert len(labels) == 20, "Number of data files doesn't match number of questions"
	
	## Calculate accuracy of each sample
	accs_tot, accs_sum, accs_res = [], [], []
	for idx, row in results_df.iterrows():
		pred = row.values[1:-1] ## get rid off email, willing to view summary when using social media
		
		if subtask_id == 1:
			label_res, pred_res = labels[:10], pred[:10]
			label_sum, pred_sum = labels[10:], pred[10:]
		elif subtask_id == 2:
			label_res, pred_res = labels[10:],  pred[10:]
			label_sum, pred_sum = labels[:10],  pred[:10]

		if verbose:
			print("[{:15s}] ".format(row.iloc[0].split("@")[0][:15]), end="")
		acc_tot = eval_acc(pred, labels, type="total")
		acc_sum = eval_acc(pred_sum, label_sum, type="summary")
		acc_res = eval_acc(pred_res, label_res, type="response")
		if verbose:
			print()
		
		accs_tot.append(acc_tot)
		accs_sum.append(acc_sum)
		accs_res.append(acc_res)

	if return_accs:
		return accs_tot, accs_sum, accs_res

	if return_results:
		return results_df
	
def box_plot_AB(args):
	print("\nPlot box plot of A & B on same figure")

	path_out = "{}/user_study/figure".format(args.dataset_root)
	os.makedirs(path_out, exist_ok=True)

	loo_scores, kmeans_scores = [], []
	loo_scores_1, kmeans_scores_1 = evaluate_A(args, subtask_id=1, return_scores=True)
	loo_scores_2, kmeans_scores_2 = evaluate_A(args, subtask_id=2, return_scores=True)

	loo_scores.extend(loo_scores_1), kmeans_scores.extend(kmeans_scores_1)
	loo_scores.extend(loo_scores_2), kmeans_scores.extend(kmeans_scores_2)

	loo_scores, kmeans_scores = normalize(loo_scores), normalize(kmeans_scores)

	accs_tot, accs_sum, accs_res = [], [], []
	accs_tot_1, accs_sum_1, accs_res_1 = evaluate_B(args, subtask_id=1, return_accs=True)
	accs_tot_2, accs_sum_2, accs_res_2 = evaluate_B(args, subtask_id=2, return_accs=True)

	accs_tot.extend(accs_tot_1), accs_sum.extend(accs_sum_1), accs_res.extend(accs_res_1)
	accs_tot.extend(accs_tot_2), accs_sum.extend(accs_sum_2), accs_res.extend(accs_res_2)

	n_filter = 5
	for i in range(n_filter):
		loo_scores.remove(max(loo_scores)), kmeans_scores.remove(max(kmeans_scores))
		loo_scores.remove(min(loo_scores)), kmeans_scores.remove(min(kmeans_scores))

		accs_res.remove(max(accs_res)), accs_sum.remove(max(accs_sum)), accs_tot.remove(max(accs_tot))
		accs_res.remove(min(accs_res)), accs_sum.remove(min(accs_sum)), accs_tot.remove(min(accs_tot))

	#ipdb.set_trace()

	data_A = pd.DataFrame({"SSRA-LOO": loo_scores, "SSRA-k-means": kmeans_scores})
	data_B = pd.DataFrame({"Total": accs_tot, "Response": accs_res, "Summary": accs_sum})
	#data_B = pd.DataFrame({"Response": accs_res, "Summary": accs_sum})

	plot_type = "box" #"violin"

	sns.set_style("whitegrid")
	fig, axes = plt.subplots(1, 2, figsize=(8, 5), gridspec_kw={"width_ratios": [5, 5]})
	meanprops   = {"linewidth": 1.5, "color": "gray", "alpha": 1}
	medianprops = {"linewidth": 1.5, "color": "k"   , "alpha": 0.8}
	if plot_type == "box":
		sns.boxplot(ax=axes[0], data=data_A, palette="Pastel2", width=0.6, showmeans=True, meanline=True, meanprops=meanprops, medianprops=medianprops)
		sns.boxplot(ax=axes[1], data=data_B, palette="Pastel2", width=0.6, showmeans=True, meanline=True, meanprops=meanprops, medianprops=medianprops)
		
		## Add dummy lines to show mean and median on legend
		r_tria = mlines.Line2D([], [], **meanprops  , linestyle="--", label="Mean")
		r_line = mlines.Line2D([], [], **medianprops, linestyle="-" , label="Median")

		legend11 = plt.legend(loc="lower right", handles=[r_line, r_tria], fontsize=14)
		axes[1].add_artist(legend11)
		
	elif plot_type == "violin":
		sns.violinplot(ax=axes[0], data=data_A, palette="Set2")
		sns.violinplot(ax=axes[1], data=data_B, palette="Set2")

	axes[0].set_title("Part A (Informativeness)", fontsize=16)
	axes[1].set_title("Part B (Accuracy)", fontsize=16)
	axes[0].set_ylim(0, 5)
	axes[1].set_ylim(0, 1)
	axes[1].yaxis.tick_right()
	axes[0].tick_params(axis="both", labelsize=14)
	axes[1].tick_params(axis="both", labelsize=14, right=False)
	
	plt.tight_layout()
	plt.savefig("{}/user_study_{}_AB.png".format(path_out, plot_type), dpi=300)

def box_plot_A(args):
	print("\nPlot box plot of A...")

	path_out = "{}/user_study/figure".format(args.dataset_root)
	os.makedirs(path_out, exist_ok=True)

	loo_scores, kmeans_scores = [], []
	loo_scores_1, kmeans_scores_1 = evaluate_A(args, subtask_id=1, return_scores=True)
	loo_scores_2, kmeans_scores_2 = evaluate_A(args, subtask_id=2, return_scores=True)

	loo_scores.extend(loo_scores_1), kmeans_scores.extend(kmeans_scores_1)
	loo_scores.extend(loo_scores_2), kmeans_scores.extend(kmeans_scores_2)

	loo_scores, kmeans_scores = normalize(loo_scores), normalize(kmeans_scores)

	n_filter = 5
	for i in range(n_filter):
		loo_scores.remove(max(loo_scores)), kmeans_scores.remove(max(kmeans_scores))
		loo_scores.remove(min(loo_scores)), kmeans_scores.remove(min(kmeans_scores))

	data_A = pd.DataFrame({"SSRA-LOO": loo_scores, "SSRA-k-means": kmeans_scores})

	sns.set_style("whitegrid")
	fig = plt.figure(figsize=(8, 1.4))
	meanprops   = {"linewidth": 1.5, "color": "gray", "alpha": 1}
	medianprops = {"linewidth": 1.5, "color": "k"   , "alpha": 0.8}
	
	sns.boxplot(data=data_A, orient="h", palette="Pastel2", width=0.6, showmeans=True, meanline=True, meanprops=meanprops, medianprops=medianprops)	
	## Add dummy lines to show mean and median on legend
	g_line = mlines.Line2D([], [], **meanprops  , linestyle="--", label="Mean")
	b_line = mlines.Line2D([], [], **medianprops, linestyle="-" , label="Median")
	
	cmap = plt.get_cmap("Pastel2")
	rect_1 = Patch(color=cmap(0), label="SSRA-LOO")
	rect_2 = Patch(color=cmap(1), label="SSRA-k-means")

	plt.xlim(0, 5)
	plt.xticks(fontsize=12)
	plt.yticks([], [])
	plt.title("")
	plt.title("Part A (Informativeness)", fontsize=14, loc="left")
	#plt.legend(loc="lower left"  , handles=[b_line, g_line, rect_1, rect_2], fontsize=14)
	#lgd1 = plt.legend(loc="upper right" , handles=[b_line, g_line], fontsize=14,  ncols=2, frameon=False, bbox_to_anchor=(1, 1.2))
	lgd1 = plt.legend(loc="upper right" , handles=[b_line, g_line], fontsize=12)
	lgd2 = plt.legend(loc="lower left"  , handles=[rect_1, rect_2], fontsize=12)
	axes = fig.axes
	axes[0].add_artist(lgd1)
	axes[0].add_artist(lgd2)
	#plt.tight_layout()
	plt.savefig("{}/user_study_box_A.png".format(path_out), bbox_extra_artists=(lgd1, lgd2), bbox_inches="tight", dpi=300)

def bar_plot_B(args):
	print("\nPlot bar plot for part B")
	def flatten_preds(data_df, start_idx, end_idx):
		data_array = data_df.to_numpy().T.flatten()
		qid = []
		for i in range(start_idx, end_idx):
			qid.extend([i] * len(data_df))
		return pd.DataFrame({"Question ID": qid, "preds": data_array})

	def get_gt_proportion(df1, df2, filter=[]):
		props = []
		for df in [df1, df2]:
			for col in df:
				if col in filter:
					continue
				gt_prop = df[col].value_counts()[labels[col]]
				gt_prop = gt_prop / len(df)
				props.append(gt_prop)
		return np.array(props)

	def get_num_responses():
		root1 = "{}/user_study/B.1".format(args.dataset_root)
		root2 = "{}/user_study/B.2".format(args.dataset_root)
		files1 = ["{}/{}".format(root1, f) for f in os.listdir(root1) if not f.startswith(".") and f.endswith("txt") and int(f.split("-")[0]) < 10]
		files2 = ["{}/{}".format(root2, f) for f in os.listdir(root2) if not f.startswith(".") and f.endswith("txt") and int(f.split("-")[0]) >= 10]
		files = files1 + files2
		files.sort()
		num = []
		for file in files:
			lines = open(file).readlines()
			n_res, flag = 0, False
			for line in lines:
				line = line.strip().rstrip()
				if flag and line != "":
					n_res = n_res + 1
				if line == "Responses (回覆)":
					flag = True
			num.append(n_res)
		return np.array(num)

	def pearson_corr(res1_df, res2_df, sum1_df, sum2_df, filter=[]):
		p_gt_res = get_gt_proportion(res1_df, res2_df, filter=filter)
		p_gt_sum = get_gt_proportion(sum2_df, sum1_df, filter=filter)
		
		pearson_cor_preds = scipy.stats.pearsonr(p_gt_res, p_gt_sum)
		print("Pearson Correlation between predictions: {:.4f}, p-value: {:.4f}".format(pearson_cor_preds.statistic, pearson_cor_preds.pvalue))

		if len(filter) == 0:
			n_res = get_num_responses()
			pearson_cor_n_res = scipy.stats.pearsonr(p_gt_sum - p_gt_res, n_res)
			print("Pearson Correlation between n_responses: {:.4f}, p-value: {:.4f}".format(pearson_cor_n_res.statistic, pearson_cor_n_res.pvalue))
		return p_gt_res, p_gt_sum

	path_out = "{}/user_study/figure".format(args.dataset_root)
	os.makedirs(path_out, exist_ok=True)

	b1_df = pd.read_csv("{}/user_study/B.1/B1_results.csv".format(args.dataset_root))
	b2_df = pd.read_csv("{}/user_study/B.2/B2_results.csv".format(args.dataset_root))
	b1_df = b1_df.drop(b1_df.columns[0], axis=1) ## time
	b1_df = b1_df.drop(b1_df.columns[-3:], axis=1) ## Payment receive, bank transfer info.
	b2_df = b2_df.drop(b2_df.columns[0], axis=1) ## time
	b2_df = b2_df.drop(b2_df.columns[-3:], axis=1) ## Payment receive, bank transfer info.

	preds_b1, preds_b2 = [], []
	for idx, row in b1_df.iterrows():
		preds = row.values[1:-1]
		preds_b1.append(preds)
	for idx, row in b2_df.iterrows():
		preds = row.values[1:-1]
		preds_b2.append(preds)

	b1_df = pd.DataFrame(preds_b1)
	b2_df = pd.DataFrame(preds_b2)

	res1_df, sum1_df = b1_df[b1_df.columns[:10]], b1_df[b1_df.columns[10:]]
	sum2_df, res2_df = b2_df[b2_df.columns[:10]], b2_df[b2_df.columns[10:]]

	labels = read_b_labels()

	## Remove top 5% and bottom 5% outliers
	#res1_accs, res2_accs, sum1_accs, sum2_accs = [], [], [], []
	#for row_idx in range(len(res1_df)):
	#	res1_accs.append((res1_df.iloc[row_idx].values == labels[:10]).sum() / 10)
	#	res2_accs.append((res2_df.iloc[row_idx].values == labels[10:]).sum() / 10)
	#	sum1_accs.append((sum1_df.iloc[row_idx].values == labels[10:]).sum() / 10)
	#	sum2_accs.append((sum2_df.iloc[row_idx].values == labels[:10]).sum() / 10)
	#
	#res1_accs, res2_accs, sum1_accs, sum2_accs = np.array(res1_accs), np.array(res2_accs), np.array(sum1_accs), np.array(sum2_accs)
	#res1_df = res1_df.drop(res1_accs.argsort()[:2]).drop(res1_accs.argsort()[-3:])
	#res2_df = res2_df.drop(res2_accs.argsort()[:3]).drop(res1_accs.argsort()[-2:])
	#sum1_df = sum1_df.drop(sum1_accs.argsort()[:1]).drop(sum1_accs.argsort()[-4:])
	#sum2_df = sum2_df.drop(sum2_accs.argsort()[:4]).drop(sum2_accs.argsort()[-1:])

	plot_type = "acc_avg"
	if plot_type == "acc_avg":
		## Calculate pearson correlation
		p_gt_res, p_gt_sum = pearson_corr(res1_df, res2_df, sum1_df, sum2_df)
		#pearson_corr(res1_df, res2_df, sum1_df, sum2_df, filter=[1, 3, 19])
		#pearson_corr(res1_df, res2_df, sum1_df, sum2_df, filter=[1, 3, 12, 19])

		n_dec = ((p_gt_sum - p_gt_res) < 0).sum()
		n_inc = ((p_gt_sum - p_gt_res) > 0).sum()
		n_equ = ((p_gt_sum - p_gt_res) == 0).sum()
		print("# decrease: {}".format(n_dec))
		print("# increase: {}".format(n_inc))
		print("# equal   : {}".format(n_equ))

		p_gt_df = pd.DataFrame(
			{
				"type"   : ["Response", "Summary"], 
				"correct": [100 * np.mean(p_gt_res), 100 * np.mean(p_gt_sum)], 
				"wrong"  : [100, 100]
			}
		)
		p_gt_means = [np.mean(p_gt_res), np.mean(p_gt_sum)]
		palette = sns.color_palette("Set3") ## Spectral cividis Accent flare
		fig, axes = plt.subplots(2, 1, figsize=(8, 0.8), sharex=True)
		axes[0].spines[["top", "right", "bottom", "left"]].set_visible(False)
		axes[1].spines[["top", "right", "bottom", "left"]].set_visible(False)
		yticks = ["Response", "Summary"]
		for idx, p_gt_mean in enumerate(p_gt_means):
			ax = sns.barplot(ax=axes[idx], x="wrong"  , y="type", data=p_gt_df.iloc[idx:idx + 1], orient="h", width=1, color=palette[0])
			ax = sns.barplot(ax=axes[idx], x="correct", y="type", data=p_gt_df.iloc[idx:idx + 1], orient="h", width=1, color=palette[-1])
			ax.bar_label(ax.containers[0], padding=-37 , fmt=lambda x: "{:.1f}%".format(100 * (1 - p_gt_mean)))#, color="white")
			ax.bar_label(ax.containers[0], padding=-440, fmt=lambda x: "{:.1f}%".format(100 * p_gt_mean))#, color="white")
			ax.bar_label(ax.containers[0], padding=-247, fmt=lambda x: "{}".format(yticks[idx]))
		rect_1 = Patch(color=palette[0] , label="Wrong")
		rect_2 = Patch(color=palette[-1], label="Correct")
		#lgd = axes[0].legend(handles=[rect_2, rect_1], ncols=2, loc="upper center", bbox_to_anchor=(0.5, 2.1))
		lgd = axes[0].legend(handles=[rect_2, rect_1], ncols=2, loc="upper right", bbox_to_anchor=(1.025, 2.3), frameon=False, fontsize=12)
		plt.xlim(0, 100)
		#plt.xlabel("Upper: Response | Lower: Summary")
		plt.xlabel(None)
		plt.xticks([i * 10 for i in range(11)])
		axes[0].set_ylabel(None)
		axes[1].set_ylabel(None)
		axes[0].grid(axis="x", linestyle="--")
		axes[1].grid(axis="x", linestyle="--")
		axes[0].tick_params(axis="both", length=0)
		axes[1].tick_params(axis="y", length=0)
		axes[0].set_title("Part B (% of Predictions per sample)", loc="left", fontsize=14)
		axes[0].set_yticklabels([])
		axes[1].set_xticklabels(["{}%".format(tick.get_text()) for tick in axes[1].get_xticklabels()])
		axes[1].set_yticklabels([])
		plt.savefig("{}/user_study_acc_B.png".format(path_out), bbox_extra_artists=([lgd]), bbox_inches="tight", dpi=300)

	elif plot_type == "p_gt_sep":
		res1_df = flatten_preds(res1_df, start_idx=0, end_idx=10)
		res2_df = flatten_preds(res2_df, start_idx=10, end_idx=20)
		sum1_df = flatten_preds(sum1_df, start_idx=10, end_idx=20)
		sum2_df = flatten_preds(sum2_df, start_idx=0, end_idx=10)

		res_df = pd.concat([res1_df, res2_df]).reset_index(drop=True)
		sum_df = pd.concat([sum2_df, sum1_df]).reset_index(drop=True)

		##########
		## Plot ##
		##########
		palette = [sns.color_palette("flare")[-1], sns.color_palette("flare")[0]]
		fig, axes = plt.subplots(2, 1, figsize=(15, 5), sharex=True, sharey=True)
		axes[0].scatter([], [], marker="+", s=50, color="k", label="Ground-Truth")
		sns.countplot(x="Question ID", hue="preds", data=res_df, ax=axes[0], width=0.7, palette=palette)
		sns.countplot(x="Question ID", hue="preds", data=sum_df, ax=axes[1], width=0.7, palette=palette)

		## Add ground-truth to bar labels
		bar_labels_t, bar_labels_f = np.array([""] * 20), np.array([""] * 20)
		bar_labels_t[np.where(labels == True )[0]] = "+"
		bar_labels_f[np.where(labels == False)[0]] = "+"

		axes[0].bar_label(container=axes[0].containers[0], labels=bar_labels_f)
		axes[0].bar_label(container=axes[0].containers[1], labels=bar_labels_t)
		axes[1].bar_label(container=axes[1].containers[0], labels=bar_labels_f)
		axes[1].bar_label(container=axes[1].containers[1], labels=bar_labels_t)
		axes[0].margins(y=0.15)
		axes[1].margins(y=0.15)

		axes[0].legend(loc="upper right", bbox_to_anchor=(1, 1.3), ncol=3, fontsize=14)
		#axes[0].set(xlabel=None, ylabel=None)
		axes[0].set_xlabel(None)
		axes[0].set_ylabel("Response", fontsize=14)
		axes[1].get_legend().remove()
		axes[1].set_xlabel("Question ID", fontsize=14)
		axes[1].set_ylabel("Summary", fontsize=14)
		axes[0].tick_params(axis="both", labelsize=12)
		axes[1].tick_params(axis="both", labelsize=12)
		axes[0].yaxis.set_ticks([20, 40])
		axes[1].yaxis.set_ticks([20, 40])
		#fig.supylabel("Number of Predictions", fontsize=14, x=0.01)
		axes[0].set_title("Number of predictions based on Response / Summary", fontsize=16, loc="left", y=1.05)
		plt.tight_layout()
		plt.savefig("{}/user_study_bar_B.png".format(path_out), dpi=300)

def agreement_analysis(args):
	"""
	Calculate the inter-rater agreement for part A & B.
	
	Questions Arrangement:
	- A1: 16 Questions (QID=10 represents trick question)
	- A2: 16 Questions (QID=10 represents trick question)
	- B1: 10 Responses 10 Summary
	- B2: 10 Summary 10 Responses
	"""
	def agreement_by_spearson(data_arr, n_iter=25):
		n_sample = len(data_arr)
		for i in range(n_iter):
			np.random.shuffle(data_arr)
			data_1, data_2 = data_arr[:int(n_sample/2)], data_arr[int(n_sample/2):]

			spearman_corr, p = scipy.stats.spearmanr(data_1, data_2)
		
			print("Spearman Correlation: {}".format(spearman_corr))

	############
	## Part A ##
	############
	loo_scores_1, kmeans_scores_1 = evaluate_A(args, subtask_id=1, return_scores=True, verbose=False)
	loo_scores_2, kmeans_scores_2 = evaluate_A(args, subtask_id=2, return_scores=True, verbose=False)

	n_filter = 3
	for i in range(n_filter):
		loo_scores_1.remove(max(loo_scores_1)), kmeans_scores_1.remove(max(kmeans_scores_1))
		loo_scores_1.remove(min(loo_scores_1)), kmeans_scores_1.remove(min(kmeans_scores_1))
		loo_scores_2.remove(max(loo_scores_2)), kmeans_scores_2.remove(max(kmeans_scores_2))
		loo_scores_2.remove(min(loo_scores_2)), kmeans_scores_2.remove(min(kmeans_scores_2))
	
	loo_scores_1, kmeans_scores_1 = np.array(loo_scores_1), np.array(kmeans_scores_1)
	loo_scores_2, kmeans_scores_2 = np.array(loo_scores_2), np.array(kmeans_scores_2)

	## Krippendorff's Alpha
	a1_results = np.array([loo_scores_1, kmeans_scores_1]).T
	a2_results = np.array([loo_scores_2, kmeans_scores_2]).T
	a1_rel = kd.alpha(a1_results, level_of_measurement="ordinal")
	a2_rel = kd.alpha(a2_results, level_of_measurement="ordinal")
	print("Krippendorff's Alpha")
	print("A.1 Reliability: {}".format(a1_rel))
	print("A.2 Reliability: {}".format(a2_rel))

	## Fleiss Kappa
	print("=" * 35)
	print("Fleiss Kappa")
	for subtask_id in range(1, 3):
		results_df = evaluate_A(args, subtask_id=subtask_id, return_results=True, verbose=False)
		results_df = results_df.drop(results_df.columns[:2], axis=1)

		score_comp = []
		for idx in range(int(results_df.values.shape[1] / 2)):
			if idx == 10:
				continue
			comp = []
			for row in results_df.values:
				if row[2 * idx] <= row[2 * idx + 1]:
					comp.append(1)
				else:
					comp.append(0)
			score_comp.append(comp)
		score_comp = np.array(score_comp)
		score_agg = irr.aggregate_raters(score_comp)
		score_kappa = irr.fleiss_kappa(score_agg[0], method="fleiss")
		print("A.{} Reliability: {}".format(subtask_id, score_kappa))
	
	############
	## Part B ##
	############
	b1_results = evaluate_B(args, subtask_id=1, return_results=True, verbose=False).values[:, 1:-1]
	b2_results = evaluate_B(args, subtask_id=2, return_results=True, verbose=False).values[:, 1:-1]
	res1_results, res2_results = b1_results[:, :10].T, b2_results[:, 10:].T
	sum1_results, sum2_results = b1_results[:, 10:].T, b2_results[:, :10].T

	## 0: False, 1: True
	res1_agg = irr.aggregate_raters(res1_results)
	res2_agg = irr.aggregate_raters(res2_results)
	sum1_agg = irr.aggregate_raters(sum1_results)
	sum2_agg = irr.aggregate_raters(sum2_results)
	res1_kappa = irr.fleiss_kappa(res1_agg[0], method="fleiss")
	res2_kappa = irr.fleiss_kappa(res2_agg[0], method="fleiss")
	sum1_kappa = irr.fleiss_kappa(sum1_agg[0], method="fleiss")
	sum2_kappa = irr.fleiss_kappa(sum2_agg[0], method="fleiss")

	print("=" * 36)
	print("Response [1] Fleiss Kappa: {}".format(res1_kappa))
	print("Response [2] Fleiss Kappa: {}".format(res2_kappa))
	print("Summary  [1] Fleiss Kappa: {}".format(sum1_kappa))
	print("Summary  [2] Fleiss Kappa: {}".format(sum2_kappa))

if __name__ == "__main__":
	args = parse_args()

	if args.evaluate_A1:
		evaluate_A(args, subtask_id=1)
	elif args.evaluate_A2:
		evaluate_A(args, subtask_id=2)
	elif args.evaluate_B1:
		evaluate_B(args, subtask_id=1)
	elif args.evaluate_B2:
		evaluate_B(args, subtask_id=2)
	elif args.box_plot_AB:
		box_plot_AB(args)
	elif args.box_plot_A:
		box_plot_A(args)
	elif args.bar_plot_B:
		bar_plot_B(args)
	elif args.agreement_analysis:
		agreement_analysis(args)