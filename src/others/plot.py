import os
import ipdb
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from tqdm import tqdm
from scipy.interpolate import make_interp_spline, PchipInterpolator

plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["mathtext.fontset"] = "stixsans"

dataset_abb_map = {
	"semeval2019": "re2019", 
	"twitter15": "t15", 
	"twitter16": "t16"
}

def parse_args():
	parser = argparse.ArgumentParser(description="Analyze experiment results")

	## Which experiment
	parser.add_argument("--plot_extract_ratio", action="store_true")
	parser.add_argument("--plot_all_extract_ratio", action="store_true")
	parser.add_argument("--plot_response_impact", action="store_true")

	## Others
	parser.add_argument("--dataset_name", type=str, default="semeval2019", choices=["semeval2019", "Pheme", "twitter15", "twitter16"])
	parser.add_argument("--dataset_root", type=str, default="../dataset/processed")
	parser.add_argument("--dataset_root_V2", type=str, default="../dataset/processedV2")
	parser.add_argument("--fold", type=str, default="0,1,2,3,4", help="either use 5-fold data or train/dev/test from rumoureval2019 competition")
	parser.add_argument("--result_path", type=str, default="/mnt/1T/projects/RumorV2/results")

	args = parser.parse_args()

	return args

def plot_extract_ratio(args):
	print("\nPlot extract ratio for [{}]".format(args.dataset_name))
	df = pd.read_csv("{}/{}/bi-tgn/adv-stage2/ext_ratio.tsv".format(args.result_path, args.dataset_name), sep="\t")
	df = df.dropna()
	data_df = df.copy()
	data_df["params"] = data_df["Summarizer"].apply(lambda x: x.replace("CARS (", "").replace(")", ""))
	data_df["rho"] = data_df["params"].apply(lambda x: x.split(",")[0].split("=")[-1])
	data_df["k"] = data_df["params"].apply(lambda x: x.split(",")[-1].split("=")[-1])

	data_df["F1-Macro"] = data_df["F1-Macro"].astype(float)	

	mf1_top, mf1_bottom = data_df.loc[0]["F1-Macro"], data_df.loc[1]["F1-Macro"]
	asr_top = float(data_df.loc[1]["ASR"])
	data_df = data_df.drop([0, 1])

	data_df["k"] = data_df["k"].astype(int)
	data_df["rho"] = data_df["rho"].astype(float)
	data_df["ASR"] = data_df["ASR"].astype(float)

	img_dir = "hor"

	cmap = plt.get_cmap("Dark2")
	markers = ["o", "^", "s", "x", "D"]
	if img_dir == "ver":
		fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

		#colors = [cmap(0), cmap(0), cmap(1), cmap(2), cmap(6)]
		for i, (k, group) in enumerate(data_df.groupby("k")):
			if k == 1:
				continue
			x = group["rho"].tolist()
			mf1 = group["F1-Macro"].astype(float).tolist()
			asr = group["ASR"].astype(float).tolist()

			x = [1 - x_ for x_ in x]

			axes[0].plot(x, mf1, label="DAS, k={}".format(k), marker=markers[i], color=cmap(i), linewidth=2, markersize=10)
			axes[1].plot(x, asr, label="DAS, k={}".format(k), marker=markers[i], color=cmap(i), linewidth=2, markersize=10)

		axes[0].set_xlim([0.05, 1])
		axes[1].set_xlim([0.05, 1])
		#axes[0].set_ylim([0.2, 0.9])
		#axes[1].set_ylim([0.05, 1])

		## Add dummy lines to make new line on legend
		#axes[0].plot(x, mf1, label=" ", color="w", linewidth=0, markersize=10)
		#axes[0].plot(x, mf1, label=" ", color="w", linewidth=0, markersize=10)
		#axes[0].plot(x, mf1, label=" ", color="w", linewidth=0, markersize=10)

		axes[0].hlines(y=mf1_top   , xmin=x[0], xmax=x[-1], linestyles='--', linewidth=3, color=cmap(0))#, color=cmap(6), label="w/o DAS, w/o attack")
		axes[0].hlines(y=mf1_bottom, xmin=x[0], xmax=x[-1], linestyles='--', linewidth=3, color="crimson")#, color=cmap(7), label="w/o DAS, w/ attack")
		axes[1].hlines(y=asr_top   , xmin=x[0], xmax=x[-1], linestyles='--', linewidth=3, color="crimson")#, color=cmap(7), label="w/o DAS, w/ attack")

		## Add dummy lines to make new line on legend
		#axes[1].plot(x, asr, label=" ", color="w", linewidth=0, markersize=10)
		#axes[1].plot(x, asr, label=" ", color="w", linewidth=0, markersize=10)
		#axes[1].plot(x, asr, label=" ", color="w", linewidth=0, markersize=10)

		axes[0].set_ylabel("Macro F$_1$", fontsize=20)#, loc="top")#, rotation=0)
		axes[1].set_xlabel(r"$1 - \rho$", fontsize=20), axes[1].set_ylabel("ASR", fontsize=20)#, loc="top")#, rotation=0)

		axes[0].tick_params(axis='both', which='both', length=0, labelsize=15)
		axes[1].tick_params(axis='both', which='both', length=0, labelsize=15)

		axes[0].legend(loc="lower left", fontsize=20, frameon=False, ncols=1, bbox_to_anchor=(0.67, 0.05))
		axes[1].legend(loc="upper left", fontsize=20, frameon=False, ncols=1, bbox_to_anchor=(0.67, 0.95))
		#axes[0].legend(loc="lower left", fontsize=20, frameon=False, ncols=1, bbox_to_anchor=(0.03, 0.05))
		#axes[1].legend(loc="upper left", fontsize=20, frameon=False, ncols=1, bbox_to_anchor=(0.03, 0.95))

		plt.tight_layout()
		#plt.gca().invert_xaxis()
		plt.savefig("{}/{}/bi-tgn/adv-stage2/ext_ratio.png".format(args.result_path, args.dataset_name), dpi=300)
	
	elif img_dir == "hor":
		fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)

		for i, (k, group) in enumerate(data_df.groupby("k")):
			if k == 1:
				continue
			x = group["rho"].tolist()
			mf1 = group["F1-Macro"].astype(float).tolist()
			asr = group["ASR"].astype(float).tolist()

			x = [1 - x_ for x_ in x]

			axes[0].plot(x, mf1, label="DAS, k={}".format(k), marker=markers[i], color=cmap(i), linewidth=2, markersize=10)
			axes[1].plot(x, asr, label="DAS, k={}".format(k), marker=markers[i], color=cmap(i), linewidth=2, markersize=10)

		gline = axes[0].hlines(y=mf1_top   , xmin=x[0], xmax=x[-1], linestyles='--', linewidth=3, label="w/o DAS, w/o attack", color=cmap(0))
		rline = axes[0].hlines(y=mf1_bottom, xmin=x[0], xmax=x[-1], linestyles='--', linewidth=3, label="w/o DAS, w/ attack" , color="crimson")
		axes[1].hlines(y=asr_top, xmin=x[0], xmax=x[-1], linestyles='--', linewidth=3, color="crimson")

		axes[0].set_title("Macro F$_1$", fontsize=20)
		axes[1].set_title("ASR", fontsize=20)

		axes[1].yaxis.tick_right()
		axes[0].tick_params(axis="both", which="both", labelsize=15)
		axes[1].tick_params(axis="both", which="both", labelsize=15)
		#axes[0].set_ylabel("Macro F$_1$", fontsize=15)#, loc="top")#, rotation=0)
		#axes[1].yaxis.set_label_position("right")
		#axes[1].set_ylabel("ASR", fontsize=15, rotation=-90)

		axes[0].legend(loc="lower left", fontsize=18, frameon=False, ncols=1, bbox_to_anchor=(0.01, 0.04), handles=[gline, rline])
		axes[1].legend(loc="upper left", fontsize=18, frameon=False, ncols=1, bbox_to_anchor=(0.45, 0.95))

		fig.supxlabel(r"$1 - \rho$", fontsize=20, y=0.05)
		plt.tight_layout()
		plt.savefig("{}/{}/bi-tgn/adv-stage2/ext_ratio_{}.png".format(args.result_path, args.dataset_name, dataset_abb_map[args.dataset_name]), dpi=300)

def plot_all_extract_ratio(args):
	datasets = ["semeval2019", "twitter15", "twitter16"]
	
	fig, axes = plt.subplots(3, 2, figsize=(10, 10.5), sharex=True)

	for dataset_idx, dataset in enumerate(datasets):
		print("Plot extract ratio for [{}]".format(dataset))
		df = pd.read_csv("{}/{}/bi-tgn/adv-stage2/ext_ratio.tsv".format(args.result_path, dataset), sep="\t")
		df = df.dropna()
		data_df = df.copy()
		data_df["params"] = data_df["Summarizer"].apply(lambda x: x.replace("CARS (", "").replace(")", ""))
		data_df["rho"] = data_df["params"].apply(lambda x: x.split(",")[0].split("=")[-1])
		data_df["k"] = data_df["params"].apply(lambda x: x.split(",")[-1].split("=")[-1])

		data_df["F1-Macro"] = data_df["F1-Macro"].astype(float)	

		mf1_top, mf1_bottom = data_df.loc[0]["F1-Macro"], data_df.loc[1]["F1-Macro"]
		asr_top = float(data_df.loc[1]["ASR"])
		data_df = data_df.drop([0, 1])

		data_df["k"] = data_df["k"].astype(int)
		data_df["rho"] = data_df["rho"].astype(float)
		data_df["ASR"] = data_df["ASR"].astype(float)

		cmap = plt.get_cmap("Dark2")
		markers = ["o", "^", "s", "x", "D"]

		for i, (k, group) in enumerate(data_df.groupby("k")):
			if k == 1:
				continue
			x = group["rho"].tolist()
			mf1 = group["F1-Macro"].astype(float).tolist()
			asr = group["ASR"].astype(float).tolist()

			x = [1 - x_ for x_ in x]

			axes[dataset_idx, 0].plot(x, mf1, label="DAS, k={}".format(k), marker=markers[i], color=cmap(i), linewidth=2, markersize=10)
			axes[dataset_idx, 1].plot(x, asr, label="DAS, k={}".format(k), marker=markers[i], color=cmap(i), linewidth=2, markersize=10)

		gline = axes[dataset_idx, 0].hlines(y=mf1_top   , xmin=x[0], xmax=x[-1], linestyles='--', linewidth=3, label="w/o DAS, w/o attack", color=cmap(0))
		rline = axes[dataset_idx, 0].hlines(y=mf1_bottom, xmin=x[0], xmax=x[-1], linestyles='--', linewidth=3, label="w/o DAS, w/ attack" , color="crimson")
		axes[dataset_idx, 1].hlines(y=asr_top, xmin=x[0], xmax=x[-1], linestyles='--', linewidth=3, color="crimson")

		#axes[dataset_idx, 0].set_ylabel(dataset.capitalize().replace("Semeval", "RE"), fontsize=18)
		if dataset_idx == 0:
			axes[dataset_idx, 0].set_title("Macro F$_1$", fontsize=20)
			axes[dataset_idx, 1].set_title("ASR", fontsize=20)

			axes[dataset_idx, 0].legend(loc="lower left", fontsize=16, frameon=False, ncols=1, bbox_to_anchor=(0.01, 0.02), handles=[gline, rline])
			axes[dataset_idx, 1].legend(loc="upper left", fontsize=16, frameon=False, ncols=1, bbox_to_anchor=(0.52, 0.95))

		axes[dataset_idx, 1].yaxis.tick_right()
		axes[dataset_idx, 0].tick_params(axis="both", which="both", labelsize=15)
		axes[dataset_idx, 1].tick_params(axis="both", which="both", labelsize=15)

		if dataset_idx < len(datasets) - 1:
			axes[dataset_idx, 0].tick_params(axis="x", length=0)
			axes[dataset_idx, 1].tick_params(axis="x", length=0)

		#if dataset_idx == len(datasets) - 1:
		
	fig.supxlabel(r"$1 - \rho$", fontsize=20, y=0.02)

	plt.tight_layout()
	plt.savefig("./ext_ratio_all.png", dpi=300)

def __plot_response_impact__(ax, thread_df, num_labels, label_list, linestyles, colormaps, smooth=False):
	"""Plot response impact of specific thread."""
	gt_label = thread_df["gt-label"].values[0]

	## Plot variation of predicted probability vs. number of responses
	x  = np.array(range(len(thread_df)))
	x_ = np.linspace(x.min(), x.max(), 500) if smooth else x
	
	ys = []
	for label_i in range(num_labels):
		y_i = thread_df["class_{}".format(label_i)].values

		if smooth:
			spl = make_interp_spline(x, y_i) #spl = PchipInterpolator(x, y_i)
			y_i = spl(x_)

		ax.plot(x_, y_i, linestyle=linestyles[label_i], linewidth=3, label=label_list[label_i], color=colormaps[label_i])
		
		if label_i == gt_label:
			y_gt = y_i

		ys.append(y_i)
	
	ax.fill_between(x_, y_gt, 0, facecolor=colormaps[gt_label], alpha=0.1)

	## Find intersection points of ground-truth class and other classes
	for label_i, y_i in enumerate(ys):
		
		if label_i == gt_label:
			continue

		pos_idx = np.argwhere((np.diff(np.sign(y_gt - y_i)) > 0).astype(float) * (y_gt > 0.1).astype(float)[:-1]).flatten()
		neg_idx = np.argwhere((np.diff(np.sign(y_gt - y_i)) < 0).astype(float) * (y_gt > 0.1).astype(float)[:-1]).flatten()
		ax.plot((x_[pos_idx] + x_[pos_idx + 1]) / 2, (y_gt[pos_idx] + y_i[pos_idx]) / 2, "o", color="r", markerfacecolor="none", markersize=12, markeredgewidth=3)
		#ax.plot((x_[pos_idx] + x_[pos_idx + 1]) / 2, (y_gt[pos_idx] + y_i[pos_idx]) / 2, "x", color="r", markerfacecolor="none", markersize=12, markeredgewidth=3)
		#ax.plot((x_[neg_idx] + x_[neg_idx + 1]) / 2, (y_gt[neg_idx] + y_i[neg_idx]) / 2, "x", color="r", markerfacecolor="none", markersize=12, markeredgewidth=3)
		ax.plot((x_[neg_idx] + x_[neg_idx + 1]) / 2, (y_gt[neg_idx] + y_i[neg_idx]) / 2, "o", color="r", markerfacecolor="none", markersize=12, markeredgewidth=3)
	
	if len(thread_df) > 20:
		xticks = np.arange(0, len(thread_df), 5)
	elif len(thread_df) > 15:
		xticks = np.arange(0, len(thread_df), 3)
	elif len(thread_df) > 10:
		xticks = np.arange(0, len(thread_df), 2)
	else:
		xticks = np.arange(0, len(thread_df), 1)
	ax.set_xticks(xticks)
	#ax.set_yticks(np.arange(0, 1.1, 0.2))
	ax.set_yticks([0, 1])
	#ax.set_ylim(0, 1.05)
	ax.set_ylim(0, 1.2)

def has_framing(thread):
	gt_label = thread["gt-label"].values[0]
				
	prev_pred = None
	for row_idx, row in thread.iterrows():
		hard_pred = row["hard_pred"]
	
		if prev_pred is None:
			prev_pred = hard_pred
			continue
		
		## Framing exists
		if (hard_pred == gt_label or prev_pred == gt_label) and (prev_pred != hard_pred):
			return True
	
		prev_pred = hard_pred

	return False

def plot_response_impact(args):
	"""Plot `Number of Responses` vs. `Predicted Probability` for framing effect experiments."""

	def get_label(dataset_name):
		data_df = pd.read_csv("{}/{}/data.csv".format(args.dataset_root_V2, dataset_name))
	
		label_list = list(set(data_df["veracity"]))
		label_list.sort()
		num_labels = len(label_list)

		return label_list, num_labels
	
	#mode = "single-img"
	mode = "multi-img"
	smooth = True

	## Read data and set label name
	label_list, num_labels = get_label(args.dataset_name)

	## Settings of the plot
	linestyles = ["solid", "dashed", "dashdot", "dotted"]
	colormaps  = [(240, 146, 53), (0, 0, 245), (135, 25, 203), (128, 128, 128)]
	colormaps  = [(map_[0] / 255, map_[1] / 255, map_[2] / 255) for map_ in colormaps]

	if mode == "single-img":

		models = ["bi-tgn-bart", "transformer-bart"]
		
		for model in models:
			for fold in range(5):
				print("Plotting {:12s}, Fold [{}], {:20s}".format(args.dataset_name, fold, model))
		
				predict_dir = "{}/{}/exp/{}/{}".format(args.result_path, args.dataset_name, model, fold)
				figures_dir = "{}/figures".format(predict_dir)
				figures_dir = "{}_smooth".format(figures_dir) if smooth else figures_dir

				os.makedirs(figures_dir, exist_ok=True)
		
				## Read predictions
				predict_file = "{}/predictions.csv".format(predict_dir, fold)
				predict_df = pd.read_csv(predict_file)
		
				group_thread = predict_df.groupby("source_id")
		
				for src_id, thread in tqdm(group_thread):
		
					if has_framing(thread) and len(thread) >= 4:
						fig, ax = plt.subplots()
						__plot_response_impact__(ax, thread, num_labels, label_list, linestyles, colormaps, smooth=smooth)

						ax.set_xlabel("Number of Responses")
						ax.set_ylabel("Predicted Probability")

						plt.tight_layout()
						plt.legend(loc="upper right")
						
						if smooth:
							if figures_dir == ".":
								src_id += "_smooth"
						
						plt.savefig("{}/{}.png".format(figures_dir, src_id))
						plt.close()
			print()

	elif mode == "multi-img":

		filepath_config = [
			#{"dataset": "semeval2019", "model": "bi-tgn-bart", "fold": 0, "source_id": "524944399890124801"}, 
			#{"dataset": "semeval2019", "model": "bi-tgn-bart", "fold": 0, "source_id": "544294893146091520"}, 
			#{"dataset": "semeval2019", "model": "bi-tgn-bart", "fold": 0, "source_id": "764927075522260992"}
			{"dataset": "twitter15", "model": "bi-tgn-bart", "fold": 0, "source_id": "689922870689009664"}, ## Correct Prediction: Type 1
			{"dataset": "twitter15", "model": "bi-tgn-bart", "fold": 0, "source_id": "450341615979069440"}, ## Correct Prediction: Type 2
			#{"dataset": "twitter15", "model": "bi-tgn-bart", "fold": 0, "source_id": "543436472842334209"}, ## Correct Prediction: Type 3
			{"dataset": "twitter15", "model": "bi-tgn-bart", "fold": 0, "source_id": "501881262240694272"}, ## Wrong Prediction: Type 1
		]

		label_list, num_labels = get_label("twitter15")

		threads = []
		## Read and collect predictions dataframe
		for config in filepath_config:
			## Read predictions
			predict_dir = "{}/{}/exp/{}/{}".format(args.result_path, config["dataset"], config["model"], config["fold"])
			predict_file = "{}/predictions.csv".format(predict_dir)
			predict_df = pd.read_csv(predict_file)
			predict_df["source_id"] = predict_df["source_id"].astype(str)

			thread = predict_df.loc[predict_df["source_id"] == config["source_id"]]
			threads.append(thread)

		fig, axes = plt.subplots(3, 1, figsize=(10, 4.5), sharex=True, sharey=True)

		for th_idx, thread in enumerate(threads):
			__plot_response_impact__(axes[th_idx], thread, num_labels, label_list, linestyles, colormaps, smooth=smooth)

		for ax in axes:
			ax.tick_params(bottom=False, labelsize=15)

		## Vertical settings
		xlabel_idx = -1
		ylabel_idx = int(len(axes) / 2)
		axes[xlabel_idx].tick_params(bottom=True)
		axes[0].legend(loc="lower right", ncols=2, fontsize=16)
		axes[xlabel_idx].set_xlabel("Number of Responses"  , fontsize=18)
		axes[ylabel_idx].set_ylabel("Predicted Probability", fontsize=18)
		axes[-1].set_xlim([-1, 31])

		## Plot legend of markers
		#k_cir = mlines.Line2D([], [], color="k", marker="o", linestyle="none", markerfacecolor="none", markeredgewidth=3, markersize=12, label="Positive CR (12.85%)")
		#r_cro = mlines.Line2D([], [], color="r", marker="x", linestyle="none", markerfacecolor="none", markeredgewidth=3, markersize=12, label="Negative CR (6.05%)")
		#axes[1].legend(loc="upper center", handles=[k_cir, r_cro], bbox_to_anchor=(0.5, 2.5), ncols=3, fontsize=18, frameon=False)

		k_cir = mlines.Line2D([], [], color="r", marker="o", linestyle="none", markerfacecolor="none", markeredgewidth=3, markersize=12, label="Critical Responses (18.9%)")
		#r_cro = mlines.Line2D([], [], color="r", marker="x", linestyle="none", markerfacecolor="none", markeredgewidth=3, markersize=12, label="Critical Responses (18.9%)")
		axes[1].legend(loc="upper center", handles=[k_cir], bbox_to_anchor=(0.5, 2.5), ncols=3, fontsize=18, frameon=False)

		plt.tight_layout()
		plt.subplots_adjust(hspace=0.05)

		plt.savefig("{}/critical_resp_t15.png".format(args.dataset_root_V2), bbox_inches="tight", dpi=300)
		
if __name__ == "__main__":
	args = parse_args()

	if args.plot_extract_ratio:
		plot_extract_ratio(args)
	elif args.plot_all_extract_ratio:
		plot_all_extract_ratio(args)
	elif args.plot_response_impact:
		plot_response_impact(args)