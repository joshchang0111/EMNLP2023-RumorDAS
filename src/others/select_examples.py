import os
import ipdb
import json
import shutil
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

def parse_args():
	parser = argparse.ArgumentParser(description="Analyze experiment results")

	## Which experiment
	parser.add_argument("--select_for_user_study", action="store_true")
	parser.add_argument("--generate_for_google_forms", action="store_true")
	parser.add_argument("--merge_en_zh", action="store_true")
	parser.add_argument("--distribute_samples_for_diff_tasks", action="store_true")
	parser.add_argument("--select_which_to_swap", action="store_true")

	## Others
	parser.add_argument("--dataset_name", type=str, default="twitter15", choices=["semeval2019", "twitter15", "twitter16"])
	parser.add_argument("--dataset_root", type=str, default="../dataset/processed")
	parser.add_argument("--dataset_root_V2", type=str, default="../dataset/processedV2")
	parser.add_argument("--fold", type=str, default="0,1,2,3,4", help="either use 5-fold data or train/dev/test from rumoureval2019 competition")
	parser.add_argument("--result_path", type=str, default="/mnt/1T/projects/RumorV2/results")

	args = parser.parse_args()

	return args

def select_for_user_study(args):
	print("Select examples for user study...")
	print("Dataset: {}".format(args.dataset_name))

	## Read input data
	path_data = "{}/{}/data.csv".format(args.dataset_root_V2, args.dataset_name)
	data_df = pd.read_csv(path_data)
	data_group = data_df.groupby("source_id")

	## Set output path
	path_out = "{}/{}/user_study/for_selection".format(args.result_path, args.dataset_name)
	#if os.path.isdir(path_out):
	#	#shutil.rmtree(path_out)
	#	#print("Path out: {} exists, remove it before writing...".format(path_out))
	#	raise ValueError("Path out: {} exists!!! Please remove it or change the target directory!".format(path_out))
	os.makedirs(path_out, exist_ok=True)
	os.makedirs("{}".format(path_out), exist_ok=True)
	for label in data_df["veracity"].unique():
		os.makedirs("{}/{}".format(path_out, label), exist_ok=True)

	folds = args.fold.split(",")
	for fold in folds:
		#path_sum_loo = "{}/{}/ssra_loo/{}/summary-3g.csv".format(args.result_path, args.dataset_name, fold)
		#path_sum_loo = "{}/{}/ssra_loo/{}/summary-3g-10.csv".format(args.result_path, args.dataset_name, fold)
		path_sum_loo = "{}/{}/ssra_loo/{}/summary-3g-10-128.csv".format(args.result_path, args.dataset_name, fold)
		#path_summary = "{}/{}/ssra_kmeans_3/{}/summary.csv".format(args.result_path, args.dataset_name, fold)
		#path_summary = "{}/{}/ssra_kmeans_3/{}/summary-10.csv".format(args.result_path, args.dataset_name, fold)
		path_summary = "{}/{}/ssra_kmeans_3/{}/summary-10-128.csv".format(args.result_path, args.dataset_name, fold)
		path_cluster = "{}/{}/split_{}/cluster_summary/train/kmeans-3.csv".format(args.dataset_root_V2, args.dataset_name, fold)
		
		sum_loo_df = pd.read_csv(path_sum_loo)
		sum_loo_df["cluster_id"] = sum_loo_df["source_id"].apply(lambda x: x.split("_")[1])
		sum_loo_df["source_id"] = sum_loo_df["source_id"].apply(lambda x: x.split("_")[0])
		sum_loo_group = sum_loo_df.groupby("source_id")

		summary_df = pd.read_csv(path_summary)
		summary_group = summary_df.groupby("source_id")

		cluster_df = pd.read_csv(path_cluster)
		cluster_group = cluster_df.groupby("source_id")

		for src_id, summaries in tqdm(summary_group, desc="Fold [{}]".format(fold)):
			thread = data_group.get_group(src_id)
			response = thread.iloc[1:]
			
			cluster = cluster_group.get_group(src_id)
			medoids = cluster.loc[cluster["is_centroid"] == 1]
			medoids = medoids.copy()

			## Remove more than one medoids
			medoids["subcluster_id"] = medoids["cluster_id"].apply(lambda x: x.split("_")[-1])
			medoids.loc[:, "cluster_id"] = medoids.cluster_id.apply(lambda x: x.split("_")[0])
			
			medoids_group = medoids.groupby("cluster_id")
			medoids = []
			for cid, medoid in medoids_group:
				medoids.append(medoid.sample(n=1))
			medoids = pd.concat(medoids)

			tids = []
			for tid in response["tweet_id"]:
				cid = cluster.loc[cluster["tweet_id"] == tid]["cluster_id"].iloc[0]
				cen = cluster.loc[cluster["tweet_id"] == tid]["is_centroid"].iloc[0]
				#ipdb.set_trace()
				if cen == 1:
					tids.append("*{},{}".format(cid, tid))
				elif cen == 0:
					tids.append("{},{}".format(cid, tid))
				#tids.append("{},{}".format(cid, tid))
				#tids.append(tid)

			obj = {
				"source id"  : src_id, 
				"source post": thread.iloc[0]["text"], 
				"label": thread["veracity"].tolist()[0], 
				"responses"  : dict(zip(tids, response["text"])), 
				#"summary_loo": sum_loo_group.get_group(src_id)["summary"].iloc[0], ## Baseline
				"summary_loo": dict(zip(sum_loo_group.get_group(str(src_id))["cluster_id"], sum_loo_group.get_group(str(src_id))["summary"])), 
				#"summary_ext": dict(zip(medoids["cluster_id"], medoids["tweet_id"])), 
				"summary_abs": dict(zip(summaries["cluster_id"], summaries["summary"]))
			}
			obj_json = json.dumps(obj, indent=4, ensure_ascii=False)
			with open("{}/{}/{}.json".format(path_out, thread["veracity"].tolist()[0], src_id), "w") as fw:
				fw.write(obj_json)

def generate_for_google_forms(args):
	"""Output a string that can be directly copied and pasted to google forms."""

	def add_ext(output_str, sample_dict, summ_idx):
		## Summary (Extractive-3)
		output_str = "{}\nSummary {}\n".format(output_str, summ_idx)
		ext_summs = []
		for cluster_idx, text in sample_dict["summary_ext"].items():
			ext_summs.append(sample_dict["responses"][text])
		output_str = "{}{}\n".format(output_str, "\n".join(ext_summs))
		return output_str

	def add_abs_loo(output_str, sample_dict, summ_idx):
		## Summary (SSRA-LOO)
		output_str = "{}\nSummary {}\n".format(output_str, summ_idx)
		loo_summs = []
		for cluster_idx, text in sample_dict["summary_loo"].items():
			loo_summs.append(text)
		output_str = "{}{}\n".format(output_str, "\n".join(loo_summs))
		return output_str
	
	def add_abs_kmeans(output_str, sample_dict, summ_idx):
		## Summary (SSRA-KMeans-3)
		output_str = "{}\nSummary {}\n".format(output_str, summ_idx)
		abs_summs = []
		for cluster_idx, text in sample_dict["summary_abs"].items():
			abs_summs.append(text)
		output_str = "{}{}\n".format(output_str, "\n".join(abs_summs))
		return output_str

	file_configs = [
		"semeval2019,true,524969201102901248", ## NEW
		"semeval2019,true,544319274072817664", 
		"semeval2019,true,544319832486064128", 
		"semeval2019,true,544515538383564801", 
		"semeval2019,true,553508098825261056", 
		"semeval2019,true,500378223977721856", ## B
		"semeval2019,true,784118929799073793", ## B
		"semeval2019,true,987801763850809344", ## B
		"semeval2019,false,553197863971610624", 
		"semeval2019,false,580339547269144576", 
		"semeval2019,false,763098277986209792", ## NEW
		"semeval2019,false,872367878871408640", 
		"semeval2019,false,904111594912841729", ## NEW
		"semeval2019,false,544271069146656768", ## B
		"semeval2019,false,580340476949086208", ## B
		"twitter15,true,407181813174788096", ## Pual Walker
		"twitter15,true,504109775358287872", ##
		"twitter15,true,509466295344304129", ## Microsoft buying minecraft
		"twitter15,true,510916647373516800", ## ISIS video
		"twitter15,true,567596847330373632", ## 
		"twitter15,true,538374385665449985", ## B
		"twitter15,true,536558567206420480", ## B
		"twitter15,true,532197572698320896", ## B
		"twitter15,false,351767344097398785", 
		"twitter15,false,387353560356118528", 
		"twitter15,false,489872500462190593", 
		"twitter15,false,497073936077959169", ## Justin Bieber, Bear attack
		"twitter15,false,563123548857053184", 
		"twitter15,false,357300409292959744", ## B
		"twitter15,false,511918957255991296", ## B
		"twitter16,true,544380742076088320", 
		"twitter16,true,614594259900080128", 
		"twitter16,true,623599854661541888", 
		"twitter16,true,641082932740947972", 
		"twitter16,true,650128194209730561", 
		"twitter16,false,613393657161510913", 
		"twitter16,false,620916279608651776", ## NASA
		"twitter16,false,662151653462790144", 
		"twitter16,false,664000310856310784", 
		"twitter16,false,672632863452299264", 
	]

	datasets = ["semeval2019", "twitter15", "twitter16"]
	for dataset_name in datasets:
		path_out = "{}/{}/user_study/google_forms/en".format(args.result_path, dataset_name)
		if os.path.isdir(path_out):
			#shutil.rmtree(path_out)
			#print("Path out: {} exists, removing it before writing...".format(path_out))
			raise ValueError("Path out: {} exists!!! Please remove it or change the target directory!".format(path_out))
		os.makedirs(path_out, exist_ok=True)

	for config in file_configs:
		config = config.split(",")

		path_in  = "{}/{}/user_study/for_selection/{}/{}.json".format(args.result_path, config[0], config[1], config[2])
		path_out = "{}/{}/user_study/google_forms/en".format(args.result_path, config[0])
		print(path_in)
		sample_dict = json.load(open(path_in))

		## Source Post & Responses
		rid2ridx = {}
		output_str = "{}\n{}\n\nSource Post\n{}\n\nResponses\n".format(sample_dict["source id"], sample_dict["label"], sample_dict["source post"])
		for ridx, (key, text) in enumerate(sample_dict["responses"].items()):
			output_str = "{}{}\n".format(output_str, text)
			rid2ridx[key] = ridx + 1
		
		#output_str = add_ext(output_str, sample_dict, summ_idx=1)
		output_str = add_abs_loo(output_str, sample_dict, summ_idx=1)
		output_str = add_abs_kmeans(output_str, sample_dict, summ_idx=2)
		
		with open("{}/{}.txt".format(path_out, sample_dict["source id"]), "w") as fw:
			fw.write(output_str)

def merge_en_zh(args):
	def filter_hidden_files_dirs(files):
		files = [file for file in files if not file.startswith(".")]
		files.sort()
		return files

	def merge_2_lines(filename, lines_en, lines_zh):
		assert len(lines_en) == len(lines_zh), "File {} has different content between en & zh versions!".format(filename)

		lines_en_zh = []
		for line_idx in range(len(lines_en)):
			en_line = lines_en[line_idx].strip().rstrip()
			zh_line = lines_zh[line_idx].strip().rstrip()

			if en_line == "Source Post" or \
			   en_line == "Responses" or \
			   en_line == "Summary 1" or \
			   en_line == "Summary 2":
				zh_line = "({})".format(zh_line)

			if en_line == zh_line:
				new_str = "{}\n".format(en_line)
			else:
				new_str = "{} {}\n".format(en_line, zh_line)
			
			lines_en_zh.append(new_str)

		return lines_en_zh

	datasets = ["semeval2019", "twitter15", "twitter16"]
	for dataset_name in datasets:
		path_en = "{}/{}/user_study/google_forms/en".format(args.result_path, dataset_name)
		path_zh = "{}/{}/user_study/google_forms/zh".format(args.result_path, dataset_name)

		path_out = "{}/{}/user_study/google_forms/en+zh".format(args.result_path, dataset_name)
		os.makedirs(path_out, exist_ok=True)

		files_en = filter_hidden_files_dirs(os.listdir(path_en))
		files_zh = filter_hidden_files_dirs(os.listdir(path_zh))

		## Check if two dirs contain same files
		assert len(files_en) == len(files_zh)
		for file_idx in range(len(files_en)):
			if files_en[file_idx] != files_zh[file_idx]:
				raise ValueError("English files and Chinese files are not the same!")
		
		## Merge files
		for file_idx in range(len(files_en)):
			print("{}/{}".format(path_en, files_en[file_idx]))
			lines_en = open("{}/{}".format(path_en, files_en[file_idx])).readlines()
			lines_zh = open("{}/{}".format(path_zh, files_zh[file_idx])).readlines()

			lines_en_zh = merge_2_lines(files_en[file_idx], lines_en, lines_zh)
			
			open("{}/{}".format(path_out, files_en[file_idx]), "w").write("".join(lines_en_zh))

def distribute_samples_for_diff_tasks(args):

	def copy_files_of_task(task, ids):
		print("Task {}: {} samples".format(task, len(ids)))
		os.makedirs("{}/{}".format(path_out, task), exist_ok=True)
		for idx, id_ in enumerate(ids):
			id_ = id_.split(",")
			shutil.copyfile(
				"{}/{}/user_study/google_forms/en+zh/{}.txt".format(args.result_path, id_[0], id_[-1]), 
				"{}/{}/{:02d}-{}.txt".format(path_out, task, idx, id_[-1])
			)
	
	def copy_task_B(subtask_id, mode, ids, start_idx=0):
		assert mode == "responses" or mode == "summary", "Mode not correctly specified!"

		os.makedirs("{}/B.{}".format(path_out, subtask_id), exist_ok=True)
		for idx, id_ in enumerate(ids, start_idx):
			id_ = id_.split(",")
			lines = open("{}/{}/user_study/google_forms/en+zh/{}.txt".format(args.result_path, id_[0], id_[-1]))
			f_out = open("{}/B.{}/{:02d}-{}.txt".format(path_out, subtask_id, idx, id_[-1]), "w")
			
			write_flag = True
			for line in lines:
				if mode == "responses" and "Summary 1" in line:
					break
				if mode == "summary":
					if "Responses (回覆)" in line: write_flag = False
					if "Summary 2" in line: write_flag = True
				if write_flag:
					f_out.write(line)
			f_out.close()

	path_out = "{}/user_study".format(args.dataset_root_V2)
	#if os.path.isdir(path_out):
	#	shutil.rmtree(path_out)
	
	task_2_ids = {
		"A": [
			"semeval2019,true,524969201102901248", ## NEW
			"semeval2019,true,544319274072817664", 
			"semeval2019,true,544319832486064128", 
			"semeval2019,true,544515538383564801", 
			"semeval2019,true,553508098825261056", 
			"semeval2019,false,553197863971610624", 
			"semeval2019,false,580339547269144576", 
			"semeval2019,false,763098277986209792", ## NEW
			"semeval2019,false,872367878871408640", 
			"semeval2019,false,904111594912841729", ## NEW
			"twitter15,true,407181813174788096", ## Pual Walker
			"twitter15,true,504109775358287872", ##
			"twitter15,true,509466295344304129", ## Microsoft buying minecraft
			"twitter15,true,510916647373516800", ## ISIS video
			"twitter15,true,567596847330373632", ## 
			"twitter15,false,351767344097398785", 
			"twitter15,false,387353560356118528", 
			"twitter15,false,489872500462190593", 
			"twitter15,false,497073936077959169", ## Justin Bieber, Bear attack
			"twitter15,false,563123548857053184", 
			"twitter16,true,544380742076088320", 
			"twitter16,true,614594259900080128", 
			"twitter16,true,623599854661541888", 
			"twitter16,true,641082932740947972", 
			"twitter16,true,650128194209730561", 
			"twitter16,false,613393657161510913", 
			"twitter16,false,620916279608651776", ## NASA
			"twitter16,false,662151653462790144", 
			"twitter16,false,664000310856310784", 
			"twitter16,false,672632863452299264", 
		], 
		"B": [
			"semeval2019,true,500378223977721856", 
			"semeval2019,true,784118929799073793", 
			"semeval2019,true,987801763850809344", 
			"semeval2019,false,544271069146656768", 
			"semeval2019,false,580340476949086208", 
			"semeval2019,true,544319274072817664",
			"semeval2019,true,553508098825261056",
			"semeval2019,false,580339547269144576",
			"semeval2019,false,904111594912841729",
			"semeval2019,false,872367878871408640",
			## 544319274072817664 ## V T
			## 524969201102901248 ## X T -> 553508098825261056
			## 580339547269144576 ## = F
			## 553197863971610624 ## = F -> 904111594912841729
			## 763098277986209792 ## = F -> 872367878871408640
			"twitter15,true,538374385665449985", 
			"twitter15,true,536558567206420480", 
			"twitter15,true,532197572698320896", 
			"twitter15,false,357300409292959744", 
			"twitter15,false,511918957255991296", 
			"twitter15,true,567596847330373632",
			"twitter15,true,509466295344304129",
			"twitter15,false,387353560356118528",
			"twitter15,false,563123548857053184",
			"twitter15,false,489872500462190593", 
			## 567596847330373632 ## V T
			## 509466295344304129 ## V T
			## 387353560356118528 ## V F
			## 497073936077959169 ## = F -> 563123548857053184
			## 351767344097398785 ## X F -> 489872500462190593
		]
	}

	#np.random.shuffle(task_2_ids["A"])
	#task_2_ids["A.1"] = task_2_ids["A"][:15]
	#task_2_ids["A.2"] = task_2_ids["A"][15:]

	#t15_t_ids, t15_f_ids, re2019_t_ids, re2019_f_ids = [], [], [], []
	#for id_A in task_2_ids["A"]:
	#	if "twitter15" in id_A:
	#		if "true"  in id_A: t15_t_ids.append(id_A)
	#		if "false" in id_A: t15_f_ids.append(id_A)
	#	elif "semeval2019" in id_A:
	#		if "true"  in id_A: re2019_t_ids.append(id_A)
	#		if "false" in id_A: re2019_f_ids.append(id_A)

	#np.random.shuffle(re2019_t_ids)
	#np.random.shuffle(re2019_f_ids)
	#np.random.shuffle(t15_t_ids)
	#np.random.shuffle(t15_f_ids)

	#task_2_ids["B"].extend(re2019_t_ids[:2])
	#task_2_ids["B"].extend(re2019_f_ids[:3])
	#task_2_ids["B"].extend(t15_t_ids[:2])
	#task_2_ids["B"].extend(t15_f_ids[:3])

	#copy_files_of_task(task="A.1", ids=task_2_ids["A.1"])
	#copy_files_of_task(task="A.2", ids=task_2_ids["A.2"])

	print("Task B  : {} samples".format(len(task_2_ids["B"])))

	np.random.shuffle(task_2_ids["B"])

	copy_task_B(subtask_id="1", mode="responses", ids=task_2_ids["B"][:10])
	copy_task_B(subtask_id="1", mode="summary"  , ids=task_2_ids["B"][10:], start_idx=10)

	copy_task_B(subtask_id="2", mode="summary"  , ids=task_2_ids["B"][:10])
	copy_task_B(subtask_id="2", mode="responses", ids=task_2_ids["B"][10:], start_idx=10)

def select_which_to_swap(args):
	np.random.seed(123)

	a1_orders = np.random.randint(2, size=15)
	a2_orders = np.random.randint(2, size=15)

	print("A.1 whether to swap: {}".format(list(a1_orders)))
	print("A.2 whether to swap: {}".format(list(a2_orders)))

	def create_map_df(orders):
		summary_1, summary_2 = [], []
		for idx in range(orders.shape[0]):
			swap_ = orders[idx]

			if swap_ == 0:
				summary_1.append("LOO")
				summary_2.append("kmeans")
			elif swap_ == 1:
				summary_1.append("kmeans")
				summary_2.append("LOO")
	
		map = {
			"summary 1": summary_1, 
			"summary 2": summary_2
		}
		map_df = pd.DataFrame(map)
		map_df.index = map_df.index + 1
		return map_df
	
	map_df = create_map_df(a1_orders)
	map_df.to_csv("{}/user_study/A_1_map.csv".format(args.dataset_root_V2))

	map_df = create_map_df(a2_orders)
	map_df.to_csv("{}/user_study/A_2_map.csv".format(args.dataset_root_V2))

if __name__ == "__main__":
	args = parse_args()

	if args.select_for_user_study:
		select_for_user_study(args)
	elif args.generate_for_google_forms:
		generate_for_google_forms(args)
	elif args.merge_en_zh:
		merge_en_zh(args)
	elif args.distribute_samples_for_diff_tasks:
		distribute_samples_for_diff_tasks(args)
	elif args.select_which_to_swap:
		select_which_to_swap(args)