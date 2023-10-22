import os
import csv
import ipdb
import json
import openai
import shutil
import signal
import backoff
import tiktoken
import argparse
import numpy as np
import pandas as pd
import preprocessor as pre ## TweetPreprocessor

from tqdm import tqdm

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def parse_args():
	parser = argparse.ArgumentParser(description="Generate summary by ChatGPT")

	## Which experiment
	parser.add_argument("--generate_summary", action="store_true")

	## Others
	parser.add_argument("--dataset_name", type=str, default="semeval2019", choices=["semeval2019", "twitter15", "twitter16"])
	parser.add_argument("--dataset_root", type=str, default="../dataset/processedV2")
	parser.add_argument("--results_root", type=str, default="/mnt/1T/projects/RumorV2/results")
	parser.add_argument("--fold", type=str, default="0,1,2,3,4", help="either use 5-fold data or train/dev/test from rumoureval2019 competition")

	parser.add_argument("--timeout_sec", type=int, default=10)

	args = parser.parse_args()

	return args

def _request_timeout(signum, frame):
	raise TimeoutError

def Twitter_or_Reddit(src_id):
	if isinstance(src_id, int) or src_id.isnumeric():
		return "Twitter"
	else:
		return "Reddit"

def clean_text(text):
	"""Remove @'s and url's"""
	pre.set_options(pre.OPT.URL, pre.OPT.MENTION)
	return pre.tokenize(text).replace("$MENTION$", "").replace("$URL$", "")

def truncate_text(text, max_tokens=4000):
	return encoding.decode(encoding.encode(text)[:max_tokens])

def concat_responses(responses):
	res_str = ""
	for ridx, response in enumerate(responses):
		res_str = "{}{}. {}\n".format(res_str, ridx + 1, clean_text(response))
	return res_str

def load_data(args):
	print("\nLoad dataset: [{}]".format(args.dataset_name))
	data_df = pd.read_csv("{}/{}/data.csv".format(args.dataset_root, args.dataset_name))
	group_src = data_df.groupby("source_id")

	prompt = "Summarize the conversation thread from Twitter below, with a maximum of 32 tokens.\n\n"

	## Read 5-Fold
	inputs_folds, src_ids_folds = [], []
	for fold_i in args.fold.split(","):
		path_fold = "{}/{}/split_{}".format(args.dataset_root, args.dataset_name, fold_i)
		train_df = pd.read_csv("{}/train.csv".format(path_fold))
		test_df  = pd.read_csv("{}/test.csv".format(path_fold))

		train_ids = train_df["source_id"].tolist()
		test_ids  = test_df["source_id"].tolist()

		inputs, src_ids = [], []
		for id_ in test_ids:
			thread_df = group_src.get_group(id_)
			responses = thread_df.iloc[1:]
			responses = responses["text"].tolist()
			
			## Formulate input string
			res_str = concat_responses(responses)
			inp_str = "{}{}".format(prompt.replace("Twitter", Twitter_or_Reddit(id_)), res_str)
			inp_str = truncate_text(inp_str)
			
			## Collect source ids & inputs
			src_ids.append(id_)
			inputs.append(inp_str)
		
		## Collect data for each fold
		src_ids_folds.append(src_ids)
		inputs_folds.append(inputs)
		
	return data_df, src_ids_folds, inputs_folds

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def request_chatgpt(input_, timeout_sec):
	while True:
		try:
			signal.signal(signal.SIGALRM, _request_timeout)
			signal.alarm(timeout_sec) ## Start timing, if more than 10 seconds then raise TimeoutError
			response = openai.ChatCompletion.create(
				model="gpt-3.5-turbo",
				max_tokens=32, 
				messages=[
					{"role": "system", "content": "You are a chatbot"}, 
					{"role": "user"  , "content": input_}
				]
			)
			signal.alarm(0)
			break
		except TimeoutError:
			print("Request time out, retry!")
	return response

def main(args):
	output_root = "{}/{}/chatgpt".format(args.results_root, args.dataset_name)
	os.makedirs(output_root, exist_ok=True)

	## Set up API key
	key = open("openai_key.txt", "r").readline()
	openai.api_key = key

	## Load data
	data_df, src_ids_folds, inputs_folds = load_data(args)

	char = input("\nAre you sure you want to execute this program? Your ChatGPT API credit would be consumed!\nType `c` to continue and other character to quit: ")
	if char != "c":
		print("Program terminated!")
		return

	## Request ChatGPT
	print("\nRequest ChatGPT...")
	start_fold = int(input("Please enter start fold: "))
	assert start_fold <= 5 and start_fold >= 0, "Invalid fold name! Should be one of 0, 1, 2, 3, 4."

	for fold_i, inputs in enumerate(inputs_folds):
		if fold_i < start_fold:
			print("Skip fold [{}]".format(fold_i))
			continue

		src_ids  = src_ids_folds[fold_i]
		fold_dir = "{}/{}".format(output_root, fold_i)
		os.makedirs(fold_dir, exist_ok=True)
		
		start_idx = 0
		#start_idx = int(input("Please enter start index for fold [{}]: ".format(fold_i)))
		assert start_idx < len(inputs) and start_idx >= 0, "Invalid index!"
		
		with open("{}/summary.csv".format(fold_dir), "a") as f_csv:
			writer = csv.writer(f_csv)
			writer.writerow(["source_id", "summary"])

			for input_idx, input_ in enumerate(tqdm(inputs[start_idx:], desc="Fold [{}]".format(fold_i))):
				src_id = src_ids[start_idx + input_idx]
				
				while True:
					try:
						response = request_chatgpt(input_, timeout_sec=args.timeout_sec)
						break
					except:
						print("Unexpected error occur, retry!")

				summary = response.choices[0].message.content
				writer.writerow([src_id, summary])

def read_generated_summaries(args):
	fold = 0
	path_in = "{}/{}/chatgpt/{}/summary.csv".format(args.results_root, args.dataset_name, fold)
	
	summary_df = pd.read_csv(path_in)
	ipdb.set_trace()

def find_index_of_id(data_list, fold, src_id):
	index = data_list[fold].index(src_id)
	print("Index of {}: {}".format(src_id, index))
	ipdb.set_trace()

if __name__ == "__main__":
	args = parse_args()
	main(args)