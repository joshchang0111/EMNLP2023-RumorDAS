import os
import csv
import ipdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn.functional as F

## Self-defined
from .utils import (
	calculate_asr, 
	write_result, 
	write_generated_responses, 
	write_response_summary, 
	write_cls_predictions, 
	calculate_PHEME_results
)
from .plot import __plot_response_impact__

#################################
## Processes of train_detector ##
#################################
def train_process(data_args, model_args, training_args, trainer, train_dataset, last_checkpoint):
	checkpoint = None
	if training_args.resume_from_checkpoint is not None:
		checkpoint = training_args.resume_from_checkpoint
	elif last_checkpoint is not None:
		checkpoint = last_checkpoint
	train_result = trainer.train(resume_from_checkpoint=checkpoint)
	metrics = train_result.metrics
	max_train_samples = (
		data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
	)
	metrics["train_samples"] = min(max_train_samples, len(train_dataset))

	trainer.log_metrics("train", metrics)

	write_result(trainer.best_metrics, data_args, model_args, training_args, mode="train")#, summarizer=trainer.model.summarizer)

def eval_process(logger, data_args, model_args, training_args, trainer, eval_dataset):
	"""
	Evaluate the best model after whole training process.
	"""
	logger.info("*** Evaluate ***")

	metrics = trainer.evaluate(eval_dataset=eval_dataset)

	max_eval_samples = (
		data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
	)
	metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

	write_result(metrics, data_args, model_args, training_args, mode="evaluate")#, summarizer=trainer.model.summarizer)

def predict_process(logger, data_args, model_args, training_args, trainer, predict_dataset):
	logger.info("*** Predict ***")
	
	#predict_dataset = predict_dataset.remove_columns("label")
	results = trainer.predict(predict_dataset, metric_key_prefix="predict")

	## Get rumor detection predictions
	if trainer.model.__class__.__name__ == "BartForRumorDetectionAndResponseGeneration":
		predictions = results.predictions[0]
	else:
		predictions = results.predictions

	predictions = {
		"soft": F.softmax(torch.Tensor(predictions)), 
		"hard": np.argmax(predictions, axis=1)
	}

	output_file = "{}/predictions.csv".format(training_args.output_dir)
	output_dict = {
		"source_id": predict_dataset["source_id"], 
		"hard_pred": predictions["hard"].tolist(), 
		"gt_label": predict_dataset["labels_det"]
	}

	## Add soft prediction of each class to output_dict
	for label_i in range(data_args.num_labels):
		output_dict["class_{}".format(label_i)] = predictions["soft"][:, label_i].tolist()

	output_df = pd.DataFrame(data=output_dict)
	output_df.to_csv(output_file, index=False)
	
	return output_df

def explain_process(logger, data_args, model_args, training_args, trainer, predict_dataset):
	logger.info("*** Evaluate detection results w/ and w/o summarizer. ***")


def framing_process(logger, data_args, model_args, training_args, trainer, predict_dataset, use_cached_predictions=False):
	logger.info("*** Make predictions for framing effect experiment. ***")

	framing_type = 1
	use_cached_predictions = True

	## Obtain predictions
	predict_file = "{}/predictions.csv".format(training_args.output_dir)
	if not os.path.isfile(predict_file):
		use_cached_predictions = False

	if use_cached_predictions:
		predict_df = pd.read_csv(predict_file)
	else:
		predict_df = predict_process(logger, data_args, model_args, training_args, trainer, predict_dataset)
	group_thread = predict_df.groupby("source_id")

	stats_resp = {"num_frame": 0, "pos_frame": 0, "neg_frame": 0}
	stats_post = {"all_frame": [], "pos_frame": [], "neg_frame": []}

	## Analyze the impact of each response in each thread
	if framing_type == 1:
		pred_diff_base = 0.5

		## 1. Responses that change the model's hard prediction
		for src_id, thread in tqdm(group_thread):
			
			gt_label = thread["gt-label"].values[0]

			prev_pred_hard, prev_pred_soft = None, None
			for row_idx, row in thread.iterrows():
				hard_pred = row["hard_pred"]
				soft_pred = row["class_{}".format(gt_label)]
		
				if prev_pred_hard is None:
					prev_pred_hard = hard_pred
					prev_pred_soft = soft_pred
					continue

				pred_diff = abs(soft_pred - prev_pred_soft)
				
				if (hard_pred == gt_label or prev_pred_hard == gt_label) \
					and (prev_pred_hard != hard_pred) \
					and (pred_diff >= pred_diff_base):
					stats_resp["num_frame"] = stats_resp["num_frame"] + 1
					stats_post["all_frame"].append(src_id)
					if hard_pred == gt_label:
						stats_resp["pos_frame"] = stats_resp["pos_frame"] + 1
						stats_post["pos_frame"].append(src_id)
					elif prev_pred_hard == gt_label:
						stats_resp["neg_frame"] = stats_resp["neg_frame"] + 1
						stats_post["neg_frame"].append(src_id)
		
				prev_pred_hard = hard_pred
				prev_pred_soft = soft_pred

	elif framing_type == 2:
		## 2. Identify influential responses (changes the predicted probability considerably after certain response)
		## select threads with more than 10 nodes
		os.makedirs("{}/figures".format(training_args.output_dir), exist_ok=True)
		os.makedirs("{}/figures_smooth".format(training_args.output_dir), exist_ok=True)

		pred_diff_base = 0.2
		linestyles = ["dashed", "solid", "dashdot", "dotted", "densely dashdotdotted", "dashdotdotted"]
		colormaps  = [(240, 146, 53), (0, 0, 245), (135, 25, 203), (128, 128, 128)]
		colormaps  = [(map_[0] / 255, map_[1] / 255, map_[2] / 255) for map_ in colormaps]
		for src_id, thread in tqdm(group_thread):
			
			#if len(thread) < 10:
			#	continue
			
			gt_label = thread["gt-label"].values[0]

			## Iterate through all responses
			prev_pred, pred_diff, pred_delta, pred_diffs = None, 0, [], []
			for row_idx, row in thread.iterrows():
				soft_pred = row["class_{}".format(gt_label)]

				if prev_pred is None:
					pred_diff = 0
				else:
					pred_diff = soft_pred - prev_pred

				prev_pred = soft_pred
				pred_delta.append(pred_diff)
				pred_diffs.append(abs(pred_diff))

			pred_delta = np.array(pred_delta)
			pred_diffs = np.array(pred_diffs)
			if max(pred_diffs) > pred_diff_base:

				## Update statistics
				num_frame = (pred_diffs > pred_diff_base).sum()
				pos_frame = ((pred_diffs > pred_diff_base) & (pred_delta >= 0)).sum()
				neg_frame = ((pred_diffs > pred_diff_base) & (pred_delta <  0)).sum()

				stats_resp["num_frame"] = stats_resp["num_frame"] + num_frame
				stats_resp["pos_frame"] = stats_resp["pos_frame"] + pos_frame
				stats_resp["neg_frame"] = stats_resp["neg_frame"] + neg_frame
				
				stats_post["all_frame"].append(src_id)
				if pos_frame > 0: stats_post["pos_frame"].append(src_id)
				if neg_frame > 0: stats_post["neg_frame"].append(src_id)

	n_post = len(set(predict_df["source_id"]))
	n_resp = len(predict_df["source_id"]) - n_post
	with open(training_args.framing_stats_file, "a") as fw:
		fw.write("{:4s}\t{:4d}\t{:10.4f}\t{:10d}\t{:10d}\t{:10d}\t{:10d}\t{:10d}\t{:10d}\t{:10d}\t{:10d}\n".format(
			data_args.fold,
			framing_type, 
			pred_diff_base, 

			stats_resp["num_frame"], 
			stats_resp["pos_frame"], 
			stats_resp["neg_frame"], 
			n_resp, 

			len(set(stats_post["all_frame"])), 
			len(set(stats_post["pos_frame"])), 
			len(set(stats_post["neg_frame"])), 
			n_post
		))

	print()
	print("***** Framing Type {} *****".format(framing_type))
	print("==========================")
	print("==       RESPONSES      ==")
	print("==========================")
	print("=  # frame: {:5d}/{:5d}  =".format(stats_resp["num_frame"], n_resp))
	print("=  # pos. : {:5d}/{:5d}  =".format(stats_resp["pos_frame"], stats_resp["num_frame"]))
	print("=  # neg. : {:5d}/{:5d}  =".format(stats_resp["neg_frame"], stats_resp["num_frame"]))
	print("==========================")
	print("==========================")
	print("==         POSTS        ==")
	print("==========================")
	print("=  # frame: {:5d}/{:5d}  =".format(len(set(stats_post["all_frame"])), n_post))
	print("=  # pos. : {:5d}/{:5d}  =".format(len(set(stats_post["pos_frame"])), len(set(stats_post["all_frame"]))))
	print("=  # neg. : {:5d}/{:5d}  =".format(len(set(stats_post["neg_frame"])), len(set(stats_post["all_frame"]))))
	print("==========================")

	#ipdb.set_trace()

#######################################
## Processes of adversarial training ##
#######################################
def train_adv(data_args, model_args, training_args, trainer, train_dataset, last_checkpoint):
	if training_args.task_type == "train_adv_stage1":
		print("\nAdv. training stage [1]: train detector & generator (attacker)")
	elif training_args.task_type == "train_adv_stage2":
		print("\nAdv. training stage [2]: train generator (attacker) while fixing detector")

	checkpoint = None
	if training_args.resume_from_checkpoint is not None:
		checkpoint = training_args.resume_from_checkpoint
	elif last_checkpoint is not None:
		checkpoint = last_checkpoint
	train_result = trainer.train(resume_from_checkpoint=checkpoint)
	#trainer.save_model() ## Saves the tokenizer too for easy upload

	metrics = train_result.metrics
	max_train_samples = (
		data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
	)
	metrics["train_samples"] = min(max_train_samples, len(train_dataset))

	trainer.log_metrics("train", metrics)

	write_result(trainer.best_metrics, data_args, model_args, training_args, mode="train")

def eval_adv_detector(logger, data_args, model_args, training_args, trainer, eval_dataset, log_flag=False, new_line=True):
	print("Evaluate the detector trained with generator (attacker)")
	logger.info("*** Evaluate ***")

	results = trainer.predict(eval_dataset, metric_key_prefix="eval")
	predictions, label_ids, metrics = results.predictions, results.label_ids, results.metrics

	max_eval_samples = (
		data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
	)
	metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

	trainer.log_metrics("eval", metrics)
	#trainer.save_metrics("eval", metrics)

	## Write results to `overall_results.txt`
	if log_flag:
		new_line_ = False if (model_args.filter_ratio is not None) or (model_args.extract_ratio is not None) else new_line
		new_line  = new_line and new_line_
		write_result(
			metrics, 
			data_args, 
			model_args, 
			training_args, 
			mode="evaluate", 
			gen_flag=trainer.model.gen_flag, 
			sum_flag=trainer.model.sum_flag, 
			new_line=new_line
		)

	## Output classification results for PHEME at stage-1 evaluation
	if data_args.dataset_name == "PHEME" and training_args.task_type == "train_adv_stage1":
		write_cls_predictions(data_args, training_args, predictions, label_ids)

		## If evaluating on the last fold, calculate the overall results
		if int(data_args.fold) == 8:
			calculate_PHEME_results(data_args, training_args)

	return predictions, label_ids

def eval_adv_asr(logger, data_args, model_args, training_args, tokenizer, trainer, eval_dataset):
	"""Evaluate attack success rate"""

	print("\nEvaluate detector `without adversarial attack`...")
	trainer.model.gen_flag = False
	clean_predictions, clean_label_ids = eval_adv_detector(logger, data_args, model_args, training_args, trainer, eval_dataset, log_flag=True)

	print("\nEvaluate detector `under adversarial attack`...")
	trainer.model.gen_flag = True
	dirty_predictions, dirty_label_ids = eval_adv_detector(logger, data_args, model_args, training_args, trainer, eval_dataset, log_flag=True, new_line=False)

	## Evaluate attack success rate (ASR)
	clean_pred_det = np.argmax(clean_predictions[0], axis=1)
	dirty_pred_det = np.argmax(dirty_predictions[0], axis=1)
	labels_det = clean_label_ids[0]

	ori_correct = (clean_pred_det == labels_det)
	as_vec = ori_correct & (dirty_pred_det != labels_det)
	#asr = sum(as_vec) / len(as_vec) ## correct -> wrong
	asr = as_vec.sum() / ori_correct.sum()
	print("\n********** ASR (Attack Success Rate) = {:.4f} **********".format(asr))
	with open(training_args.overall_results_path, "a") as fw:
		fw.write("\t{:.4f}\n".format(asr))
	#write_generated_responses(training_args, trainer, tokenizer, eval_dataset, labels_det, clean_pred_det, dirty_pred_det, dirty_predictions, as_vec)

def eval_adv_asr_with_summ(logger, data_args, model_args, training_args, tokenizer, trainer, eval_dataset):
	"""
	Evaluate attack success rate with summarizer.
	Also evaluate summarizer defense success rate.
	"""
	extract_ratio = model_args.filter_ratio if model_args.filter_ratio is not None else model_args.extract_ratio

	print("\nEvaluate detector `without adversarial attack`...")
	trainer.model.gen_flag = False
	clean_predictions, label_ids = eval_adv_detector(
		logger, data_args, model_args, training_args, trainer, eval_dataset, 
		log_flag=True if extract_ratio is None else False
	)
	
	print("\nEvaluate detector `under adversarial attack` without summarizer...")
	trainer.model.gen_flag = True
	trainer.model.sum_flag = False
	dirty_predictions, _ = eval_adv_detector(
		logger, data_args, model_args, training_args, trainer, eval_dataset, 
		log_flag=True if extract_ratio is None else False, new_line=False
	)
	
	## Evaluate attack success rate (ASR)
	clean_pred_det = np.argmax(clean_predictions[0], axis=1)
	dirty_pred_det = np.argmax(dirty_predictions[0], axis=1)
	labels_det = label_ids[0]
	asr_wo_summ = calculate_asr(labels_det, clean_pred_det, dirty_pred_det)
	
	print("\n********** ASR (Attack Success Rate) w/o summarizer = {:.4f} **********".format(asr_wo_summ))
	if extract_ratio is None:
		with open(training_args.overall_results_path, "a") as fw:
			fw.write("\t{:.4f}\n".format(asr_wo_summ))

	print("\nEvaluate detector `under adversarial attack` with summarizer...")
	trainer.model.gen_flag = True
	trainer.model.sum_flag = True
	summ_predictions, _ = eval_adv_detector(
		logger, data_args, model_args, training_args, trainer, eval_dataset, 
		log_flag=True, new_line=False
	)

	## Evaluate attack success rate (ASR)
	summ_pred_det = np.argmax(summ_predictions[0], axis=1)
	asr_wi_summ = calculate_asr(labels_det, clean_pred_det, summ_pred_det)

	print("\n********** ASR (Attack Success Rate) w/  summarizer = {:.4f} **********".format(asr_wi_summ))
	with open(training_args.overall_results_path, "a") as fw:
		fw.write("\t{:.4f}".format(asr_wi_summ))
		if extract_ratio is not None:
			fw.write("\textract_ratio={:.4f}".format(extract_ratio))
		fw.write("\tk={}".format(model_args.num_clusters))
		fw.write("\n")
		#fw.write("\t{:.4f}\tn_filter_layers={}\n".format(asr_wi_summ, model_args.filter_layer_enc))

	"""
	write_generated_responses(
		training_args, trainer, tokenizer, eval_dataset, 
		labels_det, clean_pred_det, dirty_pred_det, dirty_predictions, as_vec=as_vec_wi_summ
	)
	if model_args.summarizer_type == "abs":
		write_response_summary(training_args, trainer, tokenizer, eval_dataset, summ_predictions)
	"""

def explain_process(logger, data_args, model_args, training_args, trainer, eval_dataset):
	"""Evaluate detector w/ DAS, w/o attack"""

	trainer.model.gen_flag = False
	trainer.model.sum_flag = True if trainer.model.summarizer is not None else False
	predictions, label_ids = eval_adv_detector(logger, data_args, model_args, training_args, trainer, eval_dataset, log_flag=True, new_line=False)

	with open(training_args.overall_results_path, "a") as fw:
		fw.write("extract_ratio={}, k={}\n".format(model_args.filter_ratio, model_args.num_clusters))

##########################################
## Processes for fine-tuning abstractor ##
##########################################
def finetune_abstractor(data_args, model_args, training_args, trainer, train_dataset, last_checkpoint):
	"""Fine-tune abstractor"""
	checkpoint = None
	if training_args.resume_from_checkpoint is not None:
		checkpoint = training_args.resume_from_checkpoint
	elif last_checkpoint is not None:
		checkpoint = last_checkpoint
	train_result = trainer.train(resume_from_checkpoint=checkpoint)
	#trainer.save_model()  ## Saves the tokenizer too for easy upload
	
	metrics = train_result.metrics
	max_train_samples = (data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset))
	metrics["train_samples"] = min(max_train_samples, len(train_dataset))
	
	trainer.log_metrics("train", metrics)

	write_result(trainer.best_metrics, data_args, model_args, training_args, prefix="eval")

def eval_abstractor(data_args, model_args, training_args, tokenizer, trainer, eval_dataset):
	"""Evaluate SSRA"""
	max_length = data_args.max_tweet_length
	num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
	predict_results = trainer.predict(
		eval_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
	)
	metrics = predict_results.metrics
	max_predict_samples = (
		data_args.max_predict_samples if data_args.max_predict_samples is not None else len(eval_dataset)
	)
	metrics["predict_samples"] = min(max_predict_samples, len(eval_dataset))
	
	predictions = tokenizer.batch_decode(
		predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
	)

	#print("ROUGE-1: {}, ROUGE-2: {}, ROUGE-L: {}\n".format(
	#	metrics["predict_rouge1"], 
	#	metrics["predict_rouge2"], 
	#	metrics["predict_rougeL"]
	#))
	#if training_args.
	#write_result(metrics, data_args, model_args, training_args, prefix="predict")

	output_file = "summary"
	if data_args.split_3_groups:
		output_file = "{}-3g".format(output_file)
	if data_args.min_target_length is not None:
		output_file = "{}-{}".format(output_file, data_args.min_target_length)
	if data_args.max_target_length is not None:
		output_file = "{}-{}".format(output_file, data_args.max_target_length)

	## Output predicted summary
	if training_args.task_type == "ssra_loo":
		output_dict = {
			"source_id": eval_dataset["source_id"], 
			"summary": predictions
		}
		
		output_df = pd.DataFrame(data=output_dict)
		output_df.to_csv("{}/{}.csv".format(training_args.output_dir, output_file), index=False)

	if training_args.task_type == "ssra_kmeans" and model_args.cluster_type is not None:
		output_dict = {
			"source_id": eval_dataset["source_id"], 
			"cluster_id": eval_dataset["cluster_id"], 
			"summary": predictions
		}
		
		output_df = pd.DataFrame(data=output_dict)
		output_df.to_csv("{}/{}.csv".format(training_args.output_dir, output_file), index=False)