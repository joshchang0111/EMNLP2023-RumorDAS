import os
import sys
import ipdb
import wandb
import logging
import numpy as np
import pandas as pd

import datasets
import transformers
from datasets import load_metric
from transformers.trainer_utils import get_last_checkpoint

import torch

def mean_pooling(hidden_states, attn_mask, dim=1):
	"""
	Takes a batch of hidden states and attention masks as inputs.
	Inputs:
		- hidden_states: (bs, seq_len, hidden_dim)
		- attn_mask    : (bs, seq_len)
	"""
	sum_ = (hidden_states * attn_mask.unsqueeze(dim=-1)).sum(dim=1)
	avg_ = (sum_ / attn_mask.sum(dim=1).unsqueeze(dim=-1))
	return avg_

def setup_logging(logger, training_args, data_args, model_args):
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		handlers=[logging.StreamHandler(sys.stdout)],
	)

	#log_level = training_args.get_process_log_level()
	log_level = logging.WARNING ## only report errors & warnings
	logger.setLevel(log_level)
	datasets.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.enable_default_handler()
	transformers.utils.logging.enable_explicit_format()

	## Log on each process the small summary:
	logger.warning(
		f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
		+ f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
	)
	logger.info(f"Training/evaluation parameters {training_args}")

	## For summarization
	if data_args.source_prefix is None and model_args.model_name_or_path in [
		"t5-small",
		"t5-base",
		"t5-large",
		"t5-3b",
		"t5-11b",
	]:
		logger.warning(
			"You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
			"`--source_prefix 'summarize: ' `"
		)

	## Detecting last checkpoint.
	last_checkpoint = None
	if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
		last_checkpoint = get_last_checkpoint(training_args.output_dir)
		if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
			raise ValueError(
				f"Output directory ({training_args.output_dir}) already exists and is not empty. "
				"Use --overwrite_output_dir to overcome."
			)
		elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
			logger.info(
				f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
				"the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
			)
			
	return last_checkpoint

def find_ckpt_dir(root_dir):
	"""Find checkpoint directory name given root directory."""
	ckpt_dir = None
	for file_or_dir in os.listdir(root_dir):
		if file_or_dir.startswith("checkpoint"):
			ckpt_dir = file_or_dir
			break
	if ckpt_dir is None:
		raise ValueError("Pre-trained checkpoint directory doesn't exists!")
	return ckpt_dir

def calculate_asr(labels, clean_predictions, dirty_predictions):
	"""Evaluate the attack success rate given clean predictions & dirty predictions"""
	ori_correct = (clean_predictions == labels)
	now_wrong   = (dirty_predictions != labels)
	as_vec = ori_correct & now_wrong
	#asr = sum(as_vec) / len(as_vec) ## correct -> wrong
	asr = as_vec.sum() / ori_correct.sum()
	return asr#, as_vec

def write_result(metrics, data_args, model_args, training_args, mode=None, gen_flag=False, sum_flag=False, new_line=True, prefix=""):
	"""Write metrics to `overall_results.txt`"""
	## Write overall results (model, acc, f1-macro, ...)

	if training_args.task_type == "train_detector":
		sum_flag = "-" if not sum_flag else "v"
		
		metrics2report = [
			"{:20s}".format(model_args.model_name[:20]),  ## Model
			"{:10s}".format(sum_flag), ## sum_flag
			"{:8s}".format(mode), ## Mode
			"{:20s}".format(data_args.fold[:20]), ## Fold
			"{:<10.4f}".format(metrics["eval_accuracy"]),
			"{:<10.4f}".format(metrics["eval_f1_macro"])
		]

		f1s = ["{:<10.4f}".format(metrics["eval_f1_{}".format(label_i)]) for label_i in range(data_args.num_labels)]
		metrics2report.extend(f1s)
		
		with open(training_args.overall_results_path, "a") as fw:
			fw.write("{}\n".format("\t".join(metrics2report)))

	elif training_args.task_type == "train_adv_stage1" or training_args.task_type == "train_adv_stage2":
		gen_flag = "v" if gen_flag else "-"
		sum_flag = "v" if sum_flag else "-"

		if sum_flag == "v":
			if "filter" in model_args.extractor_name_or_path or "kmeans" in model_args.extractor_name_or_path:
				sum_flag = "DRE"
				
				if model_args.abstractor_name_or_path:
					sum_flag = "DAS"
		
		metrics2report = [
			"{:8s}".format(mode), ## Mode
			"{:10s}".format(gen_flag), 
			"{:10s}".format(sum_flag), 
			"{:20s}".format(data_args.fold[:20]), ## Fold
			"{:<10.4f}".format(metrics["eval_accuracy"]),
			"{:<10.4f}".format(metrics["eval_f1_macro"])
		]
		
		f1s = ["{:<10.4f}".format(metrics["eval_f1_{}".format(label_i)]) for label_i in range(data_args.num_labels)]
		metrics2report.extend(f1s)
		
		if "eval_rouge1" in metrics:
			metrics2report.append("{:<10.4f}".format(metrics["eval_rouge1"]))
			metrics2report.append("{:<10.4f}".format(metrics["eval_rouge2"]))
			metrics2report.append("{:<10.4f}".format(metrics["eval_rougeL"]))
		
		with open(training_args.overall_results_path, "a") as fw:
			fw.write("{}".format("\t".join(metrics2report)))
			if new_line:
				fw.write("\n")

	elif training_args.task_type == "ssra_loo" or \
		 training_args.task_type == "ssra_kmeans":

		metrics2report = ["{:20s}".format(data_args.fold[:20])]
		metrics2report.append("{:<10.4f}".format(metrics["{}_rouge1".format(prefix)]))
		metrics2report.append("{:<10.4f}".format(metrics["{}_rouge2".format(prefix)]))
		metrics2report.append("{:<10.4f}".format(metrics["{}_rougeL".format(prefix)]))
		with open(training_args.overall_results_path, "a") as fw:
			fw.write("{}\n".format("\t".join(metrics2report)))

def write_generated_responses(
		training_args, trainer, tokenizer, eval_dataset, 
		labels_det, clean_pred_det, dirty_pred_det, dirty_predictions, as_vec
	):
	"""Write generated adversarial response to file"""
	pred_gen = dirty_predictions[1] ## same as clean_predictions[1]
	if trainer.is_world_process_zero():
		pred_gen = tokenizer.batch_decode(
			pred_gen, skip_special_tokens=True, clean_up_tokenization_spaces=True
		)
		pred_gen = [pred.strip() for pred in pred_gen]

	adv_response_path = "{}/generated_response.txt".format(training_args.output_dir)
	print("\nWriting generated adversarial responses to {}".format(adv_response_path))
	with open(adv_response_path, "w") as fw:
		for pred_idx, response in enumerate(pred_gen):
			if as_vec[pred_idx]:
				success_or_fail = "success"
			else:
				success_or_fail = "failure"

			if clean_pred_det[pred_idx] == dirty_pred_det[pred_idx]:
				success_or_fail = "-------"
			if (clean_pred_det[pred_idx] != labels_det[pred_idx]) and (dirty_pred_det[pred_idx] != labels_det[pred_idx]):
				success_or_fail = "-------"

			if isinstance(eval_dataset["source_id"][pred_idx], int):
				fw.write(
					"{:20d}\t{:10s}[{:10s}->{:10s}]\t{}\t{}\n".format(
						eval_dataset["source_id"][pred_idx], 
						trainer.model.config.id2label[labels_det[pred_idx]], 
						trainer.model.config.id2label[clean_pred_det[pred_idx]], 
						trainer.model.config.id2label[dirty_pred_det[pred_idx]], 
						success_or_fail, response
					)
				)
			else:
				fw.write(
					"{:20s}\t{:10s}[{:10s}->{:10s}]\t{}\t{}\n".format(
						eval_dataset["source_id"][pred_idx], 
						trainer.model.config.id2label[labels_det[pred_idx]], 
						trainer.model.config.id2label[clean_pred_det[pred_idx]], 
						trainer.model.config.id2label[dirty_pred_det[pred_idx]], 
						success_or_fail, response
					)
				)

def write_response_summary(training_args, trainer, tokenizer, eval_dataset, summ_predictions):
	"""Write response summary to file"""
	summary_tokens = summ_predictions[2]
	if trainer.is_world_process_zero():
		summary_tokens = tokenizer.batch_decode(
			summary_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
		)
		summary_tokens = [pred.strip() for pred in summary_tokens]

	response_summary_path = "{}/response_summary.txt".format(training_args.output_dir)
	print("\nWriting response summary to {}".format(response_summary_path))
	with open(response_summary_path, "w") as fw:
		for pred_idx, summary in enumerate(summary_tokens):
			if isinstance(eval_dataset["source_id"][pred_idx], int):
				fw.write("{:20d}\t{}\n".format(eval_dataset["source_id"][pred_idx], summary))
			else:
				fw.write("{:20s}\t{}\n".format(eval_dataset["source_id"][pred_idx], summary))

def write_cls_predictions(data_args, training_args, predictions, label_ids):
	"""Output prediction result of classification task (for PHEME)"""
	output_dir = "{}/{}/{}/cls_predictions".format(training_args.output_root, data_args.dataset_name, training_args.exp_name)
	os.makedirs(output_dir, exist_ok=True)

	preds = np.argmax(predictions[0], axis=1)
	data_ = pd.DataFrame({
		"preds": preds, 
		"label": label_ids[0]
	})

	data_.to_csv("{}/{}.csv".format(output_dir, data_args.fold), index=False)

def calculate_PHEME_results(data_args, training_args):
	"""Calculate Accuracy / macro-averaged F1 / weighted-averaged F1 for PHEME from predictions of all folds."""
	n_folds = 9
	metric = {
			"accuracy": load_metric("accuracy"), 
			"f1"      : load_metric("f1")
	}
	input_dir = "{}/{}/{}/cls_predictions".format(training_args.output_root, data_args.dataset_name, training_args.exp_name)

	preds_all, label_all = [], []
	for fold in range(n_folds):
		result_csv = pd.read_csv("{}/{}.csv".format(input_dir, fold))
		preds_all.append(result_csv["preds"].to_numpy())
		label_all.append(result_csv["label"].to_numpy())

	preds_all = np.concatenate(preds_all, axis=0)
	label_all = np.concatenate(label_all, axis=0)

	accuracy = metric["accuracy"].compute(predictions=preds_all, references=label_all)["accuracy"]
	f1_macro = metric["f1"].compute(predictions=preds_all, references=label_all, average="macro")["f1"]
	f1_weighted = metric["f1"].compute(predictions=preds_all, references=label_all, average="weighted")["f1"]

	with open(training_args.overall_results_path, "a") as fw:
		fw.write("{:10s}\t{:10s}\t{:10s}\n".format("Acc", "mF1", "wF1"))
		fw.write("{:<10.4f}\t{:10.4f}\t{:10.4f}\n".format(accuracy, f1_macro, f1_weighted))

def post_process_generative_model(data_args, model_args, model):
	"""Post processing for generative models."""
	if model.config.decoder_start_token_id is None:
		raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
	
	if (
		hasattr(model.config, "max_position_embeddings")
		and model.config.max_position_embeddings < data_args.max_source_length
	):
		if model_args.resize_position_embeddings is None:
			logger.warning(
				f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
				f"to {data_args.max_source_length}."
			)
			model.resize_position_embeddings(data_args.max_source_length)
		elif model_args.resize_position_embeddings:
			model.resize_position_embeddings(data_args.max_source_length)
		else:
			raise ValueError(
				f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
				f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
				"resize the model's position encodings by passing `--resize_position_embeddings`."
			)
	return model