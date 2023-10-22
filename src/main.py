"""
This code is developed based on:
	https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
"""
import ipdb
import logging

import nltk  ## Here to have a nice missing dependency error message early on

import transformers
from transformers import HfArgumentParser, set_seed
from transformers.file_utils import is_offline_mode
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from filelock import FileLock

## Self-defined
from data.build_datasets import build_datasets
from models.build_model import build_model
from pipelines.build_trainer import build_trainer
from others.args import (
	args_post_init, 
	CustomTrainingArguments, 
	DataTrainingArguments, 
	ModelArguments
)
from others.utils import setup_logging
from others.processes import (
	train_process, 
	eval_process, 
	predict_process, 
	## ------------- 
	train_adv, 
	eval_adv_detector, 
	eval_adv_asr, 
	eval_adv_asr_with_summ, 
	## --------------------
	finetune_abstractor, 
	eval_abstractor
)

## Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.18.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)

try:
	nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
	if is_offline_mode():
		raise LookupError(
			"Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
		)
	with FileLock(".lock") as lock:
		nltk.download("punkt", quiet=True)

def main():
	## Parse args
	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
	model_args, data_args, training_args = parser.parse_args_into_dataclasses()
	model_args, data_args, training_args = args_post_init(model_args, data_args, training_args)

	## Settings
	last_checkpoint = setup_logging(logger, training_args, data_args, model_args)
	set_seed(training_args.seed) ## Set seed before initializing model.
	
	################
	## Load model ##
	################
	config, tokenizer, model = build_model(data_args, model_args, training_args)
	
	###############################
	## Load & preprocess dataset ##
	###############################
	train_dataset, eval_dataset, test_dataset = build_datasets(data_args, model_args, training_args, 
															   config, tokenizer, model)

	###################
	## Build trainer ##
	###################
	trainer = build_trainer(
		data_args, model_args, training_args, 
		train_dataset, eval_dataset, 
		model, tokenizer
	)
	
	##########################
	## Train / Test process ##
	##########################
	if training_args.task_type == "train_detector":
		if training_args.do_train:
			train_process(data_args, model_args, training_args, trainer, train_dataset, last_checkpoint)
		elif training_args.do_eval and not training_args.do_predict:
			eval_process(logger, data_args, model_args, training_args, trainer, test_dataset)
		elif training_args.do_eval and training_args.do_predict:
			predict_process(logger, data_args, model_args, training_args, trainer, test_dataset)
	
	elif training_args.task_type == "train_adv_stage1":
		if training_args.do_train:
			train_adv(data_args, model_args, training_args, trainer, train_dataset, last_checkpoint)
		elif training_args.do_eval:
			eval_adv_detector(logger, data_args, model_args, training_args, trainer, test_dataset, log_flag=True)

	elif training_args.task_type == "train_adv_stage2":
		if training_args.do_train:
			train_adv(data_args, model_args, training_args, trainer, train_dataset, last_checkpoint)
		elif training_args.do_eval:
			if model.summarizer is None:
				eval_adv_asr(logger, data_args, model_args, training_args, tokenizer, trainer, test_dataset)
			else:
				eval_adv_asr_with_summ(logger, data_args, model_args, training_args, tokenizer, trainer, test_dataset)

	elif training_args.task_type == "ssra_loo" or \
		 training_args.task_type == "ssra_kmeans":
		if training_args.do_train:
			finetune_abstractor(data_args, model_args, training_args, trainer, train_dataset, last_checkpoint)
		elif training_args.do_eval:
			eval_abstractor(data_args, model_args, training_args, tokenizer, trainer, test_dataset)

	elif training_args.task_type == "train_filter":
		trainer.train()

	elif training_args.task_type == "build_cluster_summary":
		trainer.build_cluster_summary()

	else:
		raise ValueError("training_args.task_type not correctly specified!")

if __name__ == "__main__":
	main()

