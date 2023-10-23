import os
import ipdb

import torch

from transformers import (
	AutoConfig,
	AutoModelForSequenceClassification,
	AutoTokenizer,
	AutoModelForSeq2SeqLM
)

## Self-defined
from .detector import (
	RobertaForRumorDetection, 
	BertForRumorDetection, 
	BartEncoderForRumorDetection
)
from .detector_generator import BartForRumorDetectionAndResponseGeneration
from .modeling_filter import TransformerAutoEncoder
from .modeling_abstractor import BartForAbstractiveResponseSummarization#, RobertaForExtractiveResponseSummarization, ResponseExtractor
from others.utils import find_ckpt_dir, post_process_generative_model

def build_model(data_args, model_args, training_args):
	"""Build models according to different tasks"""

	## Load pre-trained model checkpoint
	config, tokenizer, model = load_model(data_args, model_args, training_args)

	## Initialize other modules
	if hasattr(model, "init_args_modules"):
		model.init_args_modules(data_args, model_args, training_args, tokenizer=tokenizer)

	## Load trained model
	if training_args.task_type == "train_detector":
		if not training_args.do_train and training_args.do_eval:
			model_args.model_name_or_path = "{}/{}".format(training_args.output_dir, find_ckpt_dir(training_args.output_dir))
			model.load_state_dict(torch.load("{}/pytorch_model.bin".format(model_args.model_name_or_path)))
			print("Detector checkpoint: {}".format(model_args.model_name_or_path))

	## Build model for adversarial training
	if training_args.task_type == "train_adv_stage1" or \
	   training_args.task_type == "train_adv_stage2":
	   
		## Load trained model
		ckpt_path = None
		if training_args.task_type == "train_adv_stage1":
			if training_args.do_eval and not training_args.do_train:
				ckpt_path = "{}/{}/{}/{}".format(
					training_args.output_root, data_args.dataset_name, training_args.exp_name, data_args.fold)
				print("\nLoading detector checkpoint from adversarial training stage 1...")

		elif training_args.task_type == "train_adv_stage2":
			if training_args.do_train:
				print("\nLoading model checkpoint from adversarial training stage 1...")
				ckpt_path = "{}/{}/{}/adv-stage1/{}".format(
					training_args.output_root, data_args.dataset_name, training_args.exp_name.split("/")[0], data_args.fold)
			elif training_args.do_eval:
				print("\nLoading detector & attacker from adversarial training stage 2...")
				ckpt_path = "{}/{}/{}/{}".format(
					training_args.output_root, data_args.dataset_name, training_args.exp_name, data_args.fold)

		if ckpt_path is not None:
			ckpt_dir = find_ckpt_dir(ckpt_path)
			print("Checkpoint path: {}".format("{}/{}/pytorch_model.bin".format(ckpt_path, ckpt_dir)))

			## Partially load the model checkpoint (ignore summarizer)
			ckpt_state_dict = torch.load("{}/{}/pytorch_model.bin".format(ckpt_path, ckpt_dir))
			ckpt_state_dict = {
				k: v
				for k, v in ckpt_state_dict.items()
				if not k.startswith("summarizer")
			}
			model.load_state_dict(ckpt_state_dict, strict=False)
	
	print("Model name: {}".format(model.__class__.__name__))

	return config, tokenizer, model

def load_model(data_args, model_args, training_args):
	"""Load pre-trained models according to different tasks"""
	print("\nLoading pre-trained model & tokenizer...")

	if training_args.task_type == "train_detector":
		if "roberta" in model_args.model_name_or_path:
			detector = RobertaForRumorDetection
		elif "bert" in model_args.model_name_or_path:
			detector = BertForRumorDetection
		elif "bart" in model_args.model_name_or_path:
			detector = BartEncoderForRumorDetection

		## Load trained checkpoint if doing evaluation
		if not training_args.do_train and training_args.do_eval:
			ckpt_dir = find_ckpt_dir(training_args.output_dir)
			model_args.model_name_or_path = "{}/{}".format(training_args.output_dir, ckpt_dir)

		print("Detector checkpoint: {}".format(model_args.model_name_or_path))
		
		config = AutoConfig.from_pretrained(
			model_args.config_name if model_args.config_name else model_args.model_name_or_path,
			num_labels=data_args.num_labels,
			finetuning_task=data_args.task_name,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
		tokenizer = AutoTokenizer.from_pretrained(
			model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			use_fast=model_args.use_fast_tokenizer,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
		model = detector.from_pretrained(
			model_args.model_name_or_path,
			from_tf=bool(".ckpt" in model_args.model_name_or_path),
			config=config,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None
		)

	elif training_args.task_type == "predict_summary":
		print("Pre-trained encoder-decoder: {}".format(model_args.model_name_or_path))
		config = AutoConfig.from_pretrained(
			model_args.config_name if model_args.config_name else model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
		tokenizer = AutoTokenizer.from_pretrained(
			model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			use_fast=model_args.use_fast_tokenizer,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
		model = AutoModelForSeq2SeqLM.from_pretrained(
			model_args.model_name_or_path,
			from_tf=bool(".ckpt" in model_args.model_name_or_path),
			config=config,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
		model = post_process_generative_model(data_args, model_args, model)

	elif training_args.task_type == "train_adv_stage1" or \
		 training_args.task_type == "train_adv_stage2":

		print("Pre-trained encoder-decoder: {}".format(model_args.model_name_or_path))
		config = AutoConfig.from_pretrained(
			model_args.config_name if model_args.config_name else model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
		tokenizer = AutoTokenizer.from_pretrained(
			model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			use_fast=model_args.use_fast_tokenizer,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
		model = BartForRumorDetectionAndResponseGeneration.from_pretrained(
			model_args.model_name_or_path,
			from_tf=bool(".ckpt" in model_args.model_name_or_path),
			config=config,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
		model = post_process_generative_model(data_args, model_args, model)

	elif training_args.task_type == "ssra_loo" or \
		 training_args.task_type == "ssra_kmeans":
		
		## Load model for evaluation
		if (training_args.do_eval and not training_args.do_train):
			if training_args.task_type == "ssra_loo" and model_args.model_name_or_path == "ssra_loo":
				print("Load model from SSRA-LOO...")
				ckpt_path = "{}/{}/ssra_loo/{}".format(training_args.output_root, data_args.dataset_name, data_args.fold)
				ckpt_path = "{}/{}".format(ckpt_path, find_ckpt_dir(ckpt_path))
				model_args.model_name_or_path = ckpt_path
			elif training_args.task_type == "ssra_kmeans":
				print("Load model from SSRA-KMeans...")
				ckpt_path = "{}/{}/{}/{}".format(training_args.output_root, data_args.dataset_name, training_args.exp_name, data_args.fold)
				ckpt_path = "{}/{}".format(ckpt_path, find_ckpt_dir(ckpt_path))
				model_args.model_name_or_path = ckpt_path

		print("Pre-trained summarizer checkpoint: {}".format(model_args.model_name_or_path))
		config = AutoConfig.from_pretrained(
			model_args.config_name if model_args.config_name else model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
		tokenizer = AutoTokenizer.from_pretrained(
			model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			use_fast=model_args.use_fast_tokenizer,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
		model = BartForAbstractiveResponseSummarization.from_pretrained(
			model_args.model_name_or_path,
			from_tf=bool(".ckpt" in model_args.model_name_or_path),
			config=config,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
		model = post_process_generative_model(data_args, model_args, model)

	elif training_args.task_type == "train_filter":
		print("\nLoading rumor detector from adversarial training stage 2 for embedding layer...")
		ckpt_path = "{}/{}/bi-tgn/adv-stage2/{}".format(training_args.output_root, data_args.dataset_name, data_args.fold)
		ckpt_path = "{}/{}".format(ckpt_path, find_ckpt_dir(ckpt_path))
		print(ckpt_path)

		config = AutoConfig.from_pretrained(
			#model_args.config_name if model_args.config_name else model_args.model_name_or_path,
			ckpt_path, 
			num_labels=data_args.num_labels,
			finetuning_task=data_args.task_name,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
		tokenizer = AutoTokenizer.from_pretrained(
			#model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
			ckpt_path, 
			cache_dir=model_args.cache_dir,
			use_fast=model_args.use_fast_tokenizer,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
		model_emb = BartForRumorDetectionAndResponseGeneration.from_pretrained(
			model_args.model_name_or_path,
			from_tf=bool(".ckpt" in model_args.model_name_or_path),
			config=config,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None
		)
		model_args.td_gcn = True
		model_args.bu_gcn = True
		model_emb.init_args_modules(data_args, model_args, training_args, tokenizer=tokenizer)
		model_emb.load_state_dict(torch.load("{}/pytorch_model.bin".format(ckpt_path)))
		model = TransformerAutoEncoder(
			model_emb=model_emb, 
			num_layers_enc=model_args.filter_layer_enc, 
			num_layers_dec=model_args.filter_layer_dec
		)

	elif training_args.task_type == "build_cluster_summary":
		config = AutoConfig.from_pretrained(
			model_args.config_name if model_args.config_name else model_args.model_name_or_path,
			num_labels=data_args.num_labels,
			finetuning_task=data_args.task_name,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
		tokenizer = AutoTokenizer.from_pretrained(
			model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
			cache_dir=model_args.cache_dir,
			use_fast=model_args.use_fast_tokenizer,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None,
		)
		
		print("\nLoading rumor detector from adversarial training stage 2...")
		ckpt_path = "{}/{}/bi-tgn/adv-stage2/{}".format(training_args.output_root, data_args.dataset_name, data_args.fold)
		ckpt_dir  = find_ckpt_dir(ckpt_path)

		model = BartForRumorDetectionAndResponseGeneration.from_pretrained(
			model_args.model_name_or_path,
			from_tf=bool(".ckpt" in model_args.model_name_or_path),
			config=config,
			cache_dir=model_args.cache_dir,
			revision=model_args.model_revision,
			use_auth_token=True if model_args.use_auth_token else None
		)
		model.load_state_dict(torch.load("{}/{}/pytorch_model.bin".format(ckpt_path, ckpt_dir)), strict=False)

	else:
		raise ValueError("training_args.task_type not specified!")

	return config, tokenizer, model