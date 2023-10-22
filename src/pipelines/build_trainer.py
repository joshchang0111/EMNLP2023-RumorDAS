import ipdb
import numpy as np

import nltk

from datasets import load_metric
from transformers import (
	EvalPrediction, 
	DataCollatorWithPadding, 
	DataCollatorForSeq2Seq, 
	default_data_collator, 
	Seq2SeqTrainer
)

## Self-defined
from .trainer import CustomTrainer
from others.metrics import f1_score_3_class, f1_score_4_class

def build_trainer(
		data_args, model_args, training_args, 
		train_dataset, eval_dataset, 
		model, tokenizer
	):
	"""Building trainer according to different tasks."""
	print("\nBuilding trainer...")

	if training_args.task_type == "train_detector":
		## Get the metric function
		metric = {
			"accuracy": load_metric("accuracy"), 
			"f1": load_metric("f1")
		}
		
		## You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
		## predictions and label_ids field) and has to return a dictionary string to float.
		def compute_metrics(pred: EvalPrediction):
			preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
			#preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
			preds = np.argmax(preds, axis=1)
			
			result = {
				"accuracy": metric["accuracy"].compute(predictions=preds, references=pred.label_ids)["accuracy"], 
				"f1_macro": metric["f1"].compute(predictions=preds, references=pred.label_ids, average="macro")["f1"]
			}
			for label_i in range(data_args.num_labels):
				result["f1_{}".format(label_i)] = metric["f1"].compute(predictions=preds, references=pred.label_ids, average=None)["f1"][label_i]
			return result
		
		## Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
		## we already did the padding.
		if data_args.pad_to_max_length:
			data_collator = default_data_collator
		elif training_args.fp16:
			data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
		else:
			data_collator = None
		
		trainer = CustomTrainer(
			model=model,
			args=training_args,
			train_dataset=train_dataset if training_args.do_train else None,
			eval_dataset=eval_dataset if training_args.do_eval else None,
			compute_metrics=compute_metrics,
			tokenizer=tokenizer,
			data_collator=data_collator,
		)

	elif training_args.task_type == "train_adv_stage1" or \
		 training_args.task_type == "train_adv_stage2":
		## Metric
		metric = {
			"accuracy": load_metric("accuracy"), ## For detector
			"f1"      : load_metric("f1"), ## For detector
			"rouge"   : load_metric("rouge") ## For generator
		}
		
		def postprocess_text(preds, labels):
			preds  = [pred.strip() for pred in preds]
			labels = [label.strip() for label in labels]
			
			## rougeLSum expects newline after each sentence
			preds  = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
			labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
			
			return preds, labels
			
		def compute_metrics(eval_preds):
			"""This function is called in `trainer.evaluation_loop` and `trainer.prediction_loop`"""
			preds, labels = eval_preds
			preds_det, labels_det = preds[0], labels[0] ## For detector
			preds_gen, labels_gen = preds[1], labels[1] ## For generator

			####################
			## For generation ##
			####################
			decoded_preds = tokenizer.batch_decode(preds_gen, skip_special_tokens=True)
			if data_args.ignore_pad_token_for_loss:
				## Replace -100 in the labels as we can't decode them.
				labels_gen = np.where(labels_gen != -100, labels_gen, tokenizer.pad_token_id)
			decoded_labels = tokenizer.batch_decode(labels_gen, skip_special_tokens=True)
			
			## Some simple post-processing
			decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
			
			result = metric["rouge"].compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
			## Extract a few results from ROUGE
			result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
			
			prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
			result["gen_len"] = np.mean(prediction_lens)
			result = {k: round(v, 4) for k, v in result.items()}

			########################
			## For classification ##
			########################
			preds_det = np.argmax(preds_det, axis=1)
			result["accuracy"] = metric["accuracy"].compute(predictions=preds_det, references=labels_det)["accuracy"]
			
			## Calculate F1 scores
			"""
			result["f1_macro"] = metric["f1"].compute(predictions=preds_det, references=labels_det, average="macro")["f1"]
			
			for label_i in range(data_args.num_labels):
				result["f1_{}".format(label_i)] = metric["f1"].compute(predictions=preds_det, references=labels_det, average=None)["f1"][label_i]
			"""

			num_labels = data_args.num_labels#len(set(labels_det))
			if num_labels == 3:
				F1_all = f1_score_3_class(preds_det, labels_det)
			elif num_labels == 4:
				F1_all = f1_score_4_class(preds_det, labels_det)
			F1_all = np.array(F1_all)

			## Only take classes that exist in `labels_det`
			indices = np.array(list(set(labels_det)))
			F1_filt = F1_all[indices]
			result["f1_macro"] = np.mean(F1_filt)

			for label_i in range(data_args.num_labels):
				## -1: Labels that do not exist in test set, so ignore it
				result["f1_{}".format(label_i)] = F1_all[label_i] if label_i in indices else -1.

			return result
		
		## Data collator
		label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
		data_collator = DataCollatorForSeq2Seq(
			tokenizer,
			model=model,
			label_pad_token_id=label_pad_token_id,
			pad_to_multiple_of=8 if training_args.fp16 else None,
		)
		
		## Initialize our Trainer
		from .trainer_adv import CustomSeq2SeqTrainer
		trainer = CustomSeq2SeqTrainer(
			model=model,
			args=training_args,
			model_args=model_args, 
			data_args=data_args, 
			train_dataset=train_dataset if training_args.do_train else None,
			eval_dataset=eval_dataset if training_args.do_eval else None,
			tokenizer=tokenizer,
			data_collator=data_collator,
			compute_metrics=compute_metrics if training_args.predict_with_generate else None,
		)

	elif training_args.task_type == "ssra_loo" or \
		 training_args.task_type == "ssra_kmeans":
		## Metric
		metric = {
			"rouge": load_metric("rouge"),
			"perplexity": load_metric("perplexity")
		}
		
		def postprocess_text(preds, labels):
			preds  = [pred.strip() for pred in preds]
			labels = [label.strip() for label in labels]
			
			# rougeLSum expects newline after each sentence
			preds  = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
			labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
			
			return preds, labels
			
		def compute_metrics(eval_preds):
			preds, labels = eval_preds
			if isinstance(preds, tuple):
				preds = preds[0]
			decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
			if data_args.ignore_pad_token_for_loss:
				# Replace -100 in the labels as we can't decode them.
				labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
			decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
			
			## Some simple post-processing
			decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
			
			result = metric["rouge"].compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
			## Extract a few results from ROUGE
			result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
			
			prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
			result["gen_len"] = np.mean(prediction_lens)
			result = {k: round(v, 4) for k, v in result.items()}
		
			## NEW: add perplexity
			#result["perplexity"] = metric["perplexity"].compute(predictions=decoded_preds, model_id="gpt2")
			return result
		
		## Data collator
		label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
		data_collator = DataCollatorForSeq2Seq(
			tokenizer,
			model=model,
			label_pad_token_id=label_pad_token_id,
			pad_to_multiple_of=8 if training_args.fp16 else None,
		)
		
		## Initialize our Trainer
		from .trainer_abstractor import CustomSeq2SeqTrainer
		trainer = CustomSeq2SeqTrainer(
			model=model,
			args=training_args,
			model_args=model_args, 
			data_args=data_args, 
			train_dataset=train_dataset if training_args.do_train else None,
			eval_dataset=eval_dataset if training_args.do_eval else None,
			tokenizer=tokenizer,
			data_collator=data_collator,
			compute_metrics=compute_metrics if training_args.predict_with_generate else None,
		)
	
	elif training_args.task_type == "train_filter":
		from .trainer_filter import FilterTrainer
		trainer = FilterTrainer(
			model=model,
			data_args=data_args,
			model_args=model_args,
			training_args=training_args,
			train_dataset=train_dataset,
			eval_dataset=eval_dataset,
		)

	elif training_args.task_type == "build_cluster_summary":
		from .builder_cluster_summary import ClusterSummaryBuilder
		trainer = ClusterSummaryBuilder(
			model=model, 
			data_args=data_args, 
			model_args=model_args, 
			training_args=training_args, 
			train_dataset=train_dataset, 
			eval_dataset=eval_dataset
		)

	else:
		raise ValueError("training_args.task_type not specified!")
	return trainer
