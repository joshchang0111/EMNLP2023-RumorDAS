"""
This code is developed based on:
	- https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
	- https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_seq2seq.py
"""

import os
import ipdb
import shutil
from collections import Counter
from typing import Optional, Union, Dict, Any, Callable, Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import Seq2SeqTrainer
from transformers.file_utils import (
	is_sagemaker_mp_enabled, 
	is_torch_tpu_available
)
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_pt_utils import (
	find_batch_size, 
	nested_concat, 
	nested_numpify, 
	nested_truncate
)
from transformers.trainer_utils import (
	EvalPrediction, 
	EvalLoopOutput, 
	denumpify_detensorize, 
	has_length
)
from transformers.trainer_callback import TrainerCallback
from transformers.utils import logging
from transformers.deepspeed import is_deepspeed_zero3_enabled

from others.args import DataTrainingArguments, ModelArguments

logger = logging.get_logger(__name__)

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
	"""
	Customized Seq2SeqTrainer for adversarial training
	
	Functions that are modified:
		- __init__
		- _maybe_log_save_evaluate: model saving strategy
		- _rotate_checkpoints     : model saving strategy
		- _save                   : model saving strategy
		- compute_loss: called in `self.training_step` and `self.prediction_step`
		- evaluation_loop
		- prediction_loop (TO BE ADDED!)
		- training_step  : called in `self.evaluation_loop` and `self.prediction_loop`
		- prediction_step: called in `self.evaluation_loop` and `self.prediction_loop`

	Functions that are added:
		- _freeze_all_params
		- _unfreeze_specified_params
	"""
	def __init__(
		self,
		model: Union[PreTrainedModel, nn.Module] = None,
		args: TrainingArguments = None, 
		model_args: ModelArguments = None, 
		data_args: DataTrainingArguments = None, 
		data_collator: Optional[DataCollator] = None,
		train_dataset: Optional[Dataset] = None,
		eval_dataset: Optional[Dataset] = None,
		tokenizer: Optional[PreTrainedTokenizerBase] = None,
		model_init: Callable[[], PreTrainedModel] = None,
		compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
		callbacks: Optional[List[TrainerCallback]] = None,
		optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
		preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
	):
		Seq2SeqTrainer.__init__(
			self, model, args, data_collator, train_dataset, 
			eval_dataset, tokenizer, model_init, compute_metrics, 
			callbacks, optimizers, preprocess_logits_for_metrics
		)
		
		## Override
		self.label_names = ["labels_det", "labels_gen"]

		self.data_args  = data_args
		self.model_args = model_args

		## For best model saving
		self._ckpt_eval_loss = {}
		if self.args.save_model_accord_to_metric:
			self._ckpt_eval_metric = {}
		self.best_metrics = None
		self.best_checkpoint_path = None

		## Make specified parameters trainable
		train_only, freeze_again = None, None
		if self.args.task_type == "train_adv_stage2":
			train_only = "model.decoder lm_head"
			freeze_again = "model.shared" ## embedding layer shared by encoder and decoder
		self._freeze_all_params(self.model)
		self._unfreeze_specified_params(self.model, train_only, freeze_again)

		## Show number of parameters
		all_param_num = sum([p.nelement() for p in self.model.parameters()]) 
		trainable_param_num = sum([
			p.nelement()
			for p in self.model.parameters()
			if p.requires_grad == True
		])
		print("All       parameters: {}".format(all_param_num))
		print("Trainable parameters: {}".format(trainable_param_num))

		## Setup loss weight for classification
		if train_dataset is not None:
			num_classes = Counter(train_dataset["labels_det"])
			num_classes = sorted(num_classes.items())
			num_classes = torch.LongTensor([n[1] for n in num_classes])
			loss_weight = num_classes.max() / num_classes
			loss_weight = loss_weight.to(self.model.device)
			
			self.model.loss_weight = loss_weight
			if self.model.detector_head.__class__.__name__ == "GCNForClassification":
				self.model.detector_head.loss_weight = loss_weight

	def _freeze_all_params(self, model):
		for param in model.parameters():
			param.requires_grad = False

	def _unfreeze_specified_params(self, model, train_only=None, freeze_again=None):
		if train_only is not None:
			names = train_only.split()
			for train_name in names:
				for name, sub_module in self.model.named_modules():
					#if train_name in name:
					if name.startswith(train_name):
						for param in sub_module.parameters():
							param.requires_grad = True
		else:
			for param in model.parameters():
				param.requires_grad = True

		## Freeze parameters with confusing names:
		## [model.shared, model.encoder.embed_tokens, model.decoder.embed_tokens] are the same objects!
		if freeze_again is not None:
			names = freeze_again.split()
			for freeze_name in names:
				for name, sub_module in self.model.named_modules():
					if name.startswith(freeze_name):
						for param in sub_module.parameters():
							param.requires_grad = False

	def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
		"""
		Modification:
			- record current eval loss / metric for best model saving
		"""
		if self.control.should_log:
			if is_torch_tpu_available():
				xm.mark_step()

			logs: Dict[str, float] = {}

			# all_gather + mean() to get average loss over all processes
			tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

			# reset tr_loss to zero
			tr_loss -= tr_loss

			logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
			logs["learning_rate"] = self._get_learning_rate()

			self._total_loss_scalar += tr_loss_scalar
			self._globalstep_last_logged = self.state.global_step
			self.store_flos()

			self.log(logs)

		metrics = None
		if self.control.should_evaluate:
			metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
			self._report_to_hp_search(trial, epoch, metrics)

			## NEW: display metrics properly
			print("\n***** Epoch [{:.4f}] Evaluation Results *****".format(metrics["epoch"]))
			print("Loss D  : {:7.4f}, Loss G  : {:7.4f}".format(metrics["eval_loss_det"], metrics["eval_loss_gen"]))
			print("Accuracy: {:7.4f}, F1-Macro: {:7.4f}".format(metrics["eval_accuracy"], metrics["eval_f1_macro"]))
			print("ROUGE-1 : {:7.4f}, ROUGE-2 : {:7.4f}, ROUGE-L: {:7.4f}".format(metrics["eval_rouge1"], metrics["eval_rouge2"], metrics["eval_rougeL"]))

			## NEW: record metric
			## -> currently based on the detector's performance
			if self.args.save_model_accord_to_metric:
				self._cur_eval_metric = metrics["eval_f1_macro"]
			self._cur_eval_loss = metrics["eval_loss_det"]

			if self.args.task_type == "train_adv_stage2":
				if self.args.attack_type == "untargeted":
					self._cur_eval_metric = -1 * self._cur_eval_metric
					
					best_f1_macro = 100 if self.best_metrics is None else self.best_metrics["eval_f1_macro"]
					if metrics["eval_f1_macro"] < best_f1_macro:
						self.best_metrics = metrics
			else:
				best_f1_macro = 0 if self.best_metrics is None else self.best_metrics["eval_f1_macro"]
				if metrics["eval_f1_macro"] > best_f1_macro:
					self.best_metrics = metrics

		if self.control.should_save:
			self._save_checkpoint(model, trial, metrics=metrics)
			self.control = self.callback_handler.on_save(self.args, self.state, self.control)

	def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
		"""
		Modification:
			- record eval loss / metric and maintain best model

		NOTE: 
			to make this function works properly, 
			the save_steps should be multiples of evaluation_steps
		"""
		## NEW
		if self.args.save_strategy == "steps":
			if self.args.eval_steps != self.args.save_steps:
				raise Exception(
					"To properly store best models, please make sure eval_steps equals to save_steps."
				)

		if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
			return

		# Check if we should delete older checkpoint(s)
		checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)

		## NEW: record the eval metric for the last checkpoint
		self._ckpt_eval_loss[checkpoints_sorted[-1]] = self._cur_eval_loss
		if self.args.save_model_accord_to_metric:
			self._ckpt_eval_metric[checkpoints_sorted[-1]] = self._cur_eval_metric

		if len(checkpoints_sorted) <= self.args.save_total_limit:
			return

		"""
		# If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
		# we don't do to allow resuming.
		save_total_limit = self.args.save_total_limit
		if (
			self.state.best_model_checkpoint is not None
			and self.args.save_total_limit == 1
			and checkpoints_sorted[-1] != self.state.best_model_checkpoint
		):
			save_total_limit = 2
		"""

		number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)

		## NEW: sort checkpoints path
		if self.args.save_model_accord_to_metric:
			## sort according to metric (ascending for metric)
			checkpoints_sorted = [
				k for k, v in sorted(self._ckpt_eval_metric.items(),
									 key=lambda x: x[1],
									 reverse=False)
			]
			checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
		else:
			## sort according to loss (descending for loss)
			checkpoints_sorted = [
				k for k, v in sorted(self._ckpt_eval_loss.items(),
									 key=lambda x: x[1],
									 reverse=True)
			]

		#checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
		for checkpoint in checkpoints_to_be_deleted:
			logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
			shutil.rmtree(checkpoint)

			## NEW: remove the deleted ckpt
			del self._ckpt_eval_loss[checkpoint]
			if self.args.save_model_accord_to_metric:
				del self._ckpt_eval_metric[checkpoint]

		self.best_checkpoint_path = checkpoints_sorted[-1]

	def _save(self, output_dir: Optional[str] = None, state_dict=None):
		"""
		Modification:
			- Also record model_args and data_args
		"""
		# If we are executing this function, we are the process zero, so we don't check for that.
		output_dir = output_dir if output_dir is not None else self.args.output_dir
		os.makedirs(output_dir, exist_ok=True)
		logger.info(f"Saving model checkpoint to {output_dir}")
		# Save a trained model and configuration using `save_pretrained()`.
		# They can then be reloaded using `from_pretrained()`
		if not isinstance(self.model, PreTrainedModel):
			if isinstance(unwrap_model(self.model), PreTrainedModel):
				if state_dict is None:
					state_dict = self.model.state_dict()
				unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
			else:
				logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
				if state_dict is None:
					state_dict = self.model.state_dict()
				torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
		else:
			self.model.save_pretrained(output_dir, state_dict=state_dict)
		if self.tokenizer is not None:
			self.tokenizer.save_pretrained(output_dir)

		# Good practice: save your training arguments together with the trained model
		torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
		## New
		torch.save(self.data_args , os.path.join(output_dir, "data_args.bin"))
		torch.save(self.model_args, os.path.join(output_dir, "model_args.bin"))
	
	def compute_loss(self, model, inputs, return_outputs=False):
		"""
		Modification:
			- compute loss for both detector & generator
			- enable weighted-loss during training
		"""
		if self.label_smoother is not None and "labels" in inputs:
			labels_det = inputs.get("labels_det")
			labels_gen = inputs.get("labels_gen")
		else:
			labels_det = None
			labels_gen = None

		## Forward
		outputs = model(**inputs)

		## Save past state if it exists
		## TODO: this needs to be fixed and made cleaner later.
		if self.args.past_index >= 0:
			self._past = outputs[self.args.past_index]
		
		## Label Smoother, currently not used!
		#if labels_det is not None and labels_gen is not None:
		#	loss_det = self.label_smoother(outputs_det, labels_det)
		#	loss_gen = self.label_smoother(outputs_gen, labels_gen)
		#else:
		## We don't use .loss here since the model may return tuples instead of ModelOutput.
		loss_det = outputs["loss_det"] #if isinstance(outputs, dict) else outputs_det[0]
		loss_gen = outputs["loss"] #if isinstance(outputs, dict) else outputs_gen[0]

		loss = loss_det + loss_gen

		return (loss, outputs) if return_outputs else loss

	def evaluation_loop(
		self,
		dataloader: DataLoader,
		description: str,
		prediction_loss_only: Optional[bool] = None,
		ignore_keys: Optional[List[str]] = None,
		metric_key_prefix: str = "eval",
	) -> EvalLoopOutput:
		"""
		Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

		Works both with or without labels.
		"""
		args = self.args

		prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

		# if eval is called w/o train init deepspeed here
		if args.deepspeed and not self.deepspeed:

			# XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
			# from the checkpoint eventually
			deepspeed_engine, _, _ = deepspeed_init(
				self, num_training_steps=0, resume_from_checkpoint=None, inference=True
			)
			self.model = deepspeed_engine.module
			self.model_wrapped = deepspeed_engine
			self.deepspeed = deepspeed_engine

		model = self._wrap_model(self.model, training=False)

		# if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
		# while ``train`` is running, cast it to the right dtype first and then put on device
		if not self.is_in_train:
			if args.fp16_full_eval:
				model = model.to(dtype=torch.float16, device=args.device)
			elif args.bf16_full_eval:
				model = model.to(dtype=torch.bfloat16, device=args.device)

		batch_size = dataloader.batch_size

		logger.info(f"***** Running {description} *****")
		if has_length(dataloader.dataset):
			logger.info(f"  Num examples = {self.num_examples(dataloader)}")
		else:
			logger.info("  Num examples: Unknown")
		logger.info(f"  Batch size = {batch_size}")

		model.eval()

		self.callback_handler.eval_dataloader = dataloader
		# Do this before wrapping.
		eval_dataset = dataloader.dataset

		if is_torch_tpu_available():
			dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

		if args.past_index >= 0:
			self._past = None

		# Initialize containers
		# losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
		losses_host_det, losses_host_gen = None, None
		labels_host_det, labels_host_gen = None, None
		preds_host_det , preds_host_gen  = None, None
		# losses/preds/labels on CPU (final containers)
		all_losses_det, all_losses_gen = None, None
		all_labels_det, all_labels_gen = None, None
		all_preds_det , all_preds_gen  = None, None
		# Will be useful when we have an iterable dataset so don't know its length.
		all_summary = []
		n_extracted_attack = 0

		observed_num_examples = 0
		# Main evaluation loop
		for step, inputs in enumerate(dataloader):
			# Update the observed num examples
			observed_batch_size = find_batch_size(inputs)
			if observed_batch_size is not None:
				observed_num_examples += observed_batch_size
				# For batch samplers, batch_size is not known by the dataloader in advance.
				if batch_size is None:
					batch_size = observed_batch_size

			# Prediction step
			loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
			
			loss_det  , loss_gen   = loss
			labels_det, labels_gen = labels
			logits_det, logits_gen, summary_tokens, n_ext_atk = logits
			
			all_summary.append(summary_tokens)
			if n_ext_atk is not None:
				n_extracted_attack = n_extracted_attack + n_ext_atk

			if is_torch_tpu_available():
				xm.mark_step()

			# Update containers on host
			if loss_det is not None:
				losses_det = self._nested_gather(loss_det.repeat(batch_size))
				losses_gen = self._nested_gather(loss_gen.repeat(batch_size))
				losses_host_det = losses_det if losses_host_det is None else torch.cat((losses_host_det, losses_det), dim=0)
				losses_host_gen = losses_gen if losses_host_gen is None else torch.cat((losses_host_gen, losses_gen), dim=0)
			if labels_det is not None:
				labels_det = self._pad_across_processes(labels_det)
				labels_gen = self._pad_across_processes(labels_gen)
				labels_det = self._nested_gather(labels_det)
				labels_gen = self._nested_gather(labels_gen)
				labels_host_det = labels_det if labels_host_det is None else nested_concat(labels_host_det, labels_det, padding_index=-100)
				labels_host_gen = labels_gen if labels_host_gen is None else nested_concat(labels_host_gen, labels_gen, padding_index=-100)
			if logits_det is not None:
				logits_det = self._pad_across_processes(logits_det)
				logits_gen = self._pad_across_processes(logits_gen)
				logits_det = self._nested_gather(logits_det)
				logits_gen = self._nested_gather(logits_gen)
				if self.preprocess_logits_for_metrics is not None:
					logits_det = self.preprocess_logits_for_metrics(logits_det, labels_det)
					logits_gen = self.preprocess_logits_for_metrics(logits_gen, labels_gen)
				preds_host_det = logits_det if preds_host_det is None else nested_concat(preds_host_det, logits_det, padding_index=-100)
				preds_host_gen = logits_gen if preds_host_gen is None else nested_concat(preds_host_gen, logits_gen, padding_index=-100)
			self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

			# Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
			if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
				if losses_host_det is not None:
					losses_det = nested_numpify(losses_host_det)
					losses_gen = nested_numpify(losses_host_gen)
					all_losses_det = losses_det if all_losses_det is None else np.concatenate((all_losses_det, losses_det), axis=0)
					all_losses_gen = losses_gen if all_losses_gen is None else np.concatenate((all_losses_gen, losses_gen), axis=0)
				if preds_host_det is not None:
					logits_det = nested_numpify(preds_host_det)
					logits_gen = nested_numpify(preds_host_gen)
					all_preds_det = logits_det if all_preds_det is None else nested_concat(all_preds_det, logits_det, padding_index=-100)
					all_preds_gen = logits_gen if all_preds_gen is None else nested_concat(all_preds_gen, logits_gen, padding_index=-100)
				if labels_host_det is not None:
					labels_det = nested_numpify(labels_host_det)
					labels_gen = nested_numpify(labels_host_gen)
					all_labels_det = (labels_det if all_labels_det is None else nested_concat(all_labels_det, labels_det, padding_index=-100))
					all_labels_gen = (labels_gen if all_labels_gen is None else nested_concat(all_labels_gen, labels_gen, padding_index=-100))

				# Set back to None to begin a new accumulation
				losses_host_det, preds_host_det, labels_host_det = None, None, None
				losses_host_gen, preds_host_gen, labels_host_gen = None, None, None

		if args.past_index and hasattr(self, "_past"):
			# Clean the state at the end of the evaluation loop
			delattr(self, "_past")

		# Gather all remaining tensors and put them back on the CPU
		if losses_host_det is not None:
			losses_det = nested_numpify(losses_host_det)
			losses_gen = nested_numpify(losses_host_gen)
			all_losses_det = losses_det if all_losses_det is None else np.concatenate((all_losses_det, losses_det), axis=0)
			all_losses_gen = losses_gen if all_losses_gen is None else np.concatenate((all_losses_gen, losses_gen), axis=0)
		if preds_host_det is not None:
			logits_det = nested_numpify(preds_host_det)
			logits_gen = nested_numpify(preds_host_gen)
			all_preds_det = logits_det if all_preds_det is None else nested_concat(all_preds_det, logits_det, padding_index=-100)
			all_preds_gen = logits_gen if all_preds_gen is None else nested_concat(all_preds_gen, logits_gen, padding_index=-100)
		if labels_host_det is not None:
			labels_det = nested_numpify(labels_host_det)
			labels_gen = nested_numpify(labels_host_gen)
			all_labels_det = labels_det if all_labels_det is None else nested_concat(all_labels_det, labels_det, padding_index=-100)
			all_labels_gen = labels_gen if all_labels_gen is None else nested_concat(all_labels_gen, labels_gen, padding_index=-100)

		# Number of samples
		if has_length(eval_dataset):
			num_samples = len(eval_dataset)
		# The instance check is weird and does not actually check for the type, but whether the dataset has the right
		# methods. Therefore we need to make sure it also has the attribute.
		elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
			num_samples = eval_dataset.num_examples
		else:
			num_samples = observed_num_examples

		# Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
		# samplers has been rounded to a multiple of batch_size, so we truncate.
		if all_losses_det is not None:
			all_losses_det = all_losses_det[:num_samples]
			all_losses_gen = all_losses_gen[:num_samples]
		if all_preds_det is not None:
			all_preds_det = nested_truncate(all_preds_det, num_samples)
			all_preds_gen = nested_truncate(all_preds_gen, num_samples)
		if all_labels_det is not None:
			all_labels_det = nested_truncate(all_labels_det, num_samples)
			all_labels_gen = nested_truncate(all_labels_gen, num_samples)

		# Metrics!
		if self.compute_metrics is not None and all_preds_det is not None and all_labels_det is not None:
			metrics = self.compute_metrics(EvalPrediction(predictions=(all_preds_det, all_preds_gen), label_ids=(all_labels_det, all_labels_gen)))
		else:
			metrics = {}

		# To be JSON-serializable, we need to remove numpy types or zero-d tensors
		metrics = denumpify_detensorize(metrics)

		if all_losses_det is not None:
			metrics[f"{metric_key_prefix}_loss_det"] = all_losses_det.mean().item()
			metrics[f"{metric_key_prefix}_loss_gen"] = all_losses_gen.mean().item()

		# Prefix all keys with metric_key_prefix + '_'
		for key in list(metrics.keys()):
			if not key.startswith(f"{metric_key_prefix}_"):
				metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
				
		## NEW: summary
		all_summary = torch.cat(all_summary, dim=0).type(torch.LongTensor) if summary_tokens is not None else all_summary

		return EvalLoopOutput(predictions=(all_preds_det, all_preds_gen, all_summary, n_extracted_attack), label_ids=(all_labels_det, all_labels_gen), metrics=metrics, num_samples=num_samples)

	def prediction_step(
		self,
		model: nn.Module,
		inputs: Dict[str, Union[torch.Tensor, Any]],
		prediction_loss_only: bool,
		ignore_keys: Optional[List[str]] = None,
	) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
		"""
		Perform an evaluation step on `model` using `inputs`.

		Subclass and override to inject custom behavior.

		Args:
			model (`nn.Module`):
				The model to evaluate.
			inputs (`Dict[str, Union[torch.Tensor, Any]]`):
				The inputs and targets of the model.

				The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
				argument `labels`. Check your model's documentation for all accepted arguments.
			prediction_loss_only (`bool`):
				Whether or not to return the loss only.

		Return:
			Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
			labels (each being optional).
		"""

		if not self.args.predict_with_generate or prediction_loss_only:
			return super().prediction_step(
				model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
			)

		## Currently by detector labels
		has_labels = "labels_det" in inputs #"labels" in inputs
		inputs = self._prepare_inputs(inputs)

		# XXX: adapt synced_gpus for fairscale as well
		gen_kwargs = {
			#"max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
			#"num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
			"max_length": self.data_args.max_tweet_length, 
			"num_beams": 1, 
			"synced_gpus": True if is_deepspeed_zero3_enabled() else False,
			"output_hidden_states": True if "labels_gen" not in inputs else None, 
			"output_scores": True if "labels_gen" not in inputs else None, 
			"return_dict_in_generate": True if "labels_gen" not in inputs else None
		}

		if "attention_mask" in inputs:
			gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

		# prepare generation inputs
		# some encoder-decoder models can have varying encder's and thus
		# varying model input names
		if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
			generation_inputs = inputs[self.model.encoder.main_input_name]
		else:
			generation_inputs = inputs[self.model.main_input_name]

		generated_tokens = self.model.generate(
			generation_inputs,
			**gen_kwargs,
		)

		## NEW: get decoder_hidden_states when testing
		if not isinstance(generated_tokens, torch.Tensor):
			encoder_hidden_states = generated_tokens["encoder_hidden_states"]
			decoder_hidden_states = generated_tokens["decoder_hidden_states"]
			generated_tokens = generated_tokens["sequences"]

			## Set decoder_hidden_states as part of inputs
			inputs["decoder_hidden_states"] = decoder_hidden_states

		# in case the batch is shorter than max length, the output should be padded
		if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
			generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

		with torch.no_grad():
			with self.autocast_smart_context_manager():
				outputs = model(**inputs)
			if has_labels:
				if self.label_smoother is not None: ## Currently not used!
					loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
				else:
					loss_det = outputs["loss_det"].mean().detach()#(outputs["loss_det"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
					loss_gen = outputs["loss"].mean().detach() if "loss" in outputs else loss_det
			else:
				loss_det = None
				loss_gen = None

		if self.args.prediction_loss_only:
			return (loss, None, None)

		if has_labels:
			labels_det = inputs["labels_det"]
			labels_gen = inputs["labels_gen"] if "labels_gen" in inputs else labels_det
			if labels_gen.shape[-1] < gen_kwargs["max_length"]:
				labels_gen = self._pad_tensors_to_max_len(labels_gen, gen_kwargs["max_length"])
		else:
			labels_gen = None
			labels_det = None

		## NEW
		summary_tokens = outputs["summary_tokens"] if "summary_tokens" in outputs else None
		#n_extracted_attack = outputs["n_extracted_attack"] if "n_extracted_attack" in outputs else None
		n_extracted_attack = outputs["n_ext_adv"] if "n_ext_adv" in outputs else None

		return ((loss_det, loss_gen), (outputs["logits_det"], generated_tokens, summary_tokens, n_extracted_attack), (labels_det, labels_gen))


