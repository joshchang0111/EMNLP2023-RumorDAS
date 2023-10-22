"""
This code is developed based on:
	https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
"""

import ipdb
import os
import shutil
from collections import Counter
from typing import Optional, Union, Dict, Any, Callable, Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from transformers.utils import logging

from others.args import DataTrainingArguments, ModelArguments

logger = logging.get_logger(__name__)

class CustomTrainer(Trainer):
	"""Customized Trainer"""
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
		Trainer.__init__(
			self, model, args, data_collator, train_dataset, 
			eval_dataset, tokenizer, model_init, compute_metrics, 
			callbacks, optimizers, preprocess_logits_for_metrics
		)

		self.data_args  = data_args
		self.model_args = model_args

		## Override
		if self.args.task_type == "train_detector":
			self.label_names = ["labels_det"]

		## For best model saving
		self._ckpt_eval_loss = {}
		if self.args.save_model_accord_to_metric:
			self._ckpt_eval_metric = {}
		self.best_metrics = None
		self.best_checkpoint_path = None

		## Make specified parameters trainable
		freeze_only = None
		if self.args.task_type == "train_detector":
			freeze_only = "summarizer"
			self._freeze_specified_params(self.model, freeze_only=freeze_only)

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
		
	def _freeze_specified_params(self, model, freeze_only=None):
		if freeze_only is not None:
			names = freeze_only.split()
			for freeze_name in names:
				for name, sub_module in model.named_modules():
					if name.startswith(freeze_name):
						for param in sub_module.parameters():
							param.requires_grad = False
	
	def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
		"""
		Modification:
			- record current eval loss / metric for best model saving
		"""
		if self.control.should_log:
			#if is_torch_tpu_available():
			#	xm.mark_step()

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
			print("Loss D  : {:7.4f}".format(metrics["eval_loss"]))
			print("Accuracy: {:7.4f}, F1-Macro: {:7.4f}".format(metrics["eval_accuracy"], metrics["eval_f1_macro"]))

			## NEW: record metric
			if self.args.save_model_accord_to_metric:
				self._cur_eval_metric = metrics["eval_f1_macro"]
			self._cur_eval_loss = metrics["eval_loss"]

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

