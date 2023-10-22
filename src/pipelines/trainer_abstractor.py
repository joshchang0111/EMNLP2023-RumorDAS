"""
This code is developed based on:
	- https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
	- https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_seq2seq.py
"""

import os
import ipdb
import shutil
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
		self.label_names = ["labels"] #["labels_det", "labels_gen"]

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
			print("\n***** Evaluation Results *****")
			#print("Epoch  [{:7.4f}], Loss   : {:7.4f}, PPL    : {:7.4f}".format(metrics["epoch"], metrics["eval_loss"], metrics["perplexity"]))
			print("Epoch  [{:7.4f}], Loss   : {:7.4f}".format(metrics["epoch"], metrics["eval_loss"]))
			print("ROUGE-1: {:7.4f}, ROUGE-2: {:7.4f}, ROUGE-L: {:7.4f}".format(metrics["eval_rouge1"], metrics["eval_rouge2"], metrics["eval_rougeL"]))

			## NEW: record metric
			## -> currently based on the detector's performance
			if self.args.save_model_accord_to_metric:
				self._cur_eval_metric = metrics["eval_rouge1"]
			self._cur_eval_loss = metrics["eval_loss"]

			best_rouge1 = 0 if self.best_metrics is None else self.best_metrics["eval_rouge1"]
			if metrics["eval_rouge1"] > best_rouge1:
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
	
	def prediction_step(
		self,
		model: nn.Module,
		inputs: Dict[str, Union[torch.Tensor, Any]],
		prediction_loss_only: bool,
		ignore_keys: Optional[List[str]] = None,
	) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
		"""
		Modification:
			- Add a new argument `redundancy_weights` as part of encoder's input during prediction phase.
		"""

		if not self.args.predict_with_generate or prediction_loss_only:
			return super().prediction_step(
				model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
			)

		has_labels = "labels" in inputs
		inputs = self._prepare_inputs(inputs)

		# XXX: adapt synced_gpus for fairscale as well
		if not self.args.do_train and self.args.do_eval:
			self._max_length = self.data_args.max_target_length

		gen_kwargs = {
			"max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
			"min_length": self.data_args.min_target_length if self.data_args.min_target_length is not None else None, 
			"num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
			"synced_gpus": True if is_deepspeed_zero3_enabled() else False,
		}

		## NEW: add a argument as part of encoder's inputs when generating
		gen_kwargs["redundancy_weights"] = inputs.get("redundancy_weights", None)

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
		# in case the batch is shorter than max length, the output should be padded
		if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
			generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

		with torch.no_grad():
			with self.autocast_smart_context_manager():
				outputs = model(**inputs)
			if has_labels:
				if self.label_smoother is not None:
					loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
				else:
					loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
			else:
				loss = None

		if self.args.prediction_loss_only:
			return (loss, None, None)

		if has_labels:
			labels = inputs["labels"]
			if labels.shape[-1] < gen_kwargs["max_length"]:
				labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
		else:
			labels = None

		return (loss, generated_tokens, labels)

