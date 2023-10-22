import ipdb
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss

from transformers.models.bart.modeling_bart import BartPretrainedModel, BartEncoder
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bert.modeling_bert import (
	BertPreTrainedModel, 
	BertPooler, 
	BertModel
)
from transformers.models.roberta.modeling_roberta import (
	RobertaPreTrainedModel,
	RobertaModel,
	RobertaClassificationHead
)
from transformers.modeling_outputs import SequenceClassifierOutput

## Self-defined
from .generation_utils import generate_with_grad
from .modeling_gcn import GCNForClassification

class BaseModelForRumorDetectionWithResponseSummarization:
	"""
	Base model class with 2 methods.
		- response_summarization
		- get_gen_hidden_states_from_tuple
	"""
	def __init__(self):
		pass

	def init_args_modules(self, data_args, model_args, training_args, tokenizer=None, add_pooling_layer=True, summarizer=None):
		"""Initialize other arguments and modules after loading pre-trained model checkpoint."""
		self.tokenizer = tokenizer
		self.data_args = data_args
		self.model_args = model_args
		self.training_args = training_args

		## Setup response summarizer
		self.summarizer = summarizer
		if self.summarizer is not None:
			bound_method = generate_with_grad.__get__(self.summarizer, self.summarizer.__class__)
			setattr(self.summarizer, "generate_with_grad", bound_method)

		if isinstance(self, BartEncoderForRumorDetection):
			self.num_det_labels = data_args.num_labels

			## Setup detector
			self.pooler = BertPooler(self.config) if add_pooling_layer else None
			self.dropout = nn.Dropout(self.config.classifier_dropout)
			self.detector_head = nn.Linear(self.config.hidden_size, self.num_det_labels)

	def response_summarization(
		self, 
		input_ids=None,
		attention_mask=None, 
		inputs_embeds=None
	):
		"""Response summarization"""
		
		## Ignore source post, only take responses as input
		response_input_ids, response_inputs_embeds = None, None
		if input_ids is not None:
			response_input_ids = input_ids[:, self.data_args.max_tweet_length:]
		elif inputs_embeds is not None:
			response_inputs_embeds = inputs_embeds[:, self.data_args.max_tweet_length:, :]
		else:
			raise ValueError("You need to specifiy either `input_ids` or `inputs_embeds`.")
		response_attention_mask = attention_mask[:, self.data_args.max_tweet_length:]

		summ_kwargs = {
			#"max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
			"max_length": self.data_args.max_tweet_length, 
			#"num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
			"num_beams": 1, 
			"output_hidden_states": True, 
			"return_dict_in_generate": True
		}

		if self.training_args.task_type == "train_adv_stage2" and self.training_args.do_train:
			## Enable back-propagation to generator
			summarizer_outputs = self.summarizer.generate_with_grad(
				input_ids=response_input_ids,
				attention_mask=response_attention_mask,
				inputs_embeds=response_inputs_embeds,
				**summ_kwargs
			)
		else:
			#summarizer_outputs = self.summarizer.generate(
			summarizer_outputs = self.summarizer.generate_with_grad(
				input_ids=response_input_ids,
				attention_mask=response_attention_mask,
				inputs_embeds=response_inputs_embeds,
				**summ_kwargs
			)

		## Pad summary_tokens to data_args.max_tweet_length!
		summary_tokens = summarizer_outputs["sequences"]
		pad_batch = torch.randn((summary_tokens.shape[0], self.data_args.max_tweet_length - summary_tokens.shape[1])).to(self.device)
		torch.full((summary_tokens.shape[0], 32 - summary_tokens.shape[1]), self.config.pad_token_id, out=pad_batch)
		summary_tokens = torch.cat((summary_tokens, pad_batch), dim=1)

		summ_hidden_states = self.get_gen_hidden_states_from_tuple(summarizer_outputs["decoder_hidden_states"])
		return summary_tokens, summ_hidden_states

	def get_gen_hidden_states_from_tuple(self, decoder_hidden_states):
		"""Convert decoder_hidden_states of type tuple into torch tensor."""
		if isinstance(self, BertForRumorDetection):
			#bos_token_id = 
			bos_embedding = self.bert.embeddings.word_embeddings.weight[self.config.bos_token_id]
			pad_embedding = self.bert.embeddings.word_embeddings.weight[self.config.bos_token_id]
		elif isinstance(self, RobertaForRumorDetection):
			bos_embedding = self.roberta.embeddings.word_embeddings.weight[self.config.bos_token_id]
			pad_embedding = self.roberta.embeddings.word_embeddings.weight[self.config.bos_token_id]
		elif isinstance(self, BartEncoderForRumorDetection):
			bos_embedding = self.shared.weight[self.config.bos_token_id]
			pad_embedding = self.shared.weight[self.config.pad_token_id]
		bos_embedding = bos_embedding.reshape(1, 1, -1).repeat(decoder_hidden_states[0][0].shape[0], 1, 1)
		pad_embedding = pad_embedding.reshape(1, 1, -1).repeat(decoder_hidden_states[0][0].shape[0], 1, 1)

		## Add start token
		gen_hidden_states = [bos_embedding]
		for token_idx in range(len(decoder_hidden_states)): ## Iterate through all tokens
			token_hidden_states = decoder_hidden_states[token_idx] ## Get token hidden states of all layers
			token_last_hidden_state = token_hidden_states[-1] ## Get last hidden state of current token
			gen_hidden_states.append(token_last_hidden_state)

		## Padding
		paddings = [pad_embedding] * (self.data_args.max_tweet_length - len(gen_hidden_states))
		gen_hidden_states.extend(paddings)
		gen_hidden_states = torch.cat(gen_hidden_states, dim=1)

		return gen_hidden_states

class BertForRumorDetection(BertPreTrainedModel, BaseModelForRumorDetectionWithResponseSummarization):
	"""
	Rumor detector with an optional response summarizer.
	A subclass of `BertPreTrainedModel`, copy and modified from `BertForSequenceClassification`.
	"""
	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels
		self.config = config

		self.bert = BertModel(config)
		classifier_dropout = (
			config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
		)
		self.dropout = nn.Dropout(classifier_dropout)
		self.classifier = nn.Linear(config.hidden_size, config.num_labels)

		# Initialize weights and apply final processing
		self.post_init()
	
	'''
	def init_args_modules(self, data_args, model_args, training_args, tokenizer=None, summarizer=None):
		"""Initialize other arguments and modules after loading BART checkpoint."""
		self.data_args = data_args
		self.model_args = model_args
		self.training_args = training_args

		self.tokenizer = tokenizer

		## Setup response summarizer
		self.summarizer = summarizer
		if self.summarizer is not None:
			bound_method = generate_with_grad.__get__(self.summarizer, self.summarizer.__class__)
			setattr(self.summarizer, "generate_with_grad", bound_method)
	'''

	def forward(
		self,
		input_ids: Optional[torch.Tensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		token_type_ids: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.Tensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		inputs_embeds: Optional[torch.Tensor] = None,
		labels_det: Optional[torch.Tensor] = None, ## Labels for detection
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, SequenceClassifierOutput]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
			Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
			config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
			`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
		"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		## Use summarizer or not
		if self.summarizer is not None:
			summary_tokens, summ_hidden_states = self.response_summarization(
				input_ids=input_ids,
				attention_mask=attention_mask
			)

			## Source post + summary
			input_ids = torch.cat((input_ids[:, :self.data_args.max_tweet_length], summary_tokens.long()), dim=1)
			attention_mask = torch.cat((
				attention_mask[:, :self.data_args.max_tweet_length],
				(summary_tokens.long() != self.config.pad_token_id).long()), dim=1)

		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		pooled_output = outputs[1]

		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)

		loss = None
		if labels_det is not None:
			if self.config.problem_type is None:
				if self.num_labels == 1:
					self.config.problem_type = "regression"
				elif self.num_labels > 1 and (labels_det.dtype == torch.long or labels_det.dtype == torch.int):
					self.config.problem_type = "single_label_classification"
				else:
					self.config.problem_type = "multi_label_classification"

			if self.config.problem_type == "regression":
				loss_fct = MSELoss()
				if self.num_labels == 1:
					loss = loss_fct(logits.squeeze(), labels_det.squeeze())
				else:
					loss = loss_fct(logits, labels_det)
			elif self.config.problem_type == "single_label_classification":
				loss_fct = CrossEntropyLoss()
				loss = loss_fct(logits.view(-1, self.num_labels), labels_det.view(-1))
			elif self.config.problem_type == "multi_label_classification":
				loss_fct = BCEWithLogitsLoss()
				loss = loss_fct(logits, labels_det)
		if not return_dict:
			output = (logits,) + outputs[2:]
			return ((loss,) + output) if loss is not None else output

		return SequenceClassifierOutput(
			loss=loss,
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)

class RobertaForRumorDetectionAndResponseSummarization(RobertaPreTrainedModel, BaseModelForRumorDetectionWithResponseSummarization):
	"""
	Rumor detector with an optional response summarizer.
	A subclass of `RobertaPreTrainedModel`, copy and modified from `RobertaForSequenceClassification`.
	"""
	_keys_to_ignore_on_load_missing = [r"position_ids"]

	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels
		self.config = config

		self.roberta = RobertaModel(config, add_pooling_layer=False)
		self.classifier = RobertaClassificationHead(config)

		# Initialize weights and apply final processing
		self.post_init()

	'''
	def init_args_modules(self, data_args, model_args, training_args, tokenizer=None, summarizer=None):
		"""Initialize other arguments and modules after loading BART checkpoint."""
		self.data_args = data_args
		self.model_args = model_args
		self.training_args = training_args

		self.tokenizer = tokenizer

		## Setup response summarizer
		self.summarizer = summarizer
		if self.summarizer is not None:
			bound_method = generate_with_grad.__get__(self.summarizer, self.summarizer.__class__)
			setattr(self.summarizer, "generate_with_grad", bound_method)
	'''

	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.FloatTensor] = None,
		token_type_ids: Optional[torch.LongTensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		head_mask: Optional[torch.FloatTensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		labels_det: Optional[torch.LongTensor] = None, ## labels for detection
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, SequenceClassifierOutput]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
			Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
			config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
			`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
		"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		## Use summarizer or not
		if self.summarizer is not None:
			summary_tokens, summ_hidden_states = self.response_summarization(
				input_ids=input_ids,
				attention_mask=attention_mask
			)

			## Source post + response + summary
			#inputs_embeds = torch.cat((inputs_embeds, summ_hidden_states), dim=1)
			#attention_mask = torch.cat(
			#	(attention_mask, torch.ones((summ_hidden_states.shape[0], summ_hidden_states.shape[1])).to(self.encoder.device)), dim=1)

			## Obtain RoBERTa's embedding
			#inputs_embeds = self.roberta.embeddings.word_embeddings(input_ids)

			## Source post + summary
			#inputs_embeds = torch.cat((inputs_embeds[:, :self.data_args.max_tweet_length, :], summ_hidden_states), dim=1)
			#attention_mask = torch.cat(
			#	(
			#		attention_mask[:, :self.data_args.max_tweet_length], 
			#		torch.ones((summ_hidden_states.shape[0], summ_hidden_states.shape[1])).to(self.device)
			#	), dim=1
			#)

			input_ids = torch.cat((input_ids[:, :self.data_args.max_tweet_length], summary_tokens.long()), dim=1)
			attention_mask = torch.cat((
				attention_mask[:, :self.data_args.max_tweet_length],
				(summary_tokens.long() != self.config.pad_token_id).long()), dim=1)

		outputs = self.roberta(
			input_ids=input_ids if inputs_embeds is None else None,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		sequence_output = outputs[0]
		logits = self.classifier(sequence_output)

		loss = None
		if labels_det is not None:
			if self.config.problem_type is None:
				if self.num_labels == 1:
					self.config.problem_type = "regression"
				elif self.num_labels > 1 and (labels_det.dtype == torch.long or labels_det.dtype == torch.int):
					self.config.problem_type = "single_label_classification"
				else:
					self.config.problem_type = "multi_label_classification"

			if self.config.problem_type == "regression":
				loss_fct = MSELoss()
				if self.num_labels == 1:
					loss = loss_fct(logits.squeeze(), labels_det.squeeze())
				else:
					loss = loss_fct(logits, labels_det)
			elif self.config.problem_type == "single_label_classification":
				loss_fct = CrossEntropyLoss()
				loss = loss_fct(logits.view(-1, self.num_labels), labels_det.view(-1))
			elif self.config.problem_type == "multi_label_classification":
				loss_fct = BCEWithLogitsLoss()
				loss = loss_fct(logits, labels_det)

		if not return_dict:
			output = (logits,) + outputs[2:]
			return ((loss,) + output) if loss is not None else output

		return SequenceClassifierOutput(
			loss=loss,
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)

class BartEncoderForRumorDetection(BartPretrainedModel, BaseModelForRumorDetectionWithResponseSummarization):
	"""
	Rumor detector with an optional response summarizer.
	A subclass of `BartPretrainedModel`, copy and modified from `BartEncoder`.
	"""
	def __init__(self, config: BartConfig):#, embed_tokens: Optional[nn.Embedding]=None):
		super().__init__(config)

		padding_idx, vocab_size = config.pad_token_id, config.vocab_size
		self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
		self.encoder = BartEncoder(config, self.shared)

		## Initialize weights and apply final processing
		self.post_init()

	'''
	def init_args_modules(self, data_args, model_args, training_args, tokenizer=None, add_pooling_layer=True, summarizer=None):
		"""Initialize other arguments and modules after loading BART checkpoint."""
		self.data_args = data_args
		self.model_args = model_args
		self.training_args = training_args
		self.num_det_labels = data_args.num_labels

		self.tokenizer = tokenizer

		## Setup detector
		self.pooler = BertPooler(self.config) if add_pooling_layer else None
		self.dropout = nn.Dropout(self.config.classifier_dropout)
		self.detector_head = nn.Linear(self.config.hidden_size, self.num_det_labels)
		
		## Setup response summarizer
		self.summarizer = summarizer
		if self.summarizer is not None:
			bound_method = generate_with_grad.__get__(self.summarizer, self.summarizer.__class__)
			setattr(self.summarizer, "generate_with_grad", bound_method)
	'''

	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		past_key_values: Optional[List[torch.FloatTensor]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		labels_det: Optional[torch.LongTensor] = None, ## labels for classification
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, SequenceClassifierOutput]:

		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		use_cache = use_cache if use_cache is not None else self.config.use_cache
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		## Use summarizer or not
		if self.summarizer is not None:
			## Obtain embedding first (copy from `BartEncoder`)
			#inputs_embeds = self.encoder.embed_tokens(input_ids) * self.encoder.embed_scale

			summary_tokens, summ_hidden_states = self.response_summarization(
				input_ids=input_ids,
				attention_mask=attention_mask, 
				#inputs_embeds=inputs_embeds
			)

			## Source post + response + summary
			#inputs_embeds = torch.cat((inputs_embeds, summ_hidden_states), dim=1)
			#attention_mask = torch.cat(
			#	(attention_mask, torch.ones((summ_hidden_states.shape[0], summ_hidden_states.shape[1])).to(self.encoder.device)), dim=1)

			## Source post + summary
			#inputs_embeds = torch.cat((inputs_embeds[:, :self.data_args.max_tweet_length, :], summ_hidden_states), dim=1)
			#attention_mask = torch.cat(
			#	(
			#		attention_mask[:, :self.data_args.max_tweet_length], 
			#		torch.ones((summ_hidden_states.shape[0], summ_hidden_states.shape[1])).to(self.device)
			#	), dim=1
			#)

			input_ids = torch.cat((input_ids[:, :self.data_args.max_tweet_length], summary_tokens.long()), dim=1)
			attention_mask = torch.cat((
				attention_mask[:, :self.data_args.max_tweet_length],
				(summary_tokens.long() != self.config.pad_token_id).long()), dim=1)

		encoder_outputs = self.encoder(
			input_ids=input_ids if inputs_embeds is None else None,
			attention_mask=attention_mask,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict
		)
		
		## Classification: take the hidden state of first token
		## [bs=4, seq_len=512, hidden_size=768] -> [bs=4, seq_len=1, hidden_size=768]
		sequence_output = encoder_outputs["last_hidden_state"][:, 0:1, :]
		pooled_output = self.pooler(sequence_output) if self.pooler is not None else sequence_output
		pooled_output = self.dropout(pooled_output)
		logits_det = self.detector_head(pooled_output)

		loss = None
		if labels_det is not None:
			loss_fct = nn.CrossEntropyLoss()
			loss_det = loss_fct(logits_det, labels_det)

		if not return_dict:
			output = (logits_det,) + encoder_outputs[2:]
			return ((loss_det,) + output) if loss_det is not None else output

		"""
		if self.training_args.task_type == "train_adv_stage2" and self.training_args.attack_type == "untargeted":
			loss_det = -1 * loss_det
		"""

		return SequenceClassifierOutput(
			loss=loss_det,
			logits=logits_det,
			hidden_states=encoder_outputs.hidden_states,
			attentions=encoder_outputs.attentions,
		)

## ================================================================================================================== ##
## ================================================================================================================== ##

class RobertaForRumorDetection(RobertaPreTrainedModel):
	"""
	Rumor detector using RoBERTa as transformer encoder.
	A subclass of `RobertaPreTrainedModel`, copy and modified from `RobertaForSequenceClassification`.
	"""
	_keys_to_ignore_on_load_missing = [r"position_ids"]

	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels
		self.config = config

		self.roberta = RobertaModel(config, add_pooling_layer=False)
		#self.classifier = RobertaClassificationHead(config)

		# Initialize weights and apply final processing
		self.post_init()

	def init_args_modules(self, data_args, model_args, training_args, add_pooling_layer=True, tokenizer=None, summarizer=None):
		"""Initialize other arguments and modules after loading BART checkpoint."""
		self.tokenizer = tokenizer
		self.data_args = data_args
		self.model_args = model_args
		self.training_args = training_args
		self.num_det_labels = data_args.num_labels

		self.loss_weight = None
		self.summarizer = summarizer

		## Detector modules
		self.detector_head = GCNForClassification(
			self.data_args, 
			self.model_args, 
			self.training_args, 
			self.config.hidden_size, 
			self.num_det_labels
		)

	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.FloatTensor] = None,
		token_type_ids: Optional[torch.LongTensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		head_mask: Optional[torch.FloatTensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		tree_lens: Optional[torch.LongTensor] = None, ## Number of nodes each tree
		td_edges: Optional[torch.LongTensor] = None, ## Top-down  edge_index
		bu_edges: Optional[torch.LongTensor] = None, ## Bottom-up edge_index
		labels_det: Optional[torch.LongTensor] = None, ## labels for detection
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, SequenceClassifierOutput]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
			Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
			config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
			`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
		"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		outputs = self.roberta(
			input_ids=input_ids if inputs_embeds is None else None,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		logits_det, loss_det = self.detector_head(
			hidden_states=outputs["last_hidden_state"], 
			attention_msk=attention_mask, 
			tree_lens=tree_lens, 
			td_edges=td_edges, 
			bu_edges=bu_edges, 
			labels=labels_det
		)

		if not return_dict:
			output = (logits_det,) + outputs[2:]
			return ((loss_det,) + output) if loss_det is not None else output

		return SequenceClassifierOutput(
			loss=loss_det,
			logits=logits_det,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)
		
