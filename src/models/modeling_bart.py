import ipdb
import random
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.models.bart.modeling_bart import BartEncoder, BartModel, _expand_mask
from transformers.models.bart.configuration_bart import BartConfig
from transformers.modeling_outputs import (
	BaseModelOutput, 
	Seq2SeqModelOutput
)

## Self-defined
from .modeling_outputs import BaseModelOutputWithEmbedding

class BartModelWithEmbeddingRedundancyReduction(BartModel):
	"""
	Customized `BartModel` with embedding-based redundancy reduction.
	Only used when fine-tuning summarizer.

	Modification:
		- Add a new argument `redundancy_weights` to forward method and part of encoder's inputs
	"""
	def __init__(self, config: BartConfig):
		super().__init__(config)

		## Override encoder
		self.encoder = BartEncoderWithEmbeddingRedundancyReduction(config, self.shared)

	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		decoder_input_ids: Optional[torch.LongTensor] = None,
		decoder_attention_mask: Optional[torch.LongTensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		decoder_head_mask: Optional[torch.Tensor] = None,
		cross_attn_head_mask: Optional[torch.Tensor] = None,
		encoder_outputs: Optional[List[torch.FloatTensor]] = None,
		past_key_values: Optional[List[torch.FloatTensor]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		redundancy_weights: Optional[torch.FloatTensor] = None, ## NEW: Redundancy Weights
		decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, Seq2SeqModelOutput]:

		# different to other models, Bart automatically creates decoder_input_ids from
		# input_ids if no decoder_input_ids are provided
		if decoder_input_ids is None and decoder_inputs_embeds is None:
			if input_ids is None:
				raise ValueError(
					"If no `decoder_input_ids` or `decoder_inputs_embeds` are "
					"passed, `input_ids` cannot be `None`. Please pass either "
					"`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
				)

			decoder_input_ids = shift_tokens_right(
				input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
			)

		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		use_cache = use_cache if use_cache is not None else self.config.use_cache
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		if encoder_outputs is None:
			encoder_outputs = self.encoder(
				input_ids=input_ids,
				attention_mask=attention_mask,
				head_mask=head_mask,
				inputs_embeds=inputs_embeds,
				redundancy_weights=redundancy_weights, ## NEW: Redundancy weights
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=return_dict,
			)
		# If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
		elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
			encoder_outputs = BaseModelOutput(
				last_hidden_state=encoder_outputs[0],
				hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
				attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
			)

		# decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
		decoder_outputs = self.decoder(
			input_ids=decoder_input_ids,
			attention_mask=decoder_attention_mask,
			encoder_hidden_states=encoder_outputs[0],
			encoder_attention_mask=attention_mask,
			head_mask=decoder_head_mask,
			cross_attn_head_mask=cross_attn_head_mask,
			past_key_values=past_key_values,
			inputs_embeds=decoder_inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		if not return_dict:
			return decoder_outputs + encoder_outputs

		return Seq2SeqModelOutput(
			last_hidden_state=decoder_outputs.last_hidden_state,
			past_key_values=decoder_outputs.past_key_values,
			decoder_hidden_states=decoder_outputs.hidden_states,
			decoder_attentions=decoder_outputs.attentions,
			cross_attentions=decoder_outputs.cross_attentions,
			encoder_last_hidden_state=encoder_outputs.last_hidden_state,
			encoder_hidden_states=encoder_outputs.hidden_states,
			encoder_attentions=encoder_outputs.attentions,
		)

class BartEncoderWithEmbeddingRedundancyReduction(BartEncoder):
	"""
	Customized `BartEncoder` with embedding-based redundancy reduction.
	Only used when fine-tuning summarizer.

	Modification:
		- Add a new argument `redundancy_weights` to forward method
	"""
	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		redundancy_weights: Optional[torch.FloatTensor] = None, ## NEW: Redundancy Weights
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, BaseModelOutput]:
		r"""
		Args:
			input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
				Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
				provide it.

				Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
				[`PreTrainedTokenizer.__call__`] for details.

				[What are input IDs?](../glossary#input-ids)
			attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
				Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

				- 1 for tokens that are **not masked**,
				- 0 for tokens that are **masked**.

				[What are attention masks?](../glossary#attention-mask)
			head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
				Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

				- 1 indicates the head is **not masked**,
				- 0 indicates the head is **masked**.

			inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
				Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
				This is useful if you want more control over how to convert `input_ids` indices into associated vectors
				than the model's internal embedding lookup matrix.
			output_attentions (`bool`, *optional*):
				Whether or not to return the attentions tensors of all attention layers. See `attentions` under
				returned tensors for more detail.
			output_hidden_states (`bool`, *optional*):
				Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
				for more detail.
			return_dict (`bool`, *optional*):
				Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
		"""
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		# retrieve input_ids and inputs_embeds
		if input_ids is not None and inputs_embeds is not None:
			raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
		elif input_ids is not None:
			input_shape = input_ids.size()
			input_ids = input_ids.view(-1, input_shape[-1])
		elif inputs_embeds is not None:
			input_shape = inputs_embeds.size()[:-1]
		else:
			raise ValueError("You have to specify either input_ids or inputs_embeds")

		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

		## NEW: multiply each response with corresponding redundancy weights
		if redundancy_weights is not None:
			bs, tweet_len = input_shape[0], int(input_shape[1] / redundancy_weights.shape[1])
			redundancy_weights = redundancy_weights.view(bs, 1, -1).repeat(1, tweet_len, 1).transpose(1, 2).reshape(bs, -1)
			redundancy_weights = redundancy_weights.view(bs, input_shape[1], 1).repeat(1, 1, inputs_embeds.shape[-1])
			inputs_embeds = inputs_embeds * redundancy_weights

		embed_pos = self.embed_positions(input_shape)

		hidden_states = inputs_embeds + embed_pos
		hidden_states = self.layernorm_embedding(hidden_states)
		hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

		# expand attention_mask
		if attention_mask is not None:
			# [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
			attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

		encoder_states = () if output_hidden_states else None
		all_attentions = () if output_attentions else None

		# check if head_mask has a correct number of layers specified if desired
		if head_mask is not None:
			if head_mask.size()[0] != (len(self.layers)):
				raise ValueError(
					f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
				)

		for idx, encoder_layer in enumerate(self.layers):
			if output_hidden_states:
				encoder_states = encoder_states + (hidden_states,)
			# add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
			dropout_probability = random.uniform(0, 1)
			if self.training and (dropout_probability < self.layerdrop):  # skip the layer
				layer_outputs = (None, None)
			else:
				if self.gradient_checkpointing and self.training:

					def create_custom_forward(module):
						def custom_forward(*inputs):
							return module(*inputs, output_attentions)

						return custom_forward

					layer_outputs = torch.utils.checkpoint.checkpoint(
						create_custom_forward(encoder_layer),
						hidden_states,
						attention_mask,
						(head_mask[idx] if head_mask is not None else None),
					)
				else:
					layer_outputs = encoder_layer(
						hidden_states,
						attention_mask,
						layer_head_mask=(head_mask[idx] if head_mask is not None else None),
						output_attentions=output_attentions,
					)

				hidden_states = layer_outputs[0]

			if output_attentions:
				all_attentions = all_attentions + (layer_outputs[1],)

		if output_hidden_states:
			encoder_states = encoder_states + (hidden_states,)

		if not return_dict:
			return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
		return BaseModelOutput(
			last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
		)

## ------------------------------------------------------------------------
class BartModelWithOutputEmbeddings(BartModel):
	"""
	Customized `BartModel` with output embeddings.
	Used in `BartForRumorDetectionAndResponseGeneration`.
	"""
	def __init__(self, config: BartConfig):
		super().__init__(config)

		## Override encoder
		self.encoder = BartEncoderWithOutputEmbeddings(config, self.shared)

class BartEncoderWithOutputEmbeddings(BartEncoder):
	"""
	Customized `BartEncoder` with output embeddings.
	Used in `BartForRumorDetectionAndResponseGeneration`.
	"""
	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, BaseModelOutput]:
		r"""
		Args:
			input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
				Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
				provide it.
				Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
				[`PreTrainedTokenizer.__call__`] for details.
				[What are input IDs?](../glossary#input-ids)
			attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
				Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
				- 1 for tokens that are **not masked**,
				- 0 for tokens that are **masked**.
				[What are attention masks?](../glossary#attention-mask)
			head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
				Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
				- 1 indicates the head is **not masked**,
				- 0 indicates the head is **masked**.
			inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
				Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
				This is useful if you want more control over how to convert `input_ids` indices into associated vectors
				than the model's internal embedding lookup matrix.
			output_attentions (`bool`, *optional*):
				Whether or not to return the attentions tensors of all attention layers. See `attentions` under
				returned tensors for more detail.
			output_hidden_states (`bool`, *optional*):
				Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
				for more detail.
			return_dict (`bool`, *optional*):
				Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
		"""
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict
	
		# retrieve input_ids and inputs_embeds
		if input_ids is not None and inputs_embeds is not None:
			raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
		elif input_ids is not None:
			input_shape = input_ids.size()
			input_ids = input_ids.view(-1, input_shape[-1])
		elif inputs_embeds is not None:
			input_shape = inputs_embeds.size()[:-1]
		else:
			raise ValueError("You have to specify either input_ids or inputs_embeds")
	
		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
	
		embed_pos = self.embed_positions(input_shape)
	
		hidden_states = inputs_embeds + embed_pos
		hidden_states = self.layernorm_embedding(hidden_states)

		## NEW
		embed_out = hidden_states.clone()

		hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
	
		# expand attention_mask
		if attention_mask is not None:
			# [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
			attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)
	
		encoder_states = () if output_hidden_states else None
		all_attentions = () if output_attentions else None
	
		# check if head_mask has a correct number of layers specified if desired
		if head_mask is not None:
			if head_mask.size()[0] != (len(self.layers)):
				raise ValueError(
					f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
					f" {head_mask.size()[0]}."
				)
	
		for idx, encoder_layer in enumerate(self.layers):
			if output_hidden_states:
				encoder_states = encoder_states + (hidden_states,)
			# add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
			dropout_probability = random.uniform(0, 1)
			if self.training and (dropout_probability < self.layerdrop):  # skip the layer
				layer_outputs = (None, None)
			else:
				if self.gradient_checkpointing and self.training:
	
					def create_custom_forward(module):
						def custom_forward(*inputs):
							return module(*inputs, output_attentions)
	
						return custom_forward
	
					layer_outputs = torch.utils.checkpoint.checkpoint(
						create_custom_forward(encoder_layer),
						hidden_states,
						attention_mask,
						(head_mask[idx] if head_mask is not None else None),
					)
				else:
					layer_outputs = encoder_layer(
						hidden_states,
						attention_mask,
						layer_head_mask=(head_mask[idx] if head_mask is not None else None),
						output_attentions=output_attentions,
					)
	
				hidden_states = layer_outputs[0]
	
			if output_attentions:
				all_attentions = all_attentions + (layer_outputs[1],)
	
		if output_hidden_states:
			encoder_states = encoder_states + (hidden_states,)
	
		if not return_dict:
			return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
		return BaseModelOutputWithEmbedding(
			last_hidden_state=hidden_states, 
			hidden_states=encoder_states, 
			attentions=all_attentions, 
			#outputs_embeds=inputs_embeds, 
			embed_tok=inputs_embeds, 
			embed_pos=embed_pos, 
			embed_out=embed_out
		)