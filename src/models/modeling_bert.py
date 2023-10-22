from transformers.models.bert.modeling_bert import BertModel

## Self-defined
from .generation_utils import generate_with_grad

class BertModelWithResponseSummarization(BertModel):
	"""
	A subclass of `BertModel`, used in `BertForRumorDetection`.
	Modification:
		- Override forward method for response summarization.
		- Add two new methods `response_summarization` & `get_gen_hidden_states_from_tuple`.
	"""
	def __init__(self, config, add_pooling_layer=True, summarizer=None):
		super().__init__(config, add_pooling_layer)

	def init_args_modules(self, data_args, model_args, training_args, summarizer=None):
		self.data_args = data_args
		self.model_args = model_args
		self.training_args = training_args
		self.summarizer = summarizer
		if self.summarizer is not None:
			bound_method = generate_with_grad.__get__(self.summarizer, self.summarizer.__class__)
			setattr(self.summarizer, "generate_with_grad", bound_method)

	def forward(
		self,
		input_ids: Optional[torch.Tensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		token_type_ids: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.Tensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		inputs_embeds: Optional[torch.Tensor] = None,
		encoder_hidden_states: Optional[torch.Tensor] = None,
		encoder_attention_mask: Optional[torch.Tensor] = None,
		past_key_values: Optional[List[torch.FloatTensor]] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions]:
		r"""
		encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
			Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
			the model is configured as a decoder.
		encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
			the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

			- 1 for tokens that are **not masked**,
			- 0 for tokens that are **masked**.
		past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
			Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

			If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
			don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
			`decoder_input_ids` of shape `(batch_size, sequence_length)`.
		use_cache (`bool`, *optional*):
			If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
			`past_key_values`).
		"""
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		if self.config.is_decoder:
			use_cache = use_cache if use_cache is not None else self.config.use_cache
		else:
			use_cache = False

		if input_ids is not None and inputs_embeds is not None:
			raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
		elif input_ids is not None:
			input_shape = input_ids.size()
		elif inputs_embeds is not None:
			input_shape = inputs_embeds.size()[:-1]
		else:
			raise ValueError("You have to specify either input_ids or inputs_embeds")

		batch_size, seq_length = input_shape
		device = input_ids.device if input_ids is not None else inputs_embeds.device

		# past_key_values_length
		past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

		if attention_mask is None:
			attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

		if token_type_ids is None:
			if hasattr(self.embeddings, "token_type_ids"):
				buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
				buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
				token_type_ids = buffered_token_type_ids_expanded
			else:
				token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

		# We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
		# ourselves in which case we just need to make it broadcastable to all heads.
		extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

		# If a 2D or 3D attention mask is provided for the cross-attention
		# we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
		if self.config.is_decoder and encoder_hidden_states is not None:
			encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
			encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
			if encoder_attention_mask is None:
				encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
			encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
		else:
			encoder_extended_attention_mask = None

		# Prepare head mask if needed
		# 1.0 in head_mask indicate we keep the head
		# attention_probs has shape bsz x n_heads x N x N
		# input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
		# and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
		head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

		embedding_output = self.embeddings(
			input_ids=input_ids,
			position_ids=position_ids,
			token_type_ids=token_type_ids,
			inputs_embeds=inputs_embeds,
			past_key_values_length=past_key_values_length,
		)

		## Use summarizer or not
		if self.summarizer is not None:
			

		encoder_outputs = self.encoder(
			embedding_output,
			attention_mask=extended_attention_mask,
			head_mask=head_mask,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_extended_attention_mask,
			past_key_values=past_key_values,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		sequence_output = encoder_outputs[0]
		pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

		if not return_dict:
			return (sequence_output, pooled_output) + encoder_outputs[1:]

		return BaseModelOutputWithPoolingAndCrossAttentions(
			last_hidden_state=sequence_output,
			pooler_output=pooled_output,
			past_key_values=encoder_outputs.past_key_values,
			hidden_states=encoder_outputs.hidden_states,
			attentions=encoder_outputs.attentions,
			cross_attentions=encoder_outputs.cross_attentions,
		)

	def response_summarization(
		self, 
		attention_mask, 
		inputs_embeds
	):
		"""Response summarization"""
		
		## Ignore source post, only take responses as input
		response_inputs_embeds  =  inputs_embeds[:, self.data_args.max_tweet_length:, :]
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
				attention_mask=response_attention_mask,
				inputs_embeds=response_inputs_embeds,
				**summ_kwargs
			)
		else:
			summarizer_outputs = self.summarizer.generate(
				attention_mask=response_attention_mask,
				inputs_embeds=response_inputs_embeds,
				**summ_kwargs
			)

		## Pad summary_tokens to data_args.max_tweet_length!
		summary_tokens = summarizer_outputs["sequences"]
		pad_batch = torch.randn((summary_tokens.shape[0], self.data_args.max_tweet_length - summary_tokens.shape[1])).to(self.model.device)
		torch.full((summary_tokens.shape[0], 32 - summary_tokens.shape[1]), self.model.config.pad_token_id, out=pad_batch)
		summary_tokens = torch.cat((summary_tokens, pad_batch), dim=1)

		summ_hidden_states = self.get_gen_hidden_states_from_tuple(summarizer_outputs["decoder_hidden_states"])
		return summary_tokens, summ_hidden_states

	def get_gen_hidden_states_from_tuple(self, decoder_hidden_states):
		"""Convert decoder_hidden_states of type tuple into torch tensor."""
		bos_embedding = self.model.shared.weight[self.config.bos_token_id]
		pad_embedding = self.model.shared.weight[self.config.pad_token_id]
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