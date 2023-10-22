import ipdb
import streamlit as st
from typing import List, Optional

import torch
import torch.nn as nn

import transformers
from transformers.models.bart.modeling_bart import BartForConditionalGeneration, shift_tokens_right
from transformers.models.bart.configuration_bart import BartConfig
from transformers.modeling_outputs import BaseModelOutput

## Self-defined
from .generation_utils import generate_with_grad
from .modeling_outputs import (
	BaseModelOutputWithEmbedding, 
	Seq2SeqWithSequenceClassifierOutput, 
	RumorDetectorOutput
)
from .modeling_gcn import GCNForClassification
from .modeling_bart import BartModelWithOutputEmbeddings
from .response_summarizer import ResponseSummarizer

class BartForRumorDetectionAndResponseGeneration(BartForConditionalGeneration):
	"""
	Rumor detector along with response generator (attacker).
	A subclass of `BartForConditionalGeneration`.
	Two branches:
		- `BartEncoder` as detector  -> classify rumor's veracity
		- `BartDecoder` as generator -> generate adversarial response
	New module:
		- Another `BartForConditionalGeneration` as response summarizer
	"""
	def __init__(self, config: BartConfig):
		super().__init__(config)
		
		## Override (self.model: BartModel) to `BartModelWithOutputEmbeddings`
		self.model = BartModelWithOutputEmbeddings(config)

	def init_args_modules(self, data_args, model_args, training_args, add_pooling_layer=True, tokenizer=None, demo=False):
		"""Initialize other arguments and modules after loading BART checkpoint."""
		self.tokenizer = tokenizer
		self.data_args = data_args
		self.model_args = model_args
		self.training_args = training_args
		self.num_det_labels = data_args.num_labels
		self.max_tree_length = self.data_args.max_tree_length #int(self.tokenizer.model_max_length / self.data_args.max_tweet_length)

		self.loss_weight = None

		## Use adversarial response as part of encoder's input
		self.gen_flag = True

		## Use Monte-Carlo dropout or not
		self.mc_dropout = False

		## Main modules
		self.encoder = self.get_encoder()
		self.decoder = self.get_decoder()

		## Detector modules
		self.detector_head = GCNForClassification(
			self.data_args, 
			self.model_args, 
			self.training_args, 
			self.config.hidden_size, 
			self.num_det_labels
		)

		## Setup response summarizer
		self.sum_flag = False
		self.summarizer = None
		if not demo: ## if demo: load summarizer outside of this function
			if self.model_args.abstractor_name_or_path is not None or \
				self.model_args.extractor_name_or_path is not None:
				self.summarizer = ResponseSummarizer(
					self.tokenizer, 
					self.data_args, 
					self.model_args, 
					self.training_args
				)

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
			decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
			decoder_hidden_states: Optional[torch.FloatTensor] = None, ## generated hidden states of decoder during testing phase
			tree_lens: Optional[torch.LongTensor] = None, ## Number of nodes each tree
			td_edges: Optional[torch.LongTensor] = None, ## Top-down  edge_index
			bu_edges: Optional[torch.LongTensor] = None, ## Bottom-up edge_index
			td_edges_gen: Optional[torch.LongTensor] = None, ## Ground-truth td edge_index for generated response
			bu_edges_gen: Optional[torch.LongTensor] = None, ## Ground-truth bu edge_index for generated response
			labels_det: Optional[torch.LongTensor] = None, ## labels for classification
			labels_gen: Optional[torch.LongTensor] = None, ## labels for generation
			use_cache: Optional[bool] = None,
			output_attentions: Optional[bool] = None,
			output_hidden_states: Optional[bool] = None,
			abstractor_kwargs: Optional[dict] = None, ## For demo: control min. / max. abstractor length
			return_dict: Optional[bool] = None,
		):
		"""
		Training flow following `AARD (ACL 2021 Findings) by Yunzhu Song`
		For training both stage 1 & stage 2.

		with torch.no_grad():
		|->	encoder: encode (source post + previous responses)
		Then forward:
		|-> decoder: generate next response
		|-> encoder: encode (source post + previous responses + next response)
		"""
		
		## Turn on dropout layers of summarizer if using Monte-Carlo dropout
		if self.model_args.use_mc_dropout_for_summarizer and self.summarizer is not None:
			for m in self.modules():
				m.training = True

		if labels_gen is not None:
			if use_cache:
				logger.warning("The `use_cache` argument is changed to `False` since `labels_gen` is provided.")
			use_cache = False
			if decoder_input_ids is None and decoder_inputs_embeds is None:
				decoder_input_ids = shift_tokens_right(
					labels_gen, self.config.pad_token_id, self.config.decoder_start_token_id
				)

		## Encode then Decode (copy from BartModel) ##
		## different to other models, Bart automatically creates decoder_input_ids from
		## input_ids if no decoder_input_ids are provided
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
			with torch.no_grad(): ## IMPORTANT!
				encoder_outputs = self.encoder(
					input_ids=input_ids,
					attention_mask=attention_mask,
					head_mask=head_mask,
					inputs_embeds=inputs_embeds,
					output_attentions=output_attentions,
					output_hidden_states=output_hidden_states,
					return_dict=return_dict,
				)
		## If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
		elif return_dict and not isinstance(encoder_outputs, BaseModelOutputWithEmbedding):
			encoder_outputs = BaseModelOutputWithEmbedding(
				last_hidden_state=encoder_outputs[0],
				hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
				attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
				#output_embeds=encoder_outputs[3] if len(encoder_outputs) > 3 else None
				embed_tok=encoder_outputs[2] if len(encoder_outputs) > 3 else None, 
				embed_pos=encoder_outputs[2] if len(encoder_outputs) > 4 else None, 
				embed_out=encoder_outputs[2] if len(encoder_outputs) > 5 else None
			)

		######################
		## Generator Branch ##
		######################
		## decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
		decoder_outputs, masked_lm_loss, lm_logits = None, None, None
		if decoder_hidden_states is None:
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
			lm_logits = self.lm_head(decoder_outputs[0]) + self.final_logits_bias

			## Generator loss
			if labels_gen is not None:
				loss_fct = nn.CrossEntropyLoss()
				masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels_gen.view(-1))

		#####################
		## Detector Branch ##
		#####################
		## Encode then classify ##
		loss_det, logits_det = None, None
		summary_tokens, n_ext_adv = None, None
		filt_ext_idxs, clus_ext_idxs, clus_ids = None, None, None
		if labels_det is not None:
			tree_lens, inputs_embeds, attention_mask, td_edges, bu_edges = \
			self.arrange_inputs_for_gen_response(
				tree_lens=tree_lens, 
				decoder_outputs=decoder_outputs, 
				decoder_hidden_states=decoder_hidden_states, 
				attention_mask=attention_mask, 
				inputs_embeds=encoder_outputs["embed_tok"], 
				td_edges=td_edges, 
				bu_edges=bu_edges, 
				td_edges_gen=td_edges_gen, 
				bu_edges_gen=bu_edges_gen
			)

			############################
			## Response Summarization ##
			############################
			if self.sum_flag:
				summarizer_outputs = self.summarizer(
					self.config, 
					self.model.shared, 
					tree_lens=tree_lens, 
					inputs_embeds=inputs_embeds, 
					attention_mask=attention_mask, 
					td_edges=td_edges, 
					bu_edges=bu_edges, 
					abstractor_kwargs=abstractor_kwargs
				)

				inputs_embeds = summarizer_outputs["inputs_embeds"]
				attention_mask = summarizer_outputs["attention_mask"]
				tree_lens = summarizer_outputs["tree_lens"]
				td_edges = summarizer_outputs["td_edges"]
				bu_edges = summarizer_outputs["bu_edges"]
				n_ext_adv = summarizer_outputs["n_ext_adv"]
				summary_tokens = summarizer_outputs["summary_tokens"] if "summary_tokens" in summarizer_outputs else None
				filt_ext_idxs = summarizer_outputs["filt_ext_idxs"] if "filt_ext_idxs" in summarizer_outputs else None
				clus_ext_idxs = summarizer_outputs["clus_ext_idxs"] if "clus_ext_idxs" in summarizer_outputs else None
				clus_ids = summarizer_outputs["clus_ids"] if "clus_ids" in summarizer_outputs else None
			
			encoder_outputs = self.encoder(
				input_ids=None,
				attention_mask=attention_mask,
				head_mask=head_mask,
				inputs_embeds=inputs_embeds,
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=return_dict,
			)

			logits_det, loss_det = self.detector_head(
				hidden_states=encoder_outputs["last_hidden_state"], #inputs_embeds,
				attention_msk=attention_mask, 
				tree_lens=tree_lens, 
				td_edges=td_edges, 
				bu_edges=bu_edges, 
				labels=labels_det
			)

			## Untargeted attack during adv. stage 2
			if self.training_args.task_type == "train_adv_stage2" and self.training_args.attack_type == "untargeted":
				loss_det = -1 * loss_det

		return RumorDetectorOutput(
			## Response Generator
			loss=masked_lm_loss,
			logits=lm_logits,
			past_key_values=decoder_outputs.past_key_values if decoder_outputs is not None else None,
			decoder_hidden_states=decoder_outputs.hidden_states if decoder_outputs is not None else None,
			decoder_attentions=decoder_outputs.attentions if decoder_outputs is not None else None,
			cross_attentions=decoder_outputs.cross_attentions if decoder_outputs is not None else None,
			encoder_last_hidden_state=encoder_outputs.last_hidden_state,
			encoder_hidden_states=encoder_outputs.hidden_states,
			encoder_attentions=encoder_outputs.attentions,
			## Rumor Detector
			loss_det=loss_det,
			logits_det=logits_det, 
			## Response Summarizer
			summary_tokens=summary_tokens, 
			n_ext_adv=n_ext_adv, 
			filt_ext_idxs=filt_ext_idxs, ## Added for demo
			clus_ext_idxs=clus_ext_idxs, ## Added for demo
			clus_ids=clus_ids ## Added for demo
		)

	def arrange_inputs_for_gen_response(
		self, 
		tree_lens, 
		decoder_outputs, 
		decoder_hidden_states, 
		attention_mask: Optional[torch.Tensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		td_edges: Optional[torch.LongTensor] = None, ## Top-down  edge_index
		bu_edges: Optional[torch.LongTensor] = None, ## Bottom-up edge_index
		td_edges_gen: Optional[torch.LongTensor] = None, ## Ground-truth td edge_index for generated response
		bu_edges_gen: Optional[torch.LongTensor] = None, ## Ground-truth bu edge_index for generated response
	):
		"""
		This function do the following operations:
			1) Get generated hidden states and its mask
			2) Remove the padding nodes of `inputs_embeds` and `attention_mask` and 
			   append the generated hidden states, then pad to `max_tree_length`
			   Ex. [ooooo--x] -> [ooooox--]

			3) Attach the generated node to the source post if 
			   `td_edges_gen` and `bu_edges_gen` is not specified
		"""
		## 1)
		if decoder_outputs is not None: ## Training
			gen_hidden_states = decoder_outputs["last_hidden_state"]
			gen_attention_msk = torch.ones((attention_mask.shape[0], gen_hidden_states.shape[1])).to(self.model.device)
		else: ## Evaluation
			gen_hidden_states, gen_attention_msk = self.get_gen_hidden_states_from_tuple(decoder_hidden_states)
		
		if self.gen_flag:
			## 2)
			## Reshape thread sequences into nodes for convenience
			inputs_embeds = torch.stack(torch.split(inputs_embeds, self.data_args.max_tweet_length, dim=1), dim=1)
			attention_mask = torch.stack(torch.split(attention_mask, self.data_args.max_tweet_length, dim=1), dim=1)
			
			max_tree_length_before_add_gen_node = inputs_embeds.shape[1]
			max_tree_length = max_tree_length_before_add_gen_node + 1

			new_emb = []
			new_msk = []
			batch_size = inputs_embeds.shape[0]
			for batch_idx in range(batch_size):
				tree_len = tree_lens[batch_idx]
				tree_emb = inputs_embeds[batch_idx]
				tree_msk = attention_mask[batch_idx]
			
				gen_emb, gen_msk = None, None
				if self.gen_flag:
					gen_emb = gen_hidden_states[batch_idx:batch_idx + 1]
					gen_msk = gen_attention_msk[batch_idx:batch_idx + 1]
			
				## Remove padding and append generated response
				tree_emb = tree_emb[:tree_len] if not self.gen_flag else torch.cat((tree_emb[:tree_len], gen_emb), dim=0)
				tree_msk = tree_msk[:tree_len] if not self.gen_flag else torch.cat((tree_msk[:tree_len], gen_msk), dim=0)
			
				## Pad again
				pad_emb = torch.rand((max_tree_length - tree_emb.shape[0], tree_emb.shape[1], tree_emb.shape[2])).to(tree_emb.device)
				pad_msk = torch.zeros(pad_emb.shape[:-1]).to(tree_msk.device)
			
				tree_emb = torch.cat((tree_emb, pad_emb), dim=0).flatten(end_dim=1)
				tree_msk = torch.cat((tree_msk, pad_msk), dim=0).flatten(end_dim=1)
			
				new_emb.append(tree_emb)
				new_msk.append(tree_msk)
			
			tree_lens = tree_lens + 1
			new_emb = torch.stack(new_emb)
			new_msk = torch.stack(new_msk)
			
			## 3)
			td_edges = self.update_edges(td_edges, td_edges_gen, edge_type="td")
			bu_edges = self.update_edges(bu_edges, bu_edges_gen, edge_type="bu")

		else: ## self.gen_flag is False
			new_emb = inputs_embeds
			new_msk = attention_mask

		return tree_lens, new_emb, new_msk, td_edges, bu_edges

	def update_edges(self, edges, edges_gen, edge_type="td"):
		if edges is None: ## No edges are provide
			return None

		if self.training_args.task_type == "train_adv_stage2" or \
		   edges_gen is None: ## Evaluating
			## Always attach generated node to source post 
			## -> making train & test sets more compatible
			child_idx  = 1 if edge_type == "td" else 0
			parent_idx = 0 if edge_type == "td" else 1
			
			new_node_idxs = edges.flatten(start_dim=1).max(dim=1).values + 1
			new_node_idxs[new_node_idxs == 0] = 1 ## those without responses
			edges_gen = new_node_idxs.unsqueeze(-1).repeat(1, 2)
			edges_gen[:, parent_idx] = 0
			edges_gen = edges_gen.unsqueeze(-1)

		return torch.cat((edges, edges_gen), dim=2)

	def get_gen_hidden_states_from_tuple(self, decoder_hidden_states):
		"""Convert decoder_hidden_states of type tuple into torch tensor."""
		bos_embedding = self.model.shared.weight[self.config.bos_token_id]
		pad_embedding = self.model.shared.weight[self.config.pad_token_id]
		bos_embedding = bos_embedding.reshape(1, 1, -1).repeat(decoder_hidden_states[0][0].shape[0], 1, 1)
		pad_embedding = pad_embedding.reshape(1, 1, -1).repeat(decoder_hidden_states[0][0].shape[0], 1, 1)

		## Add start token
		gen_hidden_states = [bos_embedding]
		gen_attention_msk = torch.zeros(self.data_args.max_tweet_length, device=self.model.device)
		gen_attention_msk[0] = 1 ## bos token
		for token_idx in range(len(decoder_hidden_states)): ## Iterate through all tokens
			token_hidden_states = decoder_hidden_states[token_idx] ## Get token hidden states of all layers
			token_last_hidden_state = token_hidden_states[-1] ## Get last hidden state of current token
			gen_hidden_states.append(token_last_hidden_state)
			gen_attention_msk[token_idx + 1] = 1

		## Padding
		paddings = [pad_embedding] * (self.data_args.max_tweet_length - len(gen_hidden_states))
		gen_hidden_states.extend(paddings)
		gen_hidden_states = torch.cat(gen_hidden_states, dim=1)
		gen_attention_msk = gen_attention_msk.unsqueeze(0).repeat(gen_hidden_states.shape[0], 1)

		return gen_hidden_states, gen_attention_msk