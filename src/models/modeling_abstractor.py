import ipdb
from typing import List, Optional, Tuple, Union

from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

import transformers
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.models.bart.configuration_bart import BartConfig
from transformers.modeling_outputs import Seq2SeqLMOutput

## Self-defined
from .modeling_bart import BartModelWithEmbeddingRedundancyReduction
#from .modeling_filter import TransformerAutoEncoder

class BartForAbstractiveResponseSummarization(BartForConditionalGeneration):
	"""
	Response Abstractor (RA) with response relevance weights.
	A subclass of `BartForConditionalGeneration`.
	
	Modification:
		- Add a new argument `redundancy_weights` as part of encoder's inputs.
	"""
	def __init__(self, config: BartConfig):
		super().__init__(config)

		## Override (self.model: BartModel) to `BartModelWithEmbeddingRedundancyReduction`
		self.model = BartModelWithEmbeddingRedundancyReduction(config)

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
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, Seq2SeqLMOutput]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
			config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
			(masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

		Returns:
		"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		if labels is not None:
			if use_cache:
				logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
			use_cache = False
			if decoder_input_ids is None and decoder_inputs_embeds is None:
				decoder_input_ids = shift_tokens_right(
					labels, self.config.pad_token_id, self.config.decoder_start_token_id
				)

		outputs = self.model(
			input_ids,
			attention_mask=attention_mask,
			decoder_input_ids=decoder_input_ids,
			encoder_outputs=encoder_outputs,
			decoder_attention_mask=decoder_attention_mask,
			head_mask=head_mask,
			decoder_head_mask=decoder_head_mask,
			cross_attn_head_mask=cross_attn_head_mask,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			redundancy_weights=redundancy_weights, ## NEW: Redundancy Weights
			decoder_inputs_embeds=decoder_inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

		masked_lm_loss = None
		if labels is not None:
			loss_fct = nn.CrossEntropyLoss()
			masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

		if not return_dict:
			output = (lm_logits,) + outputs[1:]
			return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

		return Seq2SeqLMOutput(
			loss=masked_lm_loss,
			logits=lm_logits,
			past_key_values=outputs.past_key_values,
			decoder_hidden_states=outputs.decoder_hidden_states,
			decoder_attentions=outputs.decoder_attentions,
			cross_attentions=outputs.cross_attentions,
			encoder_last_hidden_state=outputs.encoder_last_hidden_state,
			encoder_hidden_states=outputs.encoder_hidden_states,
			encoder_attentions=outputs.encoder_attentions,
		)

class RobertaForExtractiveResponseSummarization(RobertaPreTrainedModel):
	"""
	Response Extractor (RExt) with RoBERTa as base model
	Unsupervised extractive summarization is performed to select representative responses using KMeans clustering.
	"""
	_keys_to_ignore_on_load_unexpected = [r"pooler"]
	_keys_to_ignore_on_load_missing = [r"position_ids"]

	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels

		self.roberta = RobertaModel(config, add_pooling_layer=False)
		classifier_dropout = (
			config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
		)
		#self.dropout = nn.Dropout(classifier_dropout)
		#self.classifier = nn.Linear(config.hidden_size, config.num_labels)

		# Initialize weights and apply final processing
		self.post_init()

	def init_args_modules(self, data_args, model_args, training_args, tokenizer=None, summarizer=None):
		self.data_args = data_args
		self.model_args = model_args
		self.training_args = training_args
		self.tokenizer = tokenizer

		self.extract_ratio = 0.25 if self.model_args.extract_ratio is None else self.model_args.extract_ratio

	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None, ## Used to guide first token index
		attention_mask: Optional[torch.FloatTensor] = None,
		token_type_ids: Optional[torch.LongTensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		head_mask: Optional[torch.FloatTensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	):
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		outputs = self.roberta(
			input_ids=None, 
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

		## Unsupervised extractive summarization ##
		## using response embedding clustering   ##
		pseudo_gen_id = torch.zeros((input_ids.shape[0], self.data_args.max_tweet_length)).to(self.device)
		pseudo_gen_id[:, 0] = self.tokenizer.sep_token_id
		new_input_ids = torch.cat((input_ids, pseudo_gen_id), dim=1)

		max_tree_length = int(new_input_ids.shape[1] / self.data_args.max_tweet_length)
		batch_size = input_ids.shape[0]
		
		## Separately cluster each sample in a batch
		extracted_response_idxs = []
		for batch_idx in range(batch_size):
			ori_tree_length = 0 ## Tree length without generated response
			for i in range(self.data_args.max_tree_length):
				if input_ids[batch_idx][i * self.data_args.max_tweet_length] != self.tokenizer.pad_token_id:
					ori_tree_length = ori_tree_length + 1

			## Average token embeddings as response embeddings
			response_idx = torch.LongTensor(range(max_tree_length)).to(self.device)
			response_idx = response_idx.view(1, -1).T.repeat(1, self.data_args.max_tweet_length).view(-1)
			response_emb = scatter_mean(sequence_output[batch_idx], response_idx, dim=0)
			response_emb = torch.cat((response_emb[1:ori_tree_length], response_emb[-1:max_tree_length])) ## Get ori. responses + gen. response
			response_emb = response_emb.cpu().detach().numpy()

			if response_emb.size != 0:
				n_clusters = torch.ceil((response_emb.shape[0]) * torch.as_tensor(self.extract_ratio)).long().item()
				kmeans = KMeans(n_clusters=n_clusters, random_state=0)
				distances = kmeans.fit_transform(response_emb)
			
				## Find the point closest to centroids
				extracted_response_idx = [-1] ## For source post
				for cluster_idx in range(n_clusters):
					extracted_response_idx.append(distances[:, cluster_idx].argmin())
				extracted_response_idx = list(set(extracted_response_idx))
				extracted_response_idx.sort()
				extracted_response_idx = torch.LongTensor(extracted_response_idx) + 1
				extracted_response_idx[extracted_response_idx == ori_tree_length] = max_tree_length - 1 ## Correct idx for gen. response
				extracted_one_hot = torch.zeros(max_tree_length)
				extracted_one_hot[extracted_response_idx] = 1
				extracted_response_idxs.append(extracted_one_hot)
			
			else: ## No response
				extracted_one_hot = torch.ones(max_tree_length)
				extracted_response_idxs.append(extracted_one_hot)

		extracted_response_idxs = torch.stack(extracted_response_idxs).to(self.device)
		extracted_response_mask = extracted_response_idxs.view(
			batch_size, max_tree_length, 1).repeat(1, 1, self.data_args.max_tweet_length).view(batch_size, -1)

		n_extracted_attack = extracted_response_idxs[:, -1].sum().long().item()

		return extracted_response_mask, extracted_response_idxs, n_extracted_attack
'''
class ResponseExtractor(nn.Module):
	"""Extract responses based on reconstruction loss from a pre-trained autoencoder"""
	def __init__(self, data_args, model_args, training_args):
		super(ResponseExtractor, self).__init__()
		self.data_args = data_args
		self.model_args = model_args
		self.training_args = training_args

		self.extract_type = self.model_args.extract_type
		self.extract_ratio = 0.25 if self.model_args.extract_ratio is None else self.model_args.extract_ratio

		if self.extract_type == "autoencoder":
			self.model = TransformerAutoEncoder()
			#ckpt_path = "{}/{}/{}/{}/{}.pt".format(
			#ckpt_path = "{}/{}/{}/{}/{}_bart.pt".format(
			ckpt_path = "{}/{}/{}/{}/{}_rd.pt".format(
				training_args.output_root, 
				data_args.dataset_name, 
				self.extract_type, 
				data_args.fold, 
				self.extract_type
			)
			self.model.load_state_dict(torch.load(ckpt_path), strict=False)

		#elif self.extract_type == "multi_ae":
		#	self.extractor = ...
		#	self.abstractor = ...

	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None, ## Used to guide first token index
		attention_mask: Optional[torch.FloatTensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None
	):
		batch_size = inputs_embeds.shape[0]
		if self.extract_type == "autoencoder":
			batch_tree_lens, response_embeds, response_masks = [], [], []
			for batch_idx in range(inputs_embeds.shape[0]):
				ori_tree_length = 0 ## Tree length without generated response
				for i in range(self.data_args.max_tree_length):
					if input_ids[batch_idx][i * self.data_args.max_tweet_length] != self.tokenizer.pad_token_id:
						ori_tree_length = ori_tree_length + 1

				embeds = [inputs_embeds[batch_idx][i * self.data_args.max_tweet_length:(i + 1) * self.data_args.max_tweet_length] for i in range(1, ori_tree_length)]
				embeds.append(inputs_embeds[batch_idx][self.data_args.max_tree_length * self.data_args.max_tweet_length:]) ## generated response
				masks = [attention_mask[batch_idx][i * self.data_args.max_tweet_length:(i + 1) * self.data_args.max_tweet_length] for i in range(1, ori_tree_length)]
				masks.append(attention_mask[batch_idx][self.data_args.max_tree_length * self.data_args.max_tweet_length:])

				response_embeds.extend(embeds)
				response_masks.extend(masks)
				batch_tree_lens.append(ori_tree_length + 1 - 1) ## + generated response, - source post

			response_embeds = torch.stack(response_embeds)
			response_masks = torch.stack(response_masks)
			batch_tree_lens = torch.LongTensor(batch_tree_lens)

			anomaly_scores = self.model(
				attn_mask=response_masks,
				inputs_embeds=response_embeds
			)
			anomaly_scores = anomaly_scores.sum(dim=1) ## Sum up the loss of each response
			score_idx_sort = torch.argsort(anomaly_scores)

			## Separately process each thread
			extracted_response_mask, extracted_response_idxs = [], []
			max_tree_length = int(inputs_embeds.shape[1] / self.data_args.max_tweet_length)
			for batch_idx in range(inputs_embeds.shape[0]):
				rank_idx = score_idx_sort[score_idx_sort < batch_tree_lens[batch_idx]]
				rank_idx = rank_idx - rank_idx.min() ## Make each sample start from 0

				n_ext = torch.ceil(batch_tree_lens[batch_idx] * self.extract_ratio).long()
				ext_idx = rank_idx[:n_ext]
				ext_idx = ext_idx + 1 ## Considering source post
				extracted_response_idx = torch.zeros(max_tree_length).long()
				extracted_response_idx[0] = 1 ## Source post
				extracted_response_idx[ext_idx] = 1
				extracted_response_idxs.append(extracted_response_idx)

			extracted_response_idxs = torch.stack(extracted_response_idxs).to(inputs_embeds.device)
			extracted_response_mask = extracted_response_idxs.view(
				batch_size, max_tree_length, 1).repeat(1, 1, self.data_args.max_tweet_length).view(batch_size, -1)
			n_extracted_attack = extracted_response_idxs[:, -1].sum().long().item()

			return extracted_response_mask, extracted_response_idxs, n_extracted_attack
			## extracted_response_mask: [8, 512]
			## extracted_response_idxs: [8, 16]

		#elif self.extract_type == "multu_ae_abs":
'''