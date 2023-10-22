import ipdb
import streamlit as st
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

## Self-defined
from others.utils import mean_pooling

class TransformerAutoEncoder(nn.Module):
	def __init__(
		self, 
		d_z=100, 
		nhead=12, 
		d_model=768, 
		dropout=0.1,
		model_emb=None,  
		num_layers_enc=2, 
		num_layers_dec=2
	):
		super(TransformerAutoEncoder, self).__init__()
		self.d_z = d_z
		self.nhead = nhead
		self.d_model = d_model
		self.num_layers_enc = num_layers_enc
		self.num_layers_dec = num_layers_dec

		print("n_layer_enc: {}".format(self.num_layers_enc))
		print("n_layer_dec: {}".format(self.num_layers_dec))

		if model_emb is not None:
			if "roberta" in model_emb.__class__.__name__.lower():
				self.embedding_model = "roberta"
				self.embeddings = model_emb.roberta.embeddings
			elif "bart" in model_emb.__class__.__name__.lower():
				self.embedding_model = "bart"
				self.embed_scale = model_emb.model.encoder.embed_scale
				self.embed_tokens = model_emb.model.encoder.embed_tokens
				#self.embed_positions = model_emb.model.encoder.embed_positions
				#self.layernorm_embedding = model_emb.model.encoder.layernorm_embedding

		self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
		self.decoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)

		self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers_enc)
		self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=self.num_layers_dec) ## Same as encoder

		## TODO: Add activation function! ##
		self.proj_enc = nn.Linear(d_model, d_z)
		self.proj_dec = nn.Linear(d_z, d_model)

	def forward(self, input_ids=None, attn_mask=None, inputs_embeds=None):
		## Choose loss function
		loss_fct = nn.L1Loss() ## L1
		loss_fct = nn.MSELoss(reduction="none") ## L2

		## Get embeddings
		if inputs_embeds is None:
			if self.embedding_model == "roberta": 
				embedding_output = self.embeddings(input_ids=input_ids)
			elif self.embedding_model == "bart":
				## Use token embedding only
				#inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
				#embed_pos = self.embed_positions(input_ids.size())
				#embedding_output = inputs_embeds + embed_pos
				#embedding_output = self.layernorm_embedding(embedding_output)
				embedding_output = self.embed_tokens(input_ids) * self.embed_scale
		else:
			embedding_output = inputs_embeds

		encoder_outputs = self.encoder(embedding_output, src_key_padding_mask=attn_mask.T)
		z_enc = self.proj_enc(encoder_outputs)
		z_enc = torch.tanh(z_enc)
		z_dec = self.proj_dec(z_enc)
		z_dec = torch.tanh(z_dec)
		decoder_outputs = self.decoder(z_dec, src_key_padding_mask=attn_mask.T)

		## Mean pooling
		source_ = torch.mean(decoder_outputs, dim=1)
		target_ = torch.mean(embedding_output, dim=1)

		loss = loss_fct(source_, target_)
		return loss

class ResponseFilter(nn.Module):
	def __init__(self, tokenizer, data_args, model_args, training_args):
		super(ResponseFilter, self).__init__()

		self.tokenizer = tokenizer
		self.data_args = data_args
		self.model_args = model_args
		self.training_args = training_args
		self.num_layers = model_args.filter_layer_enc

		assert (model_args.filter_layer_enc == model_args.filter_layer_dec)

		self.model = TransformerAutoEncoder(
			num_layers_enc=model_args.filter_layer_enc, 
			num_layers_dec=model_args.filter_layer_dec
		)
		ckpt_path = "{}/{}/filter_{}/{}/autoencoder_rd_all.pt".format(
			self.training_args.output_root, 
			self.data_args.dataset_name, 
			self.num_layers, 
			self.data_args.fold
		)
		self.model.load_state_dict(torch.load(ckpt_path), strict=False)
		self.filter_ratio = 0.05 if self.model_args.filter_ratio is None else self.model_args.filter_ratio

	def forward(
		self, 
		tree_lens, 
		inputs_embeds: Optional[torch.FloatTensor] = None, 
		attention_mask: Optional[torch.FloatTensor] = None
	):
		batch_size = inputs_embeds.shape[0]

		## Iterate through each conversational thread and collect all response features
		batch_reply, batch_masks = [], []
		batch_src, batch_src_msk, batch_gen_idx = [], [], []
		for batch_idx in range(batch_size):
			tree_len = tree_lens[batch_idx]
			nodes = inputs_embeds[batch_idx][:tree_len]
			masks = attention_mask[batch_idx][:tree_len]

			batch_reply.append(nodes[1:])
			batch_masks.append(masks[1:])

			batch_src.append(nodes[0])
			batch_src_msk.append(masks[0])
			#batch_gen_idx.append(gen_idx)

		batch_reply = torch.cat(batch_reply, dim=0)
		batch_masks = torch.cat(batch_masks, dim=0)

		batch_src = torch.stack(batch_src)
		batch_src_msk = torch.stack(batch_src_msk)
		#batch_gen_idx = torch.LongTensor(batch_gen_idx)

		## Forward through response filter
		## Obtain anomaly score of each response
		anomaly_scores = self.model(
			attn_mask=batch_masks, 
			inputs_embeds=batch_reply
		)
		anomaly_scores = anomaly_scores.sum(dim=1) ## Sum up the loss of each response
		score_idx_sort = torch.argsort(anomaly_scores)

		## Separately process each conversational thread
		n_ext_adv, res_accum_idx = 0, 0
		ext_idxs, ext_mask = [], []
		new_tree_lens, new_inputs_embeds, new_attention_mask = [], [], []
		max_tree_length = int(self.tokenizer.model_max_length / self.data_args.max_tweet_length)
		for batch_idx in range(batch_size):
			n_response = tree_lens[batch_idx] - 1
			rank_idx = score_idx_sort[score_idx_sort < n_response]
			rank_idx = rank_idx - rank_idx.min() if rank_idx.numel() > 0 else rank_idx ## Make each sample start from 0

			n_ext = torch.floor(n_response * self.filter_ratio).long()
			n_ext = n_ext + 1 if n_ext == 0 else n_ext
			ext_idx = rank_idx[:n_ext]
			ext_idx = ext_idx + 1 ## Consider source post

			one_hot = torch.zeros(max_tree_length, dtype=torch.long)
			one_hot[0] = 1 ## source post
			one_hot[ext_idx] = 1
			ext_idxs.append(one_hot)

			"""
			## Calculate number of extracted attacks
			if one_hot[batch_gen_idx[batch_idx]] == 1:
				n_ext_adv = n_ext_adv + 1
			"""

			## Get new (after extraction) `inputs_embeds` & `attention_mask`! Ex. src+res0+res1+res2+gen
			src_emb = batch_src[batch_idx:batch_idx + 1]
			src_msk = batch_src_msk[batch_idx:batch_idx + 1]
			res_emb = batch_reply[res_accum_idx:res_accum_idx + n_response][ext_idx - 1] ## Extract
			res_msk = batch_masks[res_accum_idx:res_accum_idx + n_response][ext_idx - 1] ## Extract
			res_accum_idx = res_accum_idx + n_response

			## Concatenate source post with extracted responses
			tree_emb = torch.cat((src_emb, res_emb), dim=0)
			tree_msk = torch.cat((src_msk, res_msk), dim=0)

			new_tree_lens.append(tree_emb.shape[0])

			## Pad each `inputs_embeds` & `attention_mask` to max_tree_length
			pad_msk = torch.zeros(src_msk.shape, device=tree_emb.device)
			tree_emb = torch.cat((tree_emb, src_emb.repeat(max_tree_length - tree_emb.shape[0], 1, 1)), dim=0)
			tree_msk = torch.cat((tree_msk, pad_msk.repeat(max_tree_length - tree_msk.shape[0], 1)), dim=0) ## Note that transformers won't attend to pad nodes

			new_inputs_embeds.append(tree_emb)
			new_attention_mask.append(tree_msk)

		ext_idxs = torch.stack(ext_idxs).to(inputs_embeds.device)
		ext_mask = ext_idxs.view(batch_size, max_tree_length, 1).repeat(1, 1, self.data_args.max_tweet_length).view(batch_size, -1)
		
		new_tree_lens = torch.LongTensor(new_tree_lens).to(tree_lens.device)
		new_inputs_embeds = torch.stack(new_inputs_embeds)
		new_attention_mask = torch.stack(new_attention_mask)

		return ext_idxs, ext_mask, n_ext_adv, new_tree_lens, new_inputs_embeds, new_attention_mask


