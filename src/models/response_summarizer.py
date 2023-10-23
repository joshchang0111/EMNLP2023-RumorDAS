import ipdb
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import (
	AutoConfig, 
	AutoModelForSeq2SeqLM
)

## Self-defined
from .modeling_filter import ResponseFilter, TransformerAutoEncoder
from .modeling_clustering import ClusterModel
from .modeling_abstractor import BartForAbstractiveResponseSummarization
from others.utils import find_ckpt_dir, post_process_generative_model, mean_pooling

def load_abstractor(abstractor_name_or_path):
	config_abs = AutoConfig.from_pretrained(
		abstractor_name_or_path, 
		cache_dir=None, 
		revision="main", 
		use_auth_token=None
	)
	return BartForAbstractiveResponseSummarization.from_pretrained(
		abstractor_name_or_path, 
		from_tf=bool(".ckpt" in abstractor_name_or_path), 
		config=config_abs, 
		cache_dir=None, 
		revision="main", 
		use_auth_token=None
	)

class ResponseSummarizer(nn.Module):
	def __init__(self, tokenizer, data_args, model_args, training_args):
		super(ResponseSummarizer, self).__init__()

		self.tokenizer = tokenizer
		self.data_args = data_args
		self.model_args = model_args
		self.training_args = training_args

		self.extractor_name_or_path  = self.model_args.extractor_name_or_path
		self.abstractor_name_or_path = self.model_args.abstractor_name_or_path

		self.max_tree_length = int(self.tokenizer.model_max_length / self.data_args.max_tweet_length)

		self.output_type = model_args.summarizer_output_type
		if self.extractor_name_or_path is not None and self.abstractor_name_or_path is not None:
			self.summarizer_type = "extract_then_abstract"
		elif self.extractor_name_or_path is not None and self.abstractor_name_or_path is None:
			self.summarizer_type = "extract"
		elif self.extractor_name_or_path is None and self.abstractor_name_or_path is not None:
			self.summarizer_type = "abstract"
		
		## Whether to load extractor or not
		self.filter = None
		self.model_cluster = None
		if self.extractor_name_or_path is not None:
			## Build Response Filter
			if "filter" in self.extractor_name_or_path:
				print("Response Filter: TransformerAutoEncoder")
				self.filter = ResponseFilter(self.tokenizer, self.data_args, self.model_args, self.training_args)

			## Build Clustering Extractor
			if "kmeans" in self.extractor_name_or_path:
				print("Clustering Extractor Type: kmeans")
				self.model_cluster = ClusterModel(cluster_type="kmeans", num_clusters=self.model_args.num_clusters, extract_ratio=self.model_args.extract_ratio)

		## Whether to load abstractor or not
		self.abstractor = None
		if self.abstractor_name_or_path is not None:

			## Load trained SSRA (Setup path for SSRA)
			if "ssra" in self.abstractor_name_or_path.lower():
				print("Abstractor: Self-Supervised Response Abstractor ({})".format(self.abstractor_name_or_path))
				ckpt_path = "{}/{}/{}/{}".format(self.training_args.output_root, self.data_args.dataset_name, self.abstractor_name_or_path, self.data_args.fold)
				ckpt_path = "{}/{}".format(ckpt_path, find_ckpt_dir(ckpt_path))
				self.abstractor_name_or_path = ckpt_path
			else:
				print("Abstractor: {}".format(self.abstractor_name_or_path))

			#config_abs = AutoConfig.from_pretrained(
			#	self.abstractor_name_or_path, 
			#	cache_dir=None, 
			#	revision="main", 
			#	use_auth_token=None
			#)
			#self.abstractor = BartForAbstractiveResponseSummarization.from_pretrained(
			#	self.abstractor_name_or_path, 
			#	from_tf=bool(".ckpt" in self.abstractor_name_or_path), 
			#	config=config_abs, 
			#	cache_dir=None, 
			#	revision="main", 
			#	use_auth_token=None
			#)
			self.abstractor = load_abstractor(self.abstractor_name_or_path)
			self.abstractor = post_process_generative_model(self.data_args, self.model_args, self.abstractor)

	def forward(
		self, 
		config, 
		embedding_layer, 
		tree_lens, ## Number of nodes each tree
		attention_mask: Optional[torch.FloatTensor] = None, 
		inputs_embeds: Optional[torch.FloatTensor] = None, 
		td_edges: Optional[torch.LongTensor] = None, ## Top-down  edge index
		bu_edges: Optional[torch.LongTensor] = None,  ## Bottom-up edge index
		abstractor_kwargs: Optional[dict] = None
	):
		## Extractor only ##
		extractor_outputs = None
		if self.abstractor_name_or_path is None and \
			self.extractor_name_or_path is not None:
			return self.extract(
				tree_lens=tree_lens, 
				inputs_embeds=inputs_embeds, 
				attention_mask=attention_mask, 
				td_edges=td_edges, 
				bu_edges=bu_edges
			)

		## Abstractor only ##
		abstractor_outputs = None
		if self.abstractor_name_or_path is not None and \
			self.extractor_name_or_path is None:
			return self.abstract(
				tree_lens=tree_lens, 
				inputs_embeds=inputs_embeds, 
				attention_mask=attention_mask, 
				td_edges=td_edges, 
				bu_edges=bu_edges, 
				config=config, 
				embedding_layer=embedding_layer, 
				kwargs=abstractor_kwargs
			)

		## ResponseFilter + ClusterExtractor + SSRA-KMeans ##
		if self.abstractor_name_or_path is not None and \
			self.extractor_name_or_path is not None:
			return self.extract_then_abstract(
				tree_lens=tree_lens, 
				inputs_embeds=inputs_embeds, 
				attention_mask=attention_mask, 
				td_edges=td_edges, 
				bu_edges=bu_edges, 
				config=config, 
				embedding_layer=embedding_layer, 
				abstractor_kwargs=abstractor_kwargs
			)

	def abstract(
		self, 
		config, 
		embedding_layer,
		tree_lens=None, 
		attention_mask: Optional[torch.FloatTensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None, 
		td_edges: Optional[torch.LongTensor] = None, ## Top-down  edge index
		bu_edges: Optional[torch.LongTensor] = None, ## Bottom-up edge index
		kwargs: Optional[dict] = None
	):
		"""Generate only one abstractive summary (BART-base-SAMSum, LOO)"""
		## Ignore source post, only take responses as input
		response_inputs_embeds  =  inputs_embeds[:, self.data_args.max_tweet_length:, :]
		response_attention_mask = attention_mask[:, self.data_args.max_tweet_length:]

		abstractor_kwargs = {
			#"min_length": kwargs["min_length"] if kwargs is not None else None, 
			"min_length": 20, 
			"max_length": kwargs["max_length"] if kwargs is not None else self.data_args.max_tweet_length, 
			"num_beams": 1, 
			"output_hidden_states": True, 
			"return_dict_in_generate": True
		}

		abstractor_outputs = self.abstractor.generate(
			attention_mask=response_attention_mask, 
			inputs_embeds=response_inputs_embeds, 
			**abstractor_kwargs
		)

		summary_tokens = abstractor_outputs["sequences"]
		pad_batch = torch.randn((
			summary_tokens.shape[0], 
			self.data_args.max_tweet_length - summary_tokens.shape[1]
		)).to(response_inputs_embeds.device)
		torch.full((summary_tokens.shape[0], 32 - summary_tokens.shape[1]), config.pad_token_id, out=pad_batch)
		summary_tokens = torch.cat((summary_tokens, pad_batch), dim=1)

		summary_hidden_states, summary_attention_mask = self.get_gen_hidden_states_from_tuple(
			config, 
			embedding_layer, 
			abstractor_outputs["decoder_hidden_states"]
		)

		if self.summarizer_type == "abstract":
			## Combine abstractive summary with source post
			src_emb = inputs_embeds[:, :self.data_args.max_tweet_length]
			src_msk = attention_mask[:, :self.data_args.max_tweet_length]
			
			final_emb = torch.cat((src_emb, summary_hidden_states), dim=1)
			final_msk = torch.cat((src_msk, summary_attention_mask), dim=1)
			
			tree_lens[:] = 2
			
			## Graph Reconstruction
			td_edges = self.graph_reconstruction(td_edges=td_edges, abs_summ=summary_hidden_states.unsqueeze(1))
			bu_edges = self.graph_reconstruction(bu_edges=bu_edges, abs_summ=summary_hidden_states.unsqueeze(1))
			
			final_outputs = {
				"inputs_embeds": final_emb, 
				"attention_mask": final_msk, 
				"tree_lens": tree_lens, 
				"td_edges": td_edges, 
				"bu_edges": bu_edges, 
				"n_ext_adv": 0
			}
			return final_outputs

		elif self.summarizer_type == "extract_then_abstract":
			return summary_tokens, summary_hidden_states, summary_attention_mask

	def abstract_clusters(
		self, 
		config, 
		embedding_layer,
		tree_lens, 
		cluster_ids, 
		attention_mask: Optional[torch.FloatTensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None, 
		abstractor_kwargs: Optional[dict] = None
	):
		batch_size = inputs_embeds.shape[0]
		max_tree_length = int(self.tokenizer.model_max_length / self.data_args.max_tweet_length)

		## Prepare inputs
		emb_tree_ids, all_tree_emb, all_attn_msk = [], [], []
		for batch_idx in range(batch_size): ## For each tree
			tree_len = tree_lens[batch_idx]
			clus_ids = cluster_ids[batch_idx][:tree_len]
			tree_emb = inputs_embeds[batch_idx][:tree_len]
			attn_msk = attention_mask[batch_idx][:tree_len]

			for cid in set(clus_ids.tolist()): ## For each cluster
				if cid < 0: ## Ignore source post
					continue
				clus_emb = tree_emb[clus_ids == cid]
				clus_msk = attn_msk[clus_ids == cid]

				## Padding
				pad_msk = torch.zeros(attn_msk[:1].shape, device=tree_emb.device)
				clus_emb = torch.cat((clus_emb, tree_emb[0].repeat(max_tree_length - clus_emb.shape[0], 1, 1)), dim=0)
				clus_msk = torch.cat((clus_msk, pad_msk.repeat(max_tree_length - clus_msk.shape[0], 1)), dim=0)

				all_tree_emb.append(clus_emb.flatten(start_dim=0, end_dim=1))
				all_attn_msk.append(clus_msk.flatten())
				emb_tree_ids.append(batch_idx)

		all_tree_emb = torch.stack(all_tree_emb)
		all_attn_msk = torch.stack(all_attn_msk)
		emb_tree_ids = torch.LongTensor(emb_tree_ids)

		## Forward through abstractor
		summary_tokens, summary_hidden_states, summary_attention_mask = self.abstract(
			config, 
			embedding_layer,
			attention_mask=all_attn_msk, 
			inputs_embeds=all_tree_emb, 
			kwargs=abstractor_kwargs
		)

		## Gather summary for each tree
		new_states, new_masks, n_summary = [], [], []
		for batch_idx in range(batch_size):
			summary = summary_hidden_states[emb_tree_ids == batch_idx]
			sum_msk = summary_attention_mask[emb_tree_ids == batch_idx]
			#summary = summary.flatten(start_dim=0, end_dim=1)
			new_states.append(summary)
			new_masks.append(sum_msk)
			n_summary.append((emb_tree_ids == batch_idx).sum())
		
		abstractor_outputs = {
			"tree_lens_diff": n_summary, 
			"summary_tokens": summary_tokens, 
			"summary_hidden_states": new_states, ## Type: List
			"summary_attention_mask": new_masks  ## Type: List
		}
		return abstractor_outputs

	def extract(
		self, 
		tree_lens, 
		attention_mask: Optional[torch.FloatTensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None, 
		td_edges: Optional[torch.LongTensor] = None, ## Top-down  edge index
		bu_edges: Optional[torch.LongTensor] = None  ## Bottom-up edge index
	):
		if "filter" in self.extractor_name_or_path:
			extractor_outputs = self.extract_by_filter(
				tree_lens=tree_lens,
				attention_mask=attention_mask, 
				inputs_embeds=inputs_embeds, 
				return_flatten=True
			)
			td_edges = self.graph_reconstruction(td_edges=td_edges, ext_idxs=extractor_outputs["ext_idxs"])
			bu_edges = self.graph_reconstruction(bu_edges=bu_edges, ext_idxs=extractor_outputs["ext_idxs"])
			
			extractor_outputs["td_edges"] = td_edges
			extractor_outputs["bu_edges"] = bu_edges
		
		elif "kmeans" in self.extractor_name_or_path:
			extractor_outputs = self.extract_by_clustering(
				tree_lens=tree_lens, 
				attention_mask=attention_mask, 
				inputs_embeds=inputs_embeds, 
				return_flatten=True
			)
			td_edges = self.graph_reconstruction(td_edges=td_edges, ext_idxs=extractor_outputs["ext_idxs"])
			bu_edges = self.graph_reconstruction(bu_edges=bu_edges, ext_idxs=extractor_outputs["ext_idxs"])
			
			extractor_outputs["td_edges"] = td_edges
			extractor_outputs["bu_edges"] = bu_edges
			extractor_outputs["n_ext_adv"] = 0 ## TODO

		return extractor_outputs

	def extract_by_filter(
		self, 
		tree_lens, 
		attention_mask: Optional[torch.FloatTensor] = None, 
		inputs_embeds: Optional[torch.FloatTensor] = None, 
		return_flatten=False
	):
		"""
		In this function, note that source post and responses should be separately processed!
		"""
		## Reshape nodes
		if len(inputs_embeds.shape) != 4:
			inputs_embeds = torch.stack(torch.split(inputs_embeds, self.data_args.max_tweet_length, dim=1), dim=1)
			attention_mask = torch.stack(torch.split(attention_mask, self.data_args.max_tweet_length, dim=1), dim=1)

		ext_idxs, ext_mask, n_ext_adv, tree_lens, inputs_embeds, attention_mask = \
		self.filter(
			tree_lens=tree_lens, 
			inputs_embeds=inputs_embeds, 
			attention_mask=attention_mask
		)

		if return_flatten:
			inputs_embeds = inputs_embeds.flatten(start_dim=1, end_dim=2)
			attention_mask = attention_mask.flatten(start_dim=1, end_dim=2)

		filter_outputs = {
			"ext_idxs": ext_idxs, ## Extracted response indices in one-hot format
			"ext_mask": ext_mask, 
			"n_ext_adv": n_ext_adv, 
			"tree_lens": tree_lens, 
			"inputs_embeds": inputs_embeds, 
			"attention_mask": attention_mask
		}
		return filter_outputs
	
	def extract_by_clustering(
		self, 
		tree_lens, 
		attention_mask: Optional[torch.FloatTensor] = None, 
		inputs_embeds: Optional[torch.FloatTensor] = None, 
		return_flatten=False
	):
		"""
		inputs_embeds : shape=(8, 32, 32, 768)
		attention_mask: shape=(8, 32, 32)
		"""
		if len(inputs_embeds.shape) != 4:
			inputs_embeds = torch.stack(torch.split(inputs_embeds, self.data_args.max_tweet_length, dim=1), dim=1)
			attention_mask = torch.stack(torch.split(attention_mask, self.data_args.max_tweet_length, dim=1), dim=1)

		batch_size = inputs_embeds.shape[0]
		max_tree_length = int(self.tokenizer.model_max_length / self.data_args.max_tweet_length)

		ext_idxs, clus_ids = [], []
		new_tree_lens, new_inputs_embeds, new_attention_mask = [], [], []
		for batch_idx in range(batch_size): ## Separately cluster each tree in a batch
			tree_len = tree_lens[batch_idx]
			attn_msk = attention_mask[batch_idx][:tree_len]
			tree_emb = inputs_embeds[batch_idx][:tree_len]

			## Mean pooling
			node_feat = mean_pooling(tree_emb, attn_msk)

			## Cluster the responses
			cluster_ids, _, _, is_centroid = self.model_cluster(node_feat[1:], device=node_feat.device)

			## Extraction
			ext_idx = torch.cat((torch.LongTensor([ 1]).to(is_centroid.device), is_centroid)) ## Also extract source post
			clus_id = torch.cat((torch.LongTensor([-1]).to(cluster_ids.device), cluster_ids)) ## Assign source post to cluster -1

			tree_emb = tree_emb[ext_idx.bool()]
			attn_msk = attn_msk[ext_idx.bool()]
			new_tree_lens.append(tree_emb.shape[0])

			## Padding
			ext_idx = torch.cat((ext_idx, torch.zeros(max_tree_length - len(ext_idx), device=is_centroid.device)), dim=0).long()
			clus_id = torch.cat((clus_id, -1 * torch.ones(max_tree_length - len(clus_id), device=cluster_ids.device)), dim=0).long()
			
			pad_msk = torch.zeros(attn_msk[:1].shape, device=tree_emb.device)
			tree_emb = torch.cat((tree_emb, tree_emb[0].repeat(max_tree_length - tree_emb.shape[0], 1, 1)), dim=0)
			attn_msk = torch.cat((attn_msk, pad_msk.repeat(max_tree_length - attn_msk.shape[0], 1)), dim=0)
			
			ext_idxs.append(ext_idx)
			clus_ids.append(clus_id)
			new_inputs_embeds.append(tree_emb)
			new_attention_mask.append(attn_msk)

		new_inputs_embeds = torch.stack(new_inputs_embeds)
		new_attention_mask = torch.stack(new_attention_mask)

		if return_flatten:
			new_inputs_embeds = new_inputs_embeds.flatten(start_dim=1, end_dim=2)
			new_attention_mask = new_attention_mask.flatten(start_dim=1, end_dim=2)

		cluster_outputs = {
			"ext_idxs": torch.stack(ext_idxs), 
			"cluster_ids": torch.stack(clus_ids), 
			"inputs_embeds": new_inputs_embeds, 
			"attention_mask": new_attention_mask, 
			"tree_lens": torch.LongTensor(new_tree_lens).to(tree_lens.device)
		}
		return cluster_outputs

	def extract_then_abstract(
		self, 
		tree_lens, 
		attention_mask: Optional[torch.FloatTensor] = None, 
		inputs_embeds: Optional[torch.FloatTensor] = None, 
		td_edges: Optional[torch.LongTensor] = None, ## top-down  edge index
		bu_edges: Optional[torch.LongTensor] = None, ## bottom-up edge index
		config=None, 
		embedding_layer=None, 
		abstractor_kwargs: Optional[dict] = None
	):	
		if len(inputs_embeds.shape) != 4:
			inputs_embeds = torch.stack(torch.split(inputs_embeds, self.data_args.max_tweet_length, dim=1), dim=1)
			attention_mask = torch.stack(torch.split(attention_mask, self.data_args.max_tweet_length, dim=1), dim=1)

		#################################
		## 1st Stage - Response Filter ##
		#################################
		if self.filter is not None:
			filter_outputs = self.extract_by_filter(
				tree_lens=tree_lens, 
				attention_mask=attention_mask,
				inputs_embeds=inputs_embeds,
				return_flatten=False
			) ## filter_outputs.keys = dict_keys(['ext_idxs', 'ext_mask', 'n_ext_adv', 'tree_lens', 'inputs_embeds', 'attention_mask'])
			
			td_edges = self.graph_reconstruction(td_edges=td_edges, ext_idxs=filter_outputs["ext_idxs"])
			bu_edges = self.graph_reconstruction(bu_edges=bu_edges, ext_idxs=filter_outputs["ext_idxs"])

			tree_lens = filter_outputs["tree_lens"]
			inputs_embeds = filter_outputs["inputs_embeds"]
			attention_mask = filter_outputs["attention_mask"]

		######################################
		## 2nd Stage - Clustering Extractor ##
		######################################
		cluster_outputs = self.extract_by_clustering(
			tree_lens=tree_lens, 
			attention_mask=attention_mask, 
			inputs_embeds=inputs_embeds
		) ## cluster_outputs.keys = dict_keys(['ext_idxs', 'cluster_ids', 'inputs_embeds', 'attention_mask', 'tree_lens'])

		td_edges = self.graph_reconstruction(td_edges=td_edges, ext_idxs=cluster_outputs["ext_idxs"])
		bu_edges = self.graph_reconstruction(bu_edges=bu_edges, ext_idxs=cluster_outputs["ext_idxs"])

		if self.abstractor is None:
			## CARE (Cluster-Aware Response Extraction)
			final_outputs = {
				"inputs_embeds": cluster_outputs["inputs_embeds"], 
				"attention_mask": cluster_outputs["attention_mask"], 
				"tree_lens": cluster_outputs["tree_lens"], 
				"td_edges": td_edges, 
				"bu_edges": bu_edges, 
				"n_ext_adv": 0
			}

		############################
		## 3rd Stage - Abstractor ##
		############################
		abstractor_outputs = self.abstract_clusters(
			config=config, 
			embedding_layer=embedding_layer, 
			tree_lens=tree_lens, 
			cluster_ids=cluster_outputs["cluster_ids"], 
			attention_mask=attention_mask, 
			inputs_embeds=inputs_embeds, ## clustering is based on filter outputs
			abstractor_kwargs=abstractor_kwargs
		) ## abstractor_outputs.keys() = dict_keys(['tree_lens_diff', 'summary_tokens', 'summary_hidden_states'])

		if self.filter is None:
			## Combine source post with `abstractive summary` & `extractive summary`
			final_emb, final_msk, final_tree_lens, td_edges, bu_edges = [], [], [], [], []
			batch_size = inputs_embeds.shape[0]
			for batch_idx in range(batch_size):
			
				src_emb_ = cluster_outputs["inputs_embeds"][batch_idx][:1] ## (32, 32, 768)
				src_mask = cluster_outputs["attention_mask"][batch_idx][:1]
				ext_summ = cluster_outputs["inputs_embeds"][batch_idx] ## (32, 32, 768)
				ext_mask = cluster_outputs["attention_mask"][batch_idx]
				abs_summ = abstractor_outputs["summary_hidden_states"][batch_idx] ## (n_abs, 32, 768)
				abs_mask = abstractor_outputs["summary_attention_mask"][batch_idx] ## (n_abs, 32)

				pad_emb_, pad_mask, pad_len_ = None, None, self.model_args.num_clusters - (abs_summ.shape[0])
				if pad_len_ > 0:
					pad_emb_ = src_emb_
					pad_emb_ = torch.cat([pad_emb_] * pad_len_, dim=0)
					pad_mask = torch.Tensor([[0] * self.data_args.max_tweet_length])
					pad_mask = torch.cat([pad_mask] * pad_len_, dim=0).to(pad_emb_.device)

				## Adjust summarizer outupts
				if self.output_type is not None:
					if self.output_type == "ssra_only":
						total_emb = torch.cat((src_emb_, abs_summ), dim=0)
						total_msk = torch.cat((src_mask, abs_mask), dim=0)
						if pad_emb_ is not None:
							total_emb = torch.cat((total_emb, pad_emb_), dim=0)
							total_msk = torch.cat((total_msk, pad_mask), dim=0)
						total_emb = total_emb.flatten(start_dim=0, end_dim=1)
						total_msk = total_msk.flatten()
						tree_len = 1 + abstractor_outputs["tree_lens_diff"][batch_idx]
				else:
					total_emb = torch.cat((src_emb_, abs_summ, ext_summ[1 + abs_summ.shape[0]:]), dim=0)
					total_emb = total_emb.flatten(start_dim=0, end_dim=1)
					total_msk = torch.cat((src_mask, abs_mask, ext_mask[1 + abs_mask.shape[0]:]), dim=0)
					total_msk = total_msk.flatten()
					tree_len = 1 + abstractor_outputs["tree_lens_diff"][batch_idx] * 2
				
				final_emb.append(total_emb)
				final_msk.append(total_msk)
				final_tree_lens.append(tree_len)

				## update edges
				src_idx = torch.zeros(tree_len - 1, dtype=torch.long)
				src_idx = torch.cat((src_idx, torch.tensor([-1] * (self.model_args.num_clusters - (tree_len - 1)))), dim=0) ## Pad edges
				sum_idx = torch.tensor(range(tree_len - 1), dtype=torch.long) + 1
				sum_idx = torch.cat((sum_idx, torch.tensor([-1] * (self.model_args.num_clusters - (tree_len - 1)))), dim=0) ## Pad edges
				td_edges.append(torch.stack([src_idx, sum_idx]))
				bu_edges.append(torch.stack([sum_idx, src_idx]))

			td_edges = torch.stack(td_edges).long().to(inputs_embeds.device)
			bu_edges = torch.stack(bu_edges).long().to(inputs_embeds.device)

		else:
			## Combine `cluster_outputs` & `abstract_outputs`
			final_emb, final_msk, final_tree_lens = [], [], []
			batch_size = inputs_embeds.shape[0]
			for batch_idx in range(batch_size):
				tree_len = cluster_outputs["tree_lens"][batch_idx]
			
				ext_summ = cluster_outputs["inputs_embeds"][batch_idx] ## (32, 32, 768)
				ext_mask = cluster_outputs["attention_mask"][batch_idx]
				abs_summ = abstractor_outputs["summary_hidden_states"][batch_idx] ## (3, 32, 768)
				abs_mask = abstractor_outputs["summary_attention_mask"][batch_idx]

				if self.output_type == "dre_only":
					## Source Post + Extractive Summary
					total_emb = ext_summ.flatten(start_dim=0, end_dim=1)
					total_msk = ext_mask.flatten()
					
					final_emb.append(total_emb)
					final_msk.append(total_msk)
					final_tree_lens.append(tree_len)
				else:
					## Source Post + Extractive Summary + Abstractive Summary
					total_emb = torch.cat((ext_summ[:tree_len], abs_summ, ext_summ[tree_len + abs_summ.shape[0]:]), dim=0)
					total_emb = total_emb.flatten(start_dim=0, end_dim=1)
					total_msk = torch.cat((ext_mask[:tree_len], abs_mask, ext_mask[tree_len + abs_mask.shape[0]:]), dim=0)
					total_msk = total_msk.flatten()

					final_emb.append(total_emb)
					final_msk.append(total_msk)
					final_tree_lens.append(tree_len + abstractor_outputs["tree_lens_diff"][batch_idx])

			if self.output_type is None:
				td_edges = self.graph_reconstruction(td_edges=td_edges, abs_summ=abstractor_outputs["summary_hidden_states"])
				bu_edges = self.graph_reconstruction(bu_edges=bu_edges, abs_summ=abstractor_outputs["summary_hidden_states"])
		
		final_emb = torch.stack(final_emb)
		final_msk = torch.stack(final_msk)
		final_tree_lens = torch.LongTensor(final_tree_lens).to(final_msk.device)

		final_outputs = {
			"inputs_embeds": final_emb, 
			"attention_mask": final_msk, 
			"tree_lens": final_tree_lens, 
			"td_edges": td_edges, 
			"bu_edges": bu_edges, 
			"n_ext_adv": filter_outputs["n_ext_adv"] if self.filter is not None else None, 
			## For demo
			"filt_ext_idxs": filter_outputs["ext_idxs"] if self.filter is not None else None, 
			"clus_ext_idxs": cluster_outputs["ext_idxs"], 
			"clus_ids": cluster_outputs["cluster_ids"], 
			"summary_tokens": abstractor_outputs["summary_tokens"]
		}
		return final_outputs

	def graph_reconstruction(self, td_edges=None, bu_edges=None, ext_idxs=None, abs_summ=None):
		"""
		Reconstruct graph for Response Extractor or Clustering Abstractor.
		"""
		if td_edges is not None:
			edges = td_edges
			parent_idx, child_idx = 0, 1
		elif bu_edges is not None:
			edges = bu_edges
			parent_idx, child_idx = 1, 0
		else:
			raise ValueError("Either td_edges or bu_edges has to be specified.")

		new_edges = []
		batch_size = edges.shape[0]
		if ext_idxs is not None: ## Reconstruct extractor outputs
			for batch_idx in range(batch_size):
				edge = edges[batch_idx]
				edge = edge[:, edge[0] != -1] ## Remove padding (-1)
				ext_idx = ext_idxs[batch_idx].nonzero().flatten() ## Convert from one hot to index value
			
				edges_to_keep = torch.ones(edge.shape[1]).bool()
				
				for i in range(edge.shape[1]):
					parent = edge[parent_idx][i]
					child  = edge[child_idx][i]
					if child not in ext_idx: ## This edge should be removed
						edges_to_keep[i] = False
					else: ## This edge should be kept
						index = i
						while True: ## Find parent until it exists in new tree
							if parent in ext_idx:
								break
							else:
								index = (edge[child_idx] == parent).nonzero().item()
								child = edge[child_idx][index]
								parent = edge[parent_idx][index]
						edge[parent_idx][i] = parent
				
				## Remove unused edges
				edge = edge[:, edges_to_keep]
				
				## Give new index with correct order
				for i in range(edge.shape[0]):
					for j in range(edge.shape[1]):
						edge[i][j] = (ext_idx == edge[i][j]).nonzero().item()
			
				## Pad edge
				edge_pad = torch.LongTensor([[-1], [-1]]).to(edge.device)
				edge = torch.cat((edge, edge_pad.repeat(1, 32 - edge.shape[1])), dim=1)
				new_edges.append(edge)

		elif abs_summ is not None:
			if self.summarizer_type == "abstract": ## SSRA-LOO, BART-base-SAMSum
				edge_summ = torch.LongTensor([[0], [0]]).to(edges.device)
				edge_summ[child_idx] = 1
				new_edges = [edge_summ] * batch_size
			else:
				for batch_idx in range(batch_size):
					summ = abs_summ[batch_idx]
					edge = edges[batch_idx]
					edge = edge[:, edge[0] != -1] ## Remove padding (-1)
				
					for summ_idx in range(summ.shape[0]): ## For each abstractive summary, add an edge
						node_summ_idx = edge.max() + 1 if edge.numel() != 0 else 1
						edge_summ = torch.LongTensor([[0], [0]]).to(edge.device)
						edge_summ[child_idx] = node_summ_idx
						edge = torch.cat((edge, edge_summ), dim=1)
				
					## Pad edge
					edge_pad = torch.LongTensor([[-1], [-1]]).to(edge.device)
					edge = torch.cat((edge, edge_pad.repeat(1, 32 - edge.shape[1])), dim=1)
					new_edges.append(edge)

		new_edges = torch.stack(new_edges)
		return new_edges

	def get_gen_hidden_states_from_tuple(self, config, embedding_layer, decoder_hidden_states):
		"""Convert decoder_hidden_states of type tuple into torch tensor."""
		bos_embedding = embedding_layer.weight[config.bos_token_id]
		pad_embedding = embedding_layer.weight[config.pad_token_id]
		bos_embedding = bos_embedding.reshape(1, 1, -1).repeat(decoder_hidden_states[0][0].shape[0], 1, 1)
		pad_embedding = pad_embedding.reshape(1, 1, -1).repeat(decoder_hidden_states[0][0].shape[0], 1, 1)

		## Add start token
		gen_hidden_states = [bos_embedding]
		gen_attention_msk = torch.zeros(self.data_args.max_tweet_length, device=bos_embedding.device)
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