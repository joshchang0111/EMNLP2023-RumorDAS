import os
import ipdb
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from transformers import default_data_collator

## Self-defined
from models.modeling_clustering import ClusterModel
from others.utils import mean_pooling

class ClusterSummaryBuilder:
	"""
	Cluster Summary Dataset Builder, 
	builds dataset for 2nd-stage abstractor training.
	"""
	def __init__(
		self, 
		model=None,
		data_args=None,
		model_args=None,
		training_args=None,
		train_dataset=None,
		eval_dataset=None,
	):
		self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

		self.data_args = data_args
		self.model_args = model_args
		self.training_args = training_args

		self.model = model.to(self.device)
		self.model_cluster = ClusterModel(cluster_type=self.model_args.cluster_type, num_clusters=self.model_args.num_clusters)

		self.dataset = train_dataset
		self.cluster_mode = self.model_args.cluster_mode
		assert (self.cluster_mode == "train" or self.cluster_mode == "test"), "please specify the cluster_mode"

		## Build data loader
		assert self.training_args.per_device_train_batch_size == 1, "Batch size should be 1 -> process 1 tree at once"
		self.dataloader = DataLoader(
			self.dataset, 
			batch_size=self.training_args.per_device_train_batch_size, 
			collate_fn=default_data_collator, 
			num_workers=self.training_args.dataloader_num_workers, 
			pin_memory=self.training_args.dataloader_pin_memory, 
			shuffle=False
		)

	def build_cluster_summary(self):
		print("\nStart building cluster summary...")
		if self.model_cluster.cluster_type == "kmeans":
			self.build_by_kmeans()
		elif self.model_cluster.cluster_type == "topics":
			self.build_by_topics()

	def build_by_kmeans(self):
		tree_idx = 0
		output_dict = {
			"source_id": [], 
			"tweet_id": [], 
			"cluster_id": [], 
			"is_centroid": []
		}
		for batch in tqdm(self.dataloader):
			tree_lens = batch["tree_lens"].to(self.device)
			input_ids = batch["input_ids"].to(self.device)
			attn_mask = batch["attention_mask"].to(self.device)

			## Iterate through each sample (tree) in a batch, actually one tree per batch
			for batch_idx in range(input_ids.shape[0]):
				tree_len = tree_lens[batch_idx]
				tree_source_id = [self.dataset["source_id"][tree_idx]] * tree_len
				tree_tweet_ids = self.dataset["tweet_ids"][tree_idx].split(",")
				
				tree_input_ids = input_ids[batch_idx]
				tree_attn_mask = attn_mask[batch_idx]
				tree_input_ids = tree_input_ids[tree_input_ids[:, 0] != -1] ## Remove padding nodes
				tree_attn_mask = tree_attn_mask[tree_attn_mask[:, 0] != -1] ## Remove padding nodes
				
				## Obtain hidden representations
				with torch.no_grad():
					## TODO: make sure to use embeddings or hidden representation?
					encoder_outputs = self.model.encoder(
						input_ids=tree_input_ids, 
						attention_mask=tree_attn_mask, 
						return_dict=True
					)

					## Get node features by mean pooling on embeddings
					node_feat = torch.mean(encoder_outputs["embed_tok"], dim=1)

				## Clustering
				response_feat = node_feat[1:]
				if response_feat.shape[0] > 0: ## Has responses
					cluster_ids, cluster_centers, dist, is_centroid = \
					self.model_cluster(
						node_feat=response_feat, 
						device=self.device, 
						mode=self.cluster_mode
					)

					output_dict["source_id"].extend(tree_source_id[1:])
					output_dict["tweet_id"].extend(tree_tweet_ids[1:])
					output_dict["cluster_id"].extend(cluster_ids.tolist())
					output_dict["is_centroid"].extend(is_centroid.tolist())
				else: ## No response
					print("Ignore this tree since it has no response.")

				tree_idx = tree_idx + 1
				"""
				response_feat = node_feat[1:]
				if response_feat.shape[0] > 2: ## More than 2 responses
					cluster_ids, cluster_centers, dist = self.model_clustering.cluster(node_feat=response_feat, device=self.device)
				
					## Format:
					## source_id, tweet_id, cluster_id, is_centroid
					output_dict["source_id"].extend(tree_source_id[1:])
					output_dict["tweet_id"].extend(tree_tweet_ids[1:])
					output_dict["cluster_id"].extend(cluster_ids.tolist())
					
					## Find the centroid of each cluster
					is_centroid = torch.zeros(dist.shape[0], dtype=torch.long)
					for cluster_i in range(dist.shape[1]):
						## Ignore this cluster if no response belongs to this cluster
						if cluster_i not in cluster_ids:
							continue
				
						## Get distance of responses closest to cluster center and set them as centroid
						dist_2_i = dist[:, cluster_i]
						min_dist = dist_2_i[cluster_ids == cluster_i].min()
						min_idxs = (dist_2_i == min_dist).nonzero().flatten()
						
						is_centroid[min_idxs] = 1
					
					output_dict["is_centroid"].extend(is_centroid.tolist())
				
				elif response_feat.shape[0] > 1: ## Only 2 responses
					## Create only 1 cluster where both responses are centers (take turns to be the center)
					cluster_ids = torch.zeros(response_feat.shape[0], dtype=torch.long)
					is_centroid = torch.ones(response_feat.shape[0], dtype=torch.long) ## should have shape = 2
				
					output_dict["source_id"].extend(tree_source_id[1:])
					output_dict["tweet_id"].extend(tree_tweet_ids[1:])
					output_dict["cluster_id"].extend(cluster_ids.tolist())
					output_dict["is_centroid"].extend(is_centroid.tolist())
				
				else: ## Less than or equal to 1 response
					print("Ignore this tree since it has only 1 response or less.")
				"""
		ipdb.set_trace()
		##################################################################
		## ** NOTE ** 													##
		## - each cluster may have more than one centroids since 		##
		##	 some closest nodes have the same distances from the center ##
		## - these nodes can take turns being the target summary		##
		##################################################################
		output_df = pd.DataFrame(data=output_dict)
		update_df = []
		for source_id, tweets_df in output_df.groupby("source_id"):
			for cluster_id, cluster_df in tweets_df.groupby("cluster_id"):
				num_centroids = (cluster_df["is_centroid"] == 1).sum()
				if num_centroids > 1: ## More than one centroid!
					centroid_tids = cluster_df[cluster_df["is_centroid"] == 1]["tweet_id"]
					for sub_idx, cent_tid in enumerate(centroid_tids): ## Each centroid takes turn being the target summary
						subcluster_df = cluster_df.copy()
						subcluster_df.loc[subcluster_df["tweet_id"] != cent_tid, ["is_centroid"]] = 0
						subcluster_df["cluster_id"] = "{}_{}".format(cluster_id, sub_idx)
						update_df.append(subcluster_df)
				else: ## Only one centroid
					cluster_df["cluster_id"] = cluster_df["cluster_id"].astype(str)
					update_df.append(cluster_df)
				
		update_df = pd.concat(update_df)

		## Output
		os.makedirs(
			"{}/{}/split_{}/cluster_summary/{}".format(
				self.data_args.dataset_root, 
				self.data_args.dataset_name, 
				self.data_args.fold, 
				self.cluster_mode
			), 
			exist_ok=True
		)
		update_df.to_csv(
			"{}/{}/split_{}/cluster_summary/{}/kmeans-{}.csv".format(
				self.data_args.dataset_root, 
				self.data_args.dataset_name, 
				self.data_args.fold, 
				self.cluster_mode, 
				self.model_cluster.num_clusters
			), 
			index=False
		)

	def build_by_topics(self):
		tree_idx = 0
		output_dict = {
			"source_id": [],  
			"cluster_id": [],
			"tweet_ids": [],  
			"centroid": []
		}
		for batch in tqdm(self.dataloader):
			tree_lens = batch["tree_lens"].to(self.device)
			input_ids = batch["input_ids"].to(self.device)
			attn_mask = batch["attention_mask"].to(self.device)
			topic_ids = batch["topic_ids"].to(self.device)
			topic_msk = batch["topic_msk"].to(self.device)
			topic_probs = batch["topic_probs"].to(self.device)

			## Iterate through each sample (tree) in a batch, actually one tree per batch
			for batch_idx in range(input_ids.shape[0]):
				tree_len = tree_lens[batch_idx]
				tree_source_id = [self.dataset["source_id"][tree_idx]] * tree_len
				tree_tweet_ids = self.dataset["tweet_ids"][tree_idx].split(",")
				
				tree_input_ids = input_ids[batch_idx]
				tree_attn_mask = attn_mask[batch_idx]
				tree_input_ids = tree_input_ids[tree_input_ids[:, 0] != -1] ## Remove padding nodes
				tree_attn_mask = tree_attn_mask[tree_attn_mask[:, 0] != -1] ## Remove padding nodes
				tree_topic_ids = topic_ids[batch_idx]
				tree_topic_msk = topic_msk[batch_idx]
				tree_topic_probs = topic_probs[batch_idx]
				
				## Obtain hidden representations
				with torch.no_grad():
					## Get node embeddings
					encoder_outputs = self.model.encoder(
						input_ids=tree_input_ids, 
						attention_mask=tree_attn_mask, 
						return_dict=True
					)
				
					## Get node features by mean pooling on embedding
					pooling_mask = tree_attn_mask.clone()
					seq_lens = pooling_mask.sum(dim=1)
					for i, seq_len in enumerate(seq_lens):
						pooling_mask[i][seq_len - 1] = 0
					pooling_mask[:, 0] = 0
					node_emb = mean_pooling(encoder_outputs["embed_tok"], pooling_mask)

					## Get topic embeddings
					topic_outputs = self.model.encoder(
						input_ids=tree_topic_ids, 
						attention_mask=tree_topic_msk, 
						return_dict=True
					)
					topic_emb = topic_outputs["embed_tok"]

				response_emb = node_emb[1:]
				cluster_ridx, centroid_ridx = self.model_clustering.cluster(node_feat=response_emb, topic_feat=topic_emb, topic_probs=tree_topic_probs)
				
				response_tids = tree_tweet_ids[1:]

				tree_idx = tree_idx + 1
				if cluster_ridx is None:
					continue

				n_cluster = cluster_ridx.shape[1]
				output_dict["source_id"].extend([tree_source_id[0]] * 3)
				output_dict["cluster_id"].extend(list(range(n_cluster)))
				for cluster_id in range(n_cluster):
					cluster_tids = [response_tids[ridx] for ridx in cluster_ridx[:, cluster_id]]
					output_dict["tweet_ids"].append(",".join(cluster_tids))
				output_dict["centroid"].extend([response_tids[cent_ridx] for cent_ridx in centroid_ridx])
				#output_dict["centroid"].extend([response_tids[cent_ridx] for cent_ridx in cluster_ridx[0]])
				ipdb.set_trace()

		output_df = pd.DataFrame(data=output_dict)
		output_df.to_csv(
			"{}/{}/split_{}/cluster_summary/topics-3.csv".format(
				self.data_args.dataset_root, 
				self.data_args.dataset_name, 
				self.data_args.fold
			), 
			index=False
		)