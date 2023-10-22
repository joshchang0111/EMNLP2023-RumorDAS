import os
import ipdb
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

class GraphDataset(Dataset):
	"""
	This class is used to process graphical info. for GCN.
	"""
	def __init__(self, data_args , model_args=None):

		print("\nBuilding graph dataset...")

		self.data_args = data_args
		self.model_args = model_args

		self.tddroprate = 0#0.2
		self.budroprate = 0#0.2

		## Read dataset content
		self.data_df = pd.read_csv("{}/{}/data.csv".format(self.data_args.dataset_root, self.data_args.dataset_name))
		self.data_df["source_id"] = self.data_df["source_id"].astype(str) ## For PHEME, twitter15, twitter16
		self.data_df["tweet_id" ] = self.data_df["tweet_id" ].astype(str)
		self.data_df["self_idx" ] = self.data_df["self_idx" ].astype(str)

		## Check whether graph cache exists
		cache_path = "{}/{}/graph.pth".format(self.data_args.dataset_root, self.data_args.dataset_name)
		if os.path.exists(cache_path):
			print("Graph cache exists, directly load graph information from {}".format(cache_path))
			graph_infos = torch.load(cache_path)
			self.td_edges = graph_infos["td_edges"]
			self.bu_edges = graph_infos["bu_edges"]
		else:
			## Initialize src_id->edge_index map
			self.td_edges = {}
			self.bu_edges = {}
			for src_id in tqdm(list(set(self.data_df["source_id"]))):
				tree_df = self.data_df.loc[self.data_df["source_id"] == src_id]
				tree_df = tree_df.reset_index(drop=True)
			
				## Build edge_index, row -> parent_idx, col -> child_idx
				## Note: edges will be sorted by child_idx
				## Example: src_id = "529695367680761856"
				## 			edge_index = [[0, 0, 0, 0, 0, 5, 6, 0], 
				##				 		  [1, 2, 3, 4, 5, 6, 7, 8]]
				row = []
				col = []
				for index_i, tweet_i in tree_df.iterrows():
					for index_j, tweet_j in tree_df.iterrows():
						if tweet_i["parent_idx"] == tweet_j["self_idx"]:
							row.append(index_j)
							col.append(index_i)
				edge_index = torch.LongTensor([row, col])
			
				## Correct edge: correct parent_idx of edges that parent_idx > child_idx to root
				parent = torch.LongTensor(row)
				child  = torch.LongTensor(col)
				parent[parent > child] = 0
			
				self.td_edges[src_id] = torch.stack((parent, child), dim=0)
				self.bu_edges[src_id] = torch.stack((child, parent), dim=0)

	def __getitem__(self, src_id):
		"""
		Returns:
			- td_edge_index: top-down edge indices corresponding to the tree structure given src_id
			- bu_edge_index: bottom-up edge indices
		"""
		if src_id not in self.td_edges:
			raise ValueError("source_id not in graph_dataset mapping!")

		## Top-down edge index
		td_edge_index = self.td_edges[src_id]
		td_edge_index = td_edge_index[:, td_edge_index[1] <= self.data_args.max_tree_length] ## Truncate

		## Bottom-up edge index
		bu_edge_index = self.bu_edges[src_id]
		bu_edge_index = bu_edge_index[:, bu_edge_index[0] <= self.data_args.max_tree_length] ## Truncate
		
		return td_edge_index, bu_edge_index

	def pad(self, edge_index):
		"""
		To enable batching for huggingface data collator, 
		need to pad to data_args.max_tree_length - 1 with value -1.
		"""
		## Padding, pad value = -1
		pad_length = (self.data_args.max_tree_length - 1) - edge_index.shape[1]
		edge_index = torch.cat((edge_index, torch.full((2, pad_length), -1)), dim=1)

		return edge_index

	def drop_edge(self, edge_index, edge_type):
		if edge_type == "td" and self.tddroprate > 0:
			rand_idx = random.sample(range(edge_index.shape[1]), int(edge_index.shape[1] * (1 - self.tddroprate)))
			rand_idx.sort()
			edge_index = edge_index[:, rand_idx]
		elif edge_type == "bu" and self.budroprate > 0:
			rand_idx = random.sample(range(edge_index.shape[1]), int(edge_index.shape[1] * (1 - self.budroprate)))
			rand_idx.sort()
			edge_index = edge_index[:, rand_idx]
		return edge_index

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Rumor Detection")
	parser.add_argument("--dataset_name", type=str, default=None)
	parser.add_argument("--dataset_root", type=str, default="../dataset/processedV2")
	args = parser.parse_args()

	print("Dataset: {}".format(args.dataset_name))

	## Build graph dataset
	graph_dataset = GraphDataset(data_args=args)

	graph_infos = {
		"td_edges": graph_dataset.td_edges, 
		"bu_edges": graph_dataset.bu_edges
	}

	## Write file to dataset
	torch.save(graph_infos, "{}/{}/graph.pth".format(args.dataset_root, args.dataset_name))
