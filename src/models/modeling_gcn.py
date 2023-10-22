import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv

## Self-defined
from others.utils import mean_pooling

class GCNPooler(nn.Module):
	"""Stack of GCN layers, can be top-down / bottom-up"""

	def __init__(self, data_args, model_args, training_args, hidden_size, gcn_type=None):
		super(GCNPooler, self).__init__()
		
		print("GCN Type: {}".format(gcn_type))

		self.n_layers = 2
		self.gcn_type = gcn_type ## `td` / `bu`
		if self.gcn_type is not None:
			self.child_idx  = 1 if self.gcn_type == "td" else 0
			self.parent_idx = 0 if self.gcn_type == "td" else 1
		
		self.data_args = data_args
		self.model_args = model_args
		self.training_args = training_args

		## Edge Filter
		self.filter = None
		if self.model_args.edge_filter:
			self.filter = nn.Sequential(
				nn.Linear(2 * hidden_size, 1), 
				nn.Sigmoid()
			)

		## GCN Layers
		self.gcn = nn.ModuleList()
		for gcn_idx in range(self.n_layers):
			self.gcn.append(GCNConv(hidden_size, hidden_size))
		self.fc = nn.Linear(hidden_size, hidden_size)
		self.act = nn.Tanh()

	def prepare_batch(
		self, 
		hidden_states, 
		attention_msk, 
		tree_lens, 
		edge_index
	):
		"""
		Prepare batch data for GCNConv.
		Ex. [[0, 0], [1, 2]] + [[0, 1], [1, 2]] -> [[0, 0, 3, 4], [1, 2, 5, 6]]

		Return:
			- nodes: node features of all trees (graphs) in a batch
			- edges: edges for graph formed by all trees (graphs) in a batch
			- index: indicate each node belong to which tree (graph)
		"""
		root_idx = 0
		nodes, edges, index = [], [], []
		for batch_idx in range(edge_index.shape[0]): ## Iterate through each tree
			edge = edge_index[batch_idx]
			edge = edge[:, edge[0] != -1] ## Remove padding (-1)

			#n_tweets = edge.shape[1] + 1 ## number of nodes = number of edges + 1
			n_tweets = tree_lens[batch_idx]
			node_states = hidden_states[batch_idx][:n_tweets] ## shape: (n_tweets, 32, 768)
			node_masks  = attention_msk[batch_idx][:n_tweets] ## shape: (n_tweets, 32)
			
			## Get each node's feature by mean pooling
			node_feat = mean_pooling(node_states, node_masks)

			## Collect all trees (graphs) in a batch into a single graph
			edge = edge + root_idx
			edges.append(edge)
			nodes.append(node_feat)
			index.append(torch.ones(n_tweets, dtype=torch.long) * batch_idx)
			
			## Set root index for next graph
			root_idx = root_idx + n_tweets

		nodes = torch.cat(nodes, dim=0)
		edges = torch.cat(edges, dim=1)
		index = torch.cat(index, dim=0).to(hidden_states.device)

		## Check
		if edges.nelement() != 0: ## If edges is not empty
			if edges.max().item() > (nodes.shape[0] - 1):
				raise ValueError("Edge index more than number of nodes!")

		return nodes, edges, index

	def forward(
		self, 
		hidden_states, 
		attention_msk, 
		tree_lens, 
		edge_index=None
	):
		"""Aggregate graphs by GCN layers"""

		## Split hidden states of each sequence (conversational thread) into nodes
		## hidden_states.shape = (bs, max_len, hidden_size) -> (bs, max_tree_length, max_tweet_length, hidden_size)
		## Ex. (8, 512, 768) -> (8, 16, 32, 768) 
		hidden_states = torch.stack(torch.split(hidden_states, self.data_args.max_tweet_length, dim=1), dim=1)
		attention_msk = torch.stack(torch.split(attention_msk, self.data_args.max_tweet_length, dim=1), dim=1)

		if edge_index is None:
			## Transformer only, take the first token as tree representation
			tree_embeds = hidden_states[:, 0, 0, :]
		else:
			## Transformer + GCN
			nodes, edges, index = self.prepare_batch(
				hidden_states=hidden_states, 
				attention_msk=attention_msk, 
				tree_lens=tree_lens, 
				edge_index=edge_index
			)

			## Edge Filter
			edge_weights = None
			if self.filter is not None:
				child_nodes  = nodes[edges[self.child_idx]]
				parent_nodes = nodes[edges[self.parent_idx]]
				edge_weights = self.filter(torch.cat((child_nodes, parent_nodes), dim=1))
				edge_weights = edge_weights.view(-1)

			## GCN Layers
			for gcn_idx, conv_i in enumerate(self.gcn):
				try:
					nodes = conv_i(nodes, edges, edge_weights)
				except:
					ipdb.set_trace()
				nodes = F.relu(nodes)
				nodes = F.dropout(nodes, p=0.1, training=self.training) if gcn_idx == 0 else nodes

			## Node aggregation
			tree_embeds = scatter_mean(nodes, index, dim=0)

		pooled_output = self.fc(tree_embeds)
		pooled_output = self.act(pooled_output)
		return pooled_output

class GCNForClassification(nn.Module):
	"""Detector head with GCN classifier"""

	def __init__(
		self, 
		data_args, 
		model_args, 
		training_args, 
		hidden_size, 
		num_labels, 
		hidden_dropout_prob=0.1
	):
		super(GCNForClassification, self).__init__()

		self.data_args = data_args
		self.model_args = model_args
		self.training_args = training_args

		## Assigned at `CustomSeq2SeqTrainer.__init__()` from `trainer_adv.py`
		self.loss_weight = None

		## Decide the GCN structure to use
		self.pooler = GCNPooler(self.data_args, self.model_args, self.training_args, hidden_size) if not (self.model_args.td_gcn or self.model_args.bu_gcn) else None
		self.td_pooler = GCNPooler(self.data_args, self.model_args, self.training_args, hidden_size, gcn_type="td") if model_args.td_gcn else None
		self.bu_pooler = GCNPooler(self.data_args, self.model_args, self.training_args, hidden_size, gcn_type="bu") if model_args.bu_gcn else None
		
		self.dropout = nn.Dropout(hidden_dropout_prob)

		input_dim = hidden_size
		input_dim = 2 * input_dim if (self.model_args.td_gcn and self.model_args.bu_gcn) else input_dim
		self.classifier = nn.Linear(input_dim, num_labels)

		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=0.02) ## std: initializer_range
		elif isinstance(module, nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)
		if isinstance(module, nn.Linear) and module.bias is not None:
			module.bias.data.zero_()

	def forward(
		self, 
		hidden_states, 
		attention_msk, 
		tree_lens, 
		td_edges=None, 
		bu_edges=None, 
		labels=None
	):
		if self.pooler is not None:
			## Transformer only
			pooled_output = self.pooler(
				hidden_states=hidden_states, 
				attention_msk=attention_msk,
				tree_lens=tree_lens
			)
		else:
			## With GCN
			if self.td_pooler is not None:
				td_pooled_output = self.td_pooler(
					hidden_states=hidden_states, 
					attention_msk=attention_msk, 
					tree_lens=tree_lens, 
					edge_index=td_edges
				)
			if self.bu_pooler is not None:
				bu_pooled_output = self.bu_pooler(
					hidden_states=hidden_states, 
					attention_msk=attention_msk, 
					tree_lens=tree_lens, 
					edge_index=bu_edges
				)

			if self.td_pooler is not None and self.bu_pooler is not None: ## BiTGN
				pooled_output = torch.cat((td_pooled_output, bu_pooled_output), dim=1)
			elif self.td_pooler is not None: ## TDTGN
				pooled_output = td_pooled_output
			elif self.bu_pooler is not None: ## BUTGN
				pooled_output = bu_pooled_output
			else:
				raise ValueError("Wrong argument specification!")

		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)

		loss = None
		if labels is not None:
			loss_fct = nn.CrossEntropyLoss(weight=self.loss_weight)
			loss = loss_fct(logits, labels)

		return (logits, loss)
