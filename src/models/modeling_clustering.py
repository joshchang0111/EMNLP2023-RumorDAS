import ipdb
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from kmeans_pytorch.soft_dtw_cuda import SoftDTW
from kmeans_pytorch import initialize, pairwise_distance

def kmeans(
		X,
		num_clusters,
		distance='euclidean',
		cluster_centers=[],
		tol=1e-4,
		tqdm_flag=True,
		iter_limit=0,
		device=torch.device('cpu'),
		gamma_for_soft_dtw=0.001,
		seed=None,
):
	"""
	NOTE that this function is copied and modified from `kmeans_pytorch`.
	Modification:
	- enable clustering when `num_clusters == 1`

	perform kmeans
	:param X: (torch.tensor) matrix
	:param num_clusters: (int) number of clusters
	:param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
	:param seed: (int) seed for kmeans
	:param tol: (float) threshold [default: 0.0001]
	:param device: (torch.device) device [default: cpu]
	:param tqdm_flag: Allows to turn logs on and off
	:param iter_limit: hard limit for max number of iterations
	:param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
	:return: (torch.tensor, torch.tensor) cluster ids, cluster centers
	"""
	if tqdm_flag:
		print(f'running k-means on {device}..')

	if distance == 'euclidean':
		pairwise_distance_function = partial(pairwise_distance, device=device, tqdm_flag=tqdm_flag)
	elif distance == 'cosine':
		pairwise_distance_function = partial(pairwise_cosine, device=device)
	elif distance == 'soft_dtw':
		sdtw = SoftDTW(use_cuda=device.type == 'cuda', gamma=gamma_for_soft_dtw)
		pairwise_distance_function = partial(pairwise_soft_dtw, sdtw=sdtw, device=device)
	else:
		raise NotImplementedError

	# convert to float
	X = X.float()

	# transfer to device
	X = X.to(device)

	# initialize
	if type(cluster_centers) == list:  # ToDo: make this less annoyingly weird
		initial_state = initialize(X, num_clusters, seed=seed)
	else:
		if tqdm_flag:
			print('resuming')
		# find data point closest to the initial cluster center
		initial_state = cluster_centers
		dis = pairwise_distance_function(X, initial_state)
		choice_points = torch.argmin(dis, dim=0)
		initial_state = X[choice_points]
		initial_state = initial_state.to(device)

	iteration = 0
	if tqdm_flag:
		tqdm_meter = tqdm(desc='[running kmeans]')
	while True:

		dis = pairwise_distance_function(X, initial_state)
		
		if len(dis.shape) == 1:
			dis = dis.view(-1, 1)
		
		choice_cluster = torch.argmin(dis, dim=1)

		initial_state_pre = initial_state.clone()

		for index in range(num_clusters):
			selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

			selected = torch.index_select(X, 0, selected)

			# https://github.com/subhadarship/kmeans_pytorch/issues/16
			if selected.shape[0] == 0:
				selected = X[torch.randint(len(X), (1,))]

			initial_state[index] = selected.mean(dim=0)

		center_shift = torch.sum(
			torch.sqrt(
				torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
			))

		# increment iteration
		iteration = iteration + 1

		# update tqdm meter
		if tqdm_flag:
			tqdm_meter.set_postfix(
				iteration=f'{iteration}',
				center_shift=f'{center_shift ** 2:0.6f}',
				tol=f'{tol:0.6f}'
			)
			tqdm_meter.update()
		if center_shift ** 2 < tol:
			break
		if iter_limit != 0 and iteration >= iter_limit:
			break

	return choice_cluster.cpu(), initial_state.cpu()

class ClusterModel(nn.Module):
	def __init__(self, cluster_type="kmeans", num_clusters=3, extract_ratio=None):
		super(ClusterModel, self).__init__()

		self.num_clusters = num_clusters
		self.cluster_type = cluster_type
		self.extract_ratio = extract_ratio

		print("Cluster Model: {}".format(self.cluster_type))
		print("Num. clusters: {}".format(self.num_clusters))
	
	def forward(
		self, 
		node_feat=None, 
		topic_feat=None, 
		topic_probs=None, 
		mode="train", 
		device=None
	):
		"""
		Input:
			- node_feat: node features of all response in a tree
			- topic_feat : for cluster_by_topic
			- topic_probs: for cluster_by_topic
			- mode: for cluster_by_kmeans, either "train" or "test", whether to random sample a centroid when multiple points are closest to center
			- device: for cluster_by_kmeans, the device to run on
		Output:
			- clusters
		"""
		if self.cluster_type == "kmeans":
			return self.cluster_by_kmeans(node_feat=node_feat, mode=mode, device=device)
		elif self.cluster_type == "topics":
			return self.cluster_by_topics(node_feat, topic_feat, topic_probs)
		
	def cluster_by_kmeans(self, node_feat, mode="train", device=None):
		## Adjust number of clusters
		if self.extract_ratio is None:
			num_clusters = self.num_clusters
			while (num_clusters >= node_feat.shape[0]) and (num_clusters != 1):
				num_clusters = math.ceil(num_clusters / 2)
			#if num_clusters == 1 and node_feat.shape[0] > 1:
			#	num_clusters = num_clusters + 1
		else:
			num_nodes = torch.tensor(node_feat.shape[0])
			num_clusters = int(torch.floor(num_nodes * self.extract_ratio))
			num_clusters = num_clusters + 1 if num_clusters == 0 else num_clusters
		
		## Cluster ##
		if node_feat.shape[0] > 1:
			cluster_ids, cluster_centers = kmeans(
				X=node_feat, 
				num_clusters=num_clusters, 
				tol=1e-6, 
				distance="euclidean", 
				device=device, 
				tqdm_flag=False
			)
		else: ## When there is only one response
			cluster_ids = torch.LongTensor([0])
			cluster_centers = node_feat
		
		cluster_ids = cluster_ids.to(device)
		cluster_centers = cluster_centers.to(device)

		## Calculate distance of each node to each cluster
		dist = pairwise_distance(node_feat, cluster_centers, device=device, tqdm_flag=False)
		if len(dist.shape) <= 1: dist = dist.view(-1, 1)

		## Find the centroid of each cluster
		is_centroid = torch.zeros(dist.shape[0], dtype=torch.long).to(device)
		for cluster_i in range(dist.shape[1]):
			## Ignore this cluster if no response belongs to this cluster
			if cluster_i not in cluster_ids:
				continue
		
			## Get distance of responses closest to cluster center and set them as centroid
			dist_2_i = dist[:, cluster_i]
			min_dist = dist_2_i[cluster_ids == cluster_i].min()
			min_idxs = (dist_2_i == min_dist).nonzero().flatten()
			if mode == "test": ## testing: random sample if more than one centroid
				min_idxs = min_idxs[torch.randint(low=0, high=len(min_idxs), size=(1,))]
			is_centroid[min_idxs] = 1
			
		return cluster_ids, cluster_centers, dist, is_centroid

	def cluster_by_topics(self, node_feat, topic_feat, topic_probs):
		def masked_softmax(x, mask, temp=0.1, dim=1):
			x_masked = x.clone()
			x_masked[mask == 0] = -float("inf")
			return F.softmax(x_masked / temp, dim=dim)

		mask = (topic_probs != 0).long()
		topic_probs = masked_softmax(topic_probs, mask, temp=0.01, dim=1)
		topic_feat = (topic_feat * topic_probs.unsqueeze(-1)).sum(dim=1)

		n_topics = topic_feat.shape[0]
		n_sample = node_feat.shape[0]

		l2_dist = torch.cdist(node_feat, topic_feat)
		cluster_prob = F.softmax(-1 * l2_dist, dim=1)
		
		dist_sort = l2_dist.argsort(dim=0)
		prob_sort = cluster_prob.argsort(dim=0)

		cluster_ridx = prob_sort[:math.ceil(n_sample / n_topics), :]
		if cluster_ridx.numel() == 0: ## No response
			return None, None
			
		centroid_ridx = []
		for topic_i in range(n_topics):
			ridx = cluster_ridx[:, topic_i]
			cent_ridx = l2_dist[ridx, topic_i].argsort()[0]
			centroid_ridx.append(ridx[cent_ridx])
		centroid_ridx = torch.stack(centroid_ridx)

		return cluster_ridx, centroid_ridx