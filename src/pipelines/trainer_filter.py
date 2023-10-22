import ipdb
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from transformers import default_data_collator

class FilterTrainer:
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

		self.train_dataloader = DataLoader(
			train_dataset, 
			batch_size=self.training_args.per_device_train_batch_size, 
			collate_fn=default_data_collator, 
			num_workers=self.training_args.dataloader_num_workers, 
			pin_memory=self.training_args.dataloader_pin_memory
		)
		self.eval_dataloader = DataLoader(
			eval_dataset, 
			batch_size=self.training_args.per_device_train_batch_size, 
			collate_fn=default_data_collator, 
			num_workers=self.training_args.dataloader_num_workers, 
			pin_memory=self.training_args.dataloader_pin_memory
		)
		
		## Build optimizer
		self.optimizer = optim.Adam(self.model.parameters(), lr=self.training_args.learning_rate)
		
		## Loss function
		#loss_fct = nn.L1Loss() ## L1
		self.loss_fct = nn.MSELoss() ## L2
		
		## Freeze embedding layer
		for name, sub_module in self.model.named_modules():
			#if name.startswith("embeddings"):
			if "embed" in name:
				for param in sub_module.parameters():
					param.requires_grad = False
		
		all_param_num = sum([p.nelement() for p in self.model.parameters()]) 
		trainable_param_num = sum([
			p.nelement()
			for p in self.model.parameters()
			if p.requires_grad == True
		])
		print("All       parameters: {}".format(all_param_num))
		print("Trainable parameters: {}".format(trainable_param_num))

	def train(self):
		print("\nStart training...")
		
		## Training loops
		best_loss = float("inf")
		for epoch in range(int(self.training_args.num_train_epochs)):
			train_losses, eval_losses = [], []
		
			## Train on all batches
			self.model.train()
			for batch in tqdm(self.train_dataloader, desc="Epoch {:2d} Training".format(epoch)):
				input_ids = batch["input_ids"].to(self.device)
				attn_mask = batch["attention_mask"].to(self.device)
		
				self.model.zero_grad()
				## Forward, return loss
				train_loss = self.model(input_ids, attn_mask)
				train_loss = train_loss.mean()
				train_loss.backward()
				self.optimizer.step()
		
				train_losses.append(train_loss.detach().cpu().numpy())
		
			## Evaluation
			self.model.eval()
			with torch.no_grad():
				for batch in tqdm(self.eval_dataloader, desc="Epoch {:2d} Evaluation".format(epoch)):
					input_ids = batch["input_ids"].to(self.device)
					attn_mask = batch["attention_mask"].to(self.device)
					eval_loss = self.model(input_ids, attn_mask)
					eval_loss = eval_loss.mean()
					eval_losses.append(eval_loss.detach().cpu().numpy())
		
			train_losses = np.array(train_losses)
			eval_losses  = np.array(eval_losses)
		
			## Display results
			print("Epoch {:2d} | Train Loss: {:.4f} | Eval Loss: {:.4f}".format(epoch, np.sum(train_losses), np.sum(eval_losses)))
		
			## Save checkpoint
			if np.sum(eval_losses) < best_loss:
				print("Saving model with best reconstruction loss!")
				#ckpt_path = "{}/anomaly_scorer.pt".format(training_args.output_dir)
				#ckpt_path = "{}/anomaly_scorer_bart.pt".format(training_args.output_dir)
				#ckpt_path = "{}/anomaly_scorer_rd.pt".format(self.training_args.output_dir)
				#ckpt_path = "{}/anomaly_scorer_test.pt".format(self.training_args.output_dir)
				#ckpt_path = "{}/autoencoder_rd.pt".format(self.training_args.output_dir)
				ckpt_path = "{}/autoencoder_rd_{}.pt".format(self.training_args.output_dir, self.model_args.target_class_ext_ae)
				torch.save(self.model.state_dict(), ckpt_path)
		
				best_loss = np.sum(eval_losses)

		with open("{}/../overall_results.txt".format(self.training_args.output_dir), "a") as fw:
			fw.write("{:4s}\t{:.4f}\n".format(self.data_args.fold, best_loss))

