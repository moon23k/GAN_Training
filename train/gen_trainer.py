import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .trainer import TrainerBase


class GenTrainer(TrainerBase):
	def __init__(self, config, model, train_dataloader, valid_dataloader):
		super(GenTrainer, self).__init__(config)
		
		self.model = model
		self.train_dataloader = train_dataloader
		self.valid_dataloader = valid_dataloader

		self.optimizer
		self.lr_scheduler

		self.ckpt


	def train(self):
		return


	def train_epoch(self):
		return


	def valid_epoch(self):
		return