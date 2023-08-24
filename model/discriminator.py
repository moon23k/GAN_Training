import torch
import torch.nn as nn
from .components import Embeddings, Encoder



class Discriminator(nn.Module):
	def __init__(self, config):
		super(Discriminator, self).__init__()
		self.encoder = Encoder(config)
		self.classifier = nn.Linear(config.hidden_size, 1)

	def forward(self, x):
		return