import torch
import torch.nn as nn
from collections import namedtuple
from .components import Encoder



class Discriminator(nn.Module):
	def __init__(self, config):
		super(Discriminator, self).__init__()
		
		self.device = config.device
		self.pad_id = config.pad_id

		self.encoder = Encoder(config)
		self.classifier = nn.Linear(config.hidden_size, 1)

        self.out = namedtuple('Out', 'logit loss')
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_id, 
            label_smoothing=0.1
        ).to(self.device)


	def forward(self, x, y):
		x_mask = x == self.pad_id
		memory = self.encoder(x, x_mask)
		logit = self.classifier(memory)

        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )

        return self.out