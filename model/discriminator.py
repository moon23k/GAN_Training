import torch
import torch.nn as nn
from collections import namedtuple
from .components import Embeddings, Encoder



class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        
        self.device = config.device
        self.pad_id = config.pad_id

        self.enc_emb = Embeddings(config)
        self.encoder = Encoder(config)
        self.fc_out = nn.Linear(config.hidden_dim, 1)

        self.out = namedtuple('Out', 'logit loss')
        self.criterion = nn.CrossEntropyLoss()



    def forward(self, x, y):

        x_mask = (x == self.pad_id)
        x = self.enc_emb(x)
        x = self.encoder(x, x_mask)
        logit = self.fc_out(x)

        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            y.contiguous().view(-1)
        )

        return self.out