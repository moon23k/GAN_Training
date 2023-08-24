import json, torch
import torch.nn as nn
from .trainer import TrainerBase



class PreTrainer(TrainerBase):
    def __init__(
    	self, config, g_model, d_model, 
        tokenizer, train_dataloader, valid_dataloader
	    ):
        
        super(PreTrainer, self).__init__(config)


    def train(self):
    	self.train_generator()
    	self.generate()
    	self.train_discriminator()
    	return


    def train_generator(self):
    	return


    def train_discriminator(self):
    	return


    def generate(self):
    	return