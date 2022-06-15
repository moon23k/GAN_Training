import torch
import torch.nn as nn
from transformers import TransfoXLModel

from models.common_layers import Ranker
from models.vanilla import Vanilla_Transformer
from models.universal import Universal_Transformer
from model.hierarchical import Hierarchical_Transformer



def init_xavier(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data)



def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def load_model(config, ranker=False):
    if ranker:
    	model = Ranker(config)
    	model.to(config.device)
    	model.apply(init_xavier)
    	return model


    if config.model == 'vanilla':
        model = Vanilla_Transformer(config)
    
    elif config.model == 'universal':
        model = Universal_Transformer(config)
    
    elif config.model == 'hierarchical':
    	model = Hierarchical_Transformer(config)
    
    elif config.model == 'extra':
    	model = TransfoXLModel(config)


    model.to(config.device)
    model.apply(init_xavier)
    print(f'{config.model} Transformer model has loaded.\nThe model has {count_params(model):,} parameters\n')
    return model

