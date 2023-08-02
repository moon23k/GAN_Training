import os, torch
import torch.nn as nn
from transformers import (
    T5Config, 
    T5EncoderModel,
    T5ForConditionalGeneration
)



def update_model_config(config, model_type):
    if model_type == 'generator':
        custom_config = {}
    elif model_type == 'discriminator':
        custom_config = {}

    model_config = T5Config()
    model_config.update(custom_config)

    return model_config



def print_model_desc(model):
    def count_params(model):
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return params

    def check_size(model):
        param_size, buffer_size = 0, 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    print(f"--- Model Params: {count_params(model):,}")
    print(f"--- Model  Size : {check_size(model):.3f} MB\n")



def load_generator(config):
    if config.mode == 'pretrain':
        generator = T5ForConditionalGeneration.from_pretrained(config.g_mname)
        print(f"Generator for {config.mode.upper()} has loaded")
        print_model_desc(generator)
        return generator.to(config.device)

    generator_config = T5Config.from_pretrained(config.g_mname)
    generator = T5ForConditionalGeneration(generator_config)
    print(f"Generator for {config.mode.upper()} has loaded")

    ckpt = config.g_ckpt
    assert os.path.exists(ckpt)

    generator_state = torch.torch.load(ckpt, map_location=config.device)['model_state_dict']
    generator.load_state_dict(generator_state)

    print(f"Model States has loaded from {ckpt}")
    print_model_desc(generator)

    return generator.to(config.device)




def load_discriminator(config):

    discriminator = Discriminator(config)
    print(f"Discriminator for {config.mode.upper()} has loaded")
    
    if config.mode == 'pretrain':
        print_model_desc(discriminator)
        return discriminator.to(config.device)

    ckpt = config.g_ckpt    
    assert os.path.exists(ckpt)
    
    model_state = torch.load(config.d_base_ckpt, map_location=config.device)['model_state_dict']        
    discriminator.load_state_dict(model_state)
    print(f"Model States has loaded from {ckpt}")        
    print_model_desc(discriminator)

    return discriminator.to(config.device)