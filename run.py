import os, yaml, argparse, torch

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from module import (
    load_dataloader,
    load_generator, 
    load_discriminator,
    Tester
)

from train import (
    GenTrainer,
    DisTrainer,
    Trainer
)



def set_seed(SEED=42):
    import random
    import numpy as np
    import torch.backends.cudnn as cudnn

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True



class Config(object):
    def __init__(self, args):    

        with open('config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            for group in params.keys():
                for key, val in params[group].items():
                    setattr(self, key, val)

        self.mode = args.mode

        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' \
                           if use_cuda and self.mode != 'inference' \
                           else 'cpu'
        self.device = torch.device(self.device_type)

        self.g_ckpt = 'ckpt/generator.pt'
        self.d_ckpt = 'ckpt/discriminator.pt'        
        self.tokenizer_path = 'data/tokenizer.json'


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def load_tokenizer(config):
    assert os.path.exists(config.tokenizer_path)

    tokenizer = Tokenizer.from_file(config.tokenizer_path)    
    tokenizer.post_processor = TemplateProcessing(
        single=f"{config.bos_token} $A {config.eos_token}",
        special_tokens=[(config.bos_token, config.bos_id), 
                        (config.eos_token, config.eos_id)]
        )
    
    return tokenizer



def main(args):
    set_seed(42)
    config = Config(args)    
    tokenizer = load_tokenizer(config)


    if 'train' in config.mode:
        g_model = load_generator(config)
        d_model = load_discriminator(config)
        train_dataloader = load_dataloader(config, tokenizer, 'train')
        valid_dataloader = load_dataloader(config, tokenizer, 'valid')
    
    elif config.mode == 'test':
        g_model = load_generator(config)
        test_dataloader = load_dataloader(config, tokenizer, 'test')



    if config.mode == 'pretrain':
        GenTrainer(config, g_model, train_dataloader, valid_dataloader).train()
        DisTrainer(config, d_model, train_dataloader, valid_dataloader).train()
        
    elif config.mode == 'train':
        trainer = Trainer(config, g_model, d_model, tokenizer, train_dataloader, valid_dataloader)
        trainer.train()

    elif config.mode == 'test':
        tester = Tester(config, g_model, tokenizer, test_dataloader)
        tester.test()
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)

    args = parser.parse_args()
    assert args.mode.lower() in ['pretrain', 'train', 'test']

    if args.mode != 'pretrain':
        assert os.path.exists('ckpt/generator.pt')
        assert os.path.exists('ckpt/discriminator.pt')

    main(args)