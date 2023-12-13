import os, yaml, argparse, torch

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from module import (
    load_dataloader,
    load_model,
    PreTrainer,
    Trainer,
    Tester,
    Inferencer
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
        
        self.task = args.task
        self.mode = args.mode
        self.search_method = args.search

        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' \
                           if use_cuda and self.mode != 'inference' \
                           else 'cpu'
        self.device = torch.device(self.device_type)

        self.gen_ckpt = f'ckpt/{self.task}/gen_model.pt'
        self.gen_pre_ckpt = f'ckpt/{self.task}/pre_gen_model.pt'

        self.dis_ckpt = f'ckpt/{self.task}/dis_model.pt'
        self.dis_pre_ckpt = f'ckpt/{self.task}/pre_dis_model.pt'
        
        self.tokenizer_path = f'data/{self.task}/tokenizer.json'


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
    mode = args.mode
    config = Config(args)    
    
    tokenizer = load_tokenizer(config)
    generator = load_model(config, 'gen')
    discriminator = load_model(config, 'dis')
    

    if 'train' in mode:
        train_dataloader = load_dataloader(config, tokenizer, 'train')
        valid_dataloader = load_dataloader(config, tokenizer, 'valid')
        
        trainer_args = {
            'config': config,
            'generator': generator,
            'discriminator': discriminator,
            'train_dataloader': train_dataloader,
            'valid_dataloader': valid_dataloader
        }

        trainer = Trainer(**trainer_args) if mode == 'train' else PreTrainer(**trainer_args)
        trainer.train()

    elif config.mode == 'test':
        test_dataloader = load_dataloader(config, tokenizer, 'test')
        tester = Tester(config, generator, tokenizer, test_dataloader)
        tester.test()
        return


    elif config.mode == 'inference':
        inferencer = Inferencer(config, generator, tokenizer)
        inferencer()
        return




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-mode', required=True)
    parser.add_argument('-search', default='greedy', required=False)

    args = parser.parse_args()
    assert args.task.lower() in ['translation', 'dialogue']
    assert args.mode.lower() in ['pretrain', 'train', 'test', 'inference']
    assert args.search.lower() in ['greedy', 'beam']

    if args.mode == 'train':
        assert os.path.exists(f'ckpt/{args.task}/pre_gen_model.pt')
        assert os.path.exists(f'ckpt/{args.task}/pre_dis_model.pt')
    if args.mode in ['test', 'inference']:
        assert os.path.exists(f'ckpt/{args.task}/gen_model.pt')

    main(args)