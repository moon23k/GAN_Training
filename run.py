import os, yaml, argparse, torch

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from module import (
    load_dataloader,
    load_generator, 
    load_discriminator,
    Tester,
    Translator
)

from train import (
    GenTrainer,
    Sampler,
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
    generator = load_generator(config)
    tokenizer = load_tokenizer(config)


    if 'train' in config.mode:
        #Common Setups for Pretraining and Training Processes
        train_dataloader = load_dataloader(config, 'train')
        valid_dataloader = load_dataloader(config, 'valid')        
        discriminator = load_discriminator(config)

        #PreTraining Process
        if config.mode == 'pretrain':
            #Pretrain Generator
            gen_trainer = GenTrainer(config, generator, train_dataloader, valid_dataloader)
            gen_trainer.train()

            #Generate Samples to pretrain Discriminator
            sampler = Sampler(config, tokenizer, train_dataloader, valid_dataloader)
            sampler.generate_samples()

            dis_trainer = DisTrainer(config, discriminator, train_dataloader, valid_dataloader)
            dis_trainer.train()

        #Training Process
        elif config.mode == 'train':
            trainer = Trainer(config, generator, discriminator, train_dataloader, valid_dataloader)
            trainer.train()

    if config.mode == 'test':
        test_dataloader = load_dataloader(config, 'test')
        tester = Tester(config, generator, tokenizer, test_dataloader)
        tester.test()


    if config.mode == 'inference':
        translator = Translator(config, model, tokenizer)
        translator.translate()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-search', default='greedy', required=False)

    args = parser.parse_args()
    assert args.mode.lower() in ['pretrain', 'train', 'test', 'inference']
    assert args.search.lower() in ['greedy', 'beam']

    if args.mode != 'pretrain':
        assert os.path.exists('ckpt/generator.pt')
        assert os.path.exists('ckpt/discriminator.pt')

    main(args)