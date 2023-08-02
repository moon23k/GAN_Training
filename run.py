import os, yaml, argparse, torch

from module.data import load_dataloader
from module.model import load_generator, load_discriminator
from module.pretrain import GenTrainer, DisTrainer
from module.train import Trainer
from module.test import Tester

from module.generate import generate

from transformers import set_seed, AutoTokenizer




class Config(object):
    def __init__(self, args):    

        with open('config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            for group in params.keys():
                for key, val in params[group].items():
                    setattr(self, key, val)

        self.mode = args.mode
        self.search_method = args.search

        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' \
                           if use_cuda and self.mode != 'inference' \
                           else 'cpu'
        self.device = torch.device(self.device_type)

        self.g_ckpt = 'ckpt/generator.pt'
        self.d_ckpt = 'ckpt/discriminator.pt'        
        self.tokenizer_path = f'data/{self.task}/tokenizer.json'


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def load_tokenizer(config):    
    return



def pretrain(config, g_model, d_model, tokenizer):

    ###PreTrain Generator with Character Dataset    
    g_train_dataloader = load_dataloader(config, tokenizer, 'train')
    g_valid_dataloader = load_dataloader(config, tokenizer, 'valid')

    g_trainer = GenTrainer(config, g_model, g_train_dataloader, g_valid_dataloader)
    g_trainer.train()


    ###Generate Samples to PreTrain Discriminator
    generate(config, g_model, tokenizer)
    

    ###PreTrain Discriminator
    config.model_type = 'discriminator'
    d_train_dataloader = load_dataloader(config, tokenizer, 'train')
    d_valid_dataloader = load_dataloader(config, tokenizer, 'valid')        

    d_trainer = DisTrainer(config, d_model, d_train_dataloader, d_valid_dataloader)
    d_trainer.train()




def train(config, g_model, d_model, tokenizer):
    train_dataloader = load_dataloader(config, tokenizer, 'train')
    valid_dataloader = load_dataloader(config, tokenizer, 'valid')

    trainer = Trainer(config, g_model, d_model, tokenizer, 
                      train_dataloader, valid_dataloader)
    trainer.train()



def test(config, g_model, d_model, tokenizer):
    test_dataloader = load_dataloader(config, 'test')
    tester = Tester(config, g_model, d_model, tokenizer, test_dataloader)    
    tester.test()    



def inference(g_model, tokenizer):
    g_model.eval()
    print(f'--- Inference Process Started! ---')
    print('[ Type "quit" on user input to stop the Process ]')
    
    while True:
        input_seq = input('\nUser Input Sequence >> ').lower()

        #End Condition
        if input_seq == 'quit':
            print('\n--- Inference Process has terminated! ---')
            break        

        #convert user input_seq into model input_ids
        input_ids = tokenizer(input_seq, return_tensors='pt')['input_ids']
        output_ids = g_model.generate(input_ids, max_new_tokens=128, use_cache=True)
        output_seq = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        #Search Output Sequence
        print(f"Model Out Sequence >> {output_seq}")



def main(args):
    set_seed(42)
    config = Config(args)    
    tokenizer = load_tokenizer(config)

    g_model = load_generator(config)
    d_model = load_discriminator(config)


    if config.mode == 'pretrain':
        pretrain(config, g_model, d_model, tokenizer)
    elif config.mode == 'train':
        train(config, g_model, d_model, tokenizer)
    elif config.mode == 'test':
        test(config, g_model, d_model, tokenizer)
    elif config.mode == 'inference':
        inference(g_model, tokenizer)
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)

    args = parser.parse_args()
    assert args.mode.lower() in ['pretrain', 'train', 'test', 'inference']

    if args.mode != 'pretrain':
        assert os.path.exists('ckpt/generator.pt')
        assert os.path.exists('ckpt/discriminator.pt')


    main(args)