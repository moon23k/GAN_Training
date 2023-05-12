import os, json, argparse, torch
from module.data import load_dataloader
from module.model import load_generator, load_discriminator
from module.train import Trainer
from module.pretrain import GenTrainer, DisTrainer
from module.test import Tester
from tqdm import tqdm
from transformers import set_seed, AutoTokenizer



class Config(object):
    def __init__(self, args):    

        self.mode = args.mode
        
        self.g_mname = "t5-small"
        self.d_mname = "prajjwal1/bert-small"        


        self.early_stop = True
        self.patience = 3
        self.clip = 1
        self.lr = 5e-5
        self.max_len = 128
        self.n_epochs = 10
        self.batch_size = 32
        self.iters_to_accumulate = 4

        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' if use_cuda else 'cpu'

        if self.mode == 'inference':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(self.device_type)

        self.g_ckpt = 'ckpt/generator.pt'
        self.d_ckpt = 'ckpt/discriminator.pt'
        self.g_base_ckpt = 'ckpt/generator_base.pt'
        self.d_base_ckpt = 'ckpt/discriminator_base.pt'


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def load_tokenizers(config):
    g_tokenizer = AutoTokenizer.from_pretrained(config.g_mname, model_max_length=config.max_len)

    if config.mode == 'inference':
        d_tokenizer = None
    else:
        d_tokenizer = AutoTokenizer.from_pretrained(config.d_mname, model_max_length=config.max_len)

    return g_tokenizer, d_tokenizer



def generate(config, model, tokenizer, split):

    #Load Pretrained Generator Model States
    model_state = torch.torch.load(config.g_base_ckpt, map_location=config.device)['model_state_dict']
    model.load_state_dict(model_state)
    model.eval()

    config.mode = 'generate'
    dataloader = load_dataloader(config)

    generated = []
    for batch in tqdm(dataloader):
        uttr, resp = batch[0], batch[1]
        batch_size = len(uttr)

        uttr_encodings = tokenizer(uttr, padding=True, truncation=True, 
                                   return_tensors='pt').to(config.device)        
        
        with torch.no_grad():
            pred = model.generate(input_ids=uttr_encodings.input_ids,
                                  attention_mask=uttr_encodings.attention_mask,
                                  use_cache=True)

        pred = tokenizer.batch_decode(pred, skip_special_tokens=True)

        for i in range(batch_size):
            generated.append({'src': uttr[i], 
                              'trg': resp[i], 
                              'pred': pred[i]})

    config.mode = 'pretrain'
    with open(f'data/dis_{split}.json', 'w') as f:
        json.dump(generated, f)



def pretrain(config, g_model, d_model, g_tokenizer, d_tokenizer):

    ###PreTrain Generator with Character Dataset    
    g_train_dataloader = load_dataloader(config, 'train')
    g_valid_dataloader = load_dataloader(config, 'valid')

    g_trainer = GenTrainer(config, g_model, g_tokenizer, g_train_dataloader, g_valid_dataloader)
    g_trainer.train()


    ###Generate Samples to PreTrain Discriminator
    generate(config, g_model, g_tokenizer)
    

    ###PreTrain Discriminator
    config.model_type = 'discriminator'
    d_train_dataloader = load_dataloader(config, 'train')
    d_valid_dataloader = load_dataloader(config, 'valid')        

    d_trainer = DisTrainer(config, d_model, d_tokenizer, d_train_dataloader, d_valid_dataloader)
    d_trainer.train()




def train(config, g_model, d_model, g_tokenizer, d_tokenizer):
    train_dataloader = load_dataloader(config, 'train')
    valid_dataloader = load_dataloader(config, 'valid')

    trainer = Trainer(config, g_model, d_model, g_tokenizer, 
                      d_tokenizer, train_dataloader, valid_dataloader)
    trainer.train()



def test(config, g_model, d_model, g_tokenizer, d_tokenizer):
    test_dataloader = load_dataloader(config, 'test')
    tester = Tester(config, g_model, d_model, g_tokenizer, d_tokenizer, test_dataloader)    
    tester.test()    



def inference(g_model, g_tokenizer):
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
        input_ids = g_tokenizer(input_seq, return_tensors='pt')['input_ids']
        output_ids = g_model.generate(input_ids, max_new_tokens=128, use_cache=True)
        output_seq = g_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        #Search Output Sequence
        print(f"Model Out Sequence >> {output_seq}")



def main(args):

    set_seed(42)
    config = Config(args)    
    g_tokenizer, d_tokenizer = load_tokenizers(config)
    setattr(config, 'pad_id', g_tokenizer.pad_token_id)

    g_model = load_generator(config)
    d_model = load_discriminator(config)


    if config.mode == 'pretrain':
        pretrain(config, g_model, d_model, g_tokenizer, d_tokenizer)
    elif config.mode == 'train':
        train(config, g_model, d_model, g_tokenizer, d_tokenizer)
    elif config.mode == 'test':
        test(config, g_model, d_model, g_tokenizer, d_tokenizer)
    elif config.mode == 'inference':
        inference(g_model, g_tokenizer)
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)

    
    args = parser.parse_args()
    assert args.mode.lower() in ['pretrain', 'train', 'test', 'inference']

    if args.mode == 'train':
        assert os.path.exists('ckpt/generator_base.pt')
        assert os.path.exists('ckpt/discriminator_base.pt')
    
    elif args.mode in ['test', 'inference']:
        assert os.path.exists('ckpt/discriminator.pt')
        assert os.path.exists('ckpt/generator.pt')

    main(args)