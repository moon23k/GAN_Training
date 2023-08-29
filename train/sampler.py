import json, torch
from module import load_generator



class Sampler:
    def __init__(self, config, tokenizer, train_dataloader, valid_dataloader):
        
        config.mode = 'pretrain' #Change config.mode to get pretrained generator model
        self.model = load_generator(config)
        self.model.eval()
        config.mode = 'train' #Revert original config.mode

        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = config.device



    def generate_samples(self):
        train_samples = self._generate(self.train_dataloader)
        valid_samples = self._generate(self.valid_dataloader)        
        
        self.save_sample(train_samples, 'train_sample')
        self.save_sample(train_samples, 'valid_sample')



    def _generate(self, dataloader):
        samples = []
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch['src'].to(self.device)
                max_len = batch['trg'].size(1)

                pred = self.model.generate(x, max_len).tolist()
                pred = self.tokenizer.batch_decode(pred)

        return samples


    @staticmethod
    def save_sample(data_obj, f_name):
        with open(f'data/{f_name}.json', 'w') as f:
            json.dump(data_obj, f)