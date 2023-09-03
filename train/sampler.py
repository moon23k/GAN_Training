import json, torch
from tqdm import tqdm
from module import load_generator, load_dataloader



class Sampler:
    def __init__(self, config, tokenizer):
        
        orig_mode = config.mode
        config.mode = 'test'                #Change mode to get pretrained generator model
        self.model = load_generator(config)
        self.model.eval()
        config.mode = orig_mode             #Revert original mode


        self.tokenizer = tokenizer
        self.device = config.device
        self.train_dataloader = load_dataloader(config, tokenizer, 'train', shuffle=False)
        self.valid_dataloader = load_dataloader(config, tokenizer, 'valid', shuffle=False)



    def generate_samples(self):
        train_samples = self._generate(self.train_dataloader)
        valid_samples = self._generate(self.valid_dataloader)        
        
        self.save_sample(train_samples, 'dis_train')
        self.save_sample(train_samples, 'dis_valid')



    def _generate(self, dataloader):
        samples = []

        for batch in tqdm(dataloader):
            x = batch['src'].to(self.device)
            label = batch['trg'].tolist()

            with torch.no_grad():
                sample = self.model.generate(x).tolist()
            
            sample = self.tokenizer.batch_decode(sample)
            label = self.tokenzier.batch_decode(label)

            samples.append({'label': label, 'sample': sample})

        return samples


    @staticmethod
    def save_sample(data_obj, f_name):
        with open(f'data/{f_name}.json', 'w') as f:
            json.dump(data_obj, f)