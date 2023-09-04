import json, torch
from tqdm import tqdm
from module import load_generator, load_dataloader



class Sampler:
    def __init__(self, config, tokenizer):
        
        orig_mode = config.mode
        config.mode = 'test'                #Change mode to get pretrained generator model
        model = load_generator(config)
        self.model = torch.compile(model)
        self.model.eval()
        config.mode = orig_mode             #Revert original mode

        self.tokenizer = tokenizer
        self.device = config.device
        self.device_type = config.device_type

        self.train_dataloader = load_dataloader(config, tokenizer, 'train', shuffle=False)
        self.valid_dataloader = load_dataloader(config, tokenizer, 'valid', shuffle=False)



    def generate_samples(self):
        train_samples = self._generate(self.train_dataloader)
        valid_samples = self._generate(self.valid_dataloader)        
        
        self.save_sample(train_samples, 'dis_train')
        self.save_sample(train_samples, 'dis_valid')


    def tokenize(self, batch):
        return [self.tokenizer.decode(x) for x in batch.tolist()]


    def _generate(self, dataloader):
        samples = []

        for batch in tqdm(dataloader):
            x = batch['x'].to(self.device)
            label = batch['y']

            with torch.no_grad():
                with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                    sample = self.model.generate(x)
            
            sample = self.tokenize(sample)
            label = self.tokenize(label)

            for l, s in zip(label, sample):
                samples.append({'x': l, 'y': s})

        return samples


    @staticmethod
    def save_sample(data_obj, f_name):
        with open(f'data/{f_name}.json', 'w') as f:
            json.dump(data_obj, f)