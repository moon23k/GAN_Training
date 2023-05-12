import json, torch
from torch.utils.data import DataLoader



class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, split=None):
        super().__init__()

        self.mode = config.mode
        self.model_type = config.model_type

        self.data = self.load_data(split)


    def load_data(self, split=None):
        with open(f_name, 'r') as f:
            data = json.load(f)

        return data


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        if self.mode == 'pretrain' and self.model_type == 'discriminator':
            uttr = self.data[idx]['src']
            resp = self.data[idx]['trg']
            pred = self.data[idx]['pred']
            return uttr, resp, pred
        
        else:
            uttr = self.data[idx]['src']
            resp = self.data[idx]['trg']            
            return uttr, resp




def load_dataloader(config, split=None):
    return DataLoader(Dataset(config, split), 
                      batch_size=config.batch_size, 
                      shuffle=True if 'train' in config.mode else False,
                      num_workers=2,
                      pin_memory=True)