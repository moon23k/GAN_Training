import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import sentencepice as spm


def read_data(f_name):
    with open(f'data/{f_name}', 'r') as f:
        data = json.load(f)
    return data


class SumDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.cls_id = tokenizer.bos_token
        self.sep_id = tokenizer.sep_token


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['src'], self.data[idx]['trg']



def _collate_fn(data_batch):    
    src_batch, trg_batch = [], []

    for batch in data_batch:
        src = torch.tensor(batch[0], dtype=torch.long)
        trg = torch.tensor(batch[1], dtype=torch.long)
        src_batch.append(src)
        trg_batch.append(trg)

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=1)
    trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=1)

    return src_batch, trg_batch



def get_dataloader(split, config):        
    data = read_data(split)
    dataset = CustomDataset(data)
    iterator = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=_collate_fn, num_workers=2)
    
    return iterator