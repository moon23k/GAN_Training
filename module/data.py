import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, split):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = self.load_data(split)


    @staticmethod
    def load_data(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        x = self.tokenizer.encode(self.data[idx]['x']).ids
        y = self.tokenizer.encode(self.data[idx]['y']).ids
        return torch.LongTensor(x), torch.LongTensor(y)



class Collator(object):
    def __init__(self, pad_id, is_dis):
        self.pad_id = pad_id
        self.is_dis = is_dis


    def __call__(self, batch):
        x_batch, y_batch = zip(*batch)

        if not self.is_dis:
            return {'x': self.pad_batch(x_batch), 
                    'y': self.pad_batch(y_batch)}
        

        ### collate logic for discriminator pretraining
        batch_size = x_batch.size(0)

        x_batch = torch.stack([x_batch, y_batch], dim=0)
        x_batch = self.pad_batch(x_batch)

        y_batch = torch.cat((torch.zeros(batch_size), 
                             torch.ones(batch_size)), dim=0)

        indice = torch.randperm(batch_size * 2)

        return {'x': x_batch(indice), 
                'y': y_batch(indice)}


    def pad_batch(self, batch):
        return pad_sequence(
            batch, 
            batch_first=True, 
            padding_value=self.pad_id
        )



def load_dataloader(config, tokenizer, split, shuffle):
    pad_id = config.pad_id
    is_dis = 'dis' in split
    batch_size = config.batch_size // 4 \
                 if split == 'test' \
                 else config.batch_size 

    return DataLoader(
        Dataset(tokenizer, split), 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=Collator(pad_id, is_dis),
        pin_memory=True,
        num_workers=2
    )