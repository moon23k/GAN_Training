import time, math, json, torch

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .trainer import TrainerBase
from module import load_dataloader




class GenTrainer(TrainerBase):
    def __init__(self, config, model, tokenizer):
        super(GenTrainer, self).__init__(config)
        
        self.model = model
        self.train_dataloader = load_dataloader(config, tokenizer, 'train', shuffle=True)
        self.valid_dataloader = load_dataloader(config, tokenizer, 'valid', shuffle=True)

        self.optimizer = AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        self.ckpt = config.g_ckpt
        self.record_path = self.ckpt.replace('.pt', '.json')
        self.record_keys = ['epoch', 'train_loss', 'train_ppl',
                            'valid_loss', 'valid_ppl', 
                            'learning_rate', 'train_time']


    def print_epoch(self, record_dict):
        print(f"""Epoch {record_dict['epoch']}/{self.n_epochs} | \
              Time: {record_dict['train_time']}""".replace(' ' * 14, ''))
        
        print(f"""  >> Train Loss: {record_dict['train_loss']:.3f} | \
              Train PPL: {record_dict['train_ppl']:.2f}""".replace(' ' * 14, ''))

        print(f"""  >> Valid Loss: {record_dict['valid_loss']:.3f} | \
              Valid PPL: {record_dict['valid_ppl']:.2f}\n""".replace(' ' * 14, ''))
                            

    def train(self):
        records = []
        prev_loss, best_loss = float('inf'), float('inf')
        patience = self.patience

        for epoch in range(1, self.n_epochs + 1):
            start_time = time.time()

            record_vals = [epoch, *self.train_epoch(), *self.valid_epoch(), 
                           self.optimizer.param_groups[0]['lr'],
                           self.measure_time(start_time, time.time())]
            record_dict = {k: v for k, v in zip(self.record_keys, record_vals)}
            
            records.append(record_dict)
            self.print_epoch(record_dict)
            
            val_loss = record_dict['valid_loss']
            self.scheduler.step(val_loss)

            #save best model
            if best_loss > val_loss:
                best_loss = val_loss
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                            self.ckpt)

            #Early Stopping Process
            if self.early_stop:
                if prev_loss > val_loss:
                    patience = self.patience
            
                else:
                    patience -= 1
                    if not patience:
                        print('--- Training Ealry Stopped ---\n')
                        break

                prev_loss = val_loss

            
        #save train_records
        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)


    def train_epoch(self):
        self.model.train()
        tot_len = len(self.train_dataloader)
        epoch_loss = 0


        for idx, batch in enumerate(self.train_dataloader):
            idx += 1
            x, y = batch['x'].to(self.device), batch['y'].to(self.device)

            with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                loss = self.model(x, y).loss                
                loss = loss / self.iters_to_accumulate
            
            #Backward Loss
            self.scaler.scale(loss).backward()        
            
            if (idx % self.iters_to_accumulate == 0) or (idx == tot_len):
                #Gradient Clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                
                #Gradient Update & Scaler Update
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            epoch_loss += loss.item()
        
        epoch_loss = round(epoch_loss / tot_len, 3)
        epoch_ppl = round(math.exp(epoch_loss), 3) 

        return epoch_loss, epoch_ppl
    

    def valid_epoch(self):
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for batch in self.valid_dataloader:
                x, y = batch['x'].to(self.device), batch['y'].to(self.device)
                
                with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                    loss = self.model(x, y).loss

                epoch_loss += loss.item()
        
        epoch_loss = round(epoch_loss / len(self.valid_dataloader), 3)
        epoch_ppl = round(math.exp(epoch_loss), 3)        
        
        return epoch_loss, epoch_ppl