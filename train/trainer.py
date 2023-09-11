import time, json, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau



class TrainerBase:
    def __init__(self, config):
        
        self.mode = config.mode
        self.clip = config.clip
        self.device = config.device
        self.max_len = config.max_len
        self.n_epochs = config.n_epochs
        self.device_type = config.device_type
        self.scaler = torch.cuda.amp.GradScaler()
        self.iters_to_accumulate = config.iters_to_accumulate

        self.early_stop = config.early_stop
        self.patience = config.patience        


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"


    @staticmethod
    def save_ckpt(epoch, ckpt, model, optimizer):
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    ckpt)

        


class Trainer(TrainerBase):
    def __init__(
        self, config, g_model, d_model, 
        train_dataloader, valid_dataloader
    ):
        
        super(Trainer, self).__init__(config)

        self.g_model = g_model
        self.d_model = d_model

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        
        self.g_optimizer = AdamW(params=self.g_model.parameters(), lr=config.lr)
        self.d_optimizer = AdamW(params=self.d_model.parameters(), lr=config.lr)

        self.g_scheduler = ReduceLROnPlateau(self.g_optimizer, patience=2)
        self.d_scheduler = ReduceLROnPlateau(self.d_optimizer, patience=2)

        self.g_ckpt = config.g_ckpt
        self.d_ckpt = config.d_ckpt

        self.record_path = 'ckpt/train.json'
        self.record_keys = ['epoch', 'g_train_loss', 'd_train_loss', 
                            'g_valid_loss', 'd_valid_loss', 
                            'g_lr', 'd_lr', 'epoch_time']



    def print_epoch(self, record_dict):
        print(f"""Epoch {record_dict['epoch']}/{self.n_epochs} | \
              Time: {record_dict['epoch_time']}""".replace(' ' * 14, ''))

        print(f"""  >> Generator Train Loss: {record_dict['g_train_loss']:.3f}     | \
              Generator Valid Loss: {record_dict['g_valid_loss']:.3f}""".replace(' ' * 14, ''))

        print(f"""  >> Discriminator Train Loss: {record_dict['d_train_loss']:.3f} | \
              Discriminator Valid Loss: {record_dict['d_valid_loss']:.3f}\n""".replace(' ' * 14, ''))



    def get_losses(self, batch):
        x, y = batch['x'].to(self.device), batch['y'].to(self.device)

        with torch.autocast(device_type=self.device_type, dtype=torch.float16):
            g_loss = self.generator(x, y).loss
            g_pred = self.generator.generate(x)
            d_loss = self.discriminator(g_pred, y).loss

        #Loss Accumulate
        g_loss = g_loss / self.iters_to_accumulate
        d_loss = d_loss / self.iters_to_accumulate

        return g_loss, d_loss



    def train(self):
        records = []
        patience = self.patience
        prev_loss, g_best_loss, d_best_loss = float('inf'), float('inf'), float('inf')

        for epoch in range(1, self.n_epochs + 1):
            start_time = time.time()

            record_vals = [epoch, *self.train_epoch(), *self.valid_epoch(), 
                           self.g_optimizer.param_groups[0]['lr'],
                           self.d_optimizer.param_groups[0]['lr'],
                           self.measure_time(start_time, time.time())]

            record_dict = {k: v for k, v in zip(self.record_keys, record_vals)}
            
            records.append(record_dict)
            self.print_epoch(record_dict)
            
            g_curr_loss = record_dict['g_valid_loss']
            d_curr_loss = record_dict['d_valid_loss']

            self.g_scheduler.step(g_curr_loss)
            self.d_scheduler.step(d_curr_loss)


            #save best discriminator states
            if d_best_loss >= d_curr_loss:
                d_best_loss = d_curr_loss
                self.save_ckpt(epoch, self.d_ckpt, self.d_model, self.d_optimizer)


            #save best generator states
            if g_best_loss >= g_curr_loss:
                g_best_loss = g_curr_loss
                self.save_ckpt(epoch, self.g_ckpt, self.g_model, self.g_optimizer)

            #Early Stopping Process
            if self.early_stop:
                if prev_loss > g_curr_loss:
                    patience = self.patience
            
                else:
                    patience -= 1
                    if not patience:
                        print('--- Training Ealry Stopped ---\n')
                        break

                prev_loss = g_curr_loss


        #save train_records
        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)        
            


    def train_epoch(self):
        g_epoch_loss, d_epoch_loss = 0, 0
        tot_len = len(self.train_dataloader)
        
        self.g_model.train()
        self.d_model.train()


        for idx, batch in enumerate(self.train_dataloader):
            
            idx += 1
            g_loss, d_loss = self.get_losses(batch)
            g_loss /= self.iters_to_accumulate
            d_loss /= self.iters_to_accumulate

            g_loss.backward()
            self.scaler.scale(d_loss).backward()

            if (idx % self.iters_to_accumulate == 0) or (idx == tot_len):
                self.scaler.unscale_(self.d_optimizer)

                #Gradient Clipping
                nn.utils.clip_grad_norm_(self.g_model.parameters(), max_norm=self.clip)
                nn.utils.clip_grad_norm_(self.d_model.parameters(), max_norm=self.clip)
                
                #Gradient Update & Scaler Update
                self.g_optimizer.step()
                self.scaler.step(self.d_optimizer)
                
                self.scaler.update()
                
                self.g_optimizer.zero_grad()
                self.d_optimizer.zero_grad()

            g_epoch_loss += g_loss.item()
            d_epoch_loss += d_loss.item()
        
        g_epoch_loss = round(g_epoch_loss / tot_len, 3)
        d_epoch_loss = round(d_epoch_loss / tot_len, 3)
    
        return g_epoch_loss, d_epoch_loss
    


    def valid_epoch(self):
        g_epoch_loss, d_epoch_loss = 0, 0
        tot_len = len(self.valid_dataloader)

        self.g_model.eval()
        self.d_model.eval()
        
        with torch.no_grad():
            for batch in self.valid_dataloader:          
                g_loss, d_loss = self.get_losses(batch)

                g_epoch_loss += g_loss.item()
                d_epoch_loss += d_loss.item()
    
        g_epoch_loss = round(g_epoch_loss / tot_len, 3)
        d_epoch_loss = round(d_epoch_loss / tot_len, 3)

        return g_epoch_loss, d_epoch_loss
