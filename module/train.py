import json, torch
from typing_extensions import Required
import torch.nn as nn
import torch.amp as amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau



class TrainerBase:
    def __init__(self, config, generator, discriminator, train_dataloader, valid_dataloader):
        super(TrainerBase, self).__init__()
  
        self.task = config.task
        self.bos_id = config.bos_id
        self.vocab_size = config.vocab_size
        
        self.lr = config.lr
        self.clip = config.clip
        self.device = config.device
        self.n_epochs = config.n_epochs        

        self.generator = generator
        self.discriminator = discriminator
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader


    def set_training_attrs(self, model, prefix):
        optimizer = AdamW(model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, patience=2)

        setattr(self, f'{prefix}_optimizer', optimizer)
        setattr(self, f'{prefix}_lr_scheduler', scheduler)
        setattr(self, f'{prefix}_ckpt', f'ckpt/{self.task}/{prefix}_model.pt')
        if 'pre' in prefix:
            setattr(self, f'{prefix}_record_path', f'ckpt/{self.task}/{prefix}_records.json')


    @staticmethod
    def set_times():
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    

    @staticmethod
    def get_gpu_info():
        epoch_gpu_memory = f"{torch.cuda.memory_allocated(device=None) / 1024**3:.2f}GB"
        epoch_gpu_max_memory = f"{torch.cuda.max_memory_allocated(device=None) / 1024**3:.2f}GB"
        return epoch_gpu_memory, epoch_gpu_max_memory


    def batch2inputs(self, batch, model_type):        
        x = batch['y'].to(self.device)
        y = batch['x'].to(self.device)
        
        if 'gen' in model_type:
            return {'x': x, 'y':y}
        
        batch_size = x.size(0)

        with torch.no_grad():
            samples = self.generator(x, y).logit.argmax(-1)
        bos_batch = torch.empty(batch_size, 1, dtype=torch.long).fill_(self.bos_id).to(self.device)
        
        samples = torch.cat((bos_batch, samples), dim=-1)
        samples = torch.cat((samples, y), dim=0)

        neg_labels = torch.zeros(batch_size)
        pos_labels = torch.ones(batch_size)
        labels = torch.cat((neg_labels, pos_labels), dim=0).to(self.device)

        indice = torch.randperm(batch_size * 2)

        return {'x': samples[indice],
                'y': labels[indice]}




class Trainer(TrainerBase):
    def __init__(self, config, generator, discriminator, train_dataloader, valid_dataloader):
        super(Trainer, self).__init__(config, generator, discriminator, train_dataloader, valid_dataloader)

        self.sigmoid = nn.Sigmoid()
        self.set_training_attrs(generator, 'gen')
        self.set_training_attrs(discriminator, 'dis')
        self.record_path = f"ckpt/{self.task}/gan_repord.json"


    def print_epoch(self, epoch_report):
        gen_train_loss = f"{epoch_report['gen_train_loss']:.3f}"
        gen_valid_loss = f"{epoch_report['gen_valid_loss']:.3f}"

        dis_train_loss = f"{epoch_report['dis_train_loss']:.3f}"
        dis_valid_loss = f"{epoch_report['dis_valid_loss']:.3f}"

        gpu_memory = epoch_report['gpu_memory']
        max_memory = epoch_report['gpu_max_memory']

        max_len = max([len(f"{x:.3f}") if isinstance(x, (float)) else len(str(x)) for x in epoch_report.values()])

        elapsed_time = epoch_report['epoch_time']
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))

        txt = f"""Epoch {epoch_report['epoch']}/{self.n_epochs} | Time: {elapsed_min}m {elapsed_sec}s
            >> Gen Train Loss: {gen_train_loss:>{max_len}}  |  Gen  Valid Loss: {gen_valid_loss:>{max_len}}
            >> Dis Train Loss: {dis_train_loss:>{max_len}}  |  Dis  Valid Loss: {dis_valid_loss:>{max_len}}            
            >> GPU Memory:     {gpu_memory:>{max_len}}  |  Max Memory:      {max_memory:>{max_len}}\n"""
        print(txt.replace(' '* 11, ''))


    def get_losses(self, batch):
        gen_inputs = self.batch2inputs(batch, 'gen')
        dis_inputs = self.batch2inputs(batch, 'dis')


        dis_loss = self.discriminator(**dis_inputs).loss
        samples = self.generator(**gen_inputs).logit.argmax(-1)
    
        logit = self.sigmoid(self.discriminator(samples))
        prob = (logit > 0.5).sum() / logit.size(0)
        prob = torch.clamp(prob, 1e-4, 0.999)
        gen_loss = -torch.log(prob)
        gen_loss.requires_grad_(True)

        return gen_loss, dis_loss


    def train(self):
        records = []

        torch.cuda.empty_cache()
        for epoch in range(1, self.n_epochs + 1):
            start_time, end_time = self.set_times()
            start_time.record()

            gen_train_epoch_loss, dis_train_epoch_loss = self.train_epoch()
            gen_valid_epoch_loss, dis_valid_epoch_loss = self.valid_epoch() 

            epoch_gpu_memory, epoch_gpu_max_memory = self.get_gpu_info()

            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time) // 1000

            epoch_record = {
                #common
                'epoch': epoch,
                'epoch_time': elapsed_time,
                'gpu_memory': epoch_gpu_memory,
                'gpu_max_memory': epoch_gpu_max_memory,

                #generator                
                'gen_train_loss': gen_train_epoch_loss,
                'gen_valid_loss': gen_valid_epoch_loss,
                'gen_lr': self.gen_optimizer.param_groups[0]['lr'],

                #discriminator
                'dis_train_loss': dis_train_epoch_loss,
                'dis_valid_loss': dis_valid_epoch_loss,                
                'dis_lr': self.dis_optimizer.param_groups[0]['lr'],
            }

            records.append(epoch_record)
            self.print_epoch(epoch_record)
            
            #lr scheduler update
            self.gen_lr_scheduler.step(gen_valid_epoch_loss)
            self.dis_lr_scheduler.step(dis_valid_epoch_loss)


        torch.save({'model_state_dict': self.generator.state_dict(),
                    'optimizer_state_dict': self.gen_optimizer.state_dict()},
                    self.gen_ckpt)
        torch.save({'model_state_dict': self.discriminator.state_dict(),
                    'optimizer_state_dict': self.dis_optimizer.state_dict()},
                    self.dis_ckpt)

        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)



    def set_model_mode(self, mode):
        if mode == 'train':
            self.generator.train()
            self.discriminator.train()
        else:
            self.generator.eval()
            self.discriminator.eval()


    def train_epoch(self):
        self.set_model_mode('train')
        tot_len = len(self.train_dataloader)
        gen_epoch_loss, dis_epoch_loss = 0, 0

        for batch in self.train_dataloader:
            
            gen_loss, dis_loss = self.get_losses(batch)

            gen_loss.backward()
            dis_loss.backward()

            #Gradient Clipping
            nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=self.clip)
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.clip)

            #Gradient Update & Scaler Update
            self.gen_optimizer.step()
            self.dis_optimizer.step()

            self.gen_optimizer.zero_grad()
            self.dis_optimizer.zero_grad()

            gen_epoch_loss += gen_loss.item()
            dis_epoch_loss += dis_loss.item()

        return round(gen_epoch_loss / tot_len, 3), round(dis_epoch_loss / tot_len, 3)


    def valid_epoch(self):
        self.set_model_mode('eval')
        tot_len = len(self.valid_dataloader)
        gen_epoch_loss, dis_epoch_loss = 0, 0

        with torch.no_grad():
            for batch in self.valid_dataloader:
                gen_loss, dis_loss = self.get_losses(batch)

                gen_epoch_loss += gen_loss.item()
                dis_epoch_loss += dis_loss.item()

        return round(gen_epoch_loss / tot_len, 3), round(dis_epoch_loss / tot_len, 3)
