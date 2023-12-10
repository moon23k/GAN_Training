import json, torch
import torch.nn as nn
from .train import TrainerBase



class PreTrainer(TrainerBase):
    def __init__(self, config, generator, discriminator, train_dataloader, valid_dataloader):
        super(PreTrainer, self).__init__(config, generator, discriminator, train_dataloader, valid_dataloader)

        self.set_training_attrs(generator, 'pre_gen')
        self.set_training_attrs(discriminator, 'pre_dis')


    def load_pt_generator(self):
        state = torch.load(
            f'ckpt/{self.task}/pre_gen_model.pt',
            map_location=self.device
        )['model_state_dict']

        self.generator.load_state_dict(state)
        self.generator.eval()
        print('Pretrained Generator has loaded to pretrain discriminator\n')
        

    def print_epoch(self, epoch_report, model_type):
        train_loss = f"{epoch_report['train_loss']:.3f}"
        valid_loss = f"{epoch_report['valid_loss']:.3f}"
        gpu_memory = epoch_report['gpu_memory']
        max_memory = epoch_report['gpu_max_memory']

        max_len = max(len(f"{elem:.3f}") for elem in epoch_report.values)

        elapsed_time = epoch_report['epoch_time']
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))

        txt = f"""Epoch {epoch_report['epoch']}/{self.n_epochs} | Time: {elapsed_min}m {elapsed_sec}s
            >> Train Loss: {train_loss:>{max_len}}  |  Valid Loss: {valid_loss:>{max_len}}
            >> GPU Memory: {gpu_memory:>{max_len}}  |  Max Memory: {max_memory:>{max_len}}\n"""
        print(f"[ {'Generator' if 'gen' in model_type else 'Discriminator'} ]")
        print(txt.replace(' '* 11, ''))

    
    def train(self):
        self.train_model(self.generator, 'pre_gen')
        self.load_pt_generator()
        self.train_model(self.discriminator, 'pre_dis')


    def train_model(self, model, model_type):
        records = []
        patience = self.patience
        prev_loss, best_loss = float('inf'), float('inf')

        optimizer = getattr(self, f'{model_type}_optimizer')
        lr_scheduler = getattr(self, f'{model_type}_lr_scheduler')

        torch.cuda.empty_cache()

        for epoch in range(1, self.n_epochs + 1):
            start_time, end_time = self.set_times()
            start_time.record()

            train_epoch_loss = self.train_epoch(model, optimizer, model_type)
            valid_epoch_loss = self.valid_epoch(model, model_type)

            epoch_gpu_memory, epoch_gpu_max_memory = self.get_gpu_info()

            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time) // 1000

            epoch_record = {
                'epoch': epoch,
                'epoch_time': elapsed_time,
                'train_loss': train_epoch_loss,
                'valid_loss': valid_epoch_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'gpu_memory': epoch_gpu_memory,
                'gpu_max_memory': epoch_gpu_max_memory
            }

            records.append(epoch_record)
            self.print_epoch(epoch_record, model_type)
            lr_scheduler.step(valid_epoch_loss)

            if best_loss > valid_epoch_loss:
                best_loss = valid_epoch_loss
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                            getattr(self, f'{model_type}_ckpt'))

            if self.early_stop:
                if prev_loss > valid_epoch_loss:
                    patience = self.patience
                else:
                    patience -= 1
                    if not patience:
                        print('--- Training Early Stopped ---\n')
                        break

                prev_loss = valid_epoch_loss

        with open(getattr(self, f'{model_type}_record_path'), 'w') as fp:
            json.dump(records, fp)


    def train_epoch(self, model, optimizer, model_type):
        model.train()
        tot_len = len(self.train_dataloader)
        epoch_loss = 0

        for idx, batch in enumerate(self.train_dataloader):
            idx += 1
            inputs = self.batch2inputs(batch, model_type)

            with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                loss = model(**inputs).loss
                loss = loss / self.iters_to_accumulate

            #Backward Loss
            self.scaler.scale(loss).backward()

            if (idx % self.iters_to_accumulate == 0) or (idx == tot_len):
                #Gradient Clipping
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip)

                #Gradient Update & Scaler Update
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        return round(epoch_loss / tot_len, 3)



    def valid_epoch(self, model, model_type):
        model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for batch in self.valid_dataloader:
                inputs = self.batch2inputs(batch, model_type)

                with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                    loss = model(**inputs).loss
                    epoch_loss += loss.item()

        return round(epoch_loss / len(self.valid_dataloader), 3)
