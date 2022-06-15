import math
from torch.optim import lr_scheduler



#Custom CosineAnnealingWarmUpRestarts scheduler
#code borrowed from https://gaussian37.github.io/dl-pytorch-lr_scheduler
class CosineAnnealingWarmUpRestarts(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0=5, T_mult=1, eta_max=1e-3, T_up=5, gamma=0.7, last_epoch=-1):
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs

        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]


    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr




class NoamLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps=5):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]



def get_scheduler(scheduler_name, optimizer):
    if scheduler_name == 'noam':
        scheduler = NoamLR(optimizer)

    elif scheduler_name == 'cosine_annealing_warm':
        scheduler = CosineAnnealingWarmUpRestarts(optimizer)

    elif scheduler_name == 'cosine_annealing':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-9)
        
    elif scheduler_name == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    elif scheduler_name == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    
    return scheduler