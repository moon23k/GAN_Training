import math
import torch
import torch.nn as nn



def train_epoch(model, dataloader, criterion, optimizer, config):
    model.train()
    epoch_loss = 0
    total_len = len(dataloader)

    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        src, trg = batch[0].to(config.device), batch[1].to(config.device)

        pred = model(src, trg[:, :-1])
        pred = pred.contiguous().view(-1, config.output_dim)            
        
        trg = trg[:, 1:].contiguous().view(-1)            
        loss = criterion(pred, trg)
        loss.backward()

        nn.utils.clip_grad_norm(model.parameters(), max_norm=config.clip)
        optimizer.step()
        epoch_loss += loss.item()

        if (i + 1) % 1000 == 0:
            print(f"---- Train Step: {i+1}/{total_len} Train Loss: {loss:.3f}")

    return epoch_loss / total_len



def eval_epoch(model, dataloader, criterion, config):
    model.eval()
    epoch_loss = 0
    total_len = len(dataloader)

    for i, batch in enumerate(dataloader):
        src, trg = batch[0].to(config.device), batch[1].to(config.device)

        with torch.no_grad():
            pred = model(src, trg[:, :-1])

        pred = pred.contiguous().view(-1, config.output_dim)                    
        trg = trg[:, 1:].contiguous().view(-1)            
        loss = criterion(pred, trg)
        epoch_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f"---- Eval Step: {i+1}/{total_len} Eval Loss: {loss:.3f}")

    return epoch_loss / total_len

