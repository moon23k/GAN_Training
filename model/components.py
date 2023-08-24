import math, torch
import torch.nn as nn




class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        hidden_dim = config.hidden_dim

        pe = torch.zeros(config.max_len, hidden_dim)
        position = torch.arange(0, config.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)



class Embeddings(nn.Module):
    def __init__(self, config):
        self.tok_emb = nn.Embedding(config)
        self.scale_factor
        self.pos_emb = PositionalEncoding(config)

        self.tok_dropout = nn.Dropout(config.dropout_ratio)
        self.pos_dropout = nn.Dropout(config.dropout_ratio)


    def forward(self, x):
        return



def generate_square_subsequent_mask(sz: int, device: str = "cpu") -> torch.Tensor:
    """ Generate the attention mask for causal decoding """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    ).to(device=device)
    return mask



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.emb = Embeddings(config)

    def forward(self, x):
        x = self.emb(x)
        return    