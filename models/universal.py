import copy
import torch
import torch.nn as nn
from common_layers import *




def block_embedding(length, dim, min_timescale=1.0, max_timescale=1.0e4):
    position = np.arange(length)

    num_timescales = dim // 2
    
    log_timescale_increment = ( math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1) )
    
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float32) * -log_timescale_increment)
    
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    
    signal = np.pad(signal, [[0, 0], [0, dim % 2]], 
                    'constant', constant_values=[0.0, 0.0])
    
    signal =  signal.reshape([1, length, dim])

    return torch.from_numpy(signal).type(torch.FloatTensor)




class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.n_layers = config.n_layers
        self.layer = EncoderLayer(config)
        self.time_signal = block_embedding(1000, config.hidden_dim)
        self.pos_signal = block_embedding(config.n_layers, config.hidden_dim)

    def forward(self, src, src_mask):
        for l in range(self.n_layers):
            src += self.time_signal[:, src.shape[1], :]
            src += self.pos_signal[:, l, :].unsqueeze(1).repeat(1, src.shape[1], 1)
            src = self.layer(src, src_mask)

        return src




class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.n_layers = config.n_layers
        self.layer = DecoderLayer(config)
        self.time_signal = block_embedding()
        self.pos_signal = block_embedding()


    def forward(self, memory, trg, src_mask, trg_mask):
        for l in range(self.n_layers):
            trg += self.time_signal[:, src.shape[1], :]
            trg += self.pos_signal[:, l, :].unsqueeze(1).repeat(1, src.shape[1], 1)
            trg, attn = self.layer(memory, trg, src_mask, trg_mask)
        
        return trg, attn



class Universal_Transformer(nn.Module):
    def __init__(self, config):
        super(Light_Transformer, self).__init__()

        self.embedding = TransformerEmbedding(config)
        self.emb_fc = nn.Linear(config.emb_dim, config.hidden_dim)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        self.fc_out = nn.Linear(config.hidden_dim, config.output_dim)
        self.device = config.device


    def forward(self, src, trg):
        src_mask = create_src_mask(src)
        trg_mask = create_trg_mask(trg)

        src, trg = self.embedding(src), self.embedding(trg) 
        
        src = self.emb_fc(src)
        trg = self.emb_fc(trg)

        enc_out = self.encoder(src, src_mask)
        dec_out, _ = self.decoder(enc_out, trg, src_mask, trg_mask)

        out = self.fc_out(dec_out)

        return out
        