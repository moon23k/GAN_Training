import copy
import torch
import torch.nn as nn
from common_layers import *




class Seq_Encoder(nn.Module):
    def __init__(self, config):
        super(Seq_Encoder, self).__init__()

        self.layers = get_clones(EncoderLayer(config), config.n_layers)


    def forward(self, src, src_mask):
        for layer in self.layers:
            src = layer(src, src_mask)

        return src




class Doc_Encoder(nn.Module):
    def __init__(self, config):
        super(Doc_Encoder, self).__init__()

        self.layers = get_clones(EncoderLayer(config), config.n_layers)


    def forward(self, src, src_mask):
        for layer in self.layers:
            src = layer(src, src_mask)

        return src





class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.layers = get_clones(DecoderLayer(config), config.n_layers)


    def forward(self, memory, trg, src_mask, trg_mask):
        for layer in self.layers:
            trg, attn = layer(memory, trg, src_mask, trg_mask)
        
        return trg, attn




class Hierarchical_Transformer(nn.Module):
    def __init__(self, config):
        super(Vanilla_Transformer, self).__init__()

        self.embedding = TransformerEmbedding(config)
        self.seq_encoder = Seq_Encoder(config)
        self.doc_encoder = Doc_Encoder(config)
        self.decoder = Decoder(config)
        
        self.fc_out = nn.Linear(config.hidden_dim, config.output_dim)
        self.device = config.device


    def forward(self, src, trg):
        doc = torch.tensor()
        for seq in src:
            seq_mask = create_src_mask(seq)
            seq = self.embedding(seq)
            seq = self.Seq_Encoder(seq, seq_mask)

        src_mask = create_src_mask(src)
        trg_mask = create_trg_mask(trg)

        src, trg = self.embedding(src), self.embedding(trg) 

        enc_out = self.encoder(src, src_mask)
        dec_out, _ = self.decoder(enc_out, trg, src_mask, trg_mask)

        out = self.fc_out(dec_out)

        return out
        