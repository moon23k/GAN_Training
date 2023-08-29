import torch
import torch.nn as nn
from collections import namedtuple
from .components import Embeddings, Encoder




class DecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        tgt,
        memory=None,
        e_mask=None,
        d_mask=None,
        use_cache=False
    ):

        if not use_cache:
            return super().forward(
                tgt,
                memory,
                memory_key_padding_mask=e_mask,
                tgt_mask=d_mask
            )

        # This part is adapted from the official Pytorch implementation
        # So that only the last token gets modified and returned.

        tgt_last_tok = tgt[:, -1:, :]


        # self attention part
        tmp_tgt = self.self_attn(
            tgt_last_tok, tgt, tgt,
            attn_mask=None,  # not needed because we only care about the last token
            key_padding_mask=d_mask,
        )[0]

        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
        tgt_last_tok = self.norm1(tgt_last_tok)


        # encoder-decoder attention
        if memory is not None:
            tmp_tgt = self.multihead_attn(
                tgt_last_tok, memory, memory,
                attn_mask=d_mask,
                key_padding_mask=e_mask,
            )[0]

            tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
            tgt_last_tok = self.norm2(tgt_last_tok)

        # final feed-forward network
        tmp_tgt = self.linear2(
            self.dropout(self.activation(self.linear1(tgt_last_tok)))
        )
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)
        
        return tgt_last_tok



class Decoder(nn.TransformerDecoder):

    def forward(
        self,
        x,
        memory=None,
        cache=None,
        e_mask=None,
        d_mask=None,
        use_cache=True
    ):

        output = x

        #In case of not using Cache
        if not use_cache:
            for layer in self.layers:
                output = layer(output, memory, e_mask, d_mask, False)

            return output

        #In case of using Cache
        new_token_cache = []
        for idx, layer in enumerate(self.layers):
            output = layer(output, memory)
            new_token_cache.append(output)
            if cache is not None:
                output = torch.cat([cache[:, idx], output], dim=1)

        if cache is None:
            new_cache = torch.stack(new_token_cache, dim=1)
        else:
            new_cache = torch.cat(
                [cache, torch.stack(new_token_cache, dim=1)], 
                dim=0
            )


        return output, new_cache




class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()

        self.device = config.device
        self.bos_id = config.bos_id
        self.pad_id = config.pad_id
        self.max_len = config.max_len
        self.vocab_size = config.vocab_size

        self.enc_emb = Embeddings(config)
        self.encoder = Encoder(config)

        self.dec_emb = Embeddings(config)
        self.decoder = Decoder(
            DecoderLayer(
                d_model=config.hidden_dim, 
                nhead=config.n_heads, 
                dim_feedforward=config.pff_dim,
                batch_first=True
            ),
            num_layers=config.n_layers,
        )

        self.generator = nn.Linear(config.hidden_dim, self.vocab_size)

        self.out = namedtuple('Out', 'logit loss')
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_id, 
            label_smoothing=0.1
        )


    @staticmethod
    def shift_y(y):
        return y[:, :-1], y[:, 1:]


    def pad_mask(self, x):
        return x == self.pad_id


    def dec_mask(self, x):
        sz = x.size(1)
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)


    def encode(self, x, x_mask):
        x = self.enc_emb(x)
        x = self.encoder(x, x_mask)
        return x


    def decode(self, x, memory, cache=None, e_mask=None, d_mask=None, use_cache=False):
        x = self.dec_emb(x)
        x = self.decoder(x, memory, cache, e_mask, d_mask, use_cache)
        return x


    def forward(self, x, y):
        y, label = self.shift_y(y)
        
        e_mask = self.pad_mask(x)
        d_mask = self.dec_mask(y)

        memory = self.encode(x, e_mask)

        dec_out = self.decode(y, memory, None, e_mask, d_mask, use_cache=False)
        logit = self.generator(dec_out)

        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )

        return self.out


    def generate(self, x, y=None):

        batch_size = x.size(0)
        max_len = self.max_len if y is None else y.size(1)

        pred = torch.zeros((batch_size, max_len), dtype=torch.long)
        pred = pred.fill_(self.pad_id).to(self.device)
        pred[:, 0] = self.bos_id

        cache=None
        e_mask = self.pad_mask(x)
        memory = self.encode(x, e_mask)

        for idx in range(1, max_len):
            y = pred[:, :idx]
            d_out, cache = self.decode(y, memory, cache, e_mask, use_cache=True)
            logit = self.generator(d_out)
            pred_token = logit.argmax(dim=-1)[:, -1]
            pred[:, idx] = pred_token

        return pred