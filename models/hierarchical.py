import copy
import torch
import torch.nn as nn
from common_layers import *




class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.layers = get_clones(EncoderLayer(config), config.n_layers)


    def forward(self, src, src_mask):
        #src : [batch_size, n_seq, seq_len, emb_dim]
        
        seq_matrix = src[]
        
        empty_matrix = []
        for seq_vector in seq_matrix:
            for layer in self.layers:
                _src = layer(src, src_mask)
                empty_matrix.append(_src)

        for layer in self.layers:
            src = layer(src, src_mask)

        return src





class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.layers = get_clones(DecoderLayer(config), config.n_layers)


        for layer in self.layers:
            trg, attn = layer(memory, trg, src_mask, trg_mask)
        
        return trg, attn



def split_doc(src):
    src = src.tolist()

    return splited



class Hierarchical_Transformer(nn.Module):
    def __init__(self, config):
        super(Vanilla_Transformer, self).__init__()

        self.embedding = TransformerEmbedding(config)
        self.encoder = Seq_Encoder(config)
        self.decoder = Decoder(config)
        
        self.fc_out = nn.Linear(config.hidden_dim, config.output_dim)
        self.device = config.device


    def forward(self, src, trg):
        src = src.to_list()
        src = 
        
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



class HAN(nn.Module):
    def __init__(self, config):
        super(self, HAN).__init__()

        self.word_enc = wordEncoder(config)
        self.sent_enc = sentEncoder(config)

        self.word_attn = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.u_w = nn.Linear(config.hidden_dim * 2, 1, bias=False)

        self.sent_attn = nn.Linear()
        self.u_s = nn.Linear()

        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax()
        self.fc_out = nn.Linear(config.hidden_dim * 2, config.output_dim)

        self.device = config.device

    def forward(self, document):
        word_attn_weights = []
        batch_size = document.shape(0)
        
        sent_enc_out = torch.zeros(batch_size, 2, self.hidden_dim).to(self.device)
        h0_sent = torch.zeros(2, 1, self.hidden_dim, dtype=float).to(self.device)
        
        #Batchsize만큼의 묶음으로 들어오는 document 각각에 대한 연산을 하는 상위 for loop
        for i in range(batch_size):
            sent = document[i]
            n_sents = sent.size(0)
            word_enc_out = torch.zeros(n_sents, 2, self.hidden_dim).to(self.device)
            h0_word = torch.zeros(n_sents, 2, self.hidden_dim, dtype=float).to(self.device)
            
            #각 문장 단위에서의 프로세스 / word encoder사용해서 word vector산출하고 이를 매트릭스에 차곡차곡 적립
            for j in range(n_sents):
                _, h0_word = self.word_enc(sent[j], h0_word)
                word_enc_out[j] = h0_word.squeeze()
            
            word_enc_out = word_enc_out.view(word_enc_out.size(0), -1)
            u_word = torch.tanh(self.word_attn(word_enc_out))
            word_weights = self.softmax(self.u_w(u_word))
            word_attn_weights.append(word_weights)
            sent_sum_vector = (u_word * word_weights).sum(axis=0)

            _, h0_sent = self.sent_enc(sent_sum_vector, h0_sent)
            sent_enc_out[i] = h0_sent.squeeze()
        

        #여기는 for문 바깥
        sent_enc_out = sent_enc_out.viwe(sent_enc_out.size(0), -1)
        u_sent = torch.tanh(self.sent_attn(sent_enc_out))
        sent_weights = self.softmax(self.u_s(u_sent))

        doc_sum_vector = (u_sent * sent_weights).sum(axis=0)
        out = self.fc_out(doc_sum_vector)
        out = self.log_softmax(out)

        return word_attn_weights, sent_attn_weights, out