import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F



def create_src_mask(src, pad_idx=1):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    src_mask.to(src.device)
    return src_mask



def create_trg_mask(trg, pad_idx=1):
    trg_pad_mask = (trg != pad_idx).unsqueeze(1).unsqueeze(2)
    trg_sub_mask = torch.tril(torch.ones((trg.size(-1), trg.size(-1)))).bool()

    trg_mask = trg_pad_mask & trg_sub_mask.to(trg.device)
    trg_mask.to(trg.device)
    return trg_mask



def get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])




class TransformerEmbedding(nn.Module):
	def __init__(self, config):
		super(TransformerEmbedding, self).__init__()

		self.tok_emb = TokenEmbedding(config)
		self.pos_enc = PosEncoding(config)
		self.dropout = nn.Dropout(config.dropout_ratio)

	def forward(self, x):
		tok_emb = self.tok_emb(x)
		pos_enc = self.pos_enc(x)
		out = self.dropout(tok_emb + pos_enc)
		
		return out




class TokenEmbedding(nn.Module):
	def __init__(self, config):
		super(TokenEmbedding, self).__init__()

		self.embedding = nn.Embedding(config.input_dim, config.emb_dim)
		self.scale = torch.sqrt(torch.FloatTensor([config.emb_dim])).to(config.device)


	def forward(self, x):
		out = self.embedding(x)
		out = out * self.scale

		return out




class PosEncoding(nn.Module):
	def __init__(self, config, max_len=5000):
		super(PosEncoding, self).__init__()
		
		pe = torch.zeros(max_len, config.emb_dim)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, config.emb_dim, 2) * (-math.log(10000.0) / config.emb_dim))
		
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		self.pe = pe.to(config.device)

	
	def forward(self, x):
		out = self.pe[:x.size(1), :].detach()
		return out



class MultiHeadAttn(nn.Module):
	def __init__(self, config):
		super(MultiHeadAttn, self).__init__()

		assert config.hidden_dim % config.n_heads == 0

		self.hidden_dim = config.hidden_dim
		self.n_heads = config.n_heads
		self.head_dim = config.hidden_dim // config.n_heads

		self.fc_q = nn.Linear(config.hidden_dim, config.hidden_dim)
		self.fc_k = nn.Linear(config.hidden_dim, config.hidden_dim)
		self.fc_v = nn.Linear(config.hidden_dim, config.hidden_dim)
		
		self.fc_out = nn.Linear(config.hidden_dim, config.hidden_dim)
		


	def forward(self, query, key, value, mask=None):
		Q, K, V = self.fc_q(query), self.fc_k(key), self.fc_v(value)
		Q, K, V = self.split(Q), self.split(K), self.split(V)

		out = self.calc_attn(Q, K, V, mask)
		out = self.concat(out)
		out = self.fc_out(out)

		return out


	def calc_attn(self, query, key, value, mask=None):
		d_k = key.size(-1)
		attn_score = torch.matmul(query, key.permute(0, 1, 3, 2))
		attn_score = attn_score / math.sqrt(d_k)

		if mask is not None:
			attn_score = attn_score.masked_fill(mask==0, -1e9)

		attn_prob = F.softmax(attn_score, dim=-1)
		attn = torch.matmul(attn_prob, value)
		
		return attn


	def split(self, x):
		batch_size = x.size(0)
		out = x.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
		
		return out


	def concat(self, x):
		batch_size = x.size(0)
		out = x.permute(0, 2, 1, 3).contiguous()
		out = out.view(batch_size, -1, self.hidden_dim)	
		
		return out




class PositionwiseFFN(nn.Module):
	def __init__(self, config):
		super(PositionwiseFFN, self).__init__()

		self.fc_1 = nn.Linear(config.hidden_dim, config.pff_dim)
		self.fc_2 = nn.Linear(config.pff_dim, config.hidden_dim)
		self.dropout = nn.Dropout(config.dropout_ratio)


	def forward(self, x):
		out = self.fc_1(x)
		out = self.dropout(F.relu(out))
		out = self.fc_2(out)

		return out




class ResidualConn(nn.Module):
	def __init__(self, config):
		super(ResidualConn, self).__init__()

		self.layer_norm = nn.LayerNorm(config.hidden_dim, eps=1e-6, elementwise_affine=True)
		self.dropout = nn.Dropout(config.dropout_ratio)


	def forward(self, x, sub_layer):
		out = x + sub_layer(x)
		out = self.layer_norm(out)
		out = self.dropout(out)
		
		return out




class EncoderLayer(nn.Module):
	def __init__(self, config):
		super(EncoderLayer, self).__init__()

		self.m_attn = MultiHeadAttn(config)
		self.pff = PositionwiseFFN(config)
		self.residual_conn = get_clones(ResidualConn(config), 2)

	def forward(self, src, src_mask):
		out = self.residual_conn[0](src, lambda x: self.m_attn(src, src, src, src_mask))
		out = self.residual_conn[1](out, self.pff)
		return out




class DecoderLayer(nn.Module):
	def __init__(self, config):
		super(DecoderLayer, self).__init__()
		
		self.m_attn = MultiHeadAttn(config)
		self.pff = PositionwiseFFN(config)
		self.residual_conn = get_clones(ResidualConn(config), 3)


	def forward(self, memory, trg, src_mask, trg_mask):
		out = self.residual_conn[0](trg, lambda x: self.m_attn(trg, trg, trg, trg_mask))
		attn = self.residual_conn[1](out, lambda x: self.m_attn(out, memory, memory, src_mask))
		out = self.residual_conn[2](attn, self.pff)

		return out, attn
