# Author   : Orange
# Coding   : Utf-8
# @Time    : 2021/11/4 1:37 下午
# @File    : Seq2SeqAttnBahdanau.py
import torch
from torch import nn
import torch.nn.functional as F

"""

"""


class EncoderGRU(nn.Module):
	"""

	hidden_{-1} = [0] * hidden_dim
	"""

	def __init__(self, encoder_vocab_size: int, encoder_hidden_dim: int, decoder_hidden_dim: int, n_layer=1,
	             bidirectional=False, dropout=0.5):
		super(EncoderGRU, self).__init__()
		self.bidirectional = bidirectional
		self.embedding = nn.Embedding(encoder_vocab_size, encoder_hidden_dim)
		self.dropout = nn.Dropout(p=dropout)
		self.gru = nn.GRU(encoder_hidden_dim, encoder_hidden_dim, n_layer, bidirectional=self.bidirectional)
		if self.bidirectional:
			self.fc = nn.Linear(2 * encoder_hidden_dim, decoder_hidden_dim)
		else:
			self.fc = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)

	def forward(self, src):
		# words_inputs : [seq length, batch size]
		seq_len = len(src)
		embedded = self.dropout(self.embedding(src))

		# output : [seq length, batch size, bidirectional * encoder_hidden_dim]
		# hidden : [bidirectional * n_layer, batch size, encoder_hidden_dim]

		# output[0] 对应第一层 前向的最后一个输出
		# output[1] 对应第二层 后项的最后一个输出

		# hidden [-2, :, : ] is the last of the forwards RNN
		# hidden [-1, :, : ] is the last of the backwards RNN
		output, hidden = self.gru(embedded)

		# hidden = [batch size, decoder_hidden_dim]
		if self.bidirectional:
			hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=-1)))
		else:
			hidden = torch.tanh(self.fc(hidden[-1, :, :]))

		return output, hidden


class BahdanauAttention(nn.Module):
	"""
	q: s_{i-1}
	k, v: h
	采用的为 加性注意力机制 V * tanh(W * p + U * q)

	"""

	def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int, bidirectional=False):
		super(BahdanauAttention, self).__init__()
		self.D = 2 if bidirectional else 1
		self.attn = nn.Linear(self.D * encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
		self.V = nn.Linear(decoder_hidden_dim, 1, bias=False)

	def forward(self, hidden, encoder_output):
		"""
		hidden : [batch size, decoder_hidden_dim]
		encoder_output : [seq length, batch size, bidirectional * encoder_hidden_dim]
		"""
		batch_size = encoder_output.shape[1]
		src_len = encoder_output.shape[0]

		hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
		encoder_output = encoder_output.permute(1, 0, 2)

		# energy : [batch size, seq length, decoder_hidden_dim]
		energy = torch.tanh(self.attn(torch.cat((encoder_output, hidden), dim=-1)))

		attention = self.V(energy).squeeze(-1)

		# return [batch size, seq length]
		return F.softmax(attention, dim=-1)


class DecoderGRU(nn.Module):
	def __init__(self, decoder_vocab_size: int, encoder_hidden_dim: int, decoder_hidden_dim: int, attention, dropout=0.5,
	             tie_weight=False):
		super(DecoderGRU, self).__init__()

		self.embedding = nn.Embedding(decoder_vocab_size, decoder_hidden_dim)
		self.attention = attention





if __name__ == '__main__':
	gru = EncoderGRU(100, 5, 2, bidirectional=False)
	gru(torch.tensor([[1], [2], [4]], dtype=torch.long))
