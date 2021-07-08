#!/usr/bin/env python
# # -*- coding: utf-8 -*-
""" Encoder for Sequence to Sequence models """
__author__ = "shubhamagarwal92"

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch_utils as torch_utils

class EncoderRNN(nn.Module):
	def __init__(self, embedmatrix, enc_emb_size, enc_hidden_size, 
				rnn_type='GRU', num_layers=1, batch_first=True,
				dropout=0, bidirectional=True):
		super(EncoderRNN, self).__init__()
		self.enc_hidden_size = enc_hidden_size
		self.batch_first = batch_first
		self.num_layers = num_layers
		# self.num_directions = 2 if self.bidirectional else 1 # In case if we half encoder size
		self.embedding = nn.Embedding.from_pretrained(embedmatrix, freeze=False)
		# Warapper to handle both LSTM and GRU
		self.rnn_cell = torch_utils.rnn_cell_wrapper(rnn_type)
		self.encoder = self.rnn_cell(enc_emb_size, enc_hidden_size, num_layers = num_layers, 
									batch_first=batch_first, dropout=dropout, 
									bidirectional=bidirectional)

	def forward(self, input_seq, seq_length, hidden = None):
		sorted_lens, len_ix = seq_length.sort(0, descending=True)
		inv_ix = len_ix.clone()
		inv_ix.data[len_ix.data] = torch.arange(0, len(len_ix)).type_as(inv_ix.data)
		sorted_inputs = input_seq[len_ix].contiguous()

		embedded = self.embedding(sorted_inputs) # input_seq = (batch,seq_length)
		packed_embbed = pack(embedded, list(sorted_lens.data), batch_first=self.batch_first)
		output, hidden = self.encoder(packed_embbed)
		# If we want to provide initial hidden state use 
		# output, hidden = self.encoder(packed_embbed, hidden)
		output, output_length = unpack(output, batch_first=self.batch_first)
        
		output = output[inv_ix].contiguous()
		hidden = hidden[:, inv_ix.data, ].contiguous()
		return output, hidden
