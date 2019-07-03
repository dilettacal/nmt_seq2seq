"""
Class: Encoder

Handles the encoder part of a Seq2Seq model.
The encoder can be set up with lstm or gru.
It can handle bidirectionality.

"""

import torch
from torch import nn
from settings import DEFAULT_DEVICE


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embedding, h_dim, num_layers=1, dropout_p=0.0,
                 bidirectional = False, device = DEFAULT_DEVICE, rnn_cell="lstm"):
        super(Encoder, self).__init__()
        self.vocab_size, self.embedding_size = src_vocab_size, embedding
        self.num_layers = num_layers
        self.h_dim = h_dim
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional
        self.device = device

        # Create word embedding and LSTM
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        if rnn_cell.lower() == "lstm":
            self.rnn = nn.LSTM(self.embedding_size, self.h_dim, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=False, dropout=self.dropout_p if self.num_layers > 1 else 0)
        elif rnn_cell.lower() == "gru":
            self.rnn = nn.GRU(self.embedding_size, self.h_dim, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=False,dropout=self.dropout_p if self.num_layers > 1 else 0)
        else: raise ValueError("Cell not supported!")

        self.rnn_type = rnn_cell
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, input_lengths = None):
        # Embed text
        x = self.embedding(x)
        x = self.dropout(x)

        ## initialize encoder hidden states to be 0 at every batch iteration
        num_layers = self.num_layers * 2 if self.bidirectional else self.num_layers
        init = torch.zeros(num_layers, x.size(1), self.h_dim)
        init = init.to(self.device)
        ## create tensor for hidden state
        if self.rnn_type == "lstm":
            h0 = (init, init.clone()) # h0 = (h, c)
        else:
            h0 = (init) # h0 = h

        # Pass through RNN
        if input_lengths:
            x = nn.utils.rnn.pack_padded_sequence(x, input_lengths)

        out, h = self.rnn(x, h0)  # h is a tuple, if lstm else single
        if input_lengths:
            out, _ = nn.utils.rnn.pad_packed_sequence(x, input_lengths)

        ## handling bidirectionality
        if self.bidirectional:
            out = out[:, :, :self.h_dim] + out[:, :, self.h_dim:]
        return out, h

