"""
Class: Decoder

Handles the decoder part of a Seq2Seq model.
The decoder can be custom with lstm and gru cells.
Or it can propagate context to the embeddings and to the final output layer, as proposed in Cho et al. (2014).

"""

import torch
from torch import nn



class Decoder(nn.Module):
    """
    This is a general decoder.
    It can be used in the Seq2Seq class. It can be multilayered and works on both lstm and gru cells.
    """
    def __init__(self, trg_vocab_size, embedding, h_dim, num_layers, dropout_p=0.0, rnn_cell="lstm"):
        super(Decoder, self).__init__()
        self.vocab_size, self.embedding_size = trg_vocab_size, embedding
        self.num_layers = num_layers
        self.h_dim = h_dim
        self.dropout_p = dropout_p

        # Create word embedding, LSTM
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        if rnn_cell.lower() == "lstm":
            self.rnn = nn.LSTM(self.embedding_size, self.h_dim, self.num_layers,
                               dropout=self.dropout_p if self.num_layers > 1 else 0)
        else:
            self.rnn = nn.GRU(self.embedding_size, self.h_dim, self.num_layers,
                              dropout=self.dropout_p if self.num_layers > 1 else 0)

        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x, h0):
        x = self.embedding(x) #1,1,256
        x = self.dropout(x)
        out, h = self.rnn(x, h0)
        return out, h


class ContextDecoder(Decoder):
    """
    This decoder only works with the ContextSeq2Seq class.
    It works for a 1-layer GRU model and can propagate a fixed context from the encoder to both the embedding projects and raw outputs.

    The ContextDecoder is always unrolled  - slightly modified version of this Decoder:
    https://github.com/bentrevett/pytorch-seq2seq/blob/master/2%20-%20Learning%20Phrase%20Representations%20using%20RNN%20Encoder-Decoder%20for%20Statistical%20Machine%20Translation.ipynb
    """
    def __init__(self, trg_vocab_size, emb_size, h_dim, num_layers, dropout_p=0.0):
        super().__init__(trg_vocab_size, emb_size, h_dim, num_layers, dropout_p, rnn_cell="gru")

        self.rnn = nn.GRU(self.embedding.embedding_dim + h_dim, h_dim, num_layers=num_layers,
                          dropout=dropout_p if num_layers > 1 else 0)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, h0, context=None, val=False):
        ### context: [1,batch_size, hid_dim], 1 is the number of layers
        #### This boolean check when decoder is unrolled during the beam search, as the input does not need to be unsqueezed
        if val:
            ### during the training and validation the input is unsqueezed to be concatenated with the context
            x = x.unsqueeze(0)
        embedded = self.dropout(self.embedding(x))
        # x: [1, batch_size] or [batch_size]
        emb_con = torch.cat((embedded, context), dim=2)
        output, h = self.rnn(emb_con, h0)  # 1,64,500
        ### output shape [batch_size, trg_vocab]
        # squeeze --> tensor: [1, emb_dim] or [1, hid_dim]
        output = torch.cat((embedded.squeeze(0), h.squeeze(0), context.squeeze(0)),
                           dim=1)
        ### output shape consistent with the shape of the normal decoder: [1, batch_size, fina_dim]

        output = output.unsqueeze(0)
        return output, h

