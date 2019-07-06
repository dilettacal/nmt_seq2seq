"""
Class: Decoder

Handles the decoder part of a Seq2Seq model.
The decoder can be custom with lstm and gru cells.
"""

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
