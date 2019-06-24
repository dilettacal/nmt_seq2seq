import random

import torch
import torch.nn as nn

from project.model.decoders import Decoder, ChoDecoder
from project.model.encoders import Encoder
from project.model.layers import MaxoutLinearLayer
from settings import DEFAULT_DEVICE

LSTM = "lstm"
GRU = "gru"
VALID_CELLS = [LSTM, GRU]
VALID_MODELS = ["standard", "sutskever", "cho"]

class Seq2Seq(nn.Module):
    def __init__(self, embedding_src, embedding_trg, h_dim, num_layers, dropout_p,
                 bi,
                 rnn_type="lstm",
             tokens_bos_eos_pad_unk=[0, 1, 2, 3], reverse_input=False,
                 tie_emb=False, device=DEFAULT_DEVICE):
        super(Seq2Seq, self).__init__()

        self.hid_dim = h_dim
        self.num_layers = num_layers
        self.vocab_size_trg, self.emb_dim_trg = embedding_trg.size()
        self.bos_token = tokens_bos_eos_pad_unk[0]
        self.eos_token = tokens_bos_eos_pad_unk[1]
        self.pad_token = tokens_bos_eos_pad_unk[2]
        self.unk_token = tokens_bos_eos_pad_unk[3]
        self.reverse_input = reverse_input
        self.device = device
        print("Model inputs reversed: {}".format(self.reverse_input))

        assert rnn_type.lower() in VALID_CELLS, "Provided cell type is not supported!"

        self.encoder = Encoder(embedding_src, h_dim, num_layers, dropout_p=dropout_p, bidirectional=bi, rnn_cell=rnn_type, device=self.device)

        self.decoder = Decoder(embedding_trg, h_dim, num_layers * 2 if bi else num_layers, dropout_p=dropout_p)

        self.dropout = nn.Dropout(dropout_p)
        #### Here Hidden size!
        self.output = nn.Linear(self.hid_dim, self.vocab_size_trg)

        ### create encoder and decoder
    def forward(self, src, trg):
        src = src.to(self.device)
        trg = trg.to(self.device)
        # Reverse src tensor
        if self.reverse_input:
            inv_index = torch.arange(src.size(0)-1, -1, -1).long()
            inv_index = inv_index.to(self.device)
            src = src.index_select(0, inv_index)
        # Encode
        out_e, final_e = self.encoder(src)
        # Decode
        out_d, _ = self.decoder(trg, final_e)
        x = self.output(out_d)
        return x

    def predict(self, src, beam_size=1, max_len =30, remove_tokens=[]):
        '''Predict top 1 sentence using beam search. Note that beam_size=1 is greedy search.'''
        beam_outputs = self.beam_search(src, beam_size, max_len=max_len, remove_tokens=remove_tokens)  # returns top beam_size options (as list of tuples)
        top1 = beam_outputs[0][1]  # a list of word indices (as ints)
        return top1

    def predict_k(self, src, k, max_len=30, remove_tokens=[]):
        '''Predict top k possibilities for first max_len words.'''
        beam_outputs = self.beam_search(src, k, max_len=max_len,
                                        remove_tokens=remove_tokens)  # returns top k options (as list of tuples)
        topk = [option[1] for option in beam_outputs]  # list of k lists of word indices (as ints)
        return topk


    def beam_search(self, src, beam_size, max_len, remove_tokens=[]):
        '''Returns top beam_size sentences using beam search. Works only when src has batch size 1.'''
        src = src.to(self.device)
        # Reverse src tensor
        if self.reverse_input:
            inv_index = torch.arange(src.size(0) - 1, -1, -1).long()
            inv_index = inv_index.to(self.device)
            src = src.index_select(0, inv_index)
        # Encode
        outputs_e, states = self.encoder(src)  # batch size = 1
        # Start with '<s>'
        init_lprob = -1e10
        init_sent = [self.bos_token]
        best_options = [(init_lprob, init_sent, states)]  # beam
        # Beam search
        k = beam_size  # store best k options
        for length in range(max_len):  # maximum target length
            options = []  # candidates
            for lprob, sentence, current_state in best_options:
                # Prepare last word
                last_word = sentence[-1]
                if last_word != self.eos_token:
                    last_word_input = torch.LongTensor([last_word]).view(1, 1).to(self.device)
                    # Decode
                    outputs_d, new_state = self.decoder(last_word_input, current_state)
                #    print(outputs_d.size())

                    x = self.output(outputs_d)
                    x = x.squeeze().data.clone()
                    # Block predictions of tokens in remove_tokens
                    for t in remove_tokens: x[t] = -10e10
                    lprobs = torch.log(x.exp() / x.exp().sum())  # log softmax
                    # Add top k candidates to options list for next word
                    for index in torch.topk(lprobs, k)[1]:
                        option = (float(lprobs[index]) + lprob, sentence + [index], new_state)
                        options.append(option)
                else:  # keep sentences ending in '</s>' as candidates
                    options.append((lprob, sentence, current_state))
            options.sort(key=lambda x: x[0], reverse=True)  # sort by lprob
            best_options = options[:k]  # place top candidates in beam
        best_options.sort(key=lambda x: x[0], reverse=True)
        return best_options



class ChoSeq2Seq(Seq2Seq):
    def __init__(self, embedding_src, embedding_trg, h_dim, num_layers, dropout_p, bi,
                 tokens_bos_eos_pad_unk=[0, 1, 2, 3], reverse_input=False, tie_emb=False, device="cuda", maxout_units=None):
        super(ChoSeq2Seq, self).__init__(embedding_src=embedding_src, embedding_trg=embedding_trg,
                                         h_dim=h_dim, num_layers=num_layers,
                                         dropout_p=dropout_p, bi=bi, rnn_type="gru",
                                         tokens_bos_eos_pad_unk=tokens_bos_eos_pad_unk,
                                         reverse_input=reverse_input, tie_emb=tie_emb, device=device)

        if maxout_units:
            self.output = MaxoutLinearLayer(input_dim=self.emb_dim_trg * 2, hidden_units=maxout_units,
                                         output_dim=self.vocab_size_trg, k=2)
        else:
            self.output = nn.ReLU(nn.Linear(self.embedding.embedding_dim + h_dim * 2, self.vocab_size))
