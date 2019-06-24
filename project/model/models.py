import random

import torch
import torch.nn as nn

from project.model.decoders import Decoder, ContextDecoder
from project.model.encoders import Encoder
from project.model.layers import MaxoutLinearLayer
from settings import DEFAULT_DEVICE

LSTM = "lstm"
GRU = "gru"
VALID_CELLS = [LSTM, GRU]
VALID_MODELS = ["standard", "sutskever", "cho"]

class Seq2Seq(nn.Module):
    def __init__(self, args, tokens_bos_eos_pad_unk):
        super(Seq2Seq, self).__init__()

        self.hid_dim = args.hs
        self.num_layers = args.nlayers
        self.bos_token = tokens_bos_eos_pad_unk[0]
        self.eos_token = tokens_bos_eos_pad_unk[1]
        self.pad_token = tokens_bos_eos_pad_unk[2]
        self.unk_token = tokens_bos_eos_pad_unk[3]
        self.device = args.cuda
        rnn_type = args.rnn

        assert rnn_type.lower() in VALID_CELLS, "Provided cell type is not supported!"

        self.encoder = Encoder(args.src_vocab_size, args.emb, args.hs, args.nlayers,
                               dropout_p=args.dp, bidirectional=args.bi, rnn_cell=rnn_type, device=self.device)
        self.decoder = Decoder(args.trg_vocab_size, args.emb, args.hs,
                               args.nlayers * 2 if args.bi else args.nlayers, dropout_p=args.dp)
        self.dropout = nn.Dropout(args.dp)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(self.hid_dim, args.trg_vocab_size)

        ### create encoder and decoder
    def forward(self, src, trg):
        src = src.to(self.device)
        trg = trg.to(self.device)

        # Encode
        out_e, final_e = self.encoder(src)
        # Decode
        out_d, _ = self.decoder(trg, final_e)
        x = self.dropout(self.tanh(out_d))
        x = self.output(x)
        return x

    def predict(self, src, beam_size=1, max_len =30, remove_tokens=[]):
        '''Predict top 1 sentence using beam search. Note that beam_size=1 is greedy search.'''
        beam_outputs = self.beam_search(src, beam_size, max_len=max_len, remove_tokens=remove_tokens)  # returns top beam_size options (as list of tuples)
        top1 = beam_outputs[0][1]  # a list of word indices (as ints)
        return top1

    def beam_search(self, src, beam_size, max_len, remove_tokens=[]):
        '''Returns top beam_size sentences using beam search. Works only when src has batch size 1.'''
        src = src.to(self.device)
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
    def __init__(self, args, tokens_bos_eos_pad_unk, maxout_units=None):
        super(ChoSeq2Seq, self).__init__(args, tokens_bos_eos_pad_unk)

        if maxout_units:
            self.output = MaxoutLinearLayer(input_dim=self.emb_dim_trg * 2, hidden_units=maxout_units,
                                         output_dim=self.vocab_size_trg, k=2)
        else:
            self.output = nn.ReLU(nn.Linear(self.embedding.embedding_dim + self.hid_dim * 2, self.vocab_size))
