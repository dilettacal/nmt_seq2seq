"""
Credits for some parts of this source code:

Class Seq2Seq (with or w/o Attention) - modified version from this code:
Author: Luke Melas
Title: Machine Translation with Recurrent Neural Networks
URL: https://lukemelas.github.io/machine-translation.html and https://github.com/lukemelas/Machine-Translation/blob/master/models/Seq2seq.py (on the courtesy of the author)

Class ContextSeq2Seq - modified version from this code:

Author Ben Trevett:
Title: 2 - Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.ipynb
Implements paper (Cho et al. 2014): https://arxiv.org/abs/1406.1078
URL: https://github.com/bentrevett/pytorch-seq2seq/blob/master/2%20-%20Learning%20Phrase%20Representations%20using%20RNN%20Encoder-Decoder%20for%20Statistical%20Machine%20Translation.ipynb

"""
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

from project.experiment.setup_experiment import Experiment
from project.model.decoders import Decoder, ContextDecoder
from project.model.encoders import Encoder
from project.model.layers import Attention, Maxout
from settings import VALID_CELLS, SEED

"""
Parameters:
- Maxout (500 units): 73,303,508
- 4 layers, 500, 500: 57,270,504
- 4 layers, 1000, 1000: 146,511,004
"""

random.seed(SEED)

class Seq2Seq(nn.Module):
    def __init__(self, experiment_config: Experiment, tokens_bos_eos_pad_unk):
        super(Seq2Seq, self).__init__()

        self.model_type = experiment_config.model_type
        self.decoder_type = experiment_config.decoder_type
        self.hid_dim = experiment_config.hid_dim
        self.emb_size = experiment_config.emb_size
        self.src_vocab_size = experiment_config.src_vocab_size
        self.trg_vocab_size = experiment_config.trg_vocab_size
        self.dp = experiment_config.dp
        self.enc_bi = experiment_config.bi
        self.num_layers = experiment_config.nlayers
        self.bos_token = tokens_bos_eos_pad_unk[0]
        self.eos_token = tokens_bos_eos_pad_unk[1]
        self.pad_token = tokens_bos_eos_pad_unk[2]
        self.unk_token = tokens_bos_eos_pad_unk[3]
        self.device = experiment_config.get_device()
        self.reverse_input = experiment_config.reverse_input
        rnn_type = experiment_config.rnn_type
        self.cell = rnn_type
        self.context_model = False
        self.att_type = experiment_config.attn

        assert rnn_type.lower() in VALID_CELLS, "Provided cell type is not supported!"

        self.encoder = Encoder(self.src_vocab_size, self.emb_size, self.hid_dim, self.num_layers,
                               dropout_p=self.dp, bidirectional=self.enc_bi, rnn_cell=rnn_type, device=self.device)

        self.decoder = Decoder(self.trg_vocab_size, self.emb_size, self.hid_dim,
                                   self.num_layers * 2 if self.enc_bi else self.num_layers, rnn_cell=rnn_type,
                                   dropout_p=self.dp)

        self.attention = Attention(pad_token=self.pad_token, bidirectional=self.enc_bi, attn_type=self.att_type, h_dim=self.hid_dim)

        self.linear1 = nn.Linear(2*self.hid_dim, self.emb_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(experiment_config.dp)
        self.linear2 = nn.Linear(self.emb_size, self.trg_vocab_size) #emb size of target

        ### This part is used in the original code
        if False and self.decoder.embedding.weight.size() == self.linear2.weight.size():
            print('Weight tying!')
            self.linear2.weight = self.decoder.embedding.weight

    def init_weights(self, func=None):
        if not func:
            pass
        else:
            self.apply(func)

        ### create encoder and decoder

    def forward(self, src, trg):
        src = src.to(self.device)
        trg = trg.to(self.device)

        if self.reverse_input:
            inv_index = torch.arange(src.size(0)-1,-1,-1).long()
            inv_index = inv_index.to(self.device)
            src = src.index_select(0, inv_index)

        # Encode
        out_e, final_e = self.encoder(src)
        # Decode
        out_d, final_d = self.decoder(trg, final_e) #[seq_len, bs, hid_dim], [num_layers, bs, hid_dim]

        # Attend
        context = self.attention(src, out_e, out_d) #seq_len, bs, hid_dim
        out_cat = torch.cat((out_d, context), dim=2)
        # Predict (returns probabilities)
        x = self.linear1(out_cat)
        x = self.dropout(self.tanh(x))
        x = self.linear2(x)
        return x

    #### Original code #####
    def predict(self, src, beam_size=1, max_len=30, remove_tokens=[]):
        '''Predict top 1 sentence using beam search. Note that beam_size=1 is greedy search.'''
        beam_outputs = self.beam_search(src, beam_size, max_len=max_len, remove_tokens=remove_tokens)  # returns top beam_size options (as list of tuples)
        top1 = beam_outputs[0][1]  # a list of word indices (as ints)
        return top1

    def beam_search(self, src, beam_size, max_len, remove_tokens=[]):
        '''Returns top beam_size sentences using beam search. Works only when src has batch size 1.

            Slightly modified from: https://lukemelas.github.io/machine-translation.html
        '''
        src = src.to(self.device)
        # Encode
        outputs_e, states = self.encoder(src)  # batch size = 1
        if self.context_model:
            context = states
        else: context = None
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
                    ### shape: [1] --> [1,1], needed as we are unrolling the decoder with bs 1
                    last_word_input = torch.LongTensor([last_word]).view(1, 1).to(self.device)
                    outputs_d, new_state = self.decoder(last_word_input, current_state)
                    # Attend
                    context = self.attention(src, outputs_e, outputs_d)
                    out_cat = torch.cat((outputs_d, context), dim=2)
                    x = self.linear1(out_cat)
                    ###########################################
                    x = self.dropout(self.tanh(x))
                    x = self.linear2(x)
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


####### Use this function to set up a model from the main script #####
#### Factory method to generate the model ####
def get_nmt_model(experiment_config: Experiment, tokens_bos_eos_pad_unk):
    model_type = experiment_config.model_type
    if model_type == "custom":
        if experiment_config.bi and experiment_config.reverse_input:
            experiment_config.reverse_input = False
        return Seq2Seq(experiment_config, tokens_bos_eos_pad_unk)

    elif model_type == "s":
        #### This returs a model like in Sutskever et al. ####
        #### The architecture was multilayered, thus layers are automatically set to 2 and input sequences were reversed (this is handled in the vocabulary class)
        if not experiment_config.reverse_input: experiment_config.reverse_input = True
        if experiment_config.nlayers < 2: experiment_config.nlayers = 2
        return Seq2Seq(experiment_config, tokens_bos_eos_pad_unk)
