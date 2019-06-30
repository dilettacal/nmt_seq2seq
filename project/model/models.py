"""
Credits for some parts of this source code to this blog post:

Author: Luke Melas
Title: Machine Translation with Recurrent Neural Networks
URL: https://lukemelas.github.io/machine-translation.html

Classes:
- Seq2Seq is a vanilla Seq2Seq model. This can handle LSTMs, GRUs and bidirectional Encoders
- ContextSeq2Seq aims to replicate the model proposed by Cho et al. Learning Phrase Representations using RNN Encoder–Decoderfor Statistical Machine Translation (2014), (URL:https://arxiv.org/pdf/1406.1078.pdf)
- AttSeq2Seq is a Seq2Seq model with attention, as implemented in the blog post of Luke Melas.

All these models can handle beam search.
"""
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

from project.experiment.setup_experiment import Experiment
from project.model.decoders import Decoder, ContextDecoder
from project.model.encoders import Encoder
from project.model.utils import Beam
from settings import VALID_CELLS, SEED, TEACHER_RATIO

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
        rnn_type = experiment_config.rnn_type
        self.cell = rnn_type
        self.context_model = False
        self.attn = False

        assert rnn_type.lower() in VALID_CELLS, "Provided cell type is not supported!"

        self.encoder = Encoder(self.src_vocab_size, self.emb_size, self.hid_dim, self.num_layers,
                               dropout_p=self.dp, bidirectional=self.enc_bi, rnn_cell=rnn_type, device=self.device)

        self.decoder = Decoder(self.trg_vocab_size, self.emb_size, self.hid_dim,
                                   self.num_layers * 2 if self.enc_bi else self.num_layers, rnn_cell=rnn_type,
                                   dropout_p=self.dp)

        self.dropout = nn.Dropout(experiment_config.dp)
        self.output = nn.Linear(self.hid_dim,
                                experiment_config.trg_vocab_size)

    def init_weights(self, func=None):
        if not func:
            pass
        else:
            self.apply(func)

        ### create encoder and decoder

    def forward(self, src, trg, teacher_forcing_ratio=TEACHER_RATIO):
        src = src.to(self.device)
        trg = trg.to(self.device)

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.trg_vocab_size

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # Encode
        out_e, states = self.encoder(src)

        if self.context_model:
            context = states

        else: context = None

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        ### unrolling the decoder word by word
        used_teacher = 0
        for t in range(1, max_len):
            if self.context_model:
                out_d, states = self.decoder(input, states, context, val=True)
            else:
                out_d, states = self.decoder(input, states, val=True) #[1,64,500]
            #print("OUT size:", out_d.size())
            x = self.dropout(torch.tanh(out_d))
            x = self.output(x) #[1,batch_size,trg_vocab_size], with context dec --> [3, bs, trg]
            #print("Final:", x.size())
            outputs[t] = x
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = x.squeeze(0).max(1)[1] # x: 1, batch_size, trg_vocab_size --> [batch_size, trg_vocab_size], then select max over the probabilities --> [batch_size]
           # print("Target:", trg[t].size(), "Top1:", top1.size()) #during validation: [batch_size], during test [1], [1]
            used_teacher += 1 if teacher_force else 0
            input = (trg[t] if teacher_force else top1)
        return outputs

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
                    if self.context_model:
                        outputs_d, new_state = self.decoder(last_word_input, current_state, context, val=False)
                    else:
                        outputs_d, new_state = self.decoder(last_word_input, current_state, val=False)
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




class ContextSeq2Seq(Seq2Seq):
    """This is inspired by the "context model" as proposed by Cho et al. (2014) """
    def __init__(self, experiment_config, token_bos_eos_pad_unk):
        super().__init__(experiment_config, token_bos_eos_pad_unk)

        self.context_model = True

        self.decoder = ContextDecoder(self.trg_vocab_size, self.emb_size, self.hid_dim,
                                      self.num_layers * 2 if self.enc_bi else self.num_layers, dropout_p=self.dp)

        self.output = nn.Linear(self.emb_size + self.hid_dim*2,
                                experiment_config.trg_vocab_size)


class AttentionSeq2Seq(Seq2Seq):
    def __init__(self, experiment_config, token_bos_eos_pad_unk):
        super().__init__(experiment_config, token_bos_eos_pad_unk)
        self.attn = True
        self.context_model = False



def uniform_init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def normal_init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)


def badahnau_init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_nmt_model(experiment_config: Experiment, tokens_bos_eos_pad_unk):
    model_type = experiment_config.model_type
    decoder_type = experiment_config.decoder_type

    if decoder_type == "context":
        model_type = "c"
        experiment_config.model_type = model_type

    if model_type == "custom":
        if experiment_config.bi and experiment_config.reverse_input:
            experiment_config.reverse_input = False
        return Seq2Seq(experiment_config, tokens_bos_eos_pad_unk)

    elif model_type == "c":
        #### This returns a model like in Cho et al. #####
        #experiment_config.bi = False
        if experiment_config.bi and experiment_config.reverse_input:
            experiment_config.reverse_input = False
        experiment_config.rnn_type = "gru"
        experiment_config.decoder_type = "context"
        #experiment_config.nlayers = 1
        return ContextSeq2Seq(experiment_config, tokens_bos_eos_pad_unk)

    elif model_type == "s":
        #### This returs a model like in Sutskever et al. ####
        if experiment_config.nlayers < 2: experiment_config.nlayers = 2
        return Seq2Seq(experiment_config, tokens_bos_eos_pad_unk)
