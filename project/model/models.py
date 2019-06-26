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


import torch
import torch.nn as nn

from project.experiment.setup_experiment import Experiment
from project.model.decoders import Decoder, ContextDecoder
from project.model.encoders import Encoder
from project.model.utils import Beam
from settings import VALID_CELLS




"""
Parameters:
- Maxout (500 units): 73,303,508
- 4 layers, 500, 500: 57,270,504
- 4 layers, 1000, 1000: 146,511,004
"""


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

        assert rnn_type.lower() in VALID_CELLS, "Provided cell type is not supported!"

        self.encoder = Encoder(self.src_vocab_size, self.emb_size, self.hid_dim, self.num_layers,
                               dropout_p=self.dp, bidirectional=self.enc_bi, rnn_cell=rnn_type, device=self.device)

        self.decoder = Decoder(self.trg_vocab_size, self.emb_size, self.hid_dim,
                                   self.num_layers * 2 if self.enc_bi else self.num_layers, rnn_cell=rnn_type,
                                   dropout_p=self.dp)

        self.dropout = nn.Dropout(experiment_config.dp)
        self.output = nn.Linear(self.hid_dim, experiment_config.trg_vocab_size)

    def init_weights(self, func=None):
        if not func:
            pass
        else:
            self.apply(func)

        ### create encoder and decoder

    def forward(self, src, trg):
        src = src.to(self.device)
        trg = trg.to(self.device)

        # Encode
        out_e, final_e = self.encoder(src)

        out_d, _ = self.decoder(trg, final_e)

        x = self.dropout(torch.tanh(out_d))
        x = self.output(x)
        return x

    def predict(self, src, beam_size=1, max_len=30, sos_eos_pad=[2,3,1]):
        return self.beam_search(src=src, beam_size=beam_size, max_len=max_len, sos_eos_pad=sos_eos_pad)

    def beam_search(self, src, beam_size, max_len, sos_eos_pad=[]):
        beam = Beam(size=beam_size, bos=sos_eos_pad[0], eos=sos_eos_pad[1], pad=sos_eos_pad[2], cuda=True if self.device == "cuda" else False)

        src = src.to(self.device)
       # print(src.size())
        outputs_e, states = self.encoder(src)

        if isinstance(states, tuple):
            h, c = states
            h = h.data.repeat(1,beam_size,1)
            c = c.data.repeat(1,beam_size,1) #num_layers, beam_size, hid_dim
            current_state = (h,c)
        else:
            h = states
            h = h.data.repeat(1,beam_size,1)
            current_state = h

        for i in range(max_len):
            x = beam.get_current_state().to(self.device) #shape: beam size [beam_size]
            ### first run: [2,1], sos, pad
            # x -> [1, beam_size]

            outputs_d, current_state = self.decoder(x.unsqueeze(0), current_state)
            ## outputs_d: [1,2,500], hidden [2,2,500]

            outputs_d = self.output(outputs_d)
            #print(outputs_d.shape) # shape: [1,2,trg_vocab_size]
            #### Log softmax to retrieve probabilities
            dec_prob = torch.log(outputs_d.exp() / outputs_d.exp().sum())

            # dec_prob.data.shape: [1,beam_size,trg_vocab] -> [beam_size, trg_vocab]
            if beam.advance(dec_prob.data.squeeze(0)):
                break

            if isinstance(current_state, tuple):
                h, c = current_state
                ###beam current origin is at the beginning a list of size beam_size
                ###this contains a tensor [0,0]
                #print(h.shape) #Target sizes: [2, 5, 500].  Tensor sizes: [5, 5, 500]
                h.data.copy_(h.data.index_select(1, beam.get_current_origin()))
                c.data.copy_(c.data.index_select(1, beam.get_current_origin()))
                current_state = (h,c)
            else:
                current_state.data.copy_(current_state.data.index_select(1, beam.get_current_origin()))

        tt = torch.cuda if self.device == "cuda" else torch
        candidate = tt.LongTensor(beam.get_hyp(0))
        return candidate



class ContextSeq2Seq(Seq2Seq):
    def __init__(self, experiment_config, token_bos_eos_pad_unk):
        super().__init__(experiment_config, token_bos_eos_pad_unk)

        self.decoder = ContextDecoder(self.trg_vocab_size, self.emb_size, self.hid_dim,
                                      self.num_layers * 2 if self.enc_bi else self.num_layers, dropout_p=self.dp)

        self.output = nn.Linear(self.emb_size + self.hid_dim*2, experiment_config.trg_vocab_size)

    def forward(self, src, trg):
        src = src.to(self.device)
        trg = trg.to(self.device)

        # Encode
        out_e, final_e = self.encoder(src)
        seq_len, batch_size = trg.size()
        outputs = torch.zeros(seq_len, batch_size, self.emb_size + self.hid_dim * 2).to(self.device)
        input = trg[0, :]
        context = final_e
        for i in range(1, seq_len):
            # Decode
            out_d, _ = self.decoder(x=input, h0=final_e, context=context)
            outputs[i] = out_d
            input = trg[i]

        x = self.dropout(torch.tanh(outputs))
        x = self.output(x)
        return x

    def beam_search(self, src, beam_size, max_len, sos_eos_pad=[]):
        beam = Beam(size=beam_size, bos=sos_eos_pad[0], eos=sos_eos_pad[1], pad=sos_eos_pad[2],
                    cuda=True if self.device == "cuda" else False)

        src = src.to(self.device)
        # print(src.size())
        outputs_e, states = self.encoder(src)

        h = states
        h = h.data.repeat(1, beam_size, 1)
        current_state = h
        context = current_state.clone()

        for i in range(max_len):
            x = beam.get_current_state().to(self.device)  # shape: beam size [beam_size]
            ### first run: [2,1], sos, pad
            # x -> [1, beam_size]
          #  print(current_state.shape) # [n_layers, beam, hid]
            outputs_d, current_state = self.decoder(x.unsqueeze(0), current_state, context, val=False)
            ## outputs_d: [1,2,500], hidden [2,2,500]

            outputs_d = self.output(outputs_d)
            # shape: [1,2,trg_vocab_size]
            #### Log softmax to retrieve probabilities
            dec_prob = torch.log(outputs_d.exp() / outputs_d.exp().sum())

            # dec_prob.data.shape: [1,beam_size,trg_vocab] -> [beam_size, trg_vocab]
            if beam.advance(dec_prob.data.squeeze(0)):
                break

            current_state.data.copy_(current_state.data.index_select(1, beam.get_current_origin()))

        tt = torch.cuda if self.device == "cuda" else torch
        candidate = tt.LongTensor(beam.get_hyp(0))
        return candidate


class AttentionSeq2Seq(Seq2Seq):
    def __init__(self, experiment_config, token_bos_eos_pad_unk):
        super().__init__(experiment_config, token_bos_eos_pad_unk)
        ### add attention stuff

    def old_beam_search(self, src, beam_size, max_len, remove_tokens=[]):
        ## modifiy with attention stuff
        pass


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
        experiment_config.nlayers = 1
        return ContextSeq2Seq(experiment_config, tokens_bos_eos_pad_unk)

    elif model_type == "s":
        #### This returs a model like in Sutskever et al. ####
        experiment_config.reverse_input = True
        experiment_config.bi = False
        if experiment_config.hid_dim < 500: experiment_config.hid_dim = 500
        if experiment_config.emb_size < 500: experiment_config.emb_size = 500
        if experiment_config.nlayers < 2: experiment_config.nlayers = 2
        return Seq2Seq(experiment_config, tokens_bos_eos_pad_unk)
