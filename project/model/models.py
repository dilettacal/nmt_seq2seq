import random

import torch
import torch.nn as nn

from project.experiment.setup_experiment import Experiment
from project.model.decoders import Decoder, ContextDecoder
from project.model.encoders import Encoder
from project.model.layers import MaxoutLinearLayer
from settings import DEFAULT_DEVICE

LSTM = "lstm"
GRU = "gru"
VALID_CELLS = [LSTM, GRU]
VALID_MODELS = ["standard", "sutskever", "cho"]





class Seq2Seq(nn.Module):
    def __init__(self, experiment_config:Experiment, tokens_bos_eos_pad_unk):
        super(Seq2Seq, self).__init__()

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

        assert rnn_type.lower() in VALID_CELLS, "Provided cell type is not supported!"

        self.encoder = Encoder(self.src_vocab_size, self.emb_size, self.hid_dim, self.num_layers,
                               dropout_p=self.dp, bidirectional=self.enc_bi, rnn_cell=rnn_type, device=self.device)
        self.decoder = Decoder(self.trg_vocab_size, self.emb_size, self.hid_dim,
                               self.num_layers * 2 if self.enc_bi else self.num_layers, dropout_p=self.dp)
        self.dropout = nn.Dropout(experiment_config.dp)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(self.hid_dim, experiment_config.trg_vocab_size)

    def init_weights(self):
        pass

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


class Sutskever(Seq2Seq):
    def __init__(self, experiment_config, tokens_bos_eos_pad_unk):
        experiment_config.nlayers = 4
        experiment_config.hs = 1000
        experiment_config.emb = 1000
        super(Sutskever, self).__init__(experiment_config, tokens_bos_eos_pad_unk)

    def init_weights(self):
        self.apply(uniform_init_weights(self))

class ChoSeq2Seq(Seq2Seq):
    def __init__(self, experiment_config, tokens_bos_eos_pad_unk, maxout_units=None):
        experiment_config.emb = 500
        experiment_config.hs = 500
        maxout_units = maxout_units
        super(ChoSeq2Seq, self).__init__(experiment_config, tokens_bos_eos_pad_unk)


        if maxout_units:
            self.output = MaxoutLinearLayer(input_dim=self.emb_dim_trg * 2, hidden_units=maxout_units,
                                         output_dim=self.vocab_size_trg, k=2)
        else:
            self.output = nn.ReLU(nn.Linear(self.embedding.embedding_dim + self.hid_dim * 2, self.vocab_size))

    def init_weights(self):
        self.apply(normal_init_weights(self))



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


def get_nmt_model(experiment_config:Experiment):
    model_type = experiment_config.model_type
    assert model_type in ["custom", "s", "c"]

    if model_type == "custom":
        return Seq2Seq(experiment_config)
    elif model_type == "c":
        return ChoSeq2Seq(experiment_config, maxout_units=experiment_config.maxout)
