"""
Credits for this source code:

Class Seq2Seq (with or w/o Attention) - modified version from this code:
Author: Luke Melas
Title: Machine Translation with Recurrent Neural Networks
URL: https://lukemelas.github.io/machine-translation.html
and https://github.com/lukemelas/Machine-Translation/blob/master/models/Seq2seq.py
(Code from the github repository is used on the courtesy of the author)
"""
import random

import torch
import torch.nn as nn
from project.utils.experiment import Experiment
from project.model.decoders import Decoder
from project.model.encoders import Encoder
from project.model.layers import Attention
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

        #### Setup configuration ###
        self.model_type = experiment_config.model_type
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
        self.weight_tied = experiment_config.tied
        rnn_type = experiment_config.rnn_type
        self.cell = rnn_type
        self.context_model = False
        self.att_type = experiment_config.attn

        assert rnn_type.lower() in VALID_CELLS, "Provided cell type is not supported!"

        #### Define main elements of the Seq2Seq model ####

        self.encoder = Encoder(self.src_vocab_size, self.emb_size, self.hid_dim, self.num_layers,
                               dropout_p=self.dp, bidirectional=self.enc_bi, rnn_cell=rnn_type, device=self.device)

        self.decoder = Decoder(self.trg_vocab_size, self.emb_size, self.hid_dim,
                                   self.num_layers * 2 if self.enc_bi else self.num_layers, rnn_cell=rnn_type,
                                   dropout_p=self.dp)

        self.attention = Attention(bidirectional=self.enc_bi, attn_type=self.att_type, h_dim=self.hid_dim)


        #### This additional preoutput layer should reduce the bottleneck problem at the final layer on which
        #### the softmax operation is performed (here this operation is done by the CrossEntropyLoss object
        self.preoutput = nn.Linear(2 * self.hid_dim, self.emb_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(experiment_config.dp)

        #### output layer
        self.output = nn.Linear(self.emb_size, self.trg_vocab_size) #emb size of target

        ### Weight tying is a method to improve the language model perfomance
        ### See: https://arxiv.org/abs/1608.05859
        if self.weight_tied and self.decoder.embedding.weight.size() == self.output.weight.size():
            print('Weight tying!')
            self.output.weight = self.decoder.embedding.weight


    def load_pretrained_embeddings(self, pretrained_src, pretraiend_trg):
        assert self.encoder.embedding.weight.size() == pretrained_src.size()
        assert self.decoder.embedding.weight.size() == pretraiend_trg.size()
        enc_w = self.encoder.embedding.weight.clone()
        dec_w = self.decoder.embedding.weight.clone()
        self.encoder.embedding.weight.data.copy_(pretrained_src)
        self.decoder.embedding.weight.data.copy_(pretraiend_trg)
        assert not torch.all(torch.eq(enc_w, self.encoder.embedding.weight))
        assert not torch.all(torch.eq(dec_w, self.decoder.embedding.weight))
        print("Embeddings weights have been loaded!")

    def forward(self, enc_input, dec_input):
        enc_input = enc_input.to(self.device)
        dec_input = dec_input.to(self.device) #seq_len, bs

        if self.reverse_input:
            inv_index = torch.arange(enc_input.size(0) - 1, -1, -1).long()
            inv_index = inv_index.to(self.device)
            enc_input = enc_input.index_select(0, inv_index)

        # Encode
        ### bidirectional encoder hidden states are stacked: [fw1, bw1, fw2, bw2, ..., fwL, bwL] , L= num layers
        ### Alternatively, decoder should take num_layers and encoder hidden states should be reduced to the last 2 (forelast forward, last one last bw step)
        ### e.g. #hidden [-2, :, : ] (last FW)  and hidden [-1, :, : ] (last BW)
        encoder_outputs, final_e = self.encoder(enc_input) #final_e is [2*num_layers, bs, hid_dim] if bidiretional, else [num_layers, bs, hid_dim]

        # Decode
        ### iniitalize decoder with the final hidden states of the encoder
        decoder_outputs, final_d = self.decoder(dec_input, final_e) #[seq_len, bs, hid_dim], [num_layers, bs, hid_dim]

        # Attend
        context = self.attention(encoder_outputs, decoder_outputs) #seq_len, bs, hid_dim
        out_cat = torch.cat((decoder_outputs, context), dim=2)

        # Predict
        x = self.preoutput(out_cat)
        x = self.dropout(self.tanh(x))
        x = self.output(x) #seq_len, bs, trg_vocab_size
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
        if self.reverse_input:
            inv_index = torch.arange(src.size(0) - 1, -1, -1).long()
            inv_index = inv_index.to(self.device)
            src = src.index_select(0, inv_index)
        # Encode
        outputs_e, states = self.encoder(src)  # batch size = 1
        # Start with '<sos>'
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
                    ### Attention is computed globally on all the encoder hidden states and on the current hidden state of the decder
                    context = self.attention(outputs_e, outputs_d)
                    out_cat = torch.cat((outputs_d, context), dim=2)
                    x = self.preoutput(out_cat)
                    ###########################################
                    x = self.dropout(self.tanh(x))
                    x = self.output(x)
                    x = x.squeeze().data.clone()
                    # Block predictions of tokens in remove_tokens
                    for t in remove_tokens: x[t] = -10e10

                    #### lprobs are the beam scores computed as log probs
                    #### this step is also performed IN the CrossEntropyLoss criterion during the training phase
                    lprobs = torch.log(x.exp() / x.exp().sum())
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
        model =  Seq2Seq(experiment_config, tokens_bos_eos_pad_unk)

    else: ### sutskever model
        #### This returs a model like in Sutskever et al. ####
        #### The architecture was multilayered, thus layers are automatically set to 2 and input sequences were reversed (this is handled in the vocabulary class)
        if not experiment_config.reverse_input: experiment_config.reverse_input = True
        if experiment_config.nlayers < 2: experiment_config.nlayers = 2
        model =  Seq2Seq(experiment_config, tokens_bos_eos_pad_unk)
    return model


