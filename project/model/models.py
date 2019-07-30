"""
Credits for parts of this source code:
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
import torch.nn.functional as F
from project.utils.experiment import Experiment
from project.model.decoders import Decoder
from project.model.encoders import Encoder
from project.model.layers import Attention
from settings import VALID_CELLS, SEED
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

        self.att_type = experiment_config.attn

        assert rnn_type.lower() in VALID_CELLS, "Provided cell type is not supported!"

        self.encoder = Encoder(self.src_vocab_size, self.emb_size, self.hid_dim, self.num_layers,
                               dropout_p=self.dp, bidirectional=self.enc_bi, rnn_cell=rnn_type, device=self.device)

        self.decoder = Decoder(self.trg_vocab_size, self.emb_size, self.hid_dim,
                                   self.num_layers * 2 if self.enc_bi else self.num_layers, rnn_cell=rnn_type,
                                   dropout_p=self.dp)

        self.attention = Attention(bidirectional=self.enc_bi, attn_type=self.att_type, h_dim=self.hid_dim)

        if self.att_type == "none":
            self.context_model = False
            self.preoutput = nn.Linear(self.hid_dim, self.emb_size)
        else:
            self.context_model = True
            self.preoutput = nn.Linear(2 * self.hid_dim, self.emb_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(experiment_config.dp)

        self.output = nn.Linear(self.emb_size, self.trg_vocab_size)
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
        print("Embeddings weights have been loaded in the model!")

    def forward(self, enc_input, dec_input):
        """
        Forward pass - Teacher forcing
        :param enc_input: encoder inputs
        :param dec_input: decoder inputs
        :return: raw scores after output layer
        """
        enc_input = enc_input.to(self.device)
        dec_input = dec_input.to(self.device)

        if self.reverse_input:
            inv_index = torch.arange(enc_input.size(0) - 1, -1, -1).long()
            inv_index = inv_index.to(self.device)
            enc_input = enc_input.index_select(0, inv_index)

        encoder_outputs, final_states_enc = self.encoder(enc_input) # Encode
        decoder_outputs, final_states_dec = self.decoder(dec_input, final_states_enc) # Decode
        if self.att_type == "none":
            # no attention
            out_cat = decoder_outputs
        else:
            # Attend
            context = self.attention(encoder_outputs, decoder_outputs)
            out_cat = torch.cat((decoder_outputs, context), dim=2)

        # Predict
        output = self.preoutput(out_cat)
        output = self.dropout(self.tanh(output))
        output = self.output(output)
        return output


    #### Original code #####
    def predict(self, src, beam_size=1, max_len=30, remove_tokens=[]):
        '''Predict top 1 sentence using beam search. Note that beam_size=1 is greedy search.'''
        beam_outputs = self.beam_search(src, beam_size, max_len=max_len, remove_tokens=remove_tokens)
        top1 = beam_outputs[0][1]  # a list of word indices (as ints)
        return top1

    def beam_search(self, src, beam_size, max_len, remove_tokens=[]):
        '''Returns top beam_size sentences using beam search. Works only when src has batch size 1.
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
                last_word = sentence[-1] #decoder input always last word
                if last_word != self.eos_token:
                    last_word_input = torch.LongTensor([last_word]).view(1, 1).to(self.device)
                    outputs_d, new_state = self.decoder(last_word_input, current_state)
                    # Attend
                    ### Attention is computed globally on all the encoder hidden states and on the current hidden state of the decder
                    context = self.attention(outputs_e, outputs_d)
                    if self.att_type == "none":
                        out_cat = outputs_d
                    else: out_cat = torch.cat((outputs_d, context), dim=2)
                    x = self.preoutput(out_cat)
                    ###########################################
                    x = self.dropout(self.tanh(x))
                    x = self.output(x)
                    x = x.squeeze().data.clone()
                    # Block predictions of tokens in remove_tokens
                    for t in remove_tokens: x[t] = -10e10
                    #### scores the words with log_softmax
                    lprobs = F.log_softmax(x, dim=0)
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


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_nmt_model(experiment_config: Experiment, tokens_bos_eos_pad_unk):
    return Seq2Seq(experiment_config, tokens_bos_eos_pad_unk)


