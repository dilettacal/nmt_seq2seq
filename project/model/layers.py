"""
This class defines the Attention Layer to use when training the model with the attention mechanism.

Code integrated from this repository:
https://github.com/lukemelas/Machine-Translation
under the courtesy of the author
"""

import torch.nn as nn
import torch


class Attention(nn.Module):
    def __init__(self, bidirectional=False, attn_type='dot', h_dim=300):
        super(Attention, self).__init__()
        if attn_type not in ['dot', 'additive', 'none']:
            raise Exception('Incorrect attention type')
        self.bidirectional = bidirectional
        self.attn_type = attn_type
        self.h_dim = h_dim

        # Create parameters for additive attention
        if self.attn_type == 'additive':
            self.linear = nn.Linear(2 * self.h_dim, self.h_dim)
            self.tanh = nn.Tanh()
            self.vector = nn.Parameter(torch.zeros(self.h_dim))

    def attention(self, encoder_outputs, decoder_outputs):
        '''Produces context and attention distribution'''
        # If no attention, return context of zeros
        if self.attn_type == 'none':
            return decoder_outputs.clone() * 0, decoder_outputs.clone() * 0

        # Deal with bidirectional encoder, move batches first
        if self.bidirectional:
            encoder_outputs = encoder_outputs.contiguous().\
                view(encoder_outputs.size(0), encoder_outputs.size(1), 2, -1).\
                sum(2).view(encoder_outputs.size(0), encoder_outputs.size(1), -1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        decoder_outputs = decoder_outputs.transpose(0, 1)

        # Different types of attention
        if self.attn_type == 'dot':
            attn = encoder_outputs.bmm(decoder_outputs.transpose(1, 2))
        elif self.attn_type == 'additive':
            batch_size_src_len_trg_len_hid_dim = \
                (encoder_outputs.size(0), encoder_outputs.size(1),
                 decoder_outputs.size(1), encoder_outputs.size(2))
            out_e_resized = encoder_outputs.unsqueeze(2).\
                expand(batch_size_src_len_trg_len_hid_dim)
            out_d_resized = decoder_outputs.unsqueeze(1).\
                expand(batch_size_src_len_trg_len_hid_dim)
            attn = self.linear(torch.cat((out_e_resized, out_d_resized), dim=3))
            attn = self.tanh(attn) @ self.vector

        # Compute scores
        attn = attn.exp() / attn.exp().sum(dim=1, keepdim=True)
        attn = attn.transpose(1,2)

        # Compute the context
        context = attn.bmm(encoder_outputs)
        context = context.transpose(0,1)

        return context, attn

    def forward(self, out_e, out_d):
        '''Produces context using attention distribution'''
        context, attn = self.attention(out_e, out_d)
        return context

    def get_visualization(self, in_e, out_e, out_d):
        '''Gives attention distribution for visualization'''
        context, attn = self.attention(out_e, out_d)
        return attn

