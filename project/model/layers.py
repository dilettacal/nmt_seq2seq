import torch.nn as nn
import torch
import torch.nn.functional as F


### Attention: see https://lukemelas.github.io/machine-translation.html

class Attention(nn.Module):
    def __init__(self, pad_token=1, bidirectional=False, attn_type='dot', h_dim=300):
        super(Attention, self).__init__()
        # Check attn type and store variables
        if attn_type not in ['dot', 'additive', 'none']:
            raise Exception('Incorrect attention type')
        self.bidirectional = bidirectional
        self.attn_type = attn_type
        self.h_dim = h_dim
        self.pad_token = pad_token

        # Create parameters for additive attention
        if self.attn_type == 'additive':
            self.linear = nn.Linear(2 * self.h_dim, self.h_dim)
            self.tanh = nn.Tanh()
            self.vector = nn.Parameter(torch.zeros(self.h_dim))

    def attention(self, in_e, out_e, out_d):
        '''Produces context and attention distribution'''

        # If no attention, return context of zeros
        if self.attn_type == 'none':
            return out_d.clone() * 0, out_d.clone() * 0

        # Deal with bidirectional encoder, move batches first
        if self.bidirectional: # sum hidden states for both directions
            #print(out_e.size()) #lstm: 30,64,1024 - seq_len, bs, hid_dim*2
            out_e = out_e.contiguous().view(out_e.size(0), out_e.size(1), 2, -1).sum(2).view(out_e.size(0), out_e.size(1), -1)
        out_e = out_e.transpose(0,1) # b x sl x hd
        out_d = out_d.transpose(0,1) # b x tl x hd

        # Different types of attention
        if self.attn_type == 'dot':
            attn = out_e.bmm(out_d.transpose(1,2)) # (b x sl x hd) (b x hd x tl) --> (b x sl x tl)
        elif self.attn_type == 'additive':
            # Resize output tensors for efficient matrix multiplication, then apply additive attention
            bs_sl_tl_hdim = (out_e.size(0), out_e.size(1), out_d.size(1), out_e.size(2))
            out_e_resized = out_e.unsqueeze(2).expand(bs_sl_tl_hdim) # b x sl x tl x hd
            out_d_resized = out_d.unsqueeze(1).expand(bs_sl_tl_hdim) # b x sl x tl x hd
            attn = self.linear(torch.cat((out_e_resized, out_d_resized), dim=3)) # --> b x sl x tl x hd
            attn = self.tanh(attn) @ self.vector # --> b x sl x tl

        # Softmax and reshape
        attn = F.log_softmax(attn, dim=1)
       # attn = attn.exp() / attn.exp().sum(dim=1, keepdim=True) # in updated pytorch, make softmax
        attn = attn.transpose(1,2) # --> b x tl x sl

        # Get attention distribution
        context = attn.bmm(out_e) # --> b x tl x hd
        context = context.transpose(0,1) # --> tl x b x hd

        return context, attn

    def forward(self, in_e, out_e, out_d):
        '''Produces context using attention distribution'''
        context, attn = self.attention(in_e, out_e, out_d)
        return context

    def get_visualization(self, in_e, out_e, out_d):
        '''Gives attention distribution for visualization'''
        context, attn = self.attention(in_e, out_e, out_d)
        return attn

