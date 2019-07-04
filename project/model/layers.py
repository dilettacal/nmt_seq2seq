import torch.nn as nn
import torch
import torch.nn.functional as F

#### https://github.com/Usama113/Maxout-PyTorch/blob/master/Maxout.ipynb
from torch.nn import Parameter


class ListModule(object):
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


class MaxoutLinearLayer(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim, k=2):
        super(MaxoutLinearLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.fc1_list = ListModule(self, "fc1_")
        self.fc2_list = ListModule(self, "fc2_")
        for _ in range(k):
            self.fc1_list.append(nn.Linear(input_dim, hidden_units))
            self.fc2_list.append(nn.Linear(hidden_units, output_dim))

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.maxout(x, self.fc1_list)
        x = F.dropout(x, training=self.training)
        x = self.maxout(x, self.fc2_list)
        ### if probabilities are computed with log_softmax together with NLLoss
        #return F.log_softmax(x, dim=1)
        return x

    def maxout(self, x, layer_list):
        max_output = layer_list[0](x)
        for _, layer in enumerate(layer_list, start=1):
            max_output = torch.max(max_output, layer(x))
        return max_output



# https://github.com/pytorch/pytorch/issues/805 erogol
class Maxout(nn.Module):
    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)

    def forward(self, inputs):
        shape = list(inputs.size()) # 1 B 3H+m
        shape[-1] = self.d_out # 1 B maxoutsize
        shape.append(self.pool_size) # 1 B maxoutsize 2
        max_dim = len(shape) - 1 # 3
        out = self.lin(inputs) # 1 B 2maxout
        m, i = out.view(*shape).max(max_dim) # 1 B maxout
        return m


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
        attn = attn.exp() / attn.exp().sum(dim=1, keepdim=True) # in updated pytorch, make softmax
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

