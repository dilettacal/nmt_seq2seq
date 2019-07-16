import torch.nn as nn
import torch
import torch.nn.functional as F


### Attention: see https://lukemelas.github.io/machine-translation.html

class Attention(nn.Module):
    def __init__(self, bidirectional=False, attn_type='dot', h_dim=300):
        super(Attention, self).__init__()
        # Check attn type and store variables
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
        if self.bidirectional: # sum hidden states for both directions
            '''
            out_e: [seq_len, bs, hid_dim*2]
            out_e reshaped (encoder_outputs.contiguous().view(encoder_outputs.size(0), encoder_outputs.size(1), 2, -1)):
            [seq_len, bs, 2, hid_dim]
            Sum on 2nd dimensions (sum for all sequences as in bs): [seq_len, bs, hid_dim]
            Final output shape: [seq_len, bs, hid_dim]
            '''
            encoder_outputs = encoder_outputs.contiguous().view(encoder_outputs.size(0), encoder_outputs.size(1), 2, -1).sum(2).view(encoder_outputs.size(0), encoder_outputs.size(1), -1) # final shape: (30,64,300)
        encoder_outputs = encoder_outputs.transpose(0, 1) # b x sl x hd
        decoder_outputs = decoder_outputs.transpose(0, 1) # b x tl x hd

        # Different types of attention
        if self.attn_type == 'dot':
            attn = encoder_outputs.bmm(decoder_outputs.transpose(1, 2)) # (b x sl x hd) (b x hd x tl) --> (b x sl x tl)
        elif self.attn_type == 'additive':
            # Resize output tensors for efficient matrix multiplication, then apply additive attention
            ### tensor must have the dimension: batch_size, src_len, trg_len, hidden_dim to consider all the sequences in the batch, the length of H and the actual hidden states of the decoder
            batch_size_src_len_trg_len_hid_dim = (encoder_outputs.size(0), encoder_outputs.size(1), decoder_outputs.size(1), encoder_outputs.size(2))
            out_e_resized = encoder_outputs.unsqueeze(2).expand(batch_size_src_len_trg_len_hid_dim) # b x sl x tl x hd
            out_d_resized = decoder_outputs.unsqueeze(1).expand(batch_size_src_len_trg_len_hid_dim) # b x sl x tl x hd
            attn = self.linear(torch.cat((out_e_resized, out_d_resized), dim=3)) # --> b x sl x tl x hd
            attn = self.tanh(attn) @ self.vector # --> b x sl x tl

        # Softmax and reshape
        #### The attention weights must sum to 1 and be interpreted as a probab distribution
        attn = attn.exp() / attn.exp().sum(dim=1, keepdim=True)
        attn = attn.transpose(1,2) # --> b x tl x sl

        # Get attention distribution
        ### context weights are then multiplied with the encoder outputs
        context = attn.bmm(encoder_outputs) # --> b x tl x hd

        #### Reshaping to seq_len x batch_size x hid_dim as the model works with batch_first = False
        context = context.transpose(0,1) # --> tl x b x hd

        return context, attn

    def forward(self, out_e, out_d):
        '''Produces context using attention distribution'''
        context, attn = self.attention(out_e, out_d)
        return context

    def get_visualization(self, in_e, out_e, out_d):
        '''Gives attention distribution for visualization'''
        context, attn = self.attention(out_e, out_d)
        return attn

