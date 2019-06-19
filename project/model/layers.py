import torch.nn as nn
import torch
import torch.nn.functional as F

#### https://github.com/Usama113/Maxout-PyTorch/blob/master/Maxout.ipynb
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