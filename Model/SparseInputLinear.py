import torch
import torch.nn as nn
import numpy as np


class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))


        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters() #init的时候随即重新取

    def reset_parameters(self):
        # pre-init
        stdv = 1. / np.sqrt(self.weight.size(1))

        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x_ = torch.mm(x.view(x.shape[0]*x.shape[1], x.shape[2]), self.weight)
        bias_ = self.bias
        x_ = x_+bias_
        return x_.view(x.shape[0], x.shape[1], -1)