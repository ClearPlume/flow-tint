import math

import torch
from torch import zeros, arange
from torch.nn import Parameter


def positional_encoding(max_len: int, d_model: int) -> Parameter:
    pe = zeros(max_len, d_model)
    position = arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return Parameter(pe.unsqueeze(0), requires_grad=False)
