from torch import nn
import torch
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BaselineModel(nn.Module):

    def __init__(self, len0, len1):
        super(BaselineModel, self).__init__()
        self.p = len0 / (len0 + len1)

    def forward(self, x):
        return torch.tensor([0 if np.random.rand() < self.p else 1 for _ in range(x[0].shape[0])], dtype=torch.float)


class BaselineSimpleModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(BaselineSimpleModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        seq, lens = x
        packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)
        data = self.linear(packed.data).squeeze()
        data = self.sigmoid(data)
        packed = torch.nn.utils.rnn.PackedSequence(data, packed.batch_sizes, packed.sorted_indices,
                                                   packed.unsorted_indices)
        seq, lens = pad_packed_sequence(packed, batch_first=True)
        seq_mean = seq.sum(dim=1) / lens.float()
        return seq_mean
