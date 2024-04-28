import torch
from torch import nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class HateDetectionModule(nn.Module):

    def __init__(self, input_size, hidden_size, sizes, dropout=0, lstm_layers=1):
        super(HateDetectionModule, self).__init__()
        self.init_classifier(sizes, dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout, bidirectional=True,
                            num_layers=lstm_layers)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(300, 1)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)
        # self.freeze_lstm_classifier(False)

    def init_classifier(self, sizes, dropout=0):
        sequence = []
        for i in range(len(sizes) - 1):
            sequence += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        self.classifier = nn.Sequential(*sequence[:-2])
        for name, param in self.classifier.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)

    def freeze_lstm_classifier(self, val):
        # Imposta i parametri del LSTM come non addestrabili
        for param in self.lstm.parameters():
            param.requires_grad = val

        # Imposta i parametri del classificatore come non addestrabili
        for param in self.classifier.parameters():
            param.requires_grad = val

    def freeze_linear(self, val):
        for param in self.linear.parameters():
            param.requires_grad = val

    def forward(self, x):

        input, lengths = x
        lens = lengths.cpu()
        padded = pack_padded_sequence(input, lens, batch_first=True, enforce_sorted=False)
        data = self.linear(padded.data).squeeze()
        packed = torch.nn.utils.rnn.PackedSequence(data, padded.batch_sizes, padded.sorted_indices,
                                                   padded.unsorted_indices)
        seq, lens = pad_packed_sequence(packed, batch_first=True)
        seq_mean = seq.sum(dim=1) / lens.float().cuda()
        o, (h, c) = self.lstm(padded)
        out = torch.cat((h[-2, :, :], h[-1, :, :], seq_mean.unsqueeze(1)), dim=1)
        output = self.classifier(out).squeeze()
        return self.sigmoid(output)
