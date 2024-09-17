import torch
import torch.nn as nn

import torch.nn.functional as F


class bilstmModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.fc = torch.nn.Linear(hidden_size * 2, 2)

    def forward(self, input):
        x, (h_n, c_n) = self.bilstm(input)
        hidden = h_n.squeeze()
        # output_fw = h_n[-2, :, :]
        # output_bw = h_n[-1, :, :]
        # print(hidden[-1].size())

        # print(hidden[-2].size())
        output = torch.cat((hidden[-1], hidden[-2]), dim=1)
        out = self.fc(output)
        # print(out)
        # print(F.softmax(out))
        return F.softmax(out)
