import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaLSTM(nn.Module):
    def __init__(self):
        super(VanillaLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=178, hidden_size=64, num_layers=1)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1)
        self.dropout2 = nn.Dropout(0.5)
        self.dense = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = F.relu(out)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = torch.sigmoid(out)
        out = self.dropout2(out)
        out = self.dense(out)
        out = torch.sigmoid(out)
        return out.view(out.size(0),-1)
