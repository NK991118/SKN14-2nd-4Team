import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, Cin):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.linear1 = nn.Linear(Cin, 32)
        self.linear2 = nn.Linear(32, 128)
        self.linear3 = nn.Linear(128, 16)
        self.fc = nn.Linear(16, 1)
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out