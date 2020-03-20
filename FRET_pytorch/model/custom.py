import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Simple1DCNN(torch.nn.Module):
    def __init__(self, num_classes=2, length=667):
        super(Simple1DCNN, self).__init__()

        self.act = nn.ReLU()
        self.maxPool = nn.MaxPool1d(3)

        self.conv_first = nn.Conv1d(in_channels=3, out_channels=100, kernel_size=3)
        self.conv_iter = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3)
        self.conv_last = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=4)
        self.averagePool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_features=100, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=num_classes)

    def forward(self, x):
        x = self.act(self.conv_first(x))
        x = self.act(self.conv_iter(x))
        x = self.maxPool(x)
        x = self.act(self.conv_iter(x))
        x = self.act(self.conv_iter(x))
        x = self.maxPool(x)
        x = self.act(self.conv_iter(x))
        x = self.act(self.conv_iter(x))
        x = self.maxPool(x)
        x = self.act(self.conv_iter(x))
        x = self.act(self.conv_iter(x))
        x = self.maxPool(x)
        x = self.act(self.conv_iter(x))
        x = self.act(self.conv_last(x))
        # x = self.averagePool(x)
        x = x.view(-1, 100)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.xavier_normal_(m.bias.data)
