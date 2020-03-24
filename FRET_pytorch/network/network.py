import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class STN3d(nn.Module):
    def __init__(self, length=301):
        super(STN3d, self).__init__()
        self.length = length
        self.conv1 = nn.Conv1d(1, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.mp1 = nn.MaxPool1d(length)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x


class Feats_STN3d(nn.Module):
    def __init__(self, length=301):
        super(Feats_STN3d, self).__init__()
        self.conv1 = nn.Conv1d(3, 256, 1)
        self.conv2 = nn.Conv1d(256, 1024, 1)
        self.mp1 = nn.MaxPool1d(length)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)

        # train param
        self.trainable_param = []
        if self.backbone:
            self.trainable_param = list(filter(lambda p: p.requires_grad, self.backbone.parameters()))
        self.trainable_param += list(filter(lambda p: p.requires_grad, self.adj_infer.parameters()))
        self.trainable_param += list(filter(lambda p: p.requires_grad, self.adj_embed.parameters()))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # bz x 256 x 2048
        x = F.relu(self.bn2(self.conv2(x)))  # bz x 1024 x 2048
        x = self.mp1(x)  # bz x 1024 x 1
        x = x.view(-1, 1024)

        x = F.relu(self.bn3(self.fc1(x)))  # bz x 512
        x = F.relu(self.bn4(self.fc2(x)))  # bz x 256
        x = self.fc3(x)
        return x
