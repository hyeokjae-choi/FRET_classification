import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, num_classes=2, length=667):
        super(STN3d, self).__init__()
        self.length = length
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.mp1 = nn.MaxPool1d(length)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
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
    def __init__(self, num_classes=2, length=667):
        super(Feats_STN3d, self).__init__()
        self.conv1 = nn.Conv1d(3, 256, 1)
        self.conv2 = nn.Conv1d(256, 1024, 1)
        self.mp1 = nn.MaxPool1d(length)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # bz x 256 x 2048
        x = F.relu(self.bn2(self.conv2(x)))  # bz x 1024 x 2048
        x = self.mp1(x)  # bz x 1024 x 1
        x = x.view(-1, 1024)

        x = F.relu(self.bn3(self.fc1(x)))  # bz x 512
        x = F.relu(self.bn4(self.fc2(x)))  # bz x 256
        x = self.fc3(x)
        return x


# class PointNetfeat(nn.Module):
#     def __init__(self, num_classes=2, length=667, global_feat=True):
#         super(PointNetfeat, self).__init__()
#         self.stn = STN3d(num_classes=num_classes, length=length)  # bz x 3 x 3
#         self.conv1 = torch.nn.Conv1d(3, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.mp1 = torch.nn.MaxPool1d(length)
#         self.length = length
#         self.global_feat = global_feat
#
#     def forward(self, x):
#         batchsize = x.size()[0]
#         trans = self.stn(x)  # regressing the transforming parameters using STN
#         x = x.transpose(2, 1)  # bz x length x 3
#         x = torch.bmm(x, trans)  # (bz x length x 3) x (bz x 3 x 3)
#         x = x.transpose(2, 1)  # bz x 3 x 2048
#         x = F.relu(self.bn1(self.conv1(x)))
#         pointfeat = x  # bz x 64 x 2048
#         x = F.relu(self.bn2(self.conv2(x)))  # bz x 128 x 2048
#         x = self.bn3(self.conv3(x))  # bz x 1024 x 2048
#         x = self.mp1(x)
#         x = x.view(-1, 1024)  # bz x 1024
#         if self.global_feat:  # using global feats for classification
#             return x, trans
#         else:
#             x = x.view(-1, 1024, 1).repeat(1, 1, self.length)
#             return torch.cat([x, pointfeat], 1), trans
#
#
# class PointNetCls(nn.Module):
#     # on modelnet40, it is set to be 2048
#     def __init__(self, num_classes=2, length=667):
#         super(PointNetCls, self).__init__()
#         self.length = length
#         self.feat = PointNetfeat(length, global_feat=True)  # bz x 1024
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, num_classes)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#
#     def forward(self, x):
#         x, trans = self.feat(x)
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = self.fc3(x)  # bz x 40
#         return F.log_softmax(x), trans
#
#
# # part segmentation
# class PointNetPartDenseCls(nn.Module):
#     ###################################
#     ## Note that we must use up all the modules defined in __init___,
#     ## otherwise, when gradient clippling, it will cause errors like
#     ## param.grad.data.clamp_(-grad_clip, grad_clip)
#     ## AttributeError: 'NoneType' object has no attribute 'data'
#     ####################################
#     def __init__(self, num_classes=2, length=667, k=2):
#         super(PointNetPartDenseCls, self).__init__()
#         self.length = length
#         self.k = k
#         # T1
#         self.stn1 = STN3d(length=length)  # bz x 3 x 3, after transform => bz x 2048 x 3
#
#         self.conv1 = torch.nn.Conv1d(3, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 128, 1)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(128)
#
#         # T2
#         self.stn2 = Feats_STN3d(length=length)
#
#         self.conv4 = torch.nn.Conv1d(128, 128, 1)
#         self.conv5 = torch.nn.Conv1d(128, 512, 1)
#         self.conv6 = torch.nn.Conv1d(512, 2048, 1)
#         self.bn4 = nn.BatchNorm1d(128)
#         self.bn5 = nn.BatchNorm1d(512)
#         self.bn6 = nn.BatchNorm1d(2048)
#         # pool layer
#         self.mp1 = torch.nn.MaxPool1d(length)
#
#         # MLP(256, 256, 128)
#         self.conv7 = torch.nn.Conv1d(3024 - 16, 256, 1)
#         self.conv8 = torch.nn.Conv1d(256, 256, 1)
#         self.conv9 = torch.nn.Conv1d(256, 128, 1)
#         self.bn7 = nn.BatchNorm1d(256)
#         self.bn8 = nn.BatchNorm1d(256)
#         self.bn9 = nn.BatchNorm1d(128)
#         # last layer
#         self.conv10 = torch.nn.Conv1d(128, self.k, 1)  # 50
#         self.bn10 = nn.BatchNorm1d(self.k)
#
#     def forward(self, x, one_hot_labels):
#         batch_size = x.size()[0]
#         # T1
#         trans_1 = self.stn1(x)  # regressing the transforming parameters using STN
#         x = x.transpose(2, 1)  # bz x 2048 x 3
#         x = torch.bmm(x, trans_1)  # (bz x 2048 x 3) x (bz x 3 x 3)
#         # change back
#         x = x.transpose(2, 1)  # bz x 3 x 2048
#         out1 = F.relu(self.bn1(self.conv1(x)))  # bz x 64 x 2048
#         out2 = F.relu(self.bn2(self.conv2(out1)))  # bz x 128 x 2048
#         out3 = F.relu(self.bn3(self.conv3(out2)))  # bz x 128 x 2048
#         #######################################################################
#         # T2, currently has bugs so now remove this temporately
#         trans_2 = self.stn2(out3)  # regressing the transforming parameters using STN
#         out3_t = out3.transpose(2, 1)  # bz x 2048 x 128
#         out3_trsf = torch.bmm(out3_t, trans_2)  # (bz x 2048 x 128) x (bz x 128 x 3)
#         # change back
#         out3_trsf = out3_trsf.transpose(2, 1)  # bz x 128 x 2048
#
#         out4 = F.relu(self.bn4(self.conv4(out3_trsf)))  # bz x 128 x 2048
#         out5 = F.relu(self.bn5(self.conv5(out4)))  # bz x 512 x 2048
#         out6 = F.relu(self.bn6(self.conv6(out5)))  # bz x 2048 x 2048
#         out6 = self.mp1(out6)  # bz x 2048
#
#         # concat out1, out2, ..., out5
#         out6 = out6.view(-1, 2048, 1).repeat(1, 1, self.length)
#         # out6 = x
#         # cetegories is 16
#         # one_hot_labels: bz x 16
#         one_hot_labels = one_hot_labels.unsqueeze(2).repeat(1, 1, self.length)
#         # 64 + 128 * 3 + 512 + 2048 + 16
#         # point_feats = torch.cat([out1, out2, out3, out4, out5, out6, one_hot_labels], 1)
#         point_feats = torch.cat([out1, out2, out3, out4, out5, out6], 1)
#         # Then feed point_feats to MLP(256, 256, 128)
#         mlp = F.relu(self.bn7(self.conv7(point_feats)))
#         mlp = F.relu(self.bn8(self.conv8(mlp)))
#         mlp = F.relu(self.bn9(self.conv9(mlp)))
#
#         # last layer
#         pred_out = F.relu(self.bn10(self.conv10(mlp)))  # bz x 50(self.k) x 2048
#         pred_out = pred_out.transpose(2, 1).contiguous()
#         pred_out = F.log_softmax(pred_out.view(-1, self.k))
#         pred_out = pred_out.view(batch_size, self.length, self.k)
#         return pred_out, trans_1, trans_2
#
#
# # regular segmentation
# class PointNetDenseCls(nn.Module):
#     def __init__(self, num_classes=2, length=667, k=2):
#         super(PointNetDenseCls, self).__init__()
#         self.length = length
#         self.k = k
#         self.feat = PointNetfeat(length, global_feat=False)
#         self.conv1 = torch.nn.Conv1d(1088, 512, 1)
#         self.conv2 = torch.nn.Conv1d(512, 256, 1)
#         self.conv3 = torch.nn.Conv1d(256, 128, 1)
#         self.conv4 = torch.nn.Conv1d(128, self.k, 1)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.bn3 = nn.BatchNorm1d(128)
#
#     def forward(self, x):
#         batchsize = x.size()[0]
#         x, trans = self.feat(x)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.conv4(x)
#         x = x.transpose(2, 1).contiguous()
#         x = F.log_softmax(x.view(-1, self.k))
#         x = x.view(batchsize, self.length, self.k)
#         return x#, trans
