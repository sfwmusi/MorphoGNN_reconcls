#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM

Modified by 
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/9 9:32 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import argparse


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature      # (batch_size, 2*num_dims, num_points, k)


class PointNet(nn.Module):
    def __init__(self, args, output_channels=2):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN_cls(nn.Module):
    # 判断对错，修改output_channels
    def __init__(self, args, output_channels=2):
        super(DGCNN_cls, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        # print('x shape: ',x.shape)
        # 32*2048  batch_size*emb_dims*2
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)

        return x


class C3D(nn.Module):
    def __init__(self,
                 sample_size,
                 sample_duration,
                 num_classes=600):
        # 手动执行父类的初始化，构造方法
        super(C3D, self).__init__()
        # self.group1 = nn.Sequential(
        #     nn.Conv3d(1, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(),
        #     nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.group1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            # 64*64 --> 8192
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            # # 112*112 --> 8192
            # nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)),

            # 256*256 --> 8192
            # nn.MaxPool3d(kernel_size=(1, 10, 10), stride=(2, 2, 2))

            # 196*196 --> 8192   512*1*4*4
            # nn.MaxPool3d(kernel_size=(1, 6, 6), stride=(2, 2, 2))
        )

        last_duration = int(math.floor(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))

        self.fc1 = nn.Sequential(
            nn.Linear(4096, 2048),
            # fc1的计算？
            # nn.Linear(41472, 4096),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc = nn.Sequential(
            nn.Linear(4096, num_classes

                      ))

    def forward(self, x):
        out = self.group1(x)
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
        out = self.group5(out)
        print('out before size ', out.size())
        out = out.view(out.size(0), -1)
        print('out after size ', out.size())
        out = self.fc1(out)
        print('out fc1 size ', out.size())
        out = self.fc2(out)
        out = self.fc(out)
        print('out last size ', out.size())
        return out


# def get_fine_tuning_parameters(model, ft_portion):
#     if ft_portion == "complete":
#         return model.parameters()
#
#     elif ft_portion == "last_layer":
#         ft_module_names = []
#         ft_module_names.append('fc')
#
#         parameters = []
#         for k, v in model.named_parameters():
#             for ft_module in ft_module_names:
#                 if ft_module in k:
#                     parameters.append({'params': v})
#                     break
#             else:
#                 parameters.append({'params': v, 'lr': 0.0})
#         return parameters
#
#     else:
#         raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


# 测试


class MultiNet(nn.Module):
    def __init__(self,
                 args,
                 output_channels=2):
        # 手动执行父类的初始化，构造方法
        super(MultiNet, self).__init__()
        self.args = args
        self.k = args.k
        self.sample_size = args.sample_size
        self.sample_duration = args.sample_duration
        self.num_classes = args.n_classes

        ############C3D#############
        self.group1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            # 64*64 --> 4096
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        # self.fc00 = nn.Sequential(
        #     # 64*64*64
        #     # nn.Linear(4096, 2048),
        #     # fc1的计算? 128*128*128
        #     # 32768
        #     # 96*96*96 13824
        #     nn.Linear(32768, 4096),
        #     nn.ReLU())
        # self.fc0 = nn.Sequential(
        #     # 64*64*64
        #     # nn.Linear(4096, 2048),
        #     # fc1的计算? 128*128*128
        #     # 32768
        #     # 96*96*96 13824
        #     nn.Linear(32768, 4096),
        #     nn.ReLU())
        self.fc1 = nn.Sequential(
            # 64*64*64
            # nn.Linear(4096, 2048),
            # fc1的计算? 128*128*128
            # 32768
            # 96*96*96
            nn.Linear(32768, 2048),
            nn.ReLU())
           # nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU())
           # nn.Dropout(0.5))
        self.fc = nn.Sequential(
            nn.Linear(512, self.num_classes
                      ))
        ###############DGCNN_cls######################
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(4096, 2048, bias=False)
        # self.linear1 = nn.Linear(2048, 2048, bias=False)
        self.bn6 = nn.BatchNorm1d(2048)
        self.dp1 = nn.Dropout(p=args.dropout)

        self.linear2 = nn.Linear(2048, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.dp2 = nn.Dropout(p=args.dropout)

        self.linear3 = nn.Linear(1024, 512)
        self.bn8 = nn.BatchNorm1d(512)
        self.dp3 = nn.Dropout(p=args.dropout)

        self.linear4 = nn.Linear(512, output_channels)


    def forward(self, x, y):
        # C3D_feature
        image_out = self.group1(y)
        image_out = self.group2(image_out)
        image_out = self.group3(image_out)
        image_out = self.group4(image_out)
        image_out = self.group5(image_out)
        # print('out before size ', image_out.size())
        image_out = image_out.view(image_out.size(0), -1)
        # print('out after size ', image_out.size())
        ## 增加一层
        # image_out = self.fc0(image_out)
        # image_out = self.fc0(image_out)
        image_out = self.fc1(image_out)

        # DGCNN_feature
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)  # (batch_size, emb_dims*2)

        # print('x shape: ',x.shape)

        out = torch.cat((x,image_out),1)
        feature = out
        # 8*4096
        out = F.leaky_relu(self.bn6(self.linear1(out)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 512)
        out = self.dp1(out)
        out = F.leaky_relu(self.bn7(self.linear2(out)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        out = self.dp2(out)
        out = F.leaky_relu(self.bn8(self.linear3(out)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        out = self.dp3(out)
        out = self.linear4(out)  # (batch_size, 256) -> (batch_size, output_channels)

        return out, feature


def get_model(**kwargs):
    """
    Returns the model.
    """
    model = MultiNet(**kwargs)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--dropout', type=float, default=0.5, help='initial dropout rate')
    parser.add_argument('--sample_size', default=64,
                        type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=256,
                        type=int, help='Temporal duration of inputs')
    parser.add_argument('--n_classes', default=2, type=int,
                        help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
    args = parser.parse_args()
    model = MultiNet(args)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    print(model)

    # 256
    input_var2 = torch.randn(8, 1, 16, 196, 196)
    input_var3 = torch.randn(8, 1, 16, 196, 196)
    input_var4 = torch.randn(8, 1, 16, 196, 196)
    input_var5 = torch.randn(8, 1, 16, 196, 196)

    input_var = torch.cat((input_var2,input_var3,input_var4,input_var5),dim=1)
    input_var2 = Variable(input_var2)
    print(input_var.size())
    print(input_var[:,0,:,:,:].reshape(8,1,16,196,196).size())

    input_var1 = Variable(torch.randn(8, 3, 1024))
    # image size ---> 64*64*64
    input_var0 = Variable(torch.randn((8,1,96,96,96)))
    output, _ = model(input_var1, input_var0)

    print(output.shape)
    print(output)

