# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 22:34:32 2023

@author: nisar
"""

import torch
from torch import nn


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=8):
        super(GoogLeNet, self).__init__()
      
        # Write in_channels, etc, all explicit in self.conv1, rest will write to
        # make everything as compact as possible, kernel_size=3 instead of (3,3)
        self.conv1 = conv_block(
            in_channels=30,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,      
        )

        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # In this order: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.avgpool = nn.AvgPool1d(kernel_size=7, stride=1)
        self.conv3 = conv_block(256, 64, kernel_size=3, stride=2, padding=1)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(64, num_classes)


    def forward(self, x):
        
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
       # x = self.inception3b(x)
        x = self.avgpool(x)
        x=self.conv3(x)
        x=self.maxpool3(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return  x


class Inception_block(nn.Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
    ):
        super(Inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=(3), padding=1),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
        )


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


# if __name__ == "__main__":
#     BATCH_SIZE = 5
#     x = torch.randn(BATCH_SIZE, 30, 50)
#     model = GoogLeNet( num_classes=8)
#     print(model(x).shape)
