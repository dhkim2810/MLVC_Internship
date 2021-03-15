import os
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Inception4", "inception_v4"]


_InceptionOuputs = namedtuple("InceptionOuputs", ["logits", "aux_logits"])


class Inception_v4(nn.Module):
    def __init__(self, num_classes=10, transform_input=False):
        super(Inception_v4, self).__init__()
        self.stem = InceptionStem(num_classes=num_classes)


    def forward(self, x):
        x = self.stem(x)


class InceptionStem(self, num_channels=10):
    def __init__(self, x):
        super(InceptionStem, self).__init__()
        self.Conv2d_1 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_4 = BasicConv2d(64, 96, kernel_size=3, stride=2)
        self.Conv2d_5a_1x1 = BasicConv2d(160, 64, kernel=1)
        self.Conv2d_5b_3x3 = BasicConv2d(64, 96, kernel=3)
        self.Conv2d_6a_1x1 = BasicConv2d(160, 64, kernel=1)
        self.Conv2d_6b_7x1 = BasicConv2d(64, 64, kernel=(1,7), padding=(0,3))
        self.Conv2d_6c_1x7 = BasicConv2d(64, 64, kernel=(7,1), padding=(3,0))
        self.Conv2d_6d_3x3 = BasicConv2d(64, 96, kernel=3)
        self.Conv2d_7 = BasicConv2d(192, 192, kernel_size=3, stride=2)
    
    def forward(self, x):
        x = Conv2d_1(x)
        x = Conv2d_2(x)
        x = Conv2d_3(x)
        x1 = Conv2d_4(x)
        x2 = F.max_pool2d(x, kernel_size=3, stride=2)
        x = torch.cat((x1, x2), 1)
        x1 = Conv2d_5a_1x1(x)
        x1 = Conv2d_5b_3x3(x1)
        x2 = Conv2d_6a_1x1(x)
        x2 = Conv2d_6b_7x1(x2)
        x2 = Conv2d_6c_1x7(x2)
        x2 = Conv2d_6d_3x3(x2)
        x = torch.cat((x1, x2), 1)
        x1 = Conv2d_7(x)
        x2 = F.max_pool2d(x, kernel_size=3, stride=2)
        x = torch.cat((x1, x2), 1)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

def Inception4():
    return Inception_v4()