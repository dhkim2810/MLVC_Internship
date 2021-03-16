import os
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Inception4", "Inception_v4"]


_InceptionOuputs = namedtuple("InceptionOuputs", ["logits", "aux_logits"])


class Inception_v4(nn.Module):
    def __init__(self, num_classes=10, transform_input=False):
        super(Inception_v4, self).__init__()
        self.stem = InceptionStem(3)
        self.inceptionA = InceptionA(384)
        self.inceptionB = InceptionB(1024)
        self.inceptionC = InceptionC(1536)
        self.reductionA = ReductionA(384)
        self.reductionB = ReductionB(1024)

        self.fc = nn.Linear(1536, num_classes)

        self.transform_input = transform_input
        self.up = nn.Upsample(
                size=(299, 299),
                mode="bilinear",
                align_corners=True
            ).type(torch.cuda.FloatTensor)

    def forward(self, x):
        if self.transform_input:
            x = self.up(x)
        x = self.stem(x)

        for i in range(4):
            x = self.inceptionA(x)
        x = self.reductionA(x)

        for i in range(7):
            x = self.inceptionB(x)
        x = self.reductionB(x)

        for i in range(3):
            x = self.inceptionC(x)
                
        x = F.avg_pool2d(x, kernel_size=8)
        x = F.dropout(x, p=0.8, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class InceptionStem(nn.Module):
    def __init__(self, in_channels):
        super(InceptionStem, self).__init__()
        self.Conv2d_1 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_4 = BasicConv2d(64, 96, kernel_size=3, stride=2)
        self.Conv2d_5a_1x1 = BasicConv2d(160, 64, kernel_size=1)
        self.Conv2d_5b_3x3 = BasicConv2d(64, 96, kernel_size=3)
        self.Conv2d_6a_1x1 = BasicConv2d(160, 64, kernel_size=1)
        self.Conv2d_6b_7x1 = BasicConv2d(64, 64, kernel_size=(7,1), padding=(3,0))
        self.Conv2d_6c_1x7 = BasicConv2d(64, 64, kernel_size=(1,7), padding=(0,3))
        self.Conv2d_6d_3x3 = BasicConv2d(64, 96, kernel_size=3)
        self.Conv2d_7 = BasicConv2d(192, 192, kernel_size=3, stride=2)
    
    def forward(self, x):
        x = self.Conv2d_1(x)
        x = self.Conv2d_2(x)
        x = self.Conv2d_3(x)
        x1 = self.Conv2d_4(x)
        x2 = F.max_pool2d(x, kernel_size=3, stride=2)
        x = torch.cat((x1, x2), 1)
        x1 = self.Conv2d_5a_1x1(x)
        x1 = self.Conv2d_5b_3x3(x1)
        x2 = self.Conv2d_6a_1x1(x)
        x2 = self.Conv2d_6b_7x1(x2)
        x2 = self.Conv2d_6c_1x7(x2)
        x2 = self.Conv2d_6d_3x3(x2)
        x = torch.cat((x1, x2), 1)
        x1 = self.Conv2d_7(x)
        x2 = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [x1, x2]
        return torch.cat(outputs , 1)


class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 96, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch5x5_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, 96, kernel_size=1)
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 384, kernel_size=1)

        self.branch7x7_1 = BasicConv2d(in_channels,192, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(192, 224, kernel_size=(1,7), padding=(0,3))
        self.branch7x7_3 = BasicConv2d(224, 256, kernel_size=(7,1), padding=(3,0))

        self.branch9x9_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch9x9_2 = BasicConv2d(192, 192, kernel_size=(1,7), padding=(0,3))
        self.branch9x9_3 = BasicConv2d(192, 224, kernel_size=(7,1), padding=(3,0))
        self.branch9x9_4 = BasicConv2d(224, 224, kernel_size=(1,7), padding=(0,3))
        self.branch9x9_5 = BasicConv2d(224, 256, kernel_size=(7,1), padding=(3,0))

        self.branch_pool = BasicConv2d(in_channels, 128, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        
        branch9x9 = self.branch9x9_1(x)
        branch9x9 = self.branch9x9_2(branch9x9)
        branch9x9 = self.branch9x9_3(branch9x9)
        branch9x9 = self.branch9x9_4(branch9x9)
        branch9x9 = self.branch9x9_5(branch9x9)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch9x9, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 256, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 256, kernel_size=(1,3), padding=(0,1))
        self.branch3x3_2b = BasicConv2d(384, 256, kernel_size=(3,1), padding=(1,0))

        self.branch5x5_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(384, 448, kernel_size=(1,3), padding=(0,1))
        self.branch5x5_3 = BasicConv2d(448, 512, kernel_size=(3,1), padding=(1,0))
        self.branch5x5_4a = BasicConv2d(512, 256, kernel_size=(3,1), padding=(1,0))
        self.branch5x5_4b = BasicConv2d(512, 256, kernel_size=(1,3), padding=(0,1))

        self.branch_pool = BasicConv2d(in_channels, 256, kernel_size=1)
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3_1 = self.branch3x3_2a(branch3x3)
        branch3x3_2 = self.branch3x3_2b(branch3x3)
        branch3x3 = torch.cat([branch3x3_1, branch3x3_2], 1)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)
        branch5x5_1 = self.branch5x5_4a(branch5x5)
        branch5x5_2 = self.branch5x5_4b(branch5x5)
        branch5x5 = torch.cat([branch5x5_1, branch5x5_2], 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class ReductionA(nn.Module):
    def __init__(self,in_channels):
        super(ReductionA, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch5x5_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(192, 224, kernel_size=3, padding=1)
        self.branch5x5_3 = BasicConv2d(224, 256, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super(ReductionB, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 192, kernel_size=3, stride=2)

        self.branch7x7_1 = BasicConv2d(in_channels, 256, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(256, 256, kernel_size=(1,7), padding=(0,3))
        self.branch7x7_3 = BasicConv2d(256, 320, kernel_size=(7,1), padding=(3,0))
        self.branch7x7_4 = BasicConv2d(320, 320, kernel_size=3, stride=2)
    
    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7 = self.branch7x7_4(branch7x7)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch7x7, branch_pool]
        return torch.cat(outputs, 1)


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
    return Inception_v4(transform_input=True)