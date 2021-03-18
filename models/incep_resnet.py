import os
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["IncepResNetV2", "Inception_ResNet_V2"]


_InceptionOuputs = namedtuple("InceptionOuputs", ["logits", "aux_logits"])


class IncepResNetV2(nn.Module):
    def __init__(self, num_classes=10, transform_input=False):
        super(IncepResNetV2, self).__init__()
        self.stem = IncepResStem(3)
        self.IncepResA = Inception_ResNet_A(256)
        self.IncepResB = Inception_ResNet_B(896)
        self.IncepResC = Inception_ResNet_C(1729)
        self.ReducResA = Reduction_ResNet_A(256)
        self.ReducResB = Reduction_ResNet_B(896)

        self.fc = nn.Linear(1792, num_classes)

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

        for i in range(5):
            x = self.IncepResA(x)
        x = self.ReducResA(x)

        for i in range(7):
            x = self.IncepResB(x)
        x = self.ReducResB(x)

        for i in range(3):
            x = self.IncepResC(x)
                
        x = F.avg_pool2d(x, kernel_size=8)
        x = F.dropout(x, p=0.8, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class IncepResStem(nn.Module):
    def __init__(self, in_channels):
        super(IncepResStem, self).__init__()
        self.Conv2d_1 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_4 = BasicConv2d(64, 64, kernel_size=3, stride=2)
        self.Conv2d_5 = BasicConv2d(64, 80, kernel_size=1)
        self.conv2d_6 = BasicConv2d(80, 192, kernel_size=3)
        self.Conv2d_7 = BasicConv2d(192, 256, kernel_size=3, stride=2)

        self.bn = nn.BatchNorm2d(256, eps=0.001)
    
    def forward(self, x):
        x = self.Conv2d_1(x)
        x = self.Conv2d_2(x)
        x = self.Conv2d_3(x)
        x = self.Conv2d_4(x)
        x = self.Conv2d_5(x)
        x = self.conv2d_6(x)
        x = self.Conv2d_7(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


# 17x17 grid module
class Inception_ResNet_A(nn.Module):
    def __init__(self, in_channels, scale_residual=True):
        super(Inception_ResNet_A, self).__init__()
        self.init = 0
        self.scale_residual = scale_residual
        self.branch1x1 = BasicConv2d(in_channels, 32, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(32, 32, kernel_size=3, padding=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(32, 48, kernel_size=3, padding=1)
        self.branch5x5_3 = BasicConv2d(48, 64, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(128, 384, kernel_size=1, activation="linear")

        self.lamb = LambdaLayer(lambda x: x * 0.1)
        self.bn = nn.BatchNorm2d(256, eps=0.001)
    
    def forward(self, x):
        self.init = x

        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)

        merge = [branch1x1, branch3x3, branch5x5]
        merge = torch.cat(merge, 1)

        merge = self.branch_pool(merge)
        if self.scale_residual:
            merge = self.lamb(merge)

        outputs = self.init + merge
        outputs = self.bn(outputs)
        return F.relu(outputs, inplace=True)


class Inception_ResNet_B(nn.Module):
    def __init__(self, in_channels):
        super(Inception_ResNet_B, self).__init__()
        self.init = 0
        
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        self.branch7x7_1 = BasicConv2d(in_channels,128, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(128, 160, kernel_size=(1,7), padding=(0,3))
        self.branch7x7_3 = BasicConv2d(160, 192, kernel_size=(7,1), padding=(3,0))

        self.branch_pool = BasicConv2d(384, 1154, kernel_size=1, activation="linear")

        self.lamb = LambdaLayer(lambda x: x * 0.1)
        self.bn = nn.BatchNorm2d(896, eps=0.001)

    def forward(self, x):
        self.init = x

        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        merge = [branch1x1, branch7x7]
        merge = torch.cat(merge, 1)

        merge = self.branch_pool(merge)
        if scale_residual:
            merge = self.lamb(merge)

        outputs = self.init + merge
        outputs = self.bn(outputs)
        return F.relu(outputs, inplace=True)


class Inception_ResNet_C(nn.Module):
    def __init__(self, in_channels):
        super(Inception_ResNet_C, self).__init__()
        self.init = 0

        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 224, kernel_size=(1,3), padding=(0,1))
        self.branch3x3_3 = BasicConv2d(224, 256, kernel_size=(3,1), padding=(1,0))

        self.branch_pool = BasicConv2d(448, 2144, kernel_size=1, activation="linear")

        self.lamb = LambdaLayer(lambda x: x * 0.1)
        self.bn = nn.BatchNorm2d(1792, eps=0.001)
    
    def forward(self, x):
        self.init = x

        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        merge = [branch1x1, branch3x3]  
        merge = torch.cat(merge, 1)

        merge = self.branch_pool(merge)
        if scale_residual:
            merge = self.lamb(merge)

        outputs = self.init + merge
        outputs = self.bn(outputs)
        return F.relu(outputs, inplace=True)


class Reduction_ResNet_A(nn.Module):
    def __init__(self,in_channels, k=192, l=224, m=256, n=384):
        super(Reduction_ResNet_A, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, n, kernel_size=3, stride=2)

        self.branch5x5_1 = BasicConv2d(in_channels, k, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(k, l, kernel_size=3, padding=1)
        self.branch5x5_3 = BasicConv2d(l, m, kernel_size=3, stride=2)
        
        self.bn = nn.BatchNorm2d(896, eps=0.001)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch5x5, branch_pool]
        outputs = self.bn(outputs)
        return F.relu(outputs, inplace=True)


class Reduction_ResNet_B(nn.Module):
    def __init__(self, in_channels):
        super(Reduction_ResNet_B, self).__init__()
        self.branch3x3_1a = BasicConv2d(in_channels, 256, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch3x3_1b = BasicConv2d(in_channels, 256, kernel_size=1)
        self.branch3x3_2b = BasicConv2d(256, 288, kernel_size=3, stride=2)

        self.branch5x5_1 = BasicConv2d(in_channels, 256, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(256, 288, kernel_size=3, padding=1)
        self.branch5x5_3 = BasicConv2d(288, 320, kernel_size=3, stride=2)

        self.bn = nn.BatchNorm2d(1792, eps=0.001)
    
    def forward(self, x):
        branch3x3_a = self.branch3x3_1a(x)
        branch3x3_a = self.branch3x3_2a(branch3x3_a)

        branch3x3_b = self.branch3x3_1b(x)
        branch3x3_b = self.branch3x3_2b(branch3x3_b)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3_a, branch3x3_b, branch5x5, branch_pool]
        outputs = self.bn(outputs)
        return F.relu(outputs, inplace=True)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, activation="relu", **kwargs):
        super(BasicConv2d, self).__init__()
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        if self.activation == "relu":
            return F.relu(x, inplace=True)
        else:
            return x


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def Inception_ResNet_V2():
    return IncepResNetV2(transform_input=True)