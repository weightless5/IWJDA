from datetime import datetime
import os
import time  # 新增：用于计时
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from feature_net import mymodule

class MACNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=None, stride=1, reduction=10):
        super(MACNNBlock, self).__init__()
        if kernel_size is None:
            kernel_size = [1, 3, 5]
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size[0],
                               stride=stride, padding='same')
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size[1],
                               stride=stride, padding='same')
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size[2],
                               stride=stride, padding='same')
        self.bn = nn.BatchNorm1d(out_channels * 3)  # 多尺度卷积拼接后归一化

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x_cat = torch.cat([x1, x2, x3], dim=1)
        return F.gelu(self.bn(x_cat))  # GELU激活，适配时序特征


class myreshape(nn.Module):
    def __init__(self, *args):
        super(myreshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class MCADNNFeatureExtractor(nn.Module):


    def __init__(self, in_channels=14, hidden_channels=16, bottleneck_dim=64, block_lay_num=None):
        super(MCADNNFeatureExtractor, self).__init__()
        if block_lay_num is None:
            block_lay_num = [1, 1, 1]
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.bottleneck_dim = bottleneck_dim  # 输出维度Z=64

        # 多尺度卷积层（对应论文ResNet-50的特征提取逻辑，🔶1-98）
        self.layer1 = self._make_layer(MACNNBlock, block_lay_num[0], self.hidden_channels)
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer2 = self._make_layer(MACNNBlock, block_lay_num[1], self.hidden_channels * 2)
        self.max_pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer3 = self._make_layer(MACNNBlock, block_lay_num[2], self.hidden_channels * 4)
        self.avg_pool = nn.AdaptiveAvgPool1d(2)  # 时序维度自适应池化

        self.csa1 = mymodule.CSA(48)
        self.csa2 = mymodule.CSA(96)

        self.dlinear1=mymodule.Model(125,128)
        self.dlinear2=mymodule.Model(64,64)


        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(384, self.bottleneck_dim)  # 映射到Z=64

    def _make_layer(self, block, block_num, hidden_channels, reduction=4):
        layers = []
        for _ in range(block_num):
            layers.append(block(self.in_channels, hidden_channels, reduction=reduction))
            self.in_channels = 3 * hidden_channels  # 多尺度卷积拼接后通道数
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer1(x)
        x = self.max_pool1(x)
        x = self.drop(x)

        if self.csa1 is not None :
            x = myreshape(-1, x.size(1), x.size(2), 1)(x)
            x = self.ema1(x)
            x = myreshape(-1, x.size(1), x.size(2))(x)

        x= self.dlinear1(x)
        x = self.layer2(x)
        x = self.max_pool2(x)
        x = self.drop(x)

        if self.csa2 is not None:
            x = myreshape(-1, x.size(1), x.size(2), 1)(x)
            x = self.ema2(x)
            x = myreshape(-1, x.size(1), x.size(2))(x)
        x = self.dlinear2(x)

        x = self.layer3(x)

        x = self.avg_pool(x)
        feat1 = x.view(x.size(0), -1)  # 展平：[B, 384]
        feat2 = self.fc(feat1)
        return feat1,feat2




class Classifier(nn.Module):


    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, f_common):
        return self.fc(f_common)  # 输出P(Y|z*)

class DomainClassifier(nn.Module):
    def __init__(self, feat_dim, num_domains):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_domains)

    def forward(self, f_aux):
        return self.fc(f_aux)