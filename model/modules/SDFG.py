import torch
import torch.nn as nn
from model.modules.DWConv import DWConv
from model.modules.AttentionModule import AttentionModule

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = DWConv(in_channels=in_planes, out_channels=in_planes // ratio, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.fc2 = DWConv(in_channels=in_planes // ratio, out_channels=in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = DWConv(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Step1(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class SDFG(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.step1 = Step1(in_planes=in_channels*2)
        self.sigmoid = nn.Sigmoid()
        self.step2 = AttentionModule(in_channels=in_channels)
    
    def forward(self, input1, input2):
        step1_weight = self.sigmoid(self.step1(torch.cat([input1, input2], dim=1)))
        enhanced_input1 = input1 + step1_weight[:,:self.in_channels,:,:]*input2
        enhanced_input2 = input2 + step1_weight[:,self.in_channels:,:,:]*input1
        output = self.step2(enhanced_input1, enhanced_input2)
        return output