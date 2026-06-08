import torch
import torch.nn as nn

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
    
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = SEBlock(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x_ca = self.channel_attention(x) * x
        x_sa = self.spatial_attention(x) * x
        return x_ca + x_sa

class AttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7) -> None:
        super().__init__()
        self.block = AttentionBlock(in_channels=in_channels, reduction=reduction, kernel_size=kernel_size)
        self.cbr = nn.Sequential(
            ConvBnReLU(in_channels=in_channels*2, out_channels=in_channels),
            ConvBnReLU(in_channels=in_channels, out_channels=in_channels),
            ConvBnReLU(in_channels=in_channels, out_channels=in_channels),
        )
    
    def forward(self, input1, input2):
        output1 = self.block(input1) * input1
        output2 = self.block(input2) * input2
        output = self.cbr(torch.cat([output1, output2], dim=1))

        return output

        
