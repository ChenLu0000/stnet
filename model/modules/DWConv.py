import torch.nn as nn

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride, 
            padding=padding, groups=in_channels, bias=False)
        
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x