import torch
import torch.nn as nn
import pytorch_wavelets as pw

class HWD(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.wt = pw.DWTForward(J=1, mode='zero', wave='haar')
        self.cbr = nn.Sequential(
            nn.Conv2d(in_channels=in_channels*4, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input):
        yL, yH = self.wt(input)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.cbr(x)
        return x  