import torch 
import torch.nn as nn
import torch.nn.functional as F
from model.modules.DWConv import DWConv

class DWConvBNGELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size ,stride, padding) -> None:
        super().__init__()
        self.conv = DWConv(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
    def forward(self, input):
        return self.gelu(self.bn(self.conv(input)))
    
class DWConvBNGELU_x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
                            DWConvBNGELU(in_channels=in_channels, out_channels=out_channels//2, kernel_size=kernel_size, stride=stride, padding=padding),
                            DWConvBNGELU(in_channels=out_channels//2, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                            DWConvBNGELU(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        )
    def forward(self, input):
        return self.blocks(input)
    
class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, down_sample_channels=None, down_sample=False, down_sample_4x=False) -> None:
        super().__init__()
        self.down_sample = down_sample
        self.down_sample_4x = down_sample_4x
        if down_sample:
            self.down_sample_blocks = nn.Sequential(
                                DWConvBNGELU(in_channels=down_sample_channels, out_channels=down_sample_channels, kernel_size=3, stride=2, padding=1),
                                DWConvBNGELU(in_channels=down_sample_channels, out_channels=down_sample_channels, kernel_size=3, stride=1, padding=1),
            )
            if down_sample_4x:
                layers = list(self.down_sample_blocks.children())
                layers = layers * 2
                self.down_sample_blocks = nn.Sequential(*layers)
        
        self.step_blocks = DWConvBNGELU_x3(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, input1, input2, input3):
        if self.down_sample:
            input_concat = torch.cat([
                self.down_sample_blocks(input1),
                input2,
                F.interpolate(input3, size=input2.size()[2:], mode='bilinear', align_corners=True)
            ], dim=1)
        else:
            input_concat = torch.cat([
                input1,
                F.interpolate(input2, size=input1.size()[2:], mode='bilinear', align_corners=True)
            ], dim=1)
        return self.step_blocks(input_concat)
    
class NED(nn.Module):
    def __init__(self, in_channels_list: list, out_channel_list: list, kernel_size=3, stride=1, padding=1, aux=False) -> None:
        super().__init__()
        self.aux = aux

        self.step1_cbg1 = DenseBlock(in_channels=in_channels_list[0][0], out_channels=out_channel_list[0][0], kernel_size=kernel_size, stride=stride, padding=padding)
        self.step1_cbg2 = DenseBlock(in_channels=in_channels_list[0][1], out_channels=out_channel_list[0][1], kernel_size=kernel_size, stride=stride, padding=padding, down_sample_channels=64, down_sample=True)
        self.step1_cbg3 = DenseBlock(in_channels=in_channels_list[0][2], out_channels=out_channel_list[0][2], kernel_size=kernel_size, stride=stride, padding=padding, down_sample_channels=128, down_sample=True)
        self.step1_cbg4 = DenseBlock(in_channels=in_channels_list[0][3], out_channels=out_channel_list[0][3], kernel_size=kernel_size, stride=stride, padding=padding, down_sample_channels=256, down_sample=True, down_sample_4x=True)

        self.step2_cbg1 = DenseBlock(in_channels=in_channels_list[1][0], out_channels=out_channel_list[1][0], kernel_size=kernel_size, stride=stride, padding=padding)
        self.step2_cbg2 = DenseBlock(in_channels=in_channels_list[1][1], out_channels=out_channel_list[1][1], kernel_size=kernel_size, stride=stride, padding=padding, down_sample_channels=64, down_sample=True)
        self.step2_cbg3 = DenseBlock(in_channels=in_channels_list[1][2], out_channels=out_channel_list[1][2], kernel_size=kernel_size, stride=stride, padding=padding, down_sample_channels=128, down_sample=True)

        self.step3_cbg1 = DenseBlock(in_channels=in_channels_list[2][0], out_channels=out_channel_list[2][0], kernel_size=kernel_size, stride=stride, padding=padding)
        self.step3_cbg2 = DenseBlock(in_channels=in_channels_list[2][1], out_channels=out_channel_list[2][1], kernel_size=kernel_size, stride=stride, padding=padding, down_sample_channels=64, down_sample=True)

        self.step4_cbg1 = DenseBlock(in_channels=in_channels_list[3][0], out_channels=out_channel_list[3][0], kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, input1, input2, input3, input4, input5):
        step1_cbg1 = self.step1_cbg1(input1=input1, input2=input2, input3=None)
        step1_cbg2 = self.step1_cbg2(input1=input1, input2=input2, input3=input3)
        step1_cbg3 = self.step1_cbg3(input1=input2, input2=input3, input3=input4)
        step1_cbg4 = self.step1_cbg4(input1=input3, input2=input4, input3=input5)

        step2_cbg1 = self.step2_cbg1(input1=step1_cbg1, input2=step1_cbg2, input3=None)
        step2_cbg2 = self.step2_cbg2(input1=step1_cbg1, input2=step1_cbg2, input3=step1_cbg3)
        step2_cbg3 = self.step2_cbg3(input1=step1_cbg2, input2=step1_cbg3, input3=step1_cbg4)

        step3_cbg1 = self.step3_cbg1(input1=step2_cbg1, input2=step2_cbg2, input3=None)
        step3_cbg2 = self.step3_cbg2(input1=step2_cbg1, input2=step2_cbg2, input3=step2_cbg3)

        step4_cbg1 = self.step4_cbg1(input1=step3_cbg1, input2=step3_cbg2, input3=None)

        if self.aux:
            return input1, step1_cbg1, step2_cbg1, step3_cbg1, step4_cbg1
        else:
            return step4_cbg1