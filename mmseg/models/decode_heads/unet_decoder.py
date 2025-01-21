import torch
from torch import nn
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, with_bn=False):
        super().__init__()
        if with_bn:
            self.step = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        else:
            self.step = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
        
    def forward(self, x):
        return self.step(x)

@MODELS.register_module()
class UNet_decoder(BaseDecodeHead):
    def __init__(self, in_channels, base_channels,channels,num_classes,with_bn=True, **kwargs):
        super().__init__(in_channels = in_channels,channels=channels,num_classes=num_classes)
        init_channels = base_channels
        norm_cfg=dict(type='BN')
        act_cfg=dict(type='ReLU')
        conv_cfg=None

        self.de_0 = DoubleConv((8 + 12)*init_channels, 8*init_channels, with_bn)
        self.de_1 = DoubleConv((4 + 8)*init_channels, 4*init_channels, with_bn)
        self.de_2 = DoubleConv((2 + 4)*init_channels, 2*init_channels, with_bn)
        self.de_3 = DoubleConv((1 + 2)*init_channels, 1*init_channels, with_bn)

        self.sigmod = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        

    def forward(self, x):
        (e1,e2,e3,e4,e5) = x

        d0 = self.de_0(torch.cat([self.upsample(e5), e4], dim=1))
        d1 = self.de_1(torch.cat([self.upsample(d0), e3], dim=1))
        d2 = self.de_2(torch.cat([self.upsample(d1), e2], dim=1))
        d3 = self.de_3(torch.cat([self.upsample(d2), e1], dim=1))
        d3 = self.cls_seg(d3)
        return d3
