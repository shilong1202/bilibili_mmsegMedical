import torch
from torch import nn
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS

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
class UNet_encoder(BaseModule):
    def __init__(self, in_channels, base_channels,with_bn=True):
        super().__init__()
        init_channels = base_channels
        norm_cfg=dict(type='BN')
        act_cfg=dict(type='ReLU')
        conv_cfg=None
        # self.out_channels = out_channels

        self.en_1 = DoubleConv(in_channels    , init_channels  , with_bn)
        self.en_2 = DoubleConv(1*init_channels, 2*init_channels, with_bn)
        self.en_3 = DoubleConv(2*init_channels, 4*init_channels, with_bn)
        self.en_4 = DoubleConv(4*init_channels, 8*init_channels, with_bn)
        self.en_5 = DoubleConv(8*init_channels, 12*init_channels, with_bn)

        self.sigmod = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        

    def forward(self, x):
        e1 = self.en_1(x)
        e2 = self.en_2(self.maxpool(e1))
        e3 = self.en_3(self.maxpool(e2))
        e4 = self.en_4(self.maxpool(e3))
        e5 = self.en_5(self.maxpool(e4))  
        return (e1,e2,e3,e4,e5)
