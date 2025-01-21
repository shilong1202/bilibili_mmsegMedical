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
class UNetss(BaseModule):
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

        self.de_0 = DoubleConv((8 + 12)*init_channels, 8*init_channels, with_bn)
        self.de_1 = DoubleConv((4 + 8)*init_channels, 4*init_channels, with_bn)
        self.de_2 = DoubleConv((2 + 4)*init_channels, 2*init_channels, with_bn)
        self.de_3 = DoubleConv((1 + 2)*init_channels, 1*init_channels, with_bn)
        # self.de_4 = nn.Conv2d(init_channels, out_channels, 1)

        self.sigmod = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.down1 = ConvModule(init_channels,init_channels,kernel_size=3,stride=2,padding=1,conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.down2 = ConvModule(2*init_channels,2*init_channels,kernel_size=3,stride=2,padding=1,conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.down3 = ConvModule(4*init_channels,4*init_channels,kernel_size=3,stride=2,padding=1,conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.down4 = ConvModule(8*init_channels,8*init_channels,kernel_size=3,stride=2,padding=1,conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.upsample1 = nn.ConvTranspose2d(12*init_channels,12*init_channels,kernel_size=(2,2),stride=2)
        self.upsample2 = nn.ConvTranspose2d(8*init_channels,8*init_channels,kernel_size=(2,2),stride=2)
        self.upsample3 = nn.ConvTranspose2d(4*init_channels,4*init_channels,kernel_size=(2,2),stride=2)
        self.upsample4 = nn.ConvTranspose2d(2*init_channels,2*init_channels,kernel_size=(2,2),stride=2)
    
    def forward(self, x):
        e1 = self.en_1(x)
        e2 = self.en_2(self.down1(e1))
        e3 = self.en_3(self.down2(e2))
        e4 = self.en_4(self.down3(e3))
        e5 = self.en_5(self.down4(e4))

        d0 = self.de_0(torch.cat([self.upsample1(e5), e4], dim=1))
        d1 = self.de_1(torch.cat([self.upsample2(d0), e3], dim=1))
        d2 = self.de_2(torch.cat([self.upsample3(d1), e2], dim=1))
        d3 = self.de_3(torch.cat([self.upsample4(d2), e1], dim=1))
        # d4 = self.de_4(d3)  
        # print(d3.shape)
    
        return (e5,d0,d1,d2,d3)
