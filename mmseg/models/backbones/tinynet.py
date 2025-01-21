# Copyright (c) Open-CD. All rights reserved.
import warnings
import numpy as np
import math
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils import checkpoint as cp
import pywt
from mmseg.models.utils import SELayer, make_divisible
from mmseg.registry import MODELS
from torch.autograd import Function
import pdb

class Waveletattchannel(nn.Module):

    def __init__(self, in_planes=3):
        super().__init__()
        wavename = 'haar'
        self.downsamplewavelet = nn.Sequential(*[nn.Upsample(scale_factor=2),Downsamplewave1(wavename=wavename)])
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // 2, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        xori = x
        B, C, H, W= x.shape
        x = x.reshape(B, H, W, C)
        x = x.permute(0, 3, 2, 1)
        y = self.downsamplewavelet(x)
        
        y = self.fc(y).reshape(B, C, 1, 1)
        y = xori * y.expand_as(xori)   
        # pdb.set_trace()    
        return y

class Waveletattspace(nn.Module):

    def __init__(self,in_planes=3):
        super().__init__()
        wavename = 'haar'
        self.downsamplewavelet = nn.Sequential(*[nn.Upsample(scale_factor=2),Downsamplewave(wavename=wavename)])
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes*2, in_planes//2, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes//2, in_planes,kernel_size=1,padding= 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        xori = x
        B, C, H, W= x.shape

        x = x.reshape(B, H, W, C)
        x = x.permute(0, 3, 2, 1)        
        y = self.downsamplewavelet(x)
        y = self.fc(y)
        y = xori * y.expand_as(xori)       
        return y

class Waveleteatt(nn.Module):

    def __init__(self, in_planes=64):
        super(Waveleteatt, self).__init__()
        
        self.att2 = Waveletattchannel(in_planes=in_planes)
        self.attspace2 = Waveletattspace(in_planes=in_planes)


    def forward(self, x):
        x = self.att2(x)
        x = self.attspace2(x)
        return x

class WaveleteCPCAChannelAttention(nn.Module):
 
    def __init__(self, input_channels):
        super(WaveleteCPCAChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels * 2, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        # self.att = Waveletattchannel(in_planes=input_channels)
        self.attspace = Waveletattspace(in_planes=input_channels)
        self.out = Wide_Focus(input_channels,input_channels)
 
    def forward(self, inputs):
        
        # x = self.attspace(inputs)
        
        # x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        # x1 = self.fc1(x1)
        # x1 = F.relu(x1, inplace=True)
        # x1 = self.fc2(x1)
        # x1 = torch.sigmoid(x1) * x1.expand_as(inputs)  
        # x1 = self.attspace(x1)
        
        # x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        # x2 = self.fc1(x2)
        # x2 = F.relu(x2, inplace=True)
        # x2 = self.fc2(x2)
        # x2 = torch.sigmoid(x2)  * x2.expand_as(inputs)  
        # x2 = self.attspace(x2)

        # x = x1 + x2 + x
        
        x = self.attspace(inputs)
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        
        x3 = torch.cat([x1,x2],dim=1)
        x3 = self.fc1(x3)
        x3 = F.relu(x2, inplace=True)
        x3 = self.fc2(x3)
        x3 = torch.sigmoid(x3)  * x3.expand_as(inputs)  
        x = x + x3
        
        x = self.out(x)
        return x

class Downsamplewave(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(Downsamplewave, self).__init__()
        self.dwt = DWT_2D(wavename = wavename)   #return LL,LH,HL,HH

    def forward(self, input):
        LL,LH,HL,HH = self.dwt(input)
        return torch.cat([LL,LH+HL+HH],dim=1)

class Downsamplewave1(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(Downsamplewave1, self).__init__()
        self.dwt = DWT_2D(wavename = wavename)   #return LL,LH,HL,HH

    def forward(self, input):
        LL,LH,HL,HH = self.dwt(input)
        LL = LL+LH+HL+HH
        result = torch.sum(LL, dim=[2, 3])  # x:torch.Size([64, 256, 56, 56])
        return result  ###torch.Size([64, 256])
    

class DWT_2D(nn.Module):

    def __init__(self, wavename):
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        """
        input_lfc = \mathcal{L} * input * \mathcal{L}^T
        input_hfc_lh = \mathcal{H} * input * \mathcal{L}^T
        input_hfc_hl = \mathcal{L} * input * \mathcal{H}^T
        input_hfc_hh = \mathcal{H} * input * \mathcal{H}^T
        :param input: the 2D data to be decomposed
        :return: the low-frequency and high-frequency components of the input 2D data
        """
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)

class DWTFunction_2D(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)
        LL = torch.matmul(L, matrix_Low_1)
        LH = torch.matmul(L, matrix_High_1)
        HL = torch.matmul(H, matrix_Low_1)
        HH = torch.matmul(H, matrix_High_1)
        return LL, LH, HL, HH
    @staticmethod
    def backward(ctx, grad_LL, grad_LH, grad_HL, grad_HH):
        # ctx, grad_LL, grad_LH, grad_HL, grad_HH = ctx.float(), grad_LL.float(), grad_LH.float(), grad_HL.float(), grad_HH.float()
        # pdb.set_trace()
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_L = torch.add(torch.matmul(grad_LL.float(), matrix_Low_1.t().float()), torch.matmul(grad_LH.float(), matrix_High_1.t().float()))
        grad_H = torch.add(torch.matmul(grad_HL.float(), matrix_Low_1.t().float()), torch.matmul(grad_HH.float(), matrix_High_1.t().float()))
        grad_input = torch.add(torch.matmul(matrix_Low_0.t().float(), grad_L), torch.matmul(matrix_High_0.t().float(), grad_H))
        return grad_input.float(), None, None, None, None

class AsymGlobalAttn(BaseModule):
    def __init__(self, dim, strip_kernel_size=21):
        super().__init__()

        self.norm = build_norm_layer(dict(type='mmpretrain.LN2d', eps=1e-6), dim)[1]
        self.global_ = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.Conv2d(dim, dim, (1, strip_kernel_size), padding=(0, (strip_kernel_size-1)//2), groups=dim),
                nn.Conv2d(dim, dim, (strip_kernel_size, 1), padding=((strip_kernel_size-1)//2, 0), groups=dim)
        )

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.layer_scale = nn.Parameter(1e-6 * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        
        a = self.global_(x)
        x = a * self.v(x)
        x = self.proj(x)
        x = self.norm(x)
        x = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * x + identity

        return x




class PriorAttention(BaseModule):
    def __init__(self, 
                 channels, 
                 num_paths=1, 
                 attn_channels=None, 
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(PriorAttention, self).__init__()
        self.num_paths = num_paths # `2` is supported.
        attn_channels = attn_channels or channels // 16
        attn_channels = max(attn_channels, 8)
        
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = build_norm_layer(norm_cfg, attn_channels)[1]
        self.act = build_activation_layer(act_cfg)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

    def forward(self, x): 
        attn = x.mean((2, 3), keepdim=True)# [2, 32, 256, 256]
        attn = self.fc_reduce(attn)
        attn = self.bn(attn)
        attn = self.act(attn)
        attn = self.fc_select(attn)
        attn1 = torch.sigmoid(attn)
        return x * attn1 + x


class StemBlock(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 with_cp=False,
                 **kwargs):
        super(StemBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2], f'stride must in [1, 2]. ' \
            f'But received {stride}.'
        self.with_cp = with_cp
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **kwargs))
        layers.extend([
            ConvModule(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=hidden_dim,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                **kwargs),
        ])
        
        self.conv = nn.Sequential(*layers)
        self.interact = PriorAttention(channels=hidden_dim)
        self.post_conv = ConvModule(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
                **kwargs)

    def forward(self, x):
        identity = x

        x1 = self.conv(x)
        x1 = self.interact(x1)
        x1 = self.post_conv(x1)

        if self.use_res_connect:
            x1 = x1 + identity
        return x1


class PriorFusion(BaseModule):
    def __init__(self, channels, stack_nums=2):
        super().__init__()

        self.stem = Wide_Focus(channels,channels)

        self.pseudo_fusion = nn.Sequential(
                nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2),
                build_norm_layer(dict(type='mmpretrain.LN2d', eps=1e-6), channels * 2)[1],
                nn.GELU(),
                nn.Conv2d(channels * 2, channels, 3, padding=1, groups=channels),
        )

    
    def forward(self, x):
        
        identity = x
        
        x1 = self.stem(x)
        x1 = x1 + identity
        early_x =  torch.cat([x, x1], dim=1)
        x = self.pseudo_fusion(early_x)
        return early_x, x


class TinyBlock(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 with_cp=False,
                 with_se=False,
                 **kwargs):
        super(TinyBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2], f'stride must in [1, 2]. ' \
            f'But received {stride}.'
        self.with_cp = with_cp
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        Attention_Layer = SELayer(hidden_dim) if with_se else nn.Identity()
        if expand_ratio != 1:
            layers.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **kwargs))
        layers.extend([
            ConvModule(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=hidden_dim,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                **kwargs),
            Attention_Layer,
            ConvModule(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
                **kwargs)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):

        def _inner_forward(x):
            if self.use_res_connect:
                x = x + self.conv(x)
                return x
            else:
                return self.conv(x)

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class Wide_Focus(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.norm_cfg = dict(type='BN')
        self.act_cfg = dict(type='ReLU6')
        self.layer1 = ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.layer2 = ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=3,dilation=2, stride=1, padding=2, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.layer3 = ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=3,dilation=3, stride=1, padding=3, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        self.layer4 = ConvModule(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        added = x1 + x2 + x3
        x_out = self.layer4(added)
        return x_out

class PFC(nn.Module):
    def __init__(self,inchannel,channels, kernel_size=7):
        super(PFC, self).__init__()
        self.input_layer = nn.Sequential(
                    nn.Conv2d(inchannel, channels, kernel_size, padding=  kernel_size // 2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
        self.depthwise = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size, groups=channels, padding= kernel_size // 2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
        self.pointwise = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
    def forward(self, x):
        x = self.input_layer(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

@MODELS.register_module()
class TinyNet(BaseModule):

    change_extractor_settings = {
        'S': [[4, 16, 2], [6, 24, 2], [6, 32, 3], [6, 48, 1]],
        'B': [[4, 16, 2], [6, 24, 2], [6, 32, 3], [6, 48, 1]], 
        'L': [[4, 16, 2], [6, 24, 2], [6, 32, 6], [6, 48, 1]],}

    def __init__(self,
                 output_early_x=False,
                 arch='B',
                 stem_stack_nums=1,
                 use_global=(True, True, True, True),
                 strip_kernel_size=(41, 31, 21, 11),
                 widen_factor=1.,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.arch_settings = self.change_extractor_settings[arch]
        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        self.widen_factor = widen_factor
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == len(self.arch_settings)
        self.out_indices = out_indices
        for index in out_indices:
            if index not in range(0, 7):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 7). But received {index}')

        if frozen_stages not in range(-1, 7):
            raise ValueError('frozen_stages must be in range(-1, 7). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.in_channels = make_divisible(16 * widen_factor, 8)

        # self.conv1 = Wide_Focus(3,self.in_channels)

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.fusion_block = PriorFusion(self.in_channels, stem_stack_nums)

        self.layers = []
        self.use_global = use_global
        self.strip_kernel_size = strip_kernel_size

        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks = layer_cfg
            stride = self.strides[i]
            dilation = self.dilations[i]
            out_channels = make_divisible(channel * widen_factor, 8)
            inverted_res_layer = self.make_layer(
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                expand_ratio=expand_ratio,
                use_global=use_global[i],
                strip_kernel_size=self.strip_kernel_size[i])
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)
        
        self.output_early_x = output_early_x
        # exit()

    def make_layer(self, out_channels, num_blocks, stride, dilation,
                   expand_ratio, use_global, strip_kernel_size):
        layers = []
        for i in range(num_blocks):
            layers.append(
                TinyBlock(
                    self.in_channels,
                    out_channels,
                    stride if i == 0 else 1,
                    expand_ratio=expand_ratio,
                    dilation=dilation if i == 0 else 1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp))
            self.in_channels = out_channels
        # after stage
        if use_global:
            layers.append(
                WaveleteCPCAChannelAttention(out_channels)
            )
        # print(strip_kernel_size)

        return nn.Sequential(*layers)

    def forward(self, x1):

        x1 = self.conv1(x1)

        early_x, x = self.fusion_block(x1)

        if self.output_early_x:
            outs = [early_x]
        else:
            outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(TinyNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()