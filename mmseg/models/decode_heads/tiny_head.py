# Copyright (c) Open-CD. All rights reserved.
import pdb
import einops as E
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from torch.nn.modules.utils import _pair
from mmseg.registry import MODELS
import numpy as np
import pdb
class ConvBNLeakyReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(ConvBNLeakyReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)


class MFF(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        r = 1
        c_ = c1//r
        self.APh = nn.AdaptiveAvgPool2d((None,1))
        self.APw = nn.AdaptiveAvgPool2d((1,None))
        self.cv1 = ConvBNLeakyReLU(c1, c_, 1)
        self.cv2 = ConvBNLeakyReLU(c_, c1, 1)
        self.cv3 = ConvBNLeakyReLU(c_, c1, 1)
        self.cv4 = ConvBNLeakyReLU(c1, c2, 1)
        self.bn = nn.BatchNorm2d(c_)
        self.sl = nn.SiLU()
        self.sg = nn.Sigmoid()

    def forward(self, x):
        yh = self.APh(x)
        yw = self.APw(x).permute(0,1,3,2)
        y1 = self.sl(self.bn(self.cv1(torch.cat((yh, yw), 2))))
        y2, y3 = torch.chunk(y1, 2, dim=2)
        y3 = y3.permute(0,1,3,2)
        zh = self.sg(self.cv2(y2))
        zw = torch.sigmoid(self.cv3(y3))
        Y = self.cv4(x * zh * zw)
        return Y

class Convolutional_Attention(nn.Module):
    def __init__(self,
                 channels,
                 down_channels,
                 num_heads,
                 img_size,
                 proj_drop=0.0,
                 kernel_size=1,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv="same",
                 padding_q="same",
                 attention_bias=True
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.num_heads = num_heads
        self.proj_drop = proj_drop

        self.layer_q = nn.Sequential(
            nn.Conv2d(channels, down_channels, kernel_size, stride_q, padding_q, bias=attention_bias, groups=down_channels),
            nn.ReLU(),
        )
        self.layernorm_q = nn.LayerNorm([down_channels,img_size,img_size], eps=1e-5)

        self.layer_k = nn.Sequential(
            nn.Conv2d(channels, down_channels, kernel_size, stride_kv, padding_kv, bias=attention_bias, groups=down_channels),
            nn.ReLU(),
        )
        self.layernorm_k = nn.LayerNorm([down_channels,img_size,img_size], eps=1e-5)

        self.layer_v = nn.Sequential(
            nn.Conv2d(channels, down_channels, kernel_size, stride_kv, padding_kv, bias=attention_bias, groups=down_channels),
            nn.ReLU(),
        )
        self.layernorm_v = nn.LayerNorm([down_channels,img_size,img_size], eps=1e-5)
        
        self.attention = nn.MultiheadAttention(embed_dim=down_channels, 
                                               bias=attention_bias, 
                                               batch_first=True,
                                               dropout=self.proj_drop,
                                               num_heads=self.num_heads)

    def _build_projection(self, x, mode):
        # x shape [batch,channel,size,size]
        # mode:0->q,1->k,2->v,for torch.script can not script str

        if mode == 0:
            proj = self.layer_q(x)
            proj = self.layernorm_q(proj)
        elif mode == 1:
            proj = self.layer_k(x)
            proj = self.layernorm_k(proj)
        elif mode == 2:
            proj = self.layer_v(x)
            proj = self.layernorm_v(proj)

        return proj

    def get_qkv(self, x,cut_x):
        q = self._build_projection(cut_x, 0)
        k = self._build_projection(cut_x, 1)
        v = self._build_projection(x, 2)

        return q, k, v

    def forward(self, x,cut_x):
        q, k, v = self.get_qkv(x,cut_x)
        q = q.view(q.shape[0], q.shape[1], q.shape[2]*q.shape[3])
        k = k.view(k.shape[0], k.shape[1], k.shape[2]*k.shape[3])
        v = v.view(v.shape[0], v.shape[1], v.shape[2]*v.shape[3])
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        x1 = self.attention(query=q, value=v, key=k, need_weights=False)

        x1 = x1[0].permute(0, 2, 1)
        x1 = x1.view(x1.shape[0], x1.shape[1], np.sqrt(
            x1.shape[2]).astype(int), np.sqrt(x1.shape[2]).astype(int))

        return x1

class FCUDown(nn.Module):

    def __init__(self,out_channels):
        super().__init__()
        self.sample_pooling = nn.AvgPool2d(kernel_size=4, stride=4)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # pdb.set_trace()
        x = self.sample_pooling(x)
        x = E.rearrange(x, "B C H W  -> B C (H W)")
        x = self.activation(x)
        return x

class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        B, n_patch, hidden = x.size() 
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor)(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

@MODELS.register_module()
class TinyHead(BaseDecodeHead):

    def __init__(self, feature_strides, priori_attn=False, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        if priori_attn:
            attn_channels = self.in_channels[0]
            self.in_channels = self.in_channels[1:]
            feature_strides = feature_strides[1:]
        self.feature_strides = feature_strides
        self.priori_attn = priori_attn


        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            scale_head = []
            scale_head.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    groups=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            # scale_head.append(MFF(self.in_channels[i],self.channels))
            self.scale_heads.append(nn.Sequential(*scale_head))


        self.early_conv = ConvModule(
                    in_channels=16,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    groups=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)

        if self.priori_attn:
            self.gen_diff_attn = ConvModule(
                in_channels=attn_channels,
                out_channels=self.channels,
                kernel_size=1,
                stride=1,
                groups=1,
                norm_cfg=None,
                act_cfg=None
            )

        self.ca = ChannelAttention(self.channels * 5, ratio=1)
        self.ca1 = ChannelAttention(self.channels, ratio=1)

        self.out = ConvModule(
            in_channels=self.channels * 5,
            out_channels=self.channels,
            kernel_size=1,
            stride=1,
            groups=1,
            norm_cfg=None,
            act_cfg=None
        )

        # self.sample_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.att = Convolutional_Attention(24,8,1,64)
        # self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.conv_early = nn.Conv2d(12, 24,kernel_size=1)

    def forward(self, inputs):

        x = self._transform_inputs(inputs)

        if self.priori_attn:
            early_x = x[0]
            x = x[1:]

        early_output = self.early_conv(early_x)
        output1 = self.scale_heads[0](x[0])
        output2 = resize(self.scale_heads[1](x[1]),size=output1.shape[2:],mode='bilinear',align_corners=self.align_corners)
        output3 = resize(self.scale_heads[2](x[2]),size=output1.shape[2:],mode='bilinear',align_corners=self.align_corners)
        output4 = resize(self.scale_heads[3](x[3]),size=output1.shape[2:],mode='bilinear',align_corners=self.align_corners)


        # early_output = self.sample_pooling(early_output)
        # output_cut = self.sample_pooling(output1)
        # early_output = self.att(early_output,output1)
        # early_output = self.conv_early(early_output)


        out = torch.cat([early_output,output1, output2, output3, output4], 1)

        intra = torch.sum(torch.stack((early_output,output1, output2, output3, output4)), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1, 5, 1, 1))
        out = self.out(out)
        
        
        
        output = self.cls_seg(out)
        return output