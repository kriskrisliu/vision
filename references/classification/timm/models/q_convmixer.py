"""
    Quantized MlpMixer for ImageNet-1K, implemented in PyTorch.
"""

from multiprocessing import pool
from matplotlib.pyplot import polar
from numpy import block
import torch
import torch.nn as nn
import copy
import time
import logging
import math
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn

import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg, overlay_external_default_cfg, named_apply, resume_checkpoint, load_checkpoint
from .registry import register_model
from .mlp_mixer import _cfg, default_cfgs
from .layers import PatchEmbed, Mlp, GluMlp, GatedMlp, DropPath, lecun_normal_, to_2tuple
from .quantization_utils.quant_modules import QuantLinear,QuantAct,QuantConv2d
from .quantization_utils.quant_utils import *
from .bit_config import bit_config_dict
from .q_mlp import q_PatchEmbed
from .utils.pg_utils import PactReLU

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        'first_conv': 'stem.0',
        **kwargs
    }


default_cfgs = {
    'convmixer_1536_20': _cfg(url='https://github.com/tmp-iclr/convmixer/releases/download/timm-v1.0/convmixer_1536_20_ks9_p7.pth.tar'),
    'convmixer_768_32': _cfg(url='https://github.com/tmp-iclr/convmixer/releases/download/timm-v1.0/convmixer_768_32_ks7_p7_relu.pth.tar'),
    'convmixer_1024_20_ks9_p14': _cfg(url='https://github.com/tmp-iclr/convmixer/releases/download/timm-v1.0/convmixer_1024_20_ks9_p14.pth.tar')
}

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class q_Pool(nn.Module):
    def __init__(self, target_size):
        super().__init__()
        # self.pool=QuantAdaptiveAvgPool2d(target_size)
        # self.flat=nn.Flatten()
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            # QuantAdaptiveAvgPool2d(target_size),
            nn.Flatten()
        )
        # self.pooling = nn.Sequential(
        #     # nn.AdaptiveAvgPool2d((1, 1)),
        #     QuantAdaptiveAvgPool2d(target_size),
        #     nn.Flatten()
        # )
        self.quant=QuantAct()
        
    def set_param(self, pool):
        # self.pooling[0].set_param(pool[0])
        self.pooling[0]=pool[0]

    def forward(self, tup):
        x, scaling_factor = tup
        # print("q_Pool before pool: ", x.shape)
        # x, act_scaling_factor = self.pool(x,scaling_factor)
        # print("q_Pool: ", act_scaling_factor.shape)
        # nn.Flatten(x)
        # x = x.squeeze()
        # print("q scale: ",scaling_factor)
        # print("q befff pool: ", x.shape , x.data.min(), x.data.max())
        # y = self.pooling1(x)
        # print("q afttt pool: ", y.shape , y.data.min(), y.data.max())

        # print("q befff pool: ", x.shape , x.data.min(), x.data.max())
        x = self.pooling(x)
        print("q afttt pool: ", x.shape , x.data.min(), x.data.max())

        # x = self.pooling(tup)
        x,act_scaling_factor = self.quant(x)
        print(self.quant.__repr__())
        print("q afttt poolquant: ", x.shape , x.data.min(), x.data.max())

        return x,act_scaling_factor

class q_ExtractPatch(nn.Module):
    def __init__(self,  in_chans=3, dim=768, patch_size=16, activation=nn.GELU):
        super().__init__()

        self.patch_size = patch_size

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.quant_input=QuantAct()
        self.proj = QuantConv2d()
        self.proj.set_param(nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size))

        self.quant_act_conv=QuantAct()
        self.act = activation()
        self.qact1=QuantAct()
        self.bn=nn.BatchNorm2d(dim)
        self.qact2=QuantAct()

    def set_param(self, unit):
        self.proj.set_param(unit[0])
        self.act=unit[1]
        self.bn=unit[2]


    def forward(self, x):
        x, pre_act_scaling_factor = self.quant_input(x)
        x, conv_scaling_factor  = self.proj(x,pre_act_scaling_factor)
        x, act_scaling_factor = self.quant_act_conv(x,pre_act_scaling_factor,conv_scaling_factor)

        x = self.act(x)
        x, act_scaling_factor = self.qact1(x)
        x = self.bn(x)
        x, act_scaling_factor = self.qact2(x) 

        return x, act_scaling_factor

class q_ExtractPatch_new(nn.Module):
    def __init__(self,  in_chans=3, dim=768, patch_size=16, activation=nn.GELU):
        super().__init__()

        self.patch_size = patch_size

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.quant_input=QuantAct()
        self.proj = QuantConv2d()
        self.proj.set_param(nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size))

        self.quant_act_conv=QuantAct()
        self.act = activation()
        self.qact1=QuantAct()
        self.bn=nn.BatchNorm2d(dim)
        self.qact2=QuantAct()

    def set_param(self, unit):
        self.proj.set_param(unit[0])
        self.bn=unit[1]


    def forward(self, x):
        x, pre_act_scaling_factor = self.quant_input(x)
        x, conv_scaling_factor  = self.proj(x,pre_act_scaling_factor)
        x, act_scaling_factor = self.quant_act_conv(x,pre_act_scaling_factor,conv_scaling_factor)

        x = self.bn(x)
        x, act_scaling_factor = self.qact1(x)
        x = self.act(x)
        x, act_scaling_factor = self.qact2(x) 

        return x, act_scaling_factor

class q_convMixerBlock(nn.Module):
    def __init__(self, dim, kernel_size=9,  activation=nn.GELU,  **kwargs):
        super().__init__()
        self.num_features = dim
        self.conv_1=QuantConv2d()
        self.conv_1.set_param(nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"))
        self.qact_11=QuantAct()
        self.act_1=activation()
        self.qact_12=QuantAct()
        self.bn_1=nn.BatchNorm2d(dim)

        self.quant_act_int32 = QuantAct()

        self.conv_2=QuantConv2d()
        self.conv_2.set_param(nn.Conv2d(dim, dim, kernel_size=1))
        self.qact_21=QuantAct()
        self.act_2=activation()
        self.qact_22=QuantAct()
        self.bn_2=nn.BatchNorm2d(dim)
        self.qact_23=QuantAct()

    def set_param(self, blocki):
        self.conv_1.set_param(blocki[0].fn[0])
        self.conv_2.set_param(blocki[1])
        
        self.bn_1=blocki[0].fn[2]
        self.bn_2=blocki[3]

        self.act_1=blocki[0].fn[1]
        self.act_2=blocki[2]

    def forward(self, tup):
        x, scaling_factor= tup
        identity = x
        x, conv_scaling_factor= self.conv_1(x,scaling_factor)
        # print("q bef qact_11: ", x.data.min(), x.data.max())
        # x, act_scaling_factor = self.qact_11(x,scaling_factor,conv_scaling_factor)
        # x, act_scaling_factor = self.qact_11(x)
        # print("q aft qact_11: ", x.shape , x.data.min(), x.data.max(), act_scaling_factor)

        x = self.act_1(x)
        # print("q bef qact_12 shape / min / max: ", x.shape , x.data.min(), x.data.max())
        # print("q bef qact_12 mean / var: ", torch.mean(x),torch.var(x))
        # x_min_pt, x_max_pt = get_percentile_min_max(x.detach().contiguous().view(-1), 100 - 99,
        #                                               99, output_tensor=True)
        # print("q bef qact_23 99% percentile: ", x_min_pt,x_max_pt)
        # x_min_pt, x_max_pt = get_percentile_min_max(x.detach().contiguous().view(-1), 100 - 99.5,
        #                                               99.5, output_tensor=True)
        # print("q bef qact_23 99.5% percentile: ", x_min_pt,x_max_pt)
        # x_min_pt, x_max_pt = get_percentile_min_max(x.detach().contiguous().view(-1), 100 - 99.9,
        #                                               99.9, output_tensor=True)
        # print("q bef qact_23 99.9% percentile: ", x_min_pt,x_max_pt)
        x, act_scaling_factor = self.qact_12(x)
        # print("q aft qact_12: ", x.shape , x.data.min(), x.data.max(), act_scaling_factor)
        x = self.bn_1(x)

        x = x + identity
        x, scaling_factor_int32 = self.quant_act_int32(x)

        x, conv_scaling_factor= self.conv_2(x, scaling_factor_int32)
        # print("q bef qact_21: ", x.data.min(), x.data.max())
        # x, act_scaling_factor = self.qact_21(x, scaling_factor_int32 ,conv_scaling_factor)
        # x, act_scaling_factor = self.qact_21(x)
        # print("q aft qact_21: ", x.shape , x.data.min(), x.data.max(), act_scaling_factor)

        x = self.act_2(x)
        # print("q bef qact_22 shape / min / max: ", x.shape , x.data.min(), x.data.max())
        # print("q bef qact_22 mean / var: ", torch.mean(x),torch.var(x))
        # x_min_pt, x_max_pt = get_percentile_min_max(x.detach().contiguous().view(-1), 100 - 99,
        #                                               99, output_tensor=True)
        # print("q bef qact_23 99% percentile: ", x_min_pt,x_max_pt)
        # x_min_pt, x_max_pt = get_percentile_min_max(x.detach().contiguous().view(-1), 100 - 99.5,
        #                                               99.5, output_tensor=True)
        # print("q bef qact_23 99.5% percentile: ", x_min_pt,x_max_pt)
        # x_min_pt, x_max_pt = get_percentile_min_max(x.detach().contiguous().view(-1), 100 - 99.9,
        #                                               99.9, output_tensor=True)
        # print("q bef qact_23 99.9% percentile: ", x_min_pt,x_max_pt)
        x, act_scaling_factor = self.qact_22(x)
        # print("q aft qact_22: ", x.shape , x.data.min(), x.data.max(), act_scaling_factor)
        x = self.bn_2(x)
        # print("q aft bn2: ", x.shape , x.data.min(), x.data.max())
        # print("q bef qact_23 shape / min / max: ", x.shape , x.data.min(), x.data.max())
        # print("q bef qact_23 mean / var: ", torch.mean(x),torch.var(x))
        # x_min_pt, x_max_pt = get_percentile_min_max(x.detach().contiguous().view(-1), 100 - 99,
        #                                               99, output_tensor=True)
        # print("q bef qact_23 99% percentile: ", x_min_pt,x_max_pt)
        # x_min_pt, x_max_pt = get_percentile_min_max(x.detach().contiguous().view(-1), 100 - 99.5,
        #                                               99.5, output_tensor=True)
        # print("q bef qact_23 99.5% percentile: ", x_min_pt,x_max_pt)
        # x_min_pt, x_max_pt = get_percentile_min_max(x.detach().contiguous().view(-1), 100 - 99.9,
        #                                               99.9, output_tensor=True)
        # print("q bef qact_23 99.9% percentile: ", x_min_pt,x_max_pt)
        # print("")
        x, scaling_factor = self.qact_23(x)
        # print(self.qact_23.__repr__(),scaling_factor)
        # print("q aft qact23: ", x.shape , x.data.min(), x.data.max())

        return (x, scaling_factor)
    def ptminmax(self,mode):
        if mode==0:
            # self.quant_act_int32.ptminmax()
            self.qact_23.ptminmax()


class q_convMixerBlock_new(nn.Module):
    def __init__(self, dim, kernel_size=9,  activation=nn.GELU,  **kwargs):
        super().__init__()
        self.num_features = dim
        self.conv_1=QuantConv2d()
        self.conv_1.set_param(nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"))
        # self.qact_11=QuantAct()
        self.act_1=activation()
        self.qact_12=QuantAct()
        self.bn_1=nn.BatchNorm2d(dim)

        self.quant_act_int32 = QuantAct()

        self.conv_2=QuantConv2d()
        self.conv_2.set_param(nn.Conv2d(dim, dim, kernel_size=1))
        # self.qact_21=QuantAct()
        self.act_2=activation()
        self.qact_22=QuantAct()
        self.bn_2=nn.BatchNorm2d(dim)
        self.qact_23=QuantAct()

    def set_param(self, blocki):
        self.conv_1.set_param(blocki[0].fn[0])
        self.conv_2.set_param(blocki[1])
        
        self.bn_1=blocki[0].fn[1]
        self.bn_2=blocki[2]

    def forward(self, tup):
        x, scaling_factor= tup
        identity = x
        x, conv_scaling_factor= self.conv_1(x,scaling_factor)
        # print("q bef qact_11: ", x.data.min(), x.data.max())
        # x, act_scaling_factor = self.qact_11(x,scaling_factor,conv_scaling_factor)
        # print("q aft qact_11: ", x.shape , x.data.min(), x.data.max(), act_scaling_factor)

        x = self.bn_1(x)
        # print("q aft bn_1: ", x.shape , x.data.min(), x.data.max())
        x, act_scaling_factor = self.qact_12(x)
        # print(self.qact_12.__repr__(),act_scaling_factor)
        # print("q aft qact_12: ", x.shape , x.data.min(), x.data.max(), act_scaling_factor)
        x = self.act_1(x)

        x = x + identity
        x, scaling_factor_int32 = self.quant_act_int32(x)

        x, conv_scaling_factor= self.conv_2(x, scaling_factor_int32)
        # print("q bef qact_21: ", x.data.min(), x.data.max())
        # x, act_scaling_factor = self.qact_21(x, scaling_factor_int32 ,conv_scaling_factor)
        # x, act_scaling_factor = self.qact_21(x)
        # print("q aft qact_21: ", x.shape , x.data.min(), x.data.max(), act_scaling_factor)

        x = self.bn_2(x)
        # print("q aft bn_2: ", x.shape , x.data.min(), x.data.max())
        x, act_scaling_factor = self.qact_22(x)
        # print(self.qact_22.__repr__(),act_scaling_factor)
        x = self.act_2(x)
        # print("q aft act_2: ", x.shape , x.data.min(), x.data.max())
        # print(" ")

        x, scaling_factor = self.qact_23(x)

        return (x, scaling_factor)

class q_ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, in_chans=3, num_classes=1000, stem_layer=q_ExtractPatch,
                    block_layer=q_convMixerBlock, activation=nn.GELU, model=None, showmm=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = dim
        # self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head = QuantLinear()
        self.head.set_param(nn.Linear(dim, num_classes))
        self.stem = stem_layer(dim=dim, patch_size=patch_size, in_chans=in_chans, activation=activation)
        self.showmm = showmm
        self.num_blocks=depth

        self.blocks = nn.Sequential(
            *[
                # nn.Sequential(
                #     Residual(nn.Sequential(
                #         # nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                #         QuantConv2d(),
                #         activation(),
                #         QuantAct(),
                #         nn.BatchNorm2d(dim)
                #     )),
                #     # nn.Conv2d(dim, dim, kernel_size=1),
                #     QuantConv2d(),
                #     activation(),
                #     QuantAct(),
                #     nn.BatchNorm2d(dim) )
                block_layer(dim,kernel_size=kernel_size,activation=activation)
                for i in range(depth)]
        )

        self.pooling = q_Pool((1,1))
        self.qct_pool = QuantAct()

        if model is not None:
            stem=getattr(model, "stem")
            blocks=getattr(model, "blocks")
            pooling=getattr(model, "pooling")
            head=getattr(model, "head")

            self.stem.set_param(stem) 
            # self.stem[0].set_param(stem[0])
            # self.stem[3]=stem[2]

            for i in range(depth):
                stage = getattr(blocks, "{}".format(i))
                self.blocks[i].set_param(stage)

            # self.pooling.set_param(pooling)
            self.pooling = pooling
            self.head.set_param(head)

    def get_classifier(self):
        return self.head
        
    def forward_features(self, x):
        # print("\nq bef stem: ", x.shape , x.data.min(), x.data.max())
        # print("\n============ New iter ===============")
        x, act_scaling_factor = self.stem(x)
        # print("q aft stem: ", x.shape , x.data.min(), x.data.max(),act_scaling_factor)
        x, act_scaling_factor = self.blocks((x,act_scaling_factor))
        # print("q aft blocks: ", x.shape, x.data.min(),x.data.max(),act_scaling_factor)
        # x, act_scaling_factor = self.pooling((x,act_scaling_factor))
        x = self.pooling(x)
        # print("q aft pooling: ", x.shape, x.data.min(),x.data.max(),act_scaling_factor)
        x, act_scaling_factor = self.qct_pool(x)
        # print(self.qct_pool.__repr__(),act_scaling_factor)
        # print("ENDING EPOCH\n")
        # print("q aft Quant_pool: ", x.shape, x.data.min(),x.data.max(),act_scaling_factor)
        if self.showmm==True:
            for i in range(self.num_blocks):  
                print("block "+str(i))      
                self.blocks[i].ptminmax(0)
        return x, act_scaling_factor
    
    def forward(self, x):
        x, act_scaling_factor = self.forward_features(x)
        x, _ = self.head(x, act_scaling_factor)
        # print("q aft head: ", x.shape , x.data.min(), x.data.max() )

        return x

def _create_qconvMixer(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(q_ConvMixer, variant, pretrained, default_cfg=default_cfgs[variant], **kwargs)

@register_model
def q_convMixer_768_32(pretrained=False,model=None, **kwargs):

    if model==None:
        model=timm.create_model('convmixer_768_32',pretrained=True)
    model_args = dict(dim=768, depth=32, kernel_size=7, patch_size=7, block_layer=q_convMixerBlock, activation=nn.ReLU, model=model, **kwargs)
    model = _create_qconvMixer('convmixer_768_32', pretrained=False, **model_args)

    bit_config=bit_config_dict["q_convMixer_768_uniform8"]
    for name, m in model.named_modules():
        if name in bit_config.keys():
            setattr(m, 'quant_mode', 'symmetric')
            setattr(m, 'bias_bit', 32)
            setattr(m, 'quantize_bias', True)
            setattr(m, 'per_channel', True)
            setattr(m, 'act_percentile', 0)
            setattr(m, 'act_range_momentum', 0.99)
            setattr(m, 'weight_percentile', 0)
            setattr(m, 'fix_flag', False)
            setattr(m, 'fix_BN', True)
            setattr(m, 'fix_BN_threshold', None)
            setattr(m, 'training_BN_mode', True)
            setattr(m, 'checkpoint_iter_threshold', -1)
            setattr(m, 'fixed_point_quantization', False)
            # setattr(m, 'full_precision_flag', True)

            if type(bit_config[name]) is tuple:
                bitwidth = bit_config[name][0]
            else:
                bitwidth = bit_config[name]

            if hasattr(m, 'activation_bit'):
                bitwidth = bitwidth
                setattr(m, 'activation_bit', bitwidth)
                setattr(m, 'quant_mode', 'asymmetric')
                # setattr(m, 'quant_mode', 'asymmetric')
                # if "qact_23" in name:
                # # setattr(m, 'act_percentile', 99.9)
                #     setattr(m, 'quant_mode', 'asymmetric')
                #     print(name,m)
                    # bitwidth = bitwidth * 2
                #     setattr(m, 'act_range_momentum', 0)
                # setattr(m, 'full_precision_flag', True)
                # if bitwidth == 4:
                #     setattr(m, 'quant_mode', 'asymmetric')
            else:
                # bitwidth = bitwidth * 2
                bitwidth = bitwidth / 2
                setattr(m, 'weight_bit', bitwidth)
            # print(name,bitwidth)
    return model

@register_model
def q_convMixer_1536_20(pretrained=False,model=None, **kwargs):

    if model==None:
        model=timm.create_model('convmixer_1536_20',pretrained=True)
    model_args =  dict(dim=1536, depth=20, kernel_size=9, patch_size=7, block_layer=q_convMixerBlock, model=model, **kwargs)
    model = _create_qconvMixer('convmixer_1536_20', pretrained=False, **model_args)

    bit_config=bit_config_dict["q_convMixer_1536_uniform8"]
    for name, m in model.named_modules():
        if name in bit_config.keys():
            setattr(m, 'quant_mode', 'symmetric')
            setattr(m, 'bias_bit', 32)
            setattr(m, 'quantize_bias', True)
            setattr(m, 'per_channel', True)
            setattr(m, 'act_percentile', 0)
            setattr(m, 'act_range_momentum', 0.99)
            setattr(m, 'weight_percentile', 0)
            setattr(m, 'fix_flag', False)
            setattr(m, 'fix_BN', True)
            setattr(m, 'fix_BN_threshold', None)
            setattr(m, 'training_BN_mode', True)
            setattr(m, 'checkpoint_iter_threshold', -1)
            setattr(m, 'fixed_point_quantization', False)
            # setattr(m, 'full_precision_flag', True)

            if type(bit_config[name]) is tuple:
                bitwidth = bit_config[name][0]
            else:
                bitwidth = bit_config[name]

            if hasattr(m, 'activation_bit'):
                # bitwidth = bitwidth * 2
                setattr(m, 'activation_bit', bitwidth)
                # setattr(m, 'full_precision_flag', True)
                # if bitwidth == 4:
                #     setattr(m, 'quant_mode', 'asymmetric')
            else:
                # bitwidth = bitwidth * 2
                setattr(m, 'weight_bit', bitwidth)
            # print(name,bitwidth)
    return model

@register_model
def q_convMixer_768_32_gelu(pretrained=False,model=None, **kwargs):

    if model==None:
        model=timm.create_model('convmixer_768_32_gelu',pretrained=False)
    load_checkpoint(model,"./checkpoint-105.pth.tar",use_ema=True)
    model_args = dict(dim=768, depth=32, kernel_size=7, patch_size=7, block_layer=q_convMixerBlock, activation=nn.GELU, model=model, **kwargs)
    model = _create_qconvMixer('convmixer_768_32', pretrained=False, **model_args)

    bit_config=bit_config_dict["q_convMixer_768_uniform8"]
    for name, m in model.named_modules():
        if name in bit_config.keys():
            setattr(m, 'quant_mode', 'symmetric')
            setattr(m, 'bias_bit', 32)
            setattr(m, 'quantize_bias', True)
            setattr(m, 'per_channel', True)
            setattr(m, 'act_percentile', 0)
            setattr(m, 'act_range_momentum', 0.99)
            setattr(m, 'weight_percentile', 0)
            setattr(m, 'fix_flag', False)
            setattr(m, 'fix_BN', True)
            setattr(m, 'fix_BN_threshold', None)
            setattr(m, 'training_BN_mode', True)
            setattr(m, 'checkpoint_iter_threshold', -1)
            setattr(m, 'fixed_point_quantization', False)
            # setattr(m, 'full_precision_flag', True)

            if type(bit_config[name]) is tuple:
                bitwidth = bit_config[name][0]
            else:
                bitwidth = bit_config[name]

            if hasattr(m, 'activation_bit'):
                bitwidth = bitwidth
                setattr(m, 'activation_bit', bitwidth)
                # setattr(m, 'full_precision_flag', True)
                # if bitwidth == 4:
                #     setattr(m, 'quant_mode', 'asymmetric')
            else:
                # bitwidth = bitwidth * 2
                bitwidth = bitwidth
                setattr(m, 'weight_bit', bitwidth)
            # print(name,bitwidth)
    return model

@register_model
def q_convMixer_768_32_new(pretrained=False,model=None, **kwargs):

    if model==None:
        model=timm.create_model('convmixer_bn_768_32',pretrained=False)
    load_checkpoint(model,"./checkpoint97.pth.tar",use_ema=True)

    model_args = dict(dim=768, depth=32, kernel_size=7, patch_size=7, stem_layer=q_ExtractPatch_new ,block_layer=q_convMixerBlock_new, activation=nn.GELU, model=model, **kwargs)
    model = _create_qconvMixer('convmixer_768_32', pretrained=False, **model_args)

    bit_config=bit_config_dict["q_convMixer_768_uniform8"]
    for name, m in model.named_modules():
        # print(name)
        if name in bit_config.keys():
            setattr(m, 'quant_mode', 'symmetric')
            setattr(m, 'bias_bit', 32)
            setattr(m, 'quantize_bias', True)
            setattr(m, 'per_channel', True)
            setattr(m, 'act_percentile', 0)
            setattr(m, 'act_range_momentum', 0.99)
            setattr(m, 'weight_percentile', 0)
            setattr(m, 'fix_flag', False)
            setattr(m, 'fix_BN', True)
            setattr(m, 'fix_BN_threshold', None)
            setattr(m, 'training_BN_mode', True)
            setattr(m, 'checkpoint_iter_threshold', -1)
            setattr(m, 'fixed_point_quantization', False)
            # setattr(m, 'full_precision_flag', True)

            if type(bit_config[name]) is tuple:
                bitwidth = bit_config[name][0]
            else:
                bitwidth = bit_config[name]

            if hasattr(m, 'activation_bit'):
                # bitwidth = bitwidth * 2
                if "qact_23" in name:
                    print(name,m)
                    bitwidth = bitwidth * 2
                # if ("blocks.0" in name) or ("blocks.1" in name) or ("blocks.2" in name):
                #     print(name,m)
                #     bitwidth = bitwidth * 2
                #     if "blocks.0" in name:
                #         setattr(m, 'act_range_momentum', 0.0)
                setattr(m, 'activation_bit', bitwidth)
                # setattr(m, 'full_precision_flag', True)
                # if bitwidth == 4:
                #     setattr(m, 'quant_mode', 'asymmetric')
            else:
                # bitwidth = bitwidth/2
                # bitwidth = bitwidth * 2
                setattr(m, 'weight_bit', bitwidth)
                # setattr(m, 'full_precision_flag', True)
            # print(name,bitwidth)
    return model

@register_model
def q_convMixer_768_32_pact(pretrained=False,model=None, **kwargs):
    if model==None:
        model=timm.create_model('convmixer_768_32_pact',pretrained=False)
    load_checkpoint(model,"./pactconv_159.pth.tar",use_ema=True)
    model_args = dict(dim=768, depth=32, kernel_size=7, patch_size=7, block_layer=q_convMixerBlock, activation=partial(PactReLU, upper_bound=200.0), model=model, **kwargs)
    model = _create_qconvMixer('convmixer_768_32', pretrained=False, **model_args)

    bit_config=bit_config_dict["q_convMixer_768_uniform8"]
    for name, m in model.named_modules():
        if name in bit_config.keys():
            setattr(m, 'quant_mode', 'symmetric')
            setattr(m, 'bias_bit', 32)
            setattr(m, 'quantize_bias', True)
            setattr(m, 'per_channel', True)
            setattr(m, 'act_percentile', 0)
            setattr(m, 'act_range_momentum', 0.99)
            setattr(m, 'weight_percentile', 0)
            setattr(m, 'fix_flag', False)
            setattr(m, 'fix_BN', True)
            setattr(m, 'fix_BN_threshold', None)
            setattr(m, 'training_BN_mode', True)
            setattr(m, 'checkpoint_iter_threshold', -1)
            setattr(m, 'fixed_point_quantization', False)
            # setattr(m, 'full_precision_flag', True)

            if type(bit_config[name]) is tuple:
                bitwidth = bit_config[name][0]
            else:
                bitwidth = bit_config[name]

            if hasattr(m, 'activation_bit'):
                bitwidth = bitwidth
                setattr(m, 'activation_bit', bitwidth)
                setattr(m, 'quant_mode', 'asymmetric')
                # setattr(m, 'act_percentile', 99.9)
                # if "qact_23" in name:
                #     setattr(m, 'act_percentile', 99.9)
                #     setattr(m, 'quant_mode', 'asymmetric')
                # setattr(m, 'full_precision_flag', True)
                # if bitwidth == 4:
                #     setattr(m, 'quant_mode', 'asymmetric')
            else:
                bitwidth = 4
                # bitwidth = bitwidth * 2
                setattr(m, 'weight_bit', bitwidth)
            # print(name,bitwidth)
    return model