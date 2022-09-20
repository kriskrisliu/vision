"""
    Quantized MlpMixer for ImageNet-1K, implemented in PyTorch.
"""

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
from .mlp_mixer import _cfg, default_cfgs, Affine
from .layers import PatchEmbed, Mlp, GluMlp, GatedMlp, DropPath, lecun_normal_, to_2tuple
from .quantization_utils.quant_modules import QuantLinear,QuantAct,QuantConv2d,QuantAffine
from .bit_config import bit_config_dict

class q_PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = QuantConv2d()
        self.proj.set_param(nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size))

        self.quant_act_conv=QuantAct()
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def set_param(self, unit):
        self.proj.set_param(unit.proj)


    def forward(self, x, pre_act_scaling_factor=None):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x, conv_scaling_factor  = self.proj(x,pre_act_scaling_factor)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        x, act_scaling_factor = self.quant_act_conv(x,pre_act_scaling_factor,conv_scaling_factor)
        x = self.norm(x)

        return x

class q_Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features

        self.fc1 = QuantLinear()
        self.fc1.set_param(nn.Linear(in_features, hidden_features))
        self.quant_fc = QuantAct()
        self.act = act_layer()
        self.quant_act = QuantAct()
        self.fc2 = QuantLinear()
        self.fc2.set_param(nn.Linear(hidden_features, in_features))

        self.use_noise = False
        self.register_buffer('noise_hidden_feat', 2*torch.rand(size=(hidden_features,))-1)
        self.register_buffer('noiseScale',torch.tensor([0.05]))
        self.static_num = 0.
    def __repr__(self):
        s = super(q_Mlp, self).__repr__()
        s = "(" + s + f" use_noise={self.use_noise}, noiseScale={self.noiseScale}, static_num={self.static_num})"
        return s

    def set_param(self, unit):
        unit.fc1.in_features = self.out_features
        unit.fc1.out_features = self.hidden_features

        unit.fc2.in_features = self.hidden_features
        unit.fc2.out_features = self.out_features

        self.fc1.set_param(unit.fc1)
        self.fc2.set_param(unit.fc2)

    def forward(self, x,prev_act_scaling_factor,noise=None):
        # import pdb; pdb.set_trace()
        x, fc_scaling_factor = self.fc1(x, prev_act_scaling_factor,noise=noise)
        x, act_scaling_factor = self.quant_fc(x,prev_act_scaling_factor,fc_scaling_factor)
        x = self.act(x)

        if not self.use_noise:
            self.noiseScale = torch.tensor([0.]).type_as(self.noiseScale)
        x = x+(self.noise_hidden_feat.view(1,1,-1)*self.noiseScale+self.static_num)
        x, act_scaling_factor = self.quant_act(x)
        x, fc_scaling_factor = self.fc2(x, act_scaling_factor,noise=(self.noise_hidden_feat*self.noiseScale+self.static_num))

        return x, act_scaling_factor, fc_scaling_factor

class q_MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=(0.5, 4.0), mlp_layer=q_Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.norm1 = norm_layer(dim)
        self.quant_norm1 = QuantAct()
        self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.quant_norm2 = QuantAct()
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

        self.quant_act_int32_tokens = QuantAct()
        self.quant_act_int32_channels = QuantAct()

    def set_param(self, unit):
        self.norm1 = unit.norm1
        self.norm2 = unit.norm2
        self.mlp_tokens.set_param(unit.mlp_tokens)
        self.mlp_channels.set_param(unit.mlp_channels)

    def forward(self, tup):
        x, scaling_factor_int32 = tup
        identity=x
        x= self.norm1(x)
        x, act_scaling_factor = self.quant_norm1(x)
        x, act_scaling_factor, fc_scaling_factor= self.mlp_tokens(x.transpose(1, 2),act_scaling_factor)
        x=x.transpose(1, 2)
        x = x + identity
        x, scaling_factor_int32 = self.quant_act_int32_tokens(x, act_scaling_factor, fc_scaling_factor, identity, scaling_factor_int32, None)

        identity=x
        x = self.norm2(x)
        x, act_scaling_factor = self.quant_norm2(x)
        x, act_scaling_factor, fc_scaling_factor= self.mlp_channels(x,act_scaling_factor)
        x = x + identity
        x, scaling_factor_int32 = self.quant_act_int32_channels(x, act_scaling_factor, fc_scaling_factor, identity, scaling_factor_int32, None)
        return (x, scaling_factor_int32)

    def ptminmax(self,mode):
        if mode==0:
            self.quant_norm2.ptminmax()

class qmulti_MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=(0.5, 4.0), mlp_layer=q_Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.,tknum=4,cnnum=1):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.token_num,self.channel_num=tknum,cnnum
        self.norm1 = norm_layer(dim)
        self.quant_norm1 = torch.nn.ModuleList([QuantAct() for i in range(tknum)])
        self.mlp_tokens = torch.nn.ModuleList([mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop) for i in range(tknum)])
        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.quant_norm2 = torch.nn.ModuleList([QuantAct() for i in range(cnnum)])
        self.mlp_channels = torch.nn.ModuleList([mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop) for i in range(cnnum)])

        self.quant_act_int32_tokens = QuantAct()
        self.quant_act_int32_channels = QuantAct()

    def set_param(self, unit):
        self.norm1 = unit.norm1
        self.norm2 = unit.norm2
        for i in range(self.token_num):
            self.mlp_tokens[i].set_param(unit.mlp_tokens[i])
        for i in range(self.channel_num):
            self.mlp_channels[i].set_param(unit.mlp_channels[i])

    def forward(self, tup):
        x, scaling_factor_int32 = tup
        identity=x
        x= self.norm1(x)
        x=x.transpose(1, 2)
        y= torch.split(x,int(x.shape[1]/self.token_num),dim=1)
        y=list(y)
        for i in range (self.token_num):
            y[i], act_scaling_factor = self.quant_norm1[i](y[i])
            y[i], act_scaling_factor, fc_scaling_factor= self.mlp_tokens[i](y[i],act_scaling_factor)
        y=tuple(y)
        x=torch.cat(y,dim=1)
        x=x.transpose(1, 2)
        x = x + identity
        x, scaling_factor_int32 = self.quant_act_int32_tokens(x)

        identity=x
        x = self.norm2(x)
        y= torch.split(x,int(x.shape[1]/self.channel_num),dim=1)
        y=list(y)
        for i in range(self.channel_num):
            y[i], act_scaling_factor = self.quant_norm2[i](y[i])
            y[i], act_scaling_factor, fc_scaling_factor = self.mlp_channels[i](y[i],act_scaling_factor)
        y=tuple(y)
        x=torch.cat(y,dim=1)
        x = x + identity
        x, scaling_factor_int32 = self.quant_act_int32_channels(x)
        return (x, scaling_factor_int32)


class q_ResBlock(nn.Module):
    """ Residual MLP block w/ LayerScale and Affine 'norm'

    Based on: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=4, mlp_layer=Mlp, norm_layer=QuantAffine,
            act_layer=nn.GELU, init_values=1e-4, drop=0., drop_path=0.):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.quant_norm1 = QuantAct()
        self.linear_tokens = QuantLinear()
        self.linear_tokens.set_param(nn.Linear(seq_len, seq_len))
        self.norm2 = norm_layer(dim)
        self.quant_norm2 = QuantAct()
        self.mlp_channels = mlp_layer(dim, channel_dim, act_layer=act_layer, drop=drop)
        self.ls1 = QuantAffine(dim)
        self.ls2 = QuantAffine(dim)
        self.quant_mlp1 = QuantAct()
        self.quant_mlp2 = QuantAct()

        self.quant_act_int32_tokens = QuantAct()
        self.quant_act_int32_channels = QuantAct()

        if norm_layer==QuantAffine:
            self.setnorm=True
        else:
            self.setnorm=False

        self.minmax=True
        self.actr=[]
        self.actr_aft=[]
        self.use_noise_token=False
        self.use_noise_channel=False
        self.register_buffer('noise_token', 2*torch.rand(size=(seq_len,))-1)
        self.register_buffer('noiseScale_token',torch.tensor([0.05]))
        self.register_buffer('noise_channel', 2*torch.rand(size=(dim,))-1)
        self.register_buffer('noiseScale_channel',torch.tensor([0.05]))
        self.static_num_token = 0.
        self.static_num_channel = 0.

    def __repr__(self):
        s = super(q_ResBlock, self).__repr__()
        s = "(" + s + f" use_noise_token={self.use_noise_token}, noiseScale_token={self.noiseScale_token}" +\
            f"use_noise_channel={self.use_noise_channel}, noiseScale_channel={self.noiseScale_channel}, " +\
            f"static_num_token={self.static_num_token}, static_num_channel={self.static_num_channel})"
        return s

    def set_param(self, unit):
        if self.setnorm:
            self.norm1.set_param(unit.norm1)
            self.norm2.set_param(unit.norm2)
        else:
            self.norm1=unit.norm1
            self.norm2=unit.norm2

        self.linear_tokens.set_param(unit.linear_tokens)
        self.mlp_channels.set_param(unit.mlp_channels)
        self.ls1.set_ls(unit.ls1)
        self.ls2.set_ls(unit.ls2)

    def forward(self, tup):
        x, scaling_factor_int32 = tup
        identity=x
        if self.setnorm:
            x, norm_scaling_factor= self.norm1(x,scaling_factor_int32)
            x, act_scaling_factor = self.quant_norm1(x,scaling_factor_int32,norm_scaling_factor)
        else:
            x= self.norm1(x)
            if not self.use_noise_token:
                self.noiseScale_token = torch.tensor([0.]).type_as(self.noiseScale_token)
            noise = self.noise_token*self.noiseScale_token+self.static_num_token
            x = x+noise.view(1,-1,1)
            # noise=None
            x, act_scaling_factor = self.quant_norm1(x)

        x, fc_scaling_factor= self.linear_tokens(x.transpose(1, 2),act_scaling_factor,noise=noise)
        x=x.transpose(1, 2)
        x, norm_scaling_factor= self.ls1(x,act_scaling_factor)
        x = x + identity
        x, scaling_factor_int32 = self.quant_act_int32_tokens(x)

        identity=x
        if self.setnorm:
            raise NotImplementedError
            x, norm_scaling_factor= self.norm2(x,scaling_factor_int32)
            x, act_scaling_factor = self.quant_norm2(x,scaling_factor_int32,norm_scaling_factor)
        else:
            x= self.norm2(x)
            if not self.use_noise_channel:
                self.noiseScale_channel = torch.tensor([0.]).type_as(self.noiseScale_channel)
            noise = self.noise_channel*self.noiseScale_channel+self.static_num_channel
            x = x+noise.view(1,1,-1)
            x, act_scaling_factor = self.quant_norm2(x)
        x, act_scaling_factor, fc_scaling_factor= self.mlp_channels(x,act_scaling_factor,noise)
        x, norm_scaling_factor= self.ls2(x,act_scaling_factor)
        x = x + identity
        x, scaling_factor_int32 = self.quant_act_int32_channels(x)
        return (x, scaling_factor_int32)

    def ptminmax(self,mode):
        if mode==0:
            self.quant_norm1.ptminmax()

class qmulti_ResBlock(nn.Module):
    """ Residual MLP block w/ LayerScale and Affine 'norm'

    Based on: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=4, mlp_layer=Mlp, norm_layer=QuantAffine,
            act_layer=nn.GELU, init_values=1e-4, drop=0., drop_path=0.,tknum=4):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.token_num=tknum
        self.quant_norm1 = torch.nn.ModuleList([QuantAct() for i in range(tknum)])
        self.linear_tokens = torch.nn.ModuleList([QuantLinear() for i in range(tknum)])
        for i in range(tknum):
            self.linear_tokens[i].set_param(nn.Linear(seq_len, seq_len))
        self.norm2 = norm_layer(dim)
        self.quant_norm2 = QuantAct()
        self.mlp_channels = mlp_layer(dim, channel_dim, act_layer=act_layer, drop=drop)
        self.ls1 = QuantAffine(dim)
        self.ls2 = QuantAffine(dim)
        self.quant_mlp1 = QuantAct()
        self.quant_mlp2 = QuantAct()

        self.quant_act_int32_tokens = QuantAct()
        self.quant_act_int32_channels = QuantAct()

        if norm_layer==QuantAffine:
            self.setnorm=True
        else:
            self.setnorm=False

    def set_param(self, unit):
        if self.setnorm:
            self.norm1.set_param(unit.norm1)
            self.norm2.set_param(unit.norm2)
        else:
            self.norm1=unit.norm1
            self.norm2=unit.norm2
        for i in range(self.token_num):
            self.linear_tokens[i].set_param(unit.linear_tokens[i])
        self.mlp_channels.set_param(unit.mlp_channels)
        self.ls1.set_ls(unit.ls1)
        self.ls2.set_ls(unit.ls2)

    def forward(self, tup):
        x, scaling_factor_int32 = tup
        identity=x
        if self.setnorm:
            x, norm_scaling_factor= self.norm1(x,scaling_factor_int32)
            x, act_scaling_factor = self.quant_norm1(x,scaling_factor_int32,norm_scaling_factor)
        else:
            x= self.norm1(x).transpose(1, 2)
            y= torch.split(x,int(x.shape[1]/self.token_num),dim=1)
            y=list(y)
            for i in range(self.token_num):
                y[i], act_scaling_factor = self.quant_norm1[i](y[i])
                y[i], fc_scaling_factor= self.linear_tokens[i](y[i],act_scaling_factor)

            y=tuple(y)
            x=torch.cat(y,dim=1)

        x=x.transpose(1, 2)
        x, act_scaling_factor = self.quant_mlp1(x)
        x, norm_scaling_factor= self.ls1(x,act_scaling_factor)
        x = x + identity
        x, scaling_factor_int32 = self.quant_act_int32_tokens(x, act_scaling_factor, norm_scaling_factor, identity, scaling_factor_int32, None)

        identity=x
        if self.setnorm:
            x, norm_scaling_factor= self.norm2(x,scaling_factor_int32)
            x, act_scaling_factor = self.quant_norm2(x,scaling_factor_int32,norm_scaling_factor)
        else:
            x= self.norm2(x)
            x, act_scaling_factor = self.quant_norm2(x)
        x, act_scaling_factor, fc_scaling_factor= self.mlp_channels(x,act_scaling_factor)
        x, act_scaling_factor = self.quant_mlp2(x,act_scaling_factor,fc_scaling_factor)
        x, norm_scaling_factor= self.ls2(x,act_scaling_factor)
        x = x + identity
        x, scaling_factor_int32 = self.quant_act_int32_channels(x, act_scaling_factor, norm_scaling_factor, identity, scaling_factor_int32, None)
        return (x, scaling_factor_int32)

class q_MlpMixer(nn.Module):
    def __init__(
            self,
            model=None,
            num_classes=1000,
            img_size=224,
            in_chans=3,
            patch_size=16,
            num_blocks=8,
            embed_dim=512,
            mlp_ratio=(0.5, 4.0),
            block_layer=q_MixerBlock,
            mlp_layer=q_Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop_rate=0.,
            drop_path_rate=0.,
            nlhb=False,
            stem_norm=False,
            showmm=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.showmm=showmm
        self.num_blocks=num_blocks

        self.quant_input = QuantAct()
        self.stem = q_PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, norm_layer=norm_layer if stem_norm else None)
        self.quant_act_int32 = QuantAct()

        self.blocks = nn.Sequential(*[
            block_layer(
                embed_dim, self.stem.num_patches, mlp_ratio, mlp_layer=mlp_layer, norm_layer=norm_layer,
                act_layer=act_layer, drop=drop_rate, drop_path=drop_path_rate)
            for _ in range(num_blocks)])

        self.norm = norm_layer(embed_dim)
        self.quant_norm = QuantAct()

        if model is not None:
            blocks=getattr(model, "blocks")
            head=getattr(model, "head")
            norm=getattr(model, "norm")
            stem=getattr(model, "stem")
            self.stem.set_param(stem)
            self.norm=norm
            for i in range(num_blocks):
                stage = getattr(blocks, "{}".format(i))
                self.blocks[i].set_param(stage)

        if num_classes > 0 :
            self.head = QuantLinear()
            if model is not None:
                self.head.set_param(head)
            else:
                self.head.set_param(nn.Linear(embed_dim, num_classes))
        else :
            self.head = nn.Identity()

    def forward_features(self, x):
        x, act_scaling_factor = self.quant_input(x)
        x = self.stem(x,act_scaling_factor)
        x, act_scaling_factor = self.quant_act_int32(x)
        x, act_scaling_factor = self.blocks((x,act_scaling_factor))

        x = self.norm(x)
        x, act_scaling_factor = self.quant_norm(x)
        x = x.mean(dim=1)
        if self.showmm==True:
            for i in range(self.num_blocks):
                print("block "+str(i))
                self.blocks[i].ptminmax(0)

        return x,act_scaling_factor

    def forward(self, x):
        x, act_scaling_factor = self.forward_features(x)
        x, _ = self.head(x, act_scaling_factor)
        return x

def checkpoint_filter_fn(state_dict, model):
    """ Remap checkpoints if needed """
    if 'patch_embed.proj.weight' in state_dict:
        # Remap FB ResMlp models -> timm
        out_dict = {}
        for k, v in state_dict.items():
            k = k.replace('patch_embed.', 'stem.')
            k = k.replace('attn.', 'linear_tokens.')
            k = k.replace('mlp.', 'mlp_channels.')
            k = k.replace('gamma_', 'ls')
            if k.endswith('.alpha') or k.endswith('.beta'):
                v = v.reshape(1, 1, -1)
            out_dict[k] = v
        return out_dict
    return state_dict

def _create_qmlp(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for MLP-Mixer models.')

    model = build_model_with_cfg(
        q_MlpMixer, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)
    return model

@register_model
def q_mlpfirst(pretrained=False,model=None, **kwargs):
    """ Mixer-S/32 224x224
    Paper: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    if model==None:
        model=timm.create_model('mixer_b16_224',pretrained=True)
    model_args = dict(model=model,patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    # print(model_args['showmm'])
    model = _create_qmlp('mixer_b16_224', pretrained=False, **model_args)

    bit_config=bit_config_dict["q_mlp_uniform8"]
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
                setattr(m, 'activation_bit', bitwidth)
                setattr(m, 'quant_mode', 'asymmetric')
            else:
                bitwidth = bitwidth
                setattr(m, 'weight_bit', bitwidth)
                # setattr(m, 'full_precision_flag', True)
            # print(name,bitwidth)

    return model

def Issubblock(str):
    if str.find("quant_norm1") >= 0:
        return True
    if str.find("mlp_tokens") >= 0:
        return True
    if str.find("quant_norm2") >= 0:
        return True
    if str.find("mlp_channels") >= 0:
        return True
    return False
def switch_key(str):
    x= str.find("quant_norm")
    if x >= 0:
        # print(str[0:x+11]+str[x+13:len(str)])
        return str[0:x+11]+str[x+13:len(str)]
    x= str.find("mlp_tokens")
    if  x >= 0:
        # print(str[0:x+10]+str[x+12:len(str)])
        return str[0:x+10]+str[x+12:len(str)]
    x= str.find("mlp_channels")
    if  x>= 0:
        # print(str[0:x+12]+str[x+14:len(str)])
        return str[0:x+12]+str[x+14:len(str)]
    return None
@register_model
def q_mlpmulti(pretrained=False,model=None, **kwargs):
    """ Mixer-S/32 224x224
    Paper: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    if model==None:
        model=timm.create_model('mixer_b16_224_multi',pretrained=False)

    load_checkpoint(model,"./multi_best.pth.tar",use_ema=True)
    model_args = dict(model=model,patch_size=16, num_blocks=12, embed_dim=768,showmm=False,block_layer=qmulti_MixerBlock, **kwargs)
    model = _create_qmlp('mixer_b16_224', pretrained=False, **model_args)

    bit_config=bit_config_dict["q_mlp_uniform8"]
    # print("1",model.showmm)
    for name, m in model.named_modules():
        # print("name",name)
        if Issubblock(name) and name not in bit_config.keys():
            keyname=switch_key(name)
        else:
            keyname=name

        if keyname in bit_config.keys():
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

            if type(bit_config[keyname]) is tuple:
                bitwidth = bit_config[keyname][0]
            else:
                bitwidth = bit_config[keyname]

            if hasattr(m, 'activation_bit'):
                setattr(m, 'activation_bit', bitwidth)
                setattr(m, 'quant_mode', 'asymmetric')
            else:
                setattr(m, 'weight_bit', bitwidth)

    return model

@register_model
def q_resmlp(pretrained=False,model=None, **kwargs):
    """ ResMLP-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    if model==None:
        model=timm.create_model('resmlp_24_224',pretrained=True)
    model_args = dict(
        model= model,patch_size=16, num_blocks=24, embed_dim=384, mlp_ratio=4,
        block_layer=partial(q_ResBlock, init_values=1e-5), norm_layer=Affine, **kwargs)
    model = _create_qmlp('resmlp_24_224', pretrained=False, **model_args)

    bit_config=bit_config_dict["q_resmlp_a6w8"]
    # bit_config=bit_config_dict["q_resmlp_uniform16"]
    print(bit_config)
    for name, m in model.named_modules():
        setattr(m, 'ownname', name)
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
                setattr(m, 'activation_bit', bitwidth)
                setattr(m, 'quant_mode', 'asymmetric')
            else:
                setattr(m, 'weight_bit', bitwidth)
    return model

@register_model
def q_resmlp_relu(pretrained=False,model=None, **kwargs):
    """ ResMLP-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    if model==None:
        model=timm.create_model('resmlp_24_224_relu',pretrained=False)

    load_checkpoint(model,"./reluwd.pth.tar",use_ema=True)

    model_args = dict(
        model= model,patch_size=16, num_blocks=24, embed_dim=384, mlp_ratio=4,
        block_layer=partial(q_ResBlock, init_values=1e-5), norm_layer=QuantAffine,act_layer=nn.ReLU, showmm=False, **kwargs)
    model = _create_qmlp('resmlp_24_224', pretrained=pretrained, **model_args)

    bit_config=bit_config_dict["q_resmlp_uniform8"]
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
                setattr(m, 'activation_bit', bitwidth)

            else:
                setattr(m, 'weight_bit', bitwidth)
    return model


@register_model
def q_resmlp_bn_relu(pretrained=False,model=None, **kwargs):
    """ ResMLP-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    if model==None:
        model=timm.create_model('resmlp_24_224_bn_relu',pretrained=False)

    load_checkpoint(model,"./relu_bn.pth.tar",use_ema=True)

    model_args = dict(
        model= model,patch_size=16, num_blocks=24, embed_dim=384, mlp_ratio=4,
        block_layer=partial(q_ResBlock, init_values=1e-5), act_layer=nn.ReLU, showmm=False, **kwargs)
    model = _create_qmlp('resmlp_24_224', pretrained=pretrained, **model_args)

    bit_config=bit_config_dict["q_resmlp_uniform8"]
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

            if type(bit_config[name]) is tuple:
                bitwidth = bit_config[name][0]
            else:
                bitwidth = bit_config[name]

            if hasattr(m, 'activation_bit'):
                setattr(m, 'activation_bit', bitwidth)
            else:
                setattr(m, 'weight_bit', bitwidth)
            # print(name,bitwidth)
    return model

@register_model
def q_resmlp_bn(pretrained=False,model=None, **kwargs):
    """ ResMLP-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    if model==None:
        model=timm.create_model('resmlp_24_224_bn',pretrained=False)

    load_checkpoint(model,"./gelu_bn.pth.tar",use_ema=True)

    model_args = dict(
        model= model,patch_size=16, num_blocks=24, embed_dim=384, mlp_ratio=4,
        block_layer=partial(q_ResBlock, init_values=1e-5), **kwargs)
    model = _create_qmlp('resmlp_24_224', pretrained=False, **model_args)

    bit_config=bit_config_dict["q_resmlp_uniform8"]
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

            if type(bit_config[name]) is tuple:
                bitwidth = bit_config[name][0]
            else:
                bitwidth = bit_config[name]

            if hasattr(m, 'activation_bit'):
                setattr(m, 'activation_bit', bitwidth)
                setattr(m, 'quant_mode', 'asymmetric')
            else:
                setattr(m, 'weight_bit', bitwidth)
    return model

def Issubblock_res(str):
    if str.find("quant_norm1") >= 0:
        return True
    if str.find("linear_tokens") >= 0:
        return True
    return False
def switch_key_res(str):
    return str[0:-2]

@register_model
def qmulti_resmlp(pretrained=False,model=None, **kwargs):
    """ ResMLP-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    if model==None:
        model=timm.create_model('resmlp_24_224_multibn',pretrained=False)

    load_checkpoint(model,"./checkpoint-536.pth.tar",use_ema=True)
    model_args = dict(
        model=model,patch_size=16, num_blocks=24, embed_dim=384, mlp_ratio=4,
        block_layer=partial(qmulti_ResBlock, init_values=1e-5), **kwargs)
    model = _create_qmlp('resmlp_24_224', pretrained=False, **model_args)

    bit_config=bit_config_dict["q_resmlp_uniform8"]
    for name, m in model.named_modules():
        # print(name,type(m))
        if Issubblock_res(name) and name not in bit_config.keys():
            keyname=switch_key_res(name)
        else:
            keyname=name

        if keyname in bit_config.keys():
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

            if type(bit_config[keyname]) is tuple:
                bitwidth = bit_config[keyname][0]
            else:
                bitwidth = bit_config[keyname]

            if hasattr(m, 'activation_bit'):
                setattr(m, 'activation_bit', bitwidth)
                setattr(m, 'quant_mode', 'asymmetric')
            else:
                setattr(m, 'weight_bit', bitwidth)
    return model
