import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model
from .helpers import build_model_with_cfg, load_checkpoint
from .utils.pg_utils import PactReLU
from functools import partial


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


class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, in_chans=3, num_classes=1000, activation=nn.GELU, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = dim
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size),
            activation(),
            nn.BatchNorm2d(dim)
        )
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        activation(),
                        nn.BatchNorm2d(dim)
                    )),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    activation(),
                    nn.BatchNorm2d(dim)
            ) for i in range(depth)]
        )
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
          
    def forward_features(self, x):
        # print("cm bef stem: ", x.shape , x.data.min(), x.data.max())
        x = self.stem(x)
        # print("cm aft stem: ", x.shape , x.data.min(), x.data.max())
        x = self.blocks(x)
        # print("cm aft blocks: ", x.shape, x.data.min(),x.data.max())
        x = self.pooling(x)
        # print("cm aft pooling: ", x.shape, x.data.min(),x.data.max())
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        # print("cm after pool: ", x.shape)
        x = self.head(x)
        # print("cm aft head: ", x.shape , x.data.min(), x.data.max() )

        return x

class ConvMixer_Bn(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, in_chans=3, num_classes=1000, activation=nn.GELU, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = dim
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(dim),
            activation()
        )
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        nn.BatchNorm2d(dim),
                        activation()
                    )),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.BatchNorm2d(dim),
                    activation()
            ) for i in range(depth)]
        )
        self.pooling = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
          
    def forward_features(self, x):
        # print("cm bef stem: ", x.shape , x.data.min(), x.data.max())
        x = self.stem(x)
        # print("cm aft stem: ", x.shape , x.data.min(), x.data.max())
        x = self.blocks(x)
        # print("cm aft blocks: ", x.shape, x.data.min(),x.data.max())
        x = self.pooling(x)
        # print("cm aft pooling: ", x.shape, x.data.min(),x.data.max())
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        # print("cm after pool: ", x.shape)
        x = self.head(x)
        # print("cm aft head: ", x.shape , x.data.min(), x.data.max() )

        return x

def _create_convmixer(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ConvMixer, variant, pretrained, default_cfg=default_cfgs[variant], **kwargs)

def _create_convmixer_bn(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ConvMixer_Bn, variant, pretrained, default_cfg=default_cfgs[variant], **kwargs)

@register_model
def convmixer_1536_20(pretrained=False, **kwargs):
    model_args = dict(dim=1536, depth=20, kernel_size=9, patch_size=7, **kwargs)
    return _create_convmixer('convmixer_1536_20', pretrained, **model_args)

@register_model
def convmixer_bn_1536_20(pretrained=False, **kwargs):
    model_args = dict(dim=1536, depth=20, kernel_size=9, patch_size=7, **kwargs)

    model = _create_convmixer_bn('convmixer_1536_20', pretrained, **model_args)
    if pretrained:
        load_checkpoint(model,"./relu_bn.pth.tar",use_ema=True)
    return model

@register_model
def convmixer_768_32(pretrained=False, **kwargs):
    model_args = dict(dim=768, depth=32, kernel_size=7, patch_size=7, activation=nn.ReLU, **kwargs)
    return _create_convmixer('convmixer_768_32', pretrained, **model_args)

@register_model
def convmixer_768_32_gelu(pretrained=False, **kwargs):
    model_args = dict(dim=768, depth=32, kernel_size=7, patch_size=7, activation=nn.GELU, **kwargs)
    model = _create_convmixer('convmixer_768_32', False, **model_args)
    if pretrained:
        load_checkpoint(model,"./checkpoint-105.pth.tar",use_ema=True)
    return model

@register_model
def convmixer_bn_768_32(pretrained=False, **kwargs):
    model_args = dict(dim=768, depth=32, kernel_size=7, patch_size=7, **kwargs)

    model = _create_convmixer_bn('convmixer_768_32', False, **model_args)
    if pretrained:
        load_checkpoint(model,"./checkpoint97.pth.tar",use_ema=True)
    return model

@register_model
def convmixer_768_32_pact(pretrained=False, **kwargs):
    model_args = dict(dim=768, depth=32, kernel_size=7, patch_size=7, activation=partial(PactReLU, upper_bound=200.0), pretrained_strict=False, **kwargs)
    model = _create_convmixer('convmixer_768_32', True, **model_args)
    if pretrained:
        load_checkpoint(model,"./pactconv_159.pth.tar",use_ema=True)
    return model

@register_model
def convmixer_1024_20_ks9_p14(pretrained=False, **kwargs):
    model_args = dict(dim=1024, depth=20, kernel_size=9, patch_size=14, **kwargs)
    return _create_convmixer('convmixer_1024_20_ks9_p14', pretrained, **model_args)
