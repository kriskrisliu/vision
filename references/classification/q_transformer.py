"""
Quantize Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch import Tensor
import copy
from quant_modules import *
from pytorchcv.models.common import ConvBlock
from pytorchcv.models.shufflenetv2 import ShuffleUnit, ShuffleInitBlock
import time
import logging
import sys
from typing import Optional, List
# sys.path.append('../../../detr')
# from util.misc import NestedTensor, is_main_process

logger = logging.getLogger(__name__)

class Q_input_proj(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant_conv = QuantConv2d()
        self.quant_conv.set_param(model)
        self.quant_act_in = QuantAct()
        # print("quant_act_int activate!")
    def forward(self, x, pre_act_scaling_factor=None):
        x, act_scaling_factor = self.quant_act_in(x)
        x, weight_scaling_factor = self.quant_conv(x,act_scaling_factor)
        #x, act_scaling_factor = self.quant_act_int(x, pre_act_scaling_factor, weight_scaling_factor)
        return x #(x, act_scaling_factor)
        # return self.quant_conv(x,pre_act_scaling_factor)

class Q_query_embed(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.quant_emb = QuantAct()
    def forward(self,x):
        return self.quant_emb(x)

class Q_bbox_embed(nn.Module):
    """
    input -> act_in -> linear1 -> act1 -> linear2 -> act2 -> linear3: output
    """
    def __init__(self, module_list):
        super().__init__()
        self.quant_act_in = QuantAct()
        self.quant_linear1 = QuantLinear()
        self.quant_linear1.set_param(module_list.layers[0])
        self.quant_act1 = QuantAct()
        self.quant_linear2 = QuantLinear()
        self.quant_linear2.set_param(module_list.layers[1])
        self.quant_act2 = QuantAct()
        self.quant_linear3 = QuantLinear()
        self.quant_linear3.set_param(module_list.layers[2])
    def forward(self,hs):
        x,act_scaling_factor = self.quant_act_in(hs)
        x = self.quant_linear1(x,act_scaling_factor)
        x = F.relu(x)

        x,act_scaling_factor = self.quant_act1(x)
        x = self.quant_linear2(x,act_scaling_factor)
        x = F.relu(x)

        x,act_scaling_factor = self.quant_act2(x)
        out = self.quant_linear3(x,act_scaling_factor)
        return out

class Q_class_embed(nn.Module):
    """decoder_out:layernorm -> act -> linear"""
    def __init__(self, module):
        super().__init__()
        self.quant_in = QuantAct()
        self.quant_linear = QuantLinear()
        self.quant_linear.set_param(module)
    def forward(self, hs):
        x, act_scaling_factor = self.quant_in(hs)
        outputs_class = self.quant_linear(x, act_scaling_factor)
        return outputs_class


class QuantLayerNorm(Module):
    """
    Class to quantize given LayerNorm layer
    Parameters:
    ----------
    output_bit : int
        Bitwidth for the LayerNorm output.
    overflow_handling : bool, default True
        Whether to do overflow handling if the intermediate values are larger than 32-bit.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    force_dequant : str, default 'none'
        Force dequantize LayerNorm if either 'layernorm' or 'nonlinear' is given.
    """
    def __init__(self,
                 output_bit,
                 overflow_handling=True,
                 quant_mode='symmetric',
                 force_dequant='none'):
        super(QuantLayerNorm, self).__init__()
        self.quant_mode = quant_mode
        if force_dequant in ['nonlinear', 'layernorm']:
            logger.info("Force dequantize layernorm")
            self.quant_mode = 'none'
        self.overflow_handling = overflow_handling
        self.register_buffer('shift', torch.zeros(1))
        self.output_bit = output_bit
        self.dim_sqrt = None

        self.activation = QuantAct(output_bit, quant_mode=self.quant_mode)
        if self.quant_mode == "none":
            pass
        elif quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(quant_mode))

    def fix(self):
        self.overflow_handling = False

    def unfix(self):
        self.overflow_handling = True

    def set_param(self, ln):
        self.normalized_shape = ln.normalized_shape
        self.eps = ln.eps
        self.weight = Parameter(ln.weight.data.clone())
        self.bias = Parameter(ln.bias.data.clone())

    def set_shift(self, y_int):
        with torch.no_grad():
            y_sq_int = y_int ** 2
            var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
            shift = (torch.log2(torch.sqrt(var_int / 2**32)).ceil()).max()
            shift_old = self.shift
            self.shift = torch.max(self.shift, shift)
            logger.info("Dynamic shift adjustment: {} -> {}".format(
                int(shift_old), int(self.shift)))

    def overflow_fallback(self, y_int):
        self.set_shift(y_int)
        y_int_shifted = ste_floor.apply(y_int / 2 ** self.shift)
        y_sq_int = y_int_shifted ** 2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
        return var_int

    def forward(self, x, scaling_factor=None, exponents=None):
        if self.quant_mode == 'none':
            mean = x.mean(axis=2, keepdim=True)
            y = x - mean
            var = torch.mean(y ** 2, axis=2, keepdim=True)
            x = y / torch.sqrt(self.eps + var)
            x = x * self.weight + self.bias
            return x, None

        assert self.quant_mode == 'symmetric', \
                "unsupported quant mode: {}".format(self.quant_mode)

        if self.dim_sqrt is None:
            n = torch.tensor(x.shape[2], dtype=torch.float) # feature dim(768)
            self.dim_sqrt = torch.sqrt(n).cuda()

        # Normalization: computes mean and variance(std)
        x_int = x / scaling_factor
        mean_int = ste_round.apply(x_int.mean(axis=2, keepdim=True))
        y_int = x_int - mean_int
        y_int_shifted = ste_floor.apply(y_int / 2 ** self.shift) # avoid overflow
        y_sq_int = y_int_shifted ** 2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)

        # overflow handling in training stage
        if self.overflow_handling:
            if var_int.max() >= 2**32:
                var_int = self.overflow_fallback(y_int)
                assert var_int.max() < 2**32

        # To be replaced with integer-sqrt kernel that produces the same output
        std_int = ste_floor.apply(torch.sqrt(var_int)) * 2 ** self.shift
        factor = ste_floor.apply(2**31 / std_int)
        y_int = ste_floor.apply(y_int * factor / 2)
        scaling_factor = self.dim_sqrt / 2**30

        # scaling and shifting
        bias = self.bias.data.detach() / (self.weight.data.detach())
        bias_int = ste_floor.apply(bias / scaling_factor)

        y_int = y_int + bias_int
        scaling_factor = scaling_factor * self.weight
        x = y_int * scaling_factor

        return x, scaling_factor

class InProjector(object):
    def __init__(self,weight,bias,in_features="neveruse",out_features='neveruse'):
        self.weight = weight
        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features

# class QuantMultiheadAttention_1_12_1(nn.MultiheadAttention):
#     def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
#                  kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
#         super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn,
#                      kdim, vdim, batch_first, device, dtype)
#     def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
#                 need_weights: bool = True, attn_mask: Optional[Tensor] = None,
#                 average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
#         super().forward(query, key, value, key_padding_mask, need_weights, attn_mask, average_attn_weights)


class QuantMultiheadAttention(nn.Module):
    """
    borrow codes from HAWQ
    """
    def __init__(self,
                 weight_bit=4,
                 bias_bit=None,
                 full_precision_flag=False,
                 quant_mode="symmetric",
                 per_channel=False,
                 fix_flag=False,
                 weight_percentile=0,
                 equal_qkv=False):
        super(QuantMultiheadAttention, self).__init__()
        self.equal_qkv = equal_qkv
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.fix_flag = fix_flag
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        if self.equal_qkv:
            self.quant_in_proj = QuantLinear()
        else:
            self.in_proj_q = QuantLinear()
            self.in_proj_k = QuantLinear()
            self.in_proj_v = QuantLinear()
        self.out_proj = QuantLinear()

    def set_param(self, MHSA):
        self.embed_dim = MHSA.embed_dim
        self.num_heads = MHSA.num_heads
        self.dropout = MHSA.dropout
        self.head_dim = MHSA.head_dim

        if self.equal_qkv:
            in_proj_weight=getattr(MHSA,'in_proj_weight')
            in_proj_bias=getattr(MHSA,'in_proj_bias')
            qkvProjector = InProjector(in_proj_weight,in_proj_bias,self.embed_dim,self.embed_dim)
            self.quant_in_proj.set_param(qkvProjector)
        else:
            qProjector,kProjector,vProjector = self.assemble_qkv_projector(in_proj_weight=getattr(MHSA,'in_proj_weight'),in_proj_bias=getattr(MHSA,'in_proj_bias'))
            self.in_proj_q.set_param(qProjector)
            self.in_proj_k.set_param(kProjector)
            self.in_proj_v.set_param(vProjector)
        self.out_proj.set_param(getattr(MHSA,'out_proj'))
        # print(getattr(MHSA,'out_proj').__dict__)

    def assemble_qkv_projector(self,in_proj_weight,in_proj_bias):
        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = 0
        _end = self.embed_dim
        _w = in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        qProjector = InProjector(_w,_b,self.embed_dim,self.embed_dim)
        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = self.embed_dim
        _end = self.embed_dim * 2
        _w = in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        kProjector = InProjector(_w,_b,self.embed_dim,self.embed_dim)
        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = self.embed_dim * 2
        _end = None
        _w = in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]
        vProjector = InProjector(_w,_b,self.embed_dim,self.embed_dim)
        return (qProjector,kProjector,vProjector)
    def forward(self, query, key, value,
                key_padding_mask=None,
                need_weights=True, attn_mask=None, pre_act_scaling_factor=None):
        # if type(x) is tuple:
        #     pre_act_scaling_factor = x[1]
        #     x = x[0]
        # pre_act_scaling_factor = pre_act_scaling_factor.cuda()
        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            self.weight_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        return multi_head_attention_forward_SLIM(self,
                query, key, value, embed_dim_to_check=self.embed_dim, num_heads=self.num_heads,
                #in_proj_weight=self.in_proj_weight, in_proj_bias=self.in_proj_bias,
                bias_k=None, bias_v=None, add_zero_attn=False,
                dropout_p=self.dropout, #out_proj_weight=self.out_proj.weight, out_proj_bias=self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)

def multi_head_attention_forward_SLIM(self_arg,
                                 query,                           # type: Tensor
                                 key,                             # type: Tensor
                                 value,                           # type: Tensor
                                 embed_dim_to_check,              # type: int
                                 num_heads,                       # type: int
                                 # in_proj_weight,                  # type: Tensor
                                 # in_proj_bias,                    # type: Tensor
                                 bias_k,                          # type: Optional[Tensor]
                                 bias_v,                          # type: Optional[Tensor]
                                 add_zero_attn,                   # type: bool
                                 dropout_p,                       # type: float
                                 # out_proj_weight,                 # type: Tensor
                                 # out_proj_bias,                   # type: Tensor
                                 training=True,                   # type: bool
                                 key_padding_mask=None,           # type: Optional[Tensor]
                                 need_weights=True,               # type: bool
                                 attn_mask=None,                  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,              # type: Optional[Tensor]
                                 k_proj_weight=None,              # type: Optional[Tensor]
                                 v_proj_weight=None,              # type: Optional[Tensor]
                                 static_k=None,                   # type: Optional[Tensor]
                                 static_v=None,                   # type: Optional[Tensor]
                                 ):
    """
    pytorch source code with modification
    """

    # if not torch.jit.is_scripting():
    #     tens_ops = (query, key, value)
    #     if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
    #         raise NotImplementedError
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # print(query)
            # print(key)
            # print(value)
            # q, k, v = F.linear(query, self.quant_in_proj, in_proj_bias).chunk(3, dim=-1)
            q, k, v = self_arg.quant_in_proj(query).chunk(3, dim=-1)
            # raise NotImplementedError
        elif torch.equal(key, value):
            raise NotImplementedError
        else:
            # # This is inline in_proj function with in_proj_weight and in_proj_bias
            # query, act_scaling_factor = self_arg.quant_query(query)
            # q = self_arg.in_proj_q(query, act_scaling_factor)
            #
            # # This is inline in_proj function with in_proj_weight and in_proj_bias
            # key, act_scaling_factor = self_arg.quant_key(key)
            # k = self_arg.in_proj_k(key, act_scaling_factor)
            #
            # # This is inline in_proj function with in_proj_weight and in_proj_bias
            # value, act_scaling_factor = self_arg.quant_value(value)
            # v = self_arg.in_proj_v(value, act_scaling_factor)

            # full_precision mode
            q = self_arg.in_proj_q(query)
            k = self_arg.in_proj_k(key)
            v = self_arg.in_proj_v(value)

    else:
        raise NotImplementedError
    q = q * scaling

    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
            attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    if bias_k is not None and bias_v is not None:
        raise NotImplementedError
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        raise NotImplementedError

    if static_v is not None:
        raise NotImplementedError

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        raise NotImplementedError
    QUANT_BF_AFT_SOFT = False#self_arg.quant_softmax
    if QUANT_BF_AFT_SOFT:
        raise NotImplementedError
        # q, q_act_scaling_factor = self_arg.quant_query_aft_proj(q)
        # q_int = q/q_act_scaling_factor
        # k, k_act_scaling_factor = self_arg.quant_key_aft_proj(k)
        # k_int = k/k_act_scaling_factor
        # attn_output_weights = torch.bmm(q_int, k_int.transpose(1, 2))*q_act_scaling_factor*k_act_scaling_factor
    else:
        q_int = q
        k_int = k
        attn_output_weights = torch.bmm(q_int, k_int.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    if QUANT_BF_AFT_SOFT:
        raise NotImplementedError
        # v, v_act_scaling_factor = self_arg.quant_value_aft_proj(v)
        # attn_output_weights, attn_act_scaling_factor = self_arg.quant_attn_output_weights_aft_softmax(attn_output_weights)
        # attn_output = torch.bmm(attn_output_weights/attn_act_scaling_factor,
        #                     v/v_act_scaling_factor) *v_act_scaling_factor*attn_act_scaling_factor
    else:
        attn_output = torch.bmm(attn_output_weights, v)

    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

    # attn_output,act_scaling_factor = self_arg.quant_attn_output_before_out_proj(attn_output)
    # attn_output = self_arg.out_proj(attn_output,act_scaling_factor)
    attn_output = self_arg.out_proj(attn_output)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

class Q_TransformerDecoderLayer(nn.Module):
    def __init__(self,decoder_layer):
        super().__init__()
        self.normalize_before = decoder_layer.normalize_before

        self_attn = getattr(decoder_layer, 'self_attn')
        multihead_attn = getattr(decoder_layer,'multihead_attn')
        self.quant_self_attn = QuantMultiheadAttention()
        self.quant_self_attn.set_param(self_attn)
        self.quant_multihead_attn = QuantMultiheadAttention()
        self.quant_multihead_attn.set_param(multihead_attn)

        linear1 = getattr(decoder_layer,'linear1')
        linear2 = getattr(decoder_layer,'linear2')
        self.quant_act1 = QuantAct()
        self.quant_linear1 = QuantLinear()
        self.quant_linear1.set_param(linear1)
        self.quant_act2 = QuantAct()
        self.quant_linear2 = QuantLinear()
        self.quant_linear2.set_param(linear2)

        self.dropout = nn.Dropout(getattr(decoder_layer, 'dropout').p)
        self.dropout1 = nn.Dropout(getattr(decoder_layer, 'dropout1').p)
        self.dropout2 = nn.Dropout(getattr(decoder_layer, 'dropout2').p)
        self.dropout3 = nn.Dropout(getattr(decoder_layer, 'dropout3').p)

        norm1 = getattr(decoder_layer,'norm1')
        norm2 = getattr(decoder_layer,'norm2')
        norm3 = getattr(decoder_layer,'norm3')
        self.quant_norm1 = norm1
        self.quant_norm2 = norm2
        self.quant_norm3 = norm3

        self.activation = getattr(decoder_layer,'activation')
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    def forward(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            raise NotImplementedError
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.quant_self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.quant_norm1(tgt)
        tgt2 = self.quant_multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.quant_norm2(tgt)
        x, act_scaling_factor = self.quant_act1(tgt)
        x = self.dropout(self.activation(self.quant_linear1(x,act_scaling_factor)))
        x, act_scaling_factor = self.quant_act2(x)
        tgt2 = self.quant_linear2(x, act_scaling_factor)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.quant_norm3(tgt)
        return tgt

class Q_TransformerEncoderLayer(nn.Module):
    def __init__(self,encoder_layer):
        super().__init__()
        self.normalize_before = encoder_layer.normalize_before
        self_attn = getattr(encoder_layer,"self_attn")
        self.quant_self_attn = QuantMultiheadAttention()
        self.quant_self_attn.set_param(self_attn)

        linear1 = getattr(encoder_layer, 'linear1')
        linear2 = getattr(encoder_layer, 'linear2')
        self.quant_linear1 = QuantLinear()
        self.quant_linear1.set_param(linear1)
        self.quant_linear2 = QuantLinear()
        self.quant_linear2.set_param(linear2)

        self.dropout = nn.Dropout(getattr(encoder_layer, 'dropout').p)
        self.dropout1 = nn.Dropout(getattr(encoder_layer, 'dropout1').p)
        self.dropout2 = nn.Dropout(getattr(encoder_layer, 'dropout2').p)

        norm1 = getattr(encoder_layer, 'norm1')
        norm2 = getattr(encoder_layer, 'norm2')
        self.quant_norm1 = norm1
        self.quant_norm2 = norm2

        self.quant_act1 = QuantAct()
        self.quant_act2 = QuantAct()
        self.activation = getattr(encoder_layer, 'activation')
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    def forward(self,src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            raise NotImplementedError
        q = k = self.with_pos_embed(src, pos)
        src2 = self.quant_self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.quant_norm1(src)
        src, act_scaling_factor = self.quant_act1(src)
        x = self.quant_linear1(src, act_scaling_factor)
        x = self.activation(x)
        x = self.dropout(x)
        x, act_scaling_factor = self.quant_act2(x)
        src2 = self.quant_linear2(x, act_scaling_factor)
        src = src + self.dropout2(src2)
        src = self.quant_norm2(src)
        return src
