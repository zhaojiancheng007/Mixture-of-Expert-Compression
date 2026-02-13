# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
import torch.utils.checkpoint as checkpoint

from functools import reduce
from operator import mul
import math

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .gdn import GDN
import matplotlib.pyplot as plt

__all__ = [
    "AttentionBlock",
    "MaskedConv2d",
    "MultistageMaskedConv2d",
    "ResidualBlock",
    "ResidualBlockUpsample",
    "ResidualBlockWithStride",
    "conv3x3",
    "subpel_conv3x3",
    "QReLU",
    "RSTB"
]


class MaskedConv2d(nn.Conv2d):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, x: Tensor) -> Tensor:
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)


class MultistageMaskedConv2d(nn.Conv2d):
    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        if mask_type == 'A':
            self.mask[:, :, 0::2, 0::2] = 1
        elif mask_type == 'B':
            self.mask[:, :, 0::2, 1::2] = 1
            self.mask[:, :, 1::2, 0::2] = 1
        elif mask_type == 'C':
            self.mask[:, :, :, :] = 1
            self.mask[:, :, 1:2, 1:2] = 0
        else:
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

    def forward(self, x: Tensor) -> Tensor:
        # TODO: weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r**2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = GDN(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


class AttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out


class QReLU(Function):
    """QReLU

    Clamping input with given bit-depth range.
    Suppose that input data presents integer through an integer network
    otherwise any precision of input will simply clamp without rounding
    operation.

    Pre-computed scale with gamma function is used for backward computation.

    More details can be found in
    `"Integer networks for data compression with latent-variable models"
    <https://openreview.net/pdf?id=S1zz2i0cY7>`_,
    by Johannes Ballé, Nick Johnston and David Minnen, ICLR in 2019

    Args:
        input: a tensor data
        bit_depth: source bit-depth (used for clamping)
        beta: a parameter for modeling the gradient during backward computation
    """

    @staticmethod
    def forward(ctx, input, bit_depth, beta):
        # TODO(choih): allow to use adaptive scale instead of
        # pre-computed scale with gamma function
        ctx.alpha = 0.9943258522851727
        ctx.beta = beta
        ctx.max_value = 2**bit_depth - 1
        ctx.save_for_backward(input)

        return input.clamp(min=0, max=ctx.max_value)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        (input,) = ctx.saved_tensors

        grad_input = grad_output.clone()
        grad_sub = (
            torch.exp(
                (-ctx.alpha**ctx.beta)
                * torch.abs(2.0 * input / ctx.max_value - 1) ** ctx.beta
            )
            * grad_output.clone()
        )

        grad_input[input < 0] = grad_sub[input < 0]
        grad_input[input > ctx.max_value] = grad_sub[input > ctx.max_value]

        return grad_input, None, None


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class PatchUnEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, -1, x_size[0], x_size[1])
        return x

    def flops(self):
        flops = 0
        return flops


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        out_vis =  dict()
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        out_vis['inner_prod'] = attn.detach()

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        out_vis['rpb'] = relative_position_bias.unsqueeze(0).detach()

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        out_vis['attn'] = attn.detach()

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, out_vis

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N, img_N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * img_N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * img_N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += img_N * self.dim * self.dim
        return flops

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_moe=False, moe_config=None, use_prompt=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.actual_resolution = None
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.use_moe = use_moe
        if use_moe:
            moe_type = moe_config.get('moe_type', 'spatial')
            if moe_type == 'channel':
                self.moe_mlp = ChannelMoEBlock(
                    hidden_dim=dim,
                    num_groups=moe_config.get('num_groups', moe_config['num_experts']),
                    num_experts=moe_config['num_experts'],
                    capacity=moe_config['capacity'],
                    hid_ratio=moe_config.get('hid_ratio', 1),
                    n_shared_experts=moe_config['n_shared_experts'],
                )
            elif moe_type == 'patch':
                _ps = moe_config.get('patch_size', 4)
                self.moe_mlp = PatchMoEBlock(
                    experts=[Mlp(in_features=dim, hidden_features=mlp_hidden_dim // moe_config['hid_ratio'], act_layer=act_layer, drop=drop) for _ in range(moe_config['num_experts'])],
                    hidden_dim=dim,
                    num_experts=moe_config['num_experts'],
                    capacity=moe_config['capacity'],
                    n_shared_experts=moe_config['n_shared_experts'],
                    use_prompt=use_prompt,
                    patch_size=_ps,
                )
            else:
                self.moe_mlp = SparseMoEBlock(
                    experts=[Mlp(in_features=dim, hidden_features=mlp_hidden_dim // moe_config['hid_ratio'], act_layer=act_layer, drop=drop) for _ in range(moe_config['num_experts'])],
                    hidden_dim=dim,
                    num_experts=moe_config['num_experts'],
                    capacity=moe_config['capacity'],
                    n_shared_experts=moe_config['n_shared_experts'],
                    use_prompt=use_prompt,
                    mix_batch_token=moe_config.get('mix_batch_token', False),
                        )
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
           
        if self.shift_size > 0:
            attn_mask =  None #self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size, prompt=None):
        self.actual_resolution = x_size
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows, out_vis = self.attn(x_windows, mask=None)  # nW*B, window_size*window_size, C
        else:
            attn_windows, out_vis = self.attn(x_windows, mask=None)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        if self.use_moe:
            if isinstance(self.moe_mlp, PatchMoEBlock):
                x = x + self.drop_path(self.moe_mlp(self.norm2(x), prompt, x_size=x_size))
            elif isinstance(self.moe_mlp, SparseMoEBlock):
                x = x + self.drop_path(self.moe_mlp(self.norm2(x), prompt, x_size=x_size))
            else:
                x = x + self.drop_path(self.moe_mlp(self.norm2(x), prompt))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, out_vis

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 block_module=SwinTransformerBlock,
                 use_moe=False, moe_config=None, use_prompt=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

     
        self.blocks = nn.ModuleList([
            block_module(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,  # noqa
                norm_layer=norm_layer,
                use_moe=use_moe, moe_config=moe_config,use_prompt=use_prompt)
            for i in range(depth)])
        


    def forward(self, x, x_size, prompt=None):
        attns = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, _ = blk(x, x_size, prompt)
                attn = None
                attns.append(attn)
      
        return x, attns

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False, 
                 use_moe=False, moe_config=None, use_prompt=False):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                            input_resolution=input_resolution,
                            depth=depth,
                            num_heads=num_heads,
                            window_size=window_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop, attn_drop=attn_drop,
                            drop_path=drop_path,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            use_moe=use_moe, moe_config=moe_config,
                            use_prompt=use_prompt
                                            )

        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed()


    def forward(self, x, x_size, prompt=None):
        out = self.patch_embed(x)
        out, attns = self.residual_group(out, x_size, prompt)
        return self.patch_unembed(out, x_size) + x, attns

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops

class MoeMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, pretraining_tp=2):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.pretraining_tp = pretraining_tp

    def forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0) 
            # print(self.up_proj.weight.size(), self.down_proj.weight.size())
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=-1)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

class SparseMoEBlock(nn.Module):
    """
    A sparse MoE block with optional shared experts (spatial / pixel-level routing).

    Each forward pass records ``_last_select_count`` (how many experts chose
    each spatial token) and ``_last_grid_size`` (H, W) for heatmap visualisation.
    """
    def __init__(self, experts, hidden_dim, num_experts, n_shared_experts=0, capacity=2,
                 use_prompt=False, prompt_mod='add', mix_batch_token=False):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.capacity = capacity
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.use_prompt = use_prompt
        self.prompt_mod = prompt_mod
        self.mix_batch_token = bool(mix_batch_token)

        self.gate_weight = nn.Parameter(torch.empty((hidden_dim, num_experts)))
        nn.init.normal_(self.gate_weight, std=0.006)

        self.n_shared_experts = n_shared_experts
        if self.n_shared_experts > 0:
            intermediate_size = hidden_dim * self.n_shared_experts
            self.shared_experts = MoeMLP(
                hidden_size=hidden_dim,
                intermediate_size=intermediate_size,
                pretraining_tp=2
            )

        # Populated every forward for vis
        self._last_select_count = None   # (B, S)
        self._last_grid_size = None      # (H, W)
        self._last_gate_mass = None      # (B, S)
        self._last_index = None          # (B, E, k)
        self._last_gating = None         # (B, E, k)
        self._last_token_energy = None   # (B, S)
        self._last_smooth_loss = None    # scalar, differentiable when training

    @staticmethod
    def _factorize_hw(S):
        H = W = int(math.sqrt(S))
        if H * W == S:
            return H, W
        for h in range(int(math.sqrt(S)), 0, -1):
            if S % h == 0:
                return h, S // h
        return S, 1

    def _infer_hw(self, S, x_size=None):
        if x_size is not None and len(x_size) == 2:
            H, W = int(x_size[0]), int(x_size[1])
            if H > 0 and W > 0 and H * W == S:
                return H, W
        return self._factorize_hw(S)

    @staticmethod
    def _tv_l1(prob_map):
        """
        Total variation (anisotropic): ||dx||_1 + ||dy||_1 on (B, E, H, W).
        """
        loss = prob_map.new_zeros(())
        if prob_map.size(-2) > 1:
            loss = loss + (prob_map[:, :, 1:, :] - prob_map[:, :, :-1, :]).abs().mean()
        if prob_map.size(-1) > 1:
            loss = loss + (prob_map[:, :, :, 1:] - prob_map[:, :, :, :-1]).abs().mean()
        return loss

    def _prob_smooth_loss(self, affinity, hw):
        """
        affinity: (B, S, E) softmax probabilities over experts.
        hw: (H, W) for reshaping token sequence into a spatial map.
        """
        B, S, E = affinity.shape
        H, W = hw
        if H * W != S:
            return affinity.new_zeros(())
        prob_map = affinity.permute(0, 2, 1).reshape(B, E, H, W)
        return self._tv_l1(prob_map)

    def forward(self, x, prompt=None, x_size=None):
        """
        x: (B, S, D)  batch_size, seq_len, hidden_dim
        x_size: optional (H, W) for vis; inferred from S if absent.
        """
        identity = x
        B, S, D = x.shape

        self._last_grid_size = self._infer_hw(S, x_size=x_size)
        self._last_token_energy = (x.pow(2).mean(dim=-1)).detach()  # (B, S)
        self._last_smooth_loss = x.new_zeros(())

        if self.mix_batch_token:
            # Optional global routing across all tokens in the batch.
            # Pool size becomes (B * S), so experts can allocate compute
            # to high-value tokens regardless of which sample they come from.
            x_flat = x.reshape(B * S, D)  # (BS, D)
            logits = x_flat @ self.gate_weight  # (BS, E)
            affinity = torch.softmax(logits, dim=-1)
            if self.training:
                affinity_bse = affinity.reshape(B, S, self.num_experts)
                self._last_smooth_loss = self._prob_smooth_loss(
                    affinity_bse, self._last_grid_size
                )
            affinity_T = affinity.transpose(0, 1)  # (E, BS)

            k = max(1, int(((B * S) / self.num_experts) * self.capacity))
            gating, index = torch.topk(affinity_T, k=k, dim=-1)  # (E, k)

            select_count_flat = torch.zeros(B * S, device=x.device)
            for e in range(self.num_experts):
                select_count_flat.scatter_add_(
                    0, index[e], torch.ones_like(gating[e])
                )
            self._last_select_count = select_count_flat.reshape(B, S).detach()

            gate_mass_flat = torch.zeros(B * S, device=x.device, dtype=gating.dtype)
            for e in range(self.num_experts):
                gate_mass_flat.scatter_add_(0, index[e], gating[e])
            self._last_gate_mass = gate_mass_flat.reshape(B, S).detach()

            # Keep vis-safe: existing expert-specialization code expects (B,E,k).
            self._last_index = None
            self._last_gating = None

            x_out_flat = torch.zeros_like(x_flat)
            for e in range(self.num_experts):
                idx = index[e]                     # (k,)
                x_selected = x_flat.index_select(0, idx)  # (k, D)
                out = self.experts[e](x_selected)         # (k, D)
                contrib = out * gating[e].unsqueeze(-1)   # (k, D)
                x_out_flat.index_add_(0, idx, contrib)

            x_out = x_out_flat.reshape(B, S, D)
        else:
            # === router logits ===
            logits = x @ self.gate_weight
            affinity = torch.softmax(logits, dim=-1)
            if self.training:
                self._last_smooth_loss = self._prob_smooth_loss(
                    affinity, self._last_grid_size
                )

            # --> (B, E, S)
            affinity_T = affinity.permute(0, 2, 1)  # (B, E, S)

            # === top-k routing ===
            k = max(1, int((S / self.num_experts) * self.capacity))
            gating, index = torch.topk(affinity_T, k=k, dim=-1)  # (B, E, k)

            # === record per-token selection count for vis ===
            select_count = torch.zeros(B, S, device=x.device)
            for e in range(self.num_experts):
                ones = torch.ones_like(gating[:, e, :])   # (B, k)
                select_count.scatter_add_(1, index[:, e, :], ones)
            self._last_select_count = select_count.detach()  # (B, S)

            gate_mass = torch.zeros(B, S, device=x.device, dtype=gating.dtype)
            for e in range(self.num_experts):
                gate_mass.scatter_add_(1, index[:, e, :], gating[:, e, :])
            self._last_gate_mass = gate_mass.detach()  # (B, S)

            self._last_index = index.detach()
            self._last_gating = gating.detach()

            # === expert forward ===
            x_in = []
            for e in range(self.num_experts):
                idx = index[:, e, :]  # (B, k)
                x_selected = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, D))
                x_in.append(x_selected)

            x_e = []
            for e in range(self.num_experts):
                out = self.experts[e](x_in[e])   # (B, k, D)
                x_e.append(out)
            x_e = torch.stack(x_e, dim=1)  # (B, E, k, D)

            # === scatter back --> (B, S, D) ===
            x_out = torch.zeros_like(x)
            for e in range(self.num_experts):
                idx = index[:, e, :]  # (B, k)
                weight = gating[:, e, :].unsqueeze(-1)  # (B, k, 1)
                contrib = x_e[:, e, :, :] * weight      # (B, k, D)
                x_out = x_out.scatter_add(1, idx.unsqueeze(-1).expand(-1, -1, D), contrib)

        if self.n_shared_experts > 0:
            x_out = x_out + self.shared_experts(identity)

        return x_out


class PatchMoEBlock(nn.Module):
    """Spatial MoE with patch tokenisation.

    Instead of treating every pixel as a token (``SparseMoEBlock``), this
    block first *patchifies* the spatial feature map into non-overlapping
    ``patch_size x patch_size`` blocks, averages each block into a single
    "patch token", routes those patch tokens to experts via expert-choice
    top-k, runs each expert on the selected patch tokens, and finally
    scatters the results back and un-patchifies.

    The expert MLP operates on the *same* channel dimension ``D`` as a
    standard FFN – the only difference is that the routing granularity is
    now a patch (e.g. 4x4) instead of a single pixel.

    ``patch_size`` is read from ``moe_config['patch_size']`` (default 4).

    For the heatmap visualisation each forward pass records which tokens
    were selected by how many experts into ``self._last_select_count``
    (detached, ``(B, T)`` where ``T`` is the number of patch tokens).
    """

    def __init__(self, experts, hidden_dim, num_experts,
                 n_shared_experts=0, capacity=2, use_prompt=False,
                 patch_size=4):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.capacity = capacity
        self.num_experts = num_experts
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # ---- gate: fuse [LN(mean), complexity_embed] -> logits ----
        # Mean captures "what" the patch represents (direction).
        # Intra-patch variance (differentiable entropy proxy) captures
        # "how complex" the patch is, so the router can learn to send
        # complex patches to specialised experts.
        self.gate_norm = nn.LayerNorm(hidden_dim)
        self.complexity_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_linear = nn.Linear(hidden_dim * 2, num_experts)

        self.n_shared_experts = n_shared_experts
        if n_shared_experts > 0:
            intermediate_size = hidden_dim * n_shared_experts
            self.shared_experts = MoeMLP(
                hidden_size=hidden_dim,
                intermediate_size=intermediate_size,
                pretraining_tp=2,
            )

        # Populated every forward for vis; (B, T)
        self._last_select_count = None

    # ------------------------------------------------------------------ #
    #  helpers: patchify / un-patchify                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _patchify(x, ps):
        """(B, H, W, D) -> (B, Th, Tw, ps, ps, D) -> (B, T, ps*ps, D)"""
        B, H, W, D = x.shape
        Th, Tw = H // ps, W // ps
        x = x.reshape(B, Th, ps, Tw, ps, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()   # (B,Th,Tw,ps,ps,D)
        x = x.reshape(B, Th * Tw, ps * ps, D)           # (B,T, ps^2, D)
        return x, Th, Tw

    @staticmethod
    def _unpatchify(x, Th, Tw, ps):
        """(B, T, ps*ps, D) -> (B, H, W, D)"""
        B, T, P, D = x.shape
        x = x.reshape(B, Th, Tw, ps, ps, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B, Th * ps, Tw * ps, D)
        return x

    def forward(self, x, prompt=None, x_size=None):
        """
        Args:
            x: (B, S, D)  where S = H * W.
            x_size: optional (H, W) tuple – avoids ambiguous factorisation.
        Returns:
            (B, S, D).
        """
        B, S, D = x.shape
        ps = self.patch_size

        if x_size is not None:
            H, W = x_size
        else:
            # Infer spatial dims – assumes S = H * W with H, W divisible by ps.
            H = W = int(math.sqrt(S))
            if H * W != S:
                # Non-square: find factors closest to sqrt
                for h in range(int(math.sqrt(S)), 0, -1):
                    if S % h == 0:
                        H = h
                        W = S // h
                        break

        # Pad if H or W not divisible by ps
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h > 0 or pad_w > 0:
            x_2d = x.reshape(B, H, W, D)
            x_2d = F.pad(x_2d.permute(0, 3, 1, 2),
                         (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
            Hp, Wp = H + pad_h, W + pad_w
        else:
            x_2d = x.reshape(B, H, W, D)
            Hp, Wp = H, W

        identity = x  # (B, S, D)

        # ---- 1. patchify ----
        patches, Th, Tw = self._patchify(x_2d, ps)  # (B, T, ps^2, D)
        T = Th * Tw

        # ---- 2. gate: complexity-aware routing ----
        # patch mean: spatial average → "what" direction
        patch_mean = patches.mean(dim=2)              # (B, T, D)
        # intra-patch variance: differentiable entropy proxy → "how complex"
        patch_var = patches.var(dim=2, unbiased=False) # (B, T, D)
        # fuse: [LN(mean), proj(var)] → logits
        gate_feat = torch.cat([
            self.gate_norm(patch_mean),
            self.complexity_proj(patch_var),
        ], dim=-1)                                    # (B, T, 2D)
        logits = self.gate_linear(gate_feat)          # (B, T, E)
        affinity = torch.softmax(logits, dim=-1)      # (B, T, E)

        # ---- 3. expert-choice top-k ----
        affinity_T = affinity.permute(0, 2, 1)       # (B, E, T)
        k = max(1, int((T / self.num_experts) * self.capacity))
        gating, index = torch.topk(affinity_T, k=k, dim=-1)  # (B, E, k)

        # ---- 4. record per-token selection count for vis ----
        select_count = torch.zeros(B, T, device=x.device)
        for e in range(self.num_experts):
            ones = torch.ones_like(gating[:, e, :])   # (B, k)
            select_count.scatter_add_(1, index[:, e, :], ones)
        self._last_select_count = select_count.detach()  # (B, T)
        self._last_grid_size = (Th, Tw)  # for vis heatmap

        # ---- 5. expert forward ----
        # Each expert processes its k patch tokens.
        # A "patch token" here is the full patch (ps^2 pixels, each D-dim),
        # but the expert MLP works on D-dim vectors (same as standard FFN).
        # So we gather the selected patches, flatten pixels, run expert,
        # reshape back.
        x_out_patches = torch.zeros_like(patches)     # (B, T, ps^2, D)

        # Per-token weight normalisation
        flat_idx = index.reshape(B, self.num_experts * k)
        flat_wt  = gating.reshape(B, self.num_experts * k)
        total_wt = torch.zeros(B, T, device=x.device)
        total_wt.scatter_add_(1, flat_idx, flat_wt)
        total_wt_safe = total_wt.clamp(min=1e-8)

        P = ps * ps  # pixels per patch

        for e in range(self.num_experts):
            idx = index[:, e, :]                       # (B, k)
            wt  = gating[:, e, :]                      # (B, k)

            # Normalise weight
            tw_sel = torch.gather(total_wt_safe, 1, idx)  # (B, k)
            wt_norm = wt / tw_sel                      # (B, k)

            # Gather patches: (B, k, ps^2, D)
            idx_exp = idx[:, :, None, None].expand(B, k, P, D)
            p_sel = torch.gather(patches, 1, idx_exp)  # (B, k, P, D)

            # Expert MLP on each pixel independently: (B*k*P, D) -> (B*k*P, D)
            p_flat = p_sel.reshape(B * k * P, D)
            p_proc = self.experts[e](p_flat)
            p_proc = p_proc.reshape(B, k, P, D)

            # Weight
            wt_e = wt_norm[:, :, None, None]           # (B, k, 1, 1)
            contrib = p_proc * wt_e

            # Scatter back
            x_out_patches.scatter_add_(1, idx_exp, contrib)

        # ---- 6. shared experts ----
        if self.n_shared_experts > 0:
            sh_flat = patches.reshape(B * T * P, D)
            sh_out = self.shared_experts(sh_flat)
            x_out_patches = x_out_patches + sh_out.reshape(B, T, P, D)

        # ---- 7. un-patchify ----
        x_out_2d = self._unpatchify(x_out_patches, Th, Tw, ps)  # (B, Hp, Wp, D)

        # Remove padding if added
        if pad_h > 0 or pad_w > 0:
            x_out_2d = x_out_2d[:, :H, :W, :]

        return x_out_2d.reshape(B, S, D)


class ChannelMoEBlock(nn.Module):
    """Per-channel Mixture-of-Experts block.

    Routes individual *channels* (not spatial tokens or channel groups)
    to experts.  Each expert receives **all** channels assigned to it and
    processes them as a single feature vector per spatial position.  The
    output for each channel is the normalised weighted sum of all experts
    that processed it, ensuring channels picked by multiple experts are
    properly aggregated.  Channels not selected by any expert keep their
    identity (no information loss).

    Routing signal per channel:
        feat = concat( LN(spatial_mean), energy_embed )   dim = 2
        logits = Linear(feat) -> (B, C, E) -> softmax

    Expert-choice top-k: each expert picks the top-k channels (from C)
    with the highest affinity.  k = (C / E) * capacity.

    Expert architecture: ``k -> hidden -> k`` MLP applied per spatial
    position (strict 2D input).

    Constructor signature is kept backward-compatible with dispatch sites
    that still pass ``num_groups`` (ignored) and ``hid_ratio``.
    """

    def __init__(self, hidden_dim, num_groups=None, num_experts=4,
                 capacity=2, hid_ratio=1, n_shared_experts=0):
        """
        Args:
            hidden_dim: C – total number of channels.
            num_groups: *ignored* (kept for dispatch-site compatibility).
            num_experts: E – number of routed experts.
            capacity: top-k multiplier; k = ceil(C / E * capacity).
            hid_ratio: expert hidden = k * 4 // hid_ratio  (resolved at
                       forward time since k depends on C).
            n_shared_experts: number of shared experts (0 = none).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.capacity = capacity
        self.hid_ratio = hid_ratio
        # Keep for backward compat / vis code that reads it
        self.num_groups = hidden_dim
        self.group_dim = 1

        C = hidden_dim
        E = num_experts

        # ---- Gate ----
        # Per-channel routing: each channel is described by a 2-dim feature
        #   [LN(spatial_mean_c),  energy_embed_c]
        # LN operates per-channel (scalar), energy_embed maps scalar -> 1.
        self.gate_norm = nn.LayerNorm(1)
        self.energy_proj = nn.Linear(1, 1)
        self.gate_linear = nn.Linear(2, E)

        # ---- Experts ----
        # k (channels per expert) is deterministic given C and E
        k = max(1, math.ceil(C / E * capacity))
        expert_hidden = max(1, k * 4 // max(1, hid_ratio))
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(k, expert_hidden),
                nn.GELU(),
                nn.Linear(expert_hidden, k),
            )
            for _ in range(E)
        ])
        self._k = k  # cached for forward

        # ---- Optional shared expert ----
        self.n_shared_experts = n_shared_experts
        if n_shared_experts > 0:
            shared_hidden = max(1, C * n_shared_experts)
            self.shared_experts = nn.Sequential(
                nn.Linear(C, shared_hidden),
                nn.SiLU(),
                nn.Linear(shared_hidden, C),
            )

    def forward(self, x, prompt=None):
        """
        Args:
            x: (B, S, C) where S = H*W.
        Returns:
            (B, S, C).
        """
        B, S, C = x.shape
        E = self.num_experts
        k = self._k  # channels per expert

        # ---- 1. per-channel routing signal ----
        # spatial mean per channel: (B, C)
        ch_mean = x.mean(dim=1)                                # (B, C)
        # per-channel L2 energy (avg over spatial): (B, C)
        ch_energy = x.norm(dim=1)                              # (B, C)
        # norm operates on S dim → actually we want per-element energy,
        # use mean of abs or std; here: RMS = sqrt(mean(x^2)) per channel
        ch_energy = (x ** 2).mean(dim=1).sqrt()                # (B, C)

        # Build 2D feature per channel: (B, C, 2)
        feat_dir = self.gate_norm(ch_mean.unsqueeze(-1))       # (B, C, 1)
        feat_eng = self.energy_proj(ch_energy.unsqueeze(-1))   # (B, C, 1)
        gate_feat = torch.cat([feat_dir, feat_eng], dim=-1)    # (B, C, 2)

        # ---- 2. gate logits ----
        logits = self.gate_linear(gate_feat)                   # (B, C, E)
        # softmax over experts for each channel
        affinity = torch.softmax(logits, dim=-1)               # (B, C, E)

        # ---- 3. expert-choice top-k ----
        affinity_T = affinity.permute(0, 2, 1)                # (B, E, C)
        gating, index = torch.topk(affinity_T, k=k, dim=-1)   # (B, E, k)

        # ---- 4. per-channel weight normalisation (vectorised) ----
        flat_idx = index.reshape(B, E * k)                     # (B, E*k)
        flat_wt  = gating.reshape(B, E * k)                    # (B, E*k)

        total_weight = torch.zeros(B, C, device=x.device)
        total_weight.scatter_add_(1, flat_idx, flat_wt)        # (B, C)
        # Channels with total_weight == 0 are not selected → identity
        # For selected channels, normalise weights to sum to 1
        total_weight_safe = total_weight.clamp(min=1e-8)

        tw_per_slot = torch.gather(total_weight_safe, 1, flat_idx)  # (B, E*k)
        wt_norm = (flat_wt / tw_per_slot).reshape(B, E, k)    # (B, E, k)

        # ---- 5. expert forward ----
        # x: (B, S, C) → gather k channels per expert
        # index: (B, E, k) channel indices
        # We need (B, E, S, k) — gather along C dim of x
        idx_exp = index[:, :, None, :].expand(B, E, S, k)     # (B, E, S, k)
        x_exp = x.unsqueeze(1).expand(B, E, S, C)             # (B, E, S, C)
        x_sel = torch.gather(x_exp, 3, idx_exp)               # (B, E, S, k)

        # Each expert processes (B*S, k) → (B*S, k)  — strict 2D
        expert_out = []
        for e in range(E):
            inp_2d = x_sel[:, e].reshape(B * S, k)            # (B*S, k)
            out_2d = self.experts[e](inp_2d)                   # (B*S, k)
            expert_out.append(out_2d.reshape(B, S, k))
        expert_out = torch.stack(expert_out, dim=1)            # (B, E, S, k)

        # ---- 6. weighted aggregation ----
        # Multiply each expert's output by its normalised weight
        wt_exp = wt_norm[:, :, None, :]                        # (B, E, 1, k)
        weighted = expert_out * wt_exp                         # (B, E, S, k)

        # Scatter-add back to full C dimension
        x_out = torch.zeros(B, E, S, C, device=x.device)
        x_out.scatter_add_(3, idx_exp, weighted)               # (B, E, S, C)
        x_out = x_out.sum(dim=1)                               # (B, S, C)


        # ---- 7. shared expert ----
        if self.n_shared_experts > 0:
            # (B*S, C) → (B*S, C)
            x_shared = self.shared_experts(x.reshape(B * S, C))
            x_out = x_out + x_shared.reshape(B, S, C)

        return x_out


# class SparseMoEBlock(nn.Module):
#     """
#     A mixed expert module containing shared experts.
#     """
#     def __init__(self, experts, hidden_dim, num_experts, n_shared_experts=0, capacity=2, use_prompt=False, prompt_mod='add'):
#         super().__init__()
#         self.experts = nn.ModuleList(experts)
#         self.capacity = capacity
#         self.num_experts = num_experts

#         self.use_prompt = use_prompt
#         self.prompt_mod = prompt_mod
#         if self.prompt_mod == 'add':
#             hidden_dim = hidden_dim
#         elif self.prompt_mod == 'concat':
#             hidden_dim = hidden_dim * 2
#         else:
#             raise ValueError(f"Invalid router_mode: {self.router_mode}")

#         self.gate_weight = nn.Parameter(torch.empty((hidden_dim, num_experts)))
#         nn.init.normal_(self.gate_weight, std=0.006)

#         self.n_shared_experts = n_shared_experts

#         if self.n_shared_experts > 0:
#             intermediate_size = hidden_dim * self.n_shared_experts
#             self.shared_experts = MoeMLP(hidden_size = hidden_dim, intermediate_size = intermediate_size, pretraining_tp=2)
    

#     def forward(self, x, prompt=None):

#         if self.use_prompt and prompt is not None:
#             prompt = prompt.unsqueeze(1).expand(-1, S, -1)
#             if self.prompt_mod == 'add':
#                 x = x + prompt
#             else:
#                 x = torch.cat([x, prompt], dim=-1)

#         identity = x
#         B, S, D = x.shape
#         # 1. Compute token-expert affinity scores
#         logits = x @ self.gate_weight      # bs, seq_len, num_experts
#         affinity = logits.softmax(dim=-1)
#         affinity = torch.einsum('b s e->b e s', affinity)
#         # 2. select the top-k tokens for each experts
#         k = int( (S/self.num_experts) * self.capacity)
#         # print(k, S, self.capacity, self.num_experts)
#         gating, index = torch.topk(affinity, k=k, dim=-1, sorted=False)
#         dispatch = F.one_hot(index, num_classes=S).to(device=x.device, dtype=x.dtype)
#         # 3. Process the tokens by each expert and combine
#         x_in = torch.einsum(" b e c s, b s d -> b e c d", dispatch, x)
#         x_e = [self.experts[e](x_in[:,e]) for e in range(self.num_experts)]
#         x_e = torch.stack(x_e, dim=1)
#         x_out = torch.einsum('b e c s, b e c, b e c d -> b s d', dispatch, gating, x_e)
#         if self.n_shared_experts >0:
#             x_out = x_out + self.shared_experts(identity)
#         return x_out



# class SparseMoEBlock(nn.Module):
#     def __init__(self, experts, hidden_dim, num_experts,
#                  n_shared_experts=0, capacity=2, router_mode='add', use_prompt=False):
#         """
#         Args:
#             experts: list[nn.Module]
#             hidden_dim: feature dim (D)
#             num_experts: expert数量
#             router_mode: ['concat' | 'add'] 融合方式
#         """
#         super().__init__()
#         self.num_experts = num_experts
#         self.capacity = capacity
#         self.use_prompt_in_router = use_prompt
#         self.router_mode = router_mode
#         self.hidden_dim = hidden_dim

#         # ---------------- Router ----------------
#         if router_mode == 'concat' and self.use_prompt_in_router:
#             router_in_dim = hidden_dim * 2
#         else:
#             router_in_dim = hidden_dim

#         self.router = nn.Parameter(torch.empty((router_in_dim, num_experts)))
#         nn.init.normal_(self.router, std=0.006)

#         # ---------------- Experts ----------------
#         self.experts = nn.ModuleList(experts)
#         self.n_shared_experts = n_shared_experts
#         if self.n_shared_experts > 0:
#             intermediate_size = hidden_dim * self.n_shared_experts
#             self.shared_experts = MoeMLP(
#                 hidden_size=hidden_dim,
#                 intermediate_size=intermediate_size,
#                 pretraining_tp=2
#             )

#     def forward(self, x, prompt):
#         """
#         Args:
#             x: (B, S, D)  feature tokens
#             prompt: (B, P)  每个样本的 task prompt embedding
#         """
#         identity = x
#         B, S, D = x.shape

#         if self.use_prompt_in_router:
#             if self.router_mode == 'concat':
#                 # prompt -> broadcast 到每个 token
#                 prompt_expanded = prompt.unsqueeze(1).expand(-1, S, -1)  # [B, S, P]
#                 router_input = torch.cat([x, prompt_expanded], dim=-1)   # [B, S, D+P]
#             elif self.router_mode == 'add':
#                 prompt_expanded = prompt.unsqueeze(1).expand(-1, S, -1)
#                 router_input = x + prompt_expanded
#             else:
#                 raise ValueError(f"Invalid router_mode: {self.router_mode}")
#         else:
#             router_input = x

#         # gating logits
#         logits = router_input @ self.router      # [B, S, E]
#         affinity = F.softmax(logits, dim=-1)
#         affinity_T = affinity.permute(0, 2, 1)      # (B, E, S)

#         # ========== 2. Top-k Token Routing ==========
#         k = int((S / self.num_experts) * self.capacity)
#         gating, index = torch.topk(affinity_T, k=k, dim=-1)  # (B, E, k)

#         # ========== 3. Dispatch Tokens ==========
#         x_in = []
#         for e in range(self.num_experts):
#             idx = index[:, e, :]  # (B, k)
#             x_selected = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, D))
#             x_in.append(x_selected)

#         # ========== 4. Expert ==========
#         x_e = []
#         for e in range(self.num_experts):
#             out = self.experts[e](x_in[e])  # (B, k, D)
#             x_e.append(out)
#         x_e = torch.stack(x_e, dim=1)  # (B, E, k, D)

#         x_out = torch.zeros_like(x)
#         for e in range(self.num_experts):
#             idx = index[:, e, :]
#             weight = gating[:, e, :].unsqueeze(-1)
#             contrib = x_e[:, e, :, :] * weight
#             x_out = x_out.scatter_add(1, idx.unsqueeze(-1).expand(-1, -1, D), contrib)

#         if self.n_shared_experts > 0:
#             x_out = x_out + self.shared_experts(identity)

#         return x_out
