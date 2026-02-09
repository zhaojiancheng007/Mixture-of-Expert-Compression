
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

from timm.models.layers import DropPath

from .gdn import GDN
from natten import NeighborhoodAttention

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
    "ResViTBlock",
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

        if mask_type == 'A': # 3x3
            self.mask[:, :, 0::2, 0::2] = 1
        elif mask_type == 'B':
            self.mask[:, :, 0::2, 1::2] = 1
            self.mask[:, :, 1::2, 0::2] = 1
        elif mask_type == 'C':
            self.mask[:, :, 0::2, 1::2] = 1
            self.mask[:, :, 1::2, :] = 1
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


class NSABlock(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(
            dim, kernel_size=kernel_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.moe_mlp = SparseMoEBlock_forloop(
                experts=[Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop) for _ in range(4)],
                hidden_dim=dim,
                num_experts=4,
                capacity=1.0,
                n_shared_experts=0,
                use_prompt = False)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.moe_mlp(self.norm2(x)))
        return x


class BasicViTLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, kernel_size, mlp_ratio=4., 
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            NSABlock(dim=dim,
                     num_heads=num_heads, kernel_size=kernel_size,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop, attn_drop=attn_drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class ResViTBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, kernel_size=7, mlp_ratio=4,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0.2, norm_layer=nn.LayerNorm):
        super(ResViTBlock, self).__init__()
        self.dim = dim

        self.residual_group = BasicViTLayer(dim=dim, depth=depth, num_heads=num_heads, kernel_size=kernel_size, mlp_ratio=mlp_ratio, 
                                         qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=drop_path_rate, norm_layer=norm_layer)

    def forward(self, x):
        return self.residual_group(x.permute(0,2,3,1)).permute(0,3,1,2) + x


class SparseMoEBlock_forloop(nn.Module):
    """
    A sparse MoE block with optional shared experts.
    """
    def __init__(self, experts, hidden_dim, num_experts, n_shared_experts=0, capacity=2, use_prompt=False, prompt_mod='add'):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.capacity = capacity
        self.num_experts = num_experts

        self.use_prompt = use_prompt
        self.prompt_mod = prompt_mod

        if use_prompt:
            if self.prompt_mod == 'add':
                hidden_dim = hidden_dim
            elif self.prompt_mod == 'concat':
                hidden_dim = hidden_dim * 2
            else:
                raise ValueError(f"Invalid router_mode: {self.router_mode}")
        
        self.gate_weight = nn.Parameter(torch.empty((hidden_dim, num_experts)))
        nn.init.normal_(self.gate_weight, std=0.006)

    def forward(self, x, prompt=None):
        """
        x: (B, S, D)  batch_size, seq_len, hidden_dim
        """
        orig_shape = x.shape
        if x.dim() == 4:
            B, H, W, D = x.shape
            x = x.view(B, H*W, D)
            S= H*W
        elif x.dim() == 3:
            B, S, D = x.shape
        else:
            raise ValueError("SparseMoEBlock expects 3D or 4D input")
        
        # identity = x
        # B, S, D = x.shape

        # 1. Router: token -> expert affinity
        # shape: (B, S, E)
        if self.use_prompt and prompt is not None:
            prompt = prompt.unsqueeze(1).expand(-1, S, -1)
            if self.prompt_mod == 'add':
                x = x + prompt
            else:
                x = torch.cat([x, prompt], dim=-1)

        # === affinity matrix ===
        logits = x @ self.gate_weight  
        affinity = torch.softmax(logits, dim=-1) 

        # --> (B, E, S)
        affinity_T = affinity.permute(0, 2, 1)  # (B, E, S)

        # === 计算 top-k 路由 ===
        k = max(1, int((S / self.num_experts) * self.capacity))
        gating, index = torch.topk(affinity_T, k=k, dim=-1)  # (B, E, k)

        # === expert forward ===
        x_in = []
        for e in range(self.num_experts):
            # 取 expert e 选中的 token 索引 (B, k)
            idx = index[:, e, :]  # (B, k)
            # gather tokens (B, k, D)
            x_selected = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, D))
            x_in.append(x_selected)
        
        x_e = []
        for e in range(self.num_experts):
            out = self.experts[e](x_in[e])   # (B, k, D)
            x_e.append(out)
        x_e = torch.stack(x_e, dim=1)  # (B, E, k, D)

        # === concat --> (B, S, D) ===
        x_out = torch.zeros_like(x)
        for e in range(self.num_experts):
            idx = index[:, e, :]  # (B, k)
            weight = gating[:, e, :].unsqueeze(-1)  # (B, k, 1)
            contrib = x_e[:, e, :, :] * weight      # (B, k, D)

            x_out = x_out.scatter_add(1, idx.unsqueeze(-1).expand(-1, -1, D), contrib)

        if len(orig_shape) == 4:
            x_out = x_out.view(B, H, W, D)

        return x_out


class SparseMoEBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, experts, hidden_dim, num_experts, n_shared_experts=0, capacity=2, use_prompt=False, prompt_mod='add'):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.capacity = capacity
        self.num_experts = num_experts

        self.use_prompt = use_prompt
        self.prompt_mod = prompt_mod
        if self.prompt_mod == 'add':
            hidden_dim = hidden_dim
        elif self.prompt_mod == 'concat':
            hidden_dim = hidden_dim * 2
        else:
            raise ValueError(f"Invalid router_mode: {self.router_mode}")

        self.gate_weight = nn.Parameter(torch.empty((hidden_dim, num_experts)))
        nn.init.normal_(self.gate_weight, std=0.006)

        # self.n_shared_experts = n_shared_experts

        # if self.n_shared_experts > 0:
        #     intermediate_size = hidden_dim * self.n_shared_experts
        #     self.shared_experts = MoeMLP(hidden_size = hidden_dim, intermediate_size = intermediate_size, pretraining_tp=2)
    

    def forward(self, x, prompt=None):

        if self.use_prompt and prompt is not None:
            prompt = prompt.unsqueeze(1).expand(-1, S, -1)
            if self.prompt_mod == 'add':
                x = x + prompt
            else:
                x = torch.cat([x, prompt], dim=-1)

        identity = x

        orig_shape = x.shape
        if x.dim() == 4:
            B, H, W, D = x.shape
            x = x.view(B, H*W, D)
            S= H*W
        elif x.dim() == 3:
            B, S, D = x.shape
        else:
            raise ValueError("SparseMoEBlock expects 3D or 4D input")

        # 1. Compute token-expert affinity scores
        logits = x @ self.gate_weight      # bs, seq_len, num_experts
        affinity = logits.softmax(dim=-1)
        affinity = torch.einsum('b s e->b e s', affinity)
        # 2. select the top-k tokens for each experts
        k = int( (S/self.num_experts) * self.capacity)
        # print(k, S, self.capacity, self.num_experts)
        gating, index = torch.topk(affinity, k=k, dim=-1, sorted=False)
        dispatch = F.one_hot(index, num_classes=S).to(device=x.device, dtype=x.dtype)
        # 3. Process the tokens by each expert and combine
        x_in = torch.einsum(" b e c s, b s d -> b e c d", dispatch, x)
        x_e = [self.experts[e](x_in[:,e]) for e in range(self.num_experts)]
        x_e = torch.stack(x_e, dim=1)
        x_out = torch.einsum('b e c s, b e c, b e c d -> b s d', dispatch, gating, x_e)
        # if self.n_shared_experts >0:
        #     x_out = x_out + self.shared_experts(identity)
        if len(orig_shape) == 4:
            x_out = x_out.view(B, H, W, D)
        return x_out
