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

"""
MoE-aware TinyLIC layers.

Extends layers_tinylic with optional Mixture-of-Experts on the FFN
inside NSABlock, following the same pattern used in moe_layers.py
for the TIC model.
"""

import torch
import torch.nn as nn
from torch import Tensor

from timm.models.layers import DropPath

from .natten import NeighborhoodAttention
from .layers_tinylic import (
    MultistageMaskedConv2d,
    Mlp,
)
from .moe_layers import SparseMoEBlock, ChannelMoEBlock


__all__ = [
    "ResViTBlock",
    "MultistageMaskedConv2d",
]


class NSABlock(nn.Module):
    """Neighborhood Self-Attention Block with optional MoE on the FFN.

    When use_moe=True, the standard Mlp is replaced by a SparseMoEBlock
    whose experts are smaller Mlp instances (hidden dim reduced by
    moe_config['hid_ratio']).
    """

    def __init__(self, dim, num_heads, kernel_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_moe=False, moe_config=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_moe = use_moe

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(
            dim, kernel_size=kernel_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        if use_moe and moe_config is not None:
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
            else:
                self.moe_mlp = SparseMoEBlock(
                    experts=[
                        Mlp(in_features=dim,
                            hidden_features=mlp_hidden_dim // moe_config['hid_ratio'],
                            act_layer=act_layer, drop=drop)
                        for _ in range(moe_config['num_experts'])
                    ],
                    hidden_dim=dim,
                    num_experts=moe_config['num_experts'],
                    capacity=moe_config['capacity'],
                    n_shared_experts=moe_config['n_shared_experts'],
                    use_prompt=False,
                )
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                           act_layer=act_layer, drop=drop)

    def forward(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            x: (B, H, W, C)
        """
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)

        if self.use_moe:
            # SparseMoEBlock expects (B, S, D) -- flatten spatial dims
            B, H, W, C = x.shape
            x_flat = x.reshape(B, H * W, C)
            x_flat = x_flat + self.drop_path(self.moe_mlp(self.norm2(x_flat)))
            x = x_flat.reshape(B, H, W, C)
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicViTLayer(nn.Module):
    """Stack of NSABlocks with optional MoE."""

    def __init__(self, dim, depth, num_heads, kernel_size, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 use_moe=False, moe_config=None):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            NSABlock(
                dim=dim,
                num_heads=num_heads, kernel_size=kernel_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_moe=use_moe, moe_config=moe_config)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class ResViTBlock(nn.Module):
    """Residual Vision Transformer Block with optional MoE.

    Wraps a BasicViTLayer with a residual connection.
    """

    def __init__(self, dim, depth, num_heads, kernel_size=7, mlp_ratio=4,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.2, norm_layer=nn.LayerNorm,
                 use_moe=False, moe_config=None):
        super(ResViTBlock, self).__init__()
        self.dim = dim

        self.residual_group = BasicViTLayer(
            dim=dim, depth=depth, num_heads=num_heads,
            kernel_size=kernel_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=drop_path_rate, norm_layer=norm_layer,
            use_moe=use_moe, moe_config=moe_config)

    def forward(self, x):
        # x: (B, C, H, W) -> permute to (B, H, W, C) for ViT blocks
        return self.residual_group(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + x
