"""
Neighborhood Attention -- compatibility shim.

Uses the pip-installed ``natten`` package (>= 0.17) instead of the legacy
local CUDA JIT kernels, which fail to compile with PyTorch >= 2.x.

The only user-facing export is ``NeighborhoodAttention`` with the same
constructor signature as the old local version::

    NeighborhoodAttention(dim, kernel_size, num_heads,
                          qkv_bias=True, qk_scale=None,
                          attn_drop=0., proj_drop=0.)

Internally this delegates to ``natten.NeighborhoodAttention2D`` with
``rel_pos_bias=True`` (the old code always used RPB).
"""

from natten import NeighborhoodAttention2D as _NA2D
from natten import NeighborhoodAttention1D


class NeighborhoodAttention(_NA2D):
    """Drop-in replacement for the old local NeighborhoodAttention.

    Preserves the original positional-argument order
    ``(dim, kernel_size, num_heads, ...)`` so that existing call sites keep
    working, and forces ``rel_pos_bias=True`` to match the old behaviour
    (which unconditionally registered an ``rpb`` parameter).
    """

    def __init__(self, dim, kernel_size=7, num_heads=1,
                 qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            rel_pos_bias=True,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
