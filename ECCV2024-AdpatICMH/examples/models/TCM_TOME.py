from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
from typing import Any, Dict

from einops import rearrange 
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_, DropPath
import numpy as np
import math

from .utils import conv, update_registered_buffers


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x


def _cfg_get(container, key, default=None):
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _tome_do_nothing(x: torch.Tensor, mode: str = None):
    return x


def _init_generator(device: torch.device, fallback: torch.Generator = None):
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    if device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    if fallback is None:
        return _init_generator(torch.device("cpu"))
    return fallback


def _mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    return torch.gather(input, dim, index)


def _bipartite_soft_matching_random2d(metric: torch.Tensor,
                                      w: int, h: int, sx: int, sy: int, r: int,
                                      no_rand: bool = False,
                                      generator: torch.Generator = None):
    B, N, _ = metric.shape

    if r <= 0:
        return _tome_do_nothing, _tome_do_nothing

    gather = _mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        hsy, wsx = h // sy, w // sx
        if hsy <= 0 or wsx <= 0:
            return _tome_do_nothing, _tome_do_nothing

        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(
                sy * sx,
                size=(hsy, wsx, 1),
                device=generator.device,
                generator=generator,
            ).to(metric.device)

        idx_buffer_view = torch.zeros(hsy, wsx, sy * sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)
        del idx_buffer, idx_buffer_view

        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :]
        b_idx = rand_idx[:, :num_dst, :]

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        r = min(a.shape[1], r)
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape
        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(
            dim=-2,
            index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c),
            src=unm,
        )
        out.scatter_(
            dim=-2,
            index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c),
            src=src,
        )
        return out

    return merge, unmerge


class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W',
                 input_resolution=None, use_tome=False, tome_cfg=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.window_size = window_size
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

        tome_cfg = tome_cfg or {}
        self.use_tome = bool(use_tome and tome_cfg.get("enable", False))
        self.tome_eval_only = bool(tome_cfg.get("eval_only", True))
        self.tome_sx = int(tome_cfg.get("sx", 2))
        self.tome_sy = int(tome_cfg.get("sy", 2))
        self.tome_use_rand = bool(tome_cfg.get("use_rand", False))
        self.tome_reduce = str(tome_cfg.get("reduce", "mean"))
        self.tome_merge_attn = bool(tome_cfg.get("merge_attn", True))
        self.tome_merge_mlp = bool(tome_cfg.get("merge_mlp", False))
        # Guardrails: avoid spending more time in merge matching than saved by attention.
        self.tome_max_tokens = int(tome_cfg.get("max_tokens", 8192))
        self.tome_max_pairs = int(tome_cfg.get("max_pairs", 16000000))
        self.tome_generator = None

    def _compute_tome_ops(self, x: torch.Tensor):
        b, h, w, c = x.shape
        n = h * w
        do_nothing = _tome_do_nothing

        if not self.use_tome:
            return do_nothing, do_nothing, do_nothing, do_nothing, h, w
        if self.tome_eval_only and self.training:
            return do_nothing, do_nothing, do_nothing, do_nothing, h, w
        if self.tome_sx <= 1 and self.tome_sy <= 1:
            return do_nothing, do_nothing, do_nothing, do_nothing, h, w
        if self.tome_max_tokens > 0 and n > self.tome_max_tokens:
            return do_nothing, do_nothing, do_nothing, do_nothing, h, w

        h_out, w_out = h // self.tome_sy, w // self.tome_sx
        if h_out <= 0 or w_out <= 0:
            return do_nothing, do_nothing, do_nothing, do_nothing, h, w
        num_dst = h_out * w_out
        r = n - num_dst
        if r <= 0:
            return do_nothing, do_nothing, do_nothing, do_nothing, h, w

        # For WMSA, merged feature map must still match window partition.
        attn_can_merge = ((h_out % self.window_size) == 0 and (w_out % self.window_size) == 0)
        if not self.tome_merge_mlp and (not self.tome_merge_attn or not attn_can_merge):
            return do_nothing, do_nothing, do_nothing, do_nothing, h, w

        num_src = n - num_dst
        if self.tome_max_pairs > 0 and (num_src * num_dst) > self.tome_max_pairs:
            return do_nothing, do_nothing, do_nothing, do_nothing, h, w

        tokens = x.reshape(b, n, c)
        if self.tome_generator is None:
            self.tome_generator = _init_generator(tokens.device)
        elif self.tome_generator.device != tokens.device:
            self.tome_generator = _init_generator(tokens.device, fallback=self.tome_generator)

        use_rand = self.tome_use_rand and (tokens.shape[0] % 2 == 0)
        try:
            merge, unmerge = _bipartite_soft_matching_random2d(
                tokens,
                w=w,
                h=h,
                sx=self.tome_sx,
                sy=self.tome_sy,
                r=r,
                no_rand=not use_rand,
                generator=self.tome_generator,
            )
        except RuntimeError:
            return do_nothing, do_nothing, do_nothing, do_nothing, h, w

        if self.tome_merge_attn and attn_can_merge:
            m_a, u_a = merge, unmerge
            h_a, w_a = h_out, w_out
        else:
            m_a, u_a = do_nothing, do_nothing
            h_a, w_a = h, w

        if self.tome_merge_mlp:
            m_m, u_m = merge, unmerge
        else:
            m_m, u_m = do_nothing, do_nothing

        return m_a, m_m, u_a, u_m, h_a, w_a

    def forward(self, x):
        b, h, w, c = x.shape
        n = h * w
        m_a, m_m, u_a, u_m, h_a, w_a = self._compute_tome_ops(x)

        attn_in = self.ln1(x).reshape(b, n, c)
        attn_in = m_a(attn_in, mode=self.tome_reduce)
        attn_in = attn_in.reshape(b, h_a, w_a, c)
        attn_out = self.msa(attn_in).reshape(b, -1, c)
        attn_out = u_a(attn_out).reshape(b, h, w, c)
        x = x + self.drop_path(attn_out)

        mlp_in = self.ln2(x).reshape(b, n, c)
        mlp_in = m_m(mlp_in, mode=self.tome_reduce)
        mlp_out = self.mlp(mlp_in)
        mlp_out = u_m(mlp_out).reshape(b, h, w, c)
        x = x + self.drop_path(mlp_out)
        return x

class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W',
                 use_tome=False, tome_cfg=None):
        """SwinTransformer + Conv block with optional ToMe acceleration."""
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        assert self.type in ['W', 'SW']

        self.trans_block = Block(
            self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type,
            use_tome=use_tome, tome_cfg=tome_cfg
        )
        self.conv1_1 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.conv_block = ResidualBlock(self.conv_dim, self.conv_dim)

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res
        return x

class SWAtten(AttentionBlock):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, inter_dim=192) -> None:
        if inter_dim is not None:
            super().__init__(N=inter_dim)
            self.non_local_block = SwinBlock(inter_dim, inter_dim, head_dim, window_size, drop_path)
        else:
            super().__init__(N=input_dim)
            self.non_local_block = SwinBlock(input_dim, input_dim, head_dim, window_size, drop_path)
        if inter_dim is not None:
            self.in_conv = conv1x1(input_dim, inter_dim)
            self.out_conv = conv1x1(inter_dim, output_dim)

    def forward(self, x):
        x = self.in_conv(x)
        identity = x
        z = self.non_local_block(x)
        a = self.conv_a(x)
        b = self.conv_b(z)
        out = a * torch.sigmoid(b)
        out += identity
        out = self.out_conv(out)
        return out

class SwinBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path) -> None:
        super().__init__()
        self.block_1 = Block(input_dim, output_dim, head_dim, window_size, drop_path, type='W')
        self.block_2 = Block(input_dim, output_dim, head_dim, window_size, drop_path, type='SW')
        self.window_size = window_size

    def forward(self, x):
        resize = False
        if (x.size(-1) <= self.window_size) or (x.size(-2) <= self.window_size):
            padding_row = (self.window_size - x.size(-2)) // 2
            padding_col = (self.window_size - x.size(-1)) // 2
            x = F.pad(x, (padding_col, padding_col+1, padding_row, padding_row+1))
        trans_x = Rearrange('b c h w -> b h w c')(x)
        trans_x = self.block_1(trans_x)
        trans_x =  self.block_2(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        if resize:
            x = F.pad(x, (-padding_col, -padding_col-1, -padding_row, -padding_row-1))
        return trans_x

class TCM_TOME(CompressionModel):
    def __init__(self, config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0,
                 N=128, M=320, Z=192, num_slices=5, max_support_slices=5, args=None, **kwargs):
        """
        Args:
            N: base channel count (conv_dim = trans_dim = N, main path uses 2*N).
            M: bottleneck (latent) channel count.
            Z: hyper-prior latent channels (entropy_bottleneck size).
            num_slices / max_support_slices: channel-conditional entropy params.
        """
        super().__init__(entropy_bottleneck_channels=Z)
        base_tome_cfg = _cfg_get(args, "tome_config", kwargs.get("tome_config", None))
        if base_tome_cfg is None:
            base_tome_cfg = {}
        base_tome_cfg = dict(base_tome_cfg)
        base_tome_cfg.setdefault("enable", True)
        base_tome_cfg.setdefault("eval_only", True)
        base_tome_cfg.setdefault("sx", 2)
        base_tome_cfg.setdefault("sy", 2)
        base_tome_cfg.setdefault("use_rand", False)
        base_tome_cfg.setdefault("reduce", "mean")
        base_tome_cfg.setdefault("merge_attn", True)
        base_tome_cfg.setdefault("merge_mlp", False)
        base_tome_cfg.setdefault("max_tokens", 8192)
        base_tome_cfg.setdefault("max_pairs", 16000000)

        enc_tome = bool(_cfg_get(args, "enc_tome", kwargs.get("enc_tome", True)))
        dec_tome = bool(_cfg_get(args, "dec_tome", kwargs.get("dec_tome", True)))
        h_tome = bool(_cfg_get(args, "h_tome", kwargs.get("h_tome", False)))

        enc_tome_cfg = dict(base_tome_cfg)
        enc_tome_cfg["enable"] = bool(base_tome_cfg["enable"] and enc_tome)
        dec_tome_cfg = dict(base_tome_cfg)
        dec_tome_cfg["enable"] = bool(base_tome_cfg["enable"] and dec_tome)
        h_tome_cfg = dict(base_tome_cfg)
        h_tome_cfg["enable"] = bool(base_tome_cfg["enable"] and h_tome)

        self.tome_config = base_tome_cfg
        self.enc_tome = enc_tome_cfg["enable"]
        self.dec_tome = dec_tome_cfg["enable"]
        self.h_tome = h_tome_cfg["enable"]

        self.config = config
        self.head_dim = head_dim
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        dim = N
        self.N = N
        self.M = M
        slice_ch = M // num_slices
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        begin = 0

        # ---------- g_a (encoder) ----------
        self.m_down1 = [ConvTransBlock(dim, dim, self.head_dim[0], self.window_size, dpr[i+begin],
                                       'W' if not i%2 else 'SW',
                                       use_tome=self.enc_tome, tome_cfg=enc_tome_cfg)
                      for i in range(config[0])] + \
                      [ResidualBlockWithStride(2*N, 2*N, stride=2)]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim[1], self.window_size, dpr[i+begin],
                                       'W' if not i%2 else 'SW',
                                       use_tome=self.enc_tome, tome_cfg=enc_tome_cfg)
                      for i in range(config[1])] + \
                      [ResidualBlockWithStride(2*N, 2*N, stride=2)]
        self.m_down3 = [ConvTransBlock(dim, dim, self.head_dim[2], self.window_size, dpr[i+begin],
                                       'W' if not i%2 else 'SW',
                                       use_tome=self.enc_tome, tome_cfg=enc_tome_cfg)
                      for i in range(config[2])] + \
                      [conv3x3(2*N, M, stride=2)]

        self.g_a = nn.Sequential(*[ResidualBlockWithStride(3, 2*N, 2)] + self.m_down1 + self.m_down2 + self.m_down3)

        # ---------- g_s (decoder) ----------
        self.m_up1 = [ConvTransBlock(dim, dim, self.head_dim[3], self.window_size, dpr[i+begin],
                                     'W' if not i%2 else 'SW',
                                     use_tome=self.dec_tome, tome_cfg=dec_tome_cfg)
                      for i in range(config[3])] + \
                      [ResidualBlockUpsample(2*N, 2*N, 2)]
        self.m_up2 = [ConvTransBlock(dim, dim, self.head_dim[4], self.window_size, dpr[i+begin],
                                     'W' if not i%2 else 'SW',
                                     use_tome=self.dec_tome, tome_cfg=dec_tome_cfg)
                      for i in range(config[4])] + \
                      [ResidualBlockUpsample(2*N, 2*N, 2)]
        self.m_up3 = [ConvTransBlock(dim, dim, self.head_dim[5], self.window_size, dpr[i+begin],
                                     'W' if not i%2 else 'SW',
                                     use_tome=self.dec_tome, tome_cfg=dec_tome_cfg)
                      for i in range(config[5])] + \
                      [subpel_conv3x3(2*N, 3, 2)]

        self.g_s = nn.Sequential(*[ResidualBlockUpsample(M, 2*N, 2)] + self.m_up1 + self.m_up2 + self.m_up3)

        # ---------- h_a (hyper encoder) ----------
        self.ha_down1 = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW',
                                        use_tome=self.h_tome, tome_cfg=h_tome_cfg)
                      for i in range(config[0])] + \
                      [conv3x3(2*N, Z, stride=2)]

        self.h_a = nn.Sequential(
            *[ResidualBlockWithStride(M, 2*N, 2)] + \
            self.ha_down1
        )

        # ---------- h_mean_s (hyper decoder – mean) ----------
        self.hs_up1 = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW',
                                      use_tome=self.h_tome, tome_cfg=h_tome_cfg)
                      for i in range(config[3])] + \
                      [subpel_conv3x3(2*N, M, 2)]

        self.h_mean_s = nn.Sequential(
            *[ResidualBlockUpsample(Z, 2*N, 2)] + \
            self.hs_up1
        )

        # ---------- h_scale_s (hyper decoder – scale) ----------
        self.hs_up2 = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW',
                                      use_tome=self.h_tome, tome_cfg=h_tome_cfg)
                      for i in range(config[3])] + \
                      [subpel_conv3x3(2*N, M, 2)]

        self.h_scale_s = nn.Sequential(
            *[ResidualBlockUpsample(Z, 2*N, 2)] + \
            self.hs_up2
        )

        # ---------- entropy parameters (attention + channel-conditional) ----------
        self.atten_mean = nn.ModuleList(
            nn.Sequential(
                SWAtten((M + slice_ch*min(i, 5)), (M + slice_ch*min(i, 5)), 16, self.window_size, 0, inter_dim=128)
            ) for i in range(self.num_slices)
        )
        self.atten_scale = nn.ModuleList(
            nn.Sequential(
                SWAtten((M + slice_ch*min(i, 5)), (M + slice_ch*min(i, 5)), 16, self.window_size, 0, inter_dim=128)
            ) for i in range(self.num_slices)
        )
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_ch*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, slice_ch, stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_ch*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, slice_ch, stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_ch*min(i+1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, slice_ch, stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.gaussian_conditional = GaussianConditional(None)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
    
    def forward(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []
        mu_list = []
        scale_list = []
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            mu_list.append(mu)
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            scale_list.append(scale)
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        means = torch.cat(mu_list, dim=1)
        scales = torch.cat(scale_list, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "para":{"means": means, "scales":scales, "y":y}
        }

    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        return super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N=N, M=M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())


            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}
