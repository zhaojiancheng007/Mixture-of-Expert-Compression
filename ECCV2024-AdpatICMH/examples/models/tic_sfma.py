import math
from click import prompt
import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from layers import RSTB
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from numpy import ceil
from .tic import TIC

from compressai.models.utils import conv, deconv, update_registered_buffers

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
from torch import Tensor

def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x

SCALES_LEVELS = 64



def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class Alignment(torch.nn.Module):
    """Image Alignment for model downsample requirement"""

    def __init__(self, divisor=64., mode='pad', padding_mode='replicate'):
        super().__init__()
        self.divisor = float(divisor)
        self.mode = mode
        self.padding_mode = padding_mode
        self._tmp_shape = None

    def extra_repr(self):
        s = 'divisor={divisor}, mode={mode}'
        if self.mode == 'pad':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    @staticmethod
    def _resize(input, size):
        return F.interpolate(input, size, mode='bilinear', align_corners=False)

    def _align(self, input):
        H, W = input.size()[-2:]
        H_ = int(ceil(H / self.divisor) * self.divisor)
        W_ = int(ceil(W / self.divisor) * self.divisor)
        pad_H, pad_W = H_-H, W_-W
        if pad_H == pad_W == 0:
            self._tmp_shape = None
            return input

        self._tmp_shape = input.size()
        if self.mode == 'pad':
            return F.pad(input, (0, pad_W, 0, pad_H), mode=self.padding_mode)
        elif self.mode == 'resize':
            return self._resize(input, size=(H_, W_))

    def _resume(self, input, shape=None):
        if shape is not None:
            self._tmp_shape = shape
        if self._tmp_shape is None:
            return input

        if self.mode == 'pad':
            output = input[..., :self._tmp_shape[-2], :self._tmp_shape[-1]]
        elif self.mode == 'resize':
            output = self._resize(input, size=self._tmp_shape[-2:])

        return output

    def align(self, input):
        """align"""
        if input.dim() == 4:
            return self._align(input)

    def resume(self, input, shape=None):
        """resume"""
        if input.dim() == 4:
            return self._resume(input, shape)

    def forward(self, func, *args, **kwargs):
        pass
    
class SFMA(nn.Module):
    def __init__(self, in_dim=128, middle_dim=64,adapt_factor=1):
        super().__init__()
        self.factor = adapt_factor
        self.s_down1 = nn.Conv2d(in_dim, middle_dim, 1, 1, 0)
        self.s_down2 = nn.Conv2d(in_dim, middle_dim, 1, 1, 0)
        self.s_dw = nn.Conv2d(middle_dim, middle_dim, 5, 1, 2, groups=middle_dim)
        self.s_relu = nn.ReLU(inplace=True)
        self.s_up = nn.Conv2d(middle_dim, in_dim, 1, 1, 0)
       
        self.f_down = nn.Conv2d(in_dim, middle_dim, 1, 1, 0)
        self.f_relu1 = nn.ReLU(inplace=True)
        self.f_relu2 = nn.ReLU(inplace=True)
        self.f_up = nn.Conv2d(middle_dim, in_dim, 1, 1, 0)
        self.f_dw = nn.Conv2d(middle_dim, middle_dim, 3, 1, 1, groups=middle_dim)
        self.f_inter = nn.Conv2d(middle_dim, middle_dim, 1, 1, 0)
        self.sg = nn.Sigmoid()
    
    def forward(self, x):
        '''
        input: 
        x: intermediate feature 
        output:
        x_tilde: adapted feature
        '''
        _, _, H, W = x.shape

        y = torch.fft.rfft2(self.f_down(x), dim=(2, 3), norm='backward')
        y_amp = torch.abs(y)
        y_phs = torch.angle(y)
        # we only modulate the amplitude component for better training stability
        y_amp_modulation = self.f_inter(self.f_relu1(self.f_dw(y_amp)))
        y_amp = y_amp * self.sg(y_amp_modulation)
        y_real = y_amp * torch.cos(y_phs)
        y_img = y_amp * torch.sin(y_phs)
        y = torch.complex(y_real, y_img)
        y = torch.fft.irfft2(y, s=(H, W), norm='backward')
        
        f_modulate = self.f_up(self.f_relu2(y))
        s_modulate = self.s_up(self.s_relu(self.s_dw(self.s_down1(x)) * self.s_down2(x)))
        x_tilde = x + (s_modulate + f_modulate)*self.factor
        return x_tilde 

class TIC_SFMA(nn.Module):
    """
    Modified from TIC (Lu et al., "Transformer-based Image Compression," DCC2022.)
    """
    def __init__(self, N=128, M=192,  input_resolution=(256,256), in_channel=3):
        super().__init__()

        depths = [2, 4, 6, 2, 2, 2]
        num_heads = [8, 8, 8, 16, 16, 16]
        window_size = 8
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        use_checkpoint= False



        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        self.encoder_sfmas = nn.Sequential(
            SFMA(N),
            SFMA(N),
            SFMA(N)
            
        )
        self.decoder_sfmas  = nn.Sequential(
            SFMA(N),
            SFMA(N),
            SFMA(N)
            
        )

        self.g_a0 = conv(in_channel, N, kernel_size=5, stride=2)
        self.g_a1 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//2, input_resolution[1]//2),
                        depth=depths[0],
                        num_heads=num_heads[0],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint
        )
        self.g_a2 = conv(N, N, kernel_size=3, stride=2)
        self.g_a3 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//4, input_resolution[1]//4),
                        depth=depths[1],
                        num_heads=num_heads[1],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_a4 = conv(N, N, kernel_size=3, stride=2)
        self.g_a5 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//8, input_resolution[1]//8),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint      )
        self.g_a6 = conv(N, M, kernel_size=3, stride=2)
        self.g_a7 = RSTB(dim=M,
                        input_resolution=(input_resolution[0]//16, input_resolution[1]//16),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )

        self.h_a0 = conv(M, N, kernel_size=3, stride=2)
        self.h_a1 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//32, input_resolution[1]//32),
                         depth=depths[4],
                         num_heads=num_heads[4],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint     )
        self.h_a2 = conv(N, N, kernel_size=3, stride=2)
        self.h_a3 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//64, input_resolution[1]//64),
                         depth=depths[5],
                         num_heads=num_heads[5],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint        )

        depths = depths[::-1]
        num_heads = num_heads[::-1]
        self.h_s0 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//64, input_resolution[1]//64),
                         depth=depths[0],
                         num_heads=num_heads[0],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint        )
        self.h_s1 = deconv(N, N, kernel_size=3, stride=2)
        self.h_s2 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//32, input_resolution[1]//32),
                         depth=depths[1],
                         num_heads=num_heads[1],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint        )
        self.h_s3 = deconv(N, M*2, kernel_size=3, stride=2)

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        
        self.g_s0 = RSTB(dim=M,
                        input_resolution=(input_resolution[0]//16, input_resolution[1]//16),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s1 = deconv(M, N, kernel_size=3, stride=2)
        self.g_s2 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//8, input_resolution[1]//8),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s3 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s4 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//4, input_resolution[1]//4),
                        depth=depths[4],
                        num_heads=num_heads[4],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s5 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s6 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//2, input_resolution[1]//2),
                        depth=depths[5],
                        num_heads=num_heads[5],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s7 = deconv(N, 3, kernel_size=5, stride=2)
        self.init_std=0.02
      
        self.apply(self._init_weights)  
    def g_a(self, x, x_size=None):
        attns = []
        if x_size is None:
            x_size = x.shape[2:4]
        x = self.g_a0(x)

        x, attn = self.g_a1(x, (x_size[0]//2, x_size[1]//2))
        x =self.encoder_sfmas[0](x)
        attns.append(attn)
        x = self.g_a2(x)

        x, attn = self.g_a3(x, (x_size[0]//4, x_size[1]//4))
        x = self.encoder_sfmas[1](x)
        attns.append(attn)
        x = self.g_a4(x)

        x, attn = self.g_a5(x, (x_size[0]//8, x_size[1]//8))
        x = self.encoder_sfmas[2](x)
        attns.append(attn)
        x = self.g_a6(x)

        x, attn = self.g_a7(x, (x_size[0]//16, x_size[1]//16))
        attns.append(attn)
        return x, attns

    def g_s(self, x, x_size=None):
        attns = []
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x, attn = self.g_s0(x, (x_size[0]//16, x_size[1]//16))
        attns.append(attn)

        x = self.g_s1(x)
        x = self.decoder_sfmas[2](x)
        x, attn = self.g_s2(x, (x_size[0]//8, x_size[1]//8))
        attns.append(attn)


        x = self.g_s3(x)
        x = self.decoder_sfmas[1](x)
        x, attn = self.g_s4(x, (x_size[0]//4, x_size[1]//4))
        attns.append(attn)

        x = self.g_s5(x)
        x = self.decoder_sfmas[0](x)
        x, attn = self.g_s6(x, (x_size[0]//2, x_size[1]//2))
        attns.append(attn)

        x = self.g_s7(x)
        return x, attns

    def h_a(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x = self.h_a0(x)
        x, _ = self.h_a1(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_a2(x)
        x, _ = self.h_a3(x, (x_size[0]//64, x_size[1]//64))
        return x

    def h_s(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*64, x.shape[3]*64)
        x, _ = self.h_s0(x, (x_size[0]//64, x_size[1]//64))
        x = self.h_s1(x)
        x, _ = self.h_s2(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_s3(x)
        return x

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        y, attns_a = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset        
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = ste_round(y-means_hat)+means_hat  
        x_hat, attns_s = self.g_s(y_hat)
     
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "attn_a": attns_a,
            "attn_s": attns_s
        }

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a0.weight"].size(0)
        M = state_dict["g_a6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        x_size = (x.shape[2], x.shape[3])
        y, attns_a = self.g_a(x, x_size)
        z = self.h_a(y, x_size)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat, x_size)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat, attns_s = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class TIC_COCO(nn.Module):
    """
    Modified from TIC (Lu et al., "Transformer-based Image Compression," DCC2022.)
    """
    def __init__(self, N=128, M=192,  input_resolution=(256,256), in_channel=3, mid_dim=32):
        super().__init__()

        depths = [2, 4, 6, 2, 2, 2]
        num_heads = [8, 8, 8, 16, 16, 16]
        window_size = 8
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        use_checkpoint= False

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        self.encoder_sfmas = nn.Sequential(
            SFMA(N,middle_dim=mid_dim),
            SFMA(N,middle_dim=mid_dim),
            SFMA(N,middle_dim=mid_dim),
        )
        self.decoder_sfmas  = nn.Sequential(
            SFMA(N,middle_dim=mid_dim),
            SFMA(N,middle_dim=mid_dim),
            SFMA(N,middle_dim=mid_dim)
        )
        self.task_sfmas  = nn.Sequential(
            SFMA(N,middle_dim=mid_dim),
            SFMA(N,middle_dim=mid_dim),
            SFMA(N,middle_dim=mid_dim),
            SFMA(N,middle_dim=mid_dim),
            SFMA(N,middle_dim=mid_dim),
            SFMA(N,middle_dim=mid_dim),
        )
        self.task_heads_sfmas = nn.ModuleDict({
            "detect": nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(N, 3, kernel_size=3, padding=1),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(N//2, 3, kernel_size=3, padding=1)
            ),
            "semseg": nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(N, 3, kernel_size=3, padding=1),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(N//2, 3, kernel_size=3, padding=1)
            ),
        })
        self.task_gate_sfmas = nn.ParameterDict({
            "detect": nn.Sequential(
            nn.Conv2d(N * 3, N, kernel_size=1),
            nn.Sigmoid()),
            "semseg":nn.Sequential(
            nn.Conv2d(N * 3, N, kernel_size=1),
            nn.Sigmoid()),
        })

        self.g_a0 = conv(in_channel, N, kernel_size=5, stride=2)
        self.g_a1 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//2, input_resolution[1]//2),
                        depth=depths[0],
                        num_heads=num_heads[0],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint
        )
        self.g_a2 = conv(N, N, kernel_size=3, stride=2)
        self.g_a3 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//4, input_resolution[1]//4),
                        depth=depths[1],
                        num_heads=num_heads[1],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_a4 = conv(N, N, kernel_size=3, stride=2)
        self.g_a5 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//8, input_resolution[1]//8),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint      )
        self.g_a6 = conv(N, M, kernel_size=3, stride=2)
        self.g_a7 = RSTB(dim=M,
                        input_resolution=(input_resolution[0]//16, input_resolution[1]//16),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )

        self.h_a0 = conv(M, N, kernel_size=3, stride=2)
        self.h_a1 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//32, input_resolution[1]//32),
                         depth=depths[4],
                         num_heads=num_heads[4],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint     )
        self.h_a2 = conv(N, N, kernel_size=3, stride=2)
        self.h_a3 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//64, input_resolution[1]//64),
                         depth=depths[5],
                         num_heads=num_heads[5],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint        )

        depths = depths[::-1]
        num_heads = num_heads[::-1]
        self.h_s0 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//64, input_resolution[1]//64),
                         depth=depths[0],
                         num_heads=num_heads[0],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint        )
        self.h_s1 = deconv(N, N, kernel_size=3, stride=2)
        self.h_s2 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//32, input_resolution[1]//32),
                         depth=depths[1],
                         num_heads=num_heads[1],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint        )
        self.h_s3 = deconv(N, M*2, kernel_size=3, stride=2)

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        
        self.g_s0 = RSTB(dim=M,
                        input_resolution=(input_resolution[0]//16, input_resolution[1]//16),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s1 = deconv(M, N, kernel_size=3, stride=2)
        self.g_s2 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//8, input_resolution[1]//8),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s3 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s4 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//4, input_resolution[1]//4),
                        depth=depths[4],
                        num_heads=num_heads[4],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s5 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s6 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//2, input_resolution[1]//2),
                        depth=depths[5],
                        num_heads=num_heads[5],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s7 = deconv(N, 3, kernel_size=5, stride=2)
        self.init_std=0.02
      
        self.apply(self._init_weights)  
    def g_a(self, x, x_size=None):
        attns = []
        if x_size is None:
            x_size = x.shape[2:4]
        x = self.g_a0(x)

        x, attn = self.g_a1(x, (x_size[0]//2, x_size[1]//2))
        x =self.encoder_sfmas[0](x)
        attns.append(attn)
        x = self.g_a2(x)

        x, attn = self.g_a3(x, (x_size[0]//4, x_size[1]//4))
        x = self.encoder_sfmas[1](x)
        attns.append(attn)
        x = self.g_a4(x)

        x, attn = self.g_a5(x, (x_size[0]//8, x_size[1]//8))
        x = self.encoder_sfmas[2](x)
        attns.append(attn)
        x = self.g_a6(x)

        x, attn = self.g_a7(x, (x_size[0]//16, x_size[1]//16))
        attns.append(attn)
        return x, attns

    def g_s(self, x, x_size=None):
        attns = []
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x, attn = self.g_s0(x, (x_size[0]//16, x_size[1]//16))
        attns.append(attn)

        x = self.g_s1(x)
        x = self.decoder_sfmas[2](x)

        detect_stage4 = self.task_sfmas[0](x) # H/16
        semseg_stage4 = self.task_sfmas[1](x)

        x, attn = self.g_s2(x, (x_size[0]//8, x_size[1]//8))
        attns.append(attn)


        x = self.g_s3(x)
        x = self.decoder_sfmas[1](x)

        detect_stage3 = self.task_sfmas[2](x) #H/8
        semseg_stage3 = self.task_sfmas[3](x)

        x, attn = self.g_s4(x, (x_size[0]//4, x_size[1]//4))
        attns.append(attn)

        x = self.g_s5(x)
        x = self.decoder_sfmas[0](x)

        detect_stage2 = self.task_sfmas[4](x) # H/4
        semseg_stage2 = self.task_sfmas[5](x)

        x, attn = self.g_s6(x, (x_size[0]//2, x_size[1]//2))
        attns.append(attn)

        # combine seg
        detect_fused = self.fuse(detect_stage4, detect_stage3, detect_stage2,'detect')
        detect_residual = self.task_heads_sfmas["detect"](detect_fused) 
        

        # combine human
        semseg_fused = self.fuse(semseg_stage4, semseg_stage3, semseg_stage2,'semseg')
        semseg_residual = self.task_heads_sfmas["semseg"](semseg_fused) 

        x = self.g_s7(x)

        return x, detect_residual, semseg_residual

    def fuse(self, f_s4, f_s3, f_s2, task):
        # f_s4 --> shape of f_s2
        features = [f_s4, f_s3, f_s2]

        target_size = f_s2.shape[2:]
        upsampled_features = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
                     for f in features]
        concat = torch.cat(upsampled_features, dim=1)  # [B, C*num_scales, H, W]
        gate = self.task_gate_sfmas[task](concat)            # [B, C, H, W]
        gate = gate.unsqueeze(1)            # [B, num_scales, 1, H, W]
        stacked = torch.stack(upsampled_features, dim=1)  # [B, num_scales, C, H, W]
        fused = (gate * stacked).sum(dim=1)     # [B, C, H, W]
        return fused
    
    def h_a(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x = self.h_a0(x)
        x, _ = self.h_a1(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_a2(x)
        x, _ = self.h_a3(x, (x_size[0]//64, x_size[1]//64))
        return x

    def h_s(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*64, x.shape[3]*64)
        x, _ = self.h_s0(x, (x_size[0]//64, x_size[1]//64))
        x = self.h_s1(x)
        x, _ = self.h_s2(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_s3(x)
        return x

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        y, attns_a = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset        
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = ste_round(y-means_hat)+means_hat  
        x_hat, detect_residual, semseg_residual = self.g_s(y_hat)
     
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "detect": x_hat + detect_residual,
            "semseg": x_hat + semseg_residual,
        }

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a0.weight"].size(0)
        M = state_dict["g_a6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        x_size = (x.shape[2], x.shape[3])
        y, attns_a = self.g_a(x, x_size)
        z = self.h_a(y, x_size)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat, x_size)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat, attns_s = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class TaskPromptTuning(nn.Module):
    def __init__(self, num_tasks: int, prompt_dim: int, feat_dim: int):
        """
        Args:
            num_tasks: 任务数 N
            prompt_dim: 每个任务的 prompt 向量维度
            feat_dim: 特征通道数 C (即 latent feature 的 channel 数)
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.prompt_dim = prompt_dim
        self.feat_dim = feat_dim

        # 每个任务有一个可学习的 prompt embedding
        self.prompts = nn.Parameter(torch.randn(num_tasks, prompt_dim))

        # 一个小 MLP，把 GAP + prompt 映射到 gamma 和 beta
        hidden_dim = max(64, prompt_dim // 2)  # 轻量隐藏层
        self.mlp = nn.Sequential(
            nn.Linear(prompt_dim + feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * feat_dim)  # 输出 gamma 和 beta
        )

    def forward(self, x, task_id: int):
        """
        Args:
            x: 输入特征图, shape [B, C, H, W]
            task_id: 当前任务 id, int in [0, num_tasks-1]
        Return:
            x_tuned: 调制后的特征图, shape [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 全局平均池化 -> [B, C]
        gap = F.adaptive_avg_pool2d(x, 1).view(B, C)

        # 取出该任务的 prompt 向量 -> [prompt_dim]
        task_prompt = self.prompts[task_id].unsqueeze(0).expand(B, -1)

        # 拼接 GAP 和 prompt
        combined = torch.cat([gap, task_prompt], dim=-1)  # [B, feat_dim + prompt_dim]

        # 生成 gamma 和 beta
        gamma_beta = self.mlp(combined)  # [B, 2*C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # 各自 [B, C]

        # reshape 成 [B, C, 1, 1] 方便广播
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)

        # FiLM 调制
        x_tuned = gamma * x + beta
        return x_tuned

class Router(nn.Module):
    def __init__(self, feat_dim: int, prompt_dim: int, num_experts: int, hidden_dim: int = 128):
        super().__init__()
        self.num_experts = num_experts
        self.fc = nn.Sequential(
            nn.Linear(feat_dim + prompt_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x, prompt, tau=1.0, hard=False):
        """
        Args:
            x: feature map, [B, C, H, W]
            prompt: task prompt vector, [B, prompt_dim]
            tau: Gumbel-Softmax temperature
            hard: 是否做hard top-1
        Return:
            gate: [B, num_experts], soft/hard 权重
        """
        B, C, H, W = x.shape
        gap = F.adaptive_avg_pool2d(x, 1).view(B, C)  # [B, C]
        combined = torch.cat([gap, prompt], dim=-1)   # [B, C+prompt_dim]
        logits = self.fc(combined)                    # [B, num_experts]

        gate = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)  # [B, num_experts]
        return gate

class Expert(nn.Module):
    def __init__(self, in_dim=128, middle_dim=32,adapt_factor=1):
        super().__init__()
        self.factor = adapt_factor
        self.s_down1 = nn.Conv2d(in_dim, middle_dim, 1, 1, 0)
        self.s_down2 = nn.Conv2d(in_dim, middle_dim, 1, 1, 0)
        self.s_dw = nn.Conv2d(middle_dim, middle_dim, 5, 1, 2, groups=middle_dim)
        self.s_relu = nn.ReLU(inplace=True)
        self.s_up = nn.Conv2d(middle_dim, in_dim, 1, 1, 0)
       
        self.f_down = nn.Conv2d(in_dim, middle_dim, 1, 1, 0)
        self.f_relu1 = nn.ReLU(inplace=True)
        self.f_relu2 = nn.ReLU(inplace=True)
        self.f_up = nn.Conv2d(middle_dim, in_dim, 1, 1, 0)
        self.f_dw = nn.Conv2d(middle_dim, middle_dim, 3, 1, 1, groups=middle_dim)
        self.f_inter = nn.Conv2d(middle_dim, middle_dim, 1, 1, 0)
        self.sg = nn.Sigmoid()
    
    def forward(self, x):
        '''
        input: 
        x: intermediate feature 
        output:
        x_tilde: adapted feature
        '''
        _, _, H, W = x.shape

        y = torch.fft.rfft2(self.f_down(x), dim=(2, 3), norm='backward')
        y_amp = torch.abs(y)
        y_phs = torch.angle(y)
        # we only modulate the amplitude component for better training stability
        y_amp_modulation = self.f_inter(self.f_relu1(self.f_dw(y_amp)))
        y_amp = y_amp * self.sg(y_amp_modulation)
        y_real = y_amp * torch.cos(y_phs)
        y_img = y_amp * torch.sin(y_phs)
        y = torch.complex(y_real, y_img)
        y = torch.fft.irfft2(y, s=(H, W), norm='backward')
        
        f_modulate = self.f_up(self.f_relu2(y))
        s_modulate = self.s_up(self.s_relu(self.s_dw(self.s_down1(x)) * self.s_down2(x)))
        x_tilde = x + (s_modulate + f_modulate)*self.factor
        return x_tilde 

class MoE(nn.Module):
    def __init__(self, feat_dim: int, prompt_dim: int, num_experts: int = 4):
        super().__init__()
        self.router = Router(feat_dim, prompt_dim, num_experts)
        self.experts = nn.ModuleList([Expert(feat_dim) for _ in range(num_experts)])

    def forward(self, x, prompt, tau=1.0, hard=False):
        """
        Args:
            x: feature map, [B, C, H, W]
            prompt: [B, prompt_dim]
        Return:
            out: [B, C, H, W] task-adapted feature
        """
        B, C, H, W = x.shape
        gate = self.router(x, prompt, tau=tau, hard=hard)  # [B, num_experts]

        expert_outs = []
        for expert in self.experts:
            expert_outs.append(expert(x).unsqueeze(1))  # [B,1,C,H,W]
        expert_outs = torch.cat(expert_outs, dim=1)     # [B,num_experts,C,H,W]

        gate = gate.view(B, -1, 1, 1, 1)                # [B,num_experts,1,1,1]
        out = (gate * expert_outs).sum(dim=1)           # [B,C,H,W]
        return out

class TIC_MoE(TIC):
    def __init__(self, N=128, M=192, prompt_dim=32, num_experts=4, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        # 在解码阶段插入 MoE
        self.moe0 = MoE(M, prompt_dim, num_experts)
        self.moe1 = MoE(N, prompt_dim, num_experts)
        self.moe2 = MoE(N, prompt_dim, num_experts)
        self.moe3 = MoE(N, prompt_dim, num_experts)

    def g_s(self, x, prompt, x_size=None, tau=1.0, hard=False):
        attns = []
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)

        # stage 0
        x, attn = self.g_s0(x, (x_size[0]//16, x_size[1]//16))
        attns.append(attn)
        x = self.moe0(x, prompt, tau, hard)

        # stage 1
        x = self.g_s1(x)
        x, attn = self.g_s2(x, (x_size[0]//8, x_size[1]//8))
        attns.append(attn)
        x = self.moe1(x, prompt, tau, hard)

        # stage 2
        x = self.g_s3(x)
        x, attn = self.g_s4(x, (x_size[0]//4, x_size[1]//4))
        attns.append(attn)
        x = self.moe2(x, prompt, tau, hard)

        # stage 3
        x = self.g_s5(x)
        x, attn = self.g_s6(x, (x_size[0]//2, x_size[1]//2))
        attns.append(attn)
        x = self.moe3(x, prompt, tau, hard)

        # 最后一层不加 MoE
        x = self.g_s7(x)
        return x, attns

    def forward(self, x, prompt, tau=1.0, hard=False):
        y, attns_a = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset
        
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = ste_round(y-means_hat)+means_hat
        x_hat, attns_s = self.g_s(y_hat, prompt, tau=tau, hard=hard)
     
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "attn_a": attns_a,
            "attn_s": attns_s
        }