from types import MethodType

from layers.layers_tinylic import NSABlock

from .tinylic import TinyLIC
from .TCM_TOME import _bipartite_soft_matching_random2d, _init_generator, _tome_do_nothing


def _cfg_get(container, key, default=None):
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _compute_tome_ops(module, tokens, h, w):
    cfg = module._tome_cfg
    _, n, _ = tokens.shape
    do_nothing = _tome_do_nothing

    if not cfg["enable"]:
        return do_nothing, do_nothing, do_nothing, do_nothing, h, w
    if cfg["eval_only"] and module.training:
        return do_nothing, do_nothing, do_nothing, do_nothing, h, w
    if cfg["sx"] <= 1 and cfg["sy"] <= 1:
        return do_nothing, do_nothing, do_nothing, do_nothing, h, w
    if cfg["max_tokens"] > 0 and n > int(cfg["max_tokens"]):
        return do_nothing, do_nothing, do_nothing, do_nothing, h, w

    h_out, w_out = h // int(cfg["sy"]), w // int(cfg["sx"])
    if h_out <= 0 or w_out <= 0:
        return do_nothing, do_nothing, do_nothing, do_nothing, h, w
    num_dst = h_out * w_out
    r = n - num_dst
    if r <= 0:
        return do_nothing, do_nothing, do_nothing, do_nothing, h, w

    if (not cfg["merge_mlp"]) and (not cfg["merge_attn"]):
        return do_nothing, do_nothing, do_nothing, do_nothing, h, w

    num_src = n - num_dst
    if cfg["max_pairs"] > 0 and (num_src * num_dst) > int(cfg["max_pairs"]):
        return do_nothing, do_nothing, do_nothing, do_nothing, h, w

    if module._tome_generator is None:
        module._tome_generator = _init_generator(tokens.device)
    elif module._tome_generator.device != tokens.device:
        module._tome_generator = _init_generator(tokens.device, fallback=module._tome_generator)

    use_rand = bool(cfg["use_rand"]) and (tokens.shape[0] % 2 == 0)
    try:
        merge, unmerge = _bipartite_soft_matching_random2d(
            tokens,
            w=w,
            h=h,
            sx=int(cfg["sx"]),
            sy=int(cfg["sy"]),
            r=r,
            no_rand=not use_rand,
            generator=module._tome_generator,
        )
    except RuntimeError:
        return do_nothing, do_nothing, do_nothing, do_nothing, h, w

    if cfg["merge_attn"]:
        m_a, u_a = merge, unmerge
        h_a, w_a = h_out, w_out
    else:
        m_a, u_a = do_nothing, do_nothing
        h_a, w_a = h, w

    if cfg["merge_mlp"]:
        m_m, u_m = merge, unmerge
    else:
        m_m, u_m = do_nothing, do_nothing

    return m_a, m_m, u_a, u_m, h_a, w_a


def _nsa_forward_tome(self, x):
    b, h, w, c = x.shape
    shortcut = x

    norm1 = self.norm1(x).reshape(b, h * w, c)
    m_a, m_m, u_a, u_m, h_a, w_a = _compute_tome_ops(self, norm1, h, w)

    x_attn = m_a(norm1, mode=self._tome_cfg["reduce"]).reshape(b, h_a, w_a, c)
    x_attn = self.attn(x_attn).reshape(b, h_a * w_a, c)
    x_attn = u_a(x_attn).reshape(b, h, w, c)
    x = shortcut + self.drop_path(x_attn)

    mlp_in = self.norm2(x).reshape(b, h * w, c)
    mlp_in = m_m(mlp_in, mode=self._tome_cfg["reduce"])
    mlp_out = self.mlp(mlp_in)
    mlp_out = u_m(mlp_out).reshape(b, h, w, c)
    x = x + self.drop_path(mlp_out)
    return x


class TinyLIC_TOME(TinyLIC):
    """TinyLIC with ToMe acceleration injected into NSABlock."""

    def __init__(self, N=128, M=320, args=None, **kwargs):
        super().__init__(N=N, M=M)

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

        self.tome_config = base_tome_cfg
        self.enc_tome = enc_tome
        self.dec_tome = dec_tome
        self.h_tome = h_tome

        self._apply_tome_patch()

    def _enable_for_block(self, module_name: str) -> bool:
        if module_name.startswith("g_a"):
            return self.enc_tome
        if module_name.startswith("g_s"):
            return self.dec_tome
        if module_name.startswith("h_"):
            return self.h_tome
        return False

    def _apply_tome_patch(self):
        patched = 0
        for name, module in self.named_modules():
            if not isinstance(module, NSABlock):
                continue
            if not self._enable_for_block(name):
                continue

            cfg = dict(self.tome_config)
            cfg["enable"] = True

            if not hasattr(module, "_tome_orig_forward"):
                module._tome_orig_forward = module.forward
            module._tome_cfg = cfg
            module._tome_generator = None
            module.forward = MethodType(_nsa_forward_tome, module)
            patched += 1

        self.tome_patched_blocks = patched
