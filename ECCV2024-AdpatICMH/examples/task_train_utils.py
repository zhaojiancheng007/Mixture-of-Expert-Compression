import logging
import math
import os
import shutil
import sys
from collections import OrderedDict

import lpips
import torch
import torch.nn as nn
import torch.optim as optim

from examples.models.TIC_moe import TIC_MoE
from examples.models.tic import TIC
from examples.models.tinylic import TinyLIC
from examples.models.tinylic_moe import TinyLIC_MoE
from examples.models.tcm import TCM
from examples.models.tcm_moe import TCM_MoE
from examples.models.TCM_TOME import TCM_TOME


MODEL_REGISTRY = {
    "TIC": TIC,
    "TIC_MoE": TIC_MoE,
    "TinyLIC": TinyLIC,
    "TinyLIC_MoE": TinyLIC_MoE,
    "TCM": TCM,
    "TCM_MoE": TCM_MoE,
    "TCM_TOME": TCM_TOME,
}

MODEL_ALIASES = {
    "tic_sfma": "TIC_MoE",
    "tic_moe": "TIC_MoE",
    "tinylic_moe": "TinyLIC_MoE",
    "tcm_moe": "TCM_MoE",
    "tcm_tome": "TCM_TOME",
    "tic": "TIC",
    "tinylic": "TinyLIC",
    "tcm": "TCM",
}

_MOE_MODELS = {"TIC_MoE", "TinyLIC_MoE", "TCM_MoE", "TCM_TOME"}
_TCM_MODELS = {"TCM", "TCM_MoE", "TCM_TOME"}

_DEFAULT_NM = {
    "TIC": (128, 192),
    "TIC_MoE": (128, 192),
    "TinyLIC": (128, 320),
    "TinyLIC_MoE": (128, 320),
    "TCM": (128, 320),
    "TCM_MoE": (128, 320),
    "TCM_TOME": (128, 320),
}

_TCM_DEFAULTS = {
    "tcm_config": [2, 2, 2, 2, 2, 2],
    "head_dim": [8, 16, 32, 32, 16, 8],
    "drop_path_rate": 0,
    "Z": 192,
    "num_slices": 5,
    "max_support_slices": 5,
}

_MOE_PARAM_KEYWORDS = ("moe_mlp.",)
_HYPER_PARAM_PREFIXES = (
    "h_",
    "entropy_bottleneck.",
    "gaussian_conditional.",
    "atten_",
    "cc_",
    "lrp_",
    "sc_transform_",
    "entropy_parameters_",
)


class AverageMeter:
    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = float(val.detach().cpu().item())
        self.val = float(val)
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)


class RateDistortionLoss(nn.Module):
    """Rate-distortion loss for image compression training."""

    def __init__(self, lmbda=1e-2, distortion="mse", eps=1e-9):
        super().__init__()
        self.lmbda = float(lmbda)
        self.eps = float(eps)
        self.distortion = distortion.lower()
        self.mse_fn = nn.MSELoss()

        if self.distortion == "lpips":
            self.lpips_fn = lpips.LPIPS(net="vgg")
            for p in self.lpips_fn.parameters():
                p.requires_grad_(False)
        else:
            self.lpips_fn = None

    @staticmethod
    def psnr(output, target):
        mse = torch.mean((output - target) ** 2)
        if mse.item() == 0:
            return torch.tensor(100.0, device=output.device)
        return 10.0 * torch.log10(1.0 / mse)

    def forward(self, out_net, target):
        n, _, h, w = target.size()
        num_pixels = n * h * w

        bpp = sum(
            (torch.log(likelihoods.clamp(min=self.eps)).sum() / (-math.log(2) * num_pixels))
            for likelihoods in out_net["likelihoods"].values()
        )

        x_hat = out_net["x_hat"]
        mse = self.mse_fn(x_hat, target)
        psnr = self.psnr(torch.clamp(x_hat, 0, 1), target)

        if self.distortion == "lpips":
            lpips_val = self.lpips_fn(
                x_hat.clamp(0, 1) * 2 - 1,
                target * 2 - 1,
            ).mean()
            rdloss = self.lmbda * lpips_val + bpp
            return {
                "rdloss": rdloss,
                "bpp_loss": bpp,
                "mse_loss": mse,
                "lpips_loss": lpips_val,
                "psnr": psnr,
            }

        rdloss = self.lmbda * (255.0 ** 2) * mse + bpp
        return {
            "rdloss": rdloss,
            "bpp_loss": bpp,
            "mse_loss": mse,
            "psnr": psnr,
        }


def setup_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    fmt = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root_logger.addHandler(sh)

    logging.info("Logging file is %s", log_path)


def init_out_dir(args, task_name="task"):
    if getattr(args, "out_dir", None):
        out_dir = args.out_dir
    elif hasattr(args, "root") and hasattr(args, "exp_name"):
        suffix = getattr(args, "quality_level", getattr(args, "lmbda", "default"))
        out_dir = os.path.join(str(args.root), str(args.exp_name), str(suffix))
    else:
        out_dir = os.path.join("./checkpoints", task_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def get_bare_model(model):
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def build_model(args):
    raw_name = str(args.model)
    name = MODEL_ALIASES.get(raw_name, MODEL_ALIASES.get(raw_name.lower(), raw_name))
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    cls = MODEL_REGISTRY[name]
    n = getattr(args, "N", _DEFAULT_NM[name][0])
    m = getattr(args, "M", _DEFAULT_NM[name][1])

    tcm_kwargs = {}
    if name in _TCM_MODELS:
        for key, default in _TCM_DEFAULTS.items():
            tcm_kwargs[key] = getattr(args, key, default)
        if "tcm_config" in tcm_kwargs:
            tcm_kwargs["config"] = tcm_kwargs.pop("tcm_config")

    if name in _MOE_MODELS:
        return cls(N=n, M=m, args=args, **tcm_kwargs)
    return cls(N=n, M=m, **tcm_kwargs)


def _is_moe_param(name):
    return any(kw in name for kw in _MOE_PARAM_KEYWORDS)


def _is_hyper_param(name):
    return name.startswith(_HYPER_PARAM_PREFIXES)


def configure_optimizers(net, args):
    """Build main/aux optimizers with optional train_moe freezing policy."""
    train_moe = getattr(args, "train_moe", False)
    named_params = list(net.named_parameters())

    if train_moe:
        n_frozen = 0
        n_moe = 0
        n_hyper = 0
        for name, param in named_params:
            is_moe = _is_moe_param(name)
            is_hyper = _is_hyper_param(name)
            keep = (is_moe or is_hyper)
            param.requires_grad_(keep)
            if keep:
                if is_moe:
                    n_moe += 1
                if is_hyper:
                    n_hyper += 1
            else:
                n_frozen += 1

        moe_m = sum(p.numel() for n, p in named_params if p.requires_grad and _is_moe_param(n)) / 1e6
        hyper_m = sum(p.numel() for n, p in named_params if p.requires_grad and _is_hyper_param(n)) / 1e6
        trainable_m = sum(p.numel() for _, p in named_params if p.requires_grad) / 1e6
        logging.info(
            "[train_moe] frozen %d tensors, kept %d MoE + %d hyper-prior tensors trainable",
            n_frozen, n_moe, n_hyper,
        )
        logging.info(
            "[train_moe] trainable params: MoE=%.2fM  Hyper=%.2fM  Total=%.2fM",
            moe_m, hyper_m, trainable_m,
        )

    params_dict = dict(named_params)
    parameters = {
        n for n, p in named_params if (not n.endswith(".quantiles")) and p.requires_grad
    }
    aux_parameters = {
        n for n, p in named_params if n.endswith(".quantiles") and p.requires_grad
    }

    logging.info("Main optimizer params: %d", len(parameters))
    logging.info("Aux  optimizer params: %d", len(aux_parameters))

    optimizer = optim.AdamW(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    if aux_parameters:
        aux_optimizer = optim.AdamW(
            (params_dict[n] for n in sorted(aux_parameters)),
            lr=args.aux_learning_rate,
        )
    else:
        aux_optimizer = None
        logging.info("Aux optimizer: None (no trainable aux parameters)")

    return optimizer, aux_optimizer


def save_checkpoint(state, is_best, out_dir, filename="checkpoint.pth.tar"):
    path = os.path.join(out_dir, filename)
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, os.path.join(out_dir, "checkpoint_best_loss.pth.tar"))


def _strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return OrderedDict((k[7:], v) for k, v in state_dict.items())
    return state_dict


def load_pretrained(net, path, device):
    logging.info("Loading pretrained weights from %s", path)
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = _strip_module_prefix(state)
    net.load_state_dict(state, strict=False)


def load_checkpoint(net, optimizer, aux_optimizer, lr_scheduler, path, device):
    logging.info("Resuming from checkpoint %s", path)
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = _strip_module_prefix(state)
    net.load_state_dict(state, strict=False)

    last_epoch = int(ckpt.get("epoch", 0))
    best_loss = float(ckpt.get("loss", float("inf")))

    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if aux_optimizer is not None and "aux_optimizer" in ckpt:
        aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
    if "lr_scheduler" in ckpt:
        try:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        except Exception as exc:
            logging.warning("Skip loading lr_scheduler state: %s", exc)

    return last_epoch, best_loss
