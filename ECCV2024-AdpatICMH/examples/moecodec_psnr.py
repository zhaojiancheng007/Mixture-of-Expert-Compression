# moecodec_psnr.py
# Single-quality (single-bpp) training / evaluation script.
# Supports: TIC, TIC_MoE, TinyLIC, TinyLIC_MoE, TCM, TCM_MoE
# Supports: DDP multi-GPU training via `torchrun`.

import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime
from collections import OrderedDict

# ---- path setup ----
# The script lives in  ECCV2024-AdpatICMH/examples/
# Models use  `from layers.xxx`   → needs examples/ on sys.path
# Script uses `from examples.xxx` → needs ECCV2024-AdpatICMH/ on sys.path
# We also need `utils/`           → already inside examples/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../examples
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)                       # .../ECCV2024-AdpatICMH
for _p in (_PROJECT_DIR, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.utils import save_image

from compressai.datasets import ImageFolder
from utils.dataloader import Kodak

from examples.models.TIC_moe import TIC_MoE
from examples.models.tic import TIC
from examples.models.tinylic import TinyLIC
from examples.models.tinylic_moe import TinyLIC_MoE
from examples.models.tcm import TCM
from examples.models.tcm_moe import TCM_MoE
from examples.models.TCM_TOME import TCM_TOME
from examples.layers.moe_layers import ChannelMoEBlock, PatchMoEBlock

import lpips
import tqdm
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# DDP helpers
# -------------------------
def setup_ddp():
    """Initialise the distributed process group.

    Expects environment variables set by ``torchrun``:
      LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.

    Returns (local_rank, global_rank, world_size).
    If the env vars are absent (single-GPU run), returns (0, 0, 1)
    and does **not** initialise a process group.
    """
    if "RANK" not in os.environ:
        # Not launched with torchrun – single-GPU fallback
        return 0, 0, 1

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Return True on rank-0 (or when DDP is not active)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_bare_model(model):
    """Unwrap DDP / DataParallel wrapper to get the underlying model."""
    if isinstance(model, (DDP, nn.DataParallel)):
        return model.module
    return model


def collect_sparse_moe_smooth_loss(model):
    """Average cached smoothness loss over all SparseMoEBlock modules."""
    total = None
    n_blocks = 0
    for mod in model.modules():
        if type(mod).__name__ != "SparseMoEBlock":
            continue
        loss = getattr(mod, "_last_smooth_loss", None)
        if loss is None:
            continue
        total = loss if total is None else (total + loss)
        n_blocks += 1

    if total is None:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device), 0
    return total / max(n_blocks, 1), n_blocks


# -------------------------
# model registry
# -------------------------
MODEL_REGISTRY = {
    "TIC":         TIC,
    "TIC_MoE":     TIC_MoE,
    "TinyLIC":     TinyLIC,
    "TinyLIC_MoE": TinyLIC_MoE,
    "TCM":         TCM,
    "TCM_MoE":     TCM_MoE,
    "TCM_TOME":    TCM_TOME,
}

# Models that require `args` for MoE / prompt configuration
_MOE_MODELS = {"TIC_MoE", "TinyLIC_MoE", "TCM_MoE", "TCM_TOME"}

# Models in the TCM family (accept extra constructor params)
_TCM_MODELS = {"TCM", "TCM_MoE", "TCM_TOME"}

# Default (N, M) per model family
_DEFAULT_NM = {
    "TIC":         (128, 192),
    "TIC_MoE":     (128, 192),
    "TinyLIC":     (128, 320),
    "TinyLIC_MoE": (128, 320),
    "TCM":         (128, 320),
    "TCM_MoE":     (128, 320),
    "TCM_TOME":    (128, 320),
}

# TCM-specific constructor defaults
# NOTE: ``tcm_config`` → maps to the ``config`` kwarg of TCM.__init__
#       (we use a different name to avoid collision with the ``--config`` CLI arg)
_TCM_DEFAULTS = {
    "tcm_config":          [2, 2, 2, 2, 2, 2],
    "head_dim":            [8, 16, 32, 32, 16, 8],
    "drop_path_rate":      0,
    "Z":                   192,   # hyper-prior latent channels
    "num_slices":          5,
    "max_support_slices":  5,
}


def build_model(args):
    """Instantiate a compression model from config.

    Reads ``args.model`` (str) and dispatches to the correct constructor.
    MoE variants additionally receive the whole ``args`` namespace so they
    can read ``enc_moe``, ``dec_moe``, ``moe_config``.
    TCM / TCM_MoE additionally accept ``config``, ``head_dim``,
    ``drop_path_rate``, ``num_slices``, ``max_support_slices``.
    """
    name = args.model
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    cls = MODEL_REGISTRY[name]
    N = getattr(args, "N", _DEFAULT_NM[name][0])
    M = getattr(args, "M", _DEFAULT_NM[name][1])

    # TCM family needs extra constructor params
    tcm_kwargs = {}
    if name in _TCM_MODELS:
        for key, default in _TCM_DEFAULTS.items():
            tcm_kwargs[key] = getattr(args, key, default)
        # rename tcm_config → config to match the TCM constructor signature
        if "tcm_config" in tcm_kwargs:
            tcm_kwargs["config"] = tcm_kwargs.pop("tcm_config")

    if name in _MOE_MODELS:
        net = cls(N=N, M=M, args=args, **tcm_kwargs)
    else:
        net = cls(N=N, M=M, **tcm_kwargs)

    return net


# -------------------------
# utils
# -------------------------
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


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access module methods like aux_loss/update.
    (Kept for backward compatibility; DDP uses get_bare_model() instead.)
    """
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def setup_logger(log_path: str):
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


def init_out_dir(args):
    base_dir = args.out_dir
    os.makedirs(base_dir, exist_ok=True)
    run_dir = os.path.join(base_dir, args.model, str(args.lmbda))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# -------------------------
# loss (single lambda RD)
# -------------------------
class RateDistortionLoss(nn.Module):
    """
    Rate-Distortion loss supporting two distortion modes:

    * ``"mse"``   – rdloss = lmbda * 255^2 * MSE(x_hat, x) + BPP
    * ``"lpips"`` – rdloss = lmbda * LPIPS(x_hat, x) + BPP

    ``x`` is assumed to be in [0, 1].
    """

    def __init__(self, lmbda: float = 1e-2, distortion: str = "mse",
                 eps: float = 1e-9):
        super().__init__()
        self.lmbda = float(lmbda)
        self.eps = float(eps)
        self.distortion = distortion.lower()

        self.mse_fn = nn.MSELoss()

        if self.distortion == "lpips":
            # VGG backbone, pretrained, eval-only (no grad on LPIPS net)
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
        N, _, H, W = target.size()
        num_pixels = N * H * W

        bpp = sum(
            (torch.log(likelihoods.clamp(min=self.eps)).sum() / (-math.log(2) * num_pixels))
            for likelihoods in out_net["likelihoods"].values()
        )

        x_hat = out_net["x_hat"]
        mse = self.mse_fn(x_hat, target)
        psnr = self.psnr(torch.clamp(x_hat, 0, 1), target)

        if self.distortion == "lpips":
            # lpips expects input in [-1, 1]
            lpips_val = self.lpips_fn(
                x_hat.clamp(0, 1) * 2 - 1,
                target * 2 - 1,
            ).mean()
            rdloss = self.lmbda * lpips_val + bpp
            return {
                "rdloss": rdloss, "bpp_loss": bpp,
                "mse_loss": mse, "lpips_loss": lpips_val,
                "psnr": psnr,
            }
        else:
            rdloss = self.lmbda * (255.0 ** 2) * mse + bpp
            return {
                "rdloss": rdloss, "bpp_loss": bpp,
                "mse_loss": mse, "psnr": psnr,
            }


# -------------------------
# optimizers
# -------------------------

# Substrings that identify MoE-specific parameters.
# Any parameter whose name contains one of these is considered a MoE parameter.
_MOE_PARAM_KEYWORDS = ("moe_mlp.",)


def _is_moe_param(name: str) -> bool:
    """Return True if *name* belongs to a MoE module."""
    return any(kw in name for kw in _MOE_PARAM_KEYWORDS)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.

    *net* should be the **bare** (unwrapped) model so that parameter names
    do not carry a ``module.`` prefix.

    Supported training modes:
      1) ``all``      - train all parameters.
      2) ``moe_only`` - train only MoE-related parameters.
    """
    train_mode = str(getattr(args, "train_mode", "all")).lower()
    if train_mode not in {"all", "moe_only"}:
        raise ValueError(f"Invalid train_mode '{train_mode}'. Choose from: ['all', 'moe_only']")

    named_params = list(net.named_parameters())

    # ---- set trainable parameters according to mode ----
    if train_mode == "all":
        for _, param in named_params:
            param.requires_grad_(True)
        if is_main_process():
            trainable_m = sum(p.numel() for _, p in named_params if p.requires_grad) / 1e6
            logging.info("[train_mode=all] trainable params: Total=%.2fM", trainable_m)
    else:  # moe_only
        n_frozen = 0
        n_moe = 0
        for name, param in named_params:
            is_moe = _is_moe_param(name)
            keep = is_moe
            param.requires_grad_(keep)
            if keep:
                n_moe += 1
            else:
                n_frozen += 1
        if is_main_process():
            moe_m = sum(p.numel() for n, p in named_params if p.requires_grad and _is_moe_param(n)) / 1e6
            trainable_m = sum(p.numel() for _, p in named_params if p.requires_grad) / 1e6
            logging.info(
                "[train_mode=moe_only] frozen %d tensors, kept %d MoE tensors trainable",
                n_frozen, n_moe
            )
            logging.info(
                "[train_mode=moe_only] trainable params: MoE=%.2fM  Total=%.2fM",
                moe_m, trainable_m
            )

    # ---- collect parameter sets ----
    parameters = {
        n for n, p in named_params
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n for n, p in named_params
        if n.endswith(".quantiles") and p.requires_grad
    }

    params_dict = dict(named_params)

    if is_main_process():
        logging.info("Main optimizer params: %d", len(parameters))
        logging.info("Aux  optimizer params: %d", len(aux_parameters))

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    if aux_parameters:
        aux_optimizer = optim.Adam(
            (params_dict[n] for n in sorted(aux_parameters)),
            lr=args.aux_learning_rate,
        )
    else:
        aux_optimizer = None
        if is_main_process():
            logging.info("Aux optimizer: None (no trainable aux parameters)")

    return optimizer, aux_optimizer


# -------------------------
# train / eval
# -------------------------
def train_one_epoch(model, criterion, train_loader, optimizer, aux_optimizer,
                    lr_scheduler, epoch, clip_max_norm, train_sampler=None,
                    moe_smooth_lambda=0.0):
    model.train()
    device = next(model.parameters()).device
    use_lpips = (criterion.distortion == "lpips")
    bare_model = get_bare_model(model)

    # DDP: set epoch on sampler so data is reshuffled each epoch
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    rdl_m = AverageMeter()
    obj_m = AverageMeter()
    bpp_m = AverageMeter()
    mse_m = AverageMeter()
    smooth_m = AverageMeter()
    aux_m = AverageMeter()
    psnr_m = AverageMeter()
    lpips_m = AverageMeter() if use_lpips else None

    total_fwd_time = 0.0   # seconds spent in model forward
    total_iter = 0

    for it, x in enumerate(train_loader):
        x = x.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_net = model(x)
        torch.cuda.synchronize()
        total_fwd_time += time.perf_counter() - t0
        total_iter += 1

        out = criterion(out_net, x)
        smooth_loss = torch.tensor(0.0, device=device)
        if moe_smooth_lambda > 0:
            smooth_loss, _ = collect_sparse_moe_smooth_loss(bare_model)
        total_loss = out["rdloss"] + float(moe_smooth_lambda) * smooth_loss
        total_loss.backward()

        if aux_optimizer is not None:
            aux_loss = bare_model.aux_loss()
            aux_optimizer.zero_grad(set_to_none=True)
            aux_loss.backward()
        else:
            aux_loss = torch.tensor(0.0, device=device)

        if clip_max_norm and clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

        optimizer.step()
        if aux_optimizer is not None:
            aux_optimizer.step()
        # Update LR per optimization step (finer granularity than per-epoch).
        lr_scheduler.step()

        bs = x.size(0)
        rdl_m.update(out["rdloss"], bs)
        obj_m.update(total_loss, bs)
        bpp_m.update(out["bpp_loss"], bs)
        mse_m.update(out["mse_loss"], bs)
        smooth_m.update(smooth_loss, bs)
        aux_m.update(aux_loss, bs)
        psnr_m.update(out["psnr"], bs)
        if use_lpips:
            lpips_m.update(out["lpips_loss"], bs)

    # end-of-epoch summary (single line)
    avg_fwd_ms = (total_fwd_time / max(total_iter, 1)) * 1000
    if is_main_process():
        msg = (
            f"[Train][E{epoch}] "
            f"OBJ {obj_m.avg:.4f}  RD {rdl_m.avg:.4f}  BPP {bpp_m.avg:.4f}  "
            f"MSE {mse_m.avg:.6f}  PSNR {psnr_m.avg:.2f} dB"
        )
        if moe_smooth_lambda > 0:
            msg += f"  SMOOTH {smooth_m.avg:.6f} (x{moe_smooth_lambda:g})"
        if use_lpips:
            msg += f"  LPIPS {lpips_m.avg:.4f}"
        if aux_optimizer is not None:
            msg += f"  AUX {aux_m.avg:.1f}"
        msg += f"  |  fwd {avg_fwd_ms:.1f} ms/batch  ({total_iter} iters)"
        logging.info(msg)

    stats = {
        "obj": obj_m.avg, "rd": rdl_m.avg, "bpp": bpp_m.avg,
        "mse": mse_m.avg, "psnr": psnr_m.avg, "aux": aux_m.avg,
        "moe_smooth": smooth_m.avg,
        "fwd_ms": avg_fwd_ms,
    }
    if use_lpips:
        stats["lpips"] = lpips_m.avg
    return stats


@torch.no_grad()
def test_epoch(model, criterion, test_loader, epoch, tag="Val", save_recon_dir=None):
    model.eval()
    device = next(model.parameters()).device
    bare_model = get_bare_model(model)
    use_lpips = (criterion.distortion == "lpips")
    save_recon = save_recon_dir is not None
    saved_images = 0

    dataset_paths = None
    if save_recon:
        os.makedirs(save_recon_dir, exist_ok=True)
        ds = getattr(test_loader, "dataset", None)
        dataset_paths = getattr(ds, "image_paths", None)

    rdl_m = AverageMeter()
    bpp_m = AverageMeter()
    mse_m = AverageMeter()
    aux_m = AverageMeter()
    psnr_m = AverageMeter()
    lpips_m = AverageMeter() if use_lpips else None

    total_fwd_time = 0.0
    total_iter = 0

    for it, x in enumerate(test_loader):
        x = x.to(device, non_blocking=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_net = bare_model(x)
        torch.cuda.synchronize()
        total_fwd_time += time.perf_counter() - t0
        total_iter += 1

        out = criterion(out_net, x)

        bs = x.size(0)
        if save_recon:
            x_hat = out_net["x_hat"].detach().clamp(0, 1).cpu()
            batch_start = saved_images
            for b in range(bs):
                img_idx = batch_start + b
                if dataset_paths is not None and img_idx < len(dataset_paths):
                    stem = os.path.splitext(os.path.basename(dataset_paths[img_idx]))[0]
                else:
                    stem = f"{it:04d}_{b:02d}"
                save_path = os.path.join(save_recon_dir, f"{stem}_xhat.png")
                save_image(x_hat[b], save_path)
            saved_images += bs

        rdl_m.update(out["rdloss"], bs)
        bpp_m.update(out["bpp_loss"], bs)
        mse_m.update(out["mse_loss"], bs)
        aux_m.update(bare_model.aux_loss(), bs)
        psnr_m.update(out["psnr"], bs)
        if use_lpips:
            lpips_m.update(out["lpips_loss"], bs)

    avg_fwd_ms = (total_fwd_time / max(total_iter, 1)) * 1000
    msg = (
        f"[{tag}][E{epoch}] "
        f"RD {rdl_m.avg:.4f}  BPP {bpp_m.avg:.4f}  "
        f"MSE {mse_m.avg:.6f}  PSNR {psnr_m.avg:.2f} dB"
    )
    if use_lpips:
        msg += f"  LPIPS {lpips_m.avg:.4f}"
    msg += f"  |  fwd {avg_fwd_ms:.1f} ms/img  ({total_iter} imgs)"
    logging.info(msg)
    if save_recon:
        logging.info("[%s][E%s] Saved %d x_hat images to %s",
                     tag, epoch, saved_images, save_recon_dir)

    stats = {
        "rd": rdl_m.avg, "bpp": bpp_m.avg, "mse": mse_m.avg,
        "psnr": psnr_m.avg, "fwd_ms": avg_fwd_ms,
    }
    if use_lpips:
        stats["lpips"] = lpips_m.avg
    return stats


# -------------------------
# checkpoint
# -------------------------
def save_checkpoint(state, is_best, out_dir, filename="checkpoint.pth.tar"):
    path = os.path.join(out_dir, filename)
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, os.path.join(out_dir, "checkpoint_best_loss.pth.tar"))


def load_pretrained(net, path, device):
    """Load weights only (no optimizer/scheduler state)."""
    logging.info("Loading pretrained weights from %s", path)
    ckpt = torch.load(path, map_location=device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # strip DataParallel 'module.' prefix if present
    if list(state.keys())[0].startswith("module."):
        state = OrderedDict((k[7:], v) for k, v in state.items())
    net.load_state_dict(state, strict=False)


def load_checkpoint(net, optimizer, aux_optimizer, lr_scheduler, path, device):
    """Load full training state for resumption."""
    logging.info("Resuming from checkpoint %s", path)
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    if list(state.keys())[0].startswith("module."):
        state = OrderedDict((k[7:], v) for k, v in state.items())
    net.load_state_dict(state, strict=False)

    last_epoch = ckpt.get("epoch", 0)
    best_loss = ckpt.get("loss", float("inf"))
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "aux_optimizer" in ckpt and aux_optimizer is not None:
        aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
    if "lr_scheduler" in ckpt:
        try:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        except Exception as e:
            logging.warning("Skip loading lr_scheduler state (possibly different scheduler type): %s", e)
    return last_epoch, best_loss


# -------------------------
# args
# -------------------------
def parse_args(argv):
    p = argparse.ArgumentParser("moecodec_psnr (yaml config)")
    p.add_argument(
        "-c", "--config", type=str,
        default="config/tinylic.yaml",
        help="Path to yaml config file",
    )
    cli = p.parse_args(argv)

    with open(cli.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("YAML config must be a dict at top level.")

    args = argparse.Namespace(**cfg)
    args.config = cli.config

    # default name = timestamp
    if not getattr(args, "name", None):
        args.name = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    # mandatory fields
    must_have = [
        "out_dir", "dataset", "epochs", "batch_size", "test_batch_size",
        "num_workers", "patch_size", "lmbda", "learning_rate", "aux_learning_rate",
        "milestones", "gamma", "clip_max_norm", "cuda", "gpu_id", "seed", "save",
        "model",
    ]
    missing = [k for k in must_have if not hasattr(args, k)]
    if missing:
        raise ValueError(f"Missing keys in yaml: {missing}")

    # optional fields with defaults
    if not hasattr(args, "dataset_kodak"):
        args.dataset_kodak = ""
    if not hasattr(args, "dataset_clic"):
        args.dataset_clic = ""
    if not hasattr(args, "eval_every"):
        args.eval_every = 5
    if not hasattr(args, "TEST"):
        args.TEST = False
    if not hasattr(args, "pretrained"):
        args.pretrained = ""
    if not hasattr(args, "checkpoint"):
        args.checkpoint = ""
    if not hasattr(args, "resume"):
        args.resume = False

    # distortion type: "mse" (default) or "lpips"
    if not hasattr(args, "distortion"):
        args.distortion = "mse"

    # MoE defaults (safe for non-MoE models)
    if not hasattr(args, "enc_moe"):
        args.enc_moe = False
    if not hasattr(args, "dec_moe"):
        args.dec_moe = False
    if not hasattr(args, "moe_config"):
        args.moe_config = None

    # Trainable-parameter mode:
    #   1) all      : train all parameters
    #   2) moe_only : train only MoE parameters
    #
    # Backward compatibility:
    #   If `train_mode` is absent, fallback to legacy `train_moe`:
    #   train_moe=True  -> moe_only
    #   train_moe=False -> all
    if hasattr(args, "train_mode"):
        args.train_mode = str(args.train_mode).lower()
    else:
        legacy_train_moe = bool(getattr(args, "train_moe", False))
        args.train_mode = "moe_only" if legacy_train_moe else "all"
    if args.train_mode not in {"all", "moe_only"}:
        raise ValueError(f"Invalid train_mode '{args.train_mode}'. Choose from ['all', 'moe_only']")

    # keep legacy field for existing code paths / logs
    args.train_moe = (args.train_mode == "moe_only")

    # Optional smoothness regularizer on SparseMoE router probabilities.
    # Priority:
    #   1) top-level `moe_smooth_lambda`
    #   2) `moe_config.smooth_lambda`
    if hasattr(args, "moe_smooth_lambda"):
        args.moe_smooth_lambda = float(args.moe_smooth_lambda)
    elif isinstance(getattr(args, "moe_config", None), dict):
        args.moe_smooth_lambda = float(args.moe_config.get("smooth_lambda", 0.0))
    else:
        args.moe_smooth_lambda = 0.0

    # TCM-specific defaults (only used when model is TCM / TCM_MoE)
    for key, default in _TCM_DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, default)
    # Optional floor LR for cosine annealing.
    if not hasattr(args, "cosine_eta_min"):
        args.cosine_eta_min = 0.0

    return args


# -------------------------
# visualization: channel energy vs MoE compute allocation
# -------------------------
@torch.no_grad()
def visualize_channel_moe_kodak(model, data_loader, save_dir, tag="kodak"):
    """Channel energy-compute sorted curves for ChannelMoEBlock.

    For each block:
    1) compute per-channel mean energy over Kodak;
    2) compute per-channel mean compute = #experts selecting channel;
    3) sort channels by energy (desc) and plot compute curve.
    """
    model.eval()
    device = next(model.parameters()).device

    moe_blocks = {}
    for name, mod in model.named_modules():
        if type(mod).__name__ == "ChannelMoEBlock" and hasattr(mod, "num_groups"):
            moe_blocks[name] = mod

    if not moe_blocks:
        logging.info("[vis-channel-sort] No ChannelMoEBlock found in model – skipping.")
        return

    block_names = list(moe_blocks.keys())
    n_blocks = len(block_names)
    logging.info("[vis-channel-sort] Found %d ChannelMoEBlock(s): %s", n_blocks, block_names)

    # Per block accumulators over samples: sum over B for (C,) vectors.
    energy_sum = {n: torch.zeros(m.hidden_dim, device=device) for n, m in moe_blocks.items()}
    compute_sum = {n: torch.zeros(m.hidden_dim, device=device) for n, m in moe_blocks.items()}
    sample_count = {n: 0 for n in moe_blocks.keys()}

    hook_data = {}

    def make_hook(block_name, block_mod):
        def hook_fn(module, inp, out):
            x = inp[0]  # (B, S, C)
            B, S, C = x.shape
            E = block_mod.num_experts
            k = block_mod._k

            # per-channel energy per-sample: (B, C)
            ch_energy = (x ** 2).mean(dim=1)  # mean square energy

            # reproduce routing to get selection count per channel
            ch_mean = x.mean(dim=1)                                # (B, C)
            ch_eng  = (x ** 2).mean(dim=1).sqrt()                  # (B, C)
            feat_dir = block_mod.gate_norm(ch_mean.unsqueeze(-1))  # (B, C, 1)
            feat_eng = block_mod.energy_proj(ch_eng.unsqueeze(-1)) # (B, C, 1)
            gate_feat = torch.cat([feat_dir, feat_eng], dim=-1)    # (B, C, 2)
            logits = block_mod.gate_linear(gate_feat)              # (B, C, E)
            affinity = torch.softmax(logits, dim=-1)
            affinity_T = affinity.permute(0, 2, 1)
            _gating, index = torch.topk(affinity_T, k=k, dim=-1)   # (B, E, k)

            ch_compute = torch.zeros(B, C, device=x.device, dtype=x.dtype)
            for e in range(E):
                ch_compute.scatter_add_(
                    1, index[:, e, :],
                    torch.ones_like(index[:, e, :], dtype=x.dtype),
                )

            hook_data[block_name] = {
                "energy_sum": ch_energy.sum(dim=0),    # (C,)
                "compute_sum": ch_compute.sum(dim=0),  # (C,)
                "n_samples": B,
            }

        return hook_fn

    handles = []
    for bname, bmod in moe_blocks.items():
        handles.append(bmod.register_forward_hook(make_hook(bname, bmod)))

    for x in data_loader:
        x = x.to(device, non_blocking=True)
        _ = model(x)
        for bname in block_names:
            if bname in hook_data:
                energy_sum[bname] += hook_data[bname]["energy_sum"]
                compute_sum[bname] += hook_data[bname]["compute_sum"]
                sample_count[bname] += int(hook_data[bname]["n_samples"])
        hook_data.clear()

    for h in handles:
        h.remove()

    out_dir = os.path.join(save_dir, f"{tag}_channel_energy_compute_sorted")
    os.makedirs(out_dir, exist_ok=True)

    n_cols = min(2, n_blocks)
    n_rows = math.ceil(n_blocks / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(8 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for idx, bname in enumerate(block_names):
        ax = axes[idx // n_cols][idx % n_cols]
        n_s = max(1, sample_count[bname])
        energy_np = (energy_sum[bname] / n_s).cpu().numpy()
        compute_np = (compute_sum[bname] / n_s).cpu().numpy()

        order = np.argsort(energy_np)[::-1]
        energy_sorted = energy_np[order]
        compute_sorted = compute_np[order]
        x_rank = np.arange(compute_sorted.shape[0])

        ax.plot(x_rank, compute_sorted, color="#E45756", linewidth=1.2,
                label="Compute (#experts/channel)")
        # also show normalized energy for reference
        e_norm = energy_sorted / max(1e-12, float(energy_sorted.max()))
        ax.plot(x_rank, e_norm * max(1e-12, float(compute_sorted.max())),
                color="#4C78A8", linewidth=1.0, alpha=0.6,
                label="Energy (scaled)")

        ax.set_xlabel("Channel Rank (sorted by energy ↓)", fontsize=8)
        ax.set_ylabel("Compute", fontsize=8)
        short_name = bname.replace("residual_group.blocks.", "blk")
        ax.set_title(f"{short_name}  (C={compute_sorted.shape[0]})", fontsize=9)
        ax.legend(fontsize=6, loc="upper right")
        ax.set_xlim(-0.5, compute_sorted.shape[0] - 0.5)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    for idx in range(n_blocks, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(f"Channel Energy-Compute (sorted) ({tag})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(out_dir, f"{tag}_channel_energy_compute_sorted.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    npz_data = {"block_names": np.array(block_names)}
    for i, n in enumerate(block_names):
        n_s = max(1, sample_count[n])
        e = (energy_sum[n] / n_s).cpu().numpy()
        c = (compute_sum[n] / n_s).cpu().numpy()
        order = np.argsort(e)[::-1]
        npz_data[f"energy_{i}"] = e
        npz_data[f"compute_{i}"] = c
        npz_data[f"order_{i}"] = order
        npz_data[f"energy_sorted_{i}"] = e[order]
        npz_data[f"compute_sorted_{i}"] = c[order]
    npz_path = os.path.join(out_dir, f"{tag}_channel_energy_compute_sorted.npz")
    np.savez(npz_path, **npz_data)
    logging.info("[vis-channel-sort] Saved channel energy-compute to %s", out_dir)


# -------------------------
# Patch-MoE expert-selection heatmap visualisation
# -------------------------
@torch.no_grad()
def visualize_patch_moe_kodak(model, data_loader, save_dir, tag="kodak"):
    """Visualise per-token expert selection counts for PatchMoEBlock.

    For each PatchMoEBlock in *model*, saves a **per-image** heatmap of
    ``_last_select_count`` (how many experts chose each patch token).

    Each Kodak image gets its own heatmap figure saved individually, since
    different images have different expert usage patterns and averaging
    across images would lose this information.

    Outputs (saved to *save_dir*/{tag}_patch_heatmaps/):
      - ``{img_stem}_patch_expert_heatmap.png``  – per-image, per-block subplot figure
      - ``{img_stem}_patch_expert_heatmap.npz``  – per-image raw numpy arrays
    """
    model.eval()
    device = next(model.parameters()).device

    # ---- 1. discover PatchMoEBlock modules ----
    moe_blocks = {}
    for name, mod in model.named_modules():
        if type(mod).__name__ == "PatchMoEBlock" and hasattr(mod, "_last_select_count"):
            moe_blocks[name] = mod

    if not moe_blocks:
        logging.info("[vis-patch] No PatchMoEBlock found in model – skipping.")
        return

    block_names = list(moe_blocks.keys())
    n_blocks = len(moe_blocks)
    logging.info("[vis-patch] Found %d PatchMoEBlock(s): %s", n_blocks, block_names)

    # ---- 2. get image file paths for naming ----
    dataset = data_loader.dataset
    image_paths = getattr(dataset, "image_paths", None)

    # ---- 3. per-image output directory ----
    per_img_dir = os.path.join(save_dir, f"{tag}_patch_heatmaps")
    os.makedirs(per_img_dir, exist_ok=True)

    # Colormap: light→dark = few→many expert selections
    cmap = plt.cm.YlOrRd

    # ---- 4. iterate over images, save each one's heatmap individually ----
    img_idx = 0
    for x in data_loader:
        x = x.to(device, non_blocking=True)
        _ = model(x)

        # derive a stem name for this image
        if image_paths is not None and img_idx < len(image_paths):
            img_stem = os.path.splitext(os.path.basename(image_paths[img_idx]))[0]
        else:
            img_stem = f"img{img_idx:03d}"

        # collect heatmap for this image from every block
        per_block_maps = {}
        for bname, bmod in moe_blocks.items():
            sc = bmod._last_select_count  # (B, T)
            grid = bmod._last_grid_size   # (Th, Tw)
            if sc is None or grid is None:
                continue
            Th, Tw = grid
            B = sc.shape[0]
            sc_map = sc.reshape(B, Th, Tw).mean(dim=0)  # (Th, Tw)
            per_block_maps[bname] = sc_map

        if not per_block_maps:
            img_idx += 1
            continue

        # ---- plot per-image heatmap ----
        n_cols = min(2, n_blocks)
        n_rows = math.ceil(n_blocks / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(7 * n_cols, 5 * n_rows),
                                 squeeze=False)

        for bidx, bname in enumerate(block_names):
            ax = axes[bidx // n_cols][bidx % n_cols]
            if bname not in per_block_maps:
                ax.set_visible(False)
                continue
            heatmap = per_block_maps[bname].cpu().numpy()
            im = ax.imshow(heatmap, cmap=cmap, interpolation='nearest', aspect='auto')
            fig.colorbar(im, ax=ax, shrink=0.8, label="#experts selecting token")
            short_name = bname.replace("residual_group.blocks.", "blk")
            Th, Tw = heatmap.shape
            ax.set_title(f"{short_name}  ({Th}x{Tw} patches)", fontsize=9)
            ax.set_xlabel("Patch col")
            ax.set_ylabel("Patch row")

        # hide unused subplots
        for bidx in range(n_blocks, n_rows * n_cols):
            axes[bidx // n_cols][bidx % n_cols].set_visible(False)

        fig.suptitle(f"Patch-MoE Expert Heatmap – {img_stem} ({tag})", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        fig_path = os.path.join(per_img_dir, f"{img_stem}_patch_expert_heatmap.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

        # ---- save per-image raw data ----
        npz_path = os.path.join(per_img_dir, f"{img_stem}_patch_expert_heatmap.npz")
        npz_data = {"block_names": np.array(block_names), "image_name": np.array(img_stem)}
        for i, n in enumerate(block_names):
            if n in per_block_maps:
                npz_data[f"heatmap_{i}"] = per_block_maps[n].cpu().numpy()
        np.savez(npz_path, **npz_data)

        img_idx += 1

    logging.info("[vis-patch] Per-image heatmaps saved to %s (%d images)", per_img_dir, img_idx)


# -------------------------
# Spatial-MoE (SparseMoEBlock) expert-selection heatmap visualisation
# -------------------------
@torch.no_grad()
def visualize_spatial_moe_kodak(model, data_loader, save_dir, tag="kodak"):
    """Visualise per-pixel expert selection counts for SparseMoEBlock.

    Same structure as ``visualize_patch_moe_kodak`` but for the spatial
    (pixel-level) MoE blocks.  Each Kodak image gets its own heatmap.

    Outputs (saved to *save_dir*/{tag}_spatial_heatmaps/):
      - ``{img_stem}_spatial_expert_heatmap.png``
      - ``{img_stem}_spatial_expert_heatmap.npz``
    """
    model.eval()
    device = next(model.parameters()).device

    # ---- 1. discover SparseMoEBlock modules ----
    moe_blocks = {}
    for name, mod in model.named_modules():
        if type(mod).__name__ == "SparseMoEBlock" and hasattr(mod, "_last_select_count"):
            moe_blocks[name] = mod

    if not moe_blocks:
        logging.info("[vis-spatial] No SparseMoEBlock found in model – skipping.")
        return

    block_names = list(moe_blocks.keys())
    n_blocks = len(moe_blocks)
    logging.info("[vis-spatial] Found %d SparseMoEBlock(s)", n_blocks)

    # ---- 2. get image file paths for naming ----
    dataset = data_loader.dataset
    image_paths = getattr(dataset, "image_paths", None)

    # ---- 3. per-image output directory ----
    per_img_dir = os.path.join(save_dir, f"{tag}_spatial_heatmaps")
    os.makedirs(per_img_dir, exist_ok=True)

    cmap = plt.cm.YlOrRd

    # ---- 4. iterate over images ----
    img_idx = 0
    for x in data_loader:
        x = x.to(device, non_blocking=True)
        _ = model(x)

        # derive image name
        if image_paths is not None and img_idx < len(image_paths):
            img_stem = os.path.splitext(os.path.basename(image_paths[img_idx]))[0]
        else:
            img_stem = f"img{img_idx:03d}"

        # collect heatmaps
        per_block_maps = {}
        for bname, bmod in moe_blocks.items():
            sc = bmod._last_select_count  # (B, S)
            grid = bmod._last_grid_size   # (H, W)
            if sc is None or grid is None:
                continue
            H, W = grid
            B = sc.shape[0]
            sc_map = sc.reshape(B, H, W).mean(dim=0)  # (H, W)
            per_block_maps[bname] = sc_map

        if not per_block_maps:
            img_idx += 1
            continue

        # ---- plot per-image heatmap ----
        n_cols = min(2, n_blocks)
        n_rows = math.ceil(n_blocks / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(7 * n_cols, 5 * n_rows),
                                 squeeze=False)

        for bidx, bname in enumerate(block_names):
            ax = axes[bidx // n_cols][bidx % n_cols]
            if bname not in per_block_maps:
                ax.set_visible(False)
                continue
            heatmap = per_block_maps[bname].cpu().numpy()
            im = ax.imshow(heatmap, cmap=cmap, interpolation='nearest', aspect='auto')
            fig.colorbar(im, ax=ax, shrink=0.8, label="#experts selecting pixel")
            short_name = bname.replace("residual_group.blocks.", "blk")
            Hm, Wm = heatmap.shape
            ax.set_title(f"{short_name}  ({Hm}x{Wm})", fontsize=9)
            ax.set_xlabel("Col")
            ax.set_ylabel("Row")

        for bidx in range(n_blocks, n_rows * n_cols):
            axes[bidx // n_cols][bidx % n_cols].set_visible(False)

        fig.suptitle(f"Spatial-MoE Expert Heatmap – {img_stem} ({tag})", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        fig_path = os.path.join(per_img_dir, f"{img_stem}_spatial_expert_heatmap.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

        # ---- save per-image raw data ----
        npz_path = os.path.join(per_img_dir, f"{img_stem}_spatial_expert_heatmap.npz")
        npz_data = {"block_names": np.array(block_names), "image_name": np.array(img_stem)}
        for i, n in enumerate(block_names):
            if n in per_block_maps:
                npz_data[f"heatmap_{i}"] = per_block_maps[n].cpu().numpy()
        np.savez(npz_path, **npz_data)

        img_idx += 1

    logging.info("[vis-spatial] Per-image heatmaps saved to %s (%d images)", per_img_dir, img_idx)


# -------------------------
# Spatial-MoE: token energy vs compute
# -------------------------
def _safe_block_name(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def _aggregate_region_sum(x_map, region_size):
    """Aggregate per-pixel map into non-overlapping region sums."""
    if region_size <= 1:
        return x_map
    B, H, W = x_map.shape
    r = int(region_size)
    pad_h = (r - H % r) % r
    pad_w = (r - W % r) % r
    x4 = x_map.unsqueeze(1)
    if pad_h > 0 or pad_w > 0:
        x4 = F.pad(x4, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return (F.avg_pool2d(x4, kernel_size=r, stride=r) * (r * r)).squeeze(1)


@torch.no_grad()
def visualize_spatial_energy_compute_kodak(model, data_loader, save_dir,
                                           tag="kodak", region_size=4,
                                           max_scatter=12000):
    """Spatial patch energy-compute sorted curves for SparseMoEBlock.

    Uses non-overlapping ``region_size x region_size`` (default 4x4) patches:
      patch_energy  = sum(token_energy in patch)
      patch_compute = sum(token_compute in patch)
    Then sort patches by energy (desc) and plot compute vs patch rank.
    """
    model.eval()
    device = next(model.parameters()).device

    moe_blocks = {}
    for name, mod in model.named_modules():
        if type(mod).__name__ == "SparseMoEBlock" and hasattr(mod, "_last_select_count"):
            moe_blocks[name] = mod

    if not moe_blocks:
        logging.info("[vis-energy] No SparseMoEBlock found in model – skipping.")
        return

    patch_r = int(max(1, region_size))
    out_dir = os.path.join(save_dir, f"{tag}_spatial_patch{patch_r}_energy_compute_sorted")
    os.makedirs(out_dir, exist_ok=True)

    accum = {n: {"energy": [], "compute": []} for n in moe_blocks}

    for x in data_loader:
        x = x.to(device, non_blocking=True)
        _ = model(x)

        for bname, bmod in moe_blocks.items():
            token_energy = getattr(bmod, "_last_token_energy", None)
            select_count = getattr(bmod, "_last_select_count", None)
            grid = getattr(bmod, "_last_grid_size", None)
            if token_energy is None or select_count is None or grid is None:
                continue

            H, W = int(grid[0]), int(grid[1])
            B = token_energy.shape[0]
            if H * W != token_energy.shape[1]:
                continue

            e_map = token_energy.reshape(B, H, W).float()
            c_map = select_count.reshape(B, H, W).float()
            e_reg = _aggregate_region_sum(e_map, region_size=patch_r)  # patch total energy
            c_reg = _aggregate_region_sum(c_map, region_size=patch_r)  # patch total compute

            accum[bname]["energy"].append(e_reg.reshape(-1).cpu().numpy())
            accum[bname]["compute"].append(c_reg.reshape(-1).cpu().numpy())

    n_done = 0
    for bname in moe_blocks.keys():
        if not accum[bname]["energy"]:
            continue

        energy = np.concatenate(accum[bname]["energy"], axis=0)
        compute = np.concatenate(accum[bname]["compute"], axis=0)
        m = np.isfinite(energy) & np.isfinite(compute)
        energy = energy[m]
        compute = compute[m]
        if energy.size == 0:
            continue

        order = np.argsort(energy)[::-1]
        energy_sorted = energy[order]
        compute_sorted = compute[order]
        rank = np.arange(compute_sorted.size)

        # display subsample for readability (raw full arrays still saved)
        if compute_sorted.size > max_scatter:
            rng = np.random.default_rng(seed=42)
            idx = np.sort(rng.choice(compute_sorted.size, size=max_scatter, replace=False))
        else:
            idx = np.arange(compute_sorted.size)

        rank_plot = rank[idx]
        compute_plot = compute_sorted[idx]
        energy_plot = energy_sorted[idx]

        fig, ax = plt.subplots(1, 1, figsize=(10, 4.8))
        ax.plot(rank_plot, compute_plot, color="#E45756", linewidth=1.1,
                label="Patch Compute (sum #selected experts)")
        e_norm = energy_plot / max(1e-12, float(energy_plot.max()))
        ax.plot(rank_plot, e_norm * max(1e-12, float(compute_plot.max())),
                color="#4C78A8", linewidth=1.0, alpha=0.6,
                label="Patch Energy (scaled)")

        ax.set_xlabel(f"Patch Rank (sorted by {patch_r}x{patch_r} patch energy ↓)")
        ax.set_ylabel("Patch Compute")
        short_name = bname.replace("residual_group.blocks.", "blk")
        ax.set_title(f"Spatial Patch Energy-Compute ({short_name}, {tag})")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        safe = _safe_block_name(bname)
        fig_path = os.path.join(out_dir, f"{safe}_energy_compute.png")
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)

        np.savez(
            os.path.join(out_dir, f"{safe}_energy_compute.npz"),
            block_name=np.array(bname),
            region_size=np.array(patch_r),
            patch_energy=energy,
            patch_compute=compute,
            order=order,
            patch_energy_sorted=energy_sorted,
            patch_compute_sorted=compute_sorted,
        )
        n_done += 1

    logging.info(
        "[vis-energy] Saved sorted patch energy-compute curves to %s (%d blocks, patch=%dx%d)",
        out_dir, n_done, patch_r, patch_r,
    )


# -------------------------
# Spatial-MoE: expert specialization (which tokens each expert selects)
# -------------------------
@torch.no_grad()
def visualize_spatial_expert_specialization_kodak(model, data_loader, save_dir, tag="kodak"):
    """Visualise per-image expert-specific spatial selection maps.

    For each Kodak image and each SparseMoEBlock, save expert-wise maps:
      expert e -> selection count map (#times token selected by expert e).
    Works for both ``mix_batch_token=False`` and ``mix_batch_token=True``.
    """
    model.eval()
    device = next(model.parameters()).device

    moe_blocks = {}
    for name, mod in model.named_modules():
        if type(mod).__name__ == "SparseMoEBlock":
            moe_blocks[name] = mod

    if not moe_blocks:
        logging.info("[vis-spec] No SparseMoEBlock found in model – skipping.")
        return

    block_names = list(moe_blocks.keys())
    logging.info("[vis-spec] Found %d SparseMoEBlock(s)", len(block_names))

    out_dir = os.path.join(save_dir, f"{tag}_spatial_expert_specialization")
    os.makedirs(out_dir, exist_ok=True)
    per_img_dir = os.path.join(out_dir, "per_image")
    os.makedirs(per_img_dir, exist_ok=True)

    # Recompute routing from each block input in forward hook so visualization
    # remains available even when block internals do not expose per-expert cache.
    hook_cache = {}

    def make_hook(block_name, block_mod):
        def hook_fn(module, inp, out):
            x = inp[0]  # (B, S, D)
            if x is None or x.dim() != 3:
                return

            B, S, _D = x.shape
            E = int(max(1, getattr(block_mod, "num_experts", 1)))
            cap = float(getattr(block_mod, "capacity", 1.0))

            logits = x @ block_mod.gate_weight  # (B, S, E)
            affinity = torch.softmax(logits, dim=-1)

            sel_map = torch.zeros(B, E, S, device=x.device, dtype=torch.float32)
            gate_map = torch.zeros(B, E, S, device=x.device, dtype=torch.float32)

            if bool(getattr(block_mod, "mix_batch_token", False)):
                # Global routing over B*S tokens.
                affinity_t = affinity.reshape(B * S, E).transpose(0, 1)  # (E, BS)
                k = max(1, int(((B * S) / E) * cap))
                gating, index = torch.topk(affinity_t, k=k, dim=-1)  # (E, k)

                tok_idx = index % S
                b_idx = torch.div(index, S, rounding_mode="floor")

                for e in range(E):
                    flat_idx = b_idx[e] * S + tok_idx[e]

                    flat_sel = torch.zeros(B * S, device=x.device, dtype=torch.float32)
                    flat_sel.scatter_add_(
                        0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32)
                    )
                    sel_map[:, e, :] = flat_sel.view(B, S)

                    flat_gate = torch.zeros(B * S, device=x.device, dtype=torch.float32)
                    flat_gate.scatter_add_(0, flat_idx, gating[e].float())
                    gate_map[:, e, :] = flat_gate.view(B, S)
            else:
                affinity_t = affinity.permute(0, 2, 1)  # (B, E, S)
                k = max(1, int((S / E) * cap))
                gating, index = torch.topk(affinity_t, k=k, dim=-1)  # (B, E, k)

                for e in range(E):
                    sel_map[:, e, :].scatter_add_(
                        1, index[:, e, :], torch.ones_like(index[:, e, :], dtype=torch.float32)
                    )
                    gate_map[:, e, :].scatter_add_(1, index[:, e, :], gating[:, e, :].float())

            grid = getattr(block_mod, "_last_grid_size", None)
            if grid is not None:
                H, W = int(grid[0]), int(grid[1])
            else:
                try:
                    H, W = block_mod._infer_hw(S, x_size=None)
                except Exception:
                    H, W = 1, S
            if H * W != S:
                H, W = 1, S

            hook_cache[block_name] = {
                "sel_map": sel_map.detach().cpu(),    # (B, E, S)
                "gate_map": gate_map.detach().cpu(),  # (B, E, S)
                "grid": (H, W),
            }

        return hook_fn

    handles = []
    for bname, bmod in moe_blocks.items():
        handles.append(bmod.register_forward_hook(make_hook(bname, bmod)))

    dataset = data_loader.dataset
    image_paths = getattr(dataset, "image_paths", None)

    img_idx = 0
    n_saved = 0
    for x in data_loader:
        x = x.to(device, non_blocking=True)
        _ = model(x)

        B = x.shape[0]
        for b in range(B):
            if image_paths is not None and img_idx < len(image_paths):
                img_stem = os.path.splitext(os.path.basename(image_paths[img_idx]))[0]
            else:
                img_stem = f"img{img_idx:03d}"

            img_out_dir = os.path.join(per_img_dir, img_stem)
            os.makedirs(img_out_dir, exist_ok=True)

            for bname in block_names:
                pack = hook_cache.get(bname)
                if pack is None:
                    continue
                sel_map = pack["sel_map"]   # (B, E, S)
                gate_map = pack["gate_map"]  # (B, E, S)
                H, W = pack["grid"]
                if sel_map.shape[0] <= b:
                    continue

                E = sel_map.shape[1]
                sel_hw = sel_map[b].reshape(E, H, W).numpy()
                gate_hw = gate_map[b].reshape(E, H, W).numpy()

                n_cols = min(4, E)
                n_rows = math.ceil(E / n_cols)
                fig, axes = plt.subplots(
                    n_rows, n_cols,
                    figsize=(3.6 * n_cols, 3.2 * n_rows),
                    squeeze=False
                )

                vmax = max(1e-8, float(sel_hw.max()))
                last_im = None
                for e in range(E):
                    ax = axes[e // n_cols][e % n_cols]
                    im = ax.imshow(
                        sel_hw[e], cmap="magma", vmin=0.0, vmax=vmax,
                        interpolation="nearest", aspect="auto"
                    )
                    last_im = im
                    ax.set_title(f"Expert {e}", fontsize=9)
                    ax.set_xlabel("Col")
                    ax.set_ylabel("Row")

                for k in range(E, n_rows * n_cols):
                    axes[k // n_cols][k % n_cols].set_visible(False)

                if last_im is not None:
                    fig.colorbar(
                        last_im, ax=axes.ravel().tolist(),
                        shrink=0.85, label="#selected by expert"
                    )

                short_name = bname.replace("residual_group.blocks.", "blk")
                fig.suptitle(
                    f"Spatial Expert Specialization – {img_stem} – {short_name} ({tag})",
                    fontsize=11,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.95])

                safe = _safe_block_name(bname)
                fig_path = os.path.join(img_out_dir, f"{safe}_expert_specialization.png")
                fig.savefig(fig_path, dpi=180)
                plt.close(fig)

                np.savez(
                    os.path.join(img_out_dir, f"{safe}_expert_specialization.npz"),
                    image_name=np.array(img_stem),
                    block_name=np.array(bname),
                    select_count=sel_hw,
                    gate_mass=gate_hw,
                    H=np.array(H),
                    W=np.array(W),
                    E=np.array(E),
                )
                n_saved += 1

            img_idx += 1

    for h in handles:
        h.remove()

    logging.info(
        "[vis-spec] Saved per-image expert specialization maps to %s "
        "(%d images, %d block-figures)",
        per_img_dir, img_idx, n_saved
    )


# -------------------------
# main
# -------------------------
def main(argv):
    args = parse_args(argv)

    # ---- DDP setup ----
    local_rank, global_rank, world_size = setup_ddp()
    use_ddp = (world_size > 1)

    run_dir = init_out_dir(args)

    # Only rank-0 sets up file logging; all ranks still log to stdout.
    if is_main_process():
        setup_logger(os.path.join(run_dir, "train.log"))
        logging.info("========== %s ==========", args.name)
        # concise config summary (key settings only)
        logging.info("Config: %s", args.config)
        logging.info("Model: %s  N=%s M=%s  |  DDP: %d GPU(s)",
                     args.model, args.N, args.M, world_size)
        logging.info("Data: %s  |  patch=%d  bs=%d  workers=%d",
                     args.dataset, args.patch_size, args.batch_size, args.num_workers)
        logging.info("Epochs: %d  eval_every=%d  |  lmbda=%s  dist=%s",
                     args.epochs, args.eval_every, args.lmbda, args.distortion)
        logging.info("LR: %.1e  aux_lr=%.1e  scheduler=cosine  eta_min=%.1e",
                     args.learning_rate, args.aux_learning_rate, args.cosine_eta_min)
        if getattr(args, "moe_config", None):
            logging.info("MoE: enc=%s dec=%s  config=%s",
                         args.enc_moe, args.dec_moe, args.moe_config)
        logging.info("Train mode: %s", args.train_mode)
        if args.checkpoint:
            logging.info("Checkpoint: %s  (resume=%s)", args.checkpoint, args.resume)
        elif args.pretrained:
            logging.info("Pretrained: %s", args.pretrained)
    else:
        # non-rank-0: suppress INFO, only WARNING+
        logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

    # seed (offset by rank so each GPU sees different augmentation)
    seed = args.seed if args.seed is not None else 42
    torch.manual_seed(seed + global_rank)
    random.seed(seed + global_rank)
    np.random.seed(seed + global_rank)
    if args.cuda:
        torch.cuda.manual_seed_all(seed + global_rank)

    # device
    if use_ddp:
        device = torch.device("cuda", local_rank)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    # ---- data ----
    train_tf = transforms.Compose([
        transforms.RandomCrop(args.patch_size, pad_if_needed=True, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
    test_tf = transforms.Compose([
        transforms.CenterCrop(args.patch_size),
        transforms.ToTensor(),
    ])

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_tf)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_tf)

    # DDP: use DistributedSampler for training data
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True,
    ) if use_ddp else None

    # Scale down num_workers per process to avoid overloading CPU/IO
    nw = args.num_workers
    if use_ddp and nw > 4:
        nw = max(4, nw // world_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,       # per-GPU batch size
        shuffle=(train_sampler is None),   # only shuffle when not using DDP sampler
        sampler=train_sampler,
        num_workers=nw,
        pin_memory=(str(device) != "cpu"),
        drop_last=True,
    )
    if is_main_process():
        eff_bs = args.batch_size * world_size
        logging.info("DataLoader: eff_bs=%d (%d/gpu x %d)  workers=%d/gpu  train=%d imgs",
                     eff_bs, args.batch_size, world_size, nw, len(train_dataset))

    # Eval/test loaders: run on all ranks but keep full dataset (no distributed sampler)
    test_loader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=nw, pin_memory=(str(device) != "cpu"),
    )

    # optional eval datasets
    val_loader_kodak = None
    val_loader_clic = None
    if args.dataset_kodak:
        val_dataset_kodak = Kodak(args.dataset_kodak, transforms.ToTensor())
        val_loader_kodak = DataLoader(
            val_dataset_kodak, batch_size=1, shuffle=False,
            num_workers=nw, pin_memory=(str(device) != "cpu"),
        )
    if args.dataset_clic:
        val_dataset_clic = ImageFolder(args.dataset_clic, split="test", transform=test_tf)
        val_loader_clic = DataLoader(
            val_dataset_clic, batch_size=args.test_batch_size, shuffle=False,
            num_workers=nw, pin_memory=(str(device) != "cpu"),
        )

    # ---- model ----
    net = build_model(args)
    net = net.to(device)

    # ---- loss / optimizer / scheduler ----
    # configure_optimizers may freeze params (train_mode=moe_only), so call it before counting.
    criterion = RateDistortionLoss(
        lmbda=args.lmbda, distortion=args.distortion,
    ).to(device)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=total_steps, eta_min=float(args.cosine_eta_min),
    # )
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[m * steps_per_epoch for m in args.milestones], gamma=args.gamma,
    )

    # ---- summary (after freeze) ----
    if is_main_process():
        total_params = sum(p.numel() for p in net.parameters()) / 1e6
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
        logging.info("Model: %s  |  Total: %.2fM params  |  Trainable: %.2fM params",
                     args.model, total_params, trainable_params)
        logging.info("Loss: %s  |  lambda=%s  |  lr=%.1e  |  aux_lr=%s",
                     args.distortion, args.lmbda, args.learning_rate,
                     f"{args.aux_learning_rate:.1e}" if aux_optimizer else "N/A")
        if args.moe_smooth_lambda > 0:
            logging.info("MoE smooth regularizer: lambda=%g", args.moe_smooth_lambda)
        logging.info("LR scheduler: CosineAnnealingLR per-step  |  steps/epoch=%d  |  total_steps=%d  |  eta_min=%.1e",
                     steps_per_epoch, total_steps, args.cosine_eta_min)

    # ---- load weights (before DDP wrapping) ----
    last_epoch = 0
    best_loss = float("inf")

    if args.pretrained:
        load_pretrained(net, args.pretrained, device)

    if args.checkpoint:
        if args.resume:
            last_epoch, best_loss = load_checkpoint(
                net, optimizer, aux_optimizer, lr_scheduler,
                args.checkpoint, device,
            )
            if is_main_process():
                logging.info("Resumed from epoch %d, best_loss=%.4f", last_epoch, best_loss)
        else:
            load_pretrained(net, args.checkpoint, device)

    # ---- wrap model with DDP ----
    if use_ddp:
        net = DDP(net, device_ids=[local_rank], output_device=local_rank,
                  find_unused_parameters=False)

    bare_net = get_bare_model(net)

    # ---- eval-only mode (rank 0 only for vis, all ranks for metrics) ----
    if args.TEST:
        if is_main_process():
            logging.info("===== TEST mode (checkpoint: %s) =====", args.checkpoint)
            if val_loader_kodak is not None:
                test_epoch(
                    net, criterion, val_loader_kodak, epoch=-1, tag="Kodak",
                    save_recon_dir=os.path.join(run_dir, "recon", "kodak_test"),
                )
                # MoE visualizations on Kodak
                vis_dir = os.path.join(run_dir, "vis")
                visualize_patch_moe_kodak(bare_net, val_loader_kodak, vis_dir, tag="kodak")
                visualize_spatial_moe_kodak(bare_net, val_loader_kodak, vis_dir, tag="kodak")
                visualize_spatial_expert_specialization_kodak(bare_net, val_loader_kodak, vis_dir, tag="kodak")
            if val_loader_clic is not None:
                test_epoch(net, criterion, val_loader_clic, epoch=-1, tag="CLIC")
            # always run on the built-in test split
            test_epoch(net, criterion, test_loader, epoch=-1, tag="TestSplit")
        cleanup_ddp()
        return

    # ---- training loop ----
    tqrange = tqdm.trange(last_epoch, args.epochs, disable=not is_main_process())
    for epoch in tqrange:
        if is_main_process():
            logging.info("------ Epoch %d/%d  lr=%.1e ------",
                         epoch, args.epochs - 1, optimizer.param_groups[0]["lr"])

        tr = train_one_epoch(
            net, criterion, train_loader, optimizer, aux_optimizer,
            lr_scheduler, epoch, args.clip_max_norm, train_sampler=train_sampler,
            moe_smooth_lambda=args.moe_smooth_lambda,
        )

        # Synchronise before eval
        if use_ddp:
            dist.barrier()

        # Eval / vis / checkpoint only on rank 0
        if is_main_process():
            test_loss = test_epoch(net, criterion, test_loader, epoch, tag="Val")

            # periodic evaluation on external datasets
            is_best = False
            if args.eval_every > 0 and epoch % args.eval_every == 0:
                if val_loader_kodak is not None:
                    kodak_res = test_epoch(
                        net, criterion, val_loader_kodak, epoch, tag="Kodak",
                        save_recon_dir=os.path.join(run_dir, "recon", f"kodak_epoch{epoch}"),
                    )
                    # MoE visualizations on Kodak
                    vis_dir = os.path.join(run_dir, "vis")
                    visualize_patch_moe_kodak(
                        bare_net, val_loader_kodak, vis_dir,
                        tag=f"kodak_epoch{epoch}",
                    )
                    visualize_spatial_moe_kodak(
                        bare_net, val_loader_kodak, vis_dir,
                        tag=f"kodak_epoch{epoch}",
                    )
                    visualize_spatial_expert_specialization_kodak(
                        bare_net, val_loader_kodak, vis_dir,
                        tag=f"kodak_epoch{epoch}",
                    )

                if val_loader_clic is not None:
                    clic_res = test_epoch(net, criterion, val_loader_clic, epoch, tag="CLIC")

                # use Kodak RD loss to track best, fall back to test split
                if val_loader_kodak is not None:
                    val_rd = kodak_res["rd"]
                else:
                    val_rd = test_loss["rd"]

                is_best = (val_rd <= best_loss)
                best_loss = min(best_loss, val_rd)

                if is_best:
                    logging.info("New best at epoch %d (val RD=%.4f)", epoch, best_loss)

            if args.save and args.eval_every > 0 and epoch % args.eval_every == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        # Always save bare model state_dict (no 'module.' prefix)
                        "state_dict": bare_net.state_dict(),
                        "loss": best_loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else {},
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "args": vars(args),
                        "train_stats": tr,
                    },
                    is_best,
                    run_dir,
                    filename=f"checkpoint_{epoch}.pth.tar",
                )
        # Synchronise after eval/checkpoint
        if use_ddp:
            dist.barrier()

    cleanup_ddp()


if __name__ == "__main__":
    main(sys.argv[1:])
