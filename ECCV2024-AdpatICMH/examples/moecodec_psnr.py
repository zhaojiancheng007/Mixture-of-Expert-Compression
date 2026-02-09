# moecodec_psnr.py
# Single-quality (single-bpp) training / evaluation script.
# Supports: TIC, TIC_MoE, TinyLIC, TinyLIC_MoE, TCM, TCM_MoE

import argparse
import math
import random
import shutil
import sys
import os
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
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from utils.dataloader import Kodak

from examples.models.moe import TIC_MoE
from examples.models.tic import TIC
from examples.models.tinylic import TinyLIC
from examples.models.tinylic_moe import TinyLIC_MoE
from examples.models.tcm import TCM
from examples.models.tcm_moe import TCM_MoE

import lpips
import tqdm
import yaml


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
}

# Models that require `args` for MoE / prompt configuration
_MOE_MODELS = {"TIC_MoE", "TinyLIC_MoE", "TCM_MoE"}

# Models in the TCM family (accept extra constructor params)
_TCM_MODELS = {"TCM", "TCM_MoE"}

# Default (N, M) per model family
_DEFAULT_NM = {
    "TIC":         (128, 192),
    "TIC_MoE":     (128, 192),
    "TinyLIC":     (128, 320),
    "TinyLIC_MoE": (128, 320),
    "TCM":         (128, 320),
    "TCM_MoE":     (128, 320),
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
    """Custom DataParallel to access module methods like aux_loss/update."""
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
def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer."""
    parameters = {
        n for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    logging.info("Main optimizer params: %d", len(parameters))
    logging.info("Aux  optimizer params: %d", len(aux_parameters))

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


# -------------------------
# train / eval
# -------------------------
def train_one_epoch(model, criterion, train_loader, optimizer, aux_optimizer, epoch, clip_max_norm):
    model.train()
    device = next(model.parameters()).device
    use_lpips = (criterion.distortion == "lpips")

    rdl_m = AverageMeter()
    bpp_m = AverageMeter()
    mse_m = AverageMeter()
    aux_m = AverageMeter()
    psnr_m = AverageMeter()
    lpips_m = AverageMeter() if use_lpips else None

    for it, x in enumerate(train_loader):
        x = x.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        out_net = model(x)
        out = criterion(out_net, x)

        out["rdloss"].backward()

        aux_loss = model.aux_loss()
        aux_optimizer.zero_grad(set_to_none=True)
        aux_loss.backward()

        if clip_max_norm and clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

        optimizer.step()
        aux_optimizer.step()

        bs = x.size(0)
        rdl_m.update(out["rdloss"], bs)
        bpp_m.update(out["bpp_loss"], bs)
        mse_m.update(out["mse_loss"], bs)
        aux_m.update(aux_loss, bs)
        psnr_m.update(out["psnr"], bs)
        if use_lpips:
            lpips_m.update(out["lpips_loss"], bs)

        if (it * bs) % 1000 == 0:
            msg = (
                f"[Train][E{epoch}] {it*bs}/{len(train_loader.dataset)} | "
                f"RD {rdl_m.avg:.4f} | BPP {bpp_m.avg:.4f} | "
                f"MSE {mse_m.avg:.6f} | PSNR {psnr_m.avg:.3f} dB"
            )
            if use_lpips:
                msg += f" | LPIPS {lpips_m.avg:.4f}"
            msg += f" | AUX {aux_m.avg:.3f}"
            logging.info(msg)

    stats = {
        "rd": rdl_m.avg, "bpp": bpp_m.avg,
        "mse": mse_m.avg, "psnr": psnr_m.avg, "aux": aux_m.avg,
    }
    if use_lpips:
        stats["lpips"] = lpips_m.avg
    return stats


@torch.no_grad()
def test_epoch(model, criterion, test_loader, epoch, tag="Val"):
    model.eval()
    device = next(model.parameters()).device
    use_lpips = (criterion.distortion == "lpips")

    rdl_m = AverageMeter()
    bpp_m = AverageMeter()
    mse_m = AverageMeter()
    aux_m = AverageMeter()
    psnr_m = AverageMeter()
    lpips_m = AverageMeter() if use_lpips else None

    for x in test_loader:
        x = x.to(device, non_blocking=True)
        out_net = model(x)
        out = criterion(out_net, x)

        bs = x.size(0)
        rdl_m.update(out["rdloss"], bs)
        bpp_m.update(out["bpp_loss"], bs)
        mse_m.update(out["mse_loss"], bs)
        aux_m.update(model.aux_loss(), bs)
        psnr_m.update(out["psnr"], bs)
        if use_lpips:
            lpips_m.update(out["lpips_loss"], bs)

    msg = (
        f"[{tag}][E{epoch}] RD {rdl_m.avg:.4f} | BPP {bpp_m.avg:.4f} | "
        f"MSE {mse_m.avg:.6f} | PSNR {psnr_m.avg:.3f} dB"
    )
    if use_lpips:
        msg += f" | LPIPS {lpips_m.avg:.4f}"
    msg += f" | AUX {aux_m.avg:.3f}"
    logging.info(msg)

    stats = {"rd": rdl_m.avg, "bpp": bpp_m.avg, "mse": mse_m.avg, "psnr": psnr_m.avg}
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
    if "aux_optimizer" in ckpt:
        aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
    if "lr_scheduler" in ckpt:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
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
        args.eval_every = 10
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

    # TCM-specific defaults (only used when model is TCM / TCM_MoE)
    for key, default in _TCM_DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, default)

    return args


# -------------------------
# main
# -------------------------
def main(argv):
    args = parse_args(argv)
    run_dir = init_out_dir(args)

    setup_logger(os.path.join(run_dir, "train.log"))
    logging.info("========== %s ==========", args.name)
    for k, v in sorted(vars(args).items()):
        logging.info("  %-25s : %s", k, v)

    # seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed_all(args.seed)

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    logging.info("Device: %s", device)

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

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    # optional eval datasets
    val_loader_kodak = None
    val_loader_clic = None
    if args.dataset_kodak:
        val_dataset_kodak = Kodak(args.dataset_kodak, transforms.ToTensor())
        val_loader_kodak = DataLoader(
            val_dataset_kodak, batch_size=1, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device == "cuda"),
        )
    if args.dataset_clic:
        val_dataset_clic = ImageFolder(args.dataset_clic, split="test", transform=test_tf)
        val_loader_clic = DataLoader(
            val_dataset_clic, batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device == "cuda"),
        )

    # ---- model ----
    net = build_model(args)
    net = net.to(device)

    total_params = sum(p.numel() for p in net.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
    logging.info("Model: %s  |  Total params: %.2fM  |  Trainable: %.2fM",
                 args.model, total_params, trainable_params)

    # ---- loss / optimizer / scheduler ----
    criterion = RateDistortionLoss(
        lmbda=args.lmbda, distortion=args.distortion,
    ).to(device)
    logging.info("Distortion: %s  |  lambda: %s", args.distortion, args.lmbda)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(args.milestones), gamma=float(args.gamma),
    )

    # ---- load weights ----
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
            logging.info("Resumed from epoch %d, best_loss=%.4f", last_epoch, best_loss)
        else:
            load_pretrained(net, args.checkpoint, device)

    # ---- eval-only mode ----
    if args.TEST:
        logging.info("===== TEST mode (checkpoint: %s) =====", args.checkpoint)
        if val_loader_kodak is not None:
            test_epoch(net, criterion, val_loader_kodak, epoch=-1, tag="Kodak")
        if val_loader_clic is not None:
            test_epoch(net, criterion, val_loader_clic, epoch=-1, tag="CLIC")
        # always run on the built-in test split
        test_epoch(net, criterion, test_loader, epoch=-1, tag="TestSplit")
        return

    # ---- training loop ----
    tqrange = tqdm.trange(last_epoch, args.epochs)
    for epoch in tqrange:
        logging.info("====== Epoch %d/%d ======", epoch, args.epochs - 1)
        logging.info("Learning rate: %.6e", optimizer.param_groups[0]["lr"])

        tr = train_one_epoch(
            net, criterion, train_loader, optimizer, aux_optimizer,
            epoch, args.clip_max_norm,
        )
        test_loss = test_epoch(net, criterion, test_loader, epoch, tag="Val")

        lr_scheduler.step()

        # periodic evaluation on external datasets
        is_best = False
        if args.eval_every > 0 and epoch % args.eval_every == 0:
            if val_loader_kodak is not None:
                logging.info("Evaluate on Kodak:")
                kodak_res = test_epoch(net, criterion, val_loader_kodak, epoch, tag="Kodak")

            if val_loader_clic is not None:
                logging.info("Evaluate on CLIC:")
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
                    "state_dict": net.state_dict(),
                    "loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "args": vars(args),
                    "train_stats": tr,
                },
                is_best,
                run_dir,
                filename=f"checkpoint_{epoch}.pth.tar",
            )


if __name__ == "__main__":
    main(sys.argv[1:])
