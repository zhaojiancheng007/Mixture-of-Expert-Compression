# tic_train_stage_vis.py
# Train TIC and save block-level spatial pattern maps on Kodak during eval.

import argparse
import csv
import logging
import math
import os
import random
import shutil
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in (_PROJECT_DIR, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml

from compressai.datasets import ImageFolder
from examples.models.tic import TIC
from utils.dataloader import Kodak


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
    def __init__(self, lmbda=1e-2, eps=1e-9):
        super().__init__()
        self.lmbda = float(lmbda)
        self.eps = float(eps)
        self.mse = nn.MSELoss()

    @staticmethod
    def psnr(output, target):
        mse = torch.mean((output - target) ** 2)
        if mse.item() == 0:
            return torch.tensor(100.0, device=output.device)
        return 10.0 * torch.log10(1.0 / mse)

    def forward(self, output, target):
        n, _, h, w = target.size()
        num_pixels = n * h * w
        bpp = sum(
            (torch.log(likelihoods.clamp(min=self.eps)).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        mse = self.mse(output["x_hat"], target)
        psnr = self.psnr(torch.clamp(output["x_hat"], 0, 1), target)
        rd = self.lmbda * (255.0 ** 2) * mse + bpp
        return {
            "rdloss": rd,
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


def init_out_dir(args):
    run_dir = os.path.join(args.out_dir, "TIC", str(args.lmbda))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_checkpoint(state, is_best, out_dir, filename="checkpoint.pth.tar"):
    path = os.path.join(out_dir, filename)
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, os.path.join(out_dir, "checkpoint_best_loss.pth.tar"))


def configure_optimizers(net, args):
    params = {
        n for n, p in net.named_parameters()
        if p.requires_grad and not n.endswith(".quantiles")
    }
    aux_params = {
        n for n, p in net.named_parameters()
        if p.requires_grad and n.endswith(".quantiles")
    }
    params_dict = dict(net.named_parameters())

    optimizer = optim.AdamW((params_dict[n] for n in sorted(params)), lr=args.learning_rate)
    if aux_params:
        aux_optimizer = optim.AdamW((params_dict[n] for n in sorted(aux_params)), lr=args.aux_learning_rate)
    else:
        aux_optimizer = None

    logging.info("Main optimizer params: %d", len(params))
    logging.info("Aux optimizer params: %d", len(aux_params))
    return optimizer, aux_optimizer


def save_rgb_image(img_chw, path):
    arr = img_chw.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    plt.imsave(path, arr)


def _normalize_for_vis(map_2d, low_pct=1.0, high_pct=99.0, log_scale=True):
    arr = map_2d.detach().cpu().float().numpy()
    arr = np.maximum(arr, 0.0)
    if log_scale:
        arr = np.log1p(arr)

    lo = float(np.percentile(arr, float(low_pct)))
    hi = float(np.percentile(arr, float(high_pct)))
    if not np.isfinite(lo):
        lo = float(np.min(arr))
    if not np.isfinite(hi):
        hi = float(np.max(arr))
    if hi <= lo:
        hi = lo + 1e-6

    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-8)
    return arr


def save_heatmap(map_2d, path, cmap="inferno", low_pct=1.0, high_pct=99.0, log_scale=True):
    arr = _normalize_for_vis(map_2d, low_pct=low_pct, high_pct=high_pct, log_scale=log_scale)
    plt.imsave(path, arr, cmap=cmap)


def save_map_artifacts(map_2d, out_prefix, cmap, low_pct, high_pct, save_npy=True):
    save_heatmap(
        map_2d,
        path=f"{out_prefix}.png",
        cmap=cmap,
        low_pct=low_pct,
        high_pct=high_pct,
        log_scale=True,
    )
    if save_npy:
        np.save(f"{out_prefix}.npy", map_2d.detach().cpu().numpy().astype(np.float32))


def get_image_stem(dataset, index):
    image_paths = getattr(dataset, "image_paths", None)
    if image_paths is not None and 0 <= index < len(image_paths):
        return os.path.splitext(os.path.basename(image_paths[index]))[0]
    return f"img_{index:03d}"


def _parse_string_list(value):
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Cannot parse boolean value from: {value}")


def _extract_tensor_output(output):
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    return output


def _effective_blur_kernel(h, w, kernel_size):
    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1

    max_k = min(int(h), int(w))
    if max_k <= 1:
        return 1
    if k > max_k:
        k = max_k if max_k % 2 == 1 else max_k - 1
    return max(1, int(k))


def blur_feature_map(y, kernel_size):
    if not torch.is_tensor(y) or y.dim() != 4:
        return y

    k = _effective_blur_kernel(y.shape[-2], y.shape[-1], kernel_size)
    if k <= 1:
        return y

    pad = k // 2
    y_pad = F.pad(y, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(y_pad, kernel_size=k, stride=1)


class BlockSpatialRecorder:
    def __init__(self, model, block_names, blur_kernel=5):
        self.model = model
        self.blur_kernel = int(blur_kernel)
        self.records = {}
        self.handles = []
        self.enabled = True
        self.active_block_names = []

        module_dict = dict(model.named_modules())
        for block_name in block_names:
            module = module_dict.get(block_name, None)
            if module is None:
                logging.warning("[Spatial] Block not found, skip: %s", block_name)
                continue
            handle = module.register_forward_hook(self._build_hook(block_name))
            self.handles.append(handle)
            self.active_block_names.append(block_name)

    def _build_hook(self, block_name):
        def _hook(_module, inputs, output):
            if not self.enabled:
                return
            if not inputs:
                return

            x = inputs[0]
            y = _extract_tensor_output(output)

            if (not torch.is_tensor(x)) or (not torch.is_tensor(y)):
                return
            if x.dim() != 4 or y.dim() != 4:
                return
            if x.shape != y.shape:
                return

            with torch.no_grad():
                update_map = (y - x).pow(2).mean(dim=1, keepdim=True)
                energy_map = y.pow(2).mean(dim=1, keepdim=True)
                y_low = blur_feature_map(y, self.blur_kernel)
                y_high = y - y_low
                e_low_map = y_low.pow(2).mean(dim=1, keepdim=True)
                e_high_map = y_high.pow(2).mean(dim=1, keepdim=True)

            if y.requires_grad:
                y.retain_grad()

            self.records[block_name] = {
                "update": update_map.detach(),
                "energy": energy_map.detach(),
                "elow": e_low_map.detach(),
                "ehigh": e_high_map.detach(),
                "y_ref": y,
            }

        return _hook

    def clear(self):
        self.records.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def evaluate_kodak_with_block_maps(
    model,
    criterion,
    kodak_loader,
    epoch,
    save_root,
    block_names,
    save_max_images=8,
    blur_kernel=5,
    map_low_pct=1.0,
    map_high_pct=99.0,
    save_npy=True,
):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    rd_m = AverageMeter()
    bpp_m = AverageMeter()
    mse_m = AverageMeter()
    psnr_m = AverageMeter()

    epoch_dir = os.path.join(save_root, f"epoch_{int(epoch) + 1:04d}")
    os.makedirs(epoch_dir, exist_ok=True)

    recorder = BlockSpatialRecorder(model=model, block_names=block_names, blur_kernel=blur_kernel)
    if not recorder.active_block_names:
        logging.warning("[Spatial] No valid blocks to record. Only Kodak metrics will be evaluated.")

    csv_rows = []
    block_agg = {
        n: {"count": 0, "u": 0.0, "e": 0.0, "eh": 0.0, "el": 0.0, "g": 0.0}
        for n in recorder.active_block_names
    }

    dataset = kodak_loader.dataset
    max_images = max(0, int(save_max_images))
    global_index = 0

    for x in kodak_loader:
        x = x.to(device, non_blocking=True)
        bs = int(x.size(0))

        batch_start = global_index
        need_maps = bool(recorder.active_block_names) and (batch_start < max_images)

        model.zero_grad(set_to_none=True)
        recorder.clear()
        recorder.enabled = need_maps

        if need_maps:
            with torch.enable_grad():
                out_net = model(x)
                out = criterion(out_net, x)
                out["rdloss"].backward()
        else:
            with torch.no_grad():
                out_net = model(x)
                out = criterion(out_net, x)

        rd_m.update(out["rdloss"], bs)
        bpp_m.update(out["bpp_loss"], bs)
        mse_m.update(out["mse_loss"], bs)
        psnr_m.update(out["psnr"], bs)

        if need_maps:
            x_hat = out_net["x_hat"].detach()
            target_hw = (x.shape[-2], x.shape[-1])

            for b in range(bs):
                image_index = batch_start + b
                if image_index >= max_images:
                    continue

                image_name = get_image_stem(dataset, image_index)
                image_dir = os.path.join(epoch_dir, f"img_{image_index:03d}_{image_name}")
                os.makedirs(image_dir, exist_ok=True)

                save_rgb_image(x[b], os.path.join(image_dir, "input.png"))
                save_rgb_image(x_hat[b], os.path.join(image_dir, "x_hat.png"))

                for block_name in recorder.active_block_names:
                    rec = recorder.records.get(block_name, None)
                    if rec is None:
                        continue

                    grad_ref = rec["y_ref"].grad
                    if grad_ref is None:
                        grad_map_native = torch.zeros_like(rec["update"][b:b + 1])
                    else:
                        grad_map_native = grad_ref[b:b + 1].detach().pow(2).mean(dim=1, keepdim=True)

                    u_map = F.interpolate(rec["update"][b:b + 1], size=target_hw, mode="bilinear", align_corners=False)[0, 0]
                    e_map = F.interpolate(rec["energy"][b:b + 1], size=target_hw, mode="bilinear", align_corners=False)[0, 0]
                    eh_map = F.interpolate(rec["ehigh"][b:b + 1], size=target_hw, mode="bilinear", align_corners=False)[0, 0]
                    el_map = F.interpolate(rec["elow"][b:b + 1], size=target_hw, mode="bilinear", align_corners=False)[0, 0]
                    g_map = F.interpolate(grad_map_native, size=target_hw, mode="bilinear", align_corners=False)[0, 0]

                    block_dir = os.path.join(image_dir, block_name)
                    os.makedirs(block_dir, exist_ok=True)

                    save_map_artifacts(
                        u_map,
                        out_prefix=os.path.join(block_dir, "update_magnitude"),
                        cmap="magma",
                        low_pct=map_low_pct,
                        high_pct=map_high_pct,
                        save_npy=save_npy,
                    )
                    save_map_artifacts(
                        e_map,
                        out_prefix=os.path.join(block_dir, "feature_energy"),
                        cmap="viridis",
                        low_pct=map_low_pct,
                        high_pct=map_high_pct,
                        save_npy=save_npy,
                    )
                    save_map_artifacts(
                        eh_map,
                        out_prefix=os.path.join(block_dir, "high_freq_energy"),
                        cmap="inferno",
                        low_pct=map_low_pct,
                        high_pct=map_high_pct,
                        save_npy=save_npy,
                    )
                    save_map_artifacts(
                        el_map,
                        out_prefix=os.path.join(block_dir, "low_freq_energy"),
                        cmap="cividis",
                        low_pct=map_low_pct,
                        high_pct=map_high_pct,
                        save_npy=save_npy,
                    )
                    save_map_artifacts(
                        g_map,
                        out_prefix=os.path.join(block_dir, "grad_sensitivity"),
                        cmap="plasma",
                        low_pct=map_low_pct,
                        high_pct=map_high_pct,
                        save_npy=save_npy,
                    )

                    u_mean = float(u_map.mean().item())
                    e_mean = float(e_map.mean().item())
                    eh_mean = float(eh_map.mean().item())
                    el_mean = float(el_map.mean().item())
                    g_mean = float(g_map.mean().item())
                    hf_ratio = eh_mean / (el_mean + 1e-8)

                    csv_rows.append({
                        "epoch": int(epoch) + 1,
                        "image_index": int(image_index),
                        "image_name": image_name,
                        "block": block_name,
                        "update_mean": u_mean,
                        "energy_mean": e_mean,
                        "ehigh_mean": eh_mean,
                        "elow_mean": el_mean,
                        "hf_ratio": hf_ratio,
                        "grad_mean": g_mean,
                    })

                    block_agg[block_name]["count"] += 1
                    block_agg[block_name]["u"] += u_mean
                    block_agg[block_name]["e"] += e_mean
                    block_agg[block_name]["eh"] += eh_mean
                    block_agg[block_name]["el"] += el_mean
                    block_agg[block_name]["g"] += g_mean

        model.zero_grad(set_to_none=True)
        recorder.clear()
        global_index += bs

    recorder.remove()

    if csv_rows:
        csv_path = os.path.join(epoch_dir, "block_metrics.csv")
        fieldnames = [
            "epoch",
            "image_index",
            "image_name",
            "block",
            "update_mean",
            "energy_mean",
            "ehigh_mean",
            "elow_mean",
            "hf_ratio",
            "grad_mean",
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    logging.info(
        "[Kodak][E%d] RD %.4f | BPP %.4f | MSE %.6f | PSNR %.2f dB",
        epoch, rd_m.avg, bpp_m.avg, mse_m.avg, psnr_m.avg,
    )

    if csv_rows:
        logging.info(
            "[Spatial][E%d] saved block maps for %d images to %s",
            epoch,
            min(max_images, len(dataset)),
            epoch_dir,
        )
        for block_name in recorder.active_block_names:
            s = block_agg[block_name]
            if s["count"] <= 0:
                continue
            cnt = float(s["count"])
            logging.info(
                "[Spatial][E%d][%s] U=%.6f E=%.6f Ehigh=%.6f Elow=%.6f G=%.6f",
                epoch,
                block_name,
                s["u"] / cnt,
                s["e"] / cnt,
                s["eh"] / cnt,
                s["el"] / cnt,
                s["g"] / cnt,
            )

    if was_training:
        model.train()

    return {
        "rd": rd_m.avg,
        "bpp": bpp_m.avg,
        "mse": mse_m.avg,
        "psnr": psnr_m.avg,
    }


def train_one_epoch(model, aux_model, criterion, train_loader, optimizer, aux_optimizer,
                    lr_scheduler, epoch, clip_max_norm, global_step=0):
    model.train()
    device = next(aux_model.parameters()).device

    rd_m = AverageMeter()
    bpp_m = AverageMeter()
    mse_m = AverageMeter()
    psnr_m = AverageMeter()
    aux_m = AverageMeter()

    for it, x in enumerate(train_loader):
        x = x.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out_net = model(x)
        out = criterion(out_net, x)
        out["rdloss"].backward()

        if aux_optimizer is not None:
            aux_loss = aux_model.aux_loss()
            aux_optimizer.zero_grad(set_to_none=True)
            aux_loss.backward()
        else:
            aux_loss = torch.tensor(0.0, device=device)

        if clip_max_norm and clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(aux_model.parameters(), clip_max_norm)

        optimizer.step()
        if aux_optimizer is not None:
            aux_optimizer.step()
        lr_scheduler.step()

        bs = x.size(0)
        rd_m.update(out["rdloss"], bs)
        bpp_m.update(out["bpp_loss"], bs)
        mse_m.update(out["mse_loss"], bs)
        psnr_m.update(out["psnr"], bs)
        aux_m.update(aux_loss, bs)

        global_step += 1

        if it % 100 == 0:
            logging.info(
                "[Train][E%d][it %d/%d][step %d] RD %.4f | BPP %.4f | MSE %.6f | PSNR %.2f | AUX %.2f",
                epoch, it, len(train_loader), global_step,
                rd_m.avg, bpp_m.avg, mse_m.avg, psnr_m.avg, aux_m.avg,
            )

    logging.info(
        "[Train][E%d] RD %.4f | BPP %.4f | MSE %.6f | PSNR %.2f dB | AUX %.2f",
        epoch, rd_m.avg, bpp_m.avg, mse_m.avg, psnr_m.avg, aux_m.avg,
    )
    return global_step


def parse_args(argv):
    p = argparse.ArgumentParser("TIC training with Kodak block-level spatial tracking")
    p.add_argument("-c", "--config", type=str, default="config/tic_stage_vis.yaml")
    cli = p.parse_args(argv)

    with open(cli.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("YAML config must be a dict.")

    args = argparse.Namespace(**cfg)
    args.config = cli.config

    defaults = {
        "dataset": "",
        "dataset_kodak": "",
        "out_dir": "./checkpoints/tic_stage_vis",
        "epochs": 30,
        "batch_size": 16,
        "test_batch_size": 1,
        "num_workers": 8,
        "patch_size": 256,
        "seed": 42,
        "clip_max_norm": 1.0,
        "save": True,
        "N": 128,
        "M": 192,
        "lmbda": 5e-3,
        "learning_rate": 1e-4,
        "aux_learning_rate": 1e-3,
        "milestones": [10, 15, 20],
        "gamma": 0.5,
        "cuda": True,
        "gpu_id": 0,
        "checkpoint": "",
        "resume": False,
        "eval_every": 1,
        "spatial_save_images": 8,
        "spatial_blur_kernel": 5,
        "spatial_map_low_pct": 1.0,
        "spatial_map_high_pct": 99.0,
        "spatial_save_npy": True,
        "spatial_block_names": ["g_a1", "g_a3", "g_a5", "g_a7", "g_s0", "g_s2", "g_s4", "g_s6"],
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    args.spatial_block_names = _parse_string_list(args.spatial_block_names)
    if not args.spatial_block_names:
        args.spatial_block_names = list(defaults["spatial_block_names"])

    args.spatial_save_npy = _to_bool(args.spatial_save_npy)

    required = ["dataset", "dataset_kodak"]
    missing = [k for k in required if not getattr(args, k, "")]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    run_dir = init_out_dir(args)
    setup_logger(os.path.join(run_dir, "train.log"))

    logging.info("Config: %s", args.config)
    logging.info("Output: %s", run_dir)
    logging.info("Model: TIC N=%d M=%d", args.N, args.M)
    logging.info(
        "eval_every=%d | spatial_save_images=%d | spatial_blur_kernel=%d | blocks=%s",
        int(args.eval_every),
        int(args.spatial_save_images),
        int(args.spatial_blur_kernel),
        ",".join(args.spatial_block_names),
    )

    train_tf = transforms.Compose([
        transforms.RandomCrop(args.patch_size, pad_if_needed=True, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    train_ds = ImageFolder(args.dataset, split="train", transform=train_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    kodak_ds = Kodak(args.dataset_kodak, transforms.ToTensor())
    kodak_loader = DataLoader(
        kodak_ds,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    net = TIC(N=args.N, M=args.M).to(device)
    if args.cuda and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    train_model = net
    bare_model = net.module if isinstance(net, nn.DataParallel) else net

    criterion = RateDistortionLoss(lmbda=args.lmbda).to(device)
    optimizer, aux_optimizer = configure_optimizers(bare_model, args)

    steps_per_epoch = len(train_loader)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[m * steps_per_epoch for m in args.milestones],
        gamma=args.gamma,
    )

    total_params = sum(p.numel() for p in bare_model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in bare_model.parameters() if p.requires_grad) / 1e6
    logging.info("Total params: %.2fM | Trainable: %.2fM", total_params, trainable_params)

    start_epoch = 0
    best_psnr = -1.0
    global_step = 0

    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        if state and next(iter(state)).startswith("module."):
            state = {k[7:]: v for k, v in state.items()}
        bare_model.load_state_dict(state, strict=False)
        logging.info("Loaded checkpoint: %s", args.checkpoint)

        if args.resume:
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            best_psnr = float(ckpt.get("psnr", best_psnr))
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if aux_optimizer is not None and "aux_optimizer" in ckpt:
                aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
            if "lr_scheduler" in ckpt:
                try:
                    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
                except Exception:
                    pass
            global_step = int(ckpt.get("global_step", 0))
            logging.info("Resume from epoch=%d, global_step=%d", start_epoch, global_step)

    spatial_root = os.path.join(run_dir, "kodak_block_spatial")
    os.makedirs(spatial_root, exist_ok=True)

    for epoch in range(start_epoch, int(args.epochs)):
        logging.info("Epoch %d/%d | LR %.2e", epoch, int(args.epochs) - 1, optimizer.param_groups[0]["lr"])

        global_step = train_one_epoch(
            model=train_model,
            aux_model=bare_model,
            criterion=criterion,
            train_loader=train_loader,
            optimizer=optimizer,
            aux_optimizer=aux_optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            clip_max_norm=args.clip_max_norm,
            global_step=global_step,
        )

        need_eval = ((epoch + 1) % int(args.eval_every) == 0) or (epoch == int(args.epochs) - 1)
        stats = None
        if need_eval:
            stats = evaluate_kodak_with_block_maps(
                model=bare_model,
                criterion=criterion,
                kodak_loader=kodak_loader,
                epoch=epoch,
                save_root=spatial_root,
                block_names=args.spatial_block_names,
                save_max_images=int(args.spatial_save_images),
                blur_kernel=int(args.spatial_blur_kernel),
                map_low_pct=float(args.spatial_map_low_pct),
                map_high_pct=float(args.spatial_map_high_pct),
                save_npy=bool(args.spatial_save_npy),
            )
            best_psnr = max(best_psnr, stats["psnr"])

        if args.save:
            cur_psnr = stats["psnr"] if stats is not None else -1.0
            is_best = stats is not None and (cur_psnr >= best_psnr)
            save_checkpoint(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": bare_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else {},
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "psnr": float(cur_psnr),
                    "args": vars(args),
                },
                is_best=is_best,
                out_dir=run_dir,
                filename="checkpoint.pth.tar",
            )

    logging.info("Training done. Best Kodak PSNR %.3f dB", best_psnr)


if __name__ == "__main__":
    main(sys.argv[1:])
