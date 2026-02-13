# classification.py
# Task-aware compression training for classification.
# Supports: TIC/TIC_MoE, TinyLIC/TinyLIC_MoE, TCM/TCM_MoE.

import argparse
import logging
import os
import random
import sys
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in (_PROJECT_DIR, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet50
import yaml

from examples.task_train_utils import (
    AverageMeter,
    RateDistortionLoss,
    build_model,
    configure_optimizers,
    get_bare_model,
    init_out_dir,
    load_checkpoint,
    load_pretrained,
    save_checkpoint,
    setup_logger,
)


class FeatureHook:
    def __init__(self, module):
        self.feature = None
        module.register_forward_hook(self._attach)

    def _attach(self, module, inp, out):
        self.feature = out


class ClassificationTaskLoss(nn.Module):
    """Classification loss on reconstructed images, optional feature matching."""

    def __init__(self, device, use_feature_loss=False):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.use_feature_loss = bool(use_feature_loss)

        try:
            from torchvision.models import ResNet50_Weights
            self.classifier = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            self.classifier = resnet50(pretrained=True)

        self.classifier.requires_grad_(False)
        self.classifier.eval().to(device)

        self.hooks = [
            FeatureHook(self.classifier.layer1),
            FeatureHook(self.classifier.layer2),
            FeatureHook(self.classifier.layer3),
            FeatureHook(self.classifier.layer4),
        ]

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _norm(self, x):
        return (x - self.mean) / self.std

    def forward(self, out_net, x, y):
        x_hat = torch.clamp(out_net["x_hat"], 0, 1)
        pred = self.classifier(self._norm(x_hat))
        ce_loss = self.ce(pred, y)
        acc = (pred.argmax(dim=1) == y).float().mean()

        feat_loss = torch.tensor(0.0, device=x_hat.device)
        if self.use_feature_loss:
            rec_feats = [h.feature.clone() for h in self.hooks]
            _ = self.classifier(self._norm(x))
            ori_feats = [h.feature.clone() for h in self.hooks]
            feat_loss = torch.stack([
                nn.functional.mse_loss(rf, of)
                for rf, of in zip(rec_feats, ori_feats)
            ]).mean()

        return {
            "ce_loss": ce_loss,
            "acc": acc,
            "feat_loss": feat_loss,
        }


def _unwrap_batch(batch, device):
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        x, y = batch[0], batch[1]
    else:
        raise ValueError("Classification batch must be (image, label).")
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def train_one_epoch(model, criterion_rd, criterion_task, train_loader,
                    optimizer, aux_optimizer, clip_max_norm,
                    task_lmbda, ce_weight, feat_weight):
    model.train()
    bare_model = get_bare_model(model)
    device = next(model.parameters()).device

    bpp_m = AverageMeter()
    psnr_m = AverageMeter()
    ce_m = AverageMeter()
    feat_m = AverageMeter()
    task_m = AverageMeter()
    total_m = AverageMeter()
    aux_m = AverageMeter()
    acc_m = AverageMeter()

    for batch in train_loader:
        x, y = _unwrap_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        out_net = model(x)
        out_rd = criterion_rd(out_net, x)
        out_task = criterion_task(out_net, x, y)

        task_loss = ce_weight * out_task["ce_loss"] + feat_weight * out_task["feat_loss"]
        total_loss = out_rd["bpp_loss"] + task_lmbda * task_loss

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

        bs = x.size(0)
        bpp_m.update(out_rd["bpp_loss"], bs)
        psnr_m.update(out_rd["psnr"], bs)
        ce_m.update(out_task["ce_loss"], bs)
        feat_m.update(out_task["feat_loss"], bs)
        task_m.update(task_loss, bs)
        total_m.update(total_loss, bs)
        aux_m.update(aux_loss, bs)
        acc_m.update(out_task["acc"], bs)

    logging.info(
        "[Train] Total %.4f | Task %.4f | CE %.4f | Feat %.4f | BPP %.4f | PSNR %.2f | Acc %.4f | AUX %.2f",
        total_m.avg, task_m.avg, ce_m.avg, feat_m.avg, bpp_m.avg, psnr_m.avg, acc_m.avg, aux_m.avg,
    )

    return {
        "total": total_m.avg,
        "task": task_m.avg,
        "ce": ce_m.avg,
        "feat": feat_m.avg,
        "bpp": bpp_m.avg,
        "psnr": psnr_m.avg,
        "acc": acc_m.avg,
    }


@torch.no_grad()
def eval_epoch(model, criterion_rd, criterion_task, data_loader,
               task_lmbda, ce_weight, feat_weight, tag="Val"):
    model.eval()
    device = next(model.parameters()).device

    bpp_m = AverageMeter()
    psnr_m = AverageMeter()
    ce_m = AverageMeter()
    feat_m = AverageMeter()
    task_m = AverageMeter()
    total_m = AverageMeter()
    acc_m = AverageMeter()

    for batch in data_loader:
        x, y = _unwrap_batch(batch, device)
        out_net = model(x)
        out_rd = criterion_rd(out_net, x)
        out_task = criterion_task(out_net, x, y)

        task_loss = ce_weight * out_task["ce_loss"] + feat_weight * out_task["feat_loss"]
        total_loss = out_rd["bpp_loss"] + task_lmbda * task_loss

        bs = x.size(0)
        bpp_m.update(out_rd["bpp_loss"], bs)
        psnr_m.update(out_rd["psnr"], bs)
        ce_m.update(out_task["ce_loss"], bs)
        feat_m.update(out_task["feat_loss"], bs)
        task_m.update(task_loss, bs)
        total_m.update(total_loss, bs)
        acc_m.update(out_task["acc"], bs)

    logging.info(
        "[%s] Total %.4f | Task %.4f | CE %.4f | Feat %.4f | BPP %.4f | PSNR %.2f | Acc %.4f",
        tag, total_m.avg, task_m.avg, ce_m.avg, feat_m.avg, bpp_m.avg, psnr_m.avg, acc_m.avg,
    )

    model.train()
    return {
        "total": total_m.avg,
        "task": task_m.avg,
        "ce": ce_m.avg,
        "feat": feat_m.avg,
        "bpp": bpp_m.avg,
        "psnr": psnr_m.avg,
        "acc": acc_m.avg,
    }


def _build_imagenet_dataset(root, train_tf, val_tf):
    try:
        train_ds = torchvision.datasets.ImageNet(root, split="train", transform=train_tf)
        val_ds = torchvision.datasets.ImageNet(root, split="val", transform=val_tf)
        return train_ds, val_ds
    except Exception as exc:
        logging.warning("ImageNet loader failed (%s), fallback to ImageFolder train/val.", exc)
        train_dir = os.path.join(root, "train")
        val_dir = os.path.join(root, "val")
        train_ds = torchvision.datasets.ImageFolder(train_dir, transform=train_tf)
        val_ds = torchvision.datasets.ImageFolder(val_dir, transform=val_tf)
        return train_ds, val_ds


def parse_args(argv):
    parser = argparse.ArgumentParser("classification task training (yaml config)")
    parser.add_argument("-c", "--config", type=str, default="config/classification.yaml")
    parser.add_argument("-T", "--TEST", action="store_true")
    cli = parser.parse_args(argv)

    with open(cli.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("YAML config must be a dict at top level.")

    args = argparse.Namespace(**cfg)
    args.config = cli.config
    if cli.TEST:
        args.TEST = True

    if not hasattr(args, "name"):
        args.name = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    defaults = {
        "dataset": "imagenet",
        "epochs": 30,
        "batch_size": 16,
        "test_batch_size": 32,
        "num_workers": 8,
        "learning_rate": 1e-4,
        "aux_learning_rate": 1e-3,
        "clip_max_norm": 1.0,
        "cuda": True,
        "gpu_id": 0,
        "seed": 42,
        "save": True,
        "eval_every": 1,
        "lmbda": 5e-3,
        "distortion": "mse",
        "task_lmbda": 1.0,
        "task_ce_weight": 0.0,
        "task_feat_weight": 1.0,
        "use_task_feat": True,
        "train_subset_size": 0,
        "milestones": [15, 25],
        "gamma": 0.5,
        "checkpoint": "",
        "pretrained": "",
        "resume": False,
        "TEST": False,
        "model": "TIC_MoE",
        "N": 128,
        "M": 192,
        "enc_moe": True,
        "dec_moe": True,
        "h_moe": False,
        "moe_config": None,
        "train_moe": True,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    if not hasattr(args, "dataset_path"):
        raise ValueError("Missing required key: dataset_path")

    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    out_dir = init_out_dir(args, task_name="classification")
    setup_logger(os.path.join(out_dir, "train.log"))

    logging.info("========== %s =========", args.name)
    logging.info("Config: %s", args.config)
    logging.info("Model: %s  | N=%s M=%s", args.model, args.N, args.M)
    logging.info("Dataset: %s  | path=%s", args.dataset, args.dataset_path)
    logging.info("Device: %s", str(device))
    logging.info("Task loss: lmbda=%.4f  ce_w=%.3f  feat_w=%.3f", args.task_lmbda, args.task_ce_weight, args.task_feat_weight)

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    dataset_name = str(args.dataset).lower()
    if dataset_name in ("imagenet", "imagenet1k", "imagenet-1k"):
        train_ds, val_ds = _build_imagenet_dataset(args.dataset_path, train_tf, eval_tf)
    else:
        train_dir = os.path.join(args.dataset_path, "train")
        val_dir = os.path.join(args.dataset_path, "val")
        train_ds = torchvision.datasets.ImageFolder(train_dir, transform=train_tf)
        val_ds = torchvision.datasets.ImageFolder(val_dir, transform=eval_tf)

    if int(args.train_subset_size) > 0:
        n = min(int(args.train_subset_size), len(train_ds))
        indices = torch.randperm(len(train_ds))[:n].tolist()
        train_ds = Subset(train_ds, indices)
        logging.info("Using train subset: %d samples", len(train_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    net = build_model(args).to(device)
    if args.cuda and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    criterion_rd = RateDistortionLoss(lmbda=args.lmbda, distortion=args.distortion).to(device)
    criterion_task = ClassificationTaskLoss(
        device=device,
        use_feature_loss=(bool(args.use_task_feat) or float(args.task_feat_weight) > 0),
    ).to(device)

    bare_net = get_bare_model(net)
    optimizer, aux_optimizer = configure_optimizers(bare_net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(args.milestones),
        gamma=float(args.gamma),
    )

    total_params = sum(p.numel() for p in bare_net.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in bare_net.parameters() if p.requires_grad) / 1e6
    logging.info("Params: total=%.2fM  trainable=%.2fM", total_params, trainable_params)

    last_epoch = 0
    best_loss = float("inf")

    if args.checkpoint and os.path.isfile(args.checkpoint):
        if args.resume:
            last_epoch, best_loss = load_checkpoint(
                bare_net, optimizer, aux_optimizer, lr_scheduler,
                args.checkpoint, device,
            )
            last_epoch += 1
        else:
            load_pretrained(bare_net, args.checkpoint, device)
    elif args.pretrained and os.path.isfile(args.pretrained):
        load_pretrained(bare_net, args.pretrained, device)

    if args.TEST:
        eval_epoch(
            net, criterion_rd, criterion_task, val_loader,
            task_lmbda=float(args.task_lmbda),
            ce_weight=float(args.task_ce_weight),
            feat_weight=float(args.task_feat_weight),
            tag="Test",
        )
        return

    for epoch in range(last_epoch, int(args.epochs)):
        logging.info("Epoch %d / %d  | lr=%.2e", epoch, int(args.epochs) - 1, optimizer.param_groups[0]["lr"])
        train_one_epoch(
            net, criterion_rd, criterion_task, train_loader,
            optimizer, aux_optimizer,
            clip_max_norm=float(args.clip_max_norm),
            task_lmbda=float(args.task_lmbda),
            ce_weight=float(args.task_ce_weight),
            feat_weight=float(args.task_feat_weight),
        )

        do_eval = ((epoch + 1) % int(args.eval_every) == 0) or (epoch == int(args.epochs) - 1)
        if do_eval:
            val_stats = eval_epoch(
                net, criterion_rd, criterion_task, val_loader,
                task_lmbda=float(args.task_lmbda),
                ce_weight=float(args.task_ce_weight),
                feat_weight=float(args.task_feat_weight),
                tag="Val",
            )
            is_best = val_stats["total"] < best_loss
            best_loss = min(best_loss, val_stats["total"])
            logging.info("Best val total loss: %.4f", best_loss)
        else:
            val_stats = {"total": float("inf")}
            is_best = False

        lr_scheduler.step()

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": bare_net.state_dict(),
                    "loss": float(val_stats["total"]),
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else {},
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "args": vars(args),
                },
                is_best=is_best,
                out_dir=out_dir,
                filename="checkpoint.pth.tar",
            )


if __name__ == "__main__":
    main(sys.argv[1:])
