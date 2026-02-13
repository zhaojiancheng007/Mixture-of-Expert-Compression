# task_coco_train.py
# Shared training pipeline for detection / segmentation task-aware compression.

import argparse
import logging
import os
import pickle
import random
import sys
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
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
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.detection_utils import read_image
from detectron2.evaluation import COCOEvaluator
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone

from utils.alignment import Alignment
from utils.dataloader import Kodak, MSCOCO
from utils.predictor import ModPredictor

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


@contextmanager
def inference_context(model):
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


class PerceptualFPNLoss(nn.Module):
    """Backbone feature-distillation loss used by detection/segmentation tasks."""

    def __init__(self, cfg, device):
        super().__init__()
        self.task_net = build_resnet_fpn_backbone(cfg, ShapeSpec(channels=3))

        checkpoint = OrderedDict()
        with open(cfg.MODEL.WEIGHTS, "rb") as f:
            fpn_ckpt = pickle.load(f)
            for k, v in fpn_ckpt["model"].items():
                if "backbone" in k:
                    checkpoint[".".join(k.split(".")[1:])] = torch.from_numpy(v)

        self.task_net.load_state_dict(checkpoint, strict=True)
        self.task_net = self.task_net.to(device)
        self.task_net.requires_grad_(False)
        self.task_net.eval()

        self.align = Alignment(divisor=32).to(device)
        self.pixel_mean = torch.tensor([103.530, 116.280, 123.675], dtype=torch.float32).view(-1, 1, 1).to(device)

    def forward(self, x_hat, x, train_mode=False):
        with torch.no_grad():
            x_gt = x.flip(1).mul(255) - self.pixel_mean
            if not train_mode:
                x_gt = self.align.align(x_gt)
            gt_feats = self.task_net(x_gt)

        x_rec = torch.clamp(x_hat, 0, 1).flip(1).mul(255) - self.pixel_mean
        if not train_mode:
            x_rec = self.align.align(x_rec)
        rec_feats = self.task_net(x_rec)

        loss = 0.0
        for level in ("p2", "p3", "p4", "p5", "p6"):
            loss = loss + nn.functional.mse_loss(gt_feats[level], rec_feats[level])
        return 0.2 * loss


def _resolve_path(path):
    if not path:
        return path
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    candidate = os.path.join(_PROJECT_DIR, path)
    if os.path.exists(candidate):
        return candidate
    return path


def _resolve_coco_dirs(dataset_path):
    dataset_path = _resolve_path(dataset_path)

    candidates = [
        {
            "train": os.path.join(dataset_path, "train2017"),
            "val": os.path.join(dataset_path, "val2017"),
            "ann": os.path.join(dataset_path, "annotations", "instances_val2017.json"),
        },
        {
            "train": os.path.join(dataset_path, "coco2017", "train2017"),
            "val": os.path.join(dataset_path, "coco2017", "val2017"),
            "ann": os.path.join(dataset_path, "coco2017", "annotations", "instances_val2017.json"),
        },
    ]

    for c in candidates:
        if os.path.isdir(c["train"]) and os.path.isdir(c["val"]) and os.path.isfile(c["ann"]):
            return c

    raise FileNotFoundError(
        "Cannot resolve COCO paths from dataset_path='{}'. "
        "Expected either <root>/train2017|val2017|annotations or <root>/coco2017/...".format(dataset_path)
    )


def _register_coco_once(name, json_path, image_path):
    if name in DatasetCatalog.list():
        return
    register_coco_instances(name, {}, json_path, image_path)


def _unwrap_img_batch(batch, device):
    if isinstance(batch, (tuple, list)) and len(batch) >= 1:
        img = batch[0]
    else:
        img = batch
    return img.to(device, non_blocking=True)


def build_task_components(args, device, task_type):
    coco = _resolve_coco_dirs(args.dataset_path)

    if task_type == "detection":
        cfg_file = _resolve_path(getattr(args, "detectron_cfg", "config/faster_rcnn_R_50_FPN_3x.yaml"))
        model_weights = _resolve_path(getattr(args, "fastrcnn_path", getattr(args, "task_model_weights", "")))
        dataset_name = "compressed_coco_det"
    elif task_type == "segmentation":
        cfg_file = _resolve_path(getattr(args, "detectron_cfg", "config/mask_rcnn_R_50_FPN_3x.yaml"))
        model_weights = _resolve_path(getattr(args, "maskrcnn_path", getattr(args, "task_model_weights", "")))
        dataset_name = "compressed_coco_seg"
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    if not model_weights:
        raise ValueError("Missing task model weights (fastrcnn_path / maskrcnn_path / task_model_weights).")

    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.WEIGHTS = model_weights

    _register_coco_once(dataset_name, coco["ann"], coco["val"])
    evaluator = COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=os.path.join(args.out_dir, "coco_log"))
    test_loader = build_detection_test_loader(cfg, dataset_name)

    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN_with_Rate"
    predictor = ModPredictor(cfg)
    task_criterion = PerceptualFPNLoss(cfg, device)

    train_tf = transforms.Compose([
        transforms.RandomCrop((args.patch_size, args.patch_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_root = coco["train"]
    if not train_root.endswith("/"):
        train_root += "/"

    train_list = getattr(args, "train_img_list", os.path.join(_SCRIPT_DIR, "utils", "img_list.txt"))
    if not os.path.exists(train_list):
        train_list = None

    train_ds = MSCOCO(train_root, train_tf, train_list)

    val_loader = None
    dataset_kodak = getattr(args, "dataset_kodak", "")
    if dataset_kodak:
        val_root = _resolve_path(dataset_kodak)
        if not val_root.endswith("/"):
            val_root += "/"
        val_ds = Kodak(val_root, transforms.ToTensor())
        val_loader = DataLoader(
            val_ds,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    return train_loader, val_loader, test_loader, evaluator, predictor, task_criterion


def train_one_epoch(model, criterion_rd, criterion_task, train_loader,
                    optimizer, aux_optimizer, clip_max_norm,
                    task_lmbda, codec_align_divisor):
    model.train()
    bare_model = get_bare_model(model)
    device = next(model.parameters()).device

    align = Alignment(divisor=codec_align_divisor, mode="pad", padding_mode="constant").to(device)

    bpp_m = AverageMeter()
    psnr_m = AverageMeter()
    task_m = AverageMeter()
    total_m = AverageMeter()
    aux_m = AverageMeter()

    for batch in train_loader:
        x = _unwrap_img_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)

        x_in = align.align(x)
        out_net = model(x_in)
        out_net["x_hat"] = align.resume(out_net["x_hat"]).clamp_(0, 1)

        out_rd = criterion_rd(out_net, x)
        task_loss = criterion_task(out_net["x_hat"], x, train_mode=True)
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
        task_m.update(task_loss, bs)
        total_m.update(total_loss, bs)
        aux_m.update(aux_loss, bs)

    logging.info(
        "[Train] Total %.4f | Task %.4f | BPP %.4f | PSNR %.2f | AUX %.2f",
        total_m.avg, task_m.avg, bpp_m.avg, psnr_m.avg, aux_m.avg,
    )


@torch.no_grad()
def val_epoch(model, criterion_rd, criterion_task, val_loader,
              task_lmbda, codec_align_divisor):
    if val_loader is None:
        return None

    model.eval()
    device = next(model.parameters()).device
    align = Alignment(divisor=codec_align_divisor, mode="pad", padding_mode="constant").to(device)

    bpp_m = AverageMeter()
    psnr_m = AverageMeter()
    task_m = AverageMeter()
    total_m = AverageMeter()

    for batch in val_loader:
        x = _unwrap_img_batch(batch, device)

        x_in = align.align(x)
        out_net = model(x_in)
        out_net["x_hat"] = align.resume(out_net["x_hat"]).clamp_(0, 1)

        out_rd = criterion_rd(out_net, x)
        task_loss = criterion_task(out_net["x_hat"], x, train_mode=False)
        total_loss = out_rd["bpp_loss"] + task_lmbda * task_loss

        bs = x.size(0)
        bpp_m.update(out_rd["bpp_loss"], bs)
        psnr_m.update(out_rd["psnr"], bs)
        task_m.update(task_loss, bs)
        total_m.update(total_loss, bs)

    logging.info(
        "[Val] Total %.4f | Task %.4f | BPP %.4f | PSNR %.2f",
        total_m.avg, task_m.avg, bpp_m.avg, psnr_m.avg,
    )
    model.train()
    return total_m.avg


@torch.no_grad()
def test_epoch_coco(model, criterion_rd, criterion_task, test_loader,
                    predictor, evaluator, codec_align_divisor):
    model.eval()
    device = next(model.parameters()).device

    codec_align = Alignment(divisor=codec_align_divisor, mode="pad", padding_mode="constant").to(device)
    rcnn_align = Alignment(divisor=32).to(device)
    pixel_mean = torch.tensor([103.530, 116.280, 123.675], dtype=torch.float32).view(-1, 1, 1).to(device)

    bpp_m = AverageMeter()
    psnr_m = AverageMeter()
    task_m = AverageMeter()

    evaluator.reset()

    for i, batch in enumerate(test_loader):
        with ExitStack() as stack:
            if isinstance(predictor.model, nn.Module):
                stack.enter_context(inference_context(predictor.model))
            stack.enter_context(torch.no_grad())

            img = read_image(batch[0]["file_name"], format="BGR")
            x = torch.stack([batch[0]["image"].float().div(255)]).flip(1).to(device)

            x_in = codec_align.align(x)
            out_net = model(x_in)
            out_net["x_hat"] = codec_align.resume(out_net["x_hat"]).clamp_(0, 1)

            out_rd = criterion_rd(out_net, x)
            task_loss = criterion_task(out_net["x_hat"], x, train_mode=False)

            trand_y_tilde = out_net["x_hat"].flip(1).mul(255)
            trand_y_tilde = rcnn_align.align(trand_y_tilde - pixel_mean)

            predictions = predictor(img, trand_y_tilde)
            evaluator.process(batch, [predictions])

            bpp_m.update(out_rd["bpp_loss"])
            psnr_m.update(out_rd["psnr"])
            task_m.update(task_loss)

            if i % 500 == 0:
                logging.info(
                    "[COCO-Test] %d/%d | BPP %.4f | PSNR %.2f | Task %.4f",
                    i, len(test_loader), bpp_m.avg, psnr_m.avg, task_m.avg,
                )

    results = evaluator.evaluate()
    logging.info("[COCO-Test] BPP %.4f | PSNR %.2f | Task %.4f", bpp_m.avg, psnr_m.avg, task_m.avg)
    logging.info("[COCO-Test] Eval results: %s", str(results))

    model.train()
    return results


def parse_args(argv, task_type, default_config):
    parser = argparse.ArgumentParser(f"{task_type} task training (yaml config)")
    parser.add_argument("-c", "--config", type=str, default=default_config)
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
        "epochs": 40,
        "batch_size": 8,
        "test_batch_size": 1,
        "num_workers": 8,
        "patch_size": 256,
        "learning_rate": 1e-4,
        "aux_learning_rate": 1e-3,
        "milestones": [20, 30],
        "gamma": 0.5,
        "lmbda": 5e-3,
        "task_lmbda": 5.0,
        "clip_max_norm": 1.0,
        "cuda": True,
        "gpu_id": 0,
        "seed": 42,
        "save": True,
        "eval_every": 1,
        "test_every": 1,
        "checkpoint": "",
        "pretrained": "",
        "resume": False,
        "TEST": False,
        "distortion": "mse",
        "model": "TIC_MoE",
        "N": 128,
        "M": 192,
        "enc_moe": True,
        "dec_moe": True,
        "h_moe": False,
        "moe_config": None,
        "train_moe": True,
        "dataset_kodak": "",
        "codec_align_divisor": 64,
    }

    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    if not hasattr(args, "dataset_path"):
        raise ValueError("Missing required key: dataset_path")

    if task_type == "detection":
        if not hasattr(args, "detectron_cfg"):
            args.detectron_cfg = "config/faster_rcnn_R_50_FPN_3x.yaml"
        if not (hasattr(args, "fastrcnn_path") or hasattr(args, "task_model_weights")):
            raise ValueError("Missing required key: fastrcnn_path (or task_model_weights)")
    elif task_type == "segmentation":
        if not hasattr(args, "detectron_cfg"):
            args.detectron_cfg = "config/mask_rcnn_R_50_FPN_3x.yaml"
        if not (hasattr(args, "maskrcnn_path") or hasattr(args, "task_model_weights")):
            raise ValueError("Missing required key: maskrcnn_path (or task_model_weights)")

    return args


def main_task(argv, task_type="detection", default_config="config/detection.yaml"):
    args = parse_args(argv, task_type=task_type, default_config=default_config)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    out_dir = init_out_dir(args, task_name=task_type)
    args.out_dir = out_dir
    setup_logger(os.path.join(out_dir, "train.log"))

    logging.info("========== %s (%s) =========", args.name, task_type)
    logging.info("Config: %s", args.config)
    logging.info("Model: %s  | N=%s M=%s", args.model, args.N, args.M)
    logging.info("Task: %s  | task_lmbda=%.4f", task_type, float(args.task_lmbda))

    train_loader, val_loader, test_loader, evaluator, predictor, criterion_task = build_task_components(
        args, device, task_type=task_type
    )

    net = build_model(args).to(device)
    if args.cuda and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    criterion_rd = RateDistortionLoss(lmbda=args.lmbda, distortion=args.distortion).to(device)

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
        test_epoch_coco(
            net, criterion_rd, criterion_task, test_loader,
            predictor, evaluator,
            codec_align_divisor=int(args.codec_align_divisor),
        )
        return

    for epoch in range(last_epoch, int(args.epochs)):
        logging.info("Epoch %d / %d | lr=%.2e", epoch, int(args.epochs) - 1, optimizer.param_groups[0]["lr"])

        train_one_epoch(
            net, criterion_rd, criterion_task, train_loader,
            optimizer, aux_optimizer,
            clip_max_norm=float(args.clip_max_norm),
            task_lmbda=float(args.task_lmbda),
            codec_align_divisor=int(args.codec_align_divisor),
        )

        is_best = False
        if val_loader is not None and (((epoch + 1) % int(args.eval_every) == 0) or (epoch == int(args.epochs) - 1)):
            val_loss = val_epoch(
                net, criterion_rd, criterion_task, val_loader,
                task_lmbda=float(args.task_lmbda),
                codec_align_divisor=int(args.codec_align_divisor),
            )
            if val_loss is not None:
                is_best = val_loss < best_loss
                best_loss = min(best_loss, val_loss)
                logging.info("Best val total loss: %.4f", best_loss)

        if ((epoch + 1) % int(args.test_every) == 0) or (epoch == int(args.epochs) - 1):
            test_epoch_coco(
                net, criterion_rd, criterion_task, test_loader,
                predictor, evaluator,
                codec_align_divisor=int(args.codec_align_divisor),
            )

        lr_scheduler.step()

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": bare_net.state_dict(),
                    "loss": float(best_loss),
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else {},
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "args": vars(args),
                },
                is_best=is_best,
                out_dir=out_dir,
                filename="checkpoint.pth.tar",
            )


__all__ = ["main_task"]
