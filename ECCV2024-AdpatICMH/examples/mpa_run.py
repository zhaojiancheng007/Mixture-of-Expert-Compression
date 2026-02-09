# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import logging
import math
import os
import random
import shutil
import sys
import time
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from datetime import datetime
from numbers import Number
from typing import Callable, Optional, Tuple

import numpy as np
import pickle
import torch
import torch.distributed as dist 
import torch.nn as nn
import torch.optim as optim
import tqdm
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.data.detection_utils import read_image
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from detectron2.utils import comm
from torch.nn.parallel import DistributedDataParallel as DDP  # [DDP]
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler  # [DDP]
from datetime import timedelta
from torchvision import transforms
from torchvision.models import resnet50

from layers.layers_mpa import SparseMoEBlock
from models.mpa import MPA


from utils.alignment import Alignment
from utils.moe_dataset import create_moe_dataset, moe_collate_fn, TASK_VOCAB
from utils.predictor import ModPredictor
import yaml
import wandb


_EXPERT_INIT_DELTA = 1e-2    
_EXPERT_INIT_MODE  = "gaussian"   
_EXPERT_INIT_MASKR = 0.0 
def _tensor_std_safe(t: torch.Tensor) -> torch.Tensor:
    std = t.std().detach()
    if (not torch.isfinite(std)) or std <= 0:
        std = torch.as_tensor(1.0, device=t.device, dtype=t.dtype)
    # 加一个很小的 eps，防止极小方差
    eps = torch.finfo(t.dtype).eps if t.is_floating_point() else 1e-8
    return std + eps

def _perturb_copy_(dst_param: torch.nn.Parameter,
                   src_tensor: torch.Tensor,
                   delta: float,
                   mode: str,
                   mask_ratio: float):

    # 拷贝
    dst_param.data.copy_(src_tensor)

    # 仅扰动可训练浮点参数
    if (not dst_param.requires_grad) or (not dst_param.data.is_floating_point()):
        return

    # 跳过 LN 仿射（更稳）
    pname = getattr(dst_param, "_param_name", "").lower()
    if "ln" in pname or "layernorm" in pname or pname.endswith(".bias") and "norm" in pname:
        return

    if mode == "gaussian":
        std_scale = _tensor_std_safe(src_tensor) * float(delta)
        noise = torch.randn_like(dst_param.data) * std_scale
        dst_param.data.add_(noise)

    elif mode == "scaled":
        mul = 1.0 + (torch.randn_like(dst_param.data) * float(delta))
        dst_param.data.mul_(mul)

    elif mode == "mask":
        if mask_ratio <= 0.0:
            return
        mask = (torch.rand_like(dst_param.data) > mask_ratio).to(dst_param.data.dtype)
        dst_param.data.mul_(mask + (1 - mask) * (1.0 - float(delta)))



@contextmanager
def inference_context(model):
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
    
    def psnr(self, output, target):
        mse = torch.mean((output - target) ** 2)
        if(mse == 0):
            return 100
        max_pixel = 1.
        psnr = 10 * torch.log10(max_pixel / mse)
        return torch.mean(psnr)

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["rdloss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
        out["psnr"] = self.psnr(torch.clamp(output["x_hat"],0,1), target)
        return out


class FeatureHook():
    def __init__(self, module):
        module.register_forward_hook(self.attach)
    
    def attach(self, model, input, output):
        self.feature = output


class Clsloss(nn.Module):
    def __init__(self, device, perceptual_loss=False) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.classifier = resnet50(True)
        self.classifier.requires_grad_(False)
        self.hooks = [FeatureHook(i) for i in [
            self.classifier.layer1,
            self.classifier.layer2,
            self.classifier.layer3,
            self.classifier.layer4,
        ]]
        self.classifier = self.classifier.to(device)
        for k, p in self.classifier.named_parameters():
            p.requires_grad = False
        self.classifier.eval()
        self.perceptual_loss = perceptual_loss
        self.transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def forward(self, output, d, y_true):
        x_hat = torch.clamp(output,0,1)
        pred = self.classifier(self.transform(x_hat))
        loss = self.ce(pred, y_true)
        accu = sum(torch.argmax(pred,-1)==y_true)/pred.shape[0]
        if self.perceptual_loss:
            pred_feat = [i.feature.clone() for i in self.hooks]
            _ = self.classifier(self.transform(d))
            ori_feat = [i.feature.clone() for i in self.hooks]
            perc_loss = torch.stack([nn.functional.mse_loss(p,o, reduction='none').mean((1,2,3)) for p,o in zip(pred_feat, ori_feat)])
            perc_loss = perc_loss.mean()
            return loss, accu, perc_loss

        return loss, accu, None

class TaskLoss(nn.Module):
    def __init__(self, cfg, device) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.task_net = build_resnet_fpn_backbone(cfg, ShapeSpec(channels=3))
        checkpoint = OrderedDict()
        with open(cfg.MODEL.WEIGHTS, 'rb') as f:
            FPN_ckpt = pickle.load(f)
            for k, v in FPN_ckpt['model'].items():
                if 'backbone' in k:
                    checkpoint['.'.join(k.split('.')[1:])] = torch.from_numpy(v)
        self.task_net.load_state_dict(checkpoint, strict=True)
        self.task_net = self.task_net.to(device)
        for k, p in self.task_net.named_parameters():
            p.requires_grad = False
        self.task_net.eval()
        self.align = Alignment(divisor=32).to(device)
        self.pixel_mean = torch.Tensor([103.530, 116.280, 123.675]).view(-1, 1, 1).to(device)

    def forward(self, output, d, train_mode=False):
        with torch.no_grad():
            ## Ground truth for perceptual loss
            d = d.flip(1).mul(255)
            d = d - self.pixel_mean
            if not train_mode:
                d = self.align.align(d)
            gt_out = self.task_net(d)
        
        x_hat = torch.clamp(output, 0, 1)
        x_hat = x_hat.flip(1).mul(255)
        x_hat = x_hat - self.pixel_mean
        if not train_mode:
            x_hat = self.align.align(x_hat)
        task_net_out = self.task_net(x_hat)

        distortion_p2 = nn.MSELoss(reduction='none')(gt_out["p2"], task_net_out["p2"])
        distortion_p3 = nn.MSELoss(reduction='none')(gt_out["p3"], task_net_out["p3"])
        distortion_p4 = nn.MSELoss(reduction='none')(gt_out["p4"], task_net_out["p4"])
        distortion_p5 = nn.MSELoss(reduction='none')(gt_out["p5"], task_net_out["p5"])
        distortion_p6 = nn.MSELoss(reduction='none')(gt_out["p6"], task_net_out["p6"])

        return 0.2*(distortion_p2.mean()+distortion_p3.mean()+distortion_p4.mean()+distortion_p5.mean()+distortion_p6.mean())


def init_detection_train(args, device):
    cfg = get_cfg()
    cfg.merge_from_file("./config/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = args.fastrcnn_path
    taskcriterion = TaskLoss(cfg, device)
    return taskcriterion

def init_segmentation_train(args, device):
    cfg = get_cfg()
    cfg.merge_from_file("./config/mask_rcnn_R_50_FPN_3x.yaml")  # 你原来用的那个
    cfg.MODEL.WEIGHTS = args.maskrcnn_path
    taskcriterion = TaskLoss(cfg, device)
    return taskcriterion

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0


def init(args):
    base_dir = os.path.join(str(args.root), str(args.exp_name), str(args.lmbda['task']), str(args.moe_config['num_experts']))
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def setup_logger(log_path: str) -> None:
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Reset handlers to avoid duplicate logs when running multiple times.
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(stream_handler)

    logging.info("Logging file is %s", log_path)


def setup_device(args) -> torch.device:
    use_cuda = bool(args.cuda) and torch.cuda.is_available()
    if use_cuda:
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(device)
    else:
        if getattr(args, "cuda", False) and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available; falling back to CPU")
        device = torch.device("cpu")
    return device

def train_one_epoch(
    model,
    # ema,
    criterion_rd,
    criterion_cls,
    criterion_det,
    criterion_seg,
    train_dataloader,
    optimizer,
    lmbda,
    epoch,
    world_size,
    lr_scheduler
):
    model.train()
    device = next(model.parameters()).device
    bpps = AverageMeter()
    accus = AverageMeter()
    total_losses = AverageMeter()
    perc_losses = AverageMeter()
    cls_losses = AverageMeter()
    det_losses = AverageMeter()
    seg_losses = AverageMeter()
    data_times = AverageMeter()
    model_times = AverageMeter()
    loss_times = AverageMeter()

    start_time = time.time()
    for i, batch in enumerate(train_dataloader):
        images, labels = batch["sample"]
        imgs = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        task_ids = batch["task_id"].to(device, non_blocking=True)

        data_times.update(time.time() - start_time)
        
        optimizer.zero_grad(set_to_none=True)

        iter_start = time.time()
        out_net = model(imgs)
        model_times.update(time.time() - iter_start)

        loss_timer = time.time()
        out_criterion = criterion_rd(out_net, imgs)

        # TASK_VOCAB = {"cls": 0, "det": 1, "seg": 2}
        mask_cls = torch.tensor([t == 0 for t in task_ids], device=device, dtype=torch.bool)
        mask_det = torch.tensor([t == 1 for t in task_ids], device=device, dtype=torch.bool)
        mask_seg = torch.tensor([t == 2 for t in task_ids], device=device, dtype=torch.bool)

        perc_terms = []
        cls_perc_loss = det_perc_loss = seg_perc_loss = None
        accu = None

        if mask_cls.any():
            _, accu, cls_perc_loss = criterion_cls(
                out_net["x_hat"][mask_cls], imgs[mask_cls], labels[mask_cls]
            )
            if cls_perc_loss is not None:
                perc_terms.append(lmbda['cls'] * cls_perc_loss)

        if mask_det.any():
            det_perc_loss = criterion_det(
                out_net["x_hat"][mask_det], imgs[mask_det], train_mode=True
            )
            perc_terms.append(lmbda['det'] * det_perc_loss)

        if mask_seg.any():
            seg_perc_loss = criterion_seg(
                out_net["x_hat"][mask_seg], imgs[mask_seg], train_mode=True
            )
            perc_terms.append(lmbda['seg'] * seg_perc_loss)

        perc_loss = torch.zeros((), device=device)
        if perc_terms:
            perc_loss = torch.stack(perc_terms).sum()

        loss_times.update(time.time() - loss_timer)

        total_loss = lmbda['task'] * perc_loss + out_criterion["bpp_loss"]
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()


        # ema.update()

        batch_size = imgs.size(0)
        if accu is not None:
            accus.update(accu.item(), batch_size)
        bpps.update(out_criterion["bpp_loss"], batch_size)
        total_losses.update(total_loss.item(), batch_size)
        perc_losses.update(perc_loss, batch_size)
        if cls_perc_loss is not None:
            cls_losses.update(cls_perc_loss, batch_size)
        if det_perc_loss is not None:
            det_losses.update(det_perc_loss, batch_size)
        if seg_perc_loss is not None:
            seg_losses.update(seg_perc_loss, batch_size)

        # [DDP] 仅主进程做 wandb log 与打印
        metrics = {
            "train/epoch": epoch,
            "train/batch": i,
            # "train/loss": total_losses.val,
            "train/loss_avg": total_losses.avg,
            # "train/bpp": bpps.val,
            "train/bpp_avg": bpps.avg,
            # "train/perc_loss": perc_losses.val,
            "train/perc_loss_avg": perc_losses.avg,
        }
        if accus.count:
            # metrics["train/acc"] = accus.val
            metrics["train/acc_avg"] = accus.avg
        if cls_losses.count:
            # metrics["train/cls_perc"] = cls_losses.val
            metrics["train/cls_avg"] = cls_losses.avg
        if det_losses.count:
            # metrics["train/det_perc"] = det_losses.val
            metrics["train/det_avg"] = det_losses.avg
        if seg_losses.count:
            # metrics["train/seg_perc"] = seg_losses.val
            metrics["train/seg_avg"] = seg_losses.avg
        if is_main_process():  # [DDP] 修改
            wandb.log(metrics)

        if (i % 100 == 0 or i == len(train_dataloader) - 1) and is_main_process():  # [DDP] 修改
            update_txt = (
                f"[{i * len(imgs) * world_size}/{len(train_dataloader.dataset)}] | data time: {data_times.avg:.4f} | "
                f"model time: {model_times.avg:.4f} | loss time: {loss_times.avg:.4f} | "
                f"Loss: {total_losses.avg:.3f} | perc loss: {perc_losses.avg:.5f} | "
                f"Bpp loss: {bpps.avg:.4f} | cls: {cls_losses.avg:.5f} | det: {det_losses.avg:.5f} | "
                f"seg: {seg_losses.avg:.5f} | Accs: {accus.avg:.3f}"
            )
            print(datetime.now(), update_txt)
        start_time = time.time()

def test_epoch(epoch, test_dataloader, model, criterion_rd, criterion_cls, lmbda, stage="test"):

    model.eval()
    device = next(model.parameters()).device

    loss_am = AverageMeter()
    percloss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    accuracy = AverageMeter()
    totalloss = AverageMeter()

    with torch.no_grad():
        for i, (d, l) in enumerate(test_dataloader):
            d = d.to(device, non_blocking=True)
            l = l.to(device, non_blocking=True)

            task_ids = torch.full((d.size(0),), TASK_VOCAB["cls"], dtype=torch.long, device=device)
            out_net = model(d)
            # out_net = model(d,task_ids)
            out_criterion = criterion_rd(out_net, d)
            loss, accu, perc_loss = criterion_cls(out_net["x_hat"], d, l)
            total_loss = lmbda['task'] * perc_loss + out_criterion["bpp_loss"]

            batch_size = d.size(0)
            aux_loss.update(model.module.aux_loss() if isinstance(model, DDP) else model.aux_loss(), batch_size)  # [DDP] 兼容
            bpp_loss.update(out_criterion["bpp_loss"], batch_size)
            loss_am.update(loss, batch_size)
            mse_loss.update(out_criterion["mse_loss"], batch_size)
            psnr.update(out_criterion["psnr"], batch_size)
            accuracy.update(accu, batch_size)
            percloss.update(perc_loss, batch_size)
            totalloss.update(total_loss, batch_size)

            if i % 100 == 0 or i == len(test_dataloader) - 1:
                print(
                    datetime.now(),
                    f"[{i * len(d)}/{len(test_dataloader.dataset)}]",
                    f"{epoch}  total loss:{totalloss.avg:.3f}| bpp loss: {bpp_loss.avg:.5f} | psnr: {psnr.avg:.5f} | accu: {accuracy.avg:.5f}",
                )

    model.train()
    print(
        f"{epoch} total loss:{totalloss.avg:.3f}| bpp loss: {bpp_loss.avg:.5f} | psnr: {psnr.avg:.5f} | accu: {accuracy.avg:.5f}"
    )
    return totalloss.avg

def test_epoch_task(test_dataloader, model, criterion_rd, criterion_task, predictor, evaluator,distributed=False, task_name='det'):
    # 不要在这里提前 return；所有 rank 都要跑，D2 自己做聚合
    model.eval()
    device = next(model.parameters()).device
    pixel_mean = torch.Tensor([103.530, 116.280, 123.675]).view(-1, 1, 1).to(device)

    bpp_loss = AverageMeter()
    psnr = AverageMeter()
    feat_loss = AverageMeter()

    with torch.no_grad():
        tqdm_meter = enumerate(test_dataloader)
        for i, batch in tqdm_meter:
            with ExitStack() as stack:
                if isinstance(predictor.model, nn.Module):
                    stack.enter_context(inference_context(predictor.model))
                stack.enter_context(torch.no_grad())

                align = Alignment(divisor=256, mode='pad', padding_mode='constant').to(device)
                rcnn_align = Alignment(divisor=32).to(device)

                img = read_image(batch[0]["file_name"], format="BGR")
                d = torch.stack([batch[0]['image'].float().div(255)]).flip(1).to(device)
                align_d = align.align(d)

                task_ids = torch.full((align_d.size(0),),
                                      TASK_VOCAB[task_name],
                                      dtype=torch.long, device=device)
                out_net = model(align_d)
                out_net['x_hat'] = align.resume(out_net['x_hat']).clamp_(0, 1) 
                out_criterion = criterion_rd(out_net, d) 

                # 任务感知特征损失
                perc_loss = criterion_task(out_net['x_hat'], d)
                feat_loss.update(perc_loss.item())

                # === RCNN 输入构造 ===
                trand_y_tilde = out_net['x_hat'].flip(1).mul(255)
                trand_y_tilde = rcnn_align.align(trand_y_tilde - pixel_mean)

                bpp_loss.update(out_criterion["bpp_loss"])
                psnr.update(out_criterion['psnr'])

                predictions = predictor(img, trand_y_tilde)
                evaluator.process(batch, [predictions])

            if i % 500 == 0:
                print(datetime.now(), f"|{i*len(d)}/{len(test_dataloader.dataset)}] "
                                      f"|Bpp loss: {bpp_loss.avg:.4f} | PSNR loss: {psnr.avg:.4f}")

    results = None
    if (not distributed) or is_main_process():
        print(f"bpp loss: {bpp_loss.avg:.5f} | feat loss:{feat_loss.avg:.4f} | psnr: {psnr.avg:.5f}")
        results = evaluator.evaluate()

    model.train()
    return results

def save_checkpoint(state, base_dir, filename="checkpoint.pth.tar"):
    ckpt_path = os.path.join(base_dir, filename)
    torch.save(state, ckpt_path)
    print(f'[{filename}] save to {ckpt_path}')


def save_yaml_config(args, base_dir):
    """
    将原始 YAML 复制到 {base_dir}/configs/config_original.yaml，
    并导出合并后的运行时配置到 {base_dir}/configs/config_runtime.yaml
    仅在主进程调用。
    """
    os.makedirs(os.path.join(base_dir, "configs"), exist_ok=True)
    cfg_dir = os.path.join(base_dir, "configs")

    # 1) 复制原始 YAML
    if hasattr(args, "config") and isinstance(args.config, str) and os.path.isfile(args.config):
        dst = os.path.join(cfg_dir, "config_original.yaml")
        try:
            shutil.copy2(args.config, dst)
            logging.info(f"Copied original YAML to {dst}")
        except Exception as e:
            logging.warning(f"Could not copy original config '{args.config}': {e}")
    else:
        logging.warning("args.config not set or file missing; skip copying original YAML.")

    # 2) 导出合并后的运行时配置
    def _to_yamlable(x):
        import numpy as _np
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        if isinstance(x, (list, tuple)):
            return [_to_yamlable(i) for i in x]
        if isinstance(x, dict):
            return {str(k): _to_yamlable(v) for k, v in x.items()}
        # 常见不可序列化类型的兜底处理
        if isinstance(x, torch.device):
            return str(x)
        if isinstance(x, _np.generic):
            return x.item()
        return str(x)

    merged = {k: _to_yamlable(v) for k, v in vars(args).items()}
    runtime_path = os.path.join(cfg_dir, "config_runtime.yaml")
    try:
        with open(runtime_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(merged, f, allow_unicode=True, sort_keys=False)
        logging.info(f"Saved merged runtime config to {runtime_path}")
    except Exception as e:
        logging.warning(f"Could not write runtime config yaml: {e}")

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-c",
        "--config",
        default="config/vpt_default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        '--name', 
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), 
        type=str,
        help='Result dir name', 
    )
    given_configs, remaining = parser.parse_known_args(argv)
    with open(given_configs.config) as file:
        yaml_data= yaml.safe_load(file)
        parser.set_defaults(**yaml_data)
    
    parser.add_argument(
        "-T",
        "--TEST",
        action='store_true',
        help='Testing'
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    # [DDP] 可选：显式指定是否使用 DDP（默认依据环境变量自动判定）
    parser.add_argument("--ddp", action="store_true", help="Force enable DDP if env set properly")

    args = parser.parse_args(remaining)
    return args

def is_main_process():
    # 保留你的实现
    return not dist.is_initialized() or dist.get_rank() == 0


def main(argv):
    args = parse_args(argv)
    base_dir = init(args)

    use_ddp_env = ("RANK" in os.environ and "WORLD_SIZE" in os.environ)
    if use_ddp_env:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=100))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        if args.cuda and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu_id)
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        local_rank = 0

    # [DDP] rank/world_size
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if args.seed is not None:
        torch.manual_seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        random.seed(args.seed + rank)
    
    if is_main_process():
        try:
            if not os.getenv("WANDB_API_KEY"):
                wandb.login()
            else:
                wandb.login(key=os.getenv("WANDB_API_KEY"))
            config_name = f"task_lmbda_{args.lmbda['task']}"
            wandb.init(project='mpamoe_compress',name=config_name)
        except wandb.exc.LaunchError:
                logging.info("Could not initialize wandb. Logging is disabled.")

    if is_main_process():
        setup_logger(base_dir + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
        msg = f'======================= {args.exp_name} ======================='
        logging.info(msg)
        for k in args.__dict__:
            logging.info(k + ':' + str(args.__dict__[k]))
        logging.info('=' * len(msg))

    # -------------------------------
    # Datasets & Samplers
    # -------------------------------
    train_dataset = create_moe_dataset(args.dataset_path, split='train', tasks=args.tasks, length=args.data_length, init_probs=args.init_prob)
    val_dataset = create_moe_dataset(args.dataset_path, split='val')

    # [DDP] 使用 DistributedSampler（训练集）
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    ) if dist.is_initialized() else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,  # [DDP] per-GPU batch size
        num_workers=args.num_workers,
        sampler=train_sampler,       # [DDP] sampler 替换 shuffle
        shuffle=False if train_sampler is not None else True,
        pin_memory=True,
        drop_last=True
        # collate_fn=moe_collate_fn
    )

    cls_val_dataloader = DataLoader(val_dataset['cls'], batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    # -------------------------------
    # Model & Optim
    # -------------------------------
    net = MPA().to(device)
    trainable = []
    for p in net.parameters():
        p.requires_grad = False
    for name, p in net.named_parameters():
        if "moe" in name.lower():
            p.requires_grad = True
            trainable.append(p)
            if is_main_process():
                print(f"[TRAIN] {name} | shape={tuple(p.shape)} | params={p.numel()/1e6:.4f}M")

    ddp_kwargs = dict(device_ids=[local_rank], output_device=local_rank,
                  find_unused_parameters=True, broadcast_buffers=False)
    if dist.is_initialized():
        net = DDP(net, **ddp_kwargs)

    total_params = sum(p.numel() for p in net.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
    if is_main_process():
        print(f"Total parameters: {total_params:.2f}M")
        print(f"Trainable parameters: {trainable_params:.2f}M")

    # [DDP]：base_lr * world_size
    base_lr = args.learning_rate
    scaled_lr = base_lr * world_size 
    optimizer = optim.AdamW(
        trainable,
        lr=base_lr,
        weight_decay=1e-2
    )
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=args.epochs * len(train_dataset), 
    #     eta_min=1e-6    # min lr
    # )
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=args.milestones,
                                                  gamma=0.1)
    # criterion
    rdcriterion = RateDistortionLoss()
    clscriterion = Clsloss(device, True)
    criterion_det = init_detection_train(args, device)
    criterion_seg = init_segmentation_train(args, device)

    last_epoch = 0
    best_loss = float("inf")
    if getattr(args, "checkpoint", None):
        if is_main_process():
            logging.info("Loading " + str(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model_to_load = net.module if isinstance(net, DDP) else net

        if args.resume:
            new_state_dict = checkpoint["state_dict"]
            if list(new_state_dict.keys())[0].startswith("module."):
                new_state_dict = OrderedDict((k[7:], v) for k, v in new_state_dict.items())
            load_info = model_to_load.load_state_dict(new_state_dict, strict=False)
            if is_main_process():
                logging.info(f"[Resumed] from MoE checkpoint:{args.checkpoint}")

            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "lr_scheduler" in checkpoint:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            if "epoch" in checkpoint:
                last_epoch = checkpoint["epoch"] + 1
            if "best_loss" in checkpoint:
                best_loss = checkpoint["best_loss"]
        else:
            if is_main_process():
                logging.info(f"[MoE init] Copied moe layer from mlp")
            base_state_dict = checkpoint["state_dict"]
            if list(base_state_dict.keys())[0].startswith("module."):
                base_state_dict = OrderedDict((k[7:], v) for k, v in base_state_dict.items())
            model_to_load.load_state_dict(base_state_dict, strict=False)

            for name, module in model_to_load.named_modules():
                if isinstance(module, SparseMoEBlock):
                    mlp_prefix = ".".join(name.split(".")[:-1]) + ".mlp"
                    for i, expert in enumerate(module.experts):
                        for k, v in expert.state_dict().items():
                            orig_key = f"{mlp_prefix}.{k}"
                            if orig_key in base_state_dict:
                                with torch.no_grad():
                                    param = dict(expert.named_parameters())[k]
                                    param.copy_(base_state_dict[orig_key])

            # for name, module in model_to_load.named_modules():
                if isinstance(module, SparseMoEBlock):
                    mlp_prefix = ".".join(name.split(".")[:-1]) + ".mlp"

                    for expert_idx, expert in enumerate(module.experts):
                        for p_name, p in expert.named_parameters():
                            setattr(p, "_param_name", f"{name}.experts.{expert_idx}.{p_name}")

                        exp_params = dict(expert.named_parameters())
                        exp_buffers = dict(expert.named_buffers())

                        for k, v in expert.state_dict().items():
                            orig_key = f"{mlp_prefix}.{k}"
                            if orig_key not in base_state_dict:
                                continue

                            src = base_state_dict[orig_key].to(v.dtype).to(v.device)

                            if k in exp_params:
                                param = exp_params[k]
                                with torch.no_grad():
                                    _perturb_copy_(
                                        dst_param=param,
                                        src_tensor=src,
                                        delta=_EXPERT_INIT_DELTA,
                                        mode=_EXPERT_INIT_MODE,
                                        mask_ratio=_EXPERT_INIT_MASKR
                                    )
                            elif k in exp_buffers:
                                # buffers（如 running stats）不扰动，直接复制
                                with torch.no_grad():
                                    exp_buffers[k].copy_(src)
            


    tqrange = tqdm.trange(last_epoch, args.epochs, disable=not is_main_process())  # [DDP] 只在主进程显示进度条
    for epoch in tqrange:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_one_epoch(
            net,
            # ema,
            rdcriterion,
            clscriterion,
            criterion_det,
            criterion_seg,
            train_dataloader,
            optimizer,
            args.lmbda,
            epoch,
            world_size,
            lr_scheduler
        )

        if dist.is_initialized():
            dist.barrier()

        # [DDP] 验证与保存：仅主进程
        if is_main_process() and (epoch % 5 == 0 or epoch == args.epochs - 1):
            loss = test_epoch(epoch, cls_val_dataloader, net, rdcriterion, clscriterion, args.lmbda, 'val')

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": (net.module.state_dict() if isinstance(net, DDP) else net.state_dict()),  # [DDP]
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "best_loss": best_loss,
                    # "ema": ema.state_dict(), 
                },
                base_dir,
                filename=f'checkpoint_{epoch}.pth.tar'
            )
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            print('best_loss:', best_loss)
            save_yaml_config(args, base_dir)

        # [DDP] 同步（可选）
        if dist.is_initialized():
            dist.barrier()

        # lr_scheduler.step()

    # [DDP] 结束清理
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main(sys.argv[1:])
