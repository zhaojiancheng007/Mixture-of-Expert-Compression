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
from detectron2.evaluation import COCOEvaluator
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from layers.moe_layers import SparseMoEBlock
from examples.models.moe import TIC_MoE, TIC_MoE_vis
from models.tic import TIC
from models.transtic import TIC_PromptModel_first2
from models.tic_sfma import TIC_SFMA
from utils.alignment import Alignment
from utils.detection_utils import init_detection
from utils.moe_dataset import create_moe_dataset, moe_collate_fn
from utils.predictor import ModPredictor
from utils.segmentation_utils import init_segmentation
# from utils.moe_dataset import ID_TO_TASK 
import matplotlib.pyplot as plt
import seaborn as sns

import yaml

import wandb

TASK_VOCAB = {"cls": 0, "det": 1, "seg": 2}

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


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def init(args):
    base_dir = os.path.join(str(args.root), str(args.exp_name), str(args.lmbda['task']))
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


def configure_optimizers(net, args):
    """Set optimizer for only the parameters for propmts"""


    parameters = {
        k
        for k, p in net.named_parameters()
        if "sfma" in k
    }

    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    return optimizer


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
    criterion_rd,
    criterion_cls,
    criterion_det,
    criterion_seg,
    train_dataloader,
    optimizer,
    lmbda,
    epoch
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
        task_ids = np.asarray(batch["task"])

        data_times.update(time.time() - start_time)

        iter_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        out_net = model(imgs)
        model_times.update(time.time() - iter_start)

        loss_timer = time.time()
        out_criterion = criterion_rd(out_net, imgs)

        mask_cls = task_ids == "cls"
        mask_det = task_ids == "det"
        mask_seg = task_ids == "seg"

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


        metrics = {
            "train/epoch": epoch,
            "train/batch": i,
            "train/loss": total_losses.val,
            "train/loss_avg": total_losses.avg,
            "train/bpp": bpps.val,
            "train/bpp_avg": bpps.avg,
            "train/perc_loss": perc_losses.val,
            "train/perc_loss_avg": perc_losses.avg,
        }
        if accus.count:
            metrics["train/acc"] = accus.val
            metrics["train/acc_avg"] = accus.avg
        if cls_losses.count:
            metrics["train/cls_perc"] = cls_losses.val
        if det_losses.count:
            metrics["train/det_perc"] = det_losses.val
        if seg_losses.count:
            metrics["train/seg_perc"] = seg_losses.val
        wandb.log(metrics)

        if i % 100 == 0 or i == len(train_dataloader) - 1:
            update_txt = (
                f"[{i * len(imgs)}/{len(train_dataloader.dataset)}] | data time: {data_times.avg:.4f} | "
                f"model time: {model_times.avg:.4f} | loss time: {loss_times.avg:.4f} | "
                f"Loss: {total_losses.avg:.3f} | perc loss: {perc_losses.avg:.5f} | "
                f"Bpp loss: {bpps.avg:.4f} | cls: {cls_losses.avg:.5f} | det: {det_losses.avg:.5f} | "
                f"seg: {seg_losses.avg:.5f} | Accs: {accus.avg:.3f}"
            )
            print(datetime.now(), update_txt)
        start_time = time.time()

    
def test_epoch(epoch, test_dataloader, model, criterion_rd, criterion_cls, lmbda, stage="test", base_dir=None):
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
    model_times = AverageMeter()

    with torch.no_grad():
        for i, (d, l) in enumerate(test_dataloader):
            if i < 100:
                print(i)
                d = d.to(device, non_blocking=True)
                l = l.to(device, non_blocking=True)
                task_ids = torch.full((d.size(0),), TASK_VOCAB["cls"], dtype=torch.long, device=device)
                
                model_start = time.time()
                out_net = model(d,task_ids)
                # out_net = model(d)
                model_times.update(time.time() - model_start)
            else:
                break
        print(f"model time: {model_times.avg:.4f}")
 
    return 0.0

def test_epoch_task(test_dataloader, model, criterion_rd, criterion_task, predictor, evaluator, task_name="det",base_dir=None):
    model.eval()
    device = next(model.parameters()).device
    pixel_mean = torch.Tensor([103.530, 116.280, 123.675]).view(-1, 1, 1).to(device)

    bpp_loss = AverageMeter()
    psnr = AverageMeter()
    feat_loss =AverageMeter()

    save_dir = os.path.join(base_dir,f'{task_name}')
    os.makedirs(save_dir,exist_ok=True)
    with torch.no_grad():
        tqdm_meter = enumerate(test_dataloader)
        base_dir = os.path.join(base_dir,f'{task_name}')
        for i, batch in tqdm_meter:
            with ExitStack() as stack:
                ## model to eval()
                if isinstance(predictor.model, nn.Module):
                    stack.enter_context(inference_context(predictor.model))
                stack.enter_context(torch.no_grad())

                if i in [1,8,16,24,29,122,127,133,134,135,137,138,1052,1061,1068,1073,1074,1090,3008,3028]:
                    print(i)
                    align = Alignment(divisor=256, mode='pad',padding_mode='constant').to(device)
                    #divisor: 256 for TIC,  64 for mbt2018mean and cheng2020anchor
                    rcnn_align = Alignment(divisor=32).to(device)

                    img = read_image(batch[0]["file_name"], format="BGR")
                    d = torch.stack([batch[0]['image'].float().div(255)]).flip(1).to(device)
                    align_d = align.align(d)

                    task_ids = torch.full((align_d.size(0),),
                                        TASK_VOCAB[task_name],
                                        dtype=torch.long, device=device)

                    out_net = model(align_d, task_ids, vis_dir=base_dir, i_name=i)
                    out_net['x_hat'] = align.resume(out_net['x_hat']).clamp_(0, 1)
                    out_criterion = criterion_rd(out_net, d)

                    out_net['likelihoods']['y'] = align.resume(out_net['likelihoods']['y']).clamp_(0, 1)
                    bitmap = (torch.log(out_net['likelihoods']['y'][0]) / (-math.log(2))).mean(dim=0).cpu().numpy()
                    bitmap = sns.heatmap(bitmap, cmap="viridis", vmin=0, vmax=1.55)
                    bpp = float(out_criterion["bpp_loss"])
                    psnr = float(out_criterion["psnr"])

                    plt.axis('off')
                    plt.savefig(os.path.join(save_dir, f"{str(i).zfill(6)}_bpp{bpp:.3f}_BitMap.jpg"), dpi=300, bbox_inches="tight")
                    plt.clf()

                    recon = out_net['x_hat'][0].squeeze().permute(1,2,0).clamp(min=0, max=1).cpu().numpy()
                    plt.axis('off')
                    plt.imsave(os.path.join(save_dir, f"{str(i).zfill(6)}_bpp{bpp:.3f}_psnr{psnr:.3f}_Recon.jpg"), recon, dpi=300)
                    plt.clf()
                else:
                    continue
    return 0.0

def save_checkpoint(state, base_dir, filename="checkpoint.pth.tar"):
    ckpt_path = os.path.join(base_dir, filename)
    torch.save(state, ckpt_path)
    shutil.copyfile(ckpt_path, os.path.join(base_dir, "checkpoint_best_loss.pth.tar"))

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

    args = parser.parse_args(remaining)
    return args


def main(argv):
    args = parse_args(argv)
    base_dir = init(args)

    if args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    try:
        import wandb
        if not os.getenv("WANDB_API_KEY"):
            wandb.login()
        else:
            wandb.login(key=os.getenv("WANDB_API_KEY"))
        config_name = f"task_lmbda_{args.lmbda['task']}"
        wandb.init(project='moe_compress',name=config_name)
    except wandb.exc.LaunchError:
            logging.info("Could not initialize wandb. Logging is disabled.")

    setup_logger(base_dir + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    msg = f'======================= {args.name} ======================='
    logging.info(msg)
    for k in args.__dict__:
        logging.info(k + ':' + str(args.__dict__[k]))
    logging.info('=' * len(msg))

    train_dataset = create_moe_dataset(args.dataset_path, split='train', tasks=args.tasks, length=args.data_length)
    val_dataset = create_moe_dataset(args.dataset_path, split='val')
    
    
    train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
    pin_memory=(device == "cuda"),
    # collate_fn=moe_collate_fn
    )

    cls_val_dataloader = DataLoader(val_dataset['cls'],batch_size=args.test_batch_size,num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)
    det_val_dataloader = DataLoader(val_dataset['det'],batch_size=args.test_batch_size,num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)
    seg_val_dataloader = DataLoader(val_dataset['seg'],batch_size=args.test_batch_size,num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)

    cls_test_dataloader = cls_val_dataloader

    net = TIC_MoE(N=128,M=192, args=args)
    # net = TIC(N=128,M=192)
    # net = TIC_SFMA(N=128,M=192)
    # net = TIC_PromptModel_first2(prompt_config=args)
    net = net.to(device)
    print('total paramaters:',sum(p.numel() for p in net.parameters() )/1e6)
 
    # trainable = []
    # for p in net.parameters():
    #     p.requires_grad = False
    # for name, p in net.named_parameters():
    #     if "moe" in name.lower():
    #         p.requires_grad = True
    #         trainable.append(p)
    #         print(f"[TRAIN] {name} | shape={tuple(p.shape)} | params={p.numel()/1e6:.4f}M")
    #  total_params = sc
    # print(f"Training MoE parameters: {total_params:.2f}M")

    gate = []
    for p in net.parameters():
        p.requires_grad = False
    for name, p in net.named_parameters():
        if "gate" in name.lower():
            p.requires_grad = True
            gate.append(p)
            print(f"[TRAIN] {name} | shape={tuple(p.shape)} | params={p.numel()/1e6:.4f}M")
    gate_params = sum(p.numel() for p in gate) / 1e6
    print(f"Training Gate parameters: {gate_params:.2f}M")



    # optimizer = optim.AdamW(
    #     trainable,
    #     lr=args.learning_rate,
    #     weight_decay=1e-2
    # )
    # sched_warm = torch.optim.lr_scheduler.LinearLR(
    #     optimizer, start_factor=1.0, end_factor=1.0, total_iters=args.warmup_epochs
    # )
    # sched_main = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[5,10,15], gamma=0.1
    # )
    # lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
    #     optimizer, schedulers=[sched_warm, sched_main], milestones=[args.warmup_epochs]
    # )

    rdcriterion = RateDistortionLoss()
    clscriterion = Clsloss(device, True)

    det_test_dataloader, det_evaluator, det_predictor, criterion_det = init_detection(args, device)
    seg_test_dataloader, seg_evaluator, seg_predictor, criterion_seg = init_segmentation(args, device)

    last_epoch = 0
    best_loss = float("inf")
    if args.checkpoint: 
        logging.info("Loading "+str(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if args.resume:  
            new_state_dict = checkpoint["state_dict"]
            if list(new_state_dict.keys())[0].startswith("module."):
                from collections import OrderedDict
                new_state_dict = OrderedDict((k[7:], v) for k, v in new_state_dict.items())

            drop_keys = [k for k in new_state_dict if k.endswith("attn_mask") or ".attn_mask" in k]
            for k in drop_keys:
                new_state_dict.pop(k)
            load_info = net.load_state_dict(new_state_dict, strict=False)
            logging.info(f"[Resumed] from MoE checkpoint:{args.checkpoint}")
        else:
            logging.info(f"[MoE init] Copied moe layer from mlp")
            base_state_dict = checkpoint["state_dict"]
            if list(base_state_dict.keys())[0].startswith("module."):
                from collections import OrderedDict
                base_state_dict = OrderedDict((k[7:], v) for k, v in base_state_dict.items())
            net.load_state_dict(base_state_dict, strict=False)

            for name, module in net.named_modules():
                if isinstance(module, SparseMoEBlock):
                    mlp_prefix = ".".join(name.split(".")[:-1]) + ".mlp"
                    for i, expert in enumerate(module.experts):
                        for k, v in expert.state_dict().items():
                            orig_key = f"{mlp_prefix}.{k}"
                            if orig_key in base_state_dict:
                                with torch.no_grad():
                                    param = dict(expert.named_parameters())[k]
                                    param.copy_(base_state_dict[orig_key])

    if args.TEST:
        loss = test_epoch(last_epoch, cls_val_dataloader, net, rdcriterion,clscriterion, args.lmbda['task'],'val',base_dir)
        # test_epoch_task(det_test_dataloader, net, rdcriterion, criterion_det, det_predictor, det_evaluator, 'det',base_dir)
        # test_epoch_task(seg_test_dataloader, net, rdcriterion, criterion_seg, seg_predictor, seg_evaluator, 'seg', base_dir)
        return

    tqrange = tqdm.trange(last_epoch, args.epochs)
    for epoch in tqrange:
        train_one_epoch(
            net,
            rdcriterion,
            clscriterion,
            criterion_det,
            criterion_seg,
            train_dataloader,
            optimizer,
            args.lmbda,
            epoch
        )
        if epoch % 5 == 0 or epoch == args.epochs:
            loss = test_epoch(epoch, cls_val_dataloader, net, rdcriterion,clscriterion, args.lmbda,'val',base_dir)
            det_evaluator.reset()
            test_epoch_task(det_test_dataloader, net, rdcriterion, criterion_det, det_predictor, det_evaluator, "det",base_dir)
            seg_evaluator.reset()
            test_epoch_task(seg_test_dataloader, net, rdcriterion, criterion_seg, seg_predictor, seg_evaluator, "seg",base_dir)

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                base_dir,
                filename=f'checkpoint_{epoch}.pth.tar'
            )
        

        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        print('best_loss:',best_loss)

if __name__ == "__main__":
    main(sys.argv[1:])
