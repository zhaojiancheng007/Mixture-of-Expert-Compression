import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np
from torchvision import transforms
from utils.alignment import Alignment

from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from detectron2.layers import ShapeSpec
from detectron2.config import get_cfg
from torchvision.models import resnet50

from collections import OrderedDict
import pickle

class FeatureHook():
    def __init__(self, module):
        module.register_forward_hook(self.attach)
    
    def attach(self, model, input, output):
        self.feature = output

class Clsloss(nn.Module):
    def __init__(self, device, perceptual_loss=False) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()# single label
        self.be = nn.BCEWithLogitsLoss()
        self.classifier = resnet50(True)

        num_classes = 20  # VOC 2012分类任务类别数
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)

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
        for p in self.classifier.fc.parameters():
            p.requires_grad = True
        self.classifier.eval()
        self.perceptual_loss = perceptual_loss
        self.transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def accuracy(self, pred_logits, y_true, threshold=0.5):
        probs = torch.sigmoid(pred_logits)
        preds = (probs > threshold).float()
        correct = (preds == y_true).float()
        acc = correct.mean()  # 直接输出 tensor
        return acc
    
    def forward(self, output, d, y_true):
        x_hat = torch.clamp(output["x_hat"], 0, 1)
        pred_logits = self.classifier(self.transform(x_hat))  # shape: [B, C]
        loss = self.be(pred_logits, y_true)  # y_true shape: [B, C]

        accu = self.accuracy(pred_logits, y_true)

        if self.perceptual_loss:
            pred_feat = [i.feature.clone() for i in self.hooks]
            _ = self.classifier(self.transform(d))
            ori_feat = [i.feature.clone() for i in self.hooks]
            perc_loss = torch.stack([
                F.mse_loss(p, o, reduction='none').mean((1, 2, 3)) 
                for p, o in zip(pred_feat, ori_feat)
            ])
            perc_loss = perc_loss.mean()
            return loss, accu, perc_loss

        return loss, accu, None

class DetectLoss(nn.Module):
    def __init__(self, cfg, device) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.task_net = build_resnet_fpn_backbone(cfg, ShapeSpec(channels=3))
        print('task net:', self.task_net)
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
        
        x_hat = torch.clamp(output["x_hat"], 0, 1)
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