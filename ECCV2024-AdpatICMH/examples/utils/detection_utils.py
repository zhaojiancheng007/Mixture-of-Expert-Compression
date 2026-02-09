
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader
from detectron2.data.detection_utils import read_image
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone

from utils.predictor import ModPredictor
from utils.alignment import Alignment

import torch
import torch.nn as nn

from contextlib import ExitStack, contextmanager

from collections import OrderedDict
import pickle

from datetime import datetime

## Function for model to eval
@contextmanager
def inference_context(model):
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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

def init_detection(args, device):

    cfg = get_cfg() # get default cfg
    cfg.merge_from_file("./config/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = args.fastrcnn_path

    json_path = args.dataset_path + "/coco2017/annotations/instances_val2017.json"
    image_path = args.dataset_path + "/coco2017/val2017"
    register_coco_instances("compressed_coco_det", {}, json_path, image_path)
    evaluator = COCOEvaluator("compressed_coco_det", cfg, False, output_dir="./coco_log")
    test_dataloader = build_detection_test_loader(cfg, "compressed_coco_det")
    evaluator.reset()

    cfg.MODEL.META_ARCHITECTURE = 'GeneralizedRCNN_with_Rate'
    predictor = ModPredictor(cfg)
    taskcriterion = TaskLoss(cfg, device)

    return test_dataloader, evaluator, predictor, taskcriterion

