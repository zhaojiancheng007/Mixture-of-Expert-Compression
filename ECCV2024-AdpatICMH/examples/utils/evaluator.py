import os
import torch
import numpy as np

from detectron2.data import MetadataCatalog
from detectron2.evaluation import PascalVOCDetectionEvaluator, SemSegEvaluator


class ClassificationEvaluator:
    def __init__(self, num_classes=20, threshold=0.5):
        self.num_classes = num_classes
        self.threshold = threshold
        self.total = 0
        self.correct = 0

    def reset(self):
        self.total = 0
        self.correct = 0

    def process(self, pred_logits, gt_labels):
        probs = torch.sigmoid(pred_logits)
        preds = (probs > self.threshold).float()
        correct = (preds == gt_labels).float().sum()
        total = torch.numel(preds)
        self.correct += correct.item()
        self.total += total

    def evaluate(self):
        acc = self.correct / self.total if self.total > 0 else 0
        print(f"Classification Accuracy: {acc:.4f}")
        return {"classification_acc": acc}


class EvaluatorWrapper:
    def __init__(self, voc_root, use_detection=True, use_segmentation=True, use_classification=True):
        self.use_detection = use_detection
        self.use_segmentation = use_segmentation
        self.use_classification = use_classification

        self.dataset_name = "voc_custom"

        # 注册 Metadata (只需一次)
        MetadataCatalog.get(self.dataset_name).set(
            thing_classes=[
                "aeroplane", "bicycle", "bird", "boat", "bottle",
                "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            ],
            dirname=os.path.join(voc_root, "VOC2012"),
            year=2012,
            split="val",
        )

        if self.use_detection:
            self.det_evaluator = PascalVOCDetectionEvaluator(self.dataset_name)
        if self.use_segmentation:
            self.seg_evaluator = SemSegEvaluator(self.dataset_name, distributed=False, num_classes=21, ignore_label=255)
        if self.use_classification:
            self.cls_evaluator = ClassificationEvaluator(num_classes=20)

    def reset(self):
        if self.use_detection:
            self.det_evaluator.reset()
        if self.use_segmentation:
            self.seg_evaluator.reset()
        if self.use_classification:
            self.cls_evaluator.reset()

    def process(self, data_batch, model_output, classifier=None):
        """
        data_batch: original batch from dataloader (dict)
        model_output: model output (dict, contains x_hat)
        classifier: classification head for prediction (if classification enabled)
        """

        device = model_output["x_hat"].device
        batch_size = model_output["x_hat"].shape[0]

        for b in range(batch_size):
            image_id = data_batch['image_id'][b]

            # --------- detection ---------
            if self.use_detection:
                # gt
                gt_input = {
                    "image_id": image_id,
                    "height": model_output["x_hat"].shape[2],
                    "width": model_output["x_hat"].shape[3],
                }
                # mock one prediction for demo purpose, you should replace this by your real detection output
                pred_instances = {
                    "instances": {
                        "pred_boxes": torch.tensor([[10, 20, 100, 200]], device=device), 
                        "scores": torch.tensor([0.9], device=device),
                        "pred_classes": torch.tensor([3], device=device)
                    }
                }
                self.det_evaluator.process([gt_input], [pred_instances])

            # --------- segmentation ---------
            if self.use_segmentation:
                gt_seg = data_batch["segmentation"][b]  # numpy array (H, W)
                # 假设你有 segmentation logits 或直接预测了 mask
                pred_seg = torch.argmax(model_output["x_hat"][b], dim=0).cpu().numpy()
                self.seg_evaluator.process([gt_seg], [pred_seg])

            # --------- classification ---------
            if self.use_classification:
                x_hat = torch.clamp(model_output["x_hat"][b], 0, 1).unsqueeze(0)  # shape (1, C, H, W)
                pred_logits = classifier(x_hat)  # shape (1, 20)
                gt_cls = data_batch["classification"][b].unsqueeze(0).to(device)
                self.cls_evaluator.process(pred_logits.cpu(), gt_cls.cpu())

    def evaluate(self):
        results = {}
        if self.use_detection:
            det_results = self.det_evaluator.evaluate()
            print("Detection Results:", det_results)
            results["detection"] = det_results
        if self.use_segmentation:
            seg_results = self.seg_evaluator.evaluate()
            print("Segmentation Results:", seg_results)
            results["segmentation"] = seg_results
        if self.use_classification:
            cls_results = self.cls_evaluator.evaluate()
            results["classification"] = cls_results
        return results
