import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np

import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop
from torchvision import transforms

class TrainVOCTransform:
    def __init__(self, resize_size=256, crop_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        # 先 resize
        img = TF.resize(img, self.resize_size)
        w, h = img.size

        # resize bbox
        if "boxes" in target:
            scale_x = self.resize_size / w
            scale_y = self.resize_size / h
            boxes = target["boxes"].copy()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target["boxes"] = boxes

        # resize segmentation
        if "segmentation" in target:
            mask = Image.fromarray(target["segmentation"])
            mask = TF.resize(mask, self.resize_size, interpolation=Image.NEAREST)
            target["segmentation"] = np.array(mask, dtype=np.int32)

        # 随机水平翻转
        if random.random() > 0.5:
            img = TF.hflip(img)
            w, _ = img.size
            if "boxes" in target:
                boxes = target["boxes"]
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
            if "segmentation" in target:
                mask = Image.fromarray(target["segmentation"])
                mask = TF.hflip(mask)
                target["segmentation"] = np.array(mask, dtype=np.int32)

        # 随机裁剪 (仅对图像和 mask做crop，bbox可以选择做clip，暂时不crop避免复杂)
        i, j, h_crop, w_crop = RandomCrop.get_params(img, output_size=(self.crop_size, self.crop_size))
        img = TF.crop(img, i, j, h_crop, w_crop)

        if "segmentation" in target:
            mask = Image.fromarray(target["segmentation"])
            mask = TF.crop(mask, i, j, h_crop, w_crop)
            target["segmentation"] = np.array(mask, dtype=np.int32)

        # bbox clip （简单安全处理）
        if "boxes" in target:
            boxes = target["boxes"]
            boxes[:, [0, 2]] -= j
            boxes[:, [1, 3]] -= i
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w_crop)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h_crop)
            target["boxes"] = boxes

        # 最后 ToTensor 和 Normalize
        img = TF.to_tensor(img)
        img = TF.normalize(img, self.mean, self.std)

        return img, target

class ValVOCTransform:
    def __init__(self, resize_size=256, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.resize_size = resize_size
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        w_orig, h_orig = img.size
        img = TF.resize(img, self.resize_size)

        if "boxes" in target:
            scale_x = self.resize_size / w_orig
            scale_y = self.resize_size / h_orig
            boxes = target["boxes"].copy()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target["boxes"] = boxes

        if "segmentation" in target:
            mask = Image.fromarray(target["segmentation"])
            mask = TF.resize(mask, self.resize_size, interpolation=Image.NEAREST)
            target["segmentation"] = np.array(mask, dtype=np.int32)

        img = TF.to_tensor(img)
        img = TF.normalize(img, self.mean, self.std)
        return img, target


class PASCAL_VOC(Dataset):
    def __init__(self, voc_root, year='2012', split='train', 
                 use_detection=True, use_segmentation=False, use_classification=False, 
                 transforms=None):
        """
        voc_root: VOCdevkit 路径
        year: '2007' or '2012'
        split: 'train', 'val', 'trainval'
        """
        self.voc_root = voc_root
        self.year = year
        self.split = split
        self.transforms = transforms

        self.use_detection = use_detection
        self.use_segmentation = use_segmentation
        self.use_classification = use_classification

        self.image_dir = os.path.join(voc_root, f'VOC{year}', 'JPEGImages')
        self.annotation_dir = os.path.join(voc_root, f'VOC{year}', 'Annotations')
        self.segmentation_dir = os.path.join(voc_root, f'VOC{year}', 'SegmentationClass')

        image_set_file = os.path.join(voc_root, f'VOC{year}', 'ImageSets', 'Segmentation', f'{split}.txt')
        with open(image_set_file, 'r') as f:
            self.image_ids = [x.strip() for x in f.readlines()]

        # VOC 20 classes
        self.class_names = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        self.class_name_to_id = {name: idx for idx, name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.image_ids)

    def parse_annotation(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []
        difficult = []

        for obj in root.findall("object"):
            name = obj.find("name").text.lower().strip()
            if name not in self.class_name_to_id:
                continue  # skip unknown class

            bbox = obj.find("bndbox")
            box = [
                float(bbox.find("xmin").text) - 1,
                float(bbox.find("ymin").text) - 1,
                float(bbox.find("xmax").text) - 1,
                float(bbox.find("ymax").text) - 1
            ]
            boxes.append(box)
            labels.append(self.class_name_to_id[name])

            difficult_flag = int(obj.find("difficult").text)
            difficult.append(difficult_flag)

        return np.array(boxes), np.array(labels), np.array(difficult)

    def parse_classification_labels(self, xml_path):
        # multi-label classification
        tree = ET.parse(xml_path)
        root = tree.getroot()
        labels = np.zeros(len(self.class_names), dtype=np.float32)
        for obj in root.findall("object"):
            name = obj.find("name").text.lower().strip()
            if name in self.class_name_to_id:
                labels[self.class_name_to_id[name]] = 1.0
        return labels

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        img = Image.open(img_path).convert("RGB")

        target = {"image_id": image_id}

        # detection: use numpy before transform
        if self.use_detection:
            xml_path = os.path.join(self.annotation_dir, f"{image_id}.xml")
            boxes, labels, difficult = self.parse_annotation(xml_path)
            target["boxes"] = np.array(boxes, dtype=np.float32)
            target["labels"] = np.array(labels, dtype=np.int64)
            target["difficult"] = np.array(difficult, dtype=np.int64)

        # segmentation: use numpy before transform
        if self.use_segmentation:
            seg_path = os.path.join(self.segmentation_dir, f"{image_id}.png")
            mask = Image.open(seg_path)
            target["segmentation"] = np.array(mask, dtype=np.uint8)

        # classification: use numpy before transform
        if self.use_classification:
            xml_path = os.path.join(self.annotation_dir, f"{image_id}.xml")
            cls_labels = self.parse_classification_labels(xml_path)
            target["classification"] = np.array(cls_labels, dtype=np.float32)

        # transforms applied to PIL image and raw labels
        if self.transforms:
            img, target = self.transforms(img, target)

        # convert to tensor AFTER transform
        target["image"] = transforms.ToTensor()(img) if isinstance(img, Image.Image) else img

        if "boxes" in target:
            target["boxes"] = torch.from_numpy(target["boxes"]).float()
        if "labels" in target:
            target["labels"] = torch.from_numpy(target["labels"]).long()
        if "difficult" in target:
            target["difficult"] = torch.from_numpy(target["difficult"]).long()
        if "segmentation" in target:
            target["segmentation"] = torch.from_numpy(target["segmentation"]).long()
        if "classification" in target:
            target["classification"] = torch.from_numpy(target["classification"]).float()

        return target


