import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset

from utils.dataloader import Kodak, MSCOCO

TASK_TO_ID: Dict[str, int] = {"cls": 0, "det": 1, "seg": 2}
ID_TO_TASK: Dict[int, str] = {v: k for k, v in TASK_TO_ID.items()}


@dataclass
class TaskConfig:
    name: str
    dataset: Dataset
    task_id: int


def _imagenet_train_transforms() -> T.Compose:
    return T.Compose(
        [
            T.Resize(256),
            T.RandomCrop(256),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    )


def _coco_train_transforms() -> T.Compose:
    return T.Compose(
        [
            T.RandomCrop((256, 256)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    )


def _imagenet_val_transforms() -> T.Compose:
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(256),
            T.ToTensor(),
        ]
    )


def _identity_transforms() -> T.Compose:
    return T.Compose([T.ToTensor()])


def create_moe_dataset(
    dataset_path: str,
    split: str = "train",
    tasks: Sequence[str] = ("cls", "det", "seg"),
    length: Optional[int] = 100000,
    alpha: float = 0.9,
    init_probs: Optional[Dict[str, float]] = None,
    generator: Optional[torch.Generator] = None,
) -> Dataset:
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split {split}")

    selected_tasks = [task for task in tasks if task in TASK_TO_ID]
    if not selected_tasks:
        raise ValueError("No valid tasks selected")

    if split == "train":
        configs: List[TaskConfig] = []

        if "cls" in selected_tasks:
            imagenet = torchvision.datasets.ImageNet(
                root=f"{dataset_path}/imagenet-1k",
                split="train",
                transform=_imagenet_train_transforms(),
            )
            configs.append(TaskConfig("cls", imagenet, TASK_TO_ID["cls"]))

        if "det" in selected_tasks:
            coco_det = MSCOCO(
                f"{dataset_path}/coco2017/train2017/",
                _coco_train_transforms(),
                "./examples/utils/img_list.txt",
            )
            configs.append(TaskConfig("det", coco_det, TASK_TO_ID["det"]))

        if "seg" in selected_tasks:
            coco_seg = MSCOCO(
                f"{dataset_path}/coco2017/train2017/",
                _coco_train_transforms(),
                "./examples/utils/img_list.txt",
            )
            configs.append(TaskConfig("seg", coco_seg, TASK_TO_ID["seg"]))

        return MoEDataset(
            configs=configs,
            length=length,
            alpha=alpha,
            init_probs=init_probs,
            generator=generator,
        )

    imagenet_val = torchvision.datasets.ImageNet(
        root=f"{dataset_path}/imagenet-1k",
        split="val",
        transform=_imagenet_val_transforms(),
    )
    kodak_val = Kodak(root=f"{dataset_path}/Kodak/", transform=_identity_transforms())

    datasets: Dict[str, Dataset] = {}
    if "cls" in selected_tasks:
        datasets["cls"] = imagenet_val
    if "det" in selected_tasks:
        datasets["det"] = kodak_val
    if "seg" in selected_tasks:
        datasets["seg"] = kodak_val

    return datasets


class MoEDataset(Dataset):
    def __init__(
        self,
        configs: Sequence[TaskConfig],
        length: Optional[int],
        alpha: float = 0.9,
        init_probs: Optional[Dict[str, float]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if not configs:
            raise ValueError("MoEDataset requires at least one task")
        self.configs = list(configs)
        self.task_names = [cfg.name for cfg in self.configs]
        self.datasets = {cfg.name: cfg.dataset for cfg in self.configs}
        self.task_ids = {cfg.name: cfg.task_id for cfg in self.configs}
        self.task_lengths = {cfg.name: len(cfg.dataset) for cfg in self.configs}
        self.length = length if length is not None else sum(self.task_lengths.values())

        base_prob = 1.0 / len(self.configs)
        self.probs = {
            cfg.name: init_probs.get(cfg.name, base_prob) if init_probs else base_prob
            for cfg in self.configs
        }
        self._normalize_probs()

        self.alpha = alpha
        self.loss_ema = {cfg.name: 1.0 for cfg in self.configs}
        self.generator = generator or torch.Generator()
        self.cursor = {cfg.name: 0 for cfg in self.configs}
        self.order = {
            cfg.name: self._reshuffle(cfg.name) for cfg in self.configs
        }

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, object]:
        task_name = self._sample_task()
        dataset = self.datasets[task_name]
        sample_idx = self._next_index(task_name)
        sample = dataset[sample_idx]

        if isinstance(sample, tuple):
            image, target = sample
        else:
            image, target = sample, None

        return {
            "image": image,
            "target": target,
            "task": task_name,
            "task_id": self.task_ids[task_name],
        }

    def update_task_losses(self, task_losses: Dict[str, float]) -> None:
        updated = False
        for task, loss in task_losses.items():
            if task in self.loss_ema:
                self.loss_ema[task] = self.alpha * self.loss_ema[task] + (1.0 - self.alpha) * loss
                updated = True
        if not updated:
            return
        total = sum(self.loss_ema[t] for t in self.task_names)
        self.probs = {t: self.loss_ema[t] / total for t in self.task_names}

    def _sample_task(self) -> str:
        weights = torch.tensor([self.probs[t] for t in self.task_names], dtype=torch.float)
        task_idx = torch.multinomial(weights, num_samples=1, generator=self.generator).item()
        return self.task_names[task_idx]

    def _next_index(self, task_name: str) -> int:
        idx = self.cursor[task_name]
        order = self.order[task_name]
        if idx >= len(order):
            order = self._reshuffle(task_name)
            idx = 0
            self.order[task_name] = order
            self.cursor[task_name] = 0
        self.cursor[task_name] += 1
        return order[idx]

    def _reshuffle(self, task_name: str) -> List[int]:
        perm = torch.randperm(self.task_lengths[task_name], generator=self.generator)
        return perm.tolist()

    def _normalize_probs(self) -> None:
        total = sum(self.probs.values())
        if total <= 0:
            raise ValueError("Task probabilities must sum to a positive value")
        for task in self.probs:
            self.probs[task] /= total


def moe_collate_fn(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    images = torch.stack([sample["image"] for sample in batch], dim=0)
    task_ids = torch.tensor([sample["task_id"] for sample in batch], dtype=torch.long)
    task_names = [sample["task"] for sample in batch]

    targets: List[object] = []
    for sample, task in zip(batch, task_names):
        target = sample["target"]
        if task == "cls" and target is not None:
            targets.append(torch.tensor(target, dtype=torch.long))
        else:
            targets.append(target)

    return {
        "images": images,
        "task_id": task_ids,
        "task": task_names,
        "target": targets,
    }
