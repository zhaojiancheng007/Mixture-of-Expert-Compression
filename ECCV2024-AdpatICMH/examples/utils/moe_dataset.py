import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from utils.dataloader import MSCOCO, Kodak
import torch
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandAugment

TASK_VOCAB = {"cls": 0, "det": 1, "seg": 2}

def to_task_ids(tnames, vocab, device=None):
    # tnames can be a single task name or an iterable of task names.
    if isinstance(tnames, str):
        tnames = [tnames]
    ids = torch.tensor([vocab[t] for t in tnames], dtype=torch.long)
    if device is not None:
        ids = ids.to(device)
    return ids

def create_moe_dataset(dataset_path, split="train", tasks = ['cls'], length=100000, init_probs=None):
    """
    - split="train" 返回 MoEDataset（ImageNet + COCO）
    - split="val"   返回 dict {"cls": dataset, "det": dataset, "seg": dataset} （ImageNet + Kodak）
    """

    if split == "train":
        # ----- transforms -----
        cls_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
        ])
        coco_transforms = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        # ----- datasets -----
        datasets_dict = {}
        if 'cls' in tasks:
            cls_dataset = torchvision.datasets.ImageNet(
                root=dataset_path + "/imagenet-1k", split="train", transform=cls_transforms
            )
            datasets_dict['cls'] = cls_dataset
        if 'det' in tasks:
            det_dataset = MSCOCO(
                dataset_path + "/coco2017/train2017/",
                coco_transforms,
                "./examples/utils/img_list.txt",
            )
            datasets_dict['det'] = det_dataset
        if 'seg' in tasks:
            seg_dataset = MSCOCO(
                dataset_path + "/coco2017/train2017/",
                coco_transforms,
                "./examples/utils/img_list.txt",
            )
            datasets_dict['seg'] = seg_dataset

        return MoEDataset(datasets_dict, length=length, init_probs=init_probs)

    else:  # val
        # ----- transforms -----
        cls_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
        simple_transforms = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # ----- datasets -----
        cls_dataset = torchvision.datasets.ImageNet(
            root=dataset_path + "/imagenet-1k", split="val", transform=cls_transforms
        )
        det_dataset = Kodak(
            root=dataset_path + "/Kodak/", transform=simple_transforms
        )
        seg_dataset = Kodak(
            root=dataset_path + "/Kodak/", transform=simple_transforms
        )

        return {"cls": cls_dataset, "det": det_dataset, "seg": seg_dataset}

class MoEDataset(Dataset):
    def __init__(self, datasets_dict, init_probs=None, alpha=0.9, length=100000):
        self.datasets = datasets_dict
        self.task_names = list(datasets_dict.keys())
        self.task_lengths = {t: len(d) for t, d in datasets_dict.items()}
        self.length = length

        if init_probs is None:
            init_probs = {t: 1.0 / len(self.task_names) for t in self.task_names}
        self.probs = init_probs

        self.alpha = alpha
        self.loss_ema = {t: 1.0 for t in self.task_names}

    def __len__(self):
        return self.length

    def update_task_losses(self, task_losses):
        for t in self.task_names:
            if t in task_losses:
                self.loss_ema[t] = (
                    self.alpha * self.loss_ema[t] + (1 - self.alpha) * task_losses[t]
                )
        loss_sum = sum(self.loss_ema.values())
        self.probs = {t: self.loss_ema[t] / loss_sum for t in self.task_names}

    def __getitem__(self, idx):
        weights = [self.probs[t] for t in self.task_names] 
        task = random.choices(self.task_names, weights=weights, k=1)[0]
        dataset = self.datasets[task]
        sub_idx = random.randint(0, len(dataset) - 1)
        sample = dataset[sub_idx]

        task_id = torch.tensor(TASK_VOCAB[task], dtype=torch.long)
        return {"task": task, "task_id": task_id, "sample": sample}

def moe_collate_fn(batch):
    images, task_names, task_ids, labels = [], [], [], []

    for sample in batch:
        task = sample["task"]
        data = sample["sample"]
        task_names.append(task)

        task_id = sample.get("task_id", TASK_VOCAB[task])
        if not torch.is_tensor(task_id):
            task_id = torch.tensor(task_id, dtype=torch.long)
        task_ids.append(task_id)

        if task == "cls":
            img, label = data   # ImageNet → (Tensor, int)
            images.append(img)
            labels.append(torch.tensor(label, dtype=torch.long))

        elif task == "det":
            # COCO 可能返回 Tensor 或 (Tensor,)，这里统一解包
            img = data[0] if isinstance(data, tuple) else data
            images.append(img)
            labels.append(torch.tensor(-1, dtype=torch.long))

        elif task == "seg":
            img = data[0] if isinstance(data, tuple) else data
            images.append(img)
            labels.append(torch.tensor(-1, dtype=torch.long))

    task_ids = torch.stack(task_ids, dim=0)
    labels = torch.stack(labels, dim=0)
    images = torch.stack(images, dim=0)  # 保证里面全是 Tensor
    sample = (images, labels)
    return {
        "sample": sample,
        "task": task_ids,
        "task_ids": task_ids,
        "task_names": task_names,
    }
