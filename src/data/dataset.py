"""
数据集加载模块
支持 folder-based 按类别子文件夹组织的数据。
超声图像可能是灰度图，自动转为三通道以适配 MedSigLIP 的 RGB 输入。
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class ThyroidUltrasoundDataset(Dataset):
    """
    甲状腺超声图像数据集

    数据目录结构:
        root_dir/
            train/
                class_A/
                    img001.png
                    img002.png
                class_B/
                    img003.png
            val/
                class_A/
                class_B/
            test/
                class_A/
                class_B/
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        file_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        self.images: list = []
        self.labels: list = []

        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # 遍历子文件夹，每个子文件夹对应一个类别
        class_dirs = sorted(
            [d for d in split_dir.iterdir() if d.is_dir()]
        )
        if not class_dirs:
            raise FileNotFoundError(
                f"No class subdirectories found in {split_dir}. "
                f"Expected structure: {split_dir}/class_name/*.png"
            )

        self.class_names = [d.name for d in class_dirs]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.class_names)

        for class_dir in class_dirs:
            class_idx = self.label_encoder.transform([class_dir.name])[0]
            for ext in file_extensions:
                for img_path in class_dir.glob(f"*{ext}"):
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)
                # 也检查大写扩展名
                for img_path in class_dir.glob(f"*{ext.upper()}"):
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)

        if len(self.images) == 0:
            raise RuntimeError(
                f"No images found in {split_dir}. "
                f"Accepted extensions: {file_extensions}"
            )

        # 计算类别分布
        self._compute_class_distribution()

    def _compute_class_distribution(self):
        labels_arr = np.array(self.labels)
        self.class_counts = {
            name: int(np.sum(labels_arr == idx))
            for name, idx in zip(self.class_names, range(len(self.class_names)))
        }
        self.total_samples = len(self.images)

    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重（用于平衡损失函数）"""
        counts = np.array(list(self.class_counts.values()))
        weights = self.total_samples / (len(counts) * counts)
        weights = np.where(counts == 0, 0.0, weights)
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.images[idx]
        label = self.labels[idx]

        # 读取图像（超声通常为灰度图）
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # 灰度图 -> 三通道 RGB（MedSigLIP 需要 3 通道输入）
        image = np.stack([image] * 3, axis=-1)  # (H, W) -> (H, W, 3)

        # 应用数据增强/预处理
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]  # (C, H, W) float32

        return {
            "pixel_values": image,
            "label": torch.tensor(label, dtype=torch.long),
            "image_path": img_path,
        }


def create_dataloaders(
    data_cfg: dict,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    创建 train/val/test DataLoader

    Returns:
        train_loader, val_loader, test_loader (test_loader 可能为 None)
    """
    from src.data.transforms import get_train_transforms, get_val_transforms

    image_size = data_cfg["image_size"]
    mean = data_cfg["mean"]
    std = data_cfg["std"]
    aug_cfg = data_cfg.get("augmentation", {})
    root_dir = data_cfg["root_dir"]
    nw = data_cfg.get("num_workers", num_workers)
    pm = data_cfg.get("pin_memory", pin_memory)

    train_transform = get_train_transforms(image_size, mean, std, aug_cfg)
    val_transform = get_val_transforms(image_size, mean, std)

    train_dataset = ThyroidUltrasoundDataset(root_dir, split="train", transform=train_transform)
    val_dataset = ThyroidUltrasoundDataset(root_dir, split="val", transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=pm,
        drop_last=False,                  # 不平衡数据不要丢弃末尾样本（可能包含少数类）
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pm,
    )

    # 可选的测试集
    test_dir = Path(root_dir) / "test"
    test_loader = None
    if test_dir.exists() and any(test_dir.iterdir()):
        test_dataset = ThyroidUltrasoundDataset(root_dir, split="test", transform=val_transform)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=pm,
        )
        print(f"Found test set: {len(test_dataset)} samples in {test_dir}")

    print(f"Train: {train_dataset.total_samples} samples, "
          f"classes: {train_dataset.class_counts}")
    print(f"Val:   {val_dataset.total_samples} samples, "
          f"classes: {val_dataset.class_counts}")

    return train_loader, val_loader, test_loader, train_dataset
