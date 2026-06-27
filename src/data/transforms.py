"""
数据增强与预处理流水线
针对甲状腺超声图像设计，增强策略较为保守以避免破坏诊断特征。
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int, mean: list, std: list, aug_cfg: dict):
    """
    训练集增强流水线：
    - 灰度图转三通道（适配 MedSigLIP RGB 输入）
    - 缩放到目标尺寸
    - 轻量几何/颜色增强
    - 归一化到 [-1, 1]
    """
    transforms_list = []

    # 1. 缩放到目标尺寸（保持宽高比 + 黑边填充）
    transforms_list.append(
        A.LongestMaxSize(max_size=image_size, p=1.0)
    )
    transforms_list.append(
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0,               # 黑边填充
            p=1.0,
        )
    )

    # 2. 数据增强
    if aug_cfg.get("random_horizontal_flip", True):
        transforms_list.append(A.HorizontalFlip(p=0.5))

    rotation_deg = aug_cfg.get("random_rotation", 10)
    if rotation_deg > 0:
        transforms_list.append(
            A.Rotate(limit=rotation_deg, border_mode=0, p=0.5)
        )

    bc_limit = aug_cfg.get("random_brightness_contrast", 0.0)
    if bc_limit > 0:
        transforms_list.append(
            A.RandomBrightnessContrast(
                brightness_limit=bc_limit,
                contrast_limit=bc_limit,
                p=0.3,
            )
        )

    scale_range = aug_cfg.get("random_affine_scale", None)
    if scale_range is not None:
        transforms_list.append(
            A.Affine(
                scale=scale_range,
                rotate=0,
                translate_percent=0,
                p=0.3,
            )
        )

    # 3. 归一化 + 转 Tensor（始终执行，p=1.0）
    transforms_list.append(A.Normalize(mean=mean, std=std, p=1.0))
    transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)


def get_val_transforms(image_size: int, mean: list, std: list):
    """验证/测试集：仅缩放 + 归一化，不做增强"""
    return A.Compose([
        A.LongestMaxSize(max_size=image_size, p=1.0),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0,
            p=1.0,
        ),
        A.Normalize(mean=mean, std=std, p=1.0),
        ToTensorV2(),
    ])
