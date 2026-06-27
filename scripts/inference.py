#!/usr/bin/env python3
"""
MedSigLIP 单张/批量推理脚本

用法:
    # 单张图像推理
    python scripts/inference.py \
        --checkpoint checkpoints/binary_cls/best_model.pt \
        --config configs/binary_cls.yaml \
        --input path/to/image.png

    # 文件夹批量推理
    python scripts/inference.py \
        --checkpoint checkpoints/binary_cls/best_model.pt \
        --config configs/binary_cls.yaml \
        --input path/to/image_folder/ \
        --output results.csv

    # 输出概率而非类别
    python scripts/inference.py \
        --checkpoint checkpoints/multi_cls/best_model.pt \
        --config configs/multi_cls.yaml \
        --input path/to/image.png \
        --output_probs
"""

import sys
import os
import argparse
from pathlib import Path

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import cv2
import torch
import numpy as np
import pandas as pd

from src.models.classifier import MedSigLIPClassifier
from src.data.transforms import get_val_transforms
from src.utils.logger import load_config


def inference():
    parser = argparse.ArgumentParser(description="MedSigLIP inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--input", type=str, required=True, help="Path to image or directory of images")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (for batch)")
    parser.add_argument("--output_probs", action="store_true", help="Output probabilities instead of class")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    device = torch.device(args.device or config.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    # 加载模型
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", [str(i) for i in range(config["model"]["num_classes"])])
    model = MedSigLIPClassifier(
        model_name=config["model"]["name"],
        num_classes=config["model"]["num_classes"],
        fine_tune_strategy=config["model"]["fine_tune_strategy"],
        dropout=config["model"].get("classifier_dropout", 0.1),
        local_files_only=config["model"].get("local_files_only", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # 预处理
    transform = get_val_transforms(
        config["data"]["image_size"],
        config["data"]["mean"],
        config["data"]["std"],
    )

    print(f"Model: {config['model']['name']}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Classes: {class_names}")
    print(f"Device: {device}\n")

    # 收集输入图像
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        image_paths = sorted([
            p for p in input_path.rglob("*") if p.suffix.lower() in exts
        ])
    else:
        raise FileNotFoundError(f"Input not found: {args.input}")

    print(f"Found {len(image_paths)} image(s)\n")

    results = []
    for img_path in image_paths:
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  [WARN] Failed to load: {img_path}")
            continue

        # 灰度 → 三通道
        image = np.stack([image] * 3, axis=-1)
        transformed = transform(image=image)
        pixel_values = transformed["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(pixel_values)
            logits = outputs["logits"].cpu().numpy()[0]

        # 计算概率
        if logits.shape[0] == 1:
            probs = 1.0 / (1.0 + np.exp(-logits[0]))
            probs_all = np.array([1 - probs, probs])
        else:
            e_x = np.exp(logits - logits.max())
            probs_all = e_x / e_x.sum()

        pred_class = probs_all.argmax()
        confidence = probs_all[pred_class]

        result = {
            "image": str(img_path),
            "pred_class": pred_class,
            "pred_label": class_names[pred_class],
            "confidence": float(confidence),
        }
        if args.output_probs:
            for i, name in enumerate(class_names):
                result[f"prob_{name}"] = float(probs_all[i])

        results.append(result)

        print(f"  {img_path.name}: {class_names[pred_class]} "
              f"(conf={confidence:.4f})")

    # 保存结果
    if args.output and results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")

    return results


if __name__ == "__main__":
    inference()
