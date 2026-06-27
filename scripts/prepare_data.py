#!/usr/bin/env python3
"""
数据准备工具脚本

用法:

    # 方式1: 从 CSV 标注文件构建数据集
    # CSV 格式: image_path,label
    #   /data/thyroid/img001.png,benign
    #   /data/thyroid/img002.png,malignant
    python scripts/prepare_data.py \
        --csv annotations.csv \
        --image_col image_path \
        --label_col label \
        --output data/thyroid_binary \
        --train_ratio 0.7 --val_ratio 0.15

    # 方式2: 从按类别分文件夹的数据构建（自动切分 train/val/test）
    # 目录结构: input_dir/benign/*.png, input_dir/malignant/*.png
    python scripts/prepare_data.py \
        --from_folders data/raw_thyroid \
        --output data/thyroid_binary \
        --train_ratio 0.7 --val_ratio 0.15

    # 方式3: TIRADS 多分类
    python scripts/prepare_data.py \
        --from_folders data/raw_tirads \
        --output data/thyroid_tirads \
        --train_ratio 0.7 --val_ratio 0.15
"""

import os
import sys
import argparse
import shutil
import random
from pathlib import Path


def prepare_from_csv(csv_path, image_col, label_col, output_dir, train_ratio, val_ratio, seed):
    """从 CSV 标注文件构建数据集"""
    import pandas as pd

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    output_dir = Path(output_dir)
    for split in ["train", "val", "test"]:
        for label in df[label_col].unique():
            (output_dir / split / str(label)).mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    for label, group in df.groupby(label_col):
        images = group[image_col].tolist()
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for split, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            for img_path in split_imgs:
                src = Path(img_path)
                if not src.exists():
                    print(f"  [WARN] Missing: {src}")
                    continue
                dst = output_dir / split / str(label) / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)

        print(f"  {label}: train={len(train_imgs)} val={len(val_imgs)} test={len(test_imgs)}")

    print(f"\nDataset created at: {output_dir}")


def prepare_from_folders(input_dir, output_dir, train_ratio, val_ratio, seed):
    """从按类别分文件夹的数据切分 train/val/test"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    class_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class subdirectories found in {input_dir}")

    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")

    random.seed(seed)

    for class_dir in class_dirs:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        images = sorted([
            str(p) for p in class_dir.iterdir()
            if p.suffix.lower() in exts
        ])
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        for split, start, end in [
            ("train", 0, n_train),
            ("val", n_train, n_train + n_val),
            ("test", n_train + n_val, n),
        ]:
            dst_dir = output_dir / split / class_dir.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for img_path in images[start:end]:
                dst = dst_dir / Path(img_path).name
                if not dst.exists():
                    shutil.copy2(img_path, dst)

        print(f"  {class_dir.name}: train={n_train} val={n_val} test={n - n_train - n_val}")

    print(f"\nDataset created at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare thyroid ultrasound dataset")
    parser.add_argument("--from_folders", type=str, default=None,
                        help="Source directory with class subfolders")
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV annotation file")
    parser.add_argument("--image_col", type=str, default="image_path",
                        help="Image path column in CSV")
    parser.add_argument("--label_col", type=str, default="label",
                        help="Label column in CSV")
    parser.add_argument("--output", type=str, required=True,
                        help="Output dataset directory")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.from_folders:
        prepare_from_folders(
            args.from_folders, args.output,
            args.train_ratio, args.val_ratio, args.seed,
        )
    elif args.csv:
        prepare_from_csv(
            args.csv, args.image_col, args.label_col,
            args.output, args.train_ratio, args.val_ratio, args.seed,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
