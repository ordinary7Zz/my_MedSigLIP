#!/usr/bin/env python3
"""
MedSigLIP 甲状腺超声图像分类 - 主训练脚本

用法:
    # 二分类（良恶性）
    python scripts/train.py --config configs/binary_cls.yaml

    # TIRADS 多分类
    python scripts/train.py --config configs/multi_cls.yaml

    # 从检查点恢复训练
    python scripts/train.py --config configs/binary_cls.yaml --resume checkpoints/binary_cls/best_model.pt
"""

import sys
import os
import argparse

# 将项目根目录加入 Python 搜索路径（支持从任意位置运行）
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)
# 同时加入当前工作目录作为备选
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import torch
from src.models.classifier import MedSigLIPClassifier
from src.data.dataset import create_dataloaders
from src.trainers.trainer import Trainer
from src.utils.logger import load_config, save_config


def main():
    parser = argparse.ArgumentParser(description="Train MedSigLIP classifier on thyroid ultrasound")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (same task, full restore)")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to checkpoint for ViT weight transfer (e.g. binary -> multi-class)")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda:0, cpu, etc.)")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device

    num_cls = config["model"]["num_classes"]
    strategy = config["model"]["fine_tune_strategy"]

    print(f"\n{'='*60}")
    print(f"Config: {args.config}")
    if num_cls == 2:
        print("Task: Binary (Benign/Malignant)")
    else:
        print(f"Task: Multi-class ({num_cls} classes)")
    print(f"Fine-tuning Strategy: {strategy}")
    print(f"{'='*60}\n")

    # 设置设备
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建 DataLoader
    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
        data_cfg=config["data"],
        batch_size=config["training"]["batch_size"],
    )

    # 创建模型
    model = MedSigLIPClassifier(
        model_name=config["model"]["name"],
        num_classes=config["model"]["num_classes"],
        fine_tune_strategy=config["model"]["fine_tune_strategy"],
        unfreeze_last_n=config["model"].get("unfreeze_last_n", 6),
        dropout=config["model"].get("classifier_dropout", 0.1),
        local_files_only=config["model"].get("local_files_only", False),
    )

    # 类别权重
    class_weights = None
    if config["model"].get("use_class_weights", False):
        class_weights = train_dataset.get_class_weights()
        print(f"Class weights: {class_weights.tolist()}")

    # 从检查点恢复（完整恢复，同任务）
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Resumed from epoch {checkpoint['epoch']+1}")

    # 从预训练权重迁移 ViT 骨干（跨任务，如二分类 -> 多分类）
    if args.pretrained:
        print(f"Transferring ViT backbone from: {args.pretrained}")
        pretrained_ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        pretrained_state = pretrained_ckpt["model_state_dict"]

        # 只加载 ViT 部分的权重，跳过分类头（维度不同）
        vit_keys = [k for k in pretrained_state if k.startswith("vision_encoder.")]
        vit_state = {k: pretrained_state[k] for k in vit_keys}

        missing, unexpected = model.load_state_dict(vit_state, strict=False)
        print(f"  Transferred {len(vit_keys)} ViT parameters")
        if missing:
            skipped = [m for m in missing if not m.startswith("vision_encoder.")]
            if skipped:
                print(f"  New (random init): {skipped}")

    # 保存当前配置到 checkpoint 目录
    checkpoint_dir = config["logging"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_config(config, os.path.join(checkpoint_dir, "config.yaml"))

    # 创建训练器并训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        class_names=train_dataset.class_names,
        class_weights=class_weights,
        test_loader=test_loader,
    )

    trainer.train()

    print("\nDone! Best model saved to:", os.path.join(checkpoint_dir, "best_model.pt"))


if __name__ == "__main__":
    main()
