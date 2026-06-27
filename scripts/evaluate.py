#!/usr/bin/env python3
"""
MedSigLIP 模型评估脚本

用法:
    # 在测试集上评估
    python scripts/evaluate.py \
        --checkpoint checkpoints/binary_cls/best_model.pt \
        --config configs/binary_cls.yaml \
        --split test

    # 在验证集上评估
    python scripts/evaluate.py \
        --checkpoint checkpoints/multi_cls/best_model.pt \
        --config configs/multi_cls.yaml \
        --split val

    # 导出可视化（ROC曲线、混淆矩阵等）
    python scripts/evaluate.py \
        --checkpoint checkpoints/binary_cls/best_model.pt \
        --config configs/binary_cls.yaml \
        --split test \
        --visualize
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

from src.models.classifier import MedSigLIPClassifier
from src.data.dataset import ThyroidUltrasoundDataset
from src.data.transforms import get_val_transforms
from src.utils.metrics import MetricsCalculator
from src.utils.logger import load_config


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate MedSigLIP classifier")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--visualize", action="store_true", help="Generate visualization plots")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    device = torch.device(args.device or config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 创建数据集
    data_cfg = config["data"]
    transform = get_val_transforms(
        data_cfg["image_size"],
        data_cfg["mean"],
        data_cfg["std"],
    )

    dataset = ThyroidUltrasoundDataset(data_cfg["root_dir"], split=args.split, transform=transform)
    print(f"Loaded {args.split} split: {len(dataset)} samples")
    print(f"Classes: {dict(zip(dataset.class_names, [dataset.class_counts[n] for n in dataset.class_names]))}")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=data_cfg.get("num_workers", 4), pin_memory=data_cfg.get("pin_memory", True),
    )

    # 加载模型
    model = MedSigLIPClassifier(
        model_name=config["model"]["name"],
        num_classes=config["model"]["num_classes"],
        fine_tune_strategy=config["model"]["fine_tune_strategy"],
        dropout=config["model"].get("classifier_dropout", 0.1),
        local_files_only=config["model"].get("local_files_only", False),
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from: {args.checkpoint}")

    # 推理
    all_logits = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"]

            outputs = model(pixel_values)
            all_logits.append(outputs["logits"].cpu().numpy())
            all_labels.append(labels.numpy())
            all_paths.extend(batch.get("image_path", []))

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 计算指标
    num_classes = config["model"]["num_classes"]
    metrics_cfg = config["evaluation"]["metrics"]
    metrics_calc = MetricsCalculator(num_classes, metrics_cfg)
    metrics = metrics_calc.compute(all_logits, all_labels)

    print(f"\n{'='*60}")
    print(f"Evaluation on {args.split} split ({len(dataset)} samples)")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        elif isinstance(v, list):
            print(f"  {k:20s}: (confusion matrix)")
            print(np.array2string(np.array(v), prefix=" " * 24))

    # 详细报告
    print(f"\n{'='*60}")
    print("Detailed Classification Report:")
    print(f"{'='*60}")
    print(metrics_calc.detailed_report(all_logits, all_labels))

    # 可视化
    if args.visualize:
        os.makedirs(args.output_dir, exist_ok=True)
        _generate_visualizations(all_logits, all_labels, dataset.class_names, args.output_dir, num_classes)
        print(f"\nVisualizations saved to: {args.output_dir}/")

    # 保存错误预测的图片路径
    _save_error_analysis(all_logits, all_labels, all_paths, dataset.class_names, args.output_dir, num_classes)


def _generate_visualizations(logits, labels, class_names, output_dir, num_classes):
    """生成可视化图表"""
    plt.style.use("seaborn-v0_8-whitegrid")

    # ROC 曲线
    if num_classes == 2:
        probs = 1 / (1 + np.exp(-logits.flatten())) if logits.shape[1] == 1 else MetricsCalculator._softmax(logits)[:, 1]
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # ROC
        axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
        axes[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("ROC Curve")
        axes[0].legend(loc="lower right")

        # PR 曲线
        precision, recall, _ = precision_recall_curve(labels, probs)
        axes[1].plot(recall, precision, color="blue", lw=2)
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision-Recall Curve")
    else:
        # 多分类：每个类别的 ROC (OvR)
        probs_all = MetricsCalculator._softmax(logits)
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))

        for i in range(num_classes):
            fpr, tpr, _ = roc_curve((labels == i).astype(int), probs_all[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves (One-vs-Rest)")
        ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_pr_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, logits.argmax(axis=1))
    fig, ax = plt.subplots(figsize=(max(6, num_classes * 1.2), max(5, num_classes * 1.0)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_error_analysis(logits, labels, paths, class_names, output_dir, num_classes):
    """保存错误分析"""
    preds = logits.argmax(axis=1) if logits.shape[1] > 1 else (1 / (1 + np.exp(-logits.flatten())) >= 0.5).astype(int)
    errors = preds != labels

    error_file = os.path.join(output_dir, "error_analysis.txt")
    with open(error_file, "w") as f:
        f.write(f"Total samples: {len(labels)}\n")
        f.write(f"Errors: {errors.sum()} ({100*errors.sum()/len(labels):.2f}%)\n\n")
        for i in np.where(errors)[0]:
            f.write(
                f"Image: {paths[i]}\n"
                f"  True: {class_names[int(labels[i])]} -> Pred: {class_names[int(preds[i])]}\n"
            )
    print(f"Error analysis saved to: {error_file}")


if __name__ == "__main__":
    evaluate()
