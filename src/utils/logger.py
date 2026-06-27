"""
日志与可视化工具
"""

import os
import json
import time
import yaml
from pathlib import Path
from typing import Optional


class Logger:
    """训练日志记录器"""

    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True,
        experiment_name: Optional[str] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_tensorboard = use_tensorboard
        self.writer = None
        self.metrics_history: list = []
        self.start_time = time.time()

        if experiment_name is None:
            experiment_name = time.strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.log_dir / "tensorboard" / experiment_name
                self.writer = SummaryWriter(log_dir=str(tb_dir))
                print(f"[Logger] TensorBoard logging to: {tb_dir}")
            except ImportError:
                print("[Logger] tensorboard not installed, skipping TensorBoard logging")

    def log_metrics(self, metrics: dict, step: int, prefix: str = ""):
        """记录指标"""
        timestamp = time.time() - self.start_time

        entry = {
            "step": step,
            "timestamp": timestamp,
            **{f"{prefix}{k}": v for k, v in metrics.items() if not isinstance(v, list)},
        }
        self.metrics_history.append(entry)

        # TensorBoard
        if self.writer is not None:
            for k, v in metrics.items():
                if not isinstance(v, list):
                    self.writer.add_scalar(f"{prefix}{k}", v, step)

    def log_lr(self, lr: float, step: int):
        """记录学习率"""
        if self.writer is not None:
            self.writer.add_scalar("train/lr", lr, step)

    def log_confusion_matrix(self, cm: list, class_names: list, step: int, prefix: str = "val"):
        """在 TensorBoard 中记录混淆矩阵（需要 matplotlib）"""
        if self.writer is None:
            return
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            cm = np.array(cm)
            fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.2),
                                              max(5, len(class_names) * 1.0)))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(class_names, fontsize=9)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Confusion Matrix - {prefix}")

            # 标注数值
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]),
                            ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black",
                            fontsize=8)

            plt.colorbar(im)
            plt.tight_layout()
            self.writer.add_figure(f"{prefix}/confusion_matrix", fig, step)
            plt.close(fig)
        except Exception:
            pass

    def save_metrics(self):
        """保存指标历史到 JSON"""
        path = self.log_dir / f"{self.experiment_name}_metrics.json"
        with open(path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"[Logger] Metrics saved to: {path}")

    def close(self):
        if self.writer is not None:
            self.writer.close()


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, save_path: str):
    """保存配置到文件"""
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[Config] Saved to: {save_path}")
