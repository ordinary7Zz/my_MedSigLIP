"""
日志工具
"""

import json
import time
import yaml
from pathlib import Path
from typing import Optional


class Logger:
    """训练日志记录器，写入纯文本 log 文件"""

    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True,
        experiment_name: Optional[str] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history: list = []
        self.start_time = time.time()

        if experiment_name is None:
            experiment_name = time.strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name

        # 打开日志文件
        log_path = self.log_dir / f"{experiment_name}.log"
        self.log_file = open(log_path, "w", buffering=1)  # 行缓冲，即时写入
        print(f"[Logger] Logging to: {log_path}")

    def _write_line(self, line: str):
        self.log_file.write(line + "\n")

    def log_metrics(self, metrics: dict, step: int, prefix: str = ""):
        """记录指标"""
        timestamp = time.time() - self.start_time

        entry = {
            "step": step,
            "timestamp": timestamp,
            **{f"{prefix}{k}": v for k, v in metrics.items() if not isinstance(v, list)},
        }
        self.metrics_history.append(entry)

        # 写入 log 文件
        items = [f"step={step}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                items.append(f"{prefix}{k}={v:.6f}")
            elif not isinstance(v, list):
                items.append(f"{prefix}{k}={v}")
        self._write_line(f"[{prefix.rstrip('/')}] " + " | ".join(items))

    def log_lr(self, lr: float, step: int):
        """记录学习率"""
        self._write_line(f"[train] step={step} | lr={lr:.8f}")

    def log_confusion_matrix(self, cm: list, class_names: list, step: int, prefix: str = "val"):
        """记录混淆矩阵到 log 文件"""
        import numpy as np
        cm = np.array(cm)
        self._write_line(f"[{prefix.rstrip('/')}] step={step} confusion_matrix:")
        header = " " * 12 + "".join(f"{n:>8}" for n in class_names)
        self._write_line(header)
        for i, name in enumerate(class_names):
            row = f"  {name:>10}" + "".join(f"{cm[i, j]:>8}" for j in range(len(class_names)))
            self._write_line(row)

    def save_metrics(self):
        """保存指标历史到 JSON"""
        path = self.log_dir / f"{self.experiment_name}_metrics.json"
        with open(path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"[Logger] Metrics saved to: {path}")

    def close(self):
        self.log_file.close()


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
