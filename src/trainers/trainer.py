"""
训练引擎
支持：二分类/多分类、三种微调策略、混合精度、早停、模型保存
"""

import os
import copy
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    LambdaLR,
)

from src.utils.metrics import MetricsCalculator
from src.utils.logger import Logger


class Trainer:
    """
    MedSigLIP 分类训练器

    使用方式:
        trainer = Trainer(model, train_loader, val_loader, config)
        trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: dict,
        class_names: Optional[list] = None,
        class_weights: Optional[torch.Tensor] = None,
        test_loader=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.class_names = class_names or [str(i) for i in range(config["model"]["num_classes"])]
        self.num_classes = config["model"]["num_classes"]

        # 设备
        self.device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        # 损失函数
        loss_cfg = config["training"]["loss"]
        self._setup_loss(loss_cfg, class_weights)

        # 训练参数（必须早于 _setup_scheduler，因为调度器需要 self.epochs）
        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.warmup_epochs = train_cfg.get("warmup_epochs", 0)
        self.gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.label_smoothing = train_cfg.get("label_smoothing", 0.0)
        self.log_interval = config["logging"].get("log_interval", 20)
        self.eval_interval = config["logging"].get("eval_interval", 1)
        self.early_stopping_patience = train_cfg.get("early_stopping_patience", 10)
        self.save_best_metric = config["evaluation"]["save_best_metric"]
        self.save_best_mode = config["evaluation"]["mode"]
        self.save_top_k = config["logging"].get("save_top_k", 3)
        self.seed = config.get("seed", 42)

        # 学习率调度器
        self._setup_scheduler()

        # 混合精度
        self.use_amp = False
        self.scaler = None
        mixed_precision = config["training"].get("mixed_precision", "no")
        if mixed_precision in ("fp16", "bf16") and self.device.type == "cuda":
            self.use_amp = True
            self.amp_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
            self.scaler = torch.amp.GradScaler(enabled=(mixed_precision == "fp16"))
            print(f"[Trainer] Mixed precision: {mixed_precision}")

        # 指标计算器
        self.metrics_calc = MetricsCalculator(
            num_classes=self.num_classes,
            metrics_list=config["evaluation"]["metrics"],
        )

        # 日志
        log_dir = config["logging"]["log_dir"]
        self.logger = Logger(
            log_dir=log_dir,
            use_tensorboard=config["logging"].get("use_tensorboard", True),
        )

        # 检查点
        self.checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric_value = -float("inf") if self.save_best_mode == "max" else float("inf")
        self.best_epoch = 0
        self.no_improve_count = 0
        self.best_model_ckpt: list = []  # (metric_value, ckpt_path)

        self._set_seed(self.seed)

    def _setup_loss(self, loss_cfg: dict, class_weights: Optional[torch.Tensor]):
        """配置损失函数"""
        loss_name = loss_cfg["name"]

        if loss_name == "bce":
            # 二分类 BCEWithLogitsLoss
            pos_weight = loss_cfg.get("pos_weight")
            if pos_weight is not None:
                pos_weight = torch.tensor([pos_weight], device=self.device)
            elif class_weights is not None and len(class_weights) == 2:
                # 自动从类别权重计算 pos_weight
                pos_weight = (class_weights[1] / class_weights[0]).unsqueeze(0).to(self.device)
            else:
                pos_weight = None

            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self._loss_fn = self._bce_loss
            print(f"[Trainer] Loss: BCEWithLogitsLoss (pos_weight={pos_weight})")

        elif loss_name == "cross_entropy":
            # 多分类 CrossEntropyLoss
            weight = class_weights.to(self.device) if class_weights is not None else None
            self.criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)
            self._loss_fn = self._cross_entropy_loss
            print(f"[Trainer] Loss: CrossEntropyLoss (class_weights={weight})")

        else:
            raise ValueError(f"Unknown loss: {loss_name}")

    def _bce_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """二分类 BCE 损失"""
        return self.criterion(logits.view(-1), labels.float())

    def _cross_entropy_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """多分类 CE 损失"""
        return self.criterion(logits, labels)

    def _setup_scheduler(self):
        train_cfg = self.config["training"]
        scheduler_type = train_cfg.get("lr_scheduler", "cosine")
        warmup_epochs = train_cfg.get("warmup_epochs", 0)
        lr_min = train_cfg.get("lr_min", 1e-6)

        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - warmup_epochs,
                eta_min=lr_min,
            )
            self._scheduler_step_after_epoch = True
        elif scheduler_type == "step":
            self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)
            self._scheduler_step_after_epoch = True
        elif scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="max" if self.save_best_mode == "max" else "min",
                patience=5, factor=0.5, min_lr=lr_min,
            )
            self._scheduler_step_after_epoch = True
        else:
            self.scheduler = None
            self._scheduler_step_after_epoch = False

        self.warmup_epochs = warmup_epochs
        print(f"[Trainer] Scheduler: {type(self.scheduler).__name__ if self.scheduler else 'None'}"
              f", warmup_epochs={warmup_epochs}")

    def _warmup_lr(self, epoch: int):
        """学习率线性预热"""
        if self.warmup_epochs <= 0 or epoch >= self.warmup_epochs:
            return
        base_lr = self.config["training"]["learning_rate"]
        lr = base_lr * (epoch + 1) / self.warmup_epochs
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train(self):
        """主训练循环"""
        print(f"\n{'='*60}")
        print(f"Training on: {self.device}")
        print(f"Epochs: {self.epochs} | Batch: {self.config['training']['batch_size']}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples:   {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # 预热
            self._warmup_lr(epoch)

            # 训练
            train_loss, train_metrics = self._train_epoch()

            # 日志
            self.logger.log_lr(self.optimizer.param_groups[0]["lr"], self.global_step)
            self.logger.log_metrics({"loss": train_loss, **train_metrics}, self.global_step, prefix="train/")
            print(f"Epoch {epoch+1}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # 验证
            if (epoch + 1) % self.eval_interval == 0 or epoch == self.epochs - 1:
                val_loss, val_metrics = self._validate_epoch()
                self.logger.log_metrics({"loss": val_loss, **val_metrics}, self.global_step, prefix="val/")

                # 打印指标
                metric_str = " | ".join(
                    f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in val_metrics.items()
                    if not isinstance(v, list)
                )
                print(f"         Val Loss: {val_loss:.4f} | {metric_str}")

                # 早停判断 & 保存模型
                current_metric = val_metrics.get(self.save_best_metric, 0.0)
                is_better = (
                    (self.save_best_mode == "max" and current_metric > self.best_metric_value) or
                    (self.save_best_mode == "min" and current_metric < self.best_metric_value)
                )

                if is_better:
                    self.best_metric_value = current_metric
                    self.best_epoch = epoch
                    self.no_improve_count = 0
                    self._save_checkpoint(epoch, val_loss, val_metrics, is_best=True)
                else:
                    self.no_improve_count += 1
                    if epoch % 5 == 0:                     # 每 5 个 epoch 保存一次常规检查点
                        self._save_checkpoint(epoch, val_loss, val_metrics, is_best=False)

                # 早停
                if self.no_improve_count >= self.early_stopping_patience:
                    print(f"\n[EarlyStopping] No improvement for {self.early_stopping_patience} epochs. Stopping.")
                    break

            # 更新学习率（预热阶段不调用 scheduler）
            if self.scheduler is not None and epoch >= self.warmup_epochs:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    # Plateau 需要传入指标值
                    metric_for_lr = val_loss  # 或者 val_metrics.get(self.save_best_metric, val_loss)
                    self.scheduler.step(metric_for_lr)
                else:
                    self.scheduler.step()

        # 训练结束
        self.logger.save_metrics()
        print(f"\n{'='*60}")
        print(f"Training Complete! Best {self.save_best_metric}: {self.best_metric_value:.4f} at epoch {self.best_epoch+1}")
        print(f"{'='*60}")

        # 最终测试评估
        if self.test_loader is not None:
            self._final_evaluate()

        self.logger.close()

    def _train_epoch(self) -> tuple:
        """单轮训练"""
        self.model.train()
        total_loss = 0.0
        all_logits = []
        all_labels = []

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            # 混合精度前向传播
            if self.use_amp:
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    outputs = self.model(pixel_values)
                    loss = self._loss_fn(outputs["logits"], labels)
                    loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(pixel_values)
                loss = self._loss_fn(outputs["logits"], labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

            # 梯度累积
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.gradient_accumulation_steps
            all_logits.append(outputs["logits"].detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / len(self.train_loader)
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        metrics = self.metrics_calc.compute(all_logits, all_labels)
        return avg_loss, metrics

    @torch.no_grad()
    def _validate_epoch(self) -> tuple:
        """单轮验证"""
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []

        for batch in self.val_loader:
            pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            outputs = self.model(pixel_values)
            loss = self._loss_fn(outputs["logits"], labels)

            total_loss += loss.item()
            all_logits.append(outputs["logits"].cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        metrics = self.metrics_calc.compute(all_logits, all_labels)

        # 记录混淆矩阵
        cm = metrics.pop("confusion_matrix", None)
        if cm is not None:
            self.logger.log_confusion_matrix(cm, self.class_names, self.global_step, prefix="val")

        # 打印详细分类报告
        report = self.metrics_calc.detailed_report(all_logits, all_labels)
        print(f"\n--- Classification Report (Epoch {self.current_epoch+1}) ---\n{report}")

        return avg_loss, metrics

    def _save_checkpoint(self, epoch: int, val_loss: float, metrics: dict, is_best: bool = False):
        """保存检查点"""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "metrics": {k: v for k, v in metrics.items() if not isinstance(v, list)},
            "config": self.config,
            "class_names": self.class_names,
        }
        if self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
            torch.save(ckpt, path)
            print(f"  [Checkpoint] Best model saved ({self.save_best_metric}: {self.best_metric_value:.4f}) -> {path}")

        # 定期保存
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
        torch.save(ckpt, path)

        # 清理旧的 top-k 检查点
        all_ckpts = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if len(all_ckpts) > self.save_top_k + 5:  # 保留一些余量
            for old_ckpt in all_ckpts[:-self.save_top_k - 5]:
                old_ckpt.unlink()

    def _final_evaluate(self):
        """最终测试集评估"""
        print("\n--- Final Test Evaluation ---")
        self.model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                outputs = self.model(pixel_values)
                all_logits.append(outputs["logits"].cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        metrics = self.metrics_calc.compute(all_logits, all_labels)
        print("\nTest Metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            elif isinstance(v, list):
                print(f"  {k}: (confusion matrix)")
                print(np.array(v))

        report = self.metrics_calc.detailed_report(all_logits, all_labels)
        print(f"\nTest Classification Report:\n{report}")

    @staticmethod
    def _set_seed(seed: int):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
