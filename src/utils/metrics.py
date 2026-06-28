"""
评估指标计算模块
支持二分类和多分类的常用指标。
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    cohen_kappa_score,
    classification_report,
)


class MetricsCalculator:
    """评估指标计算器"""

    def __init__(self, num_classes: int, metrics_list: list):
        self.num_classes = num_classes
        self.metrics_list = metrics_list
        self.is_binary = num_classes == 2

    def compute(self, logits: np.ndarray, labels: np.ndarray) -> dict:
        """
        Args:
            logits: (N, num_classes) 原始 logits
            labels: (N,) 真实标签

        Returns:
            dict: 指标名 -> 标量值
        """
        results = {}

        # 先统一计算概率
        probs_all = self._softmax(logits)  # (N, C), 每行和=1

        if self.num_classes == 1:
            probs = probs_all[:, 0]  # 单类概率
            preds = (probs >= 0.5).astype(int)
        elif self.is_binary:
            probs = probs_all[:, 1]  # 正类(恶性)概率
            preds = logits.argmax(axis=1)
        else:
            probs = probs_all            # 多分类使用完整概率矩阵
            preds = logits.argmax(axis=1)

        for metric_name in self.metrics_list:
            if metric_name == "accuracy":
                results["accuracy"] = accuracy_score(labels, preds)

            elif metric_name == "f1":
                average = "binary" if self.is_binary else "macro"
                results["f1"] = f1_score(labels, preds, average=average)

            elif metric_name == "macro_f1":
                results["macro_f1"] = f1_score(labels, preds, average="macro")

            elif metric_name == "weighted_f1":
                results["weighted_f1"] = f1_score(labels, preds, average="weighted")

            elif metric_name == "precision":
                average = "binary" if self.is_binary else "macro"
                results["precision"] = precision_score(labels, preds, average=average, zero_division=0)

            elif metric_name == "recall":
                average = "binary" if self.is_binary else "macro"
                results["recall"] = recall_score(labels, preds, average=average, zero_division=0)

            elif metric_name == "sensitivity":
                # sensitivity = recall for positive class
                if self.is_binary:
                    cm = confusion_matrix(labels, preds)
                    tn, fp, fn, tp = cm.ravel()
                    results["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                else:
                    results["sensitivity"] = recall_score(labels, preds, average="macro", zero_division=0)

            elif metric_name == "specificity":
                if self.is_binary:
                    cm = confusion_matrix(labels, preds)
                    tn, fp, fn, tp = cm.ravel()
                    results["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                else:
                    # 多分类 specificity 按 macro 计算
                    cm = confusion_matrix(labels, preds)
                    specificities = []
                    for i in range(self.num_classes):
                        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
                        fp = cm[:, i].sum() - cm[i, i]
                        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
                    results["specificity"] = np.mean(specificities)

            elif metric_name == "auc_roc":
                if self.is_binary:
                    results["auc_roc"] = roc_auc_score(labels, probs)
                else:
                    results["auc_roc"] = roc_auc_score(labels, probs_all, multi_class="ovr", average="macro")

            elif metric_name == "auc_roc_ovr":
                results["auc_roc_ovr"] = roc_auc_score(
                    labels, probs_all, multi_class="ovr", average="macro"
                )

            elif metric_name == "kappa":
                results["kappa"] = cohen_kappa_score(labels, preds)

            elif metric_name == "confusion_matrix":
                results["confusion_matrix"] = confusion_matrix(labels, preds).tolist()

        return results

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - x.max(axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def detailed_report(self, logits: np.ndarray, labels: np.ndarray) -> str:
        """生成详细的分类报告"""
        if self.is_binary:
            probs = 1.0 / (1.0 + np.exp(-logits.flatten())) if logits.shape[1] == 1 else self._softmax(logits)[:, 1]
            preds = (probs >= 0.5).astype(int)
        else:
            preds = logits.argmax(axis=1)

        return classification_report(labels, preds, digits=4, zero_division=0)
