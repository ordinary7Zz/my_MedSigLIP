# MedSigLIP 甲状腺超声图像分类

基于 Google MedSigLIP 预训练权重的甲状腺超声图像迁移学习工程，支持**二分类（良恶性）**和**多分类（TIRADS 分级）**。

---

## 训练流程概述

```
┌──────────────────────────────────────────────────────────────────────┐
│  Step 1: 下载 MedSigLIP 预训练权重 (~1.6 GB)                          │
│          ↓                                                           │
│  Step 2: 准备你的甲状腺超声数据（按类别文件夹组织）                       │
│          ↓                                                           │
│  Step 3: [冻结大部分] MedSigLIP ViT Encoder                           │
│               → 提取 768/1024 维医学图像特征向量                       │
│                    → 接一个 可训练的 Linear 分类头                     │
│                         → 输出: 良性/恶性 或 TIRADS 1-5              │
│          ↓                                                           │
│  Step 4: 仅训练分类头（+可选解冻 ViT 最后几层），不重新训练整个 MedSigLIP │
└──────────────────────────────────────────────────────────────────────┘
```

**本质**：MedSigLIP 提供预训练好的"医学影像理解能力"（视觉编码器），你只需要训练一个轻量的分类头即可适配你的任务。这就是迁移学习。

---

## 项目结构

```
my_MedSigLIP/
├── configs/
│   ├── binary_cls.yaml           ← 二分类配置
│   └── multi_cls.yaml            ← TIRADS 多分类配置
├── src/
│   ├── data/
│   │   ├── dataset.py            ← 按类文件夹加载，自动灰度→三通道
│   │   └── transforms.py         ← 超声图像增强
│   ├── models/
│   │   └── classifier.py         ← MedSigLIP ViT + Linear 分类头
│   ├── trainers/
│   │   └── trainer.py            ← 训练引擎（AMP、早停、梯度累积）
│   └── utils/
│       ├── metrics.py            ← 评估指标（AUC/Sens/Spec/F1/Kappa）
│       └── logger.py             ← TensorBoard + 混淆矩阵
├── scripts/
│   ├── download_weights.py       ← 预下载权重到本地
│   ├── train.py                  ← 主训练入口
│   ├── evaluate.py               ← 评估 + 可视化
│   ├── inference.py              ← 单张/批量推理
│   └── prepare_data.py           ← 数据切分 train/val/test
├── pretrained/                   ← 预下载权重存放目录（需自行创建或运行下载）
├── requirements.txt
└── README.md
```

---

## 快速开始

### Step 1: 环境安装

```bash
pip install -r requirements.txt
```

### Step 2: 下载 MedSigLIP 预训练权重

**务必提前下载到本地**，避免训练时网络问题导致失败（权重约 1.6 GB）。

```bash
# 方式 A：用项目脚本下载（推荐）
python scripts/download_weights.py --output ./pretrained/medsiglip-448

# 国内网络慢？换镜像：
HF_ENDPOINT=https://hf-mirror.com python scripts/download_weights.py --output ./pretrained/medsiglip-448

# 方式 B：手动用 huggingface-cli 下载
pip install huggingface_hub
huggingface-cli download google/medsiglip-448 --local-dir ./pretrained/medsiglip-448
```

> **前置条件**：
> 1. 先在 https://huggingface.co/google/medsiglip-448 同意 Health AI Developer Foundations 使用条款
> 2. 执行 `huggingface-cli login` 登录你的 HuggingFace 账号

下载完成后，修改配置文件中的 `model.name` 为本地路径：

```yaml
model:
  name: "./pretrained/medsiglip-448"   # 改为本地路径
  local_files_only: true               # 禁止联网
```

如果不修改配置，首次运行 `from_pretrained("google/medsiglip-448")` 也会自动下载到 HuggingFace 缓存目录（`~/.cache/huggingface/`），但网络不稳定时可能失败。

### Step 3: 准备数据

按以下结构组织甲状腺超声图像（子文件夹名即为类别名）：

```
data/
├── thyroid_binary/              # 二分类数据
│   ├── train/
│   │   ├── benign/              # 良性图像
│   │   │   ├── case001.png
│   │   │   └── ...
│   │   └── malignant/           # 恶性图像
│   │       └── ...
│   ├── val/
│   │   ├── benign/
│   │   └── malignant/
│   └── test/                    # 可选
│       ├── benign/
│       └── malignant/
│
└── thyroid_tirads/              # TIRADS 多分类数据
    ├── train/
    │   ├── tirads_1/
    │   ├── tirads_2/
    │   ├── tirads_3/
    │   ├── tirads_4/
    │   └── tirads_5/
    ├── val/
    │   └── ...
    └── test/
        └── ...
```

如果数据还未切分，可用项目脚本一键处理：

```bash
# 从按类别分的文件夹自动切分 train/val/test
python scripts/prepare_data.py \
    --from_folders data/raw_thyroid \
    --output data/thyroid_binary \
    --train_ratio 0.7 --val_ratio 0.15

# 或从 CSV 标注文件构建
python scripts/prepare_data.py \
    --csv annotations.csv \
    --image_col image_path \
    --label_col label \
    --output data/thyroid_binary \
    --train_ratio 0.7 --val_ratio 0.15
```

### Step 4: 训练

```bash
# 二分类（良恶性）
python scripts/train.py --config configs/binary_cls.yaml

# TIRADS 多分类
python scripts/train.py --config configs/multi_cls.yaml --pretrained checkpoints/binary_cls/best_model.pt

# 指定 GPU
python scripts/train.py --config configs/binary_cls.yaml --device cuda:0

# 从断点恢复训练
python scripts/train.py --config configs/binary_cls.yaml --resume checkpoints/binary_cls/best_model.pt
```

训练过程中会：自动计算类别权重（处理不平衡）、Cosine 学习率衰减、验证集上早停、保存最优模型。

### Step 5: 评估

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/binary_cls/best_model.pt \
    --config configs/binary_cls.yaml \
    --split test \
    --visualize
```

生成：ROC/PR 曲线、混淆矩阵图、错误样本分析。

### Step 6: 推理

```bash
# 单张图像
python scripts/inference.py \
    --checkpoint checkpoints/binary_cls/best_model.pt \
    --config configs/binary_cls.yaml \
    --input path/to/ultrasound.png

# 文件夹批量推理，输出概率
python scripts/inference.py \
    --checkpoint checkpoints/binary_cls/best_model.pt \
    --config configs/binary_cls.yaml \
    --input data/ultrasound_batch/ \
    --output predictions.csv \
    --output_probs
```

---

## 微调策略选择

| 策略 | 显存需求 | 推荐数据量 | 说明 |
|------|---------|-----------|------|
| `linear_probing` | ~4-8 GB | <5,000 | 冻结所有 ViT 层，仅训练 Linear 分类头 |
| `partial` **（默认）** | ~12-16 GB | 5,000-30,000 | 解冻 ViT 最后 6 层 + 分类头 |
| `full` | ~32-40 GB | >30,000 | 全量微调，需要 A100 |

在配置文件中修改 `model.fine_tune_strategy`，无需改动代码。

---

## 评估指标

### 二分类（良恶性）
| 指标 | 含义 |
|------|------|
| AUC-ROC | 核心：整体判别能力 |
| Sensitivity | 恶性检出率（低 = 漏诊） |
| Specificity | 良性正确率（低 = 过度诊断） |
| F1 | 精确率与召回率的调和平均 |

### TIRADS 多分类
| 指标 | 含义 |
|------|------|
| Macro F1 | 核心：各类别 F1 平均 |
| Accuracy | 总体准确率 |
| Cohen's Kappa | 排除随机一致的分类一致性 |
| Weighted F1 | 按样本量加权的 F1 |

---

## 注意事项

1. **超声不在此模型的预训练分布中**（预训练覆盖：胸片、皮肤镜、眼底、病理、CT/MR），建议先 `linear_probing` 跑 baseline 验证效果，再切换 `partial`
2. **灰度图自动处理**：代码内部 `np.stack([gray]*3, axis=-1)` 转三通道 RGB
3. **类别不平衡**：`use_class_weights: true` 自动根据训练集分布计算权重
4. **TIRADS 分级**：默认 5 级，按你的标注体系修改 `num_classes`
