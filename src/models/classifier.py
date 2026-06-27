"""
MedSigLIP 分类模型
支持三种微调策略：linear_probing / partial / full
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class MedSigLIPClassifier(nn.Module):
    """
    基于 MedSigLIP 视觉编码器的医学图像分类器

    架构:
        [图像 448×448] → [MedSigLIP ViT Encoder] → [CLS Token / Pooled Output]
            → [Dropout] → [Linear(num_classes)] → [Logits]

    Args:
        model_name: HuggingFace 模型名，如 "google/medsiglip-448"
        num_classes: 分类类别数
        fine_tune_strategy: "linear_probing" | "partial" | "full"
        unfreeze_last_n: partial 模式下解冻最后 N 个 Transformer Block
        dropout: 分类头 dropout 概率
    """

    def __init__(
        self,
        model_name: str = "google/medsiglip-448",
        num_classes: int = 2,
        fine_tune_strategy: str = "partial",
        unfreeze_last_n: int = 6,
        dropout: float = 0.1,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.fine_tune_strategy = fine_tune_strategy
        self.unfreeze_last_n = unfreeze_last_n

        # 加载完整 MedSigLIP 模型（仅使用视觉编码器部分）
        # model_name 可以是 HuggingFace ID 或本地路径如 "./pretrained/medsiglip-448"
        print(f"[Model] Loading from: {model_name} (local_files_only={local_files_only})")
        self.full_model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )

        # 提取视觉编码器（ViT）
        self.vision_encoder = self.full_model.vision_model
        # 视觉编码器的配置
        config = AutoConfig.from_pretrained(model_name)
        vision_config = config.vision_config

        self.emb_dim = vision_config.hidden_size
        print(f"[Model] Vision encoder embedding dim: {self.emb_dim}")

        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.emb_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        # 应用微调策略
        self._apply_fine_tune_strategy()

        # 打印可训练参数数量
        self._log_trainable_params()

    def _apply_fine_tune_strategy(self):
        """根据策略冻结/解冻参数"""

        if self.fine_tune_strategy == "linear_probing":
            # 冻结整个 ViT，只训练分类头
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            print("[Model] Strategy: Linear Probing (frozen ViT)")

        elif self.fine_tune_strategy == "partial":
            # 先全部冻结
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

            # 解冻最后 N 个 Transformer Block
            encoder_layers = self.vision_encoder.encoder.layers
            num_total_layers = len(encoder_layers)
            unfreeze_from = max(0, num_total_layers - self.unfreeze_last_n)

            for i in range(unfreeze_from, num_total_layers):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = True

            # 同时解冻 LayerNorm（在 encoder 层级上）
            # ViT 的 pre_layrnorm 或 post_layernorm
            if hasattr(self.vision_encoder, 'pre_layrnorm'):
                for param in self.vision_encoder.pre_layrnorm.parameters():
                    param.requires_grad = True
            # 部分模型也有最后的 layernorm
            # 保守起见保留冻结，避免影响过多

            print(
                f"[Model] Strategy: Partial Fine-tuning "
                f"(unfroze last {self.unfreeze_last_n}/{num_total_layers} blocks)"
            )

        elif self.fine_tune_strategy == "full":
            for param in self.vision_encoder.parameters():
                param.requires_grad = True
            print("[Model] Strategy: Full Fine-tuning (all parameters trainable)")

        else:
            raise ValueError(
                f"Unknown fine_tune_strategy: {self.fine_tune_strategy}. "
                f"Choose: linear_probing | partial | full"
            )

    def _log_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[Model] Total params: {total/1e6:.1f}M | "
            f"Trainable: {trainable/1e6:.1f}M ({100*trainable/total:.1f}%)"
        )

    def forward(self, pixel_values: torch.Tensor) -> dict:
        """
        Args:
            pixel_values: (B, C, H, W) 归一化后的图像张量

        Returns:
            dict with:
                logits: (B, num_classes) 分类 logits
                embeddings: (B, emb_dim)  视觉嵌入
        """
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)

        # ViT 的 pooler_output = [CLS] token 经过一个 tanh 激活的全连接层
        # 如果 pooler_output 不可用，回退到 last_hidden_state[:, 0]
        if vision_outputs.pooler_output is not None:
            embeddings = vision_outputs.pooler_output
        else:
            embeddings = vision_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        embeddings = self.dropout(embeddings)
        logits = self.classifier(embeddings)

        return {"logits": logits, "embeddings": embeddings}

    def get_embeddings(self, dataloader, device: str = "cuda") -> tuple:
        """提取整个数据集的视觉嵌入（用于后续分析）"""
        self.eval()
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["label"]

                outputs = self.forward(pixel_values)
                all_embeddings.append(outputs["embeddings"].cpu())
                all_labels.append(labels)

        return (
            torch.cat(all_embeddings, dim=0),
            torch.cat(all_labels, dim=0),
        )
