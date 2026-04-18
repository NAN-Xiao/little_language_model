"""
Vision Transformer (ViT) 模块 —— 学习用实现
==============================================

本文件实现了：
  1. PatchEmbedding        — 把图像切成固定大小的 patch，再线性投影到 d_model 维
  2. ViTEncoderBlock       — 一个标准的 Transformer 编码器块（双向自注意力 + FFN）
  3. VisionTransformer     — 完整的 ViT 图像分类模型（单图识别）
  4. ImageTextProjector    — 将图像特征投影到文本 Decoder 的嵌入空间
  5. VisionLanguageModel   — 图文多模态模型（ViT 编码图像 → 特征拼接到文本 token 前 → Decoder 生成）

核心思想：
  ViT 把一张 2D 图像视为一串 1D 的 patch 序列，就像 NLP 中的 token 序列一样，
  然后用标准的 Transformer Encoder 来建模 patch 之间的关系。

  与 Decoder-Only 语言模型的区别：
  ┌──────────────────────────────────────────────────────────────────┐
  │             Decoder (语言模型)       │   Encoder (ViT)           │
  │ ─────────────────────────────────── │ ───────────────────────── │
  │ 因果掩码(causal mask)：只看过去      │ 无掩码：所有 patch 互相看  │
  │ 输入是 token id → Embedding         │ 输入是 image → PatchEmbed │
  │ 自回归生成                           │ 全局理解 / 分类           │
  └──────────────────────────────────────────────────────────────────┘

参考论文：
  - "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
  - "LLaVA: Large Language and Vision Assistant" (Liu et al., 2023)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import MultiHeadAttention
from model.feedforward import PositionwiseFeedForward


# ---------------------------------------------------------------------------
# 0. 配置
# ---------------------------------------------------------------------------
@dataclass
class ViTConfig:
    """ViT 相关的超参数。"""
    image_size: int = 224          # 输入图像的边长（假设正方形）
    patch_size: int = 16           # 每个 patch 的边长；224/16 = 14，共 14×14 = 196 个 patch
    in_channels: int = 3           # 输入通道数（RGB = 3）
    d_model: int = 768             # patch 投影后的维度，需与 Transformer 对齐
    n_heads: int = 12              # 注意力头数
    n_layers: int = 12             # ViT 编码器的层数（ViT-Base = 12）
    d_ff: int = 3072               # FFN 中间层维度（通常是 d_model × 4）
    dropout: float = 0.1           # dropout 比例
    num_classes: int = 1000        # 分类类别数（ImageNet = 1000）


# ---------------------------------------------------------------------------
# 1. Patch Embedding — 将图像切 patch 并投影
# ---------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """
    将一张图像切成不重叠的 patch，并通过线性投影映射到 d_model 维。

    实现方式：用一个 kernel_size = stride = patch_size 的 Conv2d 一步完成切分+投影。
    这等价于：
      1) 把图像按 patch_size 网格切成 N = (H/P) × (W/P) 个 patch
      2) 每个 patch 展平为 P*P*C 维向量
      3) 对每个向量做线性投影 → d_model 维

    ┌─────────────────────────────────────────────┐
    │ 输入图像 (B, C, H, W)   例如 (B, 3, 224, 224)│
    │                                             │
    │ ┌────┬────┬────┬─···─┬────┐                 │
    │ │p_1 │p_2 │p_3 │     │p_14│  ← 第1行14个patch│
    │ ├────┼────┼────┼─···─┼────┤                 │
    │ │p_15│p_16│    │     │p_28│  ← 第2行          │
    │ ├────┼────┼────┼─···─┼────┤                 │
    │ │ ...│    │    │     │ ...│                  │
    │ ├────┼────┼────┼─···─┼────┤                 │
    │ │p183│    │    │     │p196│  ← 第14行         │
    │ └────┴────┴────┴─···─┴────┘                 │
    │                                             │
    │ 每个 patch: 16×16×3 = 768 维  → 投影到 d_model│
    │ 输出: (B, 196, d_model)                      │
    └─────────────────────────────────────────────┘

    参数:
        image_size  (int): 输入图像的边长（假设正方形）
        patch_size  (int): 每个 patch 的边长
        in_channels (int): 输入图像的通道数（RGB = 3）
        d_model     (int): 投影后的特征维度
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 768,
    ):
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) 必须能被 patch_size ({patch_size}) 整除"
        )
        self.num_patches = (image_size // patch_size) ** 2  # 196

        # Conv2d(in_channels=3, out_channels=d_model, kernel=16, stride=16)
        # 效果：每个 16×16 的区域被投影为一个 d_model 维向量
        # 输出形状：(B, d_model, H/P, W/P) = (B, 768, 14, 14)
        self.projection = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) — 例如 (B, 3, 224, 224)
        返回: (B, num_patches, d_model) — 例如 (B, 196, 768)
        """
        # (B, d_model, H/P, W/P) 例如 (B, 768, 14, 14)
        x = self.projection(x)
        # flatten(2) 把空间维合并: (B, 768, 14, 14) → (B, 768, 196)
        # transpose: (B, 768, 196) → (B, 196, 768) 即 (B, num_patches, d_model)
        x = x.flatten(2).transpose(1, 2)
        return x


# ---------------------------------------------------------------------------
# 2. ViT Encoder Block — 双向自注意力 + FFN
# ---------------------------------------------------------------------------
class ViTEncoderBlock(nn.Module):
    """
    一个标准的 Transformer Encoder 块，用于 ViT。

    与 decoder.py 中的 DecoderBlock 相比：
      - DecoderBlock 使用 *因果掩码* (causal mask)，每个位置只能看到之前的 token
      - ViTEncoderBlock *不使用掩码*，每个 patch 可以关注所有其他 patch（双向注意力）

    结构（Pre-Norm 风格，与本项目 DecoderBlock 一致）：
      x  ──→ LayerNorm → MultiHeadAttention → Dropout → (+残差) ──→
         ──→ LayerNorm → FFN                → Dropout → (+残差) ──→ out

    参数:
        d_model (int):  特征维度
        n_heads (int):  注意力头数
        d_ff    (int):  FFN 中间层维度
        dropout (float): dropout 比例
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 复用已有的多头注意力和前馈网络实现
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, d_model)   seq_len = num_patches + 1（含 [CLS]）
        返回: (B, seq_len, d_model)

        注意：这里 *没有* mask 参数，因为图像 patch 之间是双向可见的，
        不需要因果掩码，也没有 padding（图像尺寸固定）。
        """
        # --- 自注意力子层 ---
        normed = self.norm1(x)
        # query = key = value = normed → 自注意力，mask=None → 双向
        x = x + self.dropout1(self.self_attn(normed, normed, normed, mask=None))

        # --- 前馈网络子层 ---
        normed = self.norm2(x)
        x = x + self.dropout2(self.ffn(normed))
        return x


# ---------------------------------------------------------------------------
# 3. VisionTransformer — 完整的图像分类模型（单图识别）
# ---------------------------------------------------------------------------
class VisionTransformer(nn.Module):
    """
    完整的 Vision Transformer (ViT)，用于图像分类。

    整体流程：
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  Image (B,3,224,224)                                            │
    │      │                                                          │
    │      ▼                                                          │
    │  PatchEmbedding  →  (B, 196, 768)                               │
    │      │                                                          │
    │      ▼                                                          │
    │  Prepend [CLS] token  →  (B, 197, 768)                         │
    │      │               ↑                                          │
    │      │   + Learnable Position Embedding (1, 197, 768)           │
    │      ▼                                                          │
    │  ViTEncoderBlock × N  →  (B, 197, 768)                         │
    │      │                                                          │
    │      ▼                                                          │
    │  LayerNorm                                                      │
    │      │                                                          │
    │      ▼                                                          │
    │  取 [CLS] 输出 → (B, 768)                                       │
    │      │                                                          │
    │      ▼                                                          │
    │  Classification Head (Linear) → (B, num_classes)                │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

    [CLS] token 的作用：
      - 一个可学习的向量，拼接在 patch 序列的最前面
      - 经过多层 Encoder 后，[CLS] 聚合了全图信息
      - 最终用 [CLS] 的输出做分类（类似 BERT 的 [CLS]）

    位置编码：
      - ViT 原版使用 *可学习的* 位置编码（nn.Parameter），而非正弦位置编码
      - 每个位置（包括 [CLS]）有一个可训练的 d_model 维向量

    参数:
        cfg (ViTConfig): ViT 配置
    """

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg

        # --- Patch Embedding ---
        self.patch_embed = PatchEmbedding(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            d_model=cfg.d_model,
        )
        num_patches = self.patch_embed.num_patches  # 196

        # --- [CLS] Token ---
        # 可学习的 [CLS] token，形状 (1, 1, d_model)
        # 训练时会随梯度更新，最终学会聚合全局图像信息
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))

        # --- 可学习位置编码 ---
        # num_patches + 1 是因为要给 [CLS] 也分配一个位置编码
        # 与 positional.py 中的正弦编码不同，这里是全可学习的
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.d_model))

        self.pos_drop = nn.Dropout(cfg.dropout)

        # --- Encoder Blocks ---
        self.blocks = nn.ModuleList([
            ViTEncoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        self.norm = nn.LayerNorm(cfg.d_model)

        # --- 分类头 ---
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)

        self._init_weights()

    def _init_weights(self):
        # 位置编码用截断正态分布初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        提取图像特征（不含分类头），供外部模型（如多模态）复用。

        pixel_values: (B, C, H, W) 例如 (B, 3, 224, 224)
        返回: (B, num_patches + 1, d_model) — 含 [CLS] 的完整序列特征
        """
        B = pixel_values.size(0)

        # 1) Patch embedding: (B, 3, 224, 224) → (B, 196, 768)
        x = self.patch_embed(pixel_values)

        # 2) 在最前面拼接 [CLS] token
        #    cls_token.expand(B, -1, -1) 把 (1,1,768) 扩展为 (B,1,768)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 197, 768)

        # 3) 加上位置编码
        x = x + self.pos_embed  # (B, 197, 768) + (1, 197, 768) → broadcast → (B, 197, 768)
        x = self.pos_drop(x)

        # 4) 通过 N 层 Encoder Block
        for block in self.blocks:
            x = block(x)

        # 5) 最终 LayerNorm
        x = self.norm(x)
        return x

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        图像分类的 forward。

        pixel_values: (B, C, H, W)
        返回: (B, num_classes) — 未经 softmax 的 logits

        使用示例:
            cfg = ViTConfig(num_classes=10)
            model = VisionTransformer(cfg)
            images = torch.randn(4, 3, 224, 224)   # batch of 4 images
            logits = model(images)                  # (4, 10)
            loss = F.cross_entropy(logits, labels)
        """
        x = self.forward_features(pixel_values)  # (B, 197, 768)
        cls_output = x[:, 0]  # 取 [CLS] token 的输出: (B, 768)
        logits = self.head(cls_output)  # (B, num_classes)
        return logits


# ---------------------------------------------------------------------------
# 4. Image-Text Projector — 图像特征 → 文本嵌入空间
# ---------------------------------------------------------------------------
class ImageTextProjector(nn.Module):
    """
    将 ViT 输出的图像特征投影到与文本 Decoder 相同的嵌入空间。

    很多多模态模型（LLaVA、Qwen-VL 等）的核心思路就是：
      图像特征 (来自 ViT)  ──→  投影层  ──→  与文本 token embedding 拼接  ──→  送入 Decoder

    本投影器使用两层 MLP + GELU 激活：
      Linear(vit_d_model → llm_d_model) → GELU → Linear(llm_d_model → llm_d_model)

    参数:
        vit_d_model (int): ViT 输出的特征维度
        llm_d_model (int): 文本 Decoder 的特征维度
    """

    def __init__(self, vit_d_model: int, llm_d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vit_d_model, llm_d_model),
            nn.GELU(),
            nn.Linear(llm_d_model, llm_d_model),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        image_features: (B, num_image_tokens, vit_d_model)
        返回:           (B, num_image_tokens, llm_d_model)
        """
        return self.proj(image_features)


# ---------------------------------------------------------------------------
# 5. VisionLanguageModel — 图文多模态理解
# ---------------------------------------------------------------------------
class VisionLanguageModel(nn.Module):
    """
    图文多模态模型：将图像理解与文本生成结合。

    整体架构（类似 LLaVA 的简化版）：
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  Image (B,3,224,224)          Text token ids (B, T)                     │
    │      │                            │                                     │
    │      ▼                            ▼                                     │
    │  VisionTransformer          Token Embedding                             │
    │  (forward_features)         (from Decoder)                              │
    │      │                            │                                     │
    │      ▼                            │                                     │
    │  (B, N_img, vit_d)                │                                     │
    │      │                            │                                     │
    │      ▼                            │                                     │
    │  ImageTextProjector               │                                     │
    │      │                            │                                     │
    │      ▼                            ▼                                     │
    │  (B, N_img, d_model)     (B, T, d_model)                               │
    │      │                        │                                         │
    │      └────── concat ──────────┘                                         │
    │                  │                                                       │
    │                  ▼                                                       │
    │  Combined: (B, N_img + T, d_model)                                      │
    │                  │                                                       │
    │                  ▼                                                       │
    │   + Position Encoding                                                   │
    │                  │                                                       │
    │                  ▼                                                       │
    │  Decoder Blocks (causal mask)                                           │
    │                  │                                                       │
    │                  ▼                                                       │
    │  Output Projection → (B, N_img + T, vocab_size)                         │
    │                  │                                                       │
    │                  ▼                                                       │
    │  取文本部分 logits → 计算 loss / 生成文本                                │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    工作原理：
      1. ViT 编码图像为一组 "视觉 token"
      2. Projector 将视觉 token 对齐到文本嵌入空间
      3. 视觉 token 拼接在文本 token 前面
      4. 整个序列送入因果 Decoder，像处理普通文本一样做自回归生成
      5. 模型看到图像+问题，就能生成回答（图文问答）

    参数:
        vit       (VisionTransformer): 预训练好的 ViT 图像编码器
        decoder   (nn.Module):         已有的文本 Decoder（从 Transformer 中获取）
        projector (ImageTextProjector): 图文投影层
        cfg       (ModelConfig-like):   需包含 vocab_size, d_model, pad_token_id 等
    """

    def __init__(
        self,
        vit: VisionTransformer,
        decoder: nn.Module,
        output_proj: nn.Linear,
        projector: ImageTextProjector,
        pad_token_id: int = 0,
        freeze_vit: bool = True,
    ):
        super().__init__()
        self.vit = vit
        self.decoder = decoder           # Decoder from decoder.py
        self.output_proj = output_proj   # Linear(d_model, vocab_size)
        self.projector = projector
        self.pad_token_id = pad_token_id

        # 通常在多模态训练中冻结 ViT，只训练 projector 和 decoder
        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        构建因果掩码（下三角矩阵），用于 Decoder 的自回归约束。

        返回: (1, 1, seq_len, seq_len) — 可广播到 (B, n_heads, seq_len, seq_len)

        示例 (seq_len=5):
          [[1, 0, 0, 0, 0],
           [1, 1, 0, 0, 0],
           [1, 1, 1, 0, 0],
           [1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1]]
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        图文多模态 forward。

        参数:
            pixel_values: (B, C, H, W) — 输入图像
            input_ids:    (B, T)       — 输入文本的 token id 序列

        返回:
            logits: (B, N_img + T, vocab_size) — 完整序列的预测 logits
                    训练时通常只对文本部分计算 loss

        使用示例（训练）:
            # 假设已经构建好模型
            logits = model(images, text_ids)
            # 只对文本区域计算 loss
            num_img_tokens = model.get_num_image_tokens()
            text_logits = logits[:, num_img_tokens:-1, :]   # 去掉最后一个预测
            text_targets = text_ids[:, 1:]                   # shifted targets
            loss = F.cross_entropy(
                text_logits.reshape(-1, text_logits.size(-1)),
                text_targets.reshape(-1),
                ignore_index=model.pad_token_id,
            )
        """
        # ---- 图像编码 ----
        # (B, N_img+1, vit_d_model)  含 [CLS]
        image_features = self.vit.forward_features(pixel_values)
        # 投影到文本空间: (B, N_img+1, llm_d_model)
        image_embeds = self.projector(image_features)

        # ---- 文本嵌入 ----
        # 直接复用 Decoder 的 token_embedding
        d_model = self.decoder.d_model
        text_embeds = self.decoder.token_embedding(input_ids) * (d_model ** 0.5)
        # (B, T, d_model)

        # ---- 拼接：[图像 tokens | 文本 tokens] ----
        combined = torch.cat([image_embeds, text_embeds], dim=1)
        # (B, N_img+1+T, d_model)

        total_len = combined.size(1)

        # ---- 位置编码 ----
        combined = self.decoder.pos_encoding(combined)

        # ---- 因果掩码 ----
        causal_mask = self._make_causal_mask(total_len, combined.device)

        # ---- 通过 Decoder Blocks ----
        x = combined
        for layer in self.decoder.layers:
            x = layer(x, tgt_mask=causal_mask)
        x = self.decoder.norm(x)

        # ---- 输出投影 ----
        logits = self.output_proj(x)  # (B, N_img+1+T, vocab_size)
        return logits

    def get_num_image_tokens(self) -> int:
        """返回图像编码后产生的 token 数量（含 [CLS]）。"""
        return self.vit.patch_embed.num_patches + 1

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        eos_token_id: int = 2,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """
        图文多模态生成：给定图像和提示文本，自回归生成回答。

        参数:
            pixel_values:   (B, C, H, W)
            input_ids:      (B, T) — 提示文本
            max_new_tokens: 最大生成 token 数
            eos_token_id:   结束符 id
            temperature:    采样温度
            top_k:          top-k 采样

        返回:
            generated_ids: (B, T + new_tokens) — 包含原始 prompt 和生成内容
        """
        self.eval()

        # 图像编码（只需做一次）
        image_features = self.vit.forward_features(pixel_values)
        image_embeds = self.projector(image_features)  # (B, N_img, d)

        generated = input_ids.clone()
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            # 文本嵌入
            d_model = self.decoder.d_model
            text_embeds = self.decoder.token_embedding(generated) * (d_model ** 0.5)

            # 拼接
            combined = torch.cat([image_embeds, text_embeds], dim=1)
            total_len = combined.size(1)
            combined = self.decoder.pos_encoding(combined)

            causal_mask = self._make_causal_mask(total_len, combined.device)

            x = combined
            for layer in self.decoder.layers:
                x = layer(x, tgt_mask=causal_mask)
            x = self.decoder.norm(x)

            # 只取最后一个位置的 logits
            next_logits = self.output_proj(x[:, -1, :]) / max(temperature, 1e-5)

            if top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < top_k_vals[:, -1:]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            next_token = next_token.masked_fill(finished.unsqueeze(1), self.pad_token_id)

            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == eos_token_id)
            if finished.all():
                break

        return generated


# ---------------------------------------------------------------------------
# 快捷构建函数
# ---------------------------------------------------------------------------
def build_vision_language_model(
    vit_cfg: ViTConfig,
    llm_transformer: nn.Module,
    freeze_vit: bool = True,
) -> VisionLanguageModel:
    """
    快捷构建 VisionLanguageModel。

    参数:
        vit_cfg:         ViT 配置
        llm_transformer: 已有的 Transformer 实例（来自 transformer.py）
        freeze_vit:      是否冻结 ViT 参数

    返回:
        VisionLanguageModel 实例

    使用示例:
        from little_language_model.config import ModelConfig, ViTConfig
        from little_language_model.model.transformer import Transformer

        model_cfg = ModelConfig(vocab_size=16000, d_model=768)
        llm = Transformer(model_cfg)

        vit_cfg = ViTConfig(d_model=768, num_classes=0)  # num_classes 在多模态中不需要
        vlm = build_vision_language_model(vit_cfg, llm, freeze_vit=True)

        images = torch.randn(2, 3, 224, 224)
        text_ids = torch.randint(0, 16000, (2, 32))
        logits = vlm(images, text_ids)
    """
    vit = VisionTransformer(vit_cfg)
    projector = ImageTextProjector(
        vit_d_model=vit_cfg.d_model,
        llm_d_model=llm_transformer.cfg.d_model,
    )
    vlm = VisionLanguageModel(
        vit=vit,
        decoder=llm_transformer.decoder,
        output_proj=llm_transformer.output_proj,
        projector=projector,
        pad_token_id=llm_transformer.cfg.pad_token_id,
        freeze_vit=freeze_vit,
    )
    return vlm
