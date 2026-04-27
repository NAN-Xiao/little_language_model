"""
Vision Transformer (ViT) 与图文多模态 —— 从"图像像素"到"文字描述"的完整旅程
================================================================================

本文件实现了一条完整的流水线：
  1. PatchEmbedding        — 把图像切成 patch，变成像文字 token 一样的序列
  2. ViTBlock              — 一层 ViT: Attention + FFN（双向自注意力）
  3. VisionTransformer     — 完整的图像分类模型（理解单张图）
  4. ImageTextProjector    — 把图像特征"翻译"成文本能懂的语言
  5. VisionLanguageModel   — 图文多模态：看图说话、图文问答

═══════════════════════════════════════════════════════════════════════════════
【核心问题：图像和文字怎么做注意力？】
═══════════════════════════════════════════════════════════════════════════════

你可能已经熟悉了文字模型里的因果注意力（casual mask）：
  "今天天气真好" → 模型只能从左往右看，预测下一个字。

但图像是"一眼全看完"的——左上角的 patch 和右下角的 patch 是同时感知的。
这就产生了两种注意力方式：

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  图像内部注意力（ViT Encoder）                                              │
│  ───────────────────────────                                                │
│  没有 causal mask！所有 patch 互相可见。                                    │
│                                                                             │
│  例: 一张猫的图片                                                           │
│      patch 0 (左上角背景) ←→ patch 50 (猫耳朵) ←→ patch 100 (猫尾巴)       │
│      它们之间两两计算注意力，互相知道彼此的存在。                            │
│                                                                             │
│  这就像你一眼看完整张图，而不是从左到右扫描。                               │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  图文混合注意力（VisionLanguageModel）                                      │
│  ───────────────────────────────────                                        │
│  图像 patch 和文字 token 拼成一个长序列：                                    │
│                                                                             │
│      [img0][img1]...[img195][文字][文字][文字]...                           │
│       ↑ 图像部分（双向可见） ↑ |  ↑ 文字部分（因果可见）                     │
│                                                                             │
│  关键：整个序列一起过 causal mask，但 mask 的设计是"精华"：                │
│                                                                             │
│    • 图像 patch 之间：互相可见（双向）                                      │
│    • 文字看图像：可见（文字可以参考图像内容）                                │
│    • 文字看文字：只能看左边（因果）                                          │
│    • 图像看文字：不可见（图像不需要知道后面的文字）                          │
│                                                                             │
│  这就像一个学生在看画报（图像）的同时写作文（文字）：                        │
│    - 他可以随时回头看画报的任意部分                                         │
│    - 但写字只能从左到右，不能跳到未来                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
【注意力掩码的矩阵可视化】
═══════════════════════════════════════════════════════════════════════════════

假设 4 个图像 token + 5 个文字 token，共 9 个位置：

      img0 img1 img2 img3  txt0 txt1 txt2 txt3 txt4
    ┌────┬────┬────┬────┬────┬────┬────┬────┬────┐
img0│ 1  │ 1  │ 1  │ 1  │ 0  │ 0  │ 0  │ 0  │ 0  │  ← 图像看图像: 全可见
img1│ 1  │ 1  │ 1  │ 1  │ 0  │ 0  │ 0  │ 0  │ 0  │
img2│ 1  │ 1  │ 1  │ 1  │ 0  │ 0  │ 0  │ 0  │ 0  │
img3│ 1  │ 1  │ 1  │ 1  │ 0  │ 0  │ 0  │ 0  │ 0  │
    ├────┼────┼────┼────┼────┼────┼────┼────┼────┤
txt0│ 1  │ 1  │ 1  │ 1  │ 1  │ 0  │ 0  │ 0  │ 0  │  ← 文字看图像+自己: 可见
txt1│ 1  │ 1  │ 1  │ 1  │ 1  │ 1  │ 0  │ 0  │ 0  │     文字看未来文字: 不可见
txt2│ 1  │ 1  │ 1  │ 1  │ 1  │ 1  │ 1  │ 0  │ 0  │
txt3│ 1  │ 1  │ 1  │ 1  │ 1  │ 1  │ 1  │ 1  │ 0  │
txt4│ 1  │ 1  │ 1  │ 1  │ 1  │ 1  │ 1  │ 1  │ 1  │
    └────┴────┴────┴────┴────┴────┴────┴────┴────┘

1 = 可见（可以 attend），0 = 被遮挡（不能 attend）

这个 mask 的本质:
  • 图像部分是一个"全1小方块"（双向注意力）
  • 文字部分是一个"下三角"（因果注意力）
  • 文字看图像的部分全是 1（文字可以参考图像）
  • 图像看文字的部分全是 0（图像不依赖文字）

═══════════════════════════════════════════════════════════════════════════════
【为什么这样设计？】
═══════════════════════════════════════════════════════════════════════════════

1. 图像是"同时感知"的：
   人眼看图不是逐字阅读，而是一眼扫全图。所以图像 patch 之间不该有顺序限制。

2. 文字是"顺序生成"的：
   生成第 N 个字时，不能偷看第 N+1 个字。所以文字部分保持因果 mask。

3. 文字需要"看着图写"：
   每个文字 token 都应该能参考所有图像 patch。否则模型在回答"图里有什么"时，
   看不到图的内容。

4. 图像不需要"看文字"：
   图像编码在第一步就完成了，后续生成文字时不需要再修改图像表示。
   这节省了计算，也符合直觉（图就是图，不因你写什么而改变）。

═══════════════════════════════════════════════════════════════════════════════
【从像素到文字的维度流转】
═══════════════════════════════════════════════════════════════════════════════

以 batch_size=2, image=224×224, patch=16×16, d_model=768, text_len=10 为例：

  图像输入:     (2, 3, 224, 224)        2张RGB图
      ↓
  PatchEmbedding:  (2, 196, 768)        每张图切成14×14=196个patch，每个768维
      ↓
  + [CLS] token:   (2, 197, 768)        在前面加一个可学习的聚合token
      ↓
  + 位置编码:      (2, 197, 768)        每个位置学一个向量
      ↓
  ViT Encoder ×12: (2, 197, 768)        12层双向自注意力，patch之间互相理解
      ↓
  Projector:       (2, 197, 768)        投影到文本空间（如果vit_d和llm_d不同会变化）
      ↓
  文本嵌入:        (2, 10, 768)         "这是一只猫"的token embedding
      ↓
  拼接:            (2, 207, 768)        [图像197个 + 文字10个] 拼在一起
      ↓
  Causal Mask:     (1, 1, 207, 207)     上面画的那个特殊mask
      ↓
  Decoder Blocks:  (2, 207, 768)        文字token参考图像+前文，生成下一个字
      ↓
  Output Proj:     (2, 207, vocab_size) 每个位置预测下一个词的概率分布

"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision.attention import ViTMultiHeadAttention
from model.feedforward import PositionwiseFeedForward


# ─────────────────────────────────────────────────────────────────────────────
# 0. 配置
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ViTConfig:
    """
    ViT 超参数配置。

    关键数字关系:
      image_size=224, patch_size=16 → num_patches = (224/16)² = 14×14 = 196
      d_model=768 → 每个 patch 投影到 768 维（和文字模型的 d_model 对齐）
      n_heads=12  → 和文字模型一致，方便共享或对接
    """
    patch_size: int = 16           # 每个 patch 的边长
    in_channels: int = 3           # RGB = 3 通道
    d_model: int = 768             # patch 投影后的维度
    n_heads: int = 12              # 注意力头数
    n_layers: int = 12             # ViT 编码器层数（ViT-Base = 12）
    d_ff: int = 3072               # FFN 中间维度（通常 d_model × 4）
    dropout: float = 0.1           # dropout 比例
    num_classes: int = 1000        # 图像分类类别数（ImageNet = 1000）
    max_grid_size: int = 32        # 2D 位置编码的最大 grid 尺寸（支持 512×512 图 = 32×32 patch）
    window_size: int = 0           # 窗口注意力大小，0=全局注意力（如 Swin 用 7）


# ─────────────────────────────────────────────────────────────────────────────
# 1. PatchEmbedding — 把"图像"切成"token"
# ─────────────────────────────────────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    """
    将 2D 图像转换成 1D token 序列。

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  生活类比: 把一张大照片剪成若干张小照片                                   │
    │                                                                         │
    │  原始照片: 224×224 像素                                                 │
    │  每张小照片: 16×16 像素                                                 │
    │  剪完后: 14 行 × 14 列 = 196 张小照片                                   │
    │                                                                         │
    │  每张小照片的内容（16×16×3=768个数字）被压平，然后投影到 d_model 维。     │
    │  最终得到 196 个"视觉 token"，每个 768 维。                              │
    │                                                                         │
    │  这和 NLP 中的 Token Embedding 完全对应：                                │
    │    NLP:  token id → Embedding Lookup → (seq, d_model)                  │
    │    ViT:   图像像素 → Conv2d 切 patch   → (num_patches, d_model)         │
    └─────────────────────────────────────────────────────────────────────────┘

    为什么用 Conv2d 实现？
      kernel_size=16, stride=16 的卷积，恰好每次滑动 16 像素，
      输出的每个"像素"就对应原图一个 16×16 的 patch。
      这比先切 patch 再分别做 Linear 快得多，且数学等价。

    维度流转:
      输入:  (B, 3, 224, 224)
      Conv:  (B, 768, 14, 14)   ← 768 是输出通道数 = d_model
      Flat:  (B, 768, 196)      ← 把 14×14 压成一维
      Trans: (B, 196, 768)      ← 把 patch 维放前面，和 NLP 序列对齐
    """

    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size

        # Conv2d 一步完成"切 patch + 线性投影"
        self.projection = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Qwen2-VL 风格：支持任意输入尺寸，自动 pad，返回 grid_size。

        输入:  (B, C, H, W)  任意尺寸，例: (2, 3, 300, 400)
        输出:  (x, (H_p, W_p))
            x:    (B, num_patches, d_model)  例: (2, 475, 768)
            H_p:  patch 的行数  例: 19
            W_p:  patch 的列数  例: 25
        """
        B, C, H, W = x.shape

        # 自动 pad 到 patch_size 的整数倍（右和下补零）
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.projection(x)        # (B, d_model, H_p, W_p)
        B, C, H_p, W_p = x.shape
        x = x.flatten(2)              # (B, d_model, H_p * W_p)
        x = x.transpose(1, 2)         # (B, num_patches, d_model)
        return x, (H_p, W_p)


# ─────────────────────────────────────────────────────────────────────────────
# 1.5 PosEmbed2D — Qwen2-VL 风格 2D 位置编码
# ─────────────────────────────────────────────────────────────────────────────
class PosEmbed2D(nn.Module):
    """
    2D 位置编码 —— 让模型知道每个 patch 在原图的"行列坐标"。

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  和 1D 位置编码的区别：                                                  │
    │                                                                         │
    │  1D 位置编码（原版 ViT）:                                                │
    │    每个位置学一个向量，位置 0,1,2,...,196 各不同                         │
    │    问题: 不知道"位置 15"是在第1行第15列还是第2行第1列                  │
    │                                                                         │
    │  2D 位置编码（Qwen2-VL）:                                                │
    │    分别学"行位置"和"列位置"，然后拼接                                  │
    │    patch(i,j) 的位置编码 = [row_embed[i], col_embed[j]]                │
    │    → 模型知道 (i,j) 的空间关系                                         │
    │                                                                         │
    │  例: 14×14 的 patch grid                                               │
    │    patch(0,0):  [row[0], col[0]]  ← 左上角                             │
    │    patch(0,13): [row[0], col[13]] ← 右上角                             │
    │    patch(13,0): [row[13], col[0]] ← 左下角                             │
    │                                                                         │
    │  为什么比 1D 好？                                                        │
    │    1. 尺度不变性: 28×28 的大图和 14×14 的小图共享行/列编码             │
    │    2. 空间关系明确: 模型知道"左边""右边""上面""下面"                   │
    │    3. 支持动态分辨率: 任意大小的图都能加位置编码                         │
    │                                                                         │
    │  实现方式: 可学习的 embedding（不是正弦/RoPE）                           │
    │    和原版 ViT 一样用 nn.Parameter，但分成 row 和 col 两个矩阵          │
    └─────────────────────────────────────────────────────────────────────────┘

    维度流转:
      row_embed: (1, max_grid_size, d_model//2)  例: (1, 32, 384)
      col_embed: (1, max_grid_size, d_model//2)  例: (1, 32, 384)

      输入 x: (B, num_patches, d_model)  例: (2, 196, 768)
      grid: (14, 14)

      row[:14]: (1, 14, 384) → expand → (1, 14, 14, 384)
      col[:14]: (1, 14, 384) → expand → (1, 14, 14, 384)
      cat: (1, 14, 14, 768) → reshape → (1, 196, 768)

      x + pos: (2, 196, 768) + (1, 196, 768) → broadcast
    """

    def __init__(self, d_model: int, max_grid_size: int = 32):
        super().__init__()
        assert d_model % 2 == 0, "d_model 必须是偶数 (row 和 col 各一半)"
        self.d_model = d_model
        self.max_grid_size = max_grid_size

        # 行位置编码 + 列位置编码，各 d_model/2 维
        self.row_embed = nn.Parameter(
            torch.zeros(1, max_grid_size, d_model // 2))
        self.col_embed = nn.Parameter(
            torch.zeros(1, max_grid_size, d_model // 2))

        nn.init.trunc_normal_(self.row_embed, std=0.02)
        nn.init.trunc_normal_(self.col_embed, std=0.02)

    def forward(
        self, x: torch.Tensor, grid_size: tuple[int, int]
    ) -> torch.Tensor:
        """
        x:         (B, num_patches, d_model)
        grid_size: (H_p, W_p)  patch 的行数和列数
        返回:      (B, num_patches, d_model)
        """
        h, w = grid_size
        assert h <= self.max_grid_size and w <= self.max_grid_size, (
            f"grid_size ({h}, {w}) 超过 max_grid_size ({self.max_grid_size})"
        )

        # 取前 h 个行编码和前 w 个列编码
        row = self.row_embed[:, :h, :]   # (1, h, d/2)
        col = self.col_embed[:, :w, :]   # (1, w, d/2)

        # 扩展并组合
        # row: (1, h, 1, d/2) → (1, h, w, d/2)
        # col: (1, 1, w, d/2) → (1, h, w, d/2)
        row = row.unsqueeze(2).expand(-1, -1, w, -1)
        col = col.unsqueeze(1).expand(-1, h, -1, -1)

        # 拼接: (1, h, w, d_model)
        pos = torch.cat([row, col], dim=-1)

        # reshape 成和 x 对齐的序列格式: (1, h*w, d_model)
        pos = pos.reshape(1, h * w, self.d_model)

        return x + pos  # broadcast: (B, N, d) + (1, N, d)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ViTBlock — 一层 ViT（Attention + FFN）
# ─────────────────────────────────────────────────────────────────────────────
class ViTBlock(nn.Module):
    """
    ViT 的 Encoder 块 —— 和文字 Decoder 块的核心区别在这里！

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  DecoderBlock (你的语言模型)     vs      ViTBlock (一层图像编码器)        │
    │  ──────────────────────────              ─────────────────────          │
    │                                                                         │
    │  LayerNorm → Attention + mask        LayerNorm → Attention (无mask)     │
    │       ↓                                    ↓                            │
    │  残差连接                            残差连接                           │
    │       ↓                                    ↓                            │
    │  LayerNorm → FFN                   LayerNorm → FFN                      │
    │       ↓                                    ↓                            │
    │  残差连接                            残差连接                           │
    │                                                                         │
    │  【关键区别: mask】                                                     │
    │                                                                         │
    │  DecoderBlock: mask = 下三角矩阵                                       │
    │    → 每个 token 只能看自己和之前的 token                                │
    │    → 适合自回归生成                                                    │
    │                                                                         │
    │  ViTBlock: mask = None                                                │
    │    → 每个 patch 可以看所有其他 patch（包括未来的）                        │
    │    → 适合全局理解图像                                                  │
    │                                                                         │
    │  类比:                                                                  │
    │    Decoder 像写日记——今天只能回忆过去，不能预知明天。                     │
    │    Encoder 像看照片——一眼扫过去，所有细节同时进入视野。                   │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        window_size: int = 0,
        shift_size: int = 0,
    ):
        super().__init__()
        self.self_attn = ViTMultiHeadAttention(
            d_model, n_heads, dropout,
            window_size=window_size, shift_size=shift_size)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        grid_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        x: (B, seq_len, d_model)   seq_len = num_patches
        grid_size: (H_p, W_p) — patch 的行列数，窗口注意力需要
        返回: (B, seq_len, d_model)
        """
        # --- 自注意力子层 ---
        normed = self.norm1(x)
        # query = key = value
        # grid_size 传给 attention，如果配置了 window_size 就启用窗口注意力
        x = x + self.dropout1(
            self.self_attn(normed, normed, normed, mask=None, grid_size=grid_size)
        )

        # --- 前馈网络子层 ---
        normed = self.norm2(x)
        x = x + self.dropout2(self.ffn(normed))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3. VisionTransformer — 完整的图像理解模型
# ─────────────────────────────────────────────────────────────────────────────
class VisionTransformer(nn.Module):
    """
    完整的 ViT，用于把图像编码成一组向量。

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Qwen2-VL 风格整体流程（以 224×224 图像为例）:                            │
    │                                                                         │
    │  Image (B, 3, 224, 224)                                                 │
    │      │                                                                  │
    │      ▼                                                                  │
    │  PatchEmbedding  →  (B, 196, 768), grid=(14,14)                        │
    │      │                    ↑ 196 个视觉 token                            │
    │      │                    ↑ 支持任意分辨率！                            │
    │      ▼                                                                  │
    │  + PosEmbed2D(grid=(14,14))  →  (B, 196, 768)                          │
    │      │                    ↑ 每个 patch 知道自己在原图的行列位置          │
    │      │                    ↑ 例: patch 15 = 第1行第15列                 │
    │      ▼                                                                  │
    │  Encoder × 12  →  (B, 196, 768)                                        │
    │      │                    ↑ 经过 12 层双向注意力，patch 之间充分交流     │
    │      ▼                                                                  │
    │  LayerNorm                                                              │
    │      │                                                                  │
    │      ▼                                                                  │
    │  全局平均池化  →  (B, 768)                                              │
    │      │                    ↑ 所有 patch 取平均，比 [CLS] 更稳定          │
    │      ▼                                                                  │
    │  Classification Head  →  (B, num_classes)                               │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    和原版 ViT 的关键区别:
    ────────────────────────
    1. 去掉 [CLS] token
       Qwen2-VL 不用 [CLS]，而是直接用所有 patch token
       分类时用"全局平均池化"替代 [CLS]

    2. 2D 位置编码 (PosEmbed2D)
       替代 1D 可学习位置编码
       模型知道每个 patch 的二维空间位置

    3. 动态分辨率
       支持任意输入尺寸，patch 数量随图像大小变化
       不再需要固定 image_size=224
    """

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg

        # Patch Embedding: 任意尺寸图像 → token 序列
        self.patch_embed = PatchEmbedding(
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            d_model=cfg.d_model,
        )

        # 2D 位置编码: 替代 1D 可学习位置编码 + [CLS]
        self.pos_embed = PosEmbed2D(
            d_model=cfg.d_model,
            max_grid_size=cfg.max_grid_size,
        )
        self.pos_drop = nn.Dropout(cfg.dropout)

        # Encoder Blocks
        # Encoder Blocks —— 偶数层正常窗口，奇数层 Shifted Window
        # Shifted Window 让相邻窗口的 patch 能交互，解决窗口注意力的"盲区"
        shift_size = cfg.window_size // 2 if cfg.window_size > 0 else 0
        self.blocks = nn.ModuleList([
            ViTBlock(
                cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
                window_size=cfg.window_size,
                shift_size=shift_size if i % 2 == 1 else 0,  # 奇数层偏移
            )
            for i in range(cfg.n_layers)
        ])

        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        提取图像特征（不含分类头），供多模态模型复用。

        输入:  (B, C, H, W)  任意尺寸
        输出:  (x, grid_size)
            x:         (B, num_patches, d_model)  所有 patch token
            grid_size: (H_p, W_p)  patch 的行列数
        """
        # 1) Patch Embedding: 任意尺寸 → patch tokens
        x, grid_size = self.patch_embed(pixel_values)
        # x: (B, num_patches, d_model)

        # 2) 加 2D 位置编码
        # 每个 patch 知道自己在原图的行列位置
        x = self.pos_embed(x, grid_size)
        x = self.pos_drop(x)

        # 3) 通过 N 层 Encoder（双向自注意力）
        # 把 grid_size 传给每一层，窗口注意力需要知道 patch 的 2D 布局
        for block in self.blocks:
            x = block(x, grid_size=grid_size)

        x = self.norm(x)
        return x, grid_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        图像分类 forward。

        输入:  (B, C, H, W)
        输出:  (B, num_classes) — 未经 softmax 的 logits
        """
        x, _ = self.forward_features(pixel_values)  # (B, N, d_model)
        # 全局平均池化: 所有 patch 取平均，替代 [CLS]
        x = x.mean(dim=1)   # (B, d_model)
        logits = self.head(x)
        # 返回的这个logits是什么？
        # 它是模型对每个类别的预测分数，未经 softmax 转换成概率。维度是 (B, num_classes)，
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# 4. ImageTextProjector — "翻译官"：把图像语言翻译成文字语言
# ─────────────────────────────────────────────────────────────────────────────
class ImageTextProjector(nn.Module):
    """
    将 ViT 输出的图像特征投影到文本 Decoder 的嵌入空间。

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  为什么需要 Projector？                                                 │
    │                                                                         │
    │  想象两个说不同语言的人：                                               │
    │    ViT 说的是"图像语"（视觉特征空间）                                   │
    │    Decoder 说的是"文字语"（词嵌入空间）                                  │
    │                                                                         │
    │  Projector 就是翻译官，把"图像语"翻译成"文字语"，                       │
    │  这样两者才能在同一张桌子上开会（同一个序列里做注意力）。                 │
    │                                                                         │
    │  为什么不是简单的 Linear？                                              │
    │    两层 MLP + GELU 的表达能力更强，能学到更复杂的映射关系。              │
    │    就像专业翻译不只是逐字翻，还要理解语境。                              │
    └─────────────────────────────────────────────────────────────────────────┘

    维度流转:
      输入:  (B, num_image_tokens, vit_d_model)  例: (2, 197, 768)
      输出:  (B, num_image_tokens, llm_d_model)  例: (2, 197, 768)

    如果 vit_d_model ≠ llm_d_model（比如 ViT 用 1024，LLM 用 768），
    Projector 会自动做维度对齐。
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


# ─────────────────────────────────────────────────────────────────────────────
# 5. VisionLanguageModel — 图文多模态：看图说话
# ─────────────────────────────────────────────────────────────────────────────
class VisionLanguageModel(nn.Module):
    """
    图文多模态模型 —— 核心问题"图像和文字怎么做注意力"的答案在这里！

    ═══════════════════════════════════════════════════════════════════════════
    【架构总览：拼接 + 统一自注意力】
    ═══════════════════════════════════════════════════════════════════════════

    本模型采用"拼接序列 + 统一 causal attention"的方案（LLaVA 风格），
    而不是单独的 cross-attention（CLIP 风格）。

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  方案对比:                                                              │
    │                                                                         │
    │  本模型用的方案（拼接 + 自注意力）:                                     │
    │    [img0][img1]...[img195][txt0][txt1][txt2]...                        │
    │      ↓ 一起过 causal mask（图像全可见，文字因果可见）                   │
    │    → 所有 token 互相 attention                                         │
    │                                                                         │
    │  另一种方案（Cross-Attention，如 Flamingo）:                           │
    │    图像 → ViT → 固定特征                                               │
    │    文字 → Decoder 的每一层都额外做一次 cross-attention 到图像特征       │
    │                                                                         │
    │  为什么选拼接方案？                                                     │
    │    1. 简单: 复用已有的 Decoder，不需要改 attention 结构                 │
    │    2. 灵活: 图像 token 和文字 token 平等参与，交互更充分                │
    │    3. 高效: 一次前向传播搞定，不需要每层的 cross-attention               │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    ═══════════════════════════════════════════════════════════════════════════
    【Causal Mask 的具体构造】
    ═══════════════════════════════════════════════════════════════════════════

    这是图文多模态最核心的代码细节：

    假设 num_image_tokens = 196（14×14 patches，无 [CLS]）
    假设 text_len = 10
    总序列长度 = 206

    mask 是一个 207×207 的矩阵，分为四个区域：

                    图像(197)      文字(10)
                  ┌──────────┬──────────┐
        图像(197) │   全1    │    全0   │   ← 图像看图像: OK；图像看文字: 不需要
                  ├──────────┼──────────┤
        文字(10)  │   全1    │  下三角  │   ← 文字看图像: OK；文字看文字: 因果
                  └──────────┴──────────┘

    代码实现:
      1. 先构造一个标准的下三角 causal mask (207, 207)
      2. 把"图像看文字"的区域（前 197 行，后 10 列）设为 0
      3. 把"图像看图像"的区域（前 197 行，前 197 列）设为 1

    但实际上，当前的实现是用了一个"简化版"mask：
      对整个序列（图像+文字）统一用下三角 causal mask。

    这意味着：
      • 图像 patch 之间: 由于是序列开头，下三角 = 全可见 ✓
      • 文字看图像: 图像在前，文字在后，下三角 = 可见 ✓
      • 文字看文字: 下三角 = 因果 ✓
      • 图像看文字: 图像在前，文字在后，下三角 = 不可见 ✓

    完美！一个普通的下三角 mask 恰好实现了我们想要的所有约束，
    不需要额外的复杂逻辑。这就是为什么"拼接 + causal mask"方案如此优雅。
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
        self.decoder = decoder
        self.output_proj = output_proj
        self.projector = projector
        self.pad_token_id = pad_token_id

        # 多模态训练中通常冻结 ViT，只训练 Projector 和 Decoder
        # 原因: ViT 已经在大规模图像数据上预训练过，特征提取能力已很强
        #       我们只需要学"怎么把图像特征和文字对齐"
        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False

    def _make_causal_mask(
        self,
        seq_len: int,
        num_image_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        构建图文混合的因果掩码。

        纯下三角 mask 不能让图像 patch 之间全可见（img0 只能看自己）。
        正确的掩码分四块：

              ┌─────────────┬─────────────┐
              │  图像部分   │  文字部分   │
        ┌─────┼─────────────┼─────────────┤
        │图像 │   全 1      │   全 0      │  ← 图像互相可见，不看文字
        ├─────┼─────────────┼─────────────┤
        │文字 │   全 1      │   下三角    │  ← 文字看图像+因果看自己
        └─────┴─────────────┴─────────────┘

        参数:
            seq_len:          总序列长度（图像token数 + 文字token数）
            num_image_tokens: 图像token数量
            device:           目标设备

        返回: (1, 1, seq_len, seq_len) — 可广播到 (B, n_heads, seq_len, seq_len)
        """
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

        # 图像部分：所有图像 patch 互相可见（左上全1方块）
        mask[:num_image_tokens, :num_image_tokens] = True

        # 文字部分：文字看图像全可见 + 文字之间因果可见
        text_start = num_image_tokens
        if text_start < seq_len:
            # 文字看所有图像（左下全1矩形）
            mask[text_start:, :num_image_tokens] = True
            # 文字之间因果（右下下三角）
            text_len = seq_len - text_start
            causal = torch.tril(torch.ones(
                text_len, text_len, device=device, dtype=torch.bool))
            mask[text_start:, text_start:] = causal

        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        图文多模态 forward。

        参数:
            pixel_values: (B, C, H, W) — 输入图像
            input_ids:    (B, T)       — 输入文本的 token id

        返回:
            logits: (B, N_img + T, vocab_size)

        维度流转示例 (B=2, 图像=224×224, 文本=10个token):

          pixel_values:    (2, 3, 224, 224)
              ↓
          vit.forward_features:  (2, 197, 768)     ← 图像特征（含[CLS]）
              ↓
          projector:       (2, 197, 768)           ← 投影到文本空间
              ↓
          text_embeds:     (2, 10, 768)            ← "这是猫"的 embedding
              ↓
          cat:             (2, 207, 768)           ← [图像197 + 文字10]
              ↓
          pos_encoding:    (2, 207, 768)           ← 加位置信息
              ↓
          causal_mask:     (1, 1, 207, 207)        ← 下三角mask
              ↓
          Decoder Blocks:  (2, 207, 768)           ← 注意力计算
              ↓
          output_proj:     (2, 207, vocab_size)    ← 预测下一个词

        训练时 loss 只算文字部分：
          模型已经看到了图像+前面的文字，要预测下一个文字。
          图像部分的输出不需要计算 loss（它们不是"词"）。
        """
        # ─── 图像编码（只做一次）───
        # Qwen2-VL 风格: 支持任意分辨率，返回所有 patch token（无 [CLS]）
        # (B, num_patches, vit_d_model)
        image_features, grid_size = self.vit.forward_features(pixel_values)
        # 投影到文本空间: (B, num_patches, llm_d_model)
        image_embeds = self.projector(image_features)

        # ─── 文本嵌入 ───
        # 复用 Decoder 的 token_embedding
        d_model = self.decoder.d_model
        text_embeds = self.decoder.token_embedding(
            input_ids) * (d_model ** 0.5)

        # ─── 拼接：[图像 tokens | 文本 tokens] ───
        # 这是多模态的核心：图像和文字在同一个序列里
        # 它们通过自注意力自然交互——文字 token 会 attend 到图像 token
        combined = torch.cat([image_embeds, text_embeds], dim=1)

        total_len = combined.size(1)

        # ─── 位置编码 ───
        combined = self.decoder.pos_encoding(combined)

        # ─── 因果掩码 ───
        # 下三角 mask 恰好实现所有约束:
        #   - 图像 patch 之间: 全可见（在序列开头）
        #   - 文字看图像: 可见（图像在文字前面）
        #   - 文字看文字: 因果（下三角）
        #   - 图像看文字: 不可见（不需要）
        num_image_tokens = image_embeds.size(1)
        causal_mask = self._make_causal_mask(
            total_len, num_image_tokens, combined.device)

        # ─── 通过 Decoder Blocks ───
        x = combined
        for layer in self.decoder.layers:
            x = layer(x, tgt_mask=causal_mask)
        x = self.decoder.norm(x)

        # ─── 输出投影 ───
        logits = self.output_proj(x)  # (B, num_patches + text_len, vocab_size)
        return logits

    def get_num_image_tokens(self, pixel_values: torch.Tensor | None = None) -> int:
        """
        返回图像编码后产生的 token 数量（不含 [CLS]，纯 patch 数）。

        如果提供了 pixel_values，动态计算；否则返回默认值 196。
        """
        if pixel_values is not None:
            _, grid_size = self.vit.forward_features(pixel_values)
            return grid_size[0] * grid_size[1]
        # 默认 224×224, patch=16 → 14×14 = 196
        return 196

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
        图文多模态生成：给定图像和提示，自回归生成回答。

        生成过程:
          1. 图像编码一次（固定不变）
          2. 文本逐 token 生成，每步都把已生成的文本和图像重新拼接
          3. 模型参考图像+已生成文字，预测下一个词

        例:
          图像: [一只猫的照片]
          提示: "这张图里有什么？"
          生成: "图" → "里" → "有" → "一" → "只" → "猫" → "。" → <eos>

        参数:
            pixel_values:   (B, C, H, W) — 输入图像
            input_ids:      (B, T) — 提示文本
            max_new_tokens: 最大生成 token 数
            eos_token_id:   结束符 id
            temperature:    采样温度（越高越随机）
            top_k:          top-k 采样

        返回:
            generated_ids: (B, T + new_tokens) — 包含原始 prompt 和生成内容
        """
        self.eval()

        # 图像编码（只需做一次，因为图不变）
        image_features, _ = self.vit.forward_features(pixel_values)
        image_embeds = self.projector(image_features)

        generated = input_ids.clone()
        finished = torch.zeros(input_ids.size(
            0), dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            # 文本嵌入
            d_model = self.decoder.d_model
            text_embeds = self.decoder.token_embedding(
                generated) * (d_model ** 0.5)

            # 拼接图像 + 当前已生成的文本
            combined = torch.cat([image_embeds, text_embeds], dim=1)
            total_len = combined.size(1)
            combined = self.decoder.pos_encoding(combined)

            num_image_tokens = image_embeds.size(1)
            causal_mask = self._make_causal_mask(
                total_len, num_image_tokens, combined.device)

            # 前向传播
            x = combined
            for layer in self.decoder.layers:
                x = layer(x, tgt_mask=causal_mask)
            x = self.decoder.norm(x)

            # 只取最后一个位置的 logits（预测下一个词）
            next_logits = self.output_proj(
                x[:, -1, :]) / max(temperature, 1e-5)

            if top_k > 0:
                top_k_vals, _ = torch.topk(
                    next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < top_k_vals[:, -1:]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = next_token.masked_fill(
                finished.unsqueeze(1), self.pad_token_id)

            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == eos_token_id)
            if finished.all():
                break

        return generated


# ─────────────────────────────────────────────────────────────────────────────
# 快捷构建函数
# ─────────────────────────────────────────────────────────────────────────────
def build_vision_language_model(
    vit_cfg: ViTConfig,
    llm_transformer: nn.Module,
    freeze_vit: bool = True,
) -> VisionLanguageModel:
    """
    快捷构建 VisionLanguageModel —— Qwen2-VL 风格。

    参数:
        vit_cfg:         ViT 配置（不再需要 image_size）
        llm_transformer: 已有的 Transformer 实例
        freeze_vit:      是否冻结 ViT 参数（推荐 True）

    使用示例:
        from config import ModelConfig, ViTConfig
        from model.transformer import Transformer

        llm = Transformer(ModelConfig(d_model=768))
        vit_cfg = ViTConfig(d_model=768, num_classes=0, max_grid_size=32)
        vlm = build_vision_language_model(vit_cfg, llm, freeze_vit=True)

        # 支持任意分辨率！
        images = torch.randn(2, 3, 224, 224)   # 224×224 → 196 tokens
        large_images = torch.randn(2, 3, 448, 448)  # 448×448 → 784 tokens
        text_ids = torch.randint(0, 16000, (2, 32))
        logits = vlm(images, text_ids)  # (2, 228, vocab_size)
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
