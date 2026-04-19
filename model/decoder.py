"""
Decoder — 堆叠多个 DecoderBlock, 逐层提取越来越抽象的特征
============================================================

这是 Decoder-Only 架构 (GPT/Llama/Qwen 风格), 只有自注意力, 没有交叉注意力。

默认配置: d_model=768, n_heads=12, n_layers=10, d_ff=3072, max_seq_len=256
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward
from .moe_feedforward import MoEFeedForward
from .positional import SinusoidalPositionalEncoding


class DecoderBlock(nn.Module):
    """
    单个 Transformer 解码器块 — 注意力 + FFN, 各带残差连接和 LayerNorm。

    ┌──────────────────────────────────────────────────────────────────┐
    │  完整流程 (B=2, seq=5, d_model=768):                            │
    │                                                                  │
    │  输入 x: (2, 5, 768)                                            │
    │      │                                                           │
    │      ├──────────────────────(+)──→ x_after_attn: (2, 5, 768)    │
    │      │                           ↑                               │
    │      │   norm1(x) → (2,5,768)    │                               │
    │      │      │                     │                               │
    │      │      ▼ Masked Self-Attn    │                               │
    │      │   (2,5,768) → (2,5,768)───┘                               │
    │      │      (q=k=v=normed, mask=因果掩码)                       │
    │      │      + dropout                                            │
    │      │                                                           │
    │      ├──────────────────────(+)──→ 输出: (2, 5, 768)            │
    │      │                           ↑                               │
    │      │ norm2(x_after_attn)        │                               │
    │      │      │                     │                               │
    │      │      ▼ FFN / MoE FFN      │                               │
    │      │   (2,5,768)→(2,5,3072)    │                               │
    │      │   →(2,5,768)──────────────┘                               │
    │      │      + dropout                                            │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────┐            │
    │  │  两个关键设计:                                    │            │
    │  │                                                    │            │
    │  │  Pre-Norm (先归一化再计算):                       │            │
    │  │    x = x + Attn(norm(x))                          │            │
    │  │    而不是 Post-Norm: x = norm(x + Attn(x))        │            │
    │  │    Pre-Norm 训练更稳定, 梯度流动更顺畅           │            │
    │  │                                                    │            │
    │  │  残差连接 (+):                                     │            │
    │  │    把输入直接加到输出上                            │            │
    │  │    好处: 梯度可以直接跳过这一层回传,              │            │
    │  │          即使这一层没学到东西(输出≈0),            │            │
    │  │          梯度也不会消失, 训练深层网络不会崩      │            │
    │  └──────────────────────────────────────────────────┘            │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────┐            │
    │  │  为什么用 LayerNorm 而不是 BatchNorm?             │            │
    │  │                                                    │            │
    │  │  BatchNorm: 对整个 batch 统计均值方差             │            │
    │  │    → NLP 的 batch 通常很小 (4~16), 统计不稳定    │            │
    │  │    → 序列长度不同时, padding 影响统计            │            │
    │  │                                                    │            │
    │  │  LayerNorm: 对单个 token 的 768 维统计            │            │
    │  │    → 不依赖 batch 大小 ✓                          │            │
    │  │    → 每个 token 独立归一化 ✓                      │            │
    │  └──────────────────────────────────────────────────┘            │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_moe: bool = False,
        moe_num_experts: int = 4,
        use_rope: bool = False,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(
            d_model, n_heads, dropout,
            use_rope=use_rope, max_seq_len=max_seq_len,
        )
        self.ffn = (
            MoEFeedForward(d_model, d_ff, moe_num_experts, dropout)
            if use_moe
            else PositionwiseFeedForward(d_model, d_ff, dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(B, seq, d_model) → (B, seq, d_model)"""
        # 自注意力: q=k=v=归一化后的x, 只看过去 (因果掩码)
        normed = self.norm1(x)
        x = x + self.dropout1(self.masked_self_attn(normed, normed, normed, tgt_mask))
        # FFN: 逐位置做非线性变换
        normed = self.norm2(x)
        x = x + self.dropout2(self.ffn(normed))
        return x


class Decoder(nn.Module):
    """
    Decoder 堆叠 — Embedding + 位置编码 + N 层 DecoderBlock + 最终 LayerNorm。

    ┌──────────────────────────────────────────────────────────────────┐
    │  完整流程 (B=2, seq=5, 默认配置):                               │
    │                                                                  │
    │  输入: token_ids (2, 5) — 5 个 token 的 id                     │
    │      │                                                           │
    │      ▼ Embedding(68, 768): 查表, 把 id 变成向量                │
    │  (2, 5, 768)                                                    │
    │      │                                                           │
    │      ▼ × √768 ≈ 27.7: 缩放 embedding                           │
    │  (2, 5, 768) — 为什么要缩放?                                    │
    │    embedding 初始化值很小 (~1/√768 ≈ 0.036)                    │
    │    乘以 √768 后值域接近 1, 和位置编码量级匹配                   │
    │    不缩放的话位置编码会盖过语义信息                             │
    │      │                                                           │
    │      ▼ 位置编码 (二选一):                                       │
    │                                                                  │
    │    use_rope=False:                                               │
    │      x = x + SinusoidalPE(pos) → (2, 5, 768)                   │
    │      在入口加一次位置编码, 之后不再注入                         │
    │                                                                  │
    │    use_rope=True:                                                │
    │      x = dropout(x)         → (2, 5, 768)                      │
    │      不加位置编码, 每层 attention 内部旋转 Q,K                  │
    │                                                                  │
    │      │                                                           │
    │      ▼ DecoderBlock × 10: 逐层提取特征                         │
    │    每层都是 (2, 5, 768) → (2, 5, 768), 形状不变               │
    │    但每个 token 的表示越来越"懂"上下文                          │
    │                                                                  │
    │      │                                                           │
    │      ▼ LayerNorm(768): 最终归一化                               │
    │    (2, 5, 768) — 稳定输出, 方便后续投影到词表                  │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        use_moe: bool = False,
        moe_num_experts: int = 4,
        use_rope: bool = False,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.d_model = d_model
        self.use_rope = use_rope

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        if self.use_rope:
            self.embed_dropout = nn.Dropout(dropout)
            self.pos_encoding = None
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
            self.embed_dropout = None

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model, n_heads, d_ff, dropout,
                    use_moe=use_moe, moe_num_experts=moe_num_experts,
                    use_rope=use_rope, max_seq_len=max_seq_len,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        token_ids: (B, seq) → hidden: (B, seq, d_model)
        """
        x = self.token_embedding(tgt) * (self.d_model**0.5)

        if self.use_rope:
            x = self.embed_dropout(x)
        else:
            x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, tgt_mask)
        return self.norm(x)
