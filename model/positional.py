"""
位置编码 — 让模型知道"每个 token 在第几个位置"
================================================

Transformer 的注意力本身没有顺序概念 (q·k 只看内容不看位置),
所以必须额外注入位置信息。本文件实现了两种方式:

  1. RoPE (旋转位置编码) — 在每层 attention 里旋转 Q 和 K, 编码相对位置
  2. 正弦位置编码 — 在输入时加一次, 编码绝对位置

默认配置: d_model=768, n_heads=12, d_k=64
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE) — Llama/Qwen/GPT-4 等主流模型的标准配置。

    ┌──────────────────────────────────────────────────────────────────┐
    │  核心思路: 用旋转角度编码位置                                    │
    │                                                                  │
    │  把 64 维的向量, 每两个维度看成 2D 平面上的一个点:              │
    │    (q₀, q₁) = 一个 2D 点                                       │
    │    (q₂, q₃) = 另一个 2D 点                                     │
    │    ...共 32 对 (d_k/2=32)                                       │
    │                                                                  │
    │  对位置 m 的 token, 每对维度旋转一个角度:                       │
    │    [q'₀]   [cos(m·θ)  -sin(m·θ)] [q₀]                         │
    │    [q'₁] = [sin(m·θ)   cos(m·θ)] [q₁]                         │
    │                                                                  │
    │  不同对用不同频率 θ:                                            │
    │    第0对: θ₀ = 1/10000^(0/64) = 1.0       旋转最快             │
    │    第1对: θ₁ = 1/10000^(2/64) = 0.749                          │
    │    第2对: θ₂ = 1/10000^(4/64) = 0.562                          │
    │    ...                                                           │
    │    第31对: θ₃₁ = 1/10000^(62/64) = 0.0001  旋转最慢           │
    │                                                                  │
    │  ┌────────────────────────────────────────────────────┐          │
    │  │  为什么频率不同?                                    │          │
    │  │                                                    │          │
    │  │  高频 (θ大): 位置差1就转很多 → 区分相邻token      │          │
    │  │  低频 (θ小): 位置差1几乎没转 → 能感受远距离关系   │          │
    │  │  → 同时编码近距离和远距离的位置关系                │          │
    │  └────────────────────────────────────────────────────┘          │
    │                                                                  │
    │  数值示例 — d_k=64, 3个 token, 只看第0对维度:                   │
    │                                                                  │
    │  θ₀ = 1.0                                                       │
    │                                                                  │
    │  位置0: 旋转 0×1.0 =   0 弧度 → 不旋转                        │
    │  位置1: 旋转 1×1.0 =   1 弧度 → 转约 57°                      │
    │  位置2: 旋转 2×1.0 =   2 弧度 → 转约 115°                     │
    │                                                                  │
    │  关键性质: q_m · k_n 只取决于 m-n (相对距离)                   │
    │    位置2的q · 位置0的k = 旋转2步后和旋转0步的点积              │
    │    位置2的q · 位置1的k = 旋转2步后和旋转1步的点积              │
    │    差都是1步, 点积值相同 → 编码的是"距离1", 不是"位置2和位置1" │
    │                                                                  │
    │  对比正弦编码:                                                   │
    │    正弦编码: "我在第 5 个位置" (绝对位置)                       │
    │    RoPE:     "我离你 3 个位置" (相对位置)                       │
    │    → 更符合语言本质 ("猫吃鱼"中关系不因出现位置而变)           │
    └──────────────────────────────────────────────────────────────────┘

    参数:
        d_k: 每个注意力头的维度 (64)
        max_len: 预计算的最大序列长度
        base: 频率基数, 默认 10000
    """

    def __init__(self, d_k: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        assert d_k % 2 == 0, "d_k 必须是偶数 (每两个维度一组做旋转)"
        self.d_k = d_k

        # 预计算频率: θ_j = 1 / base^(2j/d_k), j = 0,1,...,d_k/2-1
        # 形状: (d_k/2,) = (32,)
        # d_k=64: [1.0, 0.749, 0.562, ..., 0.0001]
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq)

        # 预计算所有位置的 cos 和 sin, 避免每次 forward 重复算
        self._build_cache(max_len)

    def _build_cache(self, max_len: int) -> None:
        """
        预计算 cos(m·θ) 和 sin(m·θ) 的缓存表。

        ┌──────────────────────────────────────────────────────────────┐
        │  数值示例 (d_k=64, max_len=5):                               │
        │                                                              │
        │  positions: [0, 1, 2, 3, 4]  形状 (5,1)                    │
        │  inv_freq:  [1.0, 0.749, ...] 形状 (1, 32)                 │
        │                                                              │
        │  freqs = positions × inv_freq: (5, 32)                      │
        │    每行是一个位置的 32 个频率值                              │
        │    位置0: [0×1.0, 0×0.749, ...] = [0, 0, ...]             │
        │    位置1: [1×1.0, 1×0.749, ...] = [1, 0.749, ...]         │
        │    位置2: [2×1.0, 2×0.749, ...] = [2, 1.498, ...]         │
        │                                                              │
        │  cat([freqs, freqs]): (5, 64)                               │
        │    为什么重复? 因为每对维度需要同样的 cos/sin              │
        │    (q₀,q₁) 用 cos[0] 和 sin[0],                            │
        │    (q₂,q₃) 也用 cos[0] 和 sin[0] — 和前半一样             │
        │    这样 x * cos + rotate_half(x) * sin 就是 2D 旋转        │
        │                                                              │
        │  cos_cached: (5, 64) — 每个位置每个维度的 cos 值           │
        │  sin_cached: (5, 64) — 每个位置每个维度的 sin 值           │
        └──────────────────────────────────────────────────────────────┘
        """
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        freqs = positions * self.inv_freq.unsqueeze(0)
        freqs = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        把向量的前半和后半交换并取负, 配合 cos/sin 实现 2D 旋转。

        ┌──────────────────────────────────────────────────────────────┐
        │  数值示例 (d_k=8):                                           │
        │                                                              │
        │  x = [a, b, c, d, e, f, g, h]                               │
        │  前半: [a, b, c, d]    后半: [e, f, g, h]                   │
        │  返回: [-e, -f, -g, -h, a, b, c, d]                         │
        │                                                              │
        │  这样 x * cos + rotate_half(x) * sin 就等于:               │
        │    a*cos₀ + (-e)*sin₀ = a*cos(θ) - e*sin(θ) ← q'₀        │
        │    b*cos₀ + (-f)*sin₀ = b*cos(θ) - f*sin(θ) ← q'₁        │
        │    ...                                                      │
        │    e*cos₀ + a*sin₀     = e*cos(θ) + a*sin(θ) ← q'₄       │
        │                                                              │
        │  等价于每对 (q₀,q₁) 做 2D 旋转, 只是用了前后半的方式      │
        │  比逐对循环更高效 (向量化)                                   │
        └──────────────────────────────────────────────────────────────┘
        """
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对 Q 和 K 应用旋转, V 不动。

        ┌──────────────────────────────────────────────────────────────┐
        │  数值示例: B=2, n_heads=12, seq=5, d_k=64                   │
        │                                                              │
        │  q: (2, 12, 5, 64) — 5 个 token, 每个 64 维                │
        │  k: (2, 12, 5, 64)                                          │
        │                                                              │
        │  取 cos/sin:                                                 │
        │    cos_cached[0:5]: (5, 64) — 5 个位置的 cos 值             │
        │    unsqueeze 两次: (1, 1, 5, 64) — 广播到所有 batch/head   │
        │                                                              │
        │  旋转:                                                       │
        │    q_rotated = q * cos_q + rotate_half(q) * sin_q           │
        │    → 逐元素乘, 形状不变: (2, 12, 5, 64)                     │
        │                                                              │
        │  为什么 V 不旋转?                                            │
        │    注意力公式: attn = softmax(q'·k'^T / √d) × v            │
        │    位置信息已经通过 q'·k'^T 的点积注入了                    │
        │    v 只是携带内容信息, 不需要位置                            │
        │                                                              │
        │  offset 参数: 推理时增量解码用                               │
        │    训练时 offset=0, 从位置0开始                              │
        │    推理时 offset=已生成token数, 让新token从正确位置旋转     │
        └──────────────────────────────────────────────────────────────┘
        """
        seq_len_q = q.size(2)
        seq_len_k = k.size(2)

        cos_q = self.cos_cached[offset : offset + seq_len_q].unsqueeze(0).unsqueeze(0)
        sin_q = self.sin_cached[offset : offset + seq_len_q].unsqueeze(0).unsqueeze(0)

        cos_k = self.cos_cached[offset : offset + seq_len_k].unsqueeze(0).unsqueeze(0)
        sin_k = self.sin_cached[offset : offset + seq_len_k].unsqueeze(0).unsqueeze(0)

        q_rotated = q * cos_q + self._rotate_half(q) * sin_q
        k_rotated = k * cos_k + self._rotate_half(k) * sin_k

        return q_rotated, k_rotated


class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦位置编码 (原始 Transformer 方式) — 在输入时加一次。

    ┌──────────────────────────────────────────────────────────────────┐
    │  公式:                                                          │
    │    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))               │
    │    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))               │
    │                                                                  │
    │  数值示例 (d_model=8, 3个位置):                                 │
    │                                                                  │
    │  位置0: [sin(0), cos(0), sin(0), cos(0), sin(0), cos(0), ...]  │
    │       = [0, 1, 0, 1, 0, 1, ...]                                │
    │                                                                  │
    │  位置1: [sin(1/1), cos(1/1), sin(1/21.5), cos(1/21.5), ...]   │
    │       = [0.84, 0.54, 0.05, 1.0, ...]                           │
    │                                                                  │
    │  位置2: [sin(2/1), cos(2/1), sin(2/21.5), cos(2/21.5), ...]   │
    │       = [0.91, -0.42, 0.09, 1.0, ...]                          │
    │                                                                  │
    │  使用方式: x' = x + PE[pos] — 直接加到 token embedding 上      │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────┐            │
    │  │  RoPE vs 正弦编码 的根本区别:                     │            │
    │  │                                                    │            │
    │  │  正弦编码:  x' = x + PE[pos]                      │            │
    │  │    在输入时加一次, 深层会逐渐稀释                 │            │
    │  │    编码绝对位置: "我在第5个位置"                   │            │
    │  │                                                    │            │
    │  │  RoPE:     q' = rotate(q, θ_pos)                  │            │
    │  │    每层都旋转 Q,K, 位置信息永不稀释              │            │
    │  │    编码相对位置: "我离你3个位置"                   │            │
    │  │                                                    │            │
    │  │  两者频率公式一样, 只是注入方式不同:              │            │
    │  │    正弦: 加法 → 改变向量值                        │            │
    │  │    RoPE: 旋转 → 改变向量方向, 不改变大小         │            │
    │  └──────────────────────────────────────────────────┘            │
    └──────────────────────────────────────────────────────────────────┘

    参数:
        d_model: 模型维度 (768)
        max_len: 预计算的最大序列长度
        dropout: dropout 比例
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term: (d_model/2,) — 和 RoPE 的 inv_freq 是同一个公式
        # 只是这里算的是 e^{-2i·ln(10000)/d_model} = 1/10000^{2i/d_model}
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        # 偶数维用 sin, 奇数维用 cos — 一对(sin,cos)编码同一个频率
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (1, max_len, d_model) — 加 batch 维, 方便后面广播
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        ┌──────────────────────────────────────────────────────────────┐
        │  x: (B, seq_len, d_model) = (2, 5, 768)                    │
        │  pe[:, offset:offset+5]: (1, 5, 768) — 取对应位置的编码    │
        │  x + pe: (2, 5, 768) — 广播相加, 每个 token 加上位置信息   │
        │                                                              │
        │  offset 参数: KV-Cache 增量解码用                            │
        │    训练/首次: offset=0, 从位置0开始 (行为不变)             │
        │    增量解码: offset=已生成token数, 新token从正确位置开始    │
        │                                                              │
        │  为什么加法能编码位置?                                       │
        │    两个不同位置的 token, 即使内容相同, 加了不同的 PE 后     │
        │    在 768 维空间里就不同了, 注意力计算时能区分              │
        └──────────────────────────────────────────────────────────────┘
        """
        x = x + self.pe[:, offset : offset + x.size(1)]
        return self.dropout(x)
