"""
多头注意力机制 — Transformer 的核心
=====================================

一句话: 让每个 token 去"问"其他所有 token, 按相关性加权汇总信息。

默认配置: d_model=768, n_heads=12, d_k=64, d_ff=3072
"""

from __future__ import annotations

import math
from typing import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import RotaryPositionEmbedding

# KV-Cache 类型: 每层缓存一对 (K, V) 张量, 初始为 None
# K/V 形状: (B, n_heads, seq_len_so_far, d_k)
KVCache: TypeAlias = list[tuple[torch.Tensor, torch.Tensor] | None]


class MultiHeadAttention(nn.Module):
    """
    多头注意力 — 把 d_model 拆成 n_heads 个头, 每个头独立算注意力, 再拼回来。

    ┌──────────────────────────────────────────────────────────────────┐
    │  为什么拆多头?                                                   │
    │                                                                  │
    │  1 个头: 768 维的 Q·K 点积 → 只能学 1 种注意力模式             │
    │  12 个头: 每个 64 维 → 可以学 12 种不同的注意力模式            │
    │    头1 可能关注语法关系 (主语→谓语)                             │
    │    头2 可能关注指代关系 (他→小明)                              │
    │    头3 可能关注相邻修饰 (漂亮→花)                              │
    │    ...                                                           │
    │  最后拼回来, 信息更丰富                                         │
    └──────────────────────────────────────────────────────────────────┘

    参数:
        d_model: 输入特征维度 (768)
        n_heads: 注意力头数 (12)
        dropout: dropout 比例
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 use_rope: bool = False, max_seq_len: int = 8192):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 768/12 = 64, 每个头的维度
        self.use_rope = use_rope

        # 四个线性变换: 把 768 维映射到 768 维
        # w_q: 把输入变成"查询" — "我在找什么信息?"
        # w_k: 把输入变成"键"   — "我能提供什么信息?"
        # w_v: 把输入变成"值"   — "我的实际内容是什么?"
        # w_o: 把多头拼接的结果投影回 768 维
        self.w_q = nn.Linear(d_model, d_model)  # (768 → 768)
        self.w_k = nn.Linear(d_model, d_model)  # (768 → 768)
        self.w_v = nn.Linear(d_model, d_model)  # (768 → 768)
        self.w_o = nn.Linear(d_model, d_model)  # (768 → 768)
        self.dropout = nn.Dropout(dropout)

        if self.use_rope:
            self.rope = RotaryPositionEmbedding(self.d_k, max_len=max_seq_len)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        缩放点积注意力 — 注意力的核心计算。

        ┌──────────────────────────────────────────────────────────────────┐
        │  数值示例: B=2, n_heads=12, seq_len=5, d_k=64                  │
        │                                                                  │
        │  输入:                                                           │
        │    q: (2, 12, 5, 64) — 2个样本, 12个头, 每个5个token, 64维     │
        │    k: (2, 12, 5, 64)                                             │
        │    v: (2, 12, 5, 64)                                             │
        │                                                                  │
        │  ① 计算注意力分数: q @ k^T / √d_k                               │
        │                                                                  │
        │    q:     (2, 12, 5, 64)                                         │
        │    k^T:   (2, 12, 64, 5)   ← 最后两维转置                      │
        │    q @ k^T: (2, 12, 5, 5) ← 每个 token 对每个 token 的分数     │
        │                                                                  │
        │    为什么除以 √64 = 8?                                           │
        │    d_k 越大, 点积的数值越大 (64 个乘积相加)                     │
        │    不除的话 softmax 会饱和 (全部概率集中到 1 个位置)             │
        │    除以 √d_k 让分数保持在合理范围, 梯度更健康                   │
        │                                                                  │
        │  ② 应用 mask (因果掩码)                                         │
        │                                                                  │
        │    假设 seq_len=5, 因果掩码 (下三角):                           │
        │    ┌                 ┐                                           │
        │    │ 1  0  0  0  0   │  token0 只能看 token0                    │
        │    │ 1  1  0  0  0   │  token1 看 token0,1                      │
        │    │ 1  1  1  0  0   │  token2 看 token0,1,2                    │
        │    │ 1  1  1  1  0   │  token3 看 token0,1,2,3                  │
        │    │ 1  1  1  1  1   │  token4 看所有                           │
        │    └                 ┘                                           │
        │    0 的位置填 -inf → softmax 后变 0 → 看不到未来               │
        │                                                                  │
        │    mask 形状: (2, 1, 5, 5)                                       │
        │    中间的 1 会广播到 n_heads=12, 即所有头共享同一个 mask       │
        │                                                                  │
        │  ③ softmax → 注意力权重                                         │
        │                                                                  │
        │    scores: (2, 12, 5, 5)                                         │
        │    dim=-1 表示沿最后一个维度(5个key)做 softmax                  │
        │    每行概率和为1: [0.6, 0.3, 0.1, 0, 0]  (0是被mask的)        │
        │    → token0 有 60% 关注 token0, 30% 关注 token1, ...           │
        │                                                                  │
        │  ④ 用注意力权重加权求和 V                                       │
        │                                                                  │
        │    attn: (2, 12, 5, 5) — 注意力权重                             │
        │    v:    (2, 12, 5, 64) — 值向量                                │
        │    attn @ v: (2, 12, 5, 64) — 每个token的加权信息汇总          │
        │                                                                  │
        │    直觉:                                                         │
        │      token0 的输出 = 0.6×V₀ + 0.3×V₁ + 0.1×V₂ + 0×V₃ + 0×V₄  │
        │      即: 从其他 token 的"值"中, 按相关性提取信息               │
        └──────────────────────────────────────────────────────────────────┘
        """
        # ═══ Step 1: 计算 q·k^T — 每个 token 对每个 token 的原始分数 ═══
        # q:   (B, 12, 5, 64)    ← 5个token的查询向量
        # k^T: (B, 12, 64, 5)    ← 转置最后两维: 把 (5,64) 变成 (64,5)
        # q @ k^T: (B, 12, 5, 5) ← 每个 token 对 5 个 token 的点积分数
        #
        # 点积的含义:
        #   scores[b, h, i, j] = q[b,h,i] · k[b,h,j] = token_i 对 token_j 的原始注意力分数
        #   点积越大 → 两个向量越"对齐" → token_i 越关注 token_j
        #
        # 为什么要转置 k?
        #   矩阵乘法规则: (..., 5, 64) @ (..., 64, 5) → (..., 5, 5)
        #   必须让 k 的 64 维和 q 的 64 维对齐才能相乘
        scores = torch.matmul(q, k.transpose(-2, -1))

        # ═══ Step 2: 缩放 — 除以 √d_k ═══
        # scores / √64 = scores / 8
        #
        # 为什么必须缩放?
        #   d_k=64 时, 点积 = 64 个乘积相加, 数值可能很大 (如 +50)
        #   softmax(+50) ≈ 1.0, softmax(-50) ≈ 0.0
        #   → 所有概率集中到1个位置, 其他位置梯度≈0 → 训练停滞
        #   除以 √d_k 后分数缩小到合理范围 (如 ±6), softmax 梯度健康
        scores = scores / math.sqrt(self.d_k)

        # ═══ Step 3: 应用因果掩码 — 让 token 看不到未来 ═══
        # mask 中 0 的位置 → 填 -inf → softmax 后变 0 → 不看
        # mask 中 1 的位置 → 不变 → softmax 后正常参与
        #
        # 例: scores[mask=0处] = -inf
        #   [2.1, -inf, -inf, -inf, -inf]   ← token0 只看 token0
        #   [1.5,  0.8, -inf, -inf, -inf]   ← token1 看 token0,1
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # ═══ Step 4: softmax — 把分数变成概率 (每行和=1) ═══
        # dim=-1: 沿最后一个维度 (5个key) 做 softmax
        #
        # 例: token1 的分数 [1.5, 0.8, -inf, -inf, -inf]
        #   softmax → [0.67, 0.33, 0, 0, 0]
        #   → token1 有 67% 关注 token0, 33% 关注 token1, 不看未来
        #
        # 数学: softmax(x_i) = exp(x_i) / Σ exp(x_j)
        #   -inf 的 exp = 0 → 被 mask 的位置概率为 0
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # ═══ Step 5: 加权求和 — 用注意力权重混合各 token 的值 ═══
        # attn_weights: (B, 12, 5, 5)  ← 注意力概率矩阵
        # v:            (B, 12, 5, 64) ← 每个 token 的值向量
        # attn @ v:     (B, 12, 5, 64) ← 每个token的输出
        #
        # 例: token1 的输出
        #   out₁ = 0.67 × V₀ + 0.33 × V₁ + 0×V₂ + 0×V₃ + 0×V₄
        #   = 按"关注度"从其他 token 的值中提取信息的加权平均
        #
        # 这就是 Attention(Q,K,V) = softmax(QK^T/√d)·V 的完整计算!
        return torch.matmul(attn_weights, v)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope_offset: int = 0,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        多头注意力完整前向传播。

        ┌──────────────────────────────────────────────────────────────────┐
        │  完整维度流转 (默认配置, 自注意力, B=2, seq=5):                 │
        │                                                                  │
        │  输入 (自注意力时 q=k=v=同一个 x):                              │
        │    query: (2, 5, 768)                                            │
        │    key:   (2, 5, 768)                                            │
        │    value: (2, 5, 768)                                            │
        │                                                                  │
        │  ① 线性投影: 768→768, 再拆成多头                               │
        │    w_q(query): (2, 5, 768) — 线性变换, 维度不变                │
        │    .view(2, 5, 12, 64) — 把 768 拆成 12头×64维                 │
        │      为什么能拆? 768 = 12 × 64, 只是换了一种看同一组数的方式   │
        │    .transpose(1,2) → (2, 12, 5, 64)                             │
        │      为什么转置? 把 n_heads 放到前面, 方便后面每个头独立计算   │
        │                                                                  │
        │    同理 k, v 也变成 (2, 12, 5, 64)                              │
        │                                                                  │
        │  ② 可选: RoPE 旋转 (use_rope=True 时)                          │
        │    q, k = rope(q, k) — 旋转后形状不变, 仍是 (2, 12, 5, 64)    │
        │    只在 q 和 k 上旋转, v 不动                                   │
        │                                                                  │
        │  ③ 缩放点积注意力:                                              │
        │    → (2, 12, 5, 64)  详见 scaled_dot_product_attention         │
        │                                                                  │
        │  ④ 拼回多头:                                                    │
        │    .transpose(1,2): (2, 12, 5, 64) → (2, 5, 12, 64)            │
        │      把 n_heads 放回 seq 后面                                    │
        │    .view(2, 5, 768): 12×64=768, 拼回原来的维度                 │
        │      12 个头的 64 维结果按顺序接起来 → 768 维                  │
        │                                                                  │
        │  ⑤ 输出投影:                                                    │
        │    w_o: (2, 5, 768) → (2, 5, 768)                               │
        │    让拼接后的各头信息互相融合, 不是简单拼在一起就完事          │
        │                                                                  │
        │  总结: (2,5,768) → (2,12,5,64) → 注意力 → (2,5,768) → (2,5,768)│
        │  进去 768, 出来还是 768, 但每个 token 现在融合了上下文信息     │
        └──────────────────────────────────────────────────────────────────┘
        """
        batch_size = query.size(0)

        # ═══ 线性投影: 768→768, 把同一个输入投影成不同角色 ═══
        # w_q/w_k/w_v 是三个不同的 Linear(768,768), 各自独立学习
        # 输入相同, 但投影方向不同 → q/k/v 含义不同
        # (B, seq, 768) → (B, seq, 768)
        q = self.w_q(query)   # "我要找什么" — 查询
        k = self.w_k(key)     # "我能提供什么" — 键
        v = self.w_v(value)   # "我的实际内容" — 值

        # ═══ 拆多头: 768 → 12头×64维 ═══
        # 为什么能拆? 768 = 12 × 64, 只是换视角看同一组数
        # (B, seq, 768) → (B, seq, 12, 64)
        q = q.view(batch_size, -1, self.n_heads, self.d_k)
        k = k.view(batch_size, -1, self.n_heads, self.d_k)
        v = v.view(batch_size, -1, self.n_heads, self.d_k)

        # ═══ 转置: 把头维提前, 方便每个头独立算注意力 ═══
        # (B, seq, 12, 64) → (B, 12, seq, 64)
        # 转置后, 第1维是头号, 每个头可以独立做 q@k^T
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ═══ 可选: RoPE 旋转位置编码 (拆多头之后才旋转) ═══
        # 注意: 必须在拼接缓存之前旋转! 缓存里的 K 已经旋转过了
        if self.use_rope:
            q, k = self.rope(q, k, offset=rope_offset)

        # ═══ KV-Cache: 拼接缓存的 K,V 和新 token 的 K,V ═══
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)  # 沿 seq_len 维拼接
            v = torch.cat([v_prev, v], dim=2)

        # ═══ 缩放点积注意力 ═══
        if mask is not None and mask.dim() == 3:
            # (B, seq_q, seq_k) → (B, 1, seq_q, seq_k)
            mask = mask.unsqueeze(1)

        # scaled_dot_product_attention中就是 Attention(Q,K,V) = softmax(QK^T/√d)·V
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)

        # ═══ 拼回多头 + 输出投影 ═══
        # (B, 12, seq, 64) → (B, seq, 12, 64) → (B, seq, 768)
        attn_output = attn_output.transpose(1, 2)  # 头维放回 seq 后面
        attn_output = attn_output.contiguous()      # 保证内存连续 (transpose后不连续)
        attn_output = attn_output.view(
            batch_size, -1, self.d_model)  # 12×64=768, 拼回
        return self.w_o(attn_output), (k, v)
