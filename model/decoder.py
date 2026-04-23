"""
Decoder — 自回归语言模型的核心：逐层提取越来越抽象的上下文特征
================================================================

这是 Decoder-Only 架构 (GPT/Llama/Qwen 风格)。
只有自注意力 (每个 token 看自己和之前的 token), 没有交叉注意力。

默认配置: d_model=768, n_heads=12, n_layers=10, d_ff=3072, max_seq_len=256

整体结构:
  token_ids → Embedding → 位置编码 → 10层DecoderBlock → LayerNorm → hidden

  每一层 DecoderBlock 内部:
    x → LayerNorm → Masked Self-Attention → 残差(+)
      → LayerNorm → FFN (或 MoE FFN)     → 残差(+)
      → 输出

  形状始终是 (B, seq, 768), 但每个 token 的表示越来越"懂"上下文。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import KVCache, MultiHeadAttention
from .feedforward import PositionwiseFeedForward
from .moe_feedforward import MoEFeedForward
from .positional import SinusoidalPositionalEncoding


class DecoderBlock(nn.Module):
    """
    单个 Transformer 解码器块。

    ┌──────────────────────────────────────────────────────────────────┐
    │  数学原理                                                       │
    │                                                                  │
    │  原始 Transformer 论文 (Post-Norm):                             │
    │    x = LayerNorm(x + Attention(x))                              │
    │    x = LayerNorm(x + FFN(x))                                    │
    │                                                                  │
    │  本项目使用 Pre-Norm (GPT-2之后的主流):                         │
    │    x = x + Attention(LayerNorm(x))                              │
    │    x = x + FFN(LayerNorm(x))                                    │
    │                                                                  │
    │  Pre-Norm 为什么更好?                                            │
    │    Post-Norm: 残差加完后才归一化 → 深层梯度可能爆炸/消失       │
    │    Pre-Norm:  先归一化再进子层 → 输入直接通过残差跳到输出      │
    │               → 梯度可以沿残差路径无衰减地传回第1层             │
    │               → 训练 10 层、100 层都不会崩                      │
    │                                                                  │
    │  残差连接 (x + ...) 的数学意义:                                 │
    │    没有 +x: 子层必须学出完整的变换 f(x)                         │
    │    有  +x:  子层只需要学残差 Δx = f(x) - x                    │
    │             如果这层没用, Δx≈0, 输出≈x, 信息无损通过          │
    │             梯度: ∂(x + f(x))/∂x = 1 + ∂f/∂x ≥ 1             │
    │             梯度永远≥1, 不会消失                                │
    │                                                                  │
    ├──────────────────────────────────────────────────────────────────┤
    │  完整维度流转 (B=2, seq=5, d_model=768, n_heads=12, d_k=64)    │
    │                                                                  │
    │  输入 x: (2, 5, 768)                                           │
    │                                                                  │
    │  ═══ 子层1: Masked Self-Attention ═══                            │
    │                                                                  │
    │  norm1 = LayerNorm(x)                                           │
    │    LayerNorm 做什么?                                             │
    │      对每个 token 的 768 维独立归一化:                           │
    │      out = (x - mean) / √(var + ε) × γ + β                    │
    │      mean, var: 该 token 768维的均值方差                        │
    │      ε=1e-5: 防除零                                            │
    │      γ, β: 可学习参数, 初始γ=1, β=0 (即初始时归一化)          │
    │    → (2, 5, 768) 形状不变, 值域归到接近 0 均值 1 方差         │
    │                                                                  │
    │  attn_out = MultiHeadAttention(normed, normed, normed, mask)    │
    │    自注意力: q = k = v = normed (同一个输入三个角色)             │
    │    mask = 因果掩码 (下三角, 只看过去)                           │
    │                                                                  │
    │    内部维度展开:                                                 │
    │      w_q(normed):   (2,5,768) @ (768,768) → (2,5,768)          │
    │        Linear 就是矩阵乘法: y = x @ W^T + b                    │
    │        W 形状: (768, 768), b 形状: (768,)                       │
    │        把 768 维映射到 768 维 (只是换了表示空间)                │
    │      .view(2,5,12,64): 把 768 拆成 12头 × 64维/头              │
    │        768 = 12 × 64, 只是换视角看同一组数                     │
    │      .transpose(1,2): (2,12,5,64)                                │
    │        把头维提前, 后面每个头独立算注意力                       │
    │                                                                  │
    │      q: (2,12,5,64)  k: (2,12,5,64)  v: (2,12,5,64)           │
    │                                                                  │
    │      [可选] RoPE: q,k = rope(q,k)                               │
    │        旋转后形状不变: (2,12,5,64) → (2,12,5,64)               │
    │        注入位置信息, 让注意力感知 token 距离                    │
    │                                                                  │
    │      scores = q @ k^T / √64:                                     │
    │        (2,12,5,64) @ (2,12,64,5) → (2,12,5,5)                   │
    │        每个 token 对 5 个 token 的注意力分数                    │
    │        除以 √64=8: 防止分数太大导致 softmax 饱和               │
    │                                                                  │
    │      mask: 因果掩码 (2,1,5,5)                                    │
    │        [1 0 0 0 0]                                               │
    │        [1 1 0 0 0]                                               │
    │        [1 1 1 0 0]                                               │
    │        [1 1 1 1 0]                                               │
    │        [1 1 1 1 1]                                               │
    │        0 的位置填 -inf → softmax 后变 0 → 看不到未来           │
    │        中间的 1 广播到 12 个头 (所有头共享同一个 mask)          │
    │                                                                  │
    │      weights = softmax(scores, dim=-1): (2,12,5,5)               │
    │        每行概率和=1: token2 的权重 [0.3, 0.4, 0.3, 0, 0]     │
    │                                                                  │
    │      out = weights @ v: (2,12,5,64)                              │
    │        每个 token 的输出 = 所有关注 token 值的加权平均          │
    │                                                                  │
    │      拼回: .transpose(1,2) → .view → (2,5,768)                  │
    │        12个头的64维结果按顺序接起来 = 768维                     │
    │      w_o: (2,5,768) → (2,5,768)                                 │
    │        输出投影: 融合各头信息                                    │
    │                                                                  │
    │  x = x + dropout(attn_out)                                       │
    │    残差连接: 原始 x + 注意力输出                                 │
    │    dropout: 训练时随机丢弃部分值, 防过拟合                       │
    │    → (2, 5, 768)                                                │
    │                                                                  │
    │  ═══ 子层2: FFN ═══                                              │
    │                                                                  │
    │  norm2 = LayerNorm(x)   → (2, 5, 768)                           │
    │                                                                  │
    │  ffn_out = FFN(norm2):                                           │
    │    Linear(768→3072):  (2,5,768) → (2,5,3072)                    │
    │      为什么768→3072? 升维4倍, 在高维空间做非线性变换            │
    │      单层768→768只有线性变换, 升维+激活才有非线性表达力          │
    │    ReLU:                (2,5,3072) → (2,5,3072)                  │
    │      负数变0, 引入非线性; 约50%神经元被关掉, 防止过拟合         │
    │    Dropout:              (2,5,3072)                               │
    │    Linear(3072→768):    (2,5,3072) → (2,5,768)                  │
    │      降维回来, 输出和输入同形状                                 │
    │    Dropout:              (2,5,768)                                │
    │                                                                  │
    │    或者 MoE FFN (use_moe=True):                                   │
    │      有4个专家FFN, 每个token只用1个:                             │
    │      router(token) → 选专家i → 专家i的FFN(token)                │
    │      维度变化和普通FFN一样: 768→3072→768                        │
    │      好处: 参数4倍, 但每次只算1倍, 省计算                      │
    │                                                                  │
    │  x = x + dropout(ffn_out)                                        │
    │    残差连接: 原始 x + FFN 输出                                   │
    │    → (2, 5, 768) — 一个 DecoderBlock 结束                       │
    │                                                                  │
    │  总结: 进(2,5,768), 出(2,5,768), 形状永远不变                  │
    │  但每个token从"只知道自己"变成"知道所有历史token的信息"         │
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
        # 掩码自注意力: 每个token只看自己和之前的token
        self.masked_self_attn = MultiHeadAttention(
            d_model, n_heads, dropout,
            use_rope=use_rope, max_seq_len=max_seq_len,
        )
        # FFN: 二选一 — 普通FFN 或 MoE FFN
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
        rope_offset: int = 0,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """(B, seq, d_model) → (B, seq, d_model), (k_cache, v_cache)"""
        # Pre-Norm + 残差: x + Attn(norm(x))
        normed = self.norm1(x)
        attn_out, new_kv_cache = self.masked_self_attn(
            normed, normed, normed, tgt_mask,
            rope_offset=rope_offset, kv_cache=kv_cache,
        )
        x = x + self.dropout1(attn_out)
        # Pre-Norm + 残差: x + FFN(norm(x))
        normed = self.norm2(x)
        x = x + self.dropout2(self.ffn(normed))
        return x, new_kv_cache


class Decoder(nn.Module):
    """
    Decoder 堆叠: token id → Embedding → 位置编码 → N层DecoderBlock → LayerNorm → hidden

    ┌──────────────────────────────────────────────────────────────────┐
    │  完整维度流转 (B=2, seq=5, vocab=68, 默认配置):                 │
    │                                                                  │
    │  输入: token_ids (2, 5) — 5个token的id                         │
    │    例如: [[3, 12, 45, 7, 2],                                    │
    │           [8, 3, 12, 0, 0]]                                     │
    │    第2个样本末尾2个pad(id=0)                                    │
    │                                                                  │
    │  ── Step 1: Token Embedding ─────────────────────────────────    │
    │                                                                  │
    │  Embedding(68, 768): 查表操作                                   │
    │    有一个 (68, 768) 的查找表, 每行是一个token的向量             │
    │    id=3  → 查第3行  → 768维向量, 如 [0.12, -0.03, ...]       │
    │    id=12 → 查第12行 → 768维向量, 如 [0.44, 0.91, ...]        │
    │    → (2, 5, 768)                                                │
    │                                                                  │
    │  × √768 ≈ 27.7: 缩放embedding                                  │
    │    → (2, 5, 768)                                                │
    │    为什么缩放?                                                   │
    │      Embedding初始化: N(0, 1/√768) → 值约±0.04               │
    │      位置编码值: sin/cos 输出约±1                               │
    │      不缩放: 位置编码值 >> embedding值 → 语义被位置淹没        │
    │      缩放后: embedding值约±1, 和位置编码同量级 → 两者平衡      │
    │      为什么乘√d_model而不是别的?                                 │
    │        原始Transformer论文的设定, 和点积注意力的缩放对应       │
    │                                                                  │
    │  ── Step 2: 位置编码 (二选一) ──────────────────────────────    │
    │                                                                  │
    │  use_rope=False (原始Transformer / GPT-2 风格):                 │
    │    x = x + SinusoidalPE(pos) → (2, 5, 768)                     │
    │                                                                  │
    │    SinusoidalPE 做什么?                                          │
    │      PE(pos, 2i)   = sin(pos / 10000^(2i/768))                 │
    │      PE(pos, 2i+1) = cos(pos / 10000^(2i/768))                 │
    │      对每个位置生成 768 维向量, 加到 token embedding 上         │
    │      位置0: [sin(0), cos(0), sin(0), cos(0), ...] = [0,1,0,1,.]│
    │      位置1: [sin(1), cos(1), sin(1/10000^...), ...]            │
    │      不同位置的 sin/cos 值不同 → token有了位置信息              │
    │      只在入口加一次, 后面10层不再注入 → 深层位置信息可能稀释  │
    │                                                                  │
    │  use_rope=True (Llama/Qwen 风格):                               │
    │    x = dropout(x) → (2, 5, 768)                                 │
    │    不加任何位置编码!                                             │
    │    位置信息在每层attention内部通过旋转Q,K注入 (见positional.py)│
    │    → 每层都重新注入 → 位置信息不会被稀释                       │
    │    → 编码相对位置 ("我离你3步") 而非绝对位置 ("我在第5位")   │
    │                                                                  │
    │  ── Step 3: 10层 DecoderBlock ──────────────────────────────    │
    │                                                                  │
    │  每层: (2, 5, 768) → (2, 5, 768), 形状不变                    │
    │                                                                  │
    │  层数越多, token 的表示越"懂"上下文:                           │
    │    第1层: token学到了相邻token的关系 (如"猫"和"吃")            │
    │    第5层: token学到了短语级的关系 (如"猫吃"是一个主谓)         │
    │    第10层: token学到了句子级的关系 (如整句是问句还是陈述)       │
    │                                                                  │
    │  ── Step 4: 最终 LayerNorm ─────────────────────────────────    │
    │                                                                  │
    │  LayerNorm(768): (2, 5, 768) → (2, 5, 768)                     │
    │    稳定输出值域, 方便后续 Linear(768→68) 投影到词表            │
    │    没有这个归一化, 不同位置的输出值域可能差异很大               │
    │                                                                  │
    │  总结: (2,5) → (2,5,768) → 10层不变 → (2,5,768)               │
    │  每个 token 从一个 id 变成了融合了所有历史上下文的 768 维向量   │
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

        # (68, 768) 的查找表, padding_idx=0 让 pad token 的梯度为0
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        # 位置编码: 二选一
        if self.use_rope:
            # RoPE模式: 不需要加法位置编码, 只保留dropout
            self.embed_dropout = nn.Dropout(dropout)
            self.pos_encoding = None
        else:
            # 正弦编码模式: 在入口加一次
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
            self.embed_dropout = None

        # 10层 DecoderBlock
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
        rope_offset: int = 0,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """token_ids: (B, seq) → hidden: (B, seq, d_model), kv_cache"""
        # Embedding + 缩放
        x = self.token_embedding(tgt) * (self.d_model**0.5)

        # 位置编码
        if self.use_rope:
            x = self.embed_dropout(x)
        else:
            # KV-Cache 增量解码时, 用 offset 让新 token 加上正确位置的编码
            offset = rope_offset if kv_cache is not None else 0
            x = self.pos_encoding(x, offset=offset)

        # 10层 DecoderBlock, 逐层透传 kv_cache
        new_kv_cache: KVCache = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None and i < len(kv_cache) else None
            x, layer_new_cache = layer(x, tgt_mask, rope_offset=rope_offset, kv_cache=layer_cache)
            new_kv_cache.append(layer_new_cache)

        # 最终归一化
        return self.norm(x), new_kv_cache
