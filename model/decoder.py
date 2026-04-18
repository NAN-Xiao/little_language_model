from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward
from .moe_feedforward import MoEFeedForward
from .positional import SinusoidalPositionalEncoding

# 本 Decoder 实现是“Decoder-Only”模型，仅包含掩码自注意力（Masked Self-Attention），不包含编码器-解码器交叉注意力。
# 适用于 GPT、Llama、Qwen 等自回归语言建模场景，不可直接做 seq2seq 任务。


class DecoderBlock(nn.Module):
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
        # 仅包含 Masked Self-Attention，没有 cross attention
        # ── 两种位置编码的选择在这里体现 ──
        # use_rope=True  → MultiHeadAttention 内部创建 RoPE，每层旋转 Q, K
        # use_rope=False → 不传 RoPE，走原来的加法正弦编码路径
        self.masked_self_attn = MultiHeadAttention(
            d_model, n_heads, dropout,
            use_rope=use_rope, max_seq_len=max_seq_len,
        )
        self.ffn = (
            MoEFeedForward(d_model, d_ff, moe_num_experts, dropout)
            if use_moe
            else PositionwiseFeedForward(d_model, d_ff, dropout)
        )
        # 规一化函数还有：GroupNorm、BatchNorm、InstanceNorm、LayerNorm等。
        # 几个类型的区别是：
        # 常用归一化（Normalization）方式的区别与使用场景详解：
        #
        # 1. BatchNorm（批归一化）:
        #    - 区别：对同一通道维（channel）在一个 batch（批次）内的所有样本空间位置统计均值与方差进行归一化。
        #            归一化维度是 [batch, spatial]，每个通道共享均值和方差。
        #    - 主要用于：图像领域（如 CNN）、大 batch 的训练，适合批量大、分布稳定时。
        #    - 不适合：NLP（序列建模）、batch size 很小时性能变差。
        #
        # 2. LayerNorm（层归一化）:
        #    - 区别：对单一样本的最后一个或几个特征维度整体进行归一化（例如 transformer 的每个 token 整行归一化），即对每个样本分别统计均值方差。
        #            归一化适用于任意 batch size，不依赖 batch 的分布。
        #    - 主要用于：NLP/Transformer类模型、自回归建模、小 batch 或 batch size＝1的任务、序列建模、强化学习等。
        #
        # 3. InstanceNorm（实例归一化）:
        #    - 区别：对每个样本、每个通道（channel）独立统计归一化（即每个样本里的每个通道单独统计），归一化维度是每个 [n, c] 的空间位置。
        #    - 主要用于：风格迁移、生成式图像模型（如 CycleGAN 等），适合图像合成、效果偏艺术化等领域。
        #
        # 4. GroupNorm（分组归一化）:
        #    - 区别：先把全部通道分成若干组，每组在单一样本内归一化（即同一组的通道一起统计均值、方差）。
        #            不依赖 batch size，兼具部分 BatchNorm 效果，又适合小 batch。
        #    - 主要用于：小 batch、高分辨率图像，常见于目标检测、分割模型等对归一化稳定性要求高的任务中。
        #
        # 总结：
        # - BatchNorm：依赖 batch，适合图像和大 batch，训练推理需注意一致性。
        # - LayerNorm：不依赖 batch，只看单条样本，最常用于 NLP/Transformer。
        # - InstanceNorm：每样本每通道归一化，多用于图像生成/风格转换。
        # - GroupNorm：通道分组归一化，适合小 batch、不依赖 batch size，兼顾 BN 部分优点。
        #
        # 在本 Decoder Block 中，采用 LayerNorm，因其更适合 NLP 任务和小 batch 的序列建模场景。
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 必须使用掩码，让每个时刻只能看到自己和以前，decode-only 的核心
        #？这里的normal1就是query、key、value的归一化。
        normed = self.norm1(x)
        # 残差网络连接，将输入x与多头注意力输出的结果相加。
        # atten中的forward是这样调用的：这里 masked_self_attn 前向会被传入 normed 作为 query、key、value（即自注意力），
        # 还有 tgt_mask 作为 mask（掩码），控制只能访问当前及之前的 token，不看未来。
        # 返回的是注意力输出，再经过 dropout，再加残差，x 完成更新。
        # 在哪计算的线性q kv w_q w_k w_v？ 在attention.py的forward函数中计算的。
        x = x + self.dropout1(self.masked_self_attn(normed, normed, normed, tgt_mask))
        normed = self.norm2(x)
        x = x + self.dropout2(self.ffn(normed))
        return x


class Decoder(nn.Module):
    """
    Decoder-Only结构。输入 token 序列，依次经过嵌入、位置编码、若干DecoderBlock（仅掩码自注意力），最终输出序列特征。
    这类结构广泛用于生成式自回归语言模型，输入与输出长度一致，没有编码器部分。

    ┌─────────────────────────────────────────────────────────────────┐
    │  两种位置编码模式对比 (use_rope 开关):                         │
    │                                                                 │
    │  use_rope=False (原始 Transformer / GPT-2 风格):               │
    │    token_ids → Embedding → + SinusoidalPE → DecoderBlock ×N   │
    │    位置信息只在入口加一次，深层会逐渐稀释                      │
    │                                                                 │
    │  use_rope=True (Llama / Qwen / GPT-NeoX 风格):                │
    │    token_ids → Embedding → DecoderBlock ×N                     │
    │                              ↑ 每个 block 内部:                │
    │                              q, k = rope(q, k)                 │
    │    不加入口位置编码，每层注意力都旋转 Q,K 注入位置信息         │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
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
        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )

        # ── 位置编码: 二选一 ──
        if self.use_rope:
            # RoPE 模式: 不需要加法位置编码
            # 位置信息由每个 DecoderBlock 内部的 MultiHeadAttention.rope 旋转注入
            # 只保留一个 dropout 层 (和原来 SinusoidalPE 里的 dropout 对应)
            self.embed_dropout = nn.Dropout(dropout)
            self.pos_encoding = None
        else:
            # 原始 Transformer 模式: 在输入时加法注入正弦位置编码
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
            self.embed_dropout = None

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                    use_moe=use_moe,
                    moe_num_experts=moe_num_experts,
                    use_rope=use_rope,
                    max_seq_len=max_seq_len,
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
        # 输入: tgt 为 token id 序列 (batch, seq)
        # 必须传入 decoder-only 掩码，确保模型自回归（无信息泄露）
        x = self.token_embedding(tgt) * (self.d_model**0.5)

        # ── 位置编码分叉 ──
        if self.use_rope:
            # RoPE 模式: 不加位置编码，只做 dropout
            # 位置信息将在每一层 attention 内部通过旋转 Q, K 注入
            x = self.embed_dropout(x)
        else:
            # 原始模式: 在入口一次性加上正弦位置编码
            x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, tgt_mask)
        return self.norm(x)
