from __future__ import annotations

import math
from operator import inv

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════
# 1. 旋转位置编码 RoPE (Rotary Position Embedding)
# ═══════════════════════════════════════════════════════════════════════════
class RotaryPositionEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE) — Llama / Qwen / GPT-4 / Gemma 等主流模型的标配。

    === RoPE vs 加法位置编码 (SinusoidalPositionalEncoding) ===

    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  加法位置编码 (原始 Transformer):                                      │
    │    x' = x + PE[pos]                                                    │
    │    位置信息在输入时加一次，之后再也不注入                              │
    │    深层 attention 中位置信息会被逐渐稀释                              │
    │    编码的是绝对位置: "我在第 5 个位置"                                │
    │                                                                        │
    │  RoPE (旋转位置编码):                                                  │
    │    不修改 x，而是在每一层 attention 计算前旋转 Q 和 K                  │
    │    q' = rotate(q, θ_pos)                                              │
    │    k' = rotate(k, θ_pos)                                              │
    │    attn = q' · k'^T                                                   │
    │    每一层都重新注入位置信息 → 不会稀释                                │
    │    编码的是相对位置: q'_m · k'_n 只取决于 m-n (距离)                  │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘

    === RoPE 的数学原理 ===

    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  核心思想: 用二维旋转矩阵编码位置                                     │
    │                                                                        │
    │  把 d_k 维的向量每两个一组，视为复数平面上的点:                       │
    │    (q_0, q_1) → q_0 + i·q_1 (一个复数)                               │
    │    (q_2, q_3) → q_2 + i·q_3                                          │
    │    ...                                                                 │
    │                                                                        │
    │  对第 m 个位置、第 j 对维度，乘以旋转因子:                            │
    │    (q_0 + i·q_1) × e^{i·m·θ_j}                                       │
    │                                                                        │
    │  其中 θ_j = 1 / 10000^{2j/d_k}  (和正弦编码的频率相同)              │
    │                                                                        │
    │  展开 e^{i·m·θ} = cos(m·θ) + i·sin(m·θ):                            │
    │    q'_0 = q_0 · cos(m·θ) - q_1 · sin(m·θ)                           │
    │    q'_1 = q_0 · sin(m·θ) + q_1 · cos(m·θ)                           │
    │                                                                        │
    │  这就是一个 2D 旋转矩阵:                                             │
    │    [q'_0]   [cos(mθ)  -sin(mθ)] [q_0]                               │
    │    [q'_1] = [sin(mθ)   cos(mθ)] [q_1]                               │
    │                                                                        │
    │  关键性质:                                                             │
    │    q'_m · k'_n = f(q, k, m-n)  ← 点积只取决于相对距离 m-n           │
    │    证明: rotate(q,m) · rotate(k,n) = rotate(q·k, m-n)                │
    │                                                                        │
    │  直觉: 每对维度以不同频率旋转                                        │
    │    - 低频维度: θ 小，旋转慢 → 编码远距离关系                         │
    │    - 高频维度: θ 大，旋转快 → 编码近距离关系                         │
    │    和正弦编码的多频率原理相同，但作用方式不同（旋转 vs 加法）         │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘

    === 为什么 RoPE 比加法编码好？ ===

    1. 相对位置: q·k 的值只取决于两个 token 的距离，不是绝对位置
       → 更符合语言的本质（"猫吃鱼"中"吃"和"鱼"的关系不因出现位置而变）

    2. 每层注入: 在每一层 attention 的 Q,K 上都做旋转
       → 位置信息不会被深层计算稀释

    3. 长度外推: 训练 4K 长度，推理时可以处理更长文本
       → 配合 NTK-aware scaling 等技术效果更好

    4. 无额外参数: 旋转角度是预计算的，不需要学习
       → 不增加任何可训练参数

    === 使用方式 ===

    与加法编码不同，RoPE 不在 Decoder 的 forward 里用，
    而是在每一层 MultiHeadAttention 的 Q, K 投影之后、点积之前用:

      # 加法编码 (旧):
      x = token_embed(ids)
      x = x + PE[pos]          ← 只在这里加一次
      for layer in layers:
          x = layer(x, mask)    ← 后面不再注入位置

      # RoPE (新):
      x = token_embed(ids)      ← 不加位置编码
      for layer in layers:
          q, k = w_q(x), w_k(x)
          q, k = rope(q, k)     ← 每层都旋转
          attn = softmax(q @ k.T / √d) @ v

    参数:
        d_k (int):     每个注意力头的维度 (d_model // n_heads)
        max_len (int): 预计算的最大序列长度
        base (float):  频率基数，默认 10000 (和正弦编码相同)
    """

    def __init__(self, d_k: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE (每两个维度一组做旋转)"
        #d_k是每个注意力头的维度，类型是int，表示每个注意力头的维度。
        #token是512的话 dk是64 8个头 512/8=64
        self.d_k = d_k

        # 预计算频率: θ_j = 1 / base^{2j/d_k}, j = 0, 1, ..., d_k/2 - 1
        # 形状: (d_k/2,)
        #arange函数是用来生成一个从0到d_k-1的整数序列。
        #inv_frep是频率的倒数。其实就是缓存了cos和sin的值。形状是(d_k/2,)。
        #每个位置是dk/2个，因为每两个维度一组做旋转。
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        #register_buffer函数是用来注册一个缓冲区。
        #缓冲区不会被视为模型的参数，不会在训练过程中更新，但会随着模型一起保存和加载。
        #inv_freq是频率的倒数。其实就是缓存了cos和sin的值。形状是(d_k/2,)。
        #每个位置是dk/2个，因为每两个维度一组做旋转。
        self.register_buffer("inv_freq", inv_freq)

        # 预计算所有位置的 cos 和 sin
        self._build_cache(max_len)

    def _build_cache(self, max_len: int) -> None:
        """预计算 cos(m·θ) 和 sin(m·θ) 的缓存表。"""
        # positions: (max_len,) → (max_len, 1)
        # positions是位置，类型是torch.Tensor，形状是(max_len, 1)。
        # 从0到max_len-1的整数。
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        # inv_freq: (d_k/2,) → (1, d_k/2)
        # freqs = positions × inv_freq: (max_len, d_k/2)
        freqs = positions * self.inv_freq.unsqueeze(0)
        # 每对维度需要同样的 cos/sin，所以 repeat → (max_len, d_k)
        freqs = torch.cat([freqs, freqs], dim=-1)
        # 缓存 cos 和 sin
        # (max_len, d_k)：表示“序列长度为 max_len，每个 token 有 d_k 维（每对维度一组做旋转）”。
        # max_len 是“最大支持的序列长度”。在这里，它用于预先计算从位置 0 到位置 max_len-1 所有的 cos/sin 旋转频率表，以便后续不同 batch、不同 token 位置都能直接索引这些频率，无需每次动态计算。
        # 也就是说：对于每个序列位置（从 0 到 max_len-1），都有 d_k 个 cos/sin 配对频率。这样同一组 cos/sin 可作用于所有 batch/head。
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        把向量的前半和后半交换并取负，用于实现旋转。

        x = [x_0, x_1, x_2, x_3, ..., x_{d/2-1}, x_{d/2}, ..., x_{d-1}]
        返回 [-x_{d/2}, ..., -x_{d-1}, x_0, ..., x_{d/2-1}]

        这样 x * cos + rotate_half(x) * sin 就等价于对每对维度做 2D 旋转:
          x'_0 = x_0 · cos - x_1 · sin
          x'_1 = x_0 · sin + x_1 · cos
        """
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,  # (batch, n_heads, seq_len_q, d_k)
        k: torch.Tensor,  # (batch, n_heads, seq_len_k, d_k)
        offset: int = 0,  # KV缓存或增量解码时的起始位置偏移
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对 Q 和 K 应用旋转位置编码。

        q: (batch, n_heads, seq_len, d_k)
        k: (batch, n_heads, seq_len, d_k)  — seq_len 可以和 q 不同 (KV cache)
        offset: 位置偏移量，用于推理时的增量解码
                第一次 offset=0，之后每次 +1

        返回: (q_rotated, k_rotated) 形状不变

        使用示例:
            rope = RotaryPositionEmbedding(d_k=64)

            # 训练时:
            q_rot, k_rot = rope(q, k)

            # 推理时 (增量解码, 每次 1 个 token):
            q_rot, k_rot = rope(q, k, offset=current_pos)
        """
        # 这里的 size(2) 表示沿着第三个维度（下标2）取序列长度，对应 (batch, n_heads, seq_len, d_k) 里的 seq_len
        # 取出来是一个整数，表示序列长度。

        # seq_len_q 是一个整数，表示这次计算时 query 有多少个 token（序列长度）。
        # 它不是“有几个头”，也不是“每个头有多少维”。
        # 每个头的维度是 d_k，不同头的 token 数都是 seq_len_q。
        """有多少个token要旋转"""
        seq_len_q = q.size(2)
        seq_len_k = k.size(2)

        # 取对应位置的 cos/sin
       
        # unsqueeze函数是用来在指定的维度上增加一个维度。
        # cos_q和sin_q的形状是(1, 1, seq_len_q, d_k)。
        #cos 和sin都是与计算好了的 不是训练的

        """
        取出来cos和sin，cos和sin是与计算好了的 不是训练的
        self.cos_cached[offset : offset + seq_len_q]取出来是一个形状为(seq_len_q, d_k)的tensor，
        (seq_len_q, d_k)现在代表q是多少个token要旋转，dk代表每个token的维度。
        ？那这里的dk是多少？dk是每个头的维度，是d_model/n_heads。

        # 还是有点疑惑这里的 d_k 的理解：
        # 假设 embedding 维度是512，n_heads=8，那么每个 head 的 d_k=64。
        # 这里的 cos_q, sin_q shape 是 (1, 1, seq_len_q, d_k)，它会作用到每个 head 的 64 个维度上。
        # 注意，每个 head 虽然都是 64 维，但这 8 个 head 的 旋转 不是“重复”应用一组 cos/sin，
        
        # 你的理解是对的！实际上，不论是 head_i 还是 head_{i+1}，
        # 对于同样的序列位置和同样的那一组（比如第一组）维度，它们用的旋转角度（cos/sin 参数）完全一样。
        # 例如：head 0 的[第0-1维]在第M个token上假如旋转30度，那么 head 1 在同样的[第0-1维]、同样的token上也是旋转30度（只是举例说明角度）。
        # 所以 RoPE 的参数不会因为 head 不同而有差异，每个 head 和其它 head 在位置/分组上的旋转方式完全一致，参数是共享并广播的。
        
        #
        # 很好，这个细节很关键！你的理解其实已经很接近本质，我们再澄清下按分 head/分组后的行为：
        #
        # 关键点——即使拆 head，每个 token 下的“第0维”和“第64维”（或者任意两对2i, 2i+1维）使用的旋转角度（θ）也是不同的！
        # 拆 head 仅仅是在张量切片/并行意义上，并不会让“同一个 token 的不同 head 的首维”都用相同旋转——它们各自对应原始 embedding 的不同切片。
        #
        # 举例（假设 embedding 512，n_heads=8, d_k=64）：
        #   - head0 的第0-1维（实际上是总向量的0-1）用 θ_0
        #   - head1 的第0-1维（总向量的64-65）用 θ_32
        #     ...依此类推
        #   - 每两维一组，组编号不同，θ_j 就不同
        #
        # 所以对于同一个 token，不同 head 的第0-1号（以 head 内视角）实际上在全量向量上对应的位置也不同，因此 θ 也不同。
        # 实际代码中的 cos_cached/sin_cached shape 是 (max_seq_len, d_k)，
        # 但你可以把“d_k”理解为“全 embedding 维度中每两维为一组、总共多少组”，head 只是张量 view 的一个维度，不会更改组内编号对应的θ。
        #
        # 具体算的时候：q 的 shape (batch, n_heads, seq_len, d_k)
        #   - 对于每个 token（seq_len 维），把 d_k 维看为 [pair0, pair1, ..., pair_(d_k//2-1)]，每对有自己的θ
        #   - 通过广播，cos_q/sin_q 扩展到所有 heads/批次
        #
        # 总结：
        # - 不同 head、不同组，不会“同θ”
        # - 拆分 head 不会让维度上的旋转角度失去丰富性
        # - 所有位置、所有维度的 θ 都还是唯一确定的（按原始 embedding 逻辑），只是物理上分片以便并行
        #
        # 这样设计的原因是：让所有 head 在自己的子空间都获得最大的信息分布和位置分辨能力，既分片又不损失原始的相对位置信息密度。
        """
        # cos_q的形状是(1, 1, seq_len_q, d_k)。代表每个token的每个维度都要旋转。
        #offset : offset + seq_len_q 表示从offset开始到offset + seq_len_q结束。
        cos_q = self.cos_cached[offset : offset + seq_len_q].unsqueeze(0).unsqueeze(0)
        sin_q = self.sin_cached[offset : offset + seq_len_q].unsqueeze(0).unsqueeze(0)

        cos_k = self.cos_cached[offset : offset + seq_len_k].unsqueeze(0).unsqueeze(0)
        sin_k = self.sin_cached[offset : offset + seq_len_k].unsqueeze(0).unsqueeze(0)

        # 旋转: x' = x * cos + rotate_half(x) * sin
        # 这里的 self._rotate_half(q) 实际上实现了“复数乘法的虚部交换”——
        # 把 q 的后一半和前一半交叉交换符号，等价于 (q_0, q_1) 视为复数时分别取实部和虚部；
        # q 的 shape 是 (..., d_k) 假设 d_k 偶数。
        # 这样做的目的是：q * cos 是原始向量的实部，rotate_half(q) * sin 是旋转后产生的虚部。
        # 代码里没有直接写一个“half q”，而是把 q 分前半和后半——原因来自于 RoPE 公式的数学本质（见类头注）。
        """
        rotate_half函数是用来将一个向量的前半部分和后半部分交换并取负。举例说明：
        假设向量是[1, 2, 3, 4, 5, 6, 7, 8]，那么rotate_half函数会返回[-4, -5, -6, -7, 1, 2, 3, 4]。
        这样做的目的是：q * cos 是原始向量的实部，rotate_half(q) * sin 是旋转后产生的虚部。
        """
        q_rotated = q * cos_q + self._rotate_half(q) * sin_q
        k_rotated = k * cos_k + self._rotate_half(k) * sin_k

        return q_rotated, k_rotated


# ═══════════════════════════════════════════════════════════════════════════
# 2. 原始正弦加法位置编码 (保留兼容)
# ═══════════════════════════════════════════════════════════════════════════
class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦位置编码，不使用学习的方式，而是使用正弦函数和余弦函数。
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    参数:
        d_model是输入的维度，类型是int，表示输入的维度。
        max_len是最大长度，类型是int，表示最大长度。最大长度是5000。指的是输出token的最大长度。
        dropout是dropout的比例，类型是float，表示dropout的比例。
    返回值:
        torch.Tensor: 位置编码，形状为 (max_len, d_model)
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        """
        初始化SinusoidalPositionalEncoding类的实例。
        1. 初始化父类nn.Module。
        2. 创建一个Dropout层，使用传入的dropout参数。
        3. 创建一个零张量pe，形状为(max_len, d_model)，用于存储位置编码。
        4. 创建一个位置张量position，形状为(max_len, 1)，包含从0到max_len-1的整数。
        5. 创建一个除数张量div_term，形状为(d_model/2)，用于计算位置编码中的除数部分。
        6. 使用正弦函数和余弦函数计算位置编码的正弦和余弦部分，并将结果存储在pe中。
        7. 在pe的第一维添加一个维度，使其形状变为(1, max_len, d_model)，以便后续与输入的token表示进行广播操作。
        8. 将pe注册为模型的一个缓冲区，这意味着它不会被视为模型的参数，不会在训练过程中更新，但会随着模型一起保存和加载。
        """
        self.dropout = nn.Dropout(dropout)
        # pe是位置编码，类型是torch.Tensor，形状是(max_len, d_model)。
        pe = torch.zeros(max_len, d_model)
        # position是位置，类型是torch.Tensor，形状是(max_len, 1)。
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term是除数，类型是torch.Tensor，形状是(d_model/2)。
        # exp函数是指数函数，torch.exp(x)返回e的x次幂。这里计算了位置编码中的除数部分，使用了指数函数来计算。
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        # 计算位置编码的正弦和余弦部分。对于偶数位置使用正弦函数，对于奇数位置使用余弦函数。
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算位置编码的正弦和余弦部分。对于偶数位置使用正弦函数，对于奇数位置使用余弦函数。
        pe[:, 1::2] = torch.cos(position * div_term)
        # unsqueeze(0)在位置编码的第一维添加一个维度，使得pe的形状变为(1, max_len, d_model)。这样做是为了在后续的计算中能够与输入的token表示进行广播操作。
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        # register_buffer方法将pe注册为模型的一个缓冲区，这意味着它不会被视为模型的参数，不会在训练过程中更新，但会随着模型一起保存和加载。
        self.register_buffer("pe", pe)

    # x是输入，类型是torch.Tensor，形状是(batch, seq_len, d_model)。
    # 输出是(batch, seq_len, d_model)。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        # x是输入序列的token表示，形状是(batch, seq_len, d_model)。
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
