from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import RotaryPositionEmbedding

"""
# 什么是交叉注意力？ 交叉注意力是解码器和编码器之间的注意力计算。
# MultiHeadAttention是多头注意力机制，支持自注意力与交叉注意力。
# 参数: d_model是输入的维度，类型是int，表示输入的维度。
# n_heads是注意力头的数量，类型是int，表示注意力头的数量。
# dropout是dropout的比例，类型是float，表示dropout的比例。
# d_k是每个头的维度，类型是int，表示每个头的维度。
# 返回值: torch.Tensor: 注意力加权后的结果，形状为 (batch_size, n_heads, seq_len_q, d_k)
# 返回值的形状是(batch_size, n_heads, seq_len_q, d_k)，表示注意力加权后的结果。


# 编码器的的注意力机制是自注意力机制，解码器的注意力机制是交叉注意力机制。
# 参数:
#   d_model (int): 输入的维度
#   n_heads (int): 注意力头的数量
#   dropout (float): dropout 的比例 (默认值为 0.1)
#   d_k (int): 每个头的维度
# 返回值:
#   torch.Tensor: 注意力加权后的结果，形状为 (batch_size, n_heads, seq_len_q, d_k)
# 返回值的形状是(batch_size, n_heads, seq_len_q, d_k)，表示注意力加权后的结果。
"""


class MultiHeadAttention(nn.Module):
    """多头注意力机制，支持自注意力与交叉注意力。

    作为自注意力时，Q/K/V 都来自同一输入。
    作为交叉注意力时，Q 来自解码器，K/V 来自编码器。

    参数:
        d_model (int): 输入的维度
        n_heads (int): 注意力头的数量
        dropout (float): dropout 的比例 (默认值为 0.1)
        d_k (int): 每个头的维度

    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 use_rope: bool = False, max_seq_len: int = 8192):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_rope = use_rope

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        """
         这里的w_o是输出线性变换，将多头拼接还原到原始维度
         原始维度就是输入的x的维度，d_model。
         wo的形状是(d_model, d_model)， 输入是(batch_size, seq_len_q, d_model)，输出是(batch_size, seq_len_q, d_model)。
         这样就实现了将多头拼接还原到原始维度。
         这样就实现了将多头拼接还原到原始维度。
        """
        self.w_o = nn.Linear(d_model, d_model)
        # dropout 用于防止过拟合 nn.Dropout(dropout)
        # 参数:
        #   dropout (float): dropout 的比例 (默认值为 0.1)
        # 返回值:
        #   torch.Tensor: 注意力加权后的结果，形状为 (batch_size, n_heads, seq_len_q, d_k)
        self.dropout = nn.Dropout(dropout)

        # ── RoPE 旋转位置编码 (可选) ──
        # use_rope=True 时，在每一层 attention 的 Q, K 投影之后做旋转
        # use_rope=False 时，走原来的加法正弦编码 (在 Decoder forward 里加)
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
        q k v已经是拆分了多头了，每个注意力头有d_k个维度。



        计算缩放点积注意力（Scaled Dot-Product Attention）。
        这个是交叉注意力的计算，也就是解码器和编码器之间的注意力计算。
        参数:
            q (torch.Tensor): 查询向量，形状为 (batch_size, n_heads, seq_len_q, d_k)
            k (torch.Tensor): 键向量，形状为 (batch_size, n_heads, seq_len_k, d_k)
            v (torch.Tensor): 值向量，形状为 (batch_size, n_heads, seq_len_v, d_k)
            mask (torch.Tensor | None): 可选的注意力掩码，形状为 (batch_size, 1, seq_len_q, seq_len_k)
                                         或 (batch_size, 1, 1, seq_len_k)。被 mask 的位置为 0。

        返回值:
            torch.Tensor: 注意力加权后的结果，形状为 (batch_size, n_heads, seq_len_q, d_k)
        """
        # 假设现在只有一个 token，那么 q 的 shape 是 (batch_size, n_heads, 1, d_k)
        # 其中 n_heads=8，d_k=d_model/8。如果 d_model=512，则 d_k=64。
        # 这意味着"1 个 token 的 512 维特征"将被分为 8 个头，每个头 64 维。
        #
        # 可视化示意:
        # 假设 batch_size=2, n_heads=8, seq_len_q=1, d_model=512, d_k=64
        #
        # 输入 token embedding: (batch_size, seq_len_q=1, d_model=512)
        #
        #     ┌─────────────────────────────────────────────┐
        #     │ token1  (512维)                             │
        #     └─────────────────────────────────────────────┘
        #  → 拆分成8个head，每个head 64维
        #         ┌─────────┬─────────┬─────────┬─────────┬─────┬─────────┐
        #         │ head1   │ head2   │ head3   │ head4   │ ... │ head8   │
        #         │ (64维)  │ (64维)  │ (64维)  │ (64维)  │     │ (64维)  │
        #         └─────────┴─────────┴─────────┴─────────┴─────┴─────────┘
        #     shape: (batch_size, n_heads=8, seq_len_q=1, d_k=64)
        #
        # 每个 head 学习不同的特征子空间，最终所有 head 拼接输出维度仍然是512。
        #
        # 数学表达：d_k = d_model // n_heads
        #
        # 下面就是qkv中的q*k^T / sqrt(d_k)，还未乘v
        """
        用矩阵的形状来解释：
        q 的形状是 (batch_size, n_heads, seq_len_q, d_k)
        k 的形状是 (batch_size, n_heads, seq_len_k, d_k)

        在做注意力时，我们要让每个 query 向量（最后一维 d_k）去和所有 key 向量（也是 d_k）做点积，输出 (seq_len_q, seq_len_k) 的分数矩阵。
        但只有保证矩阵乘法的规律：[..., m, d_k] x [..., d_k, n] = [..., m, n]，才能一次性全部计算出来。

        所以需要把 k 的最后两个维度交换（transpose(-2, -1)），这样 k 变成 (batch_size, n_heads, d_k, seq_len_k)，
        这样 q @ k^T 就可以执行了：
          - q:        (batch_size, n_heads, seq_len_q, d_k)
          - k^T:      (batch_size, n_heads, d_k, seq_len_k)
          - matmul后: (batch_size, n_heads, seq_len_q, seq_len_k)
        # 在这里解释一下四维张量 (batch_size, n_heads, seq_len_q, d_k) 的含义和为什么要这样做矩阵乘法：
        # - x (第1维): batch_size，每个 batch 独立处理
        # - y (第2维): n_heads，注意力的头数（每个头可以关注不同信息）
        # - z (第3维): seq_len_q，序列长度（多少个 query）
        # - w (第4维): d_k，每个 head 的特征维度
        #
        # 在 q @ k^T 时，q 形状:     (batch_size, n_heads, seq_len_q, d_k)
        #                  k^T 形状: (batch_size, n_heads, d_k, seq_len_k)
        # 你可以想象 n_heads 和 batch_size 是“批量”，每个 batch、每个 head 各自做：
        #    [seq_len_q, d_k] @ [d_k, seq_len_k] => [seq_len_q, seq_len_k]
        # 这样最后输出是 (batch_size, n_heads, seq_len_q, seq_len_k)，
        # 代表 batch 里每个样本、每个 head，每个 query 对所有 key 的注意力分数。
        #
        # 重点总结：
        #   - 实际乘法发生在最后两个维度
        #   - 前面的维度就是“并行处理”“分组”而已（比如分不同 head 或不同样本）
        #   - 你用过的二维矩阵乘法就是“单个 head 单个样本”的情况。
        这种操作不仅满足矩阵乘法规律（形状对得上），还能一并并行地算出所有查询和所有键的配对分数。实际上也是为了效率，利用了张量批量乘法的优势。
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # mask变化过程：
        # 随着推理的进行，mask会动态的调整，因为随着推理的进行，用户输入的文本长度会逐渐增加，所以mask会逐渐增加。
        # mask 的形状等于当前生成的目标序列（回归输入）的长度的平方，
        # 即 mask 是 (batch, 1, seq_len_q, seq_len_k)，通常在自回归推理中，seq_len_q=seq_len_k=当前生成 token 的数量。
        # 例如已经生成了6个token，则 mask 形状为 (batch, 1, 6, 6)；
        # 随着推理/生成序列长度的增加，mask 也会随之增大。
        # 这里的“目标序列”指的是当前正在生成的文本序列（decoder 的输出序列），比如机器翻译中的目标句子、文本生成中的输出token序列。
        # 例如，已经生成到第10个token，mask 的形状就是 (batch, 1, 10, 10)，用于保证每个位置只能关注自己和先前已生成的位置（实现因果掩码）。

        # 15个token时，mask形状为 (batch, 1, 15, 15)，
        # 这是因为每个生成的token都需要判断它是否可以关注序列里的每一个token（包括自己）。
        # 所以对每个query token（15个），都要有一个长度为15的mask，控制它能访问的key位置。
        # 总共就是 (15, 15)，正方形（平方关系）——每一行代表一个query，每一列代表一个key。

        # 这个掩码mask了哪些位置？它会mask掉不应该被关注的token位置。
        # 以decoder自注意力为例，mask通常掩盖“未来”token（因果掩码）和padding位置：
        #    - 因果掩码: 保证当前token只能关注自己以及历史token，不能看到未来。具体来说，第i个token只能看到1~i列。
        #    - padding掩码: mask掉填充的pad token（通常是0），防止模型关注无效输入。
        # 所以当序列长度为N时，mask的形状是(batch, 1, N, N)，
        #   其中mask[i, 0, q, k]为0时，表示query位置q不能看key位置k（被mask），为1表示可见。
        # 例如，对20个token序列就是(batch, 1, 20, 20)，
        # 具体：第7行只能关注第1~7列，其余列会被mask。padding部分（如果有）也会对应mask为0。

        """
        mask的形状是(batch, 1, seq_len_q, seq_len_k)，其中seq_len_q是当前生成的目标序列长度，seq_len_k是源序列长度。
        # 这里的 1 代表什么？—— (batch, 1, seq_len_q, seq_len_k) 里的“1”其实是用于 broadcast（自动扩展），通常对应“注意力头”（n_heads）维度。
        # 在标准实现里，mask 形状不会把 head 维度单独展开（而是保留为1），因为:
        #   - 每个head一般使用同一组mask（即因果掩码和pad掩码是一样的），
        #   - 通过 mask.shape=(batch, 1, ...) 这样的定义，PyTorch 允许自动广播到 (batch, n_heads, ...)。
        # 例如：
        #   - 假定 batch=2, n_heads=8, seq_len_q=4, seq_len_k=4。
        #   - mask.shape 只要是 (2, 1, 4, 4)，与注意力分数 scores 的 (2, 8, 4, 4) 乘法时，
        #     “1”会广播为8（即每个head用同一份mask）。
        # 也就是说，mask[batch_idx, 0, q, k] 会自动扩展为 mask[batch_idx, head_idx, q, k] 对所有 head_idx。
        # 这样，既节省内存也简化代码，只要设置一份 mask（同一batch每个head都共享），PyTorch 自动处理head的扩展。
        #
        # 小结:  这个“1”就是为 head/broadcasting 预留的（不是 head 维度本身，而是和 head 维度对齐自动扩展用的）。
        #       如果以后你想给每个head自定义mask，把1换成n_heads就行。
        """
        if mask is not None:
            # 这里的参数是mask==0， 表示mask为0的位置，则将scores的对应位置设置为-inf。
            # -inf表示负无穷， 在softmax中，负无穷的值会变成0， 这样就实现了mask的效果。

            scores = scores.masked_fill(mask == 0, float("-inf"))

        """
         这里是qkv的softmax部分
         dim=-1表示在最后一个维度上进行softmax。
         scoes的形状是(batch_size, n_heads, seq_len_q, seq_len_k)， 在最后一个维度上进行softmax。
         seq_len_q是查询序列的长度，seq_len_k是键序列的长度。
        """
        attn_weights = F.softmax(scores, dim=-1)
        # 推理的时候为什么还要做dropout？因为我们要防止过拟合。
        attn_weights = self.dropout(attn_weights)
        # 注意: 在 softmax 前已经对 scores 除以 sqrt(d_k) 了，无需在此处再除。
        # matmul 是矩阵乘法，将注意力权重和 v 相乘，得到 (batch_size, n_heads, seq_len_q, d_k)
        # 上面已经计算了scores 为什么这里又乘了一个v？因为我们要得到q的每个位置对v的每个位置的注意力分数。
        """ 这里是qkv的乘v部分"""
        return torch.matmul(attn_weights, v)

    # 在decoder.py的DecoderBlock的forward函数中被调用。
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope_offset: int = 0,
    ) -> torch.Tensor:
        """
        计算多头注意力。

        参数:
            query (torch.Tensor): 查询向量，形状为 (batch_size, seq_len_q, d_model)
            key (torch.Tensor): 键向量，形状为 (batch_size, seq_len_k, d_model)
            value (torch.Tensor): 值向量，形状为 (batch_size, seq_len_v, d_model)
            mask (torch.Tensor | None): 可选的注意力掩码
            rope_offset (int): RoPE 位置偏移，用于推理时增量解码 (仅 use_rope=True 时有效)
        """

        batch_size = query.size(0)
        """
        将查询、键、值投影到多头注意力的 Q/K/V 子空间。
        
        参数:
            query (torch.Tensor): 查询向量，形状为 (batch_size, seq_len_q, d_model)
            key (torch.Tensor): 键向量，形状为 (batch_size, seq_len_k, d_model)
            value (torch.Tensor): 值向量，形状为 (batch_size, seq_len_v, d_model)
            mask (torch.Tensor | None): 可选的注意力掩码，形状为 (batch_size, 1, seq_len_q, seq_len_k) 或 (batch_size, 1, 1, seq_len_k)。被 mask 的位置为 0。
        """

        """
        # 这里的 query, key, value 在自注意力（self-attention）时，都是输入 x，三者是相同的。
        # 也就是 q = k = v = x，经由各自的线性层 w_q, w_k, w_v 转换后送入注意力计算。
        这里就是qkv的线性变换+多头注意力拆分。
        """
        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # ── 两种位置编码的分叉点 ──────────────────────────────────
        # use_rope=True  → 在 Q, K 上做旋转 (每层都注入位置信息)
        # use_rope=False → 不做任何事 (位置信息已在 Decoder 入口通过加法编码注入)
        #
        #   if use_rope:
        #     q, k = rope(q, k, offset)    ← 旋转 Q 和 K
        #     然后照常 attn = softmax(q @ k^T / √d) @ v
        #
        #   else:
        #     直接用已经加了正弦位置编码的 q, k 计算 attention
        #     (位置信息在 Decoder.forward 里加过了)
        if self.use_rope:
            # 这里的 q、k 是已经拆分成所有 head 的后张量，每个 head 独立处理，每个 head 拿到自己所有 token 的完整 q, k。
            # 也就是 shape 是 (batch_size, n_heads, seq_len, d_k)，RoPE 是按每个 head 分别对自身全部 token 的 q/k 做旋转。
           
            q, k = self.rope(q, k, offset=rope_offset)

        """
        如果 mask 不为空且 mask 的维度为 3，则将 mask 的维度扩展为 4。

        参数:
            mask (torch.Tensor): 注意力掩码，形状为 (batch_size, seq_len_q, seq_len_k)
        返回值:
            torch.Tensor: 注意力掩码，形状为 (batch_size, 1, seq_len_q, seq_len_k)
        """
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)
        """
        计算注意力加权后的结果。

        参数:
            q (torch.Tensor): 查询向量，形状为 (batch_size, n_heads, seq_len_q, d_k)
            k (torch.Tensor): 键向量，形状为 (batch_size, n_heads, seq_len_k, d_k)
            v (torch.Tensor): 值向量，形状为 (batch_size, n_heads, seq_len_v, d_k)
            mask (torch.Tensor | None): 可选的注意力掩码，形状为 (batch_size, 1, seq_len_q, seq_len_k) 或 (batch_size, 1, 1, seq_len_k)。被 mask 的位置为 0。
        返回值:
            torch.Tensor: 注意力加权后的结果，形状为 (batch_size, n_heads, seq_len_q, d_k)
        """
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        """
        将注意力加权后的结果拼接起来，并转换为原始维度。
        contiguous() 用于确保张量在内存中是连续的，通常在 PyTorch 中用于确保张量在某些操作（如 view）之前是连续的。
        参数:
            attn_output (torch.Tensor): 注意力加权后的结果，形状为 (batch_size, n_heads, seq_len_q, d_k)
        返回值:
            torch.Tensor: 输出结果，形状为 (batch_size, seq_len_q, d_model)
        """
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        # wo是输出线性变换，将多头拼接还原到原始维度
        # 参数:
        #   attn_output (torch.Tensor): 注意力加权后的结果，形状为 (batch_size, n_heads, seq_len_q, d_k)
        # 返回值:
        #   torch.Tensor: 输出结果，形状为 (batch_size, seq_len_q, d_model)
        return self.w_o(attn_output)
