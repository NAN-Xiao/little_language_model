"""
多头注意力机制 (Multi-Head Attention) —— Transformer 的心脏
=========================================================

作为初学者，你不需要先懂数学公式。先理解一个生活类比：

【图书馆找书】
想象你走进一座图书馆，想查"猫为什么喜欢抓老鼠"。

- 你心里的问题  →  Query (查询): "猫为什么喜欢抓老鼠"
- 每本书的标签  →  Key (键):    "猫咪习性"、"哺乳动物捕食"、"宠物护理"
- 书的实际内容  →  Value (值):   书里写的具体知识

你做三件事：
1. 拿你的问题(Q)，去和每本书的标签(K)比较 —— "猫咪习性"最相关！
2. 算出每本书的"相关度分数"，转成概率 (softmax)
3. 按概率加权，把相关书的内容(V)混合在一起读

这就是"注意力"的本质：
    用问题(Q)找标签(K)，按相关度混合内容(V)。

【多头 = 多个人同时查】
你一个人查可能漏掉角度。叫 12 个朋友同时查：
- 朋友1关注"生物学角度"
- 朋友2关注"进化论角度"
- 朋友3关注"行为学角度"
- ...

12 个人各自查完，把结果拼在一起，信息更全面。
这就是"多头" —— 12 个独立的注意力同时运行。

【因果掩码 = 不能偷看未来】
写作时你写第5个字，只能看前4个字，不能偷看第6个字。
注意力也一样：生成第5个 token 时，只能关注前5个(包括自己)，
后面的位置会被"遮掉"(mask=0)。

【模型配置】
d_model=768: 每个 token 用 768 个数字表示
n_heads=12:  12 个朋友同时查
d_k=64:      每个朋友用 64 维向量做查询 (768÷12=64)
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
    多头注意力 —— 让模型学会"看上下文"。

    没有注意力时，模型处理每个 token 就像盲人摸象，只看自己。
    有了注意力，模型处理"它"的时候，会自动回看"猫"和"抓"，
    从而理解"它"指的是猫。

    ┌────────────────────────────────────────────────────────────┐
    │  生活类比: 12 个朋友分组查资料                              │
    │                                                            │
    │  输入: 一句话 "猫 抓 了 它"，每个字变成 768 维向量         │
    │                                                            │
    │  Step 1: 生成问题/标签/内容 (Q/K/V)                        │
    │    对"抓"这个字:                                           │
    │      Q("抓") = "我要找谁被抓了?"                          │
    │      K("抓") = "我是动词'抓'"                             │
    │      V("抓") = "抓这个动作的语义信息"                     │
    │                                                            │
    │  Step 2: 12 个人同时查 (多头)                              │
    │    朋友1 (64维): Q1("抓") 和 K1("猫") 很相关!            │
    │    朋友2 (64维): Q2("抓") 和 K2("它") 也有点关系...      │
    │    ...                                                     │
    │                                                            │
    │  Step 3: 混合信息                                          │
    │    "抓"字的最终表示 = 自己的信息 + 从"猫""它"提取的信息   │
    │    这样模型就知道"抓"的主语是猫、宾语是它                  │
    └────────────────────────────────────────────────────────────┘

    参数:
        d_model: 输入向量维度 (768)，每个 token 用多少数字表示
        n_heads: 注意力头数 (12)，几组独立的查询
        dropout: 随机丢弃比例，防止过拟合
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 use_rope: bool = False, max_seq_len: int = 8192,
                 n_kv_heads: int | None = None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model    # 768, 每个 token 的维度
        self.n_heads = n_heads    # 12, 注意力头的数量
        self.d_k = d_model // n_heads  # 64, 每个头处理的维度 (768÷12)
        self.use_rope = use_rope

        # GQA: n_kv_heads 控制 K/V 的投影头数
        #   None → 标准 MHA (n_kv_heads = n_heads)
        #   3    → GQA (12 个 Q 头共享 3 套 K/V)
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_rep = n_heads // self.n_kv_heads  # K/V 重复次数
        assert n_heads % self.n_kv_heads == 0, "n_heads 必须能被 n_kv_heads 整除"

        # Q/K/V 投影: Q 保持 n_heads, K/V 用 n_kv_heads (GQA 时更少)
        self.w_q = nn.Linear(d_model, n_heads * self.d_k)        # (768 → 768)
        self.w_k = nn.Linear(d_model, self.n_kv_heads * self.d_k)  # (768 → n_kv_heads×64)
        self.w_v = nn.Linear(d_model, self.n_kv_heads * self.d_k)  # (768 → n_kv_heads×64)
        self.w_o = nn.Linear(d_model, d_model)  # 把多头结果融合回 768 维
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
        注意力的核心计算 —— 打分 + 加权混合。qkv的计算在这

        用具体数字理解 (假设 B=1, 头数=1, 句子只有3个词):

        ┌────────────────────────────────────────────────────────────┐
        │  输入: "我 爱 你"                                          │
        │                                                            │
        │  q (查询): 每个词作为"提问者"时的向量                       │
        │    q[0] = "我"想问什么?   q[1] = "爱"想问什么?            │
        │                                                            │
        │  k (标签): 每个词作为"被查询者"时的向量                     │
        │    k[0] = "我"的标签      k[1] = "爱"的标签               │
        │                                                            │
        │  v (内容): 每个词的实际语义向量                             │
        │    v[0] = "我"的实际信息  v[1] = "爱"的实际信息           │
        └────────────────────────────────────────────────────────────┘

        计算过程:
        ┌────────────────────────────────────────────────────────────┐
        │  ① 打分: q · k^T                                           │
        │                                                            │
        │  点积 (dot product) 是什么?                                │
        │    两个向量的"相似度"度量。方向越一致，点积越大。          │
        │    例: q("爱") 和 k("我") 的点积 = 1.2                   │
        │        q("爱") 和 k("爱") 的点积 = 3.5  ← 最相关!        │
        │        q("爱") 和 k("你") 的点积 = 2.1                   │
        │                                                            │
        │    结果矩阵 (3×3):                                         │
        │              我      爱      你                            │
        │    我      [3.0,   1.1,   0.5]  ← "我"最关注"我"自己     │
        │    爱      [1.2,   3.5,   2.1]  ← "爱"最关注"爱"自己     │
        │    你      [0.3,   1.8,   4.0]  ← "你"最关注"你"自己     │
        │                                                            │
        │  ② 缩放: 除以 √d_k                                         │
        │                                                            │
        │  为什么要除?                                                │
        │    d_k=64 时，点积是 64 个数相乘再相加，结果容易很大。     │
        │    比如分数变成 [50, 30, 10]，softmax 后 ≈ [1.0, 0, 0]    │
        │    → 概率全挤在一个位置，其他位置学不到东西 (梯度消失)     │
        │                                                            │
        │    除以 √64 = 8 后，分数变成 [6.25, 3.75, 1.25]           │
        │    softmax 后 ≈ [0.85, 0.13, 0.02]，梯度健康，各位置都能学│
        │                                                            │
        │  ③ 掩码: 把未来的位置遮住                                  │
        │                                                            │
        │  因果掩码 (下三角为1):                                      │
        │    我  爱  你                                              │
        │  我 [1,  0,  0]  ← "我"只能看自己                         │
        │  爱 [1,  1,  0]  ← "爱"能看"我"和"爱"                    │
        │  你 [1,  1,  1]  ← "你"能看所有                           │
        │                                                            │
        │  0 的位置填 -inf，softmax 后变成 0                         │
        │  "爱"行变成: [1.2, 3.5, -inf]                             │
        │              ↓ softmax                                     │
        │              [0.08, 0.92, 0]                               │
        │  → "爱"这个词 92% 关注自己，8% 关注"我"，不看"你"        │
        │                                                            │
        │  ④ 加权混合 V                                               │
        │                                                            │
        │  "爱"字的最终输出 = 0.08×V("我") + 0.92×V("爱") + 0×V("你")│
        │                                                            │
        │  直观理解: "爱"这个字的新表示，混合了"我"的信息和"爱"本身 │
        │  的信息。这样模型就知道"爱"前面有个"我"了。                │
        └────────────────────────────────────────────────────────────┘
        """
        # ═══ Step 1: 计算相似度分数 ═══
        # q @ k^T: 每个 token 对其他所有 token 的"相关度打分"
        #
        # 形状变化:
        #   q:     (B, n_heads, seq_len, d_k)      例: (2, 12, 5, 64)
        #   k^T:   (B, n_heads, d_k, seq_len)      例: (2, 12, 64, 5)
        #   结果:  (B, n_heads, seq_len, seq_len)  例: (2, 12, 5, 5)
        #
        # 为什么是 (seq_len, seq_len)?
        #   每个位置的 token 都要给其他所有位置的 token 打分
        #   5 个 token × 5 个 token = 25 个分数
        scores = torch.matmul(q, k.transpose(-2, -1))
        #   这就是"点积注意力"的核心计算: q 和 k 的点积衡量相关度!
        #   每个分数 = q 的 64 维和 k 的 64 维对应相乘再相加，衡量它们的相似度。

        # ═══ Step 2: 缩放 —— 防止分数太大导致 softmax 饱和 ═══
        # 除以 √d_k 是 Transformer 论文里的关键技巧
        # 没有它，注意力会变得很"硬" (只关注1个位置)，模型学不好
        scores = scores / math.sqrt(self.d_k)

        # ═══ Step 3: 应用掩码 —— 遮住未来信息 ═══
        # mask == 0 的位置 (未来) 填 -inf，softmax 后概率为 0
        # 这是"因果"(causal)注意力的核心：生成时不能偷看后面的词
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # ═══ Step 4: softmax —— 分数变概率 ═══
        # dim=-1: 沿最后一个维度 (seq_len) 做 softmax
        # 每行和为 1，表示"这个 token 对所有 token 的关注度总和为100%"
        attn_weights = F.softmax(scores, dim=-1)
        # dropout: 训练时随机把一些注意力权重设成0，防止过拟合
        attn_weights = self.dropout(attn_weights)

        # ═══ Step 5: 用概率加权混合 V ═══
        # attn_weights: (B, n_heads, seq, seq)  —— 关注度概率矩阵
        # v:            (B, n_heads, seq, d_k)  —— 每个词的实际内容
        # 结果:         (B, n_heads, seq, d_k)  —— 混合后的新表示
        #
        # 每个词的输出 = 所有词的 V 按关注度加权平均
        # 这就是"Attention(Q,K,V)"公式的完整实现!
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
        完整前向传播 —— 从输入到输出的全部步骤。

        ┌────────────────────────────────────────────────────────────┐
        │  例子: 处理 "猫 抓 了 它"，batch=2，默认配置                │
        │                                                            │
        │  输入形状: (B=2, seq=4, d_model=768)                       │
        │  表示: 2 个样本，每个 4 个词，每个词 768 个数字            │
        │                                                            │
        │  ━━━ Step 1: 线性投影 (w_q, w_k, w_v) ━━━                 │
        │                                                            │
        │  三个 Linear(768→768) 分别做矩阵乘法:                      │
        │    w_q(query): "猫" → Q("猫") = "谁在抓?"                 │
        │    w_k(key):   "抓" → K("抓") = "动词-抓"                │
        │    w_v(value): "抓" → V("抓") = 抓的语义信息              │
        │                                                            │
        │  注意: w_q/w_k/w_v 是三个不同的矩阵，参数独立学习!         │
        │  同一个"抓"字，作为 Q/K/V 时被投影到不同的空间。           │
        │                                                            │
        │  形状: (2, 4, 768) → (2, 4, 768)  维度不变，内容变了      │
        │                                                            │
        │  ━━━ Step 2: 拆多头 ━━━                                    │
        │                                                            │
        │  把 768 维拆成 12 组 × 64 维:                             │
        │    (2, 4, 768) → view → (2, 4, 12, 64)                   │
        │                                                            │
        │  view() 是什么?                                            │
        │    不改变数据，只改变"看待方式"。                          │
        │    就像把 768 个数字排成一排，改成 12 行 × 64 列的表格。  │
        │    每行 (64维) 就是一个"头"要处理的数据。                  │
        │                                                            │
        │  然后 transpose:                                           │
        │    (2, 4, 12, 64) → transpose → (2, 12, 4, 64)           │
        │                                                            │
        │  为什么要 transpose?                                       │
        │    PyTorch 矩阵乘法要求最后两维是矩阵形状。                │
        │    transpose 后: (B, n_heads, seq, d_k)                  │
        │    这样每个头可以独立做 (seq, d_k) @ (d_k, seq) 的乘法   │
        │                                                            │
        │  ━━━ Step 3: RoPE 旋转 (可选) ━━━                          │
        │                                                            │
        │  如果不旋转，模型不知道"猫"在第1位还是第3位。              │
        │  RoPE 对 q 和 k 做旋转，把位置信息编码进去。               │
        │  旋转后形状不变，仍是 (2, 12, 4, 64)。                     │
        │                                                            │
        │  ━━━ Step 4: 缩放点积注意力 ━━━                            │
        │                                                            │
        │  每个头独立计算 (4×4) 的注意力矩阵。                       │
        │  返回: (2, 12, 4, 64) —— 每个头输出的新表示               │
        │                                                            │
        │  ━━━ Step 5: 拼回 + 融合 (w_o) ━━━                         │
        │                                                            │
        │  transpose 回去: (2, 12, 4, 64) → (2, 4, 12, 64)         │
        │  view 拼回:      (2, 4, 12, 64) → (2, 4, 768)            │
        │                                                            │
        │  12 个头的 64 维结果按顺序接起来，恢复 768 维。            │
        │                                                            │
        │  最后用 w_o(768→768) 融合:                                 │
        │    12 个头的信息简单拼接还不够，需要让信息"交流"。         │
        │    w_o 就像主编，把 12 个记者的稿子综合成一篇。            │
        │                                                            │
        │  输出: (2, 4, 768) —— 形状和输入一样，但内容融合了上下文  │
        └────────────────────────────────────────────────────────────┘
        """
        batch_size = query.size(0)

        # ═══ Step 1: 线性投影 —— 把输入变成 Q/K/V 三种角色 ═══
        # 三个不同的 Linear 层，各自学习不同的投影方式
        # 输入都是 (B, seq, 768)，输出也是 (B, seq, 768)，但语义不同
        q = self.w_q(query)   # "提问者"视角
        k = self.w_k(key)     # "被查询者"视角
        v = self.w_v(value)   # "内容提供者"视角

        # ═══ Step 2: 拆成多头 ═══
        # Q 拆成 n_heads 个: (B, seq, 768) → (B, seq, n_heads, d_k)
        # K/V 拆成 n_kv_heads 个: (B, seq, 768) → (B, seq, n_kv_heads, d_k)
        #   标准 MHA: n_kv_heads = n_heads = 12
        #   GQA: n_kv_heads = 3, 只有 3 套 K/V，后续重复 4 次匹配 Q
        q = q.view(batch_size, -1, self.n_heads, self.d_k)
        k = k.view(batch_size, -1, self.n_kv_heads, self.d_k)
        v = v.view(batch_size, -1, self.n_kv_heads, self.d_k)

        # ═══ Step 3: transpose —— 让 12 个头变成 12 个独立任务 ═══
        #
        # 先理解一个关键问题: 为什么 q 里有 seq 维?
        # 不是"每个头只有一个 q 向量"，而是"每个头对句子里每个词都有一个 q 向量"。
        #
        # 【本例参数】embedding=8, 2头, 每头4维(d_k=4), 句子3个词(seq=3)
        #   注意: seq=3(词数) 和 d_k=4(每头维度) 是两个不相干的数字!
        #
        #   q (3个词, 每词4维) —— 每行是一个词的4维查询向量:
        #       ┌───────────┐
        #  猫   │ a  b  c  d │   ← 词0的查询(4个数)
        #  抓   │ e  f  g  h │   ← 词1的查询(4个数)
        #  鼠   │ i  j  k  l │   ← 词2的查询(4个数)
        #       └───────────┘
        #       形状: (3, 4)  ← 3行(3个词), 4列(每词4维)
        #
        #   k (3个词, 每词4维), 转置后变成 k^T (4行, 3列):
        #       ┌───────────┐
        #       │ m  n  o   │   ← 第0列是"猫"的键(4个数)
        #       │ p  q  r   │   ← 第1列是"抓"的键(4个数)
        #       │ s  t  u   │   ← 第2列是"鼠"的键(4个数)
        #       │ v  w  x   │
        #       └───────────┘
        #       形状: (4, 3)  ← 4行(d_k=4), 3列(3个词)
        #
        #   matmul: q(3,4) @ k^T(4,3) = (3,3):
        #
        #   为什么 4 不见了? 先看最简单的情况 —— 1行q和1列k^T:
        #
        #       q(1,4)              k^T(4,1)
        #       ┌───────┐           ┌───┐
        #       │a b c d│     @     │ m │    =    [a·m+b·n+c·o+d·p]
        #       └───────┘           │ n │           (1,1)
        #                           │ o │
        #                           │ p │
        #                           └───┘
        #
        #   4 维向量变成了 1 个数! 因为矩阵乘法是"对应相乘再相加"。
        #   中间的 4 被加总压缩成了 1 个标量。
        #
        #   扩展到完整的 (3,4) @ (4,3):
        #       ┌──────────────────────────────────────────────┐
        #  猫   │a·m+b·p+c·s+d·v  a·n+b·q+c·t+d·w  a·o+b·r+c·u+d·x│
        #  抓   │e·m+f·p+g·s+h·v  e·n+f·q+g·t+h·w  e·o+f·r+g·u+h·x│
        #  鼠   │i·m+j·p+k·s+l·v  i·n+j·q+k·t+l·w  i·o+j·r+k·u+l·x│
        #       └──────────────────────────────────────────────┘
        #       形状: (3, 3)
        #       → 3行(来自q的行数), 3列(来自k^T的列数)
        #       → 中间的 4 在每次乘法里都被加总消掉了
        #
        #   完整形状: (B, 2, 3, 3) —— B个句子, 2个头, 每头一个3×3注意力矩阵
        #
        #   每个分数 = 4个字母两两相乘再相加 (点积)
        #   例: 猫→抓 = a·n + b·q + c·t + d·w
        #   这就是"猫"的查询向量和"抓"的键向量的相似度!
        #
        #  ┌─ 不 transpose 时的问题 ─────────────────────┐
        #  │  q:     (B, seq, 2, 4)                     │
        #  │  k^T:   (B, seq, 4, 2)                     │
        #  │  matmul 最后两维: (2, 4) @ (4, 2)          │
        #  │           = (2, 2) "头×头" ← 错了!         │
        #  └────────────────────────────────────────────┘
        #
        #  ┌─ transpose 后 ─────────────────────────────┐
        #  │  q:     (B, 2, seq, 4)                     │
        #  │  k^T:   (B, 2, 4, seq)                     │
        #  │  matmul 最后两维: (seq, 4) @ (4, seq)      │
        #  │           = (seq, seq) "词×词" ← 对的!     │
        #  │  前面 (B, 2) 批处理，2 个头同时算          │
        #  └────────────────────────────────────────────┘
        #
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ═══ Step 3: 可选 RoPE 位置编码 ═══
        # 必须在拆多头之后、注意力计算之前旋转
        # 只对 q 和 k 旋转，v 不需要位置信息
        if self.use_rope:
            q, k = self.rope(q, k, offset=rope_offset)

        # ═══ Step 4: KV-Cache (推理加速用) ═══
        # 生成新 token 时，不需要重新计算之前所有 token 的 K 和 V
        # 把之前缓存的 K/V 和新的 K/V 拼起来
        # 注意: KV-Cache 保存的是原始 n_kv_heads 的 K/V，不重复
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)  # 沿 seq_len 维度拼接
            v = torch.cat([v_prev, v], dim=2)

        # 保存原始 K/V 用于 KV-Cache（必须存 repeat 之前的 n_kv_heads 版本）
        k_raw, v_raw = k, v

        # ═══ Step 4.5: GQA — K/V 重复 ═══
        # GQA 时 K/V 只有 n_kv_heads 个，需要重复 n_rep 次才能和 Q 的 n_heads 匹配
        # 例: n_kv_heads=3, n_heads=12 → n_rep=4
        #   k: (B, 3, seq, 64) → repeat_interleave(dim=1, repeats=4) → (B, 12, seq, 64)
        #   即 [K0, K1, K2] → [K0,K0,K0,K0, K1,K1,K1,K1, K2,K2,K2,K2]
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # ═══ Step 5: 调整 mask 维度 ═══
        # mask 从 (B, seq_q, seq_k) 变成 (B, 1, seq_q, seq_k)
        # 中间的 1 会自动广播到所有 12 个头
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # ═══ Step 6: 核心注意力计算 ═══
        #
        # 接上例: 2个头, seq=3, d_k=4, 输入经过transpose后:
        #   q: (1, 2, 3, 4)   2个头, 每个头3个词, 每个词4维查询
        #   k: (1, 2, 3, 4)   同上
        #   v: (1, 2, 3, 4)   同上
        #
        # scaled_dot_product_attention 内部:
        #   ① q @ k^T: (1, 2, 3, 4) @ (1, 2, 4, 3) = (1, 2, 3, 3)
        #      → 2个头, 每个头一个 3×3 注意力分数矩阵
        #   ② softmax → 概率和为1的权重
        #   ③ attn_weights @ v: (1, 2, 3, 3) @ (1, 2, 3, 4) = (1, 2, 3, 4)
        #      ↑↑↑↑
        #      (seq,seq) @ (seq,d_k) = (seq,d_k)
        #      → 3×3 的权重矩阵 × 3×4 的值矩阵 = 3×4 的输出
        #      → 每个词的输出 = 所有词的值按关注度加权混合
        #
        # 注意力后输出: (1, 2, 3, 4) —— 和输入 q 的形状一样!
        #   但这已经是"混合了上下文信息"的新表示
        # q @ k^T: (1, 2, 3, 4) @ (1, 2, 4, 3) → (1, 2, 3, 3)
        # attn_weights @ v: (1, 2, 3, 3) @ (1, 2, 3, 4) → (1, 2, 3, 4) ← 变回 (seq, d_k)

        attn_output = self.scaled_dot_product_attention(q, k, v, mask)

        # ═══ Step 7: 拼回多头结果 ═══
        #
        # 现在要把 (B, 2, 3, 4) 还原回 (B, 3, 8) [d_model=8]
        #
        # ① transpose: (1, 2, 3, 4) → (1, 3, 2, 4)
        #    把头维(2)和seq维(3)换回来
        #
        #    可视化 (B=1, seq=3, 2个头, d_k=4):
        #      词0 → [头0的4维结果, 头1的4维结果]
        #      词1 → [头0的4维结果, 头1的4维结果]
        #      词2 → [头0的4维结果, 头1的4维结果]
        #
        # ② view: (1, 3, 2, 4) → (1, 3, 8)
        #    把最后两维 2×4=8 拼起来，恢复 d_model=8
        #    每个词从"2个头各4维"变成"一个8维向量"
        #
        # 完整维度流转图 (本例: 2头, d_k=4, d_model=8):
        #  输入:        (B, 3, 8)
        #    view:      (B, 3, 2, 4)      ← 拆2头 (3个词, 每词拆成2头×4维)
        #    transpose: (B, 2, 3, 4)      ← 头提到前面
        #    注意力:    (B, 2, 3, 4)      ← 混合上下文 (形状不变, 内容变了)
        #    transpose: (B, 3, 2, 4)      ← seq提到前面
        #    view:      (B, 3, 8)         ← 拼回8维
        #
        # 实际模型中对应 (12头, d_k=64, d_model=768):
        #  (B, seq, 768) → (B, seq, 12, 64) → (B, 12, seq, 64)
        #                → 注意力 → (B, 12, seq, 64) → (B, seq, 12, 64) → (B, seq, 768)
        attn_output = attn_output.transpose(1, 2)
        # contiguous(): transpose 后内存不连续，view 前需要整理内存
        attn_output = attn_output.contiguous()
        # view: 把 (B, seq, 12, 64) 拼回 (B, seq, 768)
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # ═══ Step 8: 输出投影 —— 融合各头信息 ═══
        # w_o 让每个位置的信息在各头之间"交流"，不是简单拼接
        # 同时返回 (k, v) 用于 KV-Cache
        # 输出形状: (B, seq, 768) —— 和输入一样，但内容融合了上下文
        # 返回 k_raw/v_raw（repeat 之前的 n_kv_heads 版本），保证下次拼接 shape 匹配
        return self.w_o(attn_output), (k_raw, v_raw)
