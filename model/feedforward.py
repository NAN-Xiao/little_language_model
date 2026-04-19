"""
前馈网络 (FFN) — 每个 token 独立地做非线性变换
==============================================

注意力负责 token 之间的信息交流, FFN 负责 token 自身的特征变换。
两者交替堆叠, 构成 Transformer 的基本结构。

默认配置: d_model=768, d_ff=3072 (d_model 的 4 倍)
"""

import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    两层 FFN: 升维 → 激活 → 降维。

    ┌──────────────────────────────────────────────────────────────────┐
    │  为什么先升维再降维?                                            │
    │                                                                  │
    │  d_model=768 → d_ff=3072 → d_model=768                         │
    │                                                                  │
    │  类比: 一个 768 维的向量, 用 3072 维的中间空间去"展开"它,      │
    │  在高维空间里做非线性变换 (ReLU 会砍掉一半神经元),              │
    │  再投影回 768 维。                                              │
    │                                                                  │
    │  为什么不直接 768→768?                                          │
    │    一个线性层只能做仿射变换 (旋转+平移), 没有非线性            │
    │    升维后经过 ReLU 激活, 就有了非线性, 表达能力大幅增强        │
    │    中间维度越大, 能学到的非线性模式越丰富                       │
    │                                                                  │
    │  完整维度流转 (B=2, seq=5):                                     │
    │                                                                  │
    │  输入: (2, 5, 768)                                              │
    │    │                                                              │
    │    ▼ Linear(768 → 3072): 矩阵乘法, 每个token独立变换           │
    │  (2, 5, 3072) — 升维 4 倍                                       │
    │    │                                                              │
    │    ▼ ReLU: 把负数变0, 引入非线性                                │
    │  (2, 5, 3072) — 形状不变, 但约一半值变成0                       │
    │    │                                                              │
    │    ▼ Dropout: 随机丢弃部分神经元, 防过拟合                      │
    │  (2, 5, 3072)                                                    │
    │    │                                                              │
    │    ▼ Linear(3072 → 768): 降维回来                               │
    │  (2, 5, 768)                                                    │
    │    │                                                              │
    │    ▼ Dropout                                                     │
    │  (2, 5, 768) — 输出, 和输入形状相同                            │
    │                                                                  │
    │  注意: FFN 对每个 token 独立做同样的变换                        │
    │    token 之间的交互在注意力层完成, FFN 只做"单个token的加工"   │
    │    所以叫 "Position-wise" — 逐位置, 不跨位置                   │
    └──────────────────────────────────────────────────────────────────┘

    参数:
        d_model: 输入输出维度 (768)
        d_ff: 中间维度 (3072, 通常是 d_model 的 4 倍)
        dropout: dropout 比例
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 两层 FFN: 升维 → 激活 → 降维
        #  Sequential 按顺序串联多个层, 输入依次通过每层处理, 输出最后一层结果。
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),   # (768 → 3072) 升维
            nn.ReLU(),                  # 非线性激活
            nn.Dropout(dropout),        # 防过拟合
            nn.Linear(d_ff, d_model),   # (3072 → 768) 降维
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """(B, seq, d_model) → (B, seq, d_model)"""
        return self.net(x)
