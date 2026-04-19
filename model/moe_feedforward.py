"""
稀疏 MoE 前馈层 — 把 1 个大 FFN 拆成多个小 FFN, 每个 token 只用 1 个
======================================================================

普通 FFN: 每个 token 都过同一个 768→3072→768 的网络
MoE FFN:  有 4 个 768→3072→768 的专家, 每个 token 只过其中 1 个

好处: 模型总参数多了 4 倍, 但每次计算量不变 (每个 token 只算 1 个专家)

默认配置: d_model=768, d_ff=3072, num_experts=4
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feedforward import PositionwiseFeedForward


class MoEFeedForward(nn.Module):
    """
    Switch 风格 top-1 路由 MoE。

    ┌──────────────────────────────────────────────────────────────────┐
    │  完整流程 (B=2, seq=5, d_model=768, 4个专家):                  │
    │                                                                  │
    │  输入: (2, 5, 768) = 10 个 token                                │
    │                                                                  │
    │  ① 展平: (2,5,768) → (10, 768)                                 │
    │    方便逐 token 处理, 不用管 batch 边界                          │
    │                                                                  │
    │  ② Router 路由: Linear(768 → 4)                                 │
    │    每个 token 过一个线性层, 输出 4 个分数                       │
    │    flat: (10, 768)                                               │
    │    router_logits: (10, 4) — 每个 token 对 4 个专家的打分        │
    │                                                                  │
    │    数值示例 (某个 token):                                        │
    │      logits = [2.1, -0.5, 0.8, -1.2]                            │
    │      softmax → [0.72, 0.04, 0.22, 0.02]                         │
    │      argmax → 0 号专家 (概率 72% 最高)                          │
    │                                                                  │
    │  ③ 每个 token 只用 1 个专家:                                    │
    │                                                                  │
    │    假设 10 个 token 的分配结果:                                  │
    │      token 0,3,7 → 专家0 (3个)                                  │
    │      token 1,5   → 专家1 (2个)                                  │
    │      token 2,4,8 → 专家2 (3个)                                  │
    │      token 6,9   → 专家3 (2个)                                  │
    │                                                                  │
    │    对每个专家, 只算分给它的 token:                               │
    │      专家0: flat[0,3,7] → FFN → out[0,3,7]                     │
    │      专家1: flat[1,5]   → FFN → out[1,5]                       │
    │      专家2: flat[2,4,8] → FFN → out[2,4,8]                     │
    │      专家3: flat[6,9]   → FFN → out[6,9]                       │
    │                                                                  │
    │  ④ 还原: (10, 768) → (2, 5, 768)                               │
    │    每个 token 都有了自己专家的输出, 拼回原形状                  │
    │                                                                  │
    │  ⑤ 负载均衡损失 (训练时):                                       │
    │    如果所有 token 都选专家0, 专家1/2/3 白浪费了                │
    │    LB loss 鼓励 token 均匀分配到各专家                          │
    │    f_e = 分到专家e的token比例 (detached, 不可导)                │
    │    P_e = router对专家e的平均概率 (可导)                         │
    │    LB = E × Σ(f_e × P_e) — 越均匀越小, 最小值=1              │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.router = nn.Linear(d_model, num_experts)  # (768 → 4) 路由器
        self.experts = nn.ModuleList(
            PositionwiseFeedForward(d_model, d_ff, dropout) for _ in range(num_experts)
        )
        self.last_lb_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, seq, d_model) → (B, seq, d_model)
        """
        b, t, d = x.shape

        # (2, 5, 768) → (10, 768)
        flat = x.reshape(b * t, d)

        # 路由: (10, 768) → (10, 4) → softmax → (10, 4)
        router_logits = self.router(flat)
        probs = F.softmax(router_logits, dim=-1)

        # 每个 token 选 1 个专家: (10,)
        idx = torch.argmax(probs, dim=-1)

        out = torch.zeros_like(flat)
        for e in range(self.num_experts):
            mask = idx == e  # 哪些 token 分给了专家 e
            if mask.any():
                # unsqueeze(1): (N_e, 768) → (N_e, 1, 768)
                #   FFN 内部 Linear 对最后一维做变换, 需要保留 seq 维
                # squeeze(1): (N_e, 1, 768) → (N_e, 768)
                #   去掉多余的 seq=1 维
                sub = flat[mask].unsqueeze(1)
                out[mask] = self.experts[e](sub).squeeze(1)

        # 负载均衡损失
        counts = torch.stack(
            [(idx == e).float().mean() for e in range(self.num_experts)]
        )
        P_mean = probs.mean(dim=0)
        lb = self.num_experts * (counts.detach() * P_mean).sum()
        self.last_lb_loss = lb

        return out.view(b, t, d)


def collect_moe_load_balance_loss(model: nn.Module) -> torch.Tensor:
    """
    累加所有 MoE 层的负载均衡损失, 加入总 loss 反向传播。
    防止所有 token 都挤到同 1 个专家, 导致其他专家白训了。
    """
    device = next(model.parameters()).device
    total = torch.zeros((), device=device, dtype=torch.float32)
    for m in model.modules():
        if isinstance(m, MoEFeedForward) and m.last_lb_loss is not None:
            total = total + m.last_lb_loss.to(device=device, dtype=total.dtype)
    return total
