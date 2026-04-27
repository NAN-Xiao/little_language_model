"""
稀疏 MoE 前馈层 — 把 1 个大 FFN 拆成多个小"专家"
====================================================

普通 FFN: 每个 token 都过同一个 768→3072→768 的网络
MoE FFN:  有 4 个 768→3072→768 的专家, 每个 token 只过其中 1 个

好处: 模型总参数多了 4 倍, 但每次计算量不变 (每个 token 只算 1 个专家)
就像医院有 4 个专科医生, 病人按症状挂号, 每个病人只看 1 个医生。

默认配置: d_model=768, d_ff=3072, num_experts=4
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feedforward import PositionwiseFeedForward


class MoEFeedForward(nn.Module):
    """
    Switch 风格 top-1 路由 MoE —— "专家会诊"系统。

    ┌──────────────────────────────────────────────────────────────────┐
    │  生活类比: 医院挂号系统                                          │
    │                                                                  │
    │  普通 FFN 像全科医生: 感冒、骨折、胃病都看同一个人              │
    │  MoE 像专科医院: 内科、外科、骨科、儿科各司其职                 │
    │                                                                  │
    │  病人(token)来了, 先在导诊台(router)描述症状:                  │
    │    "我咳嗽发烧" → 导诊判断: 内科(0.8), 儿科(0.15), ...       │
    │    → 挂号内科, 只让内科医生看                                   │
    │                                                                  │
    │  4 个专家各有专长:                                               │
    │    专家0: 擅长处理"名词类"token (如"猫""狗")                  │
    │    专家1: 擅长处理"动词类"token (如"抓""跑")                  │
    │    专家2: 擅长处理"形容词类"token (如"大""红")                │
    │    专家3: 擅长处理"虚词类"token (如"的""了")                  │
    │                                                                  │
    │  注意: 专家的分工不是人为指定的, 是训练自动学出来的!            │
    └──────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────┐
    │  完整维度流转 (B=2, seq=5, d_model=768, 4个专家):              │
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
        d_model: int,      # 输入维度 (768)
        d_ff: int,         # FFN 中间维度 (3072=768×4)
        num_experts: int,  # 专家数量 (默认4)
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts  # 专家总数
        self.d_model = d_model          # 每个 token 的维度

        # Router: 导诊台, 决定每个 token 去哪个专家
        # Linear(768 → 4): 输入一个 token 的 768 维向量, 输出 4 个分数
        self.router = nn.Linear(d_model, num_experts)

        # Experts: 4 个独立的 FFN 专家
        # 每个专家结构和普通 FFN 一样: 768→3072→768
        self.experts = nn.ModuleList(
            PositionwiseFeedForward(d_model, d_ff, dropout)
            for _ in range(num_experts)
        )

        # 上一次 forward 计算的负载均衡损失, 供外部收集
        self.last_lb_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MoE 前向传播 —— 每个 token 路由到最合适的专家。

        ┌──────────────────────────────────────────────────────────────┐
        │  输入 x: (B, seq, d_model) = (2, 5, 768)                    │
        │                                                              │
        │  完整流程:                                                   │
        │                                                              │
        │  1. 展平: (2, 5, 768) → (10, 768)                           │
        │     把 batch 和 seq 合并, 变成"10 个独立的 token"           │
        │                                                              │
        │  2. 路由打分: router(flat) → (10, 4)                        │
        │     每个 token 得到 4 个专家的分数                          │
        │                                                              │
        │  3. softmax: (10, 4) → 概率和为1的分布                      │
        │     [2.1, -0.5, 0.8, -1.2] → [0.72, 0.04, 0.22, 0.02]     │
        │                                                              │
        │  4. argmax: 每个 token 选概率最高的专家                     │
        │     [0, 2, 0, 1, 2, 1, 3, 0, 2, 3]                         │
        │                                                              │
        │  5. 逐个专家计算:                                            │
        │     专家0收到 token [0,2,7] → 一起算 FFN → 写回位置 [0,2,7]│
        │     专家1收到 token [3,5]   → 一起算 FFN → 写回位置 [3,5]  │
        │     ...                                                      │
        │                                                              │
        │  6. 还原形状: (10, 768) → (2, 5, 768)                       │
        │     拼回原来的 batch × seq 结构                              │
        │                                                              │
        │  7. 计算负载均衡损失 (只训练时, 不增加计算量)               │
        └──────────────────────────────────────────────────────────────┘
        """
        b, t, d = x.shape  # b=batch, t=seq_len, d=d_model

        # ═══ Step 1: 展平 ═══
        # (B, seq, 768) → (B×seq, 768)
        # 把"2个样本各5个词"变成"10个独立的词"
        # 这样每个词可以独立选择专家, 不用管它在哪个样本里
        flat = x.reshape(b * t, d)

        # ═══ Step 2: 路由打分 ═══
        # router: Linear(768 → 4)
        # 每个 token 的 768 维向量 → 4 个分数 (对4个专家的偏好)
        router_logits = self.router(flat)  # (B×seq, num_experts)

        # softmax: 分数变概率, 每行和为1
        # 例: [2.1, -0.5, 0.8, -1.2] → [0.72, 0.04, 0.22, 0.02]
        #     总和 = 0.72+0.04+0.22+0.02 = 1.0
        probs = F.softmax(router_logits, dim=-1)

        # argmax: 每个 token 只选概率最高的那个专家 (top-1)
        # idx[i] = 0~3, 表示第 i 个 token 分配给哪个专家
        idx = torch.argmax(probs, dim=-1)  # (B×seq,)

        # ═══ Step 3: 每个 token 过对应的专家 ═══
        # 初始化输出缓冲区, 形状和 flat 一样
        out = torch.zeros_like(flat)

        for e in range(self.num_experts):
            # mask: 哪些 token 分给了专家 e
            # 例: mask = [True, False, True, False, ...]
            #     表示 token0 和 token2 分给了专家 e
            mask = idx == e

            if mask.any():  # 至少有一个 token 分给这个专家才计算
                # flat[mask]: 取出分给专家 e 的所有 token
                # 形状: (N_e, 768), N_e 是分给专家 e 的 token 数
                sub = flat[mask]

                # unsqueeze(1): (N_e, 768) → (N_e, 1, 768)
                #   FFN 内部是 Linear, 期望输入 (B, seq, d) 或 (B, d)
                #   加一维让它变成"N_e 个样本, 每个 1 个 token"
                sub = sub.unsqueeze(1)

                # 专家 e 的 FFN 计算
                # FFN: (N_e, 1, 768) → (N_e, 1, 768)
                expert_out = self.experts[e](sub)

                # squeeze(1): (N_e, 1, 768) → (N_e, 768)
                #   去掉多余的维度
                expert_out = expert_out.squeeze(1)

                # 写回输出缓冲区对应位置
                out[mask] = expert_out

        # ═══ Step 4: 负载均衡损失 (Load Balance Loss) ═══
        #
        # 问题: 如果 router 学到"所有 token 都选专家0"，
        #       专家1/2/3 永远不会被训练，等于白占了参数。
        #
        # 解决: 加一个辅助损失，鼓励 token 均匀分配到各专家。
        #
        # 公式: LB = num_experts × Σ(f_e × P_e)
        #
        #   f_e = 实际分到专家e的token比例 (不可导, 只是统计)
        #         例: 10个token, 3个去专家0 → f_0 = 0.3
        #
        #   P_e = router 对专家e的平均概率 (可导, 影响梯度)
        #         例: 所有token对专家0的平均softmax概率
        #
        # 为什么这样设计?
        #   - 如果 token 都挤到专家0: f_0=1.0, P_0≈1.0 → LB 很大
        #   - 如果均匀分配: f_e=0.25, P_e=0.25 对每个e → LB = 4×4×(0.25×0.25) = 1.0 (最小)
        #
        # 为什么 f_e 要 detach(不可导)?
        #   f_e 是"实际分配结果"，router 只能控制概率 P_e，
        #   不能控制 argmax 后的硬分配。detach 让梯度只流回 P_e。

        # counts: 每个专家的实际分配比例
        # [(idx==0).mean(), (idx==1).mean(), ...]
        counts = torch.stack(
            [(idx == e).float().mean() for e in range(self.num_experts)]
        )

        # P_mean: 每个专家的平均路由概率
        # probs.mean(dim=0): 对所有token取平均, 形状 (num_experts,)
        P_mean = probs.mean(dim=0)

        # 负载均衡损失
        lb = self.num_experts * (counts.detach() * P_mean).sum()
        self.last_lb_loss = lb

        # ═══ Step 5: 还原形状 ═══
        # (B×seq, 768) → (B, seq, 768)
        return out.view(b, t, d)


def collect_moe_load_balance_loss(model: nn.Module) -> torch.Tensor:
    """
    收集模型中所有 MoE 层的负载均衡损失, 汇总后加到总 loss 上。

    为什么需要这个函数?
      模型可能有多个 MoE 层(每层 DecoderBlock 各一个),
      每层都有自己的 last_lb_loss。
      训练时需要把所有层的 LB loss 加起来, 乘以系数, 一起反向传播。

    用法 (在训练循环中):
      loss = cross_entropy(logits, targets)
      if model.cfg.use_moe:
          lb_loss = collect_moe_load_balance_loss(model)
          loss = loss + model.cfg.moe_lb_coeff * lb_loss
      loss.backward()

    参数:
        model: 整个 Transformer 模型

    返回:
        所有 MoE 层 LB loss 的总和 (标量)
    """
    # 获取模型所在设备 (cuda 或 cpu)
    device = next(model.parameters()).device

    # 累加所有 MoE 层的负载均衡损失
    total = torch.zeros((), device=device, dtype=torch.float32)
    for m in model.modules():
        if isinstance(m, MoEFeedForward) and m.last_lb_loss is not None:
            total = total + m.last_lb_loss.to(device=device, dtype=total.dtype)
    return total
