"""稀疏 MoE 前馈层（Switch 风格 top-1 路由）。

将 Transformer Block 中的稠密 PositionwiseFeedForward 替换为：
  router(x) -> softmax -> argmax 选 1 个专家；仅对该 token 计算对应专家 FFN。

训练时需加负载均衡辅助损失（Switch Transformer）：
  L_aux = E * sum_e f_e * P_e
其中 f_e 为硬路由分配到专家 e 的 token 比例（detach），P_e 为路由器对专家 e 的平均概率（可反传）。
"""

from __future__ import annotations

import torch

"""
# 导入 nn 中常用的模块：
# nn.Linear           # 线性变换层，全连接层
# nn.Module           # 所有神经网络模块的基类
# nn.ModuleList       # 用于存放子模块列表，自动注册参数
# nn.Dropout          # Dropout 随机失活层，用于防止过拟合
# nn.BatchNorm1d      # 一维批归一化，经常用于序列建模
# nn.LayerNorm        # 层归一化，Transformer里常用
# nn.ReLU             # ReLU 激活函数
# nn.GELU             # GELU 激活函数，Transformer里常用
# nn.Softmax          # Softmax 层，一般用于多分类输出
# nn.CrossEntropyLoss # 交叉熵损失，常用于分类任务
# nn.Identity         # 恒等操作，什么都不做，结构上占位用
"""
import torch.nn as nn
# F 是 torch.nn.functional 的简写，包含了大量常用的神经网络函数/操作。
# 常见模块和方法有：
# F.relu           # ReLU 激活函数
# F.gelu           # GELU 激活函数
# F.softmax        # Softmax 函数，常用于分类概率输出
# F.cross_entropy  # 交叉熵损失函数，适用于分类任务
# F.linear         # 线性变换（全连接），手动实现 y = x @ W^T + b
# y = x @ W^T + b是线性变换的公式，x是输入，W是权重，b是偏置，@是矩阵乘法，^T是转置。
# F.dropout        # Dropout 随机失活
# F.log_softmax    # 对数Softmax，多用于结合NLLLoss
# F.layer_norm     # 层归一化
# F.batch_norm     # 批归一化
# F.pad            # 各类pad操作
# F.one_hot        # one-hot编码
# 还有其它如normalize、binary_cross_entropy、mse_loss等

import torch.nn.functional as F

from .feedforward import PositionwiseFeedForward


# nn.Module是PyTorch中的一个基类，所有的神经网络模块都应该继承自nn.Module。
# nn是
class MoEFeedForward(nn.Module):
    """每个 token 只用 1 个专家，结构其实和普通FFN是一样的形式"""

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
        """router用于推理：决定每个token分给哪个专家"""
        self.router = nn.Linear(d_model, num_experts)
        """
        ModuleList就是专家网络们，每个其实就是普通FFN一套（前馈）
        """
        self.experts = nn.ModuleList(
            PositionwiseFeedForward(d_model, d_ff, dropout) for _ in range(num_experts)
        )
        # last_lb_loss 只在训练需要
        self.last_lb_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MoE（Mixture of Experts，稀疏专家混合）选择流程如下：

        1. 对每个 token，先经过 Router 网络（一个线性层），得到路由分数 logits，形状为 (batch*seq, num_experts)。
        2. 对路由分数做 softmax，得到每个专家的概率分布（概率和为1），表示每个 token 选择每个专家的可能性。
        3. 对每个 token，选择概率最高的专家（torch.argmax，选索引），每个 token 只分配给一个专家。
        4. 按照分配，将所有 token 送到其对应的专家前馈网络，再将每个 token 的输出结果收集/还原到原顺序。
        5. （训练时）额外计算每个专家的负载均衡辅助损失，鼓励 token 能更均匀地分配到各个专家。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch, seq, d_model)
        返回值:
            torch.Tensor: 输出张量，形状为 (batch, seq, d_model)
        """
        """
        x 形状: (batch, seq, d_model)
        batch是批量大小，多少个样本。
        seq是序列长度，输入了多少个token。
        d_model是输入每个token的特征维度。
        """
        b, t, d = x.shape

        """推理用：将batch、seq合并便于并行处理所有token"""
        flat = x.reshape(b * t, d)

        """
        router前向获得每个token的专家路由得分 (batch*seq, num_experts)logits的中文含义是 路由得分
        本质也是一个线性变换
        router_logits的形状是(batch*seq, num_experts)，表示每个token分配到每个专家的概率。
        """
        router_logits = self.router(flat)

        """
        softmax获得所有专家归一化概率 (batch*seq, num_experts)
        # flat最后一个维度是d_model，是因为每个token的向量都是d_model维度（即特征维度），我们把(batch, seq, d_model)拉平成(batch*seq, d_model)后，
        # 每一行依然代表一个token的d_model维度向量。这是因为专家门控只作用在token的路由分配上，而专家网络本身处理的仍然需要完整token的d_model维度输入。
        # 这里的probs是在router_logits上按最后一维（num_experts）做softmax得到的，
        # 意味着对于每一个token（也就是flat的每一行），probs那一行是该token分配给每个专家的概率分布（长度num_experts，总和为1）。
        """
        probs = F.softmax(router_logits, dim=-1)

        """推理用：获取每个token分配到的专家编号 (batch*seq,)
        idx的形状是(batch*seq,)，表示每个token分配到的专家编号。
        """
        idx = torch.argmax(probs, dim=-1)

        """推理用：初始化输出，形状同flat"""
        out = torch.zeros_like(flat)
        """
        # 这个 for 循环实现了如下功能：
        # 对于每一个专家 e，筛选所有被路由分配到专家 e 的 token（mask 为 True）。
        # 将这些 token（flat[mask]，形状为 (N_e, d_model)）送入第 e 个专家网络（一个标准 FFN）。
        # 得到第 e 个专家为自己的 token 计算的输出（形状为 (N_e, d_model)），
        # 并通过 out[mask] = ... 将每个 token 的结果还原放回原本在 flat 张量里的位置（即 batch*seq, d_model）。
        # 这样所有专家并行独立处理各自份额的 token，最终 out 存储与输入 flat 一一对应的推理结果。
        # 返回值：循环本身不返回值，但会将每个 token 按各自专家处理后的输出写入 out（形状与 flat 一致）。
        # flat的形状是(batch*seq, d_model)，表示所有token的输入。
        """
        for e in range(self.num_experts):
            #= ==判断idx是否等于e，如果等于e，则mask为True，否则为False。
            mask = idx == e
            if mask.any():
                #unsqueeze
                # unsqueeze(1)是为了将token的输入变成(batch*seq, 1, d_model)，因为专家网络的输入是(batch*seq, 1, d_model)，方便送入专家网络。
                # squeeze(1)是为了将专家网络的输出变成(batch*seq, d_model)，方便送入下一个模块。
                sub = flat[mask].unsqueeze(1)
                out[mask] = self.experts[e](sub).squeeze(1)

        # 注意：for结束后会直接return，不会有异步或延迟。推理时这里已经把所有专家的token都处理并填回out，
        # 接下来的负载均衡loss只在训练时有用（即torch.no_grad下可以忽略这个loss计算，也可以保留无害）。
        counts = torch.stack(
            [(idx == e).float().mean() for e in range(self.num_experts)]
        )
        P_mean = probs.mean(dim=0)
        lb = self.num_experts * (counts.detach() * P_mean).sum()
        self.last_lb_loss = lb

        return out.view(b, t, d)


def collect_moe_load_balance_loss(model: nn.Module) -> torch.Tensor:
    """累加模型中所有 MoEFeedForward 在上一次 forward 中写入的负载项。
    参数:
        model (nn.Module): 模型对象，包含 MoEFeedForward 层
    返回值:
        torch.Tensor: 累加的负载均衡损失
        # 本函数用于遍历给定模型中的所有 MoEFeedForward 层，将它们上一次 forward 时计算并存储的负载均衡辅助损失（last_lb_loss）取出并累加，返回所有 MoE 层的负载均衡损失之和。
        # 这样可以方便地将整个模型多层 MoE 的负载均衡损失作为一项辅助loss，统一加入总loss中进行优化，
        # 从而鼓励每个专家被均匀调用，提升专家利用率，避免部分专家分配token过多、过少造成冗余或浪费。
        # 注意该函数依赖于每个 MoEFeedForward 在forward过程中写入 last_lb_loss，仅在训练涉及MoE负载均衡时调用。
        # 典型用法是在主训练循环内累积所有MoE负载损失，加总进主loss共同反向传播。
    """
    device = next(model.parameters()).device
    total = torch.zeros((), device=device, dtype=torch.float32)
    for m in model.modules():
        if isinstance(m, MoEFeedForward) and m.last_lb_loss is not None:
            total = total + m.last_lb_loss.to(device=device, dtype=total.dtype)
    return total
