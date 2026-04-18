"""LoRA（Low-Rank Adaptation，低秩适应）参数高效微调方法的实现。

核心思想：对于预训练权重 W（d_out x d_in），LoRA 通过加入低秩更新实现：
    W' = W + (alpha / r) * B @ A
其中，A（r x d_in）和 B（d_out x r）为唯一需要训练的参数。
B 初始为零，这样模型初始化时等价于原始行为。
"""
# LoRA 结构与传统全连接层（以 nn.Linear 为例）的详细对比与注释如下：
#
#  ┌────────────┐                ┌───────────────────────┐
#  │   输入 x   │                │   输入向量 x          │
#  └─────┬──────┘                └─────────────┬─────────┘
#        |                                   ┌─┴─┐
#        |                                  /     \
#  ┌─────────────┐                  ┌──────┴───────┐    ┌─────────────────────┐
#  │   Base分支  │                  │   LoRA分支   │    │         注释        │
#  │ （原始Linear）│                  │ （低秩适应） │    │                     │
#  └─────┬───────┘                  └──────┬───────┘    └─────────────────────┘
#        |                                      |
#        v                                      v
#    x @ W^T + bias            ┌─────────────“升维”─────────────┐
#   （原始输出）              │ lora_A: shape=(r, d_in)       │
#                             │ x @ A^T = x @ lora_A.T        │
#                             └───────────────────────────────┘
#                                                   |
#                                        ┌─────“降维”───────┐
#                                        │ lora_B: (d_out, r) │
#                                        │ (x @ A^T) @ B^T    │
#                                        └────────────────────┘
#                                                   |
#                                    (alpha/r) * (x @ A^T @ B^T)   ← LoRA分支（缩放）
#
#  总输出 y = 原始分支 + LoRA部分（低秩适应）
#  -------------------------------------------------------------------------------
#  y = x @ W^T + bias
#    + (alpha / r) * [ x @ lora_A^T @ lora_B^T ]
#                         |         |
#                     升维A^T    降维B^T
#
#  详细流程说明：
#    1. Base分支：x 直接经过原始全连接层 W（参数冻结），即 (x @ W^T + bias)。
#    2. LoRA分支：
#        ├─(a) 首先通过A矩阵（lora_A, 形状为 r × d_in）升维，将输入从d_in升到较小秩r。
#        ├─(b) 升维后结果通过B矩阵（lora_B, 形状为 d_out × r）降维回输出空间（d_out）。
#        ├─(c) 低秩分支的输出乘以缩放系数 alpha/r（解决缩放不平衡问题）。
#        ├─(d) A/B为唯一可训练参数，初始时B设为全0，保证刚开始等价于原始线性输出。
#    3. 两分支输出直接相加：前者保持预训练信息，后者则在微调时捕捉新任务的变化。
#    4. 微调时冻结 W（和 bias），只优化 LoRA 部分（A/B），大大减少需更新的参数量。
#
#  直观对比：
#  ┌────────────────────────────┬───────────────────────────────┐
#  │         Base分支           │         LoRA分支              │
#  ├────────────────────────────┼───────────────────────────────┤
#  │  x @ W^T + bias（仅推理）    │  (alpha / r) * x @ A^T @ B^T  │
#  │  权重W冻结                  │  仅A、B可训练（低秩矩阵）      │
#  └────────────────────────────┴───────────────────────────────┘
#
#  应用：只有LoRA分支可以微调，Base保持预训练权重不变。

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRALinear 是 nn.Linear 的替代品，实现单个线性层的 LoRA 扩展。
    apply中根据target_modules来判断要替换哪些层
    现在只替换w_q和w_v

        输出公式：
            y = x @ W^T + bias + (alpha/r) * (x @ A^T @ B^T)
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = original.weight  # 仅对应单个q/k/v原始权重
        self.bias = original.bias

        # 单组 LoRA 可训练参数（仅针对当前linear）
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 冻结原始 W 和 bias，只训练 LoRA 分支
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x 是输入特征，形状通常为 (batch_size, seq_len, d_model)。
        本层只负责当前 Linear 的前向，不再分别处理 w_q/w_k/w_v，只操作包裹的 self.weight 和 LoRA 分支。
        """
        base_out = F.linear(x, self.weight, self.bias)
        # LoRA 分支：(alpha/r) * (x @ lora_A^T @ lora_B^T)
        lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base_out + lora_out * self.scaling

    def merge_and_unload(self) -> nn.Linear:
        """
        合并 LoRA 权重到原始 Linear 层，返回一个普通 nn.Linear，
        可用在微调结束后，将 LoRA 的效果直接“烙印”进基座权重，
        后续推理和部署均无需 LoRA 结构，只用标准 Linear 即可。

        通俗理解：微调完成后，A、B 两个低秩矩阵能和 W 合成一张大权重，
        之后不用 LoRA，也能保留微调能力（与合并前推理效果一致）。

        Note:
            LoRA 的优势是高效微调（低参数量），但合并后模型结构恢复成普通 Linear；
            训练阶段便捷，合并后推理部署更简单无感知。
        """
        # 创建标准 nn.Linear，复制 bias 配置
        merged = nn.Linear(
            self.in_features, self.out_features, bias=self.bias is not None
        )
        # 合并原始权重与 LoRA 权重修正项 (B @ A) * scaling
        merged.weight.data = (
            self.weight.data + (self.lora_B @ self.lora_A) * self.scaling
        )
        if self.bias is not None:
            merged.bias.data = self.bias.data
        return merged


# ---------------------------------------------------------------------------
# 目标模块名称设置 —— 指定哪些 nn.Linear 层引入 LoRA
# ---------------------------------------------------------------------------

DEFAULT_TARGET_MODULES = {"w_q", "w_v"}


def _match_target(name: str, targets: set[str]) -> bool:
    """
    检查参数名称的某一级是否包含目标子模块名。例如 w_q、w_v 等。
    """
    parts = name.split(".")
    return any(t in parts for t in targets)


# ---------------------------------------------------------------------------
# 对外 API 接口
# ---------------------------------------------------------------------------


def apply_lora(
    model: nn.Module,  # [参数来源] 外部传入的基础模型；[作用] 需要应用LoRA结构的神经网络模型
    rank: int = 8,  # [参数来源] 用户配置或默认；[作用] LoRA的低秩矩阵秩，决定可训练参数量（越小越省参数，通常8~64）
    alpha: float = 16.0,  # [参数来源] 用户配置或默认；[作用] LoRA缩放系数，用于调整LoRA增益在原始权重上的影响
    dropout: float = 0.0,  # [参数来源] 用户配置或默认；[作用] LoRA中的Dropout概率，可提升微调泛化，默认0不使用
    target_modules: set[str]
    | None = None,  # [参数来源] 用户配置或默认；[作用] 要插入LoRA的目标层名称集合，如{'w_q','w_v'}，None时用默认设置
) -> list[str]:
    """
     为模型的 Linear 层注入 LoRA 分支（低秩适应）。
    操作步骤：
       1. 冻结所有基座参数
       2. 将目标 nn.Linear 替换为 LoRALinear
       3. 只有 LoRA 的 A、B 参数可训练
    """
    target_modules = target_modules or DEFAULT_TARGET_MODULES

    # 先冻结整个模型参数
    for param in model.parameters():
        param.requires_grad_(False)

    replaced = []
    """""
    类似反射机制，找到nn.model的子模块(继承nn.Module的类)，并遍歷子模块的子模块。
    然后再遍歷子模块的子模块的子模块。
    找到目标層后進行替換。替換的目標是將nn.Linear替換為LoRALinear。
    """ ""

    # 1. 遍历模型所有模块（包括主模块及其所有子模块）
    for parent_name, parent_module in _named_modules_with_parent(model):
        # 2. 遍历父模块的所有直接子模块
        for attr_name, child in list(parent_module.named_children()):
            # 得到模块的完整名字，比如 encoder.layer.0.self_attn.w_q
            full_name = f"{parent_name}.{attr_name}" if parent_name else attr_name

            # [检测要替换为 LoRALinear 的依据]
            # 替换的条件是：该子模块是 nn.Linear（全连接层），并且其完整名字中含有目标关键词（如 'w_q', 'w_v' 等）。
            # 这些目标关键词由 target_modules 集合控制。
            # 检测逻辑见 _match_target：只要 full_name 的某一级名字与 target_modules 中的字符串匹配，即命中该层。
            """
            target_modules = {"w_q", "w_v"} 所以只替换原始的w_q和w_v
            """
            if isinstance(child, nn.Linear) and _match_target(
                full_name, target_modules
            ):
                # 满足条件：用 LoRALinear 替换 nn.Linear，并拷贝权重、偏置参数
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                # 保证设备与数据类型一致
                lora_layer = lora_layer.to(
                    device=child.weight.device, dtype=child.weight.dtype
                )
                # 替换子模块
                setattr(parent_module, attr_name, lora_layer)
                replaced.append(full_name)

    return replaced


def _named_modules_with_parent(model: nn.Module):
    """
    生成（模块名称，模块）元组，包括主模块本身及其所有子模块。
    """
    yield "", model
    for name, module in model.named_modules():
        if name:
            yield name, module


def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """
    提取模型中所有可训练 LoRA 参数 (A/B)。
    """
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad and ("lora_A" in name or "lora_B" in name):
            params.append(param)
    return params


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """
    仅提取模型中 LoRA 分支的参数，在保存下游微调参数时很有用。
    """
    return {
        k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k
    }


def save_lora(model: nn.Module, path: str | Path):
    """
    保存 LoRA 分支权重到文件，仅需存很小的数据量。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(get_lora_state_dict(model), path)


def load_lora(model: nn.Module, path: str | Path, device: torch.device | str = "cpu"):
    """
    加载 LoRA 分支权重进模型（前提：原模型已应用 LoRA 层结构）。
    """
    lora_state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(lora_state, strict=False)


def merge_lora(model: nn.Module) -> nn.Module:
    """
    说明：“合并 LoRA”并不是把低秩 A/B 直接拼到主权重（主权重和低秩矩阵维度肯定不一样，不能直接 cat 或叠加）。
    LoRA 本质是这样：假设 pretrained 主权重是 W（[out, in]），LoRA 分支就是生成一个与 W 同形状的补丁 ΔW = scaling * (B @ A)，
    其中 A:[r, in], B:[out, r]，B @ A 得到 [out, in]，正好和 W 同形可加！
    最终 W' = W + ΔW，也就是原始权重加上经过 LoRA 低秩分解但再投影到原始形状的扰动。

    具体实现：实际“合并/卸载”操作就是在 LoRA Linear 层对象上调用 .merge_and_unload()（内部逻辑正确处理了 patch 的加法），
    替换为普通 nn.Linear 层，权重已经包含了 LoRA 调整后的主权重，维度永远是对得上的。
    """
    for parent_name, parent_module in _named_modules_with_parent(model):
        for attr_name, child in list(parent_module.named_children()):
            if isinstance(child, LoRALinear):
                # 这里 merge_and_unload 生成了一个 weight 已合成（包含低秩补丁后）的 nn.Linear
                merged_linear = child.merge_and_unload()
                setattr(parent_module, attr_name, merged_linear)
    return model


def count_lora_parameters(model: nn.Module) -> tuple[int, int]:
    """
    统计模型（当前）可训练参数数量和总参数数量，返回 (可训练, 总数) 二元组。
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
