"""
LoRA (Low-Rank Adaptation) — 低秩适应, 参数高效微调
====================================================

核心思路: 冻结原始权重 W, 只训练两个小矩阵 A 和 B,
用 B @ A 近似一个"更新补丁", 加到 W 上。

默认配置: rank=8, alpha=16.0, target_modules={"w_q", "w_v"}
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    把 nn.Linear 替换成 LoRA 版本: 原始路径 + 低秩旁路。

    ┌──────────────────────────────────────────────────────────────────┐
    │  数值示例 (d_model=768, rank=8, alpha=16):                      │
    │                                                                  │
    │  原始 nn.Linear: W (768, 768), 有 768×768 = 590,208 个参数    │
    │                                                                  │
    │  LoRA 新增:                                                      │
    │    A: (8, 768)  =   6,144 个参数                                │
    │    B: (768, 8)  =   6,144 个参数                                │
    │    合计:          12,288 个参数 — 只有原来的 2%!               │
    │                                                                  │
    │  前向传播:                                                       │
    │    输入 x: (B, seq, 768) = (2, 5, 768)                          │
    │                                                                  │
    │    原始路径: x @ W^T + bias → (2, 5, 768)                       │
    │      W 是冻结的, 不参与微调训练                                 │
    │                                                                  │
    │    LoRA 路径:                                                    │
    │      x @ A^T: (2,5,768) @ (768,8) = (2,5,8)                    │
    │        ↑ 768 维压到 8 维 — "降维/浓缩"                          │
    │        为什么能压到这么小? 因为微调需要的"更新"是低秩的,       │
    │        8 维足以表达大部分需要调整的方向                         │
    │                                                                  │
    │      (x @ A^T) @ B^T: (2,5,8) @ (8,768) = (2,5,768)            │
    │        ↑ 8 维升回 768 维 — "还原/展开"                          │
    │                                                                  │
    │      × scaling = alpha/rank = 16/8 = 2.0                        │
    │        为什么缩放? rank 越大, B@A 的值越大,                     │
    │        除以 rank 让不同 rank 的 LoRA 输出量级一致              │
    │        乘 alpha 控制整体强度                                    │
    │                                                                  │
    │    总输出 = 原始路径 + LoRA路径 × scaling                       │
    │      = x @ W^T + bias + 2.0 × (x @ A^T @ B^T)                 │
    │      (2, 5, 768)                                                │
    │                                                                  │
    │  B 初始化为零 → 训练刚开始时 LoRA 路径输出全0                  │
    │  → 等价于原始模型, 微调从原始能力出发, 不会崩                  │
    │                                                                  │
    │  只替换 w_q 和 w_v (不替换 w_k, w_o):                           │
    │    实验表明只改 Q 和 V 的投影就足够了                           │
    │    替换越少, 可训练参数越少, 微调越高效                         │
    │    但如果想更强, 也可以把 w_k, w_o 加进 target_modules         │
    └──────────────────────────────────────────────────────────────────┘
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
        self.scaling = alpha / rank  # 16/8 = 2.0

        self.weight = original.weight   # (768, 768) 冻结
        self.bias = original.bias

        # LoRA 参数: A 和 B
        # A: (rank, in_features) = (8, 768), kaiming 初始化
        # B: (out_features, rank) = (768, 8), 零初始化
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 冻结原始权重, 只训练 LoRA
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, seq, 768) → (B, seq, 768)
        """
        base_out = F.linear(x, self.weight, self.bias)
        # LoRA: x → A^T(768→8) → B^T(8→768) → × scaling
        lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base_out + lora_out * self.scaling

    def merge_and_unload(self) -> nn.Linear:
        """
        合并: W' = W + scaling × (B @ A), 返回普通 nn.Linear。

        ┌──────────────────────────────────────────────────────────────┐
        │  数值示例:                                                    │
        │    W:  (768, 768) — 原始权重                                 │
        │    B:  (768, 8)                                              │
        │    A:  (8, 768)                                              │
        │    B @ A: (768, 8) @ (8, 768) = (768, 768) — 和 W 同形!     │
        │    ΔW = 2.0 × (B @ A): (768, 768) — 更新补丁               │
        │    W' = W + ΔW: (768, 768) — 合并后的权重                   │
        │                                                              │
        │  合并后推理不需要 A, B, 速度和原始模型一样                  │
        │  效果和合并前完全一致, 只是少了旁路计算                     │
        └──────────────────────────────────────────────────────────────┘
        """
        merged = nn.Linear(
            self.in_features, self.out_features, bias=self.bias is not None
        )
        merged.weight.data = (
            self.weight.data + (self.lora_B @ self.lora_A) * self.scaling
        )
        if self.bias is not None:
            merged.bias.data = self.bias.data
        return merged


DEFAULT_TARGET_MODULES = {"w_q", "w_v"}


def _match_target(name: str, targets: set[str]) -> bool:
    """检查模块名是否匹配目标集合, 如 {"w_q", "w_v"}。"""
    parts = name.split(".")
    return any(t in parts for t in targets)


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: set[str] | None = None,
) -> list[str]:
    """
    给模型的指定 Linear 层注入 LoRA 分支。

    ┌──────────────────────────────────────────────────────────────┐
    │  操作流程:                                                   │
    │                                                              │
    │  ① 冻结模型所有参数                                         │
    │    model.parameters() → requires_grad=False                  │
    │                                                              │
    │  ② 遍历找到 nn.Linear, 名字含 "w_q" 或 "w_v"               │
    │    共 10 层 × 2 个目标 = 20 个 Linear 要替换                │
    │    每层的 w_q: (768, 768) → LoRALinear                      │
    │    每层的 w_v: (768, 768) → LoRALinear                      │
    │                                                              │
    │  ③ 替换: setattr(parent, name, LoRALinear(original))        │
    │    LoRALinear 里 A 和 B 是 requires_grad=True                │
    │    其余 (包括原始 W) 都是 False                              │
    │                                                              │
    │  参数量对比 (默认配置):                                      │
    │    原始可训练: 768×768 × 4(每层q,k,v,o) × 10层 ≈ 23.6M     │
    │    LoRA 可训练: (8×768 + 768×8) × 2(每层q,v) × 10层        │
    │              = 12288 × 2 × 10 = 245,760 ≈ 0.25M             │
    │    只占原来的 1%!                                            │
    └──────────────────────────────────────────────────────────────┘
    """
    target_modules = target_modules or DEFAULT_TARGET_MODULES

    for param in model.parameters():
        param.requires_grad_(False)

    replaced = []
    for parent_name, parent_module in _named_modules_with_parent(model):
        for attr_name, child in list(parent_module.named_children()):
            full_name = f"{parent_name}.{attr_name}" if parent_name else attr_name

            if isinstance(child, nn.Linear) and _match_target(
                full_name, target_modules
            ):
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                lora_layer = lora_layer.to(
                    device=child.weight.device, dtype=child.weight.dtype
                )
                setattr(parent_module, attr_name, lora_layer)
                replaced.append(full_name)

    return replaced


def _named_modules_with_parent(model: nn.Module):
    """生成 (模块名, 模块) 元组, 包括模型本身和所有子模块。"""
    yield "", model
    for name, module in model.named_modules():
        if name:
            yield name, module


def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """提取所有 LoRA 可训练参数 (A 和 B)。"""
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad and ("lora_A" in name or "lora_B" in name):
            params.append(param)
    return params


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """仅提取 LoRA 分支的参数, 保存时只需存很小的文件。"""
    return {
        k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k
    }


def save_lora(model: nn.Module, path: str | Path):
    """保存 LoRA 权重到文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(get_lora_state_dict(model), path)


def load_lora(model: nn.Module, path: str | Path, device: torch.device | str = "cpu"):
    """加载 LoRA 权重进模型 (前提: 已 apply_lora)。"""
    lora_state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(lora_state, strict=False)


def merge_lora(model: nn.Module) -> nn.Module:
    """
    把所有 LoRALinear 合并成普通 nn.Linear。

    合并后: 推理不需要旁路, 速度更快, 效果不变。
    """
    for parent_name, parent_module in _named_modules_with_parent(model):
        for attr_name, child in list(parent_module.named_children()):
            if isinstance(child, LoRALinear):
                merged_linear = child.merge_and_unload()
                setattr(parent_module, attr_name, merged_linear)
    return model


def count_lora_parameters(model: nn.Module) -> tuple[int, int]:
    """返回 (可训练参数数, 总参数数)。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
