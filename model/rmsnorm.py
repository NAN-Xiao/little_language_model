"""
RMSNorm (Root Mean Square Layer Normalization)
===============================================

Llama/Qwen/DeepSeek 等现代模型的标配归一化层。

与 LayerNorm 的区别:
  LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
  RMSNorm:   y = x / sqrt(mean(x^2) + eps) * weight

RMSNorm 省去了"减均值"这一步，计算更快，效果持平。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    RMSNorm — 只用均方根做归一化。

    公式:
        rms = sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
        y = x / rms * weight

    参数:
        dim:      归一化的维度大小 (如 d_model=768)
        eps:      防止除0的小数，默认 1e-6

    维度示例:
        输入 x: (B, seq, 768)
        mean(x^2, dim=-1): (B, seq, 1)  ← 每个位置算一个 rms 值
        rms: (B, seq, 1)
        x / rms: (B, seq, 768)  ← 广播除法
        weight: (768,)  ← 可学习参数，每个维度一个缩放因子
        输出 y: (B, seq, 768)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # weight: 每个维度一个可学习的缩放因子
        # 初始化为 1.0（不缩放），训练后自动调整
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x: (..., dim) — 任意形状，最后 1 维是归一化维度

        返回:
            y: (..., dim) — 归一化后的张量，形状不变
        """
        # mean(x^2, dim=-1, keepdim=True): 每个位置算均方
        # 例: x=(2,5,768) → mean=(2,5,1)
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)

        # rms = sqrt(mean_sq + eps)
        rms = torch.rsqrt(mean_sq + self.eps)
        # rsqrt = 1/sqrt，避免先开方再除法，数值更稳定

        # x * rms: 归一化，每个位置的向量长度变为 1
        # 例: (2,5,768) * (2,5,1) = (2,5,768)
        x_norm = x * rms

        # weight: (768,) 广播到 (2,5,768)
        # 每个维度一个可学习的缩放因子
        return x_norm * self.weight
