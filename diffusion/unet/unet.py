"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  UNet —— 扩散模型的噪声预测骨干网络                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

UNet 在扩散模型（Diffusion Model）中扮演的核心角色：
  给定一张加了噪声的图片 x_t、当前时间步 t，以及可选的文本提示词嵌入 context，
  预测噪声 ε（或速度 v）。

整体结构（编码器-瓶颈-解码器，带跳跃连接 + 文本条件化）：

  输入 x_t (B, C, H, W)  +  时间步 t (B,)  +  文本嵌入 context (B, L, D_ctx)【可选】
         │                                              │
         ▼                                              │
  ┌─────────────────────────────┐                      │
  │   时间嵌入 SinusoidalEmb     │   t → (B, time_dim)  │
  └─────────────┬───────────────┘                      │
                │ time_emb                              │ context
  ┌─────────────▼──────────────────────────────────────▼─────────┐
  │  编码器 (Encoder)                                              │
  │  DownBlock: ResBlock × N → [CrossAttn(context)] → Downsample  │
  └───────────────────────────────────────────────────────────────┘
                │
  ┌─────────────▼──────────────────────────────────────▼─────────┐
  │  瓶颈 (Bottleneck)                                             │
  │  ResBlock → SelfAttn → CrossAttn(context) → ResBlock          │
  └───────────────────────────────────────────────────────────────┘
                │
  ┌─────────────▼──────────────────────────────────────▼─────────┐
  │  解码器 (Decoder)                                              │
  │  UpBlock: Upsample → concat(skip) → ResBlock × N → [CrossAttn]│
  └───────────────────────────────────────────────────────────────┘
                │
         输出头 (1×1 卷积)
                │
         ε_pred (B, C, H, W)   ← 预测的噪声

文本条件化（Cross-Attention）原理：
  - 文本经过 CLIP/T5 等文本编码器，得到 context (B, L, D_ctx)
    其中 L = 文本 token 数（如 77），D_ctx = 文本嵌入维度（如 768）
  - 在 UNet 的每个注意力层：
      Q = 图像特征（展平为序列）    ← 图像"问"文本"我该生成什么"
      K = V = context（文本嵌入）   ← 文本"告诉"图像内容
  - 这样图像生成过程就被文本内容所引导

  无文本时（context=None）：退化为无条件生成，兼容原有行为。

参考论文:
  - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
    https://arxiv.org/abs/2006.11239
  - Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion
    Models" (Stable Diffusion), CVPR 2022, https://arxiv.org/abs/2112.10752
  - Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
    Image Segmentation", MICCAI 2015
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# 0. 配置
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class UNetConfig:
    """UNet 超参数配置。

    参数:
        in_channels  (int): 输入图像的通道数，例如 RGB=3、灰度=1。
        out_channels (int): 输出通道数，通常等于 in_channels（预测噪声的通道数）。
        base_channels (int): 第一层编码器的基础通道数，后续每下采样一次翻倍。
        channel_mults (tuple): 每一层的通道倍数，长度决定下采样级数。
            例如 (1, 2, 4, 8) 表示共 4 级，通道依次为
            base*1, base*2, base*4, base*8。
        num_res_blocks (int): 每个 DownBlock/UpBlock 中堆叠的 ResBlock 数量。
        attn_resolutions (tuple): 在哪些分辨率（相对于输入）使用注意力，
            例如 (8, 16) 表示当特征图空间尺寸为 8 或 16 时加注意力。
        dropout (float): ResBlock 内部的 dropout 概率。
        time_emb_dim (int): 时间嵌入的隐层维度（内部正弦编码维度）。
        context_dim (int): 文本嵌入的维度（来自 CLIP/T5 等文本编码器）。
            设为 0 表示不使用文本条件化（无条件生成）。
            例如：CLIP ViT-L/14 输出 768 维，T5-XXL 输出 4096 维。
    """
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 64
    channel_mults: tuple = (1, 2, 4, 8)
    num_res_blocks: int = 2
    attn_resolutions: tuple = (8, 16)
    dropout: float = 0.1
    time_emb_dim: int = 256
    context_dim: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# 1. 时间嵌入（Sinusoidal Time Embedding）
# ═══════════════════════════════════════════════════════════════════════════

class SinusoidalTimeEmbedding(nn.Module):
    """将整数时间步 t 编码为高维向量（仿照 Transformer 位置编码）。

    扩散模型需要知道当前处于哪个噪声水平（第几步），才能正确预测。
    直接把整数 t 传进去效果很差，因此用正弦/余弦函数将其映射到高维空间。

    数学形式（与 Transformer 位置编码完全相同）：
        emb[2i]   = sin(t / 10000^(2i/dim))
        emb[2i+1] = cos(t / 10000^(2i/dim))

    然后再经过两层 MLP 提升表达能力：
        SinEmb(t) → Linear(dim, time_emb_dim) → SiLU → Linear(time_emb_dim, time_emb_dim)

    可视化（dim=8 时，不同 t 值产生不同的向量）：
        t=0:   [sin(0), cos(0), sin(0), cos(0), ...]  = [0, 1, 0, 1, ...]
        t=10:  [sin(10/1), cos(10/1), sin(10/100), cos(10/100), ...]
        t=500: [sin(500/1), ..., sin(500/10000), ...]

    参数:
        dim        (int): 正弦编码的基础维度（通常 = base_channels * 4）。
        out_dim    (int): 经 MLP 后输出的维度，即注入每层 ResBlock 的向量大小。
    """

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.dim = dim
        # 两层 MLP，增强时间嵌入的非线性表达能力
        self.mlp = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.SiLU(),           # SiLU(x) = x * sigmoid(x)，比 ReLU 更平滑
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        参数:
            t (torch.Tensor): 整数时间步，形状 (B,)，值域 [0, T-1]。
        返回:
            torch.Tensor: 时间嵌入向量，形状 (B, out_dim)。
        """
        # ── 正弦编码 ──────────────────────────────────────────────────────
        # half_dim = dim // 2，生成 half_dim 个频率
        half_dim = self.dim // 2
        # frequencies: [1/10000^(0/half_dim), ..., 1/10000^((half_dim-1)/half_dim)]
        # 形状: (half_dim,)
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        # t: (B,)  →  t[:, None]: (B, 1)
        # freqs[None, :]: (1, half_dim)
        # 广播相乘后: (B, half_dim)
        args = t[:, None].float() * freqs[None, :]
        # 拼接 sin 和 cos，形状: (B, dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        # ── MLP 增强 ──────────────────────────────────────────────────────
        return self.mlp(emb)


# ═══════════════════════════════════════════════════════════════════════════
# 2. 残差块（ResBlock）
# ═══════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """时间条件化残差块（ResBlock）。

    这是 UNet 中最核心的积木，结构如下：

        x ──┬── GroupNorm → SiLU → Conv(3×3) ──┬── GroupNorm → SiLU → Dropout → Conv(3×3) ──┬── out
            │                                    │                                              │
            │            time_emb                │                                              │
            │     (B, time_dim) → Linear → SiLU  │                                              │
            │           ↓ 形状: (B, out_ch, 1, 1)│                                              │
            │            加到特征图上 ─────────────┘                                              │
            │                                                                                   │
            └─── 残差映射（若通道数不同，用 1×1 卷积对齐）───────────────────────────────────────┘

    时间信息注入方式：
        将时间嵌入 time_emb (B, time_dim) 经线性层投影到 (B, out_ch)，
        然后 reshape 成 (B, out_ch, 1, 1)，直接加到特征图上。
        这样每个空间位置都得到了相同的时间偏置，使整个特征图"感知"到当前时间步。

    参数:
        in_ch    (int): 输入通道数。
        out_ch   (int): 输出通道数（若与 in_ch 不同，残差路径用 1×1 卷积对齐）。
        time_dim (int): 时间嵌入向量的维度。
        dropout  (float): Dropout 概率。
        num_groups (int): GroupNorm 的组数（要求 out_ch 能整除）。
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int,
                 dropout: float = 0.1, num_groups: int = 8):
        super().__init__()

        # ── 第一条卷积路径 ──────────────────────────────────────────────────
        # GroupNorm 将通道分为 num_groups 组，对每组独立归一化
        # 比 BatchNorm 更适合小 batch；比 LayerNorm 更适合 CNN
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        # ── 时间嵌入注入 ────────────────────────────────────────────────────
        # 将 (B, time_dim) 线性投影到 (B, out_ch)，然后加到特征图
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )

        # ── 第二条卷积路径 ──────────────────────────────────────────────────
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # ── 残差连接的通道对齐 ──────────────────────────────────────────────
        # 如果输入输出通道不一致，用 1×1 卷积把输入映射到目标通道数
        # 如果通道相同，直接用恒等映射（Identity）
        if in_ch != out_ch:
            self.residual_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x        (torch.Tensor): 输入特征图，形状 (B, in_ch, H, W)。
            time_emb (torch.Tensor): 时间嵌入向量，形状 (B, time_dim)。
        返回:
            torch.Tensor: 输出特征图，形状 (B, out_ch, H, W)。
        """
        # ── 主路径 ──────────────────────────────────────────────────────────
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # ── 注入时间信息 ─────────────────────────────────────────────────────
        # time_emb: (B, time_dim) → 经 time_proj → (B, out_ch)
        # reshape 成 (B, out_ch, 1, 1)，广播加到 (B, out_ch, H, W) 上
        # 这就像给每个时间步设置一个全局偏置，让网络知道当前噪声水平
        t = self.time_proj(time_emb)          # (B, out_ch)
        h = h + t[:, :, None, None]           # (B, out_ch, H, W)

        # ── 第二层卷积 ───────────────────────────────────────────────────────
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # ── 残差连接 ─────────────────────────────────────────────────────────
        # out = F(x) + x（如果通道不同则对 x 做线性投影）
        return h + self.residual_conv(x)


# ═══════════════════════════════════════════════════════════════════════════
# 3. 自注意力块（AttentionBlock）
# ═══════════════════════════════════════════════════════════════════════════

class AttentionBlock(nn.Module):
    """2D 特征图上的多头自注意力。

    在低分辨率的特征图（如 8×8、16×16）上使用注意力，
    可以让网络建立全局依赖关系（例如：图像左上角的纹理影响右下角的生成）。

    实现思路：
        1. 将 (B, C, H, W) reshape 为序列 (B, H*W, C)
        2. 做多头自注意力（Q=K=V=x）
        3. reshape 回 (B, C, H, W)

    可视化（H=W=8, C=256 的特征图展开为序列）:
        (B, 256, 8, 8) ──flatten空间维度──→ (B, 64, 256) ──MHA──→ (B, 64, 256) → (B, 256, 8, 8)

    参数:
        channels   (int): 特征通道数（= 序列的 embedding 维度）。
        num_heads  (int): 注意力头的数量。
        num_groups (int): GroupNorm 的组数。
    """

    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        # PyTorch 内置多头注意力（batch_first=True 表示输入格式是 (B, seq, dim)）
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x (torch.Tensor): 特征图，形状 (B, C, H, W)。
        返回:
            torch.Tensor: 与输入相同形状 (B, C, H, W)，加入了全局注意力信息。
        """
        B, C, H, W = x.shape

        # ── 归一化 ──────────────────────────────────────────────────────────
        h = self.norm(x)                          # (B, C, H, W)

        # ── reshape: 图片 → 序列 ────────────────────────────────────────────
        # 把空间维度 H×W 展平成序列长度
        # (B, C, H, W) → (B, C, H*W) → (B, H*W, C)
        h = h.reshape(B, C, H * W).transpose(1, 2)   # (B, H*W, C)

        # ── 多头自注意力（Q=K=V=h）──────────────────────────────────────────
        # need_weights=False 加速，不返回注意力权重矩阵
        h, _ = self.attn(h, h, h, need_weights=False)  # (B, H*W, C)

        # ── 还原形状 ─────────────────────────────────────────────────────────
        h = h.transpose(1, 2).reshape(B, C, H, W)   # (B, C, H, W)

        # ── 残差连接 ─────────────────────────────────────────────────────────
        return x + h


# ═══════════════════════════════════════════════════════════════════════════
# 3b. 交叉注意力块（CrossAttentionBlock）—— 文本条件化
# ═══════════════════════════════════════════════════════════════════════════

class CrossAttentionBlock(nn.Module):
    """图像特征与文本嵌入之间的交叉注意力，实现文本提示词对图像生成的控制。

    这是 Stable Diffusion 文本控制的核心机制：
        Q（查询）= 图像特征  ← 图像问："我在 context 里能找到什么信息？"
        K（键）  = 文本嵌入  ← 文本提供"检索索引"
        V（值）  = 文本嵌入  ← 文本提供"实际内容"

    直觉类比：
        想象图像特征是一个"学生"，文本是一本"教材"。
        交叉注意力就是学生根据自己的疑问（Q），
        在教材里找到相关段落（K 匹配），然后吸收那段内容（V）。
        这样图像生成就被文本内容所引导。

    数学流程：
        h = 图像特征 (B, H*W, channels)    ← Q 的来源
        c = 文本嵌入 (B, L, context_dim)   ← K、V 的来源

        Q = h @ W_q    → (B, H*W, channels)
        K = c @ W_k    → (B, L,   channels)   ← 注意：K、V 来自文本！
        V = c @ W_v    → (B, L,   channels)

        Attn = softmax(Q @ K^T / sqrt(d)) @ V   → (B, H*W, channels)

    与自注意力（AttentionBlock）的区别：
        自注意力：Q = K = V = 图像特征（图像"问"自己）
        交叉注意力：Q = 图像特征，K = V = 文本（图像"问"文本）

    可视化（prompt="一只橙色的猫"，图像特征 8×8=64 个 token）：
        图像每个位置的特征 → Q
        "一只"、"橙色"、"猫" → K, V
        注意力权重：图像左上角可能更关注"一只"，
                    橙色区域更关注"橙色"，猫咪区域更关注"猫"

    参数:
        channels    (int): 图像特征的通道数（= Q 的维度）。
        context_dim (int): 文本嵌入的维度（= K、V 的维度）。
        num_heads   (int): 注意力头的数量。
        num_groups  (int): GroupNorm 的组数。
    """

    def __init__(self, channels: int, context_dim: int,
                 num_heads: int = 4, num_groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)

        # Q 来自图像特征（channels 维），K/V 来自文本嵌入（context_dim 维）
        # 三者都投影到同一个 channels 维空间，才能做注意力计算
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(context_dim, channels, bias=False)
        self.to_v = nn.Linear(context_dim, channels, bias=False)

        # 输出线性层，将注意力结果映射回原始维度
        self.to_out = nn.Linear(channels, channels, bias=False)

        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5   # 1 / sqrt(d_k)，防止点积过大

    def forward(self, x: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x       (torch.Tensor): 图像特征图，形状 (B, channels, H, W)。
            context (torch.Tensor): 文本嵌入，形状 (B, L, context_dim)。
                                    L = 文本 token 数（如 CLIP 的 77）。
        返回:
            torch.Tensor: 注入文本信息后的特征图，形状 (B, channels, H, W)。
        """
        B, C, H, W = x.shape
        L = context.shape[1]

        # ── 归一化图像特征 ────────────────────────────────────────────────
        h = self.norm(x)

        # ── reshape 图像特征：(B, C, H, W) → (B, H*W, C) ─────────────────
        # 把空间维度展平成序列，让每个像素位置成为一个 token
        h = h.reshape(B, C, H * W).transpose(1, 2)   # (B, H*W, C)

        # ── 计算 Q、K、V ──────────────────────────────────────────────────
        # Q 来自图像，K/V 来自文本
        q = self.to_q(h)        # (B, H*W, C)   ← 图像特征投影
        k = self.to_k(context)  # (B, L,   C)   ← 文本嵌入投影（K）
        v = self.to_v(context)  # (B, L,   C)   ← 文本嵌入投影（V）

        # ── 拆分多头：(B, seq, C) → (B, heads, seq, head_dim) ────────────
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            bs, seq, _ = t.shape
            return t.reshape(bs, seq, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)   # (B, heads, H*W, head_dim)
        k = split_heads(k)   # (B, heads, L,   head_dim)
        v = split_heads(v)   # (B, heads, L,   head_dim)

        # ── 缩放点积注意力 ────────────────────────────────────────────────
        # attn_scores: (B, heads, H*W, L)
        # 每个图像 token（H*W 个）对每个文本 token（L 个）的注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)   # 在 L 维度归一化

        # 加权求和：(B, heads, H*W, head_dim)
        out = torch.matmul(attn_weights, v)

        # ── 合并多头：(B, heads, H*W, head_dim) → (B, H*W, C) ────────────
        out = out.transpose(1, 2).reshape(B, H * W, C)

        # ── 输出投影 ──────────────────────────────────────────────────────
        out = self.to_out(out)

        # ── 还原空间形状 + 残差连接 ──────────────────────────────────────
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return x + out


# ═══════════════════════════════════════════════════════════════════════════
# 4. 下采样块（DownBlock）—— 编码器
# ═══════════════════════════════════════════════════════════════════════════

class DownBlock(nn.Module):
    """编码器中的一个下采样阶段。

    结构：
        [ResBlock × num_res_blocks]（可选 AttentionBlock）→ Downsample(步幅卷积)

    下采样使用步幅为 2 的卷积，将空间分辨率减半，通道数翻倍（由外部决定）。
    这和 MaxPool 的区别是：步幅卷积的权重是可学习的，能学到更好的下采样方式。

    参数:
        in_ch       (int): 输入通道数。
        out_ch      (int): 输出通道数（通常是 in_ch 的 2 倍）。
        time_dim    (int): 时间嵌入维度。
        num_res     (int): ResBlock 数量。
        use_attn    (bool): 是否在此阶段使用注意力。
        dropout     (float): Dropout 概率。
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int,
                 num_res: int = 2, use_attn: bool = False, dropout: float = 0.1,
                 context_dim: int = 0):
        super().__init__()

        # ── ResBlock 序列 ────────────────────────────────────────────────────
        # 第一个 ResBlock 负责通道转换（in_ch → out_ch），后续保持 out_ch
        self.res_blocks = nn.ModuleList()
        for i in range(num_res):
            block_in = in_ch if i == 0 else out_ch
            self.res_blocks.append(ResBlock(block_in, out_ch, time_dim, dropout))

        # ── 可选自注意力 ─────────────────────────────────────────────────────
        self.attn = AttentionBlock(out_ch) if use_attn else None

        # ── 可选交叉注意力（文本条件化） ─────────────────────────────────────
        # context_dim > 0 时才创建，否则为 None（无条件生成）
        self.cross_attn = CrossAttentionBlock(out_ch, context_dim) if context_dim > 0 else None

        # ── 下采样：步幅 2 的卷积，空间尺寸减半 ─────────────────────────────────
        # kernel=3, stride=2, pad=1 → H_out = (H_in + 2*1 - 3) / 2 + 1 = H_in / 2
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor,
                context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            x        (torch.Tensor): 输入特征图，形状 (B, in_ch, H, W)。
            time_emb (torch.Tensor): 时间嵌入，形状 (B, time_dim)。
            context  (torch.Tensor | None): 文本嵌入，形状 (B, L, context_dim)。
                                            为 None 时跳过交叉注意力（无条件生成）。
        返回:
            Tuple[torch.Tensor, torch.Tensor]:
                - 下采样后的特征图，形状 (B, out_ch, H/2, W/2)。
                - 跳跃连接特征图（下采样前），形状 (B, out_ch, H, W)。
        """
        h = x
        for res_block in self.res_blocks:
            h = res_block(h, time_emb)

        if self.attn is not None:
            h = self.attn(h)

        # 交叉注意力：让图像特征"查询"文本嵌入，注入文本语义
        if self.cross_attn is not None and context is not None:
            h = self.cross_attn(h, context)

        # 保存下采样前的特征图，供解码器跳跃连接使用
        skip = h                         # (B, out_ch, H, W)
        h = self.downsample(h)           # (B, out_ch, H/2, W/2)

        return h, skip


# ═══════════════════════════════════════════════════════════════════════════
# 5. 上采样块（UpBlock）—— 解码器
# ═══════════════════════════════════════════════════════════════════════════

class UpBlock(nn.Module):
    """解码器中的一个上采样阶段。

    结构：
        Upsample(双线性 ×2) → concat(skip) → [ResBlock × num_res_blocks]（可选 AttentionBlock）

    上采样使用双线性插值 + 卷积，将空间分辨率翻倍。
    拼接跳跃连接后，通道数翻倍，因此第一个 ResBlock 的输入通道是 out_ch*2。

    跳跃连接（Skip Connection）的作用：
        编码器的特征图保留了高分辨率细节（边缘、纹理），
        与解码器的语义特征拼接，让解码器既有"大局观"又有"细节感"。

    示意图（UpBlock_1 为例）：
        瓶颈特征  (B, ch3, H/8, W/8)
              ↓ 上采样 ×2
        (B, ch3, H/4, W/4)
              ↓ concat(skip3: (B, ch3, H/4, W/4))
        (B, ch3*2, H/4, W/4)
              ↓ ResBlocks
        (B, ch2, H/4, W/4)

    参数:
        in_ch     (int): 上采样后的通道数（来自上一层输出）。
        skip_ch   (int): 跳跃连接的通道数（来自对应编码器层）。
        out_ch    (int): 输出通道数。
        time_dim  (int): 时间嵌入维度。
        num_res   (int): ResBlock 数量。
        use_attn  (bool): 是否使用注意力。
        dropout   (float): Dropout 概率。
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int,
                 num_res: int = 2, use_attn: bool = False, dropout: float = 0.1,
                 context_dim: int = 0):
        super().__init__()

        # ── 上采样：插值 + 卷积 ──────────────────────────────────────────────
        # 先用双线性插值放大 2 倍，再用 3×3 卷积精细化
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
        )

        # ── ResBlock 序列 ────────────────────────────────────────────────────
        # 第一个 ResBlock 输入 = in_ch（上采样） + skip_ch（跳跃连接）
        self.res_blocks = nn.ModuleList()
        for i in range(num_res):
            block_in = (in_ch + skip_ch) if i == 0 else out_ch
            self.res_blocks.append(ResBlock(block_in, out_ch, time_dim, dropout))

        # ── 可选自注意力 ─────────────────────────────────────────────────────
        self.attn = AttentionBlock(out_ch) if use_attn else None

        # ── 可选交叉注意力（文本条件化） ─────────────────────────────────────
        self.cross_attn = CrossAttentionBlock(out_ch, context_dim) if context_dim > 0 else None

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
                time_emb: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        """
        参数:
            x        (torch.Tensor): 来自上一个解码器层（或瓶颈）的特征图，
                                     形状 (B, in_ch, H, W)。
            skip     (torch.Tensor): 对应编码器层的跳跃连接特征，
                                     形状 (B, skip_ch, H*2, W*2)。
            time_emb (torch.Tensor): 时间嵌入，形状 (B, time_dim)。
            context  (torch.Tensor | None): 文本嵌入，形状 (B, L, context_dim)。
        返回:
            torch.Tensor: 上采样后特征图，形状 (B, out_ch, H*2, W*2)。
        """
        # ── 上采样：空间尺寸翻倍 ─────────────────────────────────────────────
        h = self.upsample(x)         # (B, in_ch, H*2, W*2)

        # ── 拼接跳跃连接 ─────────────────────────────────────────────────────
        # 在通道维度（dim=1）上拼接
        h = torch.cat([h, skip], dim=1)  # (B, in_ch + skip_ch, H*2, W*2)

        # ── ResBlocks ────────────────────────────────────────────────────────
        for res_block in self.res_blocks:
            h = res_block(h, time_emb)

        if self.attn is not None:
            h = self.attn(h)

        # 交叉注意力：注入文本语义
        if self.cross_attn is not None and context is not None:
            h = self.cross_attn(h, context)

        return h


# ═══════════════════════════════════════════════════════════════════════════
# 6. 瓶颈块（Bottleneck）
# ═══════════════════════════════════════════════════════════════════════════

class Bottleneck(nn.Module):
    """UNet 中间最低分辨率的瓶颈模块。

    结构：ResBlock → AttentionBlock → ResBlock

    在最低分辨率（如 8×8 或 4×4）的特征图上，
    全局注意力的计算量可接受（序列长度仅 64 或 16），
    而语义信息最为丰富，适合全局建模。

    参数:
        channels (int): 瓶颈的通道数（= 最深层的通道数）。
        time_dim (int): 时间嵌入维度。
        dropout  (float): Dropout 概率。
    """

    def __init__(self, channels: int, time_dim: int, dropout: float = 0.1,
                 context_dim: int = 0):
        super().__init__()
        self.res1 = ResBlock(channels, channels, time_dim, dropout)
        self.self_attn = AttentionBlock(channels)
        # 瓶颈是语义最丰富、分辨率最低的地方，最适合注入文本信息
        self.cross_attn = CrossAttentionBlock(channels, context_dim) if context_dim > 0 else None
        self.res2 = ResBlock(channels, channels, time_dim, dropout)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        """
        参数:
            x        (torch.Tensor): 输入特征图，形状 (B, channels, H, W)。
            time_emb (torch.Tensor): 时间嵌入，形状 (B, time_dim)。
            context  (torch.Tensor | None): 文本嵌入，形状 (B, L, context_dim)。
        返回:
            torch.Tensor: 与输入相同形状。
        """
        h = self.res1(x, time_emb)
        h = self.self_attn(h)
        # 自注意力之后紧跟交叉注意力：先理解图像自身，再融合文本语义
        if self.cross_attn is not None and context is not None:
            h = self.cross_attn(h, context)
        h = self.res2(h, time_emb)
        return h


# ═══════════════════════════════════════════════════════════════════════════
# 7. 主模型：UNet
# ═══════════════════════════════════════════════════════════════════════════

class UNet(nn.Module):
    """扩散模型的 UNet 骨干网络，输入加噪图像 x_t 和时间步 t，输出预测噪声 ε。

    完整的数据流：

        输入: x_t (B, C, H, W)  &  t (B,)
                │
        ┌───────▼────────┐
        │  时间嵌入        │   t (B,) → time_emb (B, time_emb_dim)
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │  输入卷积        │   (B, C, H, W) → (B, base_ch, H, W)
        └───────┬────────┘
                │
        ┌───────▼──────────────────────────────────────┐
        │  编码器                                        │
        │  DownBlock_0: (B, ch[0], H,   W  )  → skip_0 │
        │  DownBlock_1: (B, ch[1], H/2, W/2) → skip_1  │
        │  DownBlock_2: (B, ch[2], H/4, W/4) → skip_2  │
        │  DownBlock_3: (B, ch[3], H/8, W/8) → skip_3  │
        └───────────────────────────────────────────────┘
                │
        ┌───────▼──────────────────────────────────────┐
        │  瓶颈  (B, ch[-1], H/16, W/16)                │
        └───────────────────────────────────────────────┘
                │
        ┌───────▼──────────────────────────────────────┐
        │  解码器                                        │
        │  UpBlock_3: concat(skip_3) → (B, ch[2], H/8) │
        │  UpBlock_2: concat(skip_2) → (B, ch[1], H/4) │
        │  UpBlock_1: concat(skip_1) → (B, ch[0], H/2) │
        │  UpBlock_0: concat(skip_0) → (B, ch[0], H  ) │
        └───────────────────────────────────────────────┘
                │
        ┌───────▼────────┐
        │  输出头          │   GroupNorm → SiLU → Conv1×1 → (B, C, H, W)
        └───────┬────────┘
                │
        输出: ε_pred (B, C, H, W)   ← 预测的噪声

    参数:
        config (UNetConfig): 超参数配置对象，详见 UNetConfig 注释。
    """

    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config

        # ── 通道数列表 ───────────────────────────────────────────────────────
        # 例如 base=64, mults=(1,2,4,8) → chs=[64, 128, 256, 512]
        chs: list[int] = [config.base_channels * m for m in config.channel_mults]

        # ── 时间嵌入 ─────────────────────────────────────────────────────────
        # dim = base_channels * 4（经验值），out_dim = time_emb_dim
        time_dim = config.time_emb_dim
        self.time_embedding = SinusoidalTimeEmbedding(
            dim=config.base_channels * 4,
            out_dim=time_dim,
        )

        # ── 输入卷积（将图像从 in_channels 映射到 base_channels）──────────────
        self.input_conv = nn.Conv2d(config.in_channels, chs[0], kernel_size=3, padding=1)

        ctx = config.context_dim   # 0 表示无文本条件化

        # ── 编码器 ─────────────────────────────────────────────────────────
        self.down_blocks = nn.ModuleList()
        for i in range(len(chs) - 1):
            self.down_blocks.append(
                DownBlock(
                    in_ch=chs[i],
                    out_ch=chs[i + 1],
                    time_dim=time_dim,
                    num_res=config.num_res_blocks,
                    use_attn=False,
                    dropout=config.dropout,
                    context_dim=ctx,
                )
            )

        # ── 瓶颈 ────────────────────────────────────────────────────────────
        self.bottleneck = Bottleneck(chs[-1], time_dim, config.dropout,
                                     context_dim=ctx)

        # ── 解码器 ─────────────────────────────────────────────────────────
        self.up_blocks = nn.ModuleList()
        for i in range(len(chs) - 1, 0, -1):
            self.up_blocks.append(
                UpBlock(
                    in_ch=chs[i],
                    skip_ch=chs[i],
                    out_ch=chs[i - 1],
                    time_dim=time_dim,
                    num_res=config.num_res_blocks,
                    use_attn=False,
                    dropout=config.dropout,
                    context_dim=ctx,
                )
            )

        # ── 输出头 ─────────────────────────────────────────────────────────
        # 将最后的特征图（chs[0] 通道）映射回 out_channels
        num_groups = min(8, chs[0])
        self.output_head = nn.Sequential(
            nn.GroupNorm(num_groups, chs[0]),
            nn.SiLU(),
            nn.Conv2d(chs[0], config.out_channels, kernel_size=1),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        """
        前向传播：给定加噪图像 x_t、时间步 t 和可选的文本嵌入 context，预测噪声 ε。

        参数:
            x_t     (torch.Tensor): 加噪图像，形状 (B, C, H, W)。
            t       (torch.Tensor): 当前时间步，形状 (B,)，整数，值域 [0, T-1]。
            context (torch.Tensor | None): 文本嵌入，形状 (B, L, context_dim)。
                L = 文本 token 数（如 CLIP 的 77），context_dim 由文本编码器决定。
                传入 None 时退化为无条件生成（与原来行为完全一致）。

                典型用法：
                    # 有文本条件（文字生成图像）
                    text_emb = clip_encoder("一只橙色的猫")   # (B, 77, 768)
                    noise_pred = unet(x_t, t, context=text_emb)

                    # 无条件（无提示词，自由生成）
                    noise_pred = unet(x_t, t)                 # context=None

        返回:
            torch.Tensor: 预测的噪声 ε_θ(x_t, t, context)，形状 (B, C, H, W)。
        """
        # ── Step 1: 计算时间嵌入 ─────────────────────────────────────────────
        time_emb = self.time_embedding(t)     # (B,) → (B, time_emb_dim)

        # ── Step 2: 输入卷积 ─────────────────────────────────────────────────
        h = self.input_conv(x_t)              # (B, C, H, W) → (B, base_ch, H, W)

        # ── Step 3: 编码器（下采样，收集跳跃连接）───────────────────────────────
        # 每个 DownBlock 内部通过交叉注意力将 context 注入图像特征
        skips: list[torch.Tensor] = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, time_emb, context)
            skips.append(skip)

        # ── Step 4: 瓶颈（语义最浓缩处，文本注入效果最强）──────────────────────
        h = self.bottleneck(h, time_emb, context)

        # ── Step 5: 解码器（上采样，消耗跳跃连接）───────────────────────────────
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            h = up_block(h, skip, time_emb, context)

        # ── Step 6: 输出头 ─────────────────────────────────────────────────
        eps_pred = self.output_head(h)        # (B, base_ch, H, W) → (B, C, H, W)

        return eps_pred
