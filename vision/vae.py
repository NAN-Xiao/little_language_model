"""
2D VAE (Variational Autoencoder) 模块 —— 图像压缩/解压 学习用实现
===================================================================

ViT 处理的是单张图片，没有时间维度，所以不需要 3D VAE。
本文件实现的是 2D VAE，即 Stable Diffusion 系列使用的图像 VAE。

本文件实现了：
  1. ResBlock2D              — 2D 残差块（VAE 的基本构建单元）
  2. AttentionBlock          — 自注意力块（在最小分辨率加入全局注意力）
  3. Downsample2D            — 2D 下采样（空间压缩）
  4. Upsample2D              — 2D 上采样（空间恢复）
  5. Encoder2D               — 编码器：图像像素 → latent 分布 (μ, σ)
  6. Decoder2D               — 解码器：latent → 图像像素
  7. ImageVAE                — 完整的 2D VAE（Encoder + 重参数化 + Decoder）
  8. ImageVAEConfig          — 配置

=== 2D VAE vs 3D VAE 对比 ===

  ┌────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │  2D VAE (本文件, Stable Diffusion 使用):                               │
  │    - 处理单张图片: (B, 3, H, W)                                       │
  │    - 使用 Conv2d 做空间压缩                                           │
  │    - 无时间维度，不需要因果卷积                                       │
  │    - 压缩: (B, 3, 256, 256) → (B, 4, 32, 32) — 空间 8× 压缩        │
  │                                                                        │
  │  3D VAE (3dvae.py, CogVideoX/Sora 使用):                              │
  │    - 处理视频: (B, 3, T, H, W)                                        │
  │    - 使用 Conv3d / CausalConv3d                                       │
  │    - 同时压缩空间和时间                                               │
  │    - 压缩: (B, 3, 96, 1088, 1920) → (B, 4, 24, 136, 240)            │
  │                                                                        │
  │  对比:                                                                 │
  │    ┌──────────────┬──────────────┬──────────────────┐                  │
  │    │              │ 2D VAE       │ 3D VAE           │                  │
  │    │ 输入         │ 图片(H,W)    │ 视频(T,H,W)      │                  │
  │    │ 卷积         │ Conv2d       │ Conv3d/Causal3D  │                  │
  │    │ 压缩维度     │ 只压空间     │ 压空间+时间      │                  │
  │    │ 下采样       │ stride 2D    │ stride 3D        │                  │
  │    │ 因果性       │ 不需要       │ 时间维因果       │                  │
  │    │ 注意力       │ 2D空间注意力 │ 3D时空注意力     │                  │
  │    │ 典型应用     │ SD/SDXL/FLUX │ Sora/CogVideoX   │                  │
  │    └──────────────┴──────────────┴──────────────────┘                  │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘

=== 为什么 ViT 场景下也需要 VAE？ ===

  ViT 本身做分类/理解时不需要 VAE。
  但在图像生成场景（如 Stable Diffusion）中：
    1. 直接在像素空间做扩散太慢（256×256×3 = 196K 像素）
    2. VAE 先把图像压到 32×32×4 = 4K 的 latent 空间
    3. 扩散模型（U-Net 或 DiT）在 latent 空间里工作 → 快 50 倍+
    4. 生成完 latent 后，VAE decoder 再解码回像素

  ┌────────────────────────────────────────────────────────┐
  │  图像生成流程 (Latent Diffusion):                      │
  │                                                        │
  │  训练:                                                 │
  │    图片 → VAE.encode → z₀ → 加噪 → zₜ → 模型预测噪声  │
  │                                                        │
  │  推理:                                                 │
  │    zₜ(随机噪声) → 模型去噪 → z₀ → VAE.decode → 图片   │
  └────────────────────────────────────────────────────────┘

=== 整体架构（本文件实现的 2D VAE）===

  ┌────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │  Encoder2D:                                                            │
  │    (B,3,256,256) → Conv(3→128) → ResBlock×2                           │
  │    → Downsample → (B,128,128,128) → ResBlock×2                        │
  │    → Downsample → (B,256,64,64)   → ResBlock×2                        │
  │    → Downsample → (B,512,32,32)   → ResBlock×2                        │
  │    → Attention → ResBlock → Conv(512→8) → (μ, logvar)                 │
  │    输出: (B, 8, 32, 32) → split → μ(B,4,32,32) + logvar(B,4,32,32)  │
  │                                                                        │
  │  Decoder2D:                                                            │
  │    (B,4,32,32) → Conv(4→512) → ResBlock → Attention                   │
  │    → ResBlock×2 → Upsample → (B,512,64,64) → ResBlock×2              │
  │    → Upsample → (B,256,128,128) → ResBlock×2                          │
  │    → Upsample → (B,128,256,256) → ResBlock×2                          │
  │    → GroupNorm → SiLU → Conv(128→3) → (B,3,256,256)                   │
  │                                                                        │
  │  压缩: 256×256×3 = 196,608 → 32×32×4 = 4,096 (约 48× 压缩)          │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘

参考:
  - "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013) — VAE 原论文
  - Stable Diffusion VAE — CompVis/LDM 开源实现
  - "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# 0. 配置
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class ImageVAEConfig:
    """
    2D VAE 配置。

    默认按 Stable Diffusion 常用设置：
      输入: (B, 3, 256, 256)
      输出: (B, 4, 32, 32)  — 空间 8× 压缩
    """
    in_channels: int = 3              # 输入通道（RGB）
    latent_channels: int = 4          # latent 通道数（SD=4, SD3=16）
    base_channels: int = 128          # 编码器第一层通道数
    channel_multipliers: tuple = (1, 2, 4, 4)   # 每级的通道倍数 → 128, 256, 512, 512
    num_res_blocks: int = 2           # 每级的残差块数量
    attention_at_level: tuple = (False, False, False, True)  # 哪些级加自注意力
    dropout: float = 0.0              # dropout

    @property
    def spatial_compression(self) -> int:
        """
        空间总压缩率。

        每级下采样 2×, 总共 len(channel_multipliers)-1 次下采样
        (最后一级不下采样)
        """
        return 2 ** (len(self.channel_multipliers) - 1)  # 2³ = 8

    @property
    def encoder_channels(self) -> list[int]:
        """编码器每级的通道数。"""
        return [self.base_channels * m for m in self.channel_multipliers]


# ═══════════════════════════════════════════════════════════════════════════
# 1. ResBlock2D — 2D 残差块
# ═══════════════════════════════════════════════════════════════════════════
class ResBlock2D(nn.Module):
    """
    2D 残差块 — VAE 的基本构建单元。

    结构:
      x ─────────────────────────────────(+)──→ out
      │                                   ↑
      └→ GroupNorm → SiLU → Conv2d       │
         → GroupNorm → SiLU → Conv2d ────┘
         (+ 1×1 shortcut if in≠out)

    === 为什么用 GroupNorm？ ===

    ┌────────────────────────────────────────────────────────────────┐
    │                                                                │
    │  BatchNorm:  对整个 batch 的同一通道归一化                      │
    │    → 需要大 batch 才稳定，VAE 训练 batch 通常只有 4~16 → 不适合│
    │                                                                │
    │  LayerNorm:  对单个样本的所有通道一起归一化                      │
    │    → 不区分通道，对 CNN 特征不太合适                            │
    │                                                                │
    │  GroupNorm:  把通道分成 G 组，每组内部归一化                     │
    │    → 不依赖 batch 大小 ✓                                       │
    │    → 保留通道间差异 ✓                                          │
    │    → CNN 的最佳选择 ✓                                          │
    │                                                                │
    │  示例 (channels=256, groups=32):                               │
    │    256 个通道分成 32 组，每组 8 个通道                          │
    │    在每组的 8 个通道内部做归一化                                │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘

    === 为什么用 SiLU 而不是 ReLU？ ===

    SiLU(x) = x · σ(x) = x · sigmoid(x)

    - 处处可导（ReLU 在 0 点不可导）
    - 有负值输出（ReLU 把负值全截断为 0，丢失信息）
    - 实验表明在生成模型中效果更好
    - 也叫 Swish 激活函数

    参数:
        in_ch  (int): 输入通道
        out_ch (int): 输出通道（可与 in_ch 不同）
        dropout (float): dropout
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 当输入/输出通道不同时，需要 1×1 卷积做 shortcut
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, H, W) → (B, C_out, H, W)"""
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return self.shortcut(x) + h


# ═══════════════════════════════════════════════════════════════════════════
# 2. AttentionBlock — 自注意力块
# ═══════════════════════════════════════════════════════════════════════════
class AttentionBlock(nn.Module):
    """
    空间自注意力块 — 在最小分辨率的 feature map 上做全局注意力。

    === 为什么在 VAE 中需要注意力？ ===

    卷积的感受野是局部的（3×3 卷积只看邻近像素）。
    在低分辨率（如 32×32）加入自注意力，可以建模全局关系：
      - 图像远处的物体之间的关系
      - 全局色调/光照的一致性
      - 结构性特征（对称性等）

    === 为什么只在最低分辨率加？ ===

    注意力的复杂度是 O(n²)，n = H × W
    在 256×256 上: n = 65536 → 太慢
    在 32×32 上:   n = 1024  → 可接受

    === 具体操作 ===

    ┌──────────────────────────────────────────────────────┐
    │  输入: (B, C, H, W) — 例如 (B, 512, 32, 32)        │
    │                                                      │
    │  1. GroupNorm                                        │
    │  2. 用 1×1 Conv 产生 Q, K, V: 各 (B, C, H, W)       │
    │  3. reshape 为序列: (B, C, H*W) → (B, H*W, C)       │
    │  4. 标准自注意力: softmax(QK^T/√d) × V              │
    │  5. reshape 回空间: (B, H*W, C) → (B, C, H, W)      │
    │  6. 1×1 Conv 输出 + 残差                             │
    │                                                      │
    │  本质: 把 2D feature map 当作"长度 H*W 的序列"做注意力│
    └──────────────────────────────────────────────────────┘

    参数:
        channels (int): 输入/输出通道数
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.out = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C, H, W)"""
        B, C, H, W = x.shape
        h = self.norm(x)

        q = self.q(h).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        k = self.k(h).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        v = self.v(h).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)

        # 标准缩放点积注意力
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, HW, HW)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)  # (B, HW, C)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H, W)
        out = self.out(out)

        return x + out


# ═══════════════════════════════════════════════════════════════════════════
# 3. Downsample2D / Upsample2D — 空间维度缩放
# ═══════════════════════════════════════════════════════════════════════════
class Downsample2D(nn.Module):
    """
    2D 下采样 — 空间维度缩小 2×。

    ┌──────────────────────────────────────────────────────────────┐
    │  方法: stride=2 的 Conv2d                                    │
    │                                                              │
    │  (B, C, H, W) → (B, C, H/2, W/2)                           │
    │                                                              │
    │  为什么用 conv 而不是 pooling？                               │
    │    MaxPool: 丢弃大量信息（只保留最大值）                      │
    │    AvgPool: 模糊细节                                         │
    │    Conv(stride=2): 可学习的下采样，保留有用信息 ✓              │
    │                                                              │
    │  Stable Diffusion 的做法:                                    │
    │    先 pad 再 conv，避免边界信息丢失                           │
    │    pad(0,1,0,1) + Conv(stride=2, padding=0) → 非对称 padding │
    └──────────────────────────────────────────────────────────────┘
    """

    def __init__(self, channels: int):
        super().__init__()
        # 使用非对称 padding（SD 的做法）: 右边和下边多 pad 1
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 非对称 padding: (left=0, right=1, top=0, bottom=1)
        x = F.pad(x, (0, 1, 0, 1))
        return self.conv(x)


class Upsample2D(nn.Module):
    """
    2D 上采样 — 空间维度放大 2×。

    ┌──────────────────────────────────────────────────────────────┐
    │  方法: nearest 插值 + Conv2d                                 │
    │                                                              │
    │  (B, C, H, W) → interpolate → (B, C, 2H, 2W)              │
    │  → Conv2d → (B, C, 2H, 2W)                                 │
    │                                                              │
    │  为什么不用 ConvTranspose2d (转置卷积)?                      │
    │    转置卷积容易产生"棋盘格"伪影 (checkerboard artifacts)     │
    │    nearest + conv 更稳定，生成质量更好                       │
    │                                                              │
    │  棋盘格伪影:                                                 │
    │    ┌─┬─┬─┬─┐      ┌─┬─┬─┬─┐                               │
    │    │▓│░│▓│░│  vs  │▓│▓│▓│▓│                               │
    │    │░│▓│░│▓│      │▓│▓│▓│▓│                               │
    │    │▓│░│▓│░│      │▓│▓│▓│▓│                               │
    │    └─┴─┴─┴─┘      └─┴─┴─┴─┘                               │
    │    转置卷积伪影    nearest+conv 无伪影                       │
    └──────────────────────────────────────────────────────────────┘
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Encoder2D — 编码器
# ═══════════════════════════════════════════════════════════════════════════
class Encoder2D(nn.Module):
    """
    2D VAE 编码器。

    将图像像素压缩到 latent 分布参数 (μ, log σ²)。

    ┌────────────────────────────────────────────────────────────────────────┐
    │  输入: (B, 3, 256, 256) — 图像                                       │
    │                                                                        │
    │  Level 0: Conv(3→128) + ResBlock(128,128)×2                           │
    │           → (B, 128, 256, 256)                                        │
    │           Downsample → (B, 128, 128, 128)                             │
    │                                                                        │
    │  Level 1: ResBlock(128→256)×2                                         │
    │           → (B, 256, 128, 128)                                        │
    │           Downsample → (B, 256, 64, 64)                               │
    │                                                                        │
    │  Level 2: ResBlock(256→512)×2                                         │
    │           → (B, 512, 64, 64)                                          │
    │           Downsample → (B, 512, 32, 32)                               │
    │                                                                        │
    │  Level 3: ResBlock(512,512)×2 + Attention                             │
    │           → (B, 512, 32, 32)  (最后一级不下采样)                      │
    │                                                                        │
    │  Mid:   ResBlock → Attention → ResBlock                               │
    │  Head:  GroupNorm → SiLU → Conv(512→8)                                │
    │         → (B, 8, 32, 32)                                              │
    │         split → μ(B,4,32,32) + logvar(B,4,32,32)                     │
    │                                                                        │
    │  总压缩: 空间 8×, 通道 3→4                                            │
    └────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, cfg: ImageVAEConfig):
        super().__init__()
        channels = cfg.encoder_channels  # [128, 256, 512, 512]

        # 初始卷积
        self.conv_in = nn.Conv2d(cfg.in_channels, channels[0], kernel_size=3, padding=1)

        # 编码器各级
        self.down_blocks = nn.ModuleList()
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            block_layers = nn.ModuleList()

            # 残差块
            for j in range(cfg.num_res_blocks):
                res_in = in_ch if j == 0 else out_ch
                block_layers.append(ResBlock2D(res_in, out_ch, cfg.dropout))

            # 注意力（只在指定级加）
            if cfg.attention_at_level[i]:
                block_layers.append(AttentionBlock(out_ch))

            block = nn.ModuleDict({"layers": block_layers})

            # 下采样（最后一级不下采样）
            if i < len(channels) - 1:
                block["downsample"] = Downsample2D(out_ch)

            self.down_blocks.append(block)
            in_ch = out_ch

        # 中间瓶颈层（attention 夹在两个 resblock 之间）
        self.mid = nn.ModuleList([
            ResBlock2D(channels[-1], channels[-1], cfg.dropout),
            AttentionBlock(channels[-1]),
            ResBlock2D(channels[-1], channels[-1], cfg.dropout),
        ])

        # 输出头
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, channels[-1]),
            nn.SiLU(),
            nn.Conv2d(channels[-1], 2 * cfg.latent_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        返回: (B, 2*latent_channels, H/8, W/8) — (μ, logvar) 拼在通道维
        """
        x = self.conv_in(x)

        for block in self.down_blocks:
            for layer in block["layers"]:
                x = layer(x)
            if "downsample" in block:
                x = block["downsample"](x)

        for layer in self.mid:
            x = layer(x)

        return self.conv_out(x)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Decoder2D — 解码器
# ═══════════════════════════════════════════════════════════════════════════
class Decoder2D(nn.Module):
    """
    2D VAE 解码器 — Encoder2D 的镜像结构。

    ┌────────────────────────────────────────────────────────────────────────┐
    │  输入: (B, 4, 32, 32) — latent                                       │
    │                                                                        │
    │  Conv(4→512) → (B, 512, 32, 32)                                      │
    │                                                                        │
    │  Mid: ResBlock → Attention → ResBlock                                 │
    │                                                                        │
    │  Level 3: ResBlock(512,512)×3 + Attention                             │
    │           → (B, 512, 32, 32)                                          │
    │           Upsample → (B, 512, 64, 64)                                │
    │                                                                        │
    │  Level 2: ResBlock(512→512)×3                                         │
    │           → (B, 512, 64, 64)                                          │
    │           Upsample → (B, 512, 128, 128)                              │
    │                                                                        │
    │  Level 1: ResBlock(512→256)×3                                         │
    │           → (B, 256, 128, 128)                                        │
    │           Upsample → (B, 256, 256, 256)                              │
    │                                                                        │
    │  Level 0: ResBlock(256→128)×3                                         │
    │           → (B, 128, 256, 256)                                        │
    │                                                                        │
    │  Head: GroupNorm → SiLU → Conv(128→3) → (B, 3, 256, 256)            │
    │                                                                        │
    │  注意: 解码器每级有 num_res_blocks+1 个 ResBlock（比编码器多一个）     │
    │  这是 Stable Diffusion VAE 的设计，给解码器更多容量以提高重建质量     │
    └────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, cfg: ImageVAEConfig):
        super().__init__()
        channels = cfg.encoder_channels  # [128, 256, 512, 512]
        rev_channels = list(reversed(channels))  # [512, 512, 256, 128]

        # 输入投影
        self.conv_in = nn.Conv2d(cfg.latent_channels, rev_channels[0], kernel_size=3, padding=1)

        # 中间瓶颈层
        self.mid = nn.ModuleList([
            ResBlock2D(rev_channels[0], rev_channels[0], cfg.dropout),
            AttentionBlock(rev_channels[0]),
            ResBlock2D(rev_channels[0], rev_channels[0], cfg.dropout),
        ])

        # 解码器各级（编码器的镜像，但多一个 ResBlock）
        rev_attention = list(reversed(cfg.attention_at_level))
        self.up_blocks = nn.ModuleList()
        in_ch = rev_channels[0]
        for i, out_ch in enumerate(rev_channels):
            block_layers = nn.ModuleList()

            # 残差块（比编码器多一个）
            for j in range(cfg.num_res_blocks + 1):
                res_in = in_ch if j == 0 else out_ch
                block_layers.append(ResBlock2D(res_in, out_ch, cfg.dropout))

            # 注意力
            if rev_attention[i]:
                block_layers.append(AttentionBlock(out_ch))

            block = nn.ModuleDict({"layers": block_layers})

            # 上采样（最后一级不上采样）
            if i < len(rev_channels) - 1:
                block["upsample"] = Upsample2D(out_ch)

            self.up_blocks.append(block)
            in_ch = out_ch

        # 输出头
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, rev_channels[-1]),
            nn.SiLU(),
            nn.Conv2d(rev_channels[-1], cfg.in_channels, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, latent_channels, H', W')
        返回: (B, 3, H, W)
        """
        x = self.conv_in(z)

        for layer in self.mid:
            x = layer(x)

        for block in self.up_blocks:
            for layer in block["layers"]:
                x = layer(x)
            if "upsample" in block:
                x = block["upsample"](x)

        return self.conv_out(x)


# ═══════════════════════════════════════════════════════════════════════════
# 6. ImageVAE — 完整的 2D VAE
# ═══════════════════════════════════════════════════════════════════════════
class ImageVAE(nn.Module):
    """
    完整的 2D VAE：编码器 + 重参数化 + 解码器。

    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  === 训练 (forward) ===                                                │
    │                                                                        │
    │    image (B,3,H,W)                                                     │
    │       │                                                                │
    │       ▼                                                                │
    │    Encoder2D → (μ, logvar)                                             │
    │       │                                                                │
    │       ▼                                                                │
    │    重参数化: z = μ + σ·ε,  ε ~ N(0,I)                                 │
    │       │                                                                │
    │       ▼                                                                │
    │    Decoder2D → image_recon (B,3,H,W)                                  │
    │       │                                                                │
    │       ▼                                                                │
    │    Loss = MSE(image, image_recon)                                      │
    │         + β · KL(N(μ,σ²) ‖ N(0,I))                                   │
    │         + λ_lpips · LPIPS(image, image_recon)  ← 感知损失(可选)       │
    │         + λ_gan · GAN_loss                     ← 对抗损失(可选)       │
    │                                                                        │
    │  === 推理/扩散模型训练 (encode) ===                                    │
    │    image → Encoder → μ  (确定性编码，不采样)                          │
    │                                                                        │
    │  === 扩散模型推理 (decode) ===                                         │
    │    z₀ (去噪后的 latent) → Decoder → image                             │
    │                                                                        │
    │  === scaling factor ===                                                │
    │    SD 系列在编码后会乘一个 scaling_factor (约 0.18215):               │
    │    z_scaled = z * scaling_factor                                       │
    │    这样让 latent 的方差接近 1，有利于扩散模型的数值稳定性             │
    │    解码前要除回来: z = z_scaled / scaling_factor                       │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘

    参数:
        cfg (ImageVAEConfig): 配置
    """

    def __init__(self, cfg: ImageVAEConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = ImageVAEConfig()
        self.cfg = cfg
        self.encoder = Encoder2D(cfg)
        self.decoder = Decoder2D(cfg)

        # Stable Diffusion 使用的 scaling factor
        # 训练完 VAE 后，统计 latent 的标准差，取倒数作为 scaling_factor
        # 使 latent 的方差接近 1
        self.scaling_factor = 0.18215

    def encode(self, x: torch.Tensor, sample: bool = True, apply_scaling: bool = True
               ) -> dict[str, torch.Tensor]:
        """
        编码图像到 latent。

        x: (B, 3, H, W)
        sample:        True=从 N(μ,σ²) 采样, False=返回 μ
        apply_scaling: True=乘以 scaling_factor（给扩散模型用）

        返回 dict:
          z:      (B, C_latent, H/8, W/8) — latent
          mu:     (B, C_latent, H/8, W/8) — 均值
          logvar: (B, C_latent, H/8, W/8) — log 方差
        """
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)

        if sample:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        if apply_scaling:
            z = z * self.scaling_factor

        return {"z": z, "mu": mu, "logvar": logvar}

    def decode(self, z: torch.Tensor, apply_scaling: bool = True) -> torch.Tensor:
        """
        从 latent 解码回图像。

        z: (B, C_latent, H/8, W/8)
        apply_scaling: True=先除以 scaling_factor（撤销编码时的缩放）

        返回: (B, 3, H, W)
        """
        if apply_scaling:
            z = z / self.scaling_factor
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        训练用的完整前向传播（不应用 scaling，直接重建）。

        x: (B, 3, H, W)

        返回 dict:
          recon:  (B, 3, H, W)  — 重建图像
          z:      latent
          mu:     均值
          logvar: log 方差

        使用示例:
            vae = ImageVAE()
            images = torch.randn(4, 3, 256, 256)
            out = vae(images)
            loss = image_vae_loss(out["recon"], images, out["mu"], out["logvar"])
            loss["total"].backward()
        """
        enc = self.encode(x, sample=True, apply_scaling=False)
        recon = self.decode(enc["z"], apply_scaling=False)
        return {
            "recon": recon,
            "z": enc["z"],
            "mu": enc["mu"],
            "logvar": enc["logvar"],
        }


# ═══════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════
def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL 散度: KL(N(μ,σ²) ‖ N(0,1))

    公式推导:
      KL = ∫ q(z) log(q(z)/p(z)) dz
         = ∫ N(μ,σ²) [log N(μ,σ²) - log N(0,1)] dz
         = ½ [-1 - log σ² + μ² + σ²]
         = -½ Σ (1 + log σ² - μ² - σ²)

    直觉:
      - μ² 项: 惩罚均值偏离 0 → 让 latent 中心在原点附近
      - σ² 项: 惩罚方差偏离 1 → 让 latent 的spread 适中
      - 整体效果: 让编码分布接近标准正态 N(0,1)

    返回: 标量 — batch 平均 KL
    """
    return (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean()


def image_vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """
    图像 VAE 训练损失。

    === 各项损失的作用 ===

    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  L_recon (MSE):                                                        │
    │    - 像素级别的重建精度                                                │
    │    - 问题: MSE 偏好模糊（平均像素值）                                 │
    │                                                                        │
    │  L_KL:                                                                 │
    │    - 让 latent 分布规则化                                              │
    │    - 权重很小 (1e-6)，否则会压缩 latent 表达能力                      │
    │                                                                        │
    │  L_LPIPS (本文件未实现，实际训练会加):                                  │
    │    - 用预训练 VGG 网络提取特征做比较                                   │
    │    - 比 MSE 更接近人眼感知                                             │
    │    - "长得像" vs "像素值接近"                                          │
    │                                                                        │
    │  L_GAN (本文件未实现，实际训练会加):                                    │
    │    - 加一个 PatchGAN 判别器                                            │
    │    - 让重建图像更锐利、更真实                                          │
    │    - 弥补 MSE 导致的模糊问题                                           │
    │                                                                        │
    │  实际 Stable Diffusion VAE 的完整损失:                                 │
    │    L = L_recon + 1e-6 · L_KL + 0.5 · L_LPIPS + 0.1 · L_GAN          │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘

    参数:
        recon:     重建图像
        target:    原始图像
        mu, logvar: 编码分布参数
        kl_weight:  KL 权重
    """
    recon_loss = F.mse_loss(recon, target)
    kl_loss = kl_divergence(mu, logvar)
    total = recon_loss + kl_weight * kl_loss
    return {"total": total, "recon_loss": recon_loss, "kl_loss": kl_loss}
