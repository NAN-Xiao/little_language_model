"""
简化版 3D VAE —— 学习练习用
============================

本文件实现一个极简的 3D VAE，用于理解 encode-decode 的完整流程。
不追求性能，只追求可读性。

═══════════════════════════════════════════════════════════════════════════════
【形象图解：Conv3d 怎么降采样/升采样】
═══════════════════════════════════════════════════════════════════════════════

一、视频是什么形状？
───────────────────

视频不是一个"平面图片"，而是一个"时空立方体"：

    时间 T=8 帧 → → →
    ┌─────────────────────────────────────┐
   /│  帧0  │  帧1  │  帧2  │ ... │  帧7  │
  / │       │       │       │     │       │
 H=64      │  高   │       │     │       │  ↑ 空间高度
│  │       │       │       │     │       │
│  │       │       │       │     │       │
│  ├───────┼───────┼───────┼─────┼───────┤
│ W=64     │       │       │     │       │  → 空间宽度
↓  │       │       │       │     │       │

整体形状: (B, C=3, T=8, H=64, W=64)
          B=几段视频, C=RGB三通道, T=8帧, H=64像素高, W=64像素宽


二、3D 卷积核是什么？
────────────────────

想象一个"微型探测器"，每次同时看"一小段时空"：

    卷积核大小 kernel=3×3×3:
      时间: 3帧  ← 同时看"前一帧+当前帧+后一帧"
      高度: 3像素
      宽度: 3像素

    可视化（只看一个通道的一个位置）:

        帧t-1      帧t       帧t+1
       ┌───┐     ┌───┐     ┌───┐
       │ ■ │     │ ■ │     │ ■ │   ← 每个■是一个3×3的小窗口
       └───┘     └───┘     └───┘
           ↓         ↓         ↓
              加权求和
                  ↓
             输出 1 个数

    这个"探测器"在 3 个通道上都有一套权重（3×3×3×3=81个参数），
    最终输出 1 个数到新的特征图里。


三、降采样（Downsample）—— "跳步跳着走"
─────────────────────────────────────────

普通卷积（stride=1）：探测器一步一步挨着走，输出和输入一样大。

降采样卷积（stride=2）：探测器每次走两步，跳过一格，输出变一半。

    stride=1（普通）:              stride=2（降采样）:

    输入: 8个位置                   输入: 8个位置
    1 2 3 4 5 6 7 8               1 2 3 4 5 6 7 8
    ■ ■ ■ ■ ■ ■ ■ ■               ■   ■   ■   ■
    ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑               ↑   ↑   ↑   ↑
    输出: 8个位置                   输出: 4个位置

    时间维 stride=2: 8帧 → 4帧
    空间维 stride=2: 64×64 → 32×32

    Conv3d(kernel=3, stride=(2,2,2), padding=1):
      时间: 8 → 4  (跳步)
      高度: 64 → 32
      宽度: 64 → 32
      通道: 32 → 32 (不变，因为 in_channels = out_channels = 32)

    整体: (B,32,8,64,64) → (B,32,4,32,32)


四、升采样（Upsample）—— "先抻大再细化"
─────────────────────────────────────────

升采样不能直接用"反卷积"（容易出棋盘格伪影），标准做法是两步：

    Step 1: 最近邻插值 —— 把每个像素复制成 2×2 块

        输入 2×2:              插值后 4×4:
        ┌───┬───┐              ┌───┬───┬───┬───┐
        │ a │ b │      →       │ a │ a │ b │ b │
        ├───┼───┤              ├───┼───┼───┼───┤
        │ c │ d │              │ a │ a │ b │ b │
        └───┴───┘              ├───┼───┼───┼───┤
                               │ c │ c │ d │ d │
                               ├───┼───┼───┼───┤
                               │ c │ c │ d │ d │
                               └───┴───┴───┴───┘
        简单粗暴但有效，没有引入新信息，只是"抻大"。

    Step 2: 卷积精化 —— 用 CausalConv3d 在抻大后的数据上做平滑/融合

        抻大后的 (B,32,4,32,32)
            → Conv3d(kernel=3, stride=1, padding=1)
            → (B,32,4,32,32)
            形状不变，但像素值被邻居们重新加权混合了。

    时间维 + 空间维同时抻大:
      时间: 4帧 → 8帧（插值后帧数翻倍）
      高度: 32 → 64
      宽度: 32 → 64

    整体: (B,64,4,16,16) → 插值 → (B,64,8,32,32) → 卷积 → (B,64,8,32,32)


五、维度变化全览（Encoder 一路往下压）
────────────────────────────────────────

    层                操作                    输出形状
    ────             ──────                  ───────────
    输入              原始视频                 (2, 3,  8, 64, 64)
                       ↓
    conv_in           Conv3d(3→32)            (2, 32, 8, 64, 64)
                       ↓ 空间 stride=2
    down1             ResBlock×2 + Downsample  (2, 32, 8, 32, 32)
                       ↓ 空间+时间 stride=2
    down2             ResBlock×2 + Downsample  (2, 64, 4, 16, 16)
                       ↓ 空间 stride=2
    down3             ResBlock×2 + Downsample  (2, 128,4, 8,  8)
                       ↓
    conv_out          Conv3d(128→8)           (2, 8,  4, 8,  8)

    8 = 前4通道(μ) + 后4通道(logvar)

    总压缩: 64×64×8 = 32768  →  8×8×4 = 256，空间压缩 64 倍

═══════════════════════════════════════════════════════════════════════════════

数据流:
  输入视频 (B, 3, 8, 64, 64)
    → Encoder: Conv3d + ResBlock + Downsample
    → (μ, logvar) (B, 8, 4, 8, 8)  [μ和logvar各4通道]
    → 重参数化 → z (B, 4, 4, 8, 8)
    → Decoder: Conv3d + ResBlock + Upsample
    → 重建视频 (B, 3, 8, 64, 64)
"""




from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# 基础组件（已提供，直接使用）
# ═══════════════════════════════════════════════════════════════════════════

class  SimpleCausalConv3d(nn.Module):
    """简化版因果3D卷积。时间维只看过去和当前。"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.temporal_pad = kernel_size - 1
        self.conv = nn.Conv3d(
            in_ch, out_ch, kernel_size=kernel_size,
            padding=(0, kernel_size // 2, kernel_size // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 0, 0, 0, self.temporal_pad, 0))
        return self.conv(x)


class SimpleResBlock3D(nn.Module):
    """简化版3D残差块。"""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            SimpleCausalConv3d(channels, channels),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            SimpleCausalConv3d(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SimpleDownsample3D(nn.Module):
    """简化版3D下采样：空间和/或时间各缩2倍。"""

    def __init__(self, channels: int, spatial: bool = True, temporal: bool = False):
        super().__init__()
        stride_t = 2 if temporal else 1
        stride_s = 2 if spatial else 1
        self.conv = nn.Conv3d(
            channels, channels, kernel_size=3,
            stride=(stride_t, stride_s, stride_s), padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SimpleUpsample3D(nn.Module):
    """简化版3D上采样：先插值再卷积。"""

    def __init__(self, channels: int, spatial: bool = True, temporal: bool = False):
        super().__init__()
        self.spatial = spatial
        self.temporal = temporal
        self.conv = SimpleCausalConv3d(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = (
            2.0 if self.temporal else 1.0,
            2.0 if self.spatial else 1.0,
            2.0 if self.spatial else 1.0,
        )
        x = F.interpolate(x, scale_factor=scale, mode="nearest")
        return self.conv(x)


# ═══════════════════════════════════════════════════════════════════════════
# 【学习笔记：3通道怎么变成8通道？——卷积的通道变换原理】
# ═══════════════════════════════════════════════════════════════════════════
#
#   Conv3d(in=3, out=8, kernel=3×3×3) 的形象理解：
#
#   "雇了 8 个特征分析师，每个分析师在 3 个输入通道上各派一个探测器
#    （3×3×3 卷积核），把扫到的信息加权求和 + 偏置，变成 1 个数。
#    8 个分析师同时干，每个位置产出 8 个数。
#    核在时空上滑动一遍，最终得到 8 个通道的输出。"
#
#   具体计算（以输出第 0 通道、位置 (t,h,w) 为例）：
#
#     输出[0,t,h,w] = 红核⊙红窗口 + 绿核⊙绿窗口 + 蓝核⊙蓝窗口 + b₀
#                   ↑                              ↑
#              (3×3×3 卷积)                    (可学习偏置)
#
#   参数数量：
#     weight: out_ch × in_ch × k_t × k_h × k_w = 8 × 3 × 3 × 3 × 3 = 648
#     bias:   out_ch = 8
#     总计:   656 个可学习参数
#
#   这就是为什么 3 通道能"升"到 8 通道：不是复制，是学习到的特征变换。
#
# ═══════════════════════════════════════════════════════════════════════════
# 编码器和解码器（已提供骨架）
# ═══════════════════════════════════════════════════════════════════════════

class SimpleEncoder3D(nn.Module):
    """
    简化版编码器。
    b c t h w
    数据流 (B, 3, 8, 64, 64):
      → Conv3d(3→32)                    → (B, 32, 8, 64, 64)
      → ResBlock×2                      → (B, 32, 8, 64, 64)
      → Downsample(空间2×)               → (B, 32, 8, 32, 32)
      → ResBlock(32→64)×2               → (B, 64, 8, 32, 32)
      → Downsample(空间2×+时间2×)        → (B, 64, 4, 16, 16)
      → ResBlock(64→128)×2              → (B, 128, 4, 16, 16)
      → Downsample(空间2×)               → (B, 128, 4, 8, 8)
      → Conv3d(128→8)                   → (B, 8, 4, 8, 8)

    输出: (B, 8, 4, 8, 8) — 前4通道是μ，后4通道是logvar
    """

    def __init__(self):
        super().__init__()
        self.conv_in = SimpleCausalConv3d(3, 32)

        self.down1 = nn.Sequential(
            SimpleResBlock3D(32),
            SimpleResBlock3D(32),
            SimpleDownsample3D(32, spatial=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(32, 64, 1),
            SimpleResBlock3D(64),
            SimpleResBlock3D(64),
            SimpleDownsample3D(64, spatial=True, temporal=True),
        )
        self.down3 = nn.Sequential(
            nn.Conv3d(64, 128, 1),
            SimpleResBlock3D(128),
            SimpleResBlock3D(128),
            SimpleDownsample3D(128, spatial=True),
        )
        self.conv_out = nn.Conv3d(128, 8, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.conv_out(x)
        return x


class SimpleDecoder3D(nn.Module):
    """
    简化版解码器（编码器的镜像）。

    输入: (B, 4, 4, 8, 8) — latent z
      → Conv3d(4→128)                   → (B, 128, 4, 8, 8)
      → Upsample(空间2×)                 → (B, 128, 4, 16, 16)
      → ResBlock(128→64)×2              → (B, 64, 4, 16, 16)
      → Upsample(空间2×+时间2×)          → (B, 64, 8, 32, 32)
      → ResBlock(64→32)×2               → (B, 32, 8, 32, 32)
      → Upsample(空间2×)                 → (B, 32, 8, 64, 64)
      → ResBlock×2                      → (B, 32, 8, 64, 64)
      → Conv3d(32→3)                    → (B, 3, 8, 64, 64)
    """

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv3d(4, 128, kernel_size=1)

        self.up1 = nn.Sequential(
            SimpleUpsample3D(128, spatial=True),
            SimpleResBlock3D(128),
            nn.Conv3d(128, 64, 1),
            SimpleResBlock3D(64),
        )
        self.up2 = nn.Sequential(
            SimpleUpsample3D(64, spatial=True, temporal=True),
            SimpleResBlock3D(64),
            nn.Conv3d(64, 32, 1),
            SimpleResBlock3D(32),
        )
        self.up3 = nn.Sequential(
            SimpleUpsample3D(32, spatial=True),
            SimpleResBlock3D(32),
            SimpleResBlock3D(32),
        )
        self.conv_out = nn.Conv3d(32, 3, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(z)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.conv_out(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# 完整 VAE（需要你完成的练习）
# ═══════════════════════════════════════════════════════════════════════════

class SimpleVideoVAE(nn.Module):
    """
    简化版 3D VAE：编码器 + 重参数化 + 解码器。

    你需要完成:
      1. encode() —— 从视频编码到 (μ, logvar)，再重参数化采样得到 z
      2. encode_deterministic() —— 确定性编码（推理用，直接返回 μ）
      3. forward() —— 完整训练前向：视频 → 重建视频 + 损失

    维度参考:
      输入视频: (B, 3, 8, 64, 64)
      编码输出: (B, 8, 4, 8, 8)  [前4通道μ，后4通道logvar]
      latent z: (B, 4, 4, 8, 8)
      重建视频: (B, 3, 8, 64, 64)
    """

    def __init__(self):
        super().__init__()
        self.encoder = SimpleEncoder3D()
        self.decoder = SimpleDecoder3D()

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        编码视频到 latent（训练用，带随机采样）。

        参数:
            x: (B, 3, T, H, W) — 原始视频

        返回 dict:
            z:      (B, 4, T', H', W') — 采样后的 latent
            mu:     (B, 4, T', H', W') — 均值
            logvar: (B, 4, T', H', W') — log 方差
        """
        h = self.encoder(x)  # (B, 8, T', H', W')

        # TODO(human): 把 h 拆成 mu 和 logvar，然后做重参数化采样
        #
        # 提示:
        #   1. h 的形状是 (B, 8, T', H', W')，前4通道是 mu，后4通道是 logvar
        #   2. 用 torch.chunk(2, dim=1) 把 h 拆成两份
        #   3. 重参数化: z = mu + exp(0.5 * logvar) * eps, 其中 eps ~ N(0, I)
        #   4. eps 可以用 torch.randn_like(std) 生成
        #
        # 你的代码写在这里:
        # --------------------------------------------------------
        # mu, logvar = ...
        # std = ...
        # eps = ...
        # z = ...
        # --------------------------------------------------------

        return {"z": z, "mu": mu, "logvar": logvar}

    def encode_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """
        确定性编码（推理/DiT训练用）。

        与 encode() 的区别:
          - encode() 用于训练，从 N(μ,σ²) 采样，引入随机性
          - encode_deterministic() 用于推理，直接返回 μ，没有随机性

        为什么推理时不需要采样？
          因为 VAE 训练好后，KL loss 已经把 latent 分布压到了接近 N(0,1)。
          μ 就是分布的均值（最可能的点），直接用 μ 比采样更稳定。

        参数:
            x: (B, 3, T, H, W) — 原始视频

        返回:
            z: (B, 4, T', H', W') — 确定性 latent
        """
        # TODO(human): 实现确定性编码
        #
        # 提示:
        #   1. 先用 self.encoder(x) 得到 h
        #   2. 把 h 拆成 mu 和 logvar（和 encode() 一样）
        #   3. 直接返回 mu，不做采样
        #
        # 你的代码写在这里:
        # --------------------------------------------------------
        # h = ...
        # mu = ...
        # return mu
        # --------------------------------------------------------
        pass  # 替换这行

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """从 latent 解码回视频。"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        训练用的完整前向传播。

        参数:
            x: (B, 3, T, H, W) — 原始视频

        返回 dict:
            recon:  (B, 3, T, H, W) — 重建视频
            z:      latent
            mu:     均值
            logvar: log 方差
        """
        enc = self.encode(x)
        recon = self.decode(enc["z"])
        return {
            "recon": recon,
            "z": enc["z"],
            "mu": enc["mu"],
            "logvar": enc["logvar"],
        }


# ═══════════════════════════════════════════════════════════════════════════
# 损失函数（已提供）
# ═══════════════════════════════════════════════════════════════════════════

def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(μ,σ²) ‖ N(0,1)) = -½ Σ(1 + logσ² - μ² - σ²)"""
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """VAE 总损失 = 重建损失 + KL 权重 × KL 散度"""
    recon_loss = F.mse_loss(recon, target)
    kl_loss = kl_divergence(mu, logvar)
    total = recon_loss + kl_weight * kl_loss
    return {"total": total, "recon_loss": recon_loss, "kl_loss": kl_loss}


# ═══════════════════════════════════════════════════════════════════════════
# 测试代码（运行验证你的实现）
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 创建模型和测试输入
    vae = SimpleVideoVAE()
    video = torch.randn(2, 3, 8, 64, 64)

    print("=" * 60)
    print("测试 1: encode() —— 训练用（带采样）")
    print("=" * 60)
    enc = vae.encode(video)
    print(f"  mu shape:     {enc['mu'].shape}")
    print(f"  logvar shape: {enc['logvar'].shape}")
    print(f"  z shape:      {enc['z'].shape}")

    print("\n" + "=" * 60)
    print("测试 2: encode_deterministic() —— 推理用（无采样）")
    print("=" * 60)
    z_det = vae.encode_deterministic(video)
    print(f"  z shape:      {z_det.shape}")

    print("\n" + "=" * 60)
    print("测试 3: forward() —— 完整训练前向")
    print("=" * 60)
    out = vae(video)
    print(f"  recon shape:  {out['recon'].shape}")

    # 验证 shape
    assert out["recon"].shape == video.shape, "重建视频 shape 必须和输入一致！"
    assert enc["z"].shape == (2, 4, 4, 8, 8), "latent z shape 必须是 (B, 4, 4, 8, 8)！"

    print("\n" + "=" * 60)
    print("测试 4: 损失计算")
    print("=" * 60)
    losses = vae_loss(out["recon"], video, out["mu"], out["logvar"])
    print(f"  recon_loss: {losses['recon_loss'].item():.4f}")
    print(f"  kl_loss:    {losses['kl_loss'].item():.4f}")
    print(f"  total:      {losses['total'].item():.4f}")

    print("\n✓ 所有测试通过！")
