"""
3D VAE (Variational Autoencoder) 模块 —— 视频压缩/解压 学习用实现
===================================================================

本文件实现了：
  1. CausalConv3d         — 因果 3D 卷积（时间维单向，保证因果性）
  2. ResBlock3D           — 3D 残差块（VAE 的基本构建单元）
  3. Downsample3D         — 3D 下采样（空间/时间压缩）
  4. Upsample3D           — 3D 上采样（空间/时间恢复）
  5. Encoder3D            — 编码器：视频像素 → latent 分布 (μ, σ)
  6. Decoder3D            — 解码器：latent → 视频像素
  7. VideoVAE             — 完整的 3D VAE（Encoder + 重参数化 + Decoder）
  8. VAEConfig            — 配置

=== 为什么需要 3D VAE？ ===

  扩散模型（DiT）直接在像素空间操作的计算量太大：
    1080p 视频 96 帧:  3 × 96 × 1080 × 1920 ≈ 597M 值
    经 3D VAE 压缩后:  4 × 24 × 136  × 240  ≈ 3.1M 值   → 约 190 倍压缩

  VAE 的训练独立于 DiT，通常先训好 VAE，再训 DiT。

=== 3D VAE vs 2D VAE ===

  ┌────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │  2D VAE (Stable Diffusion 使用):                                       │
  │    - 逐帧编码/解码，帧之间独立                                        │
  │    - 不压缩时间维度                                                    │
  │    - 只做空间压缩: (B, 3, H, W) → (B, 4, H/8, W/8)                   │
  │    - 问题: 帧间无时间一致性，容易闪烁                                  │
  │                                                                        │
  │  3D VAE (CogVideoX / Sora / 可灵 使用):                                │
  │    - 同时编码时间+空间，3D 卷积核                                      │
  │    - 压缩时间 + 空间: (B, 3, T, H, W) → (B, C, T/4, H/8, W/8)       │
  │    - 优势: 时间维也被压缩，latent 更紧凑                              │
  │    - 3D 卷积天然建模帧间关系，重建更时间连贯                          │
  │                                                                        │
  │  对比:                                                                 │
  │    输入: (B, 3, 96, 1088, 1920)                                        │
  │    2D VAE: → (B, 4, 96, 136, 240)  时间维不变，还有 96 帧             │
  │    3D VAE: → (B, 4, 24, 136, 240)  时间也压缩 4×，只剩 24 帧         │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘

=== VAE 的核心思想（变分自编码器）===

  ┌────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │  普通 AutoEncoder:                                                     │
  │    x → Encoder → z → Decoder → x̂                                     │
  │    问题: z 空间不规则，不能随机采样生成新内容                          │
  │                                                                        │
  │  VAE (Variational):                                                    │
  │    x → Encoder → (μ, log σ²) → 重参数化 → z → Decoder → x̂           │
  │                                                                        │
  │    重参数化技巧:                                                       │
  │      z = μ + σ · ε,  ε ~ N(0, I)                                      │
  │      这样 z 是从学到的高斯分布 N(μ, σ²) 采样的                        │
  │      且梯度可以通过 μ 和 σ 回传（ε 是常数噪声）                       │
  │                                                                        │
  │    损失函数:                                                           │
  │      L = L_recon + β · L_KL                                           │
  │      L_recon = ‖x - x̂‖²  或  感知损失(LPIPS)                        │
  │      L_KL = KL(N(μ,σ²) ‖ N(0,I))  — 让 latent 分布接近标准正态      │
  │                                                                        │
  │    训练好后，DiT 在 latent space 里做扩散：                            │
  │      训练: x → VAE.encode → z_0 → 加噪 → z_t → DiT 学预测噪声       │
  │      推理: z_T(噪声) → DiT 去噪 → z_0 → VAE.decode → x̂(视频)       │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘

=== 整体架构（本文件实现的 3D VAE）===

  ┌────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │  Encoder3D:                                                            │
  │    input (B,3,T,H,W) ──→ Conv3d(3→128) ──→ ResBlock×2                │
  │    ──→ Downsample(空间2×) ──→ ResBlock×2                               │
  │    ──→ Downsample(空间2×) ──→ ResBlock×2                               │
  │    ──→ Downsample(空间2×+时间2×) ──→ ResBlock×2                        │
  │    ──→ Downsample(时间2×) ──→ ResBlock×2                               │
  │    ──→ Conv3d → (μ, log σ²)                                           │
  │    输出: (B, 2C, T/4, H/8, W/8)  (μ 和 logvar 拼在通道维)            │
  │                                                                        │
  │  Decoder3D:                                                            │
  │    input (B,C,T/4,H/8,W/8) ──→ Conv3d(C→512) ──→ ResBlock×2          │
  │    ──→ Upsample(时间2×) ──→ ResBlock×2                                │
  │    ──→ Upsample(空间2×+时间2×) ──→ ResBlock×2                         │
  │    ──→ Upsample(空间2×) ──→ ResBlock×2                                │
  │    ──→ Upsample(空间2×) ──→ ResBlock×2                                │
  │    ──→ Conv3d → (B,3,T,H,W)                                           │
  │                                                                        │
  │  压缩过程 (1080p 示例):                                                │
  │    (B, 3, 96, 1088, 1920)                                              │
  │     → 空间 2×: (B, 128, 96, 544, 960)                                 │
  │     → 空间 2×: (B, 256, 96, 272, 480)                                 │
  │     → 空间+时间: (B, 256, 48, 136, 240)                               │
  │     → 时间 2×: (B, 512, 24, 136, 240)                                 │
  │     → 投影: (B, 4, 24, 136, 240)  ← latent                           │
  │                                                                        │
  │  总压缩: 空间 8×, 时间 4×, 通道 3→4                                   │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘

参考:
  - "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013) — VAE 原论文
  - CogVideoX 3D VAE — 开源 3D VAE 实现
  - Stable Diffusion VAE — 2D VAE 参考
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
class VAEConfig:
    """
    3D VAE 配置。

    默认按 1920×1080 视频设置:
      输入: (B, 3, 96, 1088, 1920)   — 4 秒 24fps, pad 到 1088
      输出: (B, 4, 24, 136, 240)     — 空间 8×, 时间 4× 压缩
    """

    in_channels: int = 3  # 输入通道（RGB）
    latent_channels: int = 4  # latent 通道数（SD-VAE=4, SD3-VAE=16）
    base_channels: int = 128  # 编码器第一层通道数
    channel_multipliers: tuple = (1, 2, 2, 4)  # 每级的通道倍数 → 128, 256, 256, 512
    num_res_blocks: int = 2  # 每级的残差块数量
    spatial_downsample: tuple = (True, True, True, False)  # 每级是否空间下采样 2×
    temporal_downsample: tuple = (False, False, True, True)  # 每级是否时间下采样 2×
    dropout: float = 0.0  # dropout（VAE 训练通常不用 dropout）

    @property
    def spatial_compression(self) -> int:
        """空间总压缩率。"""
        return 2 ** sum(self.spatial_downsample)  # 2³ = 8

    @property
    def temporal_compression(self) -> int:
        """时间总压缩率。"""
        return 2 ** sum(self.temporal_downsample)  # 2² = 4

    @property
    def encoder_channels(self) -> list[int]:
        """编码器每级的通道数。"""
        return [self.base_channels * m for m in self.channel_multipliers]


# ═══════════════════════════════════════════════════════════════════════════
# 1. CausalConv3d — 因果 3D 卷积
# ═══════════════════════════════════════════════════════════════════════════
class CausalConv3d(nn.Module):
    """
    因果 3D 卷积 — 时间维只看过去和当前，不看未来。

    === 为什么需要因果卷积？ ===

    普通 Conv3d(kernel_size=3, padding=1):
      时间维上 pad 两边各 1 → 当前帧能看到前一帧和后一帧
      问题: 编码第 t 帧时用到了第 t+1 帧的信息（未来泄漏）

    因果 Conv3d:
      时间维只在前面 pad，不在后面 pad
      → 当前帧只能看到自己和过去的帧

    ┌──────────────────────────────────────────────────┐
    │  普通卷积 (kernel_t=3, pad=1):                    │
    │    t-1  t  t+1  ← 看到了未来帧                    │
    │     ▪───▪───▪                                     │
    │         ▼                                          │
    │       output_t                                     │
    │                                                    │
    │  因果卷积 (kernel_t=3, pad_front=2):              │
    │    t-2  t-1  t   ← 只看过去和当前                  │
    │     ▪───▪───▪                                     │
    │              ▼                                     │
    │           output_t                                 │
    └──────────────────────────────────────────────────┘

    空间维度仍然用普通对称 padding（空间没有因果关系）。

    参数:
        in_ch  (int):  输入通道
        out_ch (int):  输出通道
        kernel_size (int): 卷积核大小（3D 各维度相同）
        stride (int):  步长
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        # 空间维的对称 padding
        self.spatial_pad = kernel_size // 2
        # 时间维的因果 padding：只在前面 pad
        self.temporal_pad = kernel_size - 1

        self.conv = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, self.spatial_pad, self.spatial_pad),  # 时间维不 pad（手动 pad）
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W)
        返回: (B, C_out, T', H', W')
        # 你问“那么补充的是0卷积结果不是错误的吗？”
        # 其实这里在时间维前面补零（causal pad），确实导致最前面几个时间步的输出会用到补零（信息缺失），
        # 但这是所有因果卷积（包括常见的1D/2D/3D causal conv）都有的“warmup效应”：
        # 最前面 t < kernel_size-1 的输出对应的 receptive field 不完整，等价于用0作为前因填补，
        # 这样保证的是“不会泄漏未来”，而不是输出每一帧都等价于真实信息。
        # 
        # 其实这个补0只和“时间维的索引位置”有关——它不是针对哪一帧补的，也不是“把上一帧补成0”，而是整个序列最前面头几个时间步都补了0。
        # 卷积的窗口始终是滑动覆盖“当前时刻和之前（历史）”的内容，只不过序列刚开始时，历史不足，就只能用0补足窗口长度。
        # 比如 kernel=3 时：
        #   - 输出第1帧：窗口拿到 [0, 0, input[0]]，等价于前面两帧是0，只有input[0]是当前帧；
        #   - 输出第2帧：窗口拿到 [0, input[0], input[1]]，只有一帧历史、一帧补零；
        #   - 输出第3帧：窗口拿到 [input[0], input[1], input[2]]，此时才完全被真实帧填满。

        # temporal_pad = kernel_size - 1 的设置正好保证：输出的第 t 帧始终只依赖于输入的 [t-k+1, ..., t] 共 kernel_size 帧（即当前及历史）。
        # kernel_size不是常量吗？是动态的，因为kernel_size是3，所以temporal_pad是2，所以补2个0。
        # 这样刚开始的前 kernel_size-1 帧没有足够的历史，就自动从“补零的虚拟历史帧”中补足，等价于“空缺的输入帧全是0”。
        # 因此，因果卷积的因果性正是通过 kernel_size-1 的 pad 实现的：永远不会看到未来信息，只是前面用0“补历史”。
        # 这种补零不会让当前真实帧变成0，卷积窗口始终滑动——只是用0弥补不在当前序列里的“历史”部分。
        
        """
        # 对时间维做因果 padding（只在前补零，shape 会从 T → T+self.temporal_pad）

        # F.pad 参数（pad=(0,0,0,0,self.temporal_pad,0)）的含义:
        #   最后4个数 (0,0,0,0) 对应于W和H维：左/右/上/下都不填充
        #   前2个数 (self.temporal_pad, 0) 对应于T维：仅在最前面填充 self.temporal_pad 个0，后面不填充
        # 这样 pad 之后各元素的形状还是 (B, C, T+self.temporal_pad, H, W):
        #   B: batch size (不变)
        #   C: 通道数 (不变) c的形状是(in_ch) in_ch的形状是一个整数
        #   T: 时间维长度变成 T+self.temporal_pad  t的形状是(T+self.temporal_pad)也是一个整数
        #   H: 高度 (不变) h的形状是(H) H的形状是一个整数
        #   W: 宽度 (不变) w的形状是(W) W的形状是一个整数

        x = F.pad(x, (0, 0, 0, 0, self.cc, 0))
        # 现在 x 的 shape 是 (B, C, T+self.temporal_pad, H, W)
        return self.conv(x)


# ═══════════════════════════════════════════════════════════════════════════
# 2. ResBlock3D — 3D 残差块
# ═══════════════════════════════════════════════════════════════════════════
class ResBlock3D(nn.Module):
    """
    3D 残差块 — VAE 的基本构建单元。

    结构（与 ResNet 类似，但用 3D 卷积 + GroupNorm）:

      x ─────────────────────────────────(+)──→ out
      │                                   ↑
      └→ GroupNorm → SiLU → Conv3d       │
         → GroupNorm → SiLU → Conv3d ────┘

    为什么用 GroupNorm 而不是 LayerNorm/BatchNorm？
      - BatchNorm: 需要大 batch，VAE 训练 batch 通常很小 → 不稳定
      - LayerNorm: 对 CNN 不太适合（通道间差异大）
      - GroupNorm: 把通道分组归一化，不依赖 batch 大小，CNN 首选

    参数:
        channels (int): 输入/输出通道数
        dropout  (float): dropout 比例
    """

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            CausalConv3d(channels, channels, kernel_size=3),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            CausalConv3d(channels, channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W) → (B, C, T, H, W)"""
        return x + self.block(x)


class ResBlockWithChannelChange(nn.Module):
    """带通道变换的残差块，当输入/输出通道不同时使用 1×1×1 卷积做 shortcut。"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            CausalConv3d(in_ch, out_ch, kernel_size=3),
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            CausalConv3d(out_ch, out_ch, kernel_size=3),
        )
        self.shortcut = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.block(x)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Downsample3D / Upsample3D — 空间/时间维度缩放
# ═══════════════════════════════════════════════════════════════════════════
class Downsample3D(nn.Module):
    """
    3D 下采样 — 空间和/或时间维度缩小 2×。

    ┌──────────────────────────────────────────────────────┐
    │  空间下采样 (spatial=True):                           │
    │    (B, C, T, H, W) → (B, C, T, H/2, W/2)           │
    │    用 stride=(1,2,2) 的 Conv3d                       │
    │                                                      │
    │  时间下采样 (temporal=True):                          │
    │    (B, C, T, H, W) → (B, C, T/2, H, W)             │
    │    用 stride=(2,1,1) 的 Conv3d                       │
    │                                                      │
    │  两者都开:                                            │
    │    (B, C, T, H, W) → (B, C, T/2, H/2, W/2)         │
    │    用 stride=(2,2,2)                                 │
    └──────────────────────────────────────────────────────┘
    """

    def __init__(self, channels: int, spatial: bool = True, temporal: bool = False):
        super().__init__()
        stride_t = 2 if temporal else 1
        stride_s = 2 if spatial else 1
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            stride=(stride_t, stride_s, stride_s),
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    """
    3D 上采样 — 空间和/或时间维度放大 2×。

    先用 nearest-neighbor 插值放大，再用 Conv3d 精化。
    （比转置卷积更稳定，不容易出棋盘格伪影）

    ┌──────────────────────────────────────────────────────┐
    │  空间上采样 (spatial=True):                           │
    │    (B, C, T, H, W) → interpolate → (B, C, T, 2H, 2W)│
    │    → Conv3d → (B, C, T, 2H, 2W)                     │
    │                                                      │
    │  时间上采样 (temporal=True):                          │
    │    (B, C, T, H, W) → interpolate → (B, C, 2T, H, W) │
    │    → Conv3d → (B, C, 2T, H, W)                      │
    └──────────────────────────────────────────────────────┘
    """

    def __init__(self, channels: int, spatial: bool = True, temporal: bool = False):
        super().__init__()
        self.spatial = spatial
        self.temporal = temporal
        self.conv = CausalConv3d(channels, channels, kernel_size=3)

    def _compute_scale(self) -> tuple[float, float, float]:
        return (
            2.0 if self.temporal else 1.0,
            2.0 if self.spatial else 1.0,
            2.0 if self.spatial else 1.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self._compute_scale()
        x = F.interpolate(x, scale_factor=scale, mode="nearest")
        return self.conv(x)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Encoder3D — 编码器
# ═══════════════════════════════════════════════════════════════════════════
class Encoder3D(nn.Module):
    """
    3D VAE 编码器。

    将原始视频像素压缩到 latent 分布参数 (μ, log σ²)。

    ┌────────────────────────────────────────────────────────────────────────┐
    │  输入: (B, 3, 96, 1088, 1920) — 原始视频                              │
    │                                                                        │
    │  Level 0: Conv3d(3→128) + ResBlock×2                                  │
    │           → (B, 128, 96, 1088, 1920)                                  │
    │           Downsample(空间) → (B, 128, 96, 544, 960)                   │
    │                                                                        │
    │  Level 1: ResBlock(128→256)×2                                         │
    │           Downsample(空间) → (B, 256, 96, 272, 480)                   │
    │                                                                        │
    │  Level 2: ResBlock(256→256)×2                                         │
    │           Downsample(空间+时间) → (B, 256, 48, 136, 240)              │
    │                                                                        │
    │  Level 3: ResBlock(256→512)×2                                         │
    │           Downsample(时间) → (B, 512, 24, 136, 240)                   │
    │                                                                        │
    │  Head:  GroupNorm → SiLU → Conv3d(512→8)                              │
    │         → (B, 8, 24, 136, 240)                                        │
    │         split → μ (B,4,...) + log σ² (B,4,...)                        │
    │                                                                        │
    │  总压缩: 空间 8×, 时间 4×                                             │
    └────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        channels = cfg.encoder_channels  # [128, 256, 256, 512]

        # 初始卷积
        self.conv_in = CausalConv3d(cfg.in_channels, channels[0], kernel_size=3)

        # 编码器各级
        self.down_blocks = nn.ModuleList()
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            block = nn.ModuleDict()
            # 残差块（含通道变换）
            res_blocks = []
            for j in range(cfg.num_res_blocks):
                res_in = in_ch if j == 0 else out_ch
                res_blocks.append(
                    ResBlockWithChannelChange(res_in, out_ch, cfg.dropout)
                )
            block["res_blocks"] = nn.ModuleList(res_blocks)

            # 下采样
            s_down = cfg.spatial_downsample[i]
            t_down = cfg.temporal_downsample[i]
            if s_down or t_down:
                block["downsample"] = Downsample3D(
                    out_ch, spatial=s_down, temporal=t_down
                )
            self.down_blocks.append(block)
            in_ch = out_ch

        # 中间瓶颈层
        self.mid_block = nn.Sequential(
            ResBlock3D(channels[-1], cfg.dropout),
            ResBlock3D(channels[-1], cfg.dropout),
        )

        # 输出头: → 2×latent_channels (μ 和 logvar 拼接)
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], 2 * cfg.latent_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, T, H, W) — 原始视频
        返回: (B, 2*latent_channels, T', H', W') — (μ, logvar) 拼在通道维
        """
        # 1. 初始因果卷积，将输入 x 从输入通道升维到第一个 encoder 通道数
        x = self.conv_in(x)  # (B, 3, T, H, W) -> (B, 128, T, H, W)

        # 2. 编码器多层级循环
        #    遍历 self.down_blocks，每个 block 包含若干残差块和可选下采样
        for i, block in enumerate(self.down_blocks):
            # 2.1 block["res_blocks"]: 多个残差块顺序堆叠
            for j, res in enumerate(block["res_blocks"]):
                x = res(x)  # (B, C_in, T, H, W) -> (B, C_out, T, H, W)
            # 2.2 block["downsample"]: 可选的下采样模块（空间/时间）
            if "downsample" in block:
                x = block["downsample"](x)  # 可改变 T/H/W 分辨率

        # 3. 编码器中间瓶颈部分
        #    两个 ResBlock3D 进一步特征抽象
        x = self.mid_block(x)

        # 4. 输出头：GroupNorm → SiLU → 1x1x1 Conv3d 映射到 2×latent_channels
        x = self.conv_out(x)

        # 5. 返回 (μ, logvar) 拼接的最终编码特征
        return x


# ═══════════════════════════════════════════════════════════════════════════
# 5. Decoder3D — 解码器
# ═══════════════════════════════════════════════════════════════════════════
class Decoder3D(nn.Module):
    """
    3D VAE 解码器。

    将 latent 还原为视频像素，是 Encoder3D 的镜像结构。

    ┌────────────────────────────────────────────────────────────────────────┐
    │  输入: (B, 4, 24, 136, 240) — latent                                 │
    │                                                                        │
    │  Conv3d(4→512) → (B, 512, 24, 136, 240)                              │
    │                                                                        │
    │  Mid: ResBlock×2                                                      │
    │                                                                        │
    │  Level 3: Upsample(时间) → (B, 512, 48, 136, 240) + ResBlock×2      │
    │  Level 2: Upsample(空间+时间) → (B, 256, 96, 272, 480) + ResBlock×2 │
    │  Level 1: Upsample(空间) → (B, 256, 96, 544, 960) + ResBlock×2      │
    │  Level 0: Upsample(空间) → (B, 128, 96, 1088, 1920) + ResBlock×2    │
    │                                                                        │
    │  Head: GroupNorm → SiLU → Conv3d(128→3)                               │
    │  输出: (B, 3, 96, 1088, 1920) — 重建视频                             │
    └────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        channels = cfg.encoder_channels  # [128, 256, 256, 512]
        rev_channels = list(reversed(channels))  # [512, 256, 256, 128]
        rev_spatial = list(reversed(cfg.spatial_downsample))
        rev_temporal = list(reversed(cfg.temporal_downsample))

        # 输入投影
        self.conv_in = nn.Conv3d(cfg.latent_channels, rev_channels[0], kernel_size=1)

        # 中间瓶颈层
        #Sequential函数是torch.nn.Sequential，它是一个有序的容器，用于按顺序组合多个网络层。
        # mid_block 在 nn.Sequential 前通常只是单个或多个包裹在 list/tuple 里的模块，尚未形成有序可 forward 的容器结构；
        # 而通过 nn.Sequential 后，mid_block 变成一个可直接调用的有序容器，forward 时会依次通过每个子模块。
        self.ff = nn.Sequential(
            ResBlock3D(rev_channels[0], cfg.dropout),
            # 这两个 ResBlock3D 没有形状上的区别，只是参数不同，各自学习到不同的特征表示
            ResBlock3D(rev_channels[0], cfg.dropout),
        )

        # 解码器各级（编码器的镜像）
        self.up_blocks = nn.ModuleList()
        # 遍历 rev_channels，从后往前遍历，所以 in_ch 从 rev_channels[0] 开始，out_ch 从 rev_channels[1] 开始
        in_ch = rev_channels[0]
        # 这里的 for 循环负责按照解码器各级对输入特征做逐步上采样与特征变换
        # rev_channels 是编码器通道数的反转（解码器顺序）
        # 对每一级 (Level)：
        #   1. 根据 rev_spatial/rev_temporal 决定当前层是否做空间/时间维度的上采样
        #   2. 把该层要做的上采样、多个残差块分别装进 block 容器
        #   3. 每个 block（一个 dict）都包括若干 ResBlock3D（带通道调整），用于增加表达能力与非线性
        #   4. 每处理一级后 in_ch 变为 out_ch，用于下一级输入通道
        
        for i, out_ch in enumerate(rev_channels):
            # 构建每一级的操作容器（模块字典）
            block = nn.ModuleDict()

            # Step 1: 是否需要上采样（时空都支持），生成 Upsample3D 层
            s_up = rev_spatial[i]  # 当前级空间上采样倍数（如2或1）
            t_up = rev_temporal[i] # 当前级时间上采样倍数（如2或1）
            if s_up or t_up:
                # 只要有空间/时间维度需要扩展，就生成上采样模块
                block["upsample"] = Upsample3D(in_ch, spatial=s_up, temporal=t_up)

            # Step 2: 堆叠多个残差块（每一级的非线性表达）
            # 第一个 ResBlock 通道 in→out，后续的 ResBlock in=out
            res_blocks = []
            for j in range(cfg.num_res_blocks):
                res_in = in_ch if j == 0 else out_ch  # 只有第一个的输入通道与上一层相接
                res_blocks.append(
                    ResBlockWithChannelChange(res_in, out_ch, cfg.dropout)
                )
            block["res_blocks"] = nn.ModuleList(res_blocks)

            # Step 3: 把该级容器添加进 up_blocks（解码器的主流程依次调这些 block）
            self.up_blocks.append(block)
            # 更新输入通道，为下一级做准备
            in_ch = out_ch

        # 输出头
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, rev_channels[-1]),
            nn.SiLU(),
            nn.Conv3d(rev_channels[-1], cfg.in_channels, kernel_size=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, latent_channels, T', H', W') — latent
        返回: (B, 3, T, H, W) — 重建视频
        """
        x = self.conv_in(z)
        x = self.mid_block(x)

        for block in self.up_blocks:
            if "upsample" in block:
                x = block["upsample"](x)
            for res in block["res_blocks"]:
                x = res(x)

        x = self.conv_out(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# 6. VideoVAE — 完整的 3D VAE
# ═══════════════════════════════════════════════════════════════════════════
class VideoVAE(nn.Module):
    """
    完整的 3D VAE：编码器 + 重参数化 + 解码器。

    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  训练模式 (forward):                                                   │
    │                                                                        │
    │    video (B,3,T,H,W)                                                   │
    │       │                                                                │
    │       ▼                                                                │
    │    Encoder3D → (μ, logvar)                                             │
    │       │                                                                │
    │       ▼                                                                │
    │    重参数化: z = μ + σ·ε,  ε~N(0,I)                                   │
    │       │                                                                │
    │       ▼                                                                │
    │    Decoder3D → video_recon (B,3,T,H,W)                                │
    │       │                                                                │
    │       ▼                                                                │
    │    Loss = MSE(video, video_recon) + β · KL(μ,σ ‖ N(0,I))             │
    │                                                                        │
    │  推理/DiT训练 (encode):                                                │
    │    video → Encoder → (μ, logvar) → z = μ  (确定性编码，不采样)        │
    │                                                                        │
    │  DiT推理 (decode):                                                     │
    │    z_0 (DiT 去噪后的 latent) → Decoder → video                        │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘

    === KL 散度损失 ===

    KL(N(μ,σ²) ‖ N(0,1)) = -½ Σ (1 + log σ² - μ² - σ²)

    这个 loss 迫使 latent 分布接近标准正态 N(0,1)，这样：
      1. DiT 可以从 N(0,1) 采样噪声做扩散
      2. latent 空间是规则的，不会有"空洞"区域

    参数:
        cfg (VAEConfig): 配置
    """

    def __init__(self, cfg: VAEConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = VAEConfig()
        self.cfg = cfg
        self.encoder = Encoder3D(cfg)
        self.decoder = Decoder3D(cfg)

    def encode(self, x: torch.Tensor, sample: bool = True) -> dict[str, torch.Tensor]:
        """
        编码视频到 latent。

        x: (B, 3, T, H, W)
        sample: True=从 N(μ,σ²) 采样（训练用）, False=直接返回 μ（推理用）

        返回 dict:
          z:      (B, C_latent, T', H', W')  — latent 向量
          mu:     (B, C_latent, T', H', W')  — 均值
          logvar: (B, C_latent, T', H', W')  — log 方差
        """
        h = self.encoder(x)  # (B, 2*C_latent, T', H', W')
        mu, logvar = h.chunk(2, dim=1)

        if sample:
            # 重参数化: z = μ + σ · ε
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        return {"z": z, "mu": mu, "logvar": logvar}

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        从 latent 解码回视频。

        z: (B, C_latent, T', H', W')
        返回: (B, 3, T, H, W)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        训练用的完整前向传播。

        x: (B, 3, T, H, W) — 原始视频

        返回 dict:
          recon:  (B, 3, T, H, W)  — 重建视频
          z:      latent
          mu:     均值
          logvar: log 方差

        使用示例（训练）:
            vae = VideoVAE()
            video = torch.randn(2, 3, 16, 64, 64)

            out = vae(video)
            recon_loss = F.mse_loss(out["recon"], video)
            kl_loss = kl_divergence(out["mu"], out["logvar"])
            loss = recon_loss + 1e-6 * kl_loss

            loss.backward()
        """
        enc = self.encode(x, sample=True)
        recon = self.decode(enc["z"])
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
    计算 KL 散度: KL(N(μ,σ²) ‖ N(0,1))。

    公式: -½ Σ (1 + log σ² - μ² - σ²)

    mu:     (B, C, T, H, W)
    logvar: (B, C, T, H, W)
    返回:   标量 — batch 平均 KL 散度
    """
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """
    VAE 总损失。

    参数:
        recon:     (B, 3, T, H, W) — 重建视频
        target:    (B, 3, T, H, W) — 原始视频
        mu:        编码均值
        logvar:    编码 log 方差
        kl_weight: KL 损失权重 (通常很小, 1e-6 ~ 1e-4)
                   权重太大 → latent 被压到标准正态，重建质量差
                   权重太小 → latent 空间不规则，DiT 生成质量差

    返回 dict:
        total:       总损失
        recon_loss:  重建损失
        kl_loss:     KL 散度

    实际商业系统还会加:
      - 感知损失 (LPIPS): 用预训练 VGG 的特征比较，比 MSE 更符合人眼感知
      - GAN 判别器损失: 加一个判别器让重建更逼真（锐利细节）
      - 光流一致性损失: 确保重建视频的运动与原始一致
    """
    recon_loss = F.mse_loss(recon, target)
    kl_loss = kl_divergence(mu, logvar)
    total = recon_loss + kl_weight * kl_loss
    return {"total": total, "recon_loss": recon_loss, "kl_loss": kl_loss}
