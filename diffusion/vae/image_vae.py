"""
2D VAE (Variational Autoencoder) —— 图像压缩/解压
==================================================

一句话: 把大图压成小向量，再从小向量还原回大图。

  输入: 一张猫的图片 (3, 256, 256) — 196,608 个数
  压缩: latent (4, 32, 32)          —   4,096 个数  (压缩 48 倍)
  还原: 重建图片 (3, 256, 256)       — 196,608 个数

为什么需要压缩？
  Stable Diffusion 不在像素空间做扩散，而是在压缩后的 latent 空间做。
  像素空间 196,608 个数 → 每步扩散都要处理这么多 → 太慢
  latent 空间   4,096 个数 → 快 48 倍

VAE 和普通 AutoEncoder 的区别？
  普通 AE: 图片 → 一个确定的向量 → 重建
  VAE:     图片 → 一个"分布"(均值μ + 方差σ²) → 从分布采样一个向量 → 重建

  为什么要分布？因为采样有随机性，同一个图每次编码的结果略有不同，
  这让 latent 空间更"连续"——相邻的 latent 向量解码出来是类似的图。
  这样在 latent 空间做扩散时，生成质量更好。

整体流程图:

  ┌────────────────────────────────────────────────────────────────────┐
  │                                                                    │
  │  训练:                                                             │
  │                                                                    │
  │  猫图片 (1,3,256,256)                                              │
  │      │                                                             │
  │      ▼ Encoder2D                                                   │
  │  μ(1,4,32,32) + logvar(1,4,32,32)                                 │
  │      │                                                             │
  │      ▼ 重参数化: z = μ + σ × ε                                     │
  │  z (1,4,32,32)                                                     │
  │      │                                                             │
  │      ▼ Decoder2D                                                   │
  │  重建图片 (1,3,256,256)                                            │
  │      │                                                             │
  │      ▼ 算 loss                                                     │
  │  loss = MSE(原图, 重建图) + 很小 × KL(编码分布, 标准正态)         │
  │                                                                    │
  │  推理 (给扩散模型用):                                              │
  │    编码: 图片 → μ → z = μ × 0.18215                               │
  │    解码: z ÷ 0.18215 → 图片                                       │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘

本文件实现了以下组件，从简单到复杂:

  1. ImageVAEConfig  — 配置参数 (长宽、通道数等)
  2. ResBlock2D      — 残差块 (基本积木，只变通道不变空间)
  3. AttentionBlock  — 注意力块 (让每个像素能"看到"全局)
  4. Downsample2D    — 下采样 (空间缩小一半)
  5. Upsample2D      — 上采样 (空间放大一半)
  6. Encoder2D       — 编码器 (图片 → μ + logvar)
  7. Decoder2D       — 解码器 (z → 图片)
  8. ImageVAE        — 完整 VAE (编码器 + 重参数化 + 解码器)
  9. kl_divergence   — KL 散度 (让 latent 分布别太离谱)
  10. image_vae_loss — 训练损失 (重建 + KL)

参考:
  - "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
  - "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# 1. 配置
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class ImageVAEConfig:
    """
    VAE 的所有参数都在这里。默认按 Stable Diffusion 的设置。

    ┌──────────────────────────────────────────────────────────────────┐
    │  张量形状的含义: (B, C, H, W)                                    │
    │    B = batch 大小 (同时处理几张图)                               │
    │    C = 通道数 (RGB=3, latent=4, 中间特征=128/256/512)            │
    │    H = 高度                                                      │
    │    W = 宽度                                                      │
    │                                                                  │
    │  "通道"怎么理解?                                                 │
    │    RGB 图像有 3 个通道: 红、绿、蓝, 每个通道是一张灰度图        │
    │    网络中间的 128 通道 = 128 张灰度图叠在一起                   │
    │    每张灰度图捕捉不同的特征 (边缘/纹理/颜色...)                 │
    │    通道越多，能捕捉的信息越丰富                                 │
    │                                                                  │
    │  默认配置的维度变化:                                             │
    │    输入图片:  (1, 3, 256, 256) — 3通道, 256×256像素             │
    │    输出latent: (1, 4, 32, 32)  — 4通道, 32×32, 压缩48倍        │
    │                                                                  │
    │  channel_multipliers=(1,2,4,4) 决定了4个级别的通道数:           │
    │    Level 0: 128×1 = 128 通道,  空间256→128 (下采样1次)          │
    │    Level 1: 128×2 = 256 通道,  空间128→64  (下采样2次)          │
    │    Level 2: 128×4 = 512 通道,  空间64→32   (下采样3次)          │
    │    Level 3: 128×4 = 512 通道,  空间32停住   (不下采样)          │
    │                                                                  │
    │  为什么越深通道越多、空间越小?                                   │
    │    浅层: 大图片但只有粗略特征 (轮廓、颜色) → 少通道够用         │
    │    深层: 小图片但需要精细特征 (细节、语义) → 多通道存储         │
    │    空间缩小节省计算量, 通道增多保留信息量                       │
    └──────────────────────────────────────────────────────────────────┘
    """
    in_channels: int = 3              # 输入通道 (RGB=3)
    latent_channels: int = 4          # latent 通道 (SD=4)
    base_channels: int = 128          # 第一层通道数
    channel_multipliers: tuple = (1, 2, 4, 4)  # 每级通道倍数
    num_res_blocks: int = 2           # 每级残差块数量
    attention_at_level: tuple = (False, False, False, True)  # 哪级加注意力
    dropout: float = 0.0

    @property
    def spatial_compression(self) -> int:
        """空间总压缩率: 3次下采样 = 2³ = 8倍"""
        return 2 ** (len(self.channel_multipliers) - 1)

    @property
    def encoder_channels(self) -> list[int]:
        """编码器每级的通道数: [128, 256, 512, 512]"""
        return [self.base_channels * m for m in self.channel_multipliers]


# ═══════════════════════════════════════════════════════════════════════════
# 2. ResBlock2D — 残差块
# ═══════════════════════════════════════════════════════════════════════════
class ResBlock2D(nn.Module):
    """
    残差块 — VAE 的基本积木。

    ┌──────────────────────────────────────────────────────────────────┐
    │  它做什么?                                                       │
    │    对图像做两次"归一化→激活→卷积"变换, 空间大小不变,           │
    │    通道数可以变也可以不变。                                      │
    │                                                                  │
    │  "残差"是什么意思?                                               │
    │    输出 = 原始输入 + 变换结果 (而不是只输出变换结果)            │
    │    好处: 如果变换没学到有用的东西(输出≈0),                      │
    │          输出 ≈ 原始输入, 不会把信息搞丢                       │
    │                                                                  │
    │  数值示例 (in_ch=128, out_ch=256):                               │
    │                                                                  │
    │  输入 x: (1, 128, 64, 64)  ← 1张图, 128通道, 64×64像素         │
    │      │                                                           │
    │      ├──────────────────────────────────(+)──→ 输出              │
    │      │                                     ↑                     │
    │      │  GroupNorm(128) → 归一化             │                     │
    │      │  SiLU          → 激活(非线性)       │                     │
    │      │  Conv2d(128→256, 3×3)                │                     │
    │      │    → (1, 256, 64, 64)                │                     │
    │      │    ↑ 通道变了128→256, 空间64×64没变 │                     │
    │      │    ↑ 3×3卷积: 每个像素看周围3×3邻域 │                     │
    │      │    ↑ padding=1: 输出空间大小不变     │                     │
    │      │                                    │                     │
    │      │  GroupNorm(256) → 归一化            │                     │
    │      │  SiLU          → 激活               │                     │
    │      │  Conv2d(256→256, 3×3)               │                     │
    │      │    → (1, 256, 64, 64) ──────────────┘                     │
    │                                                                  │
    │  shortcut: Conv2d(128→256, 1×1)                                 │
    │    把 x 从 (1,128,64,64) 变成 (1,256,64,64)                    │
    │    1×1卷积不改变空间大小, 只改变通道数                          │
    │    为什么需要? 因为 + 要求两边形状相同才能相加                  │
    │    (1,128,64,64) + (1,256,64,64) 不行!                         │
    │    (1,256,64,64) + (1,256,64,64) ✓                             │
    │                                                                  │
    │  最终输出 = shortcut(x) + conv路径(h)                           │
    │           = (1, 256, 64, 64)                                    │
    │                                                                  │
    │  为什么用 GroupNorm 不用 BatchNorm?                              │
    │    BatchNorm: 对整个batch统计均值方差 → batch小时不稳定         │
    │    GroupNorm:  把通道分成32组, 组内归一化 → 不依赖batch大小    │
    │    VAE训练batch通常只有4~16, GroupNorm更稳定                    │
    │                                                                  │
    │  为什么用 SiLU 不用 ReLU?                                        │
    │    ReLU: 负数直接变0 → 丢失信息                                 │
    │    SiLU:  x × sigmoid(x) → 负数也有输出(只是很小)              │
    │    生成模型中 SiLU 效果更好                                     │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # 通道数不同时, 用1×1卷积调整通道, 让 shortcut 能和主路径相加
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C_in, H, W) → (B, C_out, H, W)  空间不变, 通道可变"""
        h = self.act(self.norm1(x))   # 归一化+激活
        h = self.conv1(h)             # 3×3卷积, 通道 in→out
        h = self.act(self.norm2(h))   # 再归一化+激活
        h = self.dropout(h)
        h = self.conv2(h)             # 3×3卷积, 通道 out→out
        return self.shortcut(x) + h   # 残差: 原始输入(调整通道) + 变换结果


# ═══════════════════════════════════════════════════════════════════════════
# 3. AttentionBlock — 自注意力块
# ═══════════════════════════════════════════════════════════════════════════
class AttentionBlock(nn.Module):
    """
    让每个像素位置能"看到"整张图的所有位置, 而不只是局部3×3邻域。

    ┌──────────────────────────────────────────────────────────────────┐
    │  为什么需要注意力?                                               │
    │                                                                  │
    │  卷积(3×3)的局限: 每个像素只看周围3×3=9个邻居                 │
    │    → 左上角的猫耳朵和右下角的猫尾巴, 没法直接建立联系         │
    │                                                                  │
    │  注意力: 每个像素和所有像素算相关性, 按相关性加权汇总信息      │
    │    → 猫耳朵的像素可以直接"关注"猫尾巴                         │
    │                                                                  │
    │  为什么只在 32×32 用, 不在 256×256 用?                          │
    │    注意力计算量 = 像素数²                                       │
    │    256×256: 65536² = 43亿 → 太慢太慢                           │
    │    32×32:   1024² = 100万 → 可以接受                           │
    │                                                                  │
    │  ─────────────────────────────────────────────────────────────   │
    │                                                                  │
    │  数值示例: 输入 (1, 512, 32, 32)                                │
    │    1张图, 512通道, 32×32=1024个像素位置                         │
    │                                                                  │
    │  步骤1: GroupNorm → (1, 512, 32, 32) 归一化, 形状不变          │
    │                                                                  │
    │  步骤2: 用1×1卷积产生 Q, K, V, 各 (1, 512, 32, 32)            │
    │    Q = "我在找什么?" (查询)                                     │
    │    K = "我能提供什么?" (键)                                     │
    │    V = "我的实际内容" (值)                                      │
    │    为什么用1×1卷积? 它不改变空间大小, 只改变每像素的向量       │
    │    本质是给每个像素学一个线性变换, 产生不同的Q/K/V角色          │
    │                                                                  │
    │  步骤3: 把2D图像展平成"序列"                                   │
    │    (1, 512, 32, 32)                                             │
    │    → reshape: (1, 512, 1024)   把H×W合并成一维                │
    │    → permute: (1, 1024, 512)   交换维度                        │
    │                                                                  │
    │    现在有1024个"词"(像素位置), 每个"词"用512维向量表示        │
    │    这和NLP的注意力完全一样: 1024个token, 512维embedding         │
    │                                                                  │
    │  步骤4: 算注意力                                                 │
    │    scores = Q @ K^T / √512                                      │
    │      Q: (1, 1024, 512),  K^T: (1, 512, 1024)                   │
    │      → scores: (1, 1024, 1024)                                  │
    │      这是1024×1024的相关性矩阵                                  │
    │      scores[i,j] = 第i个像素对第j个像素的关注分数              │
    │                                                                  │
    │    为什么除以√512?                                               │
    │      512维点积的值很大(512个乘积相加)                           │
    │      不除的话softmax会饱和(全概率集中到1个位置)                 │
    │      除以√512让分数在合理范围                                   │
    │                                                                  │
    │    weights = softmax(scores) → (1, 1024, 1024)                  │
    │      每行和为1, 表示第i个像素对1024个位置的关注比例            │
    │                                                                  │
    │    out = weights @ V                                             │
    │      weights: (1, 1024, 1024),  V: (1, 1024, 512)              │
    │      → out: (1, 1024, 512)                                      │
    │      每个像素的输出 = 所有像素值的加权平均                     │
    │                                                                  │
    │  步骤5: 变回2D图像                                              │
    │    (1, 1024, 512)                                                │
    │    → permute: (1, 512, 1024)                                    │
    │    → reshape: (1, 512, 32, 32)                                  │
    │                                                                  │
    │  步骤6: 1×1卷积融合 + 残差                                     │
    │    out = 1×1Conv(out)                                            │
    │    return x + out → (1, 512, 32, 32)                            │
    │                                                                  │
    │  总结: 进去(1,512,32,32), 出来(1,512,32,32), 形状完全不变      │
    │  但每个像素现在融合了全局信息                                   │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)  # 1×1卷积产生Q
        self.k = nn.Conv2d(channels, channels, kernel_size=1)  # 1×1卷积产生K
        self.v = nn.Conv2d(channels, channels, kernel_size=1)  # 1×1卷积产生V
        self.out = nn.Conv2d(channels, channels, kernel_size=1)  # 输出融合
        self.scale = channels ** -0.5  # 1/√C, 缩放用

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, C, H, W)  形状完全不变"""
        B, C, H, W = x.shape
        h = self.norm(x)

        # 产生QKV并展平成序列
        q = self.q(h).reshape(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        k = self.k(h).reshape(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        v = self.v(h).reshape(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)

        # 注意力计算
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, H*W, H*W)
        attn = F.softmax(attn, dim=-1)

        # 加权汇总并变回2D
        out = torch.bmm(attn, v)  # (B, H*W, C)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H, W)
        out = self.out(out)

        return x + out  # 残差


# ═══════════════════════════════════════════════════════════════════════════
# 4. Downsample2D — 下采样 (空间缩小一半)
# ═══════════════════════════════════════════════════════════════════════════
class Downsample2D(nn.Module):
    """
    把图像的宽高各缩小一半, 通道数不变。

    ┌──────────────────────────────────────────────────────────────────┐
    │  为什么需要下采样?                                               │
    │    编码器的目标: 256→128→64→32, 逐步缩小空间                  │
    │    每缩小一次, 网络看到的就是更大范围的"全局信息"              │
    │    (就像你离远看一张图, 看到的是整体轮廓而不是细节)            │
    │                                                                  │
    │  怎么缩小? 用 stride=2 的 Conv2d (可学习的下采样)              │
    │                                                                  │
    │  数值示例: 输入 (1, 128, 256, 256)                              │
    │                                                                  │
    │  ① 非对称 padding: 右边和下边各补1列/行0                       │
    │    (1, 128, 256, 256) → (1, 128, 257, 257)                     │
    │    为什么补? Conv2d(3×3, stride=2) 需要3×3的窗口              │
    │    256/2=128, 但边界像素会被丢掉一些, 补0保留边界信息         │
    │                                                                  │
    │  ② Conv2d(3×3, stride=2, padding=0):                           │
    │    窗口每次滑动2格 (stride=2), 所以输出空间减半                │
    │    (1, 128, 257, 257) → (1, 128, 128, 128)                    │
    │    257/2 = 128 (整数除法向下取整)                               │
    │                                                                  │
    │  为什么不用 MaxPool 或 AvgPool?                                  │
    │    MaxPool: 只保留最大值, 丢信息太多                            │
    │    AvgPool: 取平均, 变模糊                                      │
    │    Conv(stride=2): 可学习的下采样, 保留最有用的信息 ✓          │
    │                                                                  │
    │  总结: (1,128,256,256) → (1,128,128,128)                       │
    │        通道128不变, 空间 256→128 缩小2倍                       │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1))  # 右边补1列, 下边补1行
        return self.conv(x)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Upsample2D — 上采样 (空间放大一半)
# ═══════════════════════════════════════════════════════════════════════════
class Upsample2D(nn.Module):
    """
    把图像的宽高各放大一倍, 通道数不变。

    ┌──────────────────────────────────────────────────────────────────┐
    │  为什么需要上采样?                                               │
    │    解码器的目标: 32→64→128→256, 逐步还原空间                  │
    │    和编码器的下采样是反过来的                                   │
    │                                                                  │
    │  怎么放大? 先 nearest 插值, 再 3×3 卷积平滑                    │
    │                                                                  │
    │  数值示例: 输入 (1, 512, 32, 32)                                │
    │                                                                  │
    │  ① nearest 插值 (最近邻, scale_factor=2):                       │
    │    每个像素复制成2×2的方块:                                    │
    │    [A]  →  [A A]                                                │
    │              [A A]                                               │
    │    (1, 512, 32, 32) → (1, 512, 64, 64)                         │
    │    空间翻倍, 但现在图是"块状"的, 像打了马赛克                 │
    │                                                                  │
    │  ② Conv2d(3×3, padding=1):                                      │
    │    用3×3卷积平滑, 消除马赛克感                                 │
    │    (1, 512, 64, 64) → (1, 512, 64, 64)                         │
    │    空间不变, 但像素之间的过渡更自然                            │
    │                                                                  │
    │  为什么不用转置卷积 (ConvTranspose2d)?                           │
    │    转置卷积容易产生"棋盘格"伪影:                               │
    │    明亮 像素 和 暗淡 像素 交替出现, 像棋盘一样                 │
    │    nearest + conv 更稳定, 不会出现这种问题                     │
    │                                                                  │
    │  总结: (1,512,32,32) → (1,512,64,64)                           │
    │        通道512不变, 空间 32→64 放大2倍                         │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")  # 空间翻倍
        return self.conv(x)  # 平滑


# ═══════════════════════════════════════════════════════════════════════════
# 6. Encoder2D — 编码器
# ═══════════════════════════════════════════════════════════════════════════
class Encoder2D(nn.Module):
    """
    编码器: 图片 → latent 分布参数 (μ 和 log σ²)

    ┌──────────────────────────────────────────────────────────────────┐
    │  它做什么?                                                       │
    │    把一张大图 (3, 256, 256) 逐步压缩成小的特征图 (8, 32, 32),  │
    │    然后劈成两半: μ(4,32,32) 和 logvar(4,32,32)。               │
    │    μ 是每个位置的均值, logvar 是 log方差,                       │
    │    它们一起定义了一个正态分布 N(μ, σ²)。                       │
    │                                                                  │
    │  为什么输出"分布"而不是一个确定的向量?                          │
    │    如果只输出确定向量, latent 空间会有"空洞"                  │
    │    → 从空洞采样出来的向量, 解码后会生成乱七八糟的图           │
    │    输出分布 + 训练时从分布采样 → latent空间更连续              │
    │    → 相邻的 latent 解码出来是类似的图 → 生成质量更好          │
    │                                                                  │
    │  ─────────────────────────────────────────────────────────────   │
    │                                                                  │
    │  完整维度流转 (B=1, 默认配置):                                   │
    │                                                                  │
    │  输入图片: (1, 3, 256, 256)                                     │
    │      │                                                           │
    │      ▼ conv_in: Conv2d(3→128, 3×3, padding=1)                   │
    │         为什么? 把3通道RGB变成128通道特征图                     │
    │         为什么padding=1? 3×3卷积+padding=1 → 空间大小不变      │
    │  (1, 128, 256, 256)                                             │
    │      │                                                           │
    │      ▼ Level 0: 通道=128, 空间 256→128 ────────────────────     │
    │      │                                                           │
    │      │  ResBlock(128→128) ×2                                     │
    │      │    → (1, 128, 256, 256) → (1, 128, 256, 256)             │
    │      │    ResBlock不改变空间, 通道可以变(这里没变)              │
    │      │    2个ResBlock: 提取特征, 越深特征越抽象                 │
    │      │                                                           │
    │      │  Downsample(128)                                          │
    │      │    → (1, 128, 128, 128)                                   │
    │      │    空间256→128, 缩小2倍, 看到更大范围                   │
    │      │                                                           │
    │      ▼ Level 1: 通道=256, 空间 128→64 ────────────────────      │
    │      │                                                           │
    │      │  ResBlock(128→256) ×2                                     │
    │      │    → (1, 256, 128, 128)                                   │
    │      │    第1个ResBlock: 通道128→256 (信息变丰富)               │
    │      │    第2个ResBlock: 通道256→256 (深化特征)                 │
    │      │                                                           │
    │      │  Downsample(256)                                          │
    │      │    → (1, 256, 64, 64)                                     │
    │      │                                                           │
    │      ▼ Level 2: 通道=512, 空间 64→32 ─────────────────────      │
    │      │                                                           │
    │      │  ResBlock(256→512) ×2                                     │
    │      │    → (1, 512, 64, 64)                                     │
    │      │                                                           │
    │      │  Downsample(512)                                          │
    │      │    → (1, 512, 32, 32)                                     │
    │      │                                                           │
    │      ▼ Level 3: 通道=512, 空间 32 停住 ────────────────────     │
    │      │                                                           │
    │      │  ResBlock(512→512) ×2                                     │
    │      │    → (1, 512, 32, 32)                                     │
    │      │    最后一级不下采样! 32×32已经够小了                    │
    │      │                                                           │
    │      │  AttentionBlock(512)                                      │
    │      │    → (1, 512, 32, 32)                                     │
    │      │    在最小分辨率加注意力, 让每个像素看到全局              │
    │      │    只在这一级加, 因为更大分辨率太慢                      │
    │      │                                                           │
    │      ▼ Mid 瓶颈层 ──────────────────────────────────────────    │
    │      │                                                           │
    │      │  ResBlock → Attention → ResBlock                          │
    │      │    全部 (1, 512, 32, 32)                                  │
    │      │    在最深处再提炼一次特征                                 │
    │      │    注意力夹在两个ResBlock中间, 先局部再全局再局部        │
    │      │                                                           │
    │      ▼ conv_out: GroupNorm → SiLU → Conv2d(512→8, 1×1)         │
    │      │                                                           │
    │      │  为什么512→8?                                             │
    │      │    8 = 2 × latent_channels(4)                            │
    │      │    前一半是μ(4通道), 后一半是logvar(4通道)               │
    │      │    所以输出8通道, 后面劈成两个4通道                      │
    │      │                                                           │
    │      │  为什么用1×1卷积?                                         │
    │      │    不改变空间大小(32×32), 只改变通道数(512→8)            │
    │      │    1×1卷积就是每个像素位置独立的线性变换                 │
    │      │                                                           │
    │  (1, 8, 32, 32)                                                 │
    │      │                                                           │
    │      ▼ chunk(2, dim=1) — 沿通道维劈成两半                      │
    │      │  dim=1 表示通道维, 把8通道劈成两个4通道                 │
    │      │  通道0~3 → μ,  通道4~7 → logvar                         │
    │      │                                                           │
    │  μ:      (1, 4, 32, 32) ← 每个位置的均值                       │
    │  logvar: (1, 4, 32, 32) ← 每个位置的log方差                    │
    │                                                                  │
    │  ─────────────────────────────────────────────────────────────   │
    │                                                                  │
    │  总结: (1,3,256,256) → Encoder → (1,8,32,32) → split           │
    │        → μ(1,4,32,32) + logvar(1,4,32,32)                      │
    │                                                                  │
    │  信息变化:                                                       │
    │    空间: 256→128→64→32, 缩8倍                                  │
    │    通道: 3→128→256→512→8, 先扩再缩                             │
    │    总压缩: 256×256×3 = 196608 → 32×32×4×2 = 8192 (24倍)       │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, cfg: ImageVAEConfig):
        super().__init__()
        channels = cfg.encoder_channels  # [128, 256, 512, 512]

        # 入口: 3通道→128通道, 空间不变
        self.conv_in = nn.Conv2d(cfg.in_channels, channels[0], kernel_size=3, padding=1)

        # 4个下采样级别
        self.down_blocks = nn.ModuleList()
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            block_layers = nn.ModuleList()

            # 每级2个ResBlock (第一个可能变通道, 后面的通道不变)
            for j in range(cfg.num_res_blocks):
                res_in = in_ch if j == 0 else out_ch
                block_layers.append(ResBlock2D(res_in, out_ch, cfg.dropout))

            # 最后一级加注意力
            if cfg.attention_at_level[i]:
                block_layers.append(AttentionBlock(out_ch))

            block = nn.ModuleDict({"layers": block_layers})

            # 前三级下采样, 最后一级不下
            if i < len(channels) - 1:
                block["downsample"] = Downsample2D(out_ch)

            self.down_blocks.append(block)
            in_ch = out_ch

        # 中间瓶颈: ResBlock → Attention → ResBlock
        self.mid = nn.ModuleList([
            ResBlock2D(channels[-1], channels[-1], cfg.dropout),
            AttentionBlock(channels[-1]),
            ResBlock2D(channels[-1], channels[-1], cfg.dropout),
        ])

        # 输出头: 512→8通道 (8 = 2×latent_channels, 前4=μ, 后4=logvar)
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, channels[-1]),
            nn.SiLU(),
            nn.Conv2d(channels[-1], 2 * cfg.latent_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, 8, H/8, W/8)  8=2×latent_channels"""
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
# 7. Decoder2D — 解码器
# ═══════════════════════════════════════════════════════════════════════════
class Decoder2D(nn.Module):
    """
    解码器: latent → 图片。编码器的镜像, 空间逐步放大, 通道逐步缩小。

    ┌──────────────────────────────────────────────────────────────────┐
    │  完整维度流转 (B=1, 默认配置):                                   │
    │                                                                  │
    │  输入 latent: (1, 4, 32, 32)                                    │
    │      │                                                           │
    │      ▼ conv_in: Conv2d(4→512, 3×3, padding=1)                   │
    │         为什么? 把4通道latent变回512通道, 准备逐级展开         │
    │  (1, 512, 32, 32)                                               │
    │      │                                                           │
    │      ▼ Mid 瓶颈层 ──────────────────────────────────────────    │
    │      │  ResBlock(512→512) → (1, 512, 32, 32)                    │
    │      │  Attention(512)   → (1, 512, 32, 32)                     │
    │      │  ResBlock(512→512) → (1, 512, 32, 32)                    │
    │      │  和编码器mid一样: 先局部, 再全局, 再局部                 │
    │      │                                                           │
    │      ▼ Level 3→0 (通道倒序: 512→512→256→128) ────────────      │
    │      │                                                           │
    │      │  Level 3: ResBlock(512→512) ×3 + Attention               │
    │      │    → (1, 512, 32, 32)                                     │
    │      │    为什么3个ResBlock? 编码器只有2个, 解码器多1个         │
    │      │    因为解码器要还原细节, 需要更多容量                    │
    │      │  Upsample → (1, 512, 64, 64)   空间: 32→64              │
    │      │                                                           │
    │      │  Level 2: ResBlock(512→512) ×3                           │
    │      │    → (1, 512, 64, 64)                                     │
    │      │  Upsample → (1, 512, 128, 128)  空间: 64→128            │
    │      │                                                           │
    │      │  Level 1: ResBlock(512→256) ×3                           │
    │      │    → (1, 256, 128, 128)                                   │
    │      │    通道开始减少: 512→256, 信息逐步"聚焦"                │
    │      │  Upsample → (1, 256, 256, 256)  空间: 128→256           │
    │      │                                                           │
    │      │  Level 0: ResBlock(256→128) ×3                           │
    │      │    → (1, 128, 256, 256)                                   │
    │      │    最后一级不上采样, 已经是目标大小256×256               │
    │      │                                                           │
    │      ▼ conv_out: GroupNorm → SiLU → Conv2d(128→3, 3×3)         │
    │         为什么128→3? 还原回RGB 3通道                            │
    │  (1, 3, 256, 256)  ← 还原回图片                                │
    │                                                                  │
    │  总结: (1,4,32,32) → Decoder → (1,3,256,256)                   │
    │  和编码器完全对称: 编码器怎么压的, 解码器就怎么还原            │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, cfg: ImageVAEConfig):
        super().__init__()
        channels = cfg.encoder_channels  # [128, 256, 512, 512]
        rev_channels = list(reversed(channels))  # [512, 512, 256, 128]

        # latent → 512通道
        self.conv_in = nn.Conv2d(cfg.latent_channels, rev_channels[0], kernel_size=3, padding=1)

        # 瓶颈层 (和编码器一样)
        self.mid = nn.ModuleList([
            ResBlock2D(rev_channels[0], rev_channels[0], cfg.dropout),
            AttentionBlock(rev_channels[0]),
            ResBlock2D(rev_channels[0], rev_channels[0], cfg.dropout),
        ])

        # 4个上采样级别 (通道倒序)
        rev_attention = list(reversed(cfg.attention_at_level))
        self.up_blocks = nn.ModuleList()
        in_ch = rev_channels[0]
        for i, out_ch in enumerate(rev_channels):
            block_layers = nn.ModuleList()

            # 每级3个ResBlock (比编码器多1个)
            for j in range(cfg.num_res_blocks + 1):
                res_in = in_ch if j == 0 else out_ch
                block_layers.append(ResBlock2D(res_in, out_ch, cfg.dropout))

            if rev_attention[i]:
                block_layers.append(AttentionBlock(out_ch))

            block = nn.ModuleDict({"layers": block_layers})

            # 前三级上采样, 最后一级不上
            if i < len(rev_channels) - 1:
                block["upsample"] = Upsample2D(out_ch)

            self.up_blocks.append(block)
            in_ch = out_ch

        # 128通道 → 3通道RGB
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, rev_channels[-1]),
            nn.SiLU(),
            nn.Conv2d(rev_channels[-1], cfg.in_channels, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, 4, H', W') → (B, 3, H, W)"""
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
# 8. ImageVAE — 完整的 2D VAE
# ═══════════════════════════════════════════════════════════════════════════
class ImageVAE(nn.Module):
    """
    完整 VAE: 编码器 + 重参数化 + 解码器。

    ┌──────────────────────────────────────────────────────────────────┐
    │  训练流程 (forward) — B=1, 输入(1,3,256,256):                   │
    │                                                                  │
    │  ① 编码                                                         │
    │     图片 (1, 3, 256, 256)                                       │
    │       │ Encoder2D: 逐步压缩空间, 扩大通道                      │
    │       ▼                                                         │
    │     (1, 8, 32, 32) — 8 = 2 × latent_channels(4)               │
    │       │ chunk(2, dim=1): 沿通道维劈成两半                      │
    │       │  通道0~3 → μ,  通道4~7 → logvar                       │
    │       ▼                                                         │
    │     μ:      (1, 4, 32, 32)  每个位置的均值, 假设某位置=1.5    │
    │     logvar: (1, 4, 32, 32)  每个位置的log方差, 假设=0.7        │
    │                                                                  │
    │  ② 重参数化 (VAE最关键的创新!)                                 │
    │                                                                  │
    │     目标: 从正态分布 N(μ, σ²) 采样一个 z                       │
    │                                                                  │
    │     直觉: μ=1.5 表示"中心在1.5", σ=1.42表示"散布范围±1.42"    │
    │                                                                  │
    │     计算:                                                        │
    │       σ = exp(0.5 × logvar) = exp(0.35) ≈ 1.42                │
    │       ε ~ N(0,1)           — 从标准正态采样, 假设ε=0.3          │
    │       z = μ + σ × ε       = 1.5 + 1.42×0.3 = 1.93             │
    │                                                                  │
    │     为什么要拆成 μ + σ × ε, 而不是直接采样?                    │
    │       直接采样 z ~ N(μ, σ²): 采样操作不可导, 梯度断掉了!      │
    │       拆开后: z = μ + σ × ε                                     │
    │         μ 和 σ 是确定性的计算 (可求导 ✓)                       │
    │         ε 是随机的, 但它和 μ,σ 无关 (不参与求导)              │
    │         梯度可以通过 μ 和 σ 传回编码器 ✓                       │
    │       这就是"重参数化技巧" — 把随机性从参数里剥离出来          │
    │                                                                  │
    │     z: (1, 4, 32, 32) — 采样的 latent 向量                     │
    │                                                                  │
    │  ③ 解码                                                         │
    │     z (1, 4, 32, 32)                                            │
    │       │ Decoder2D: 逐步放大空间, 缩小通道                      │
    │       ▼                                                         │
    │     recon (1, 3, 256, 256) — 重建的图片                        │
    │                                                                  │
    │  ④ 计算损失                                                     │
    │     recon_loss = MSE(原图, 重建图)                              │
    │       → 重建越像原图, 值越小                                   │
    │     kl_loss = KL(编码分布, 标准正态)                            │
    │       → latent分布越接近N(0,1), 值越小                         │
    │     total = recon_loss + 0.000001 × kl_loss                    │
    │       → KL权重极小, 主要靠重建损失驱动学习                    │
    │                                                                  │
    ├──────────────────────────────────────────────────────────────────┤
    │  推理时 (给扩散模型用):                                         │
    │                                                                  │
    │  编码: 图片 → Encoder → μ → z = μ × 0.18215                   │
    │    用 μ 而不采样 — 推理要确定性, 不需要随机性                  │
    │    × 0.18215 (scaling) — 让 latent 方差≈1, 扩散模型更稳定     │
    │                                                                  │
    │  解码: z ÷ 0.18215 → Decoder → 图片                            │
    │    先除回 scaling, 撤销编码时的缩放, 再解码                    │
    │                                                                  │
    │  scaling_factor=0.18215 怎么来的?                               │
    │    训练完VAE后, 统计所有训练图编码后 latent 的标准差 ≈ 5.5     │
    │    1/5.5 ≈ 0.18215, 乘上后标准差变成1                         │
    │    扩散模型在方差≈1的空间里工作最稳定                          │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, cfg: ImageVAEConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = ImageVAEConfig()
        self.cfg = cfg
        self.encoder = Encoder2D(cfg)
        self.decoder = Decoder2D(cfg)
        self.scaling_factor = 0.18215

    def encode(self, x: torch.Tensor, sample: bool = True, apply_scaling: bool = True
               ) -> dict[str, torch.Tensor]:
        """
        编码: 图片 → latent

        ┌──────────────────────────────────────────────────────────────┐
        │  x: (1, 3, 256, 256)                                        │
        │    │ Encoder → (1, 8, 32, 32)                                │
        │    │ chunk(2, dim=1):                                        │
        │    │   前4通道 → μ:     (1, 4, 32, 32)                       │
        │    │   后4通道 → logvar: (1, 4, 32, 32)                      │
        │    │                                                          │
        │    │ sample=True (训练时):                                    │
        │    │   σ = exp(0.5 × logvar)                                 │
        │    │   ε = randn (标准正态随机数)                             │
        │    │   z = μ + σ × ε   → (1, 4, 32, 32)  带随机性          │
        │    │                                                          │
        │    │ sample=False (推理时):                                   │
        │    │   z = μ            → (1, 4, 32, 32)  确定性            │
        │    │                                                          │
        │    │ apply_scaling=True:                                      │
        │    │   z = z × 0.18215  让方差接近1                          │
        │    │                                                          │
        │    └→ 返回 {"z": z, "mu": μ, "logvar": logvar}              │
        └──────────────────────────────────────────────────────────────┘
        """
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)  # 8通道劈成两个4通道

        if sample:
            std = (0.5 * logvar).exp()   # σ = exp(0.5 × logvar)
            eps = torch.randn_like(std)   # ε ~ N(0,1)
            z = mu + std * eps            # 重参数化: z = μ + σ × ε
        else:
            z = mu  # 推理时直接用μ

        if apply_scaling:
            z = z * self.scaling_factor  # 缩放让方差≈1

        return {"z": z, "mu": mu, "logvar": logvar}

    def decode(self, z: torch.Tensor, apply_scaling: bool = True) -> torch.Tensor:
        """
        解码: latent → 图片

        ┌──────────────────────────────────────────────────────────────┐
        │  z: (1, 4, 32, 32)                                          │
        │    │ apply_scaling=True: z = z / 0.18215  撤销编码时的缩放  │
        │    │ Decoder → (1, 3, 256, 256)                               │
        │    └→ 重建的图片                                              │
        └──────────────────────────────────────────────────────────────┘
        """
        if apply_scaling:
            z = z / self.scaling_factor
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        训练用: 编码 → 采样 → 解码 → 返回 (不加scaling)。

        ┌──────────────────────────────────────────────────────────────┐
        │  x: (1, 3, 256, 256)                                        │
        │    │ encode(sample=True, scaling=False)                      │
        │    │   → z(1,4,32,32), μ(1,4,32,32), logvar(1,4,32,32)     │
        │    │ decode(scaling=False)                                   │
        │    │   → recon(1,3,256,256)                                  │
        │    └→ {"recon": recon, "z": z, "mu": μ, "logvar": logvar}   │
        │                                                              │
        │  训练时不用scaling:                                           │
        │    scaling是给扩散模型用的, VAE自己训练时不需要              │
        │    直接在原始latent空间做重建, loss更准确                    │
        └──────────────────────────────────────────────────────────────┘
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
# 9. KL 散度
# ═══════════════════════════════════════════════════════════════════════════
def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL 散度: 衡量编码分布 N(μ, σ²) 离标准正态 N(0,1) 有多远。

    ┌──────────────────────────────────────────────────────────────────┐
    │  为什么需要 KL 散度?                                             │
    │                                                                  │
    │  没有KL: 编码器可以把不同图片映射到latent空间任意位置           │
    │    → latent空间有大量"空洞"                                    │
    │    → 从空洞采样 → 解码出乱七八糟的图                          │
    │                                                                  │
    │  有KL: 强迫所有编码分布都接近 N(0,1)                           │
    │    → latent空间更"紧凑", 没有空洞                              │
    │    → 任意采样都能解码出合理的图 ✓                              │
    │                                                                  │
    │  公式: KL = -0.5 × Σ(1 + logvar - μ² - exp(logvar))           │
    │                                                                  │
    │  数值示例 — 假设某个像素位置:                                   │
    │    μ = 0.5,  logvar = 0.2                                       │
    │    σ² = exp(0.2) ≈ 1.22                                         │
    │                                                                  │
    │    逐项计算:                                                     │
    │      1      = 1          (常数项)                                │
    │      logvar = 0.2        (log方差)                               │
    │      μ²     = 0.25       (均值偏离0的程度)                      │
    │      exp(logvar) = 1.22  (方差偏离1的程度)                      │
    │                                                                  │
    │    KL = -0.5 × (1 + 0.2 - 0.25 - 1.22)                        │
    │       = -0.5 × (-0.27)                                          │
    │       = 0.135                                                    │
    │                                                                  │
    │  什么时候 KL = 0?                                                │
    │    μ=0 且 logvar=0 → σ²=1 → 完美标准正态 → KL=0               │
    │                                                                  │
    │  两项的物理含义:                                                 │
    │    μ² 项: 均值离0越远 → KL越大 → 惩罚latent偏离原点           │
    │    exp(logvar)-1 项: 方差离1越远 → KL越大 → 惩罚太集中或太散  │
    │                                                                  │
    │  .mean() 对所有位置取平均, 返回一个标量                         │
    └──────────────────────────────────────────────────────────────────┘
    """
    return (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean()


# ═══════════════════════════════════════════════════════════════════════════
# 10. 训练损失
# ═══════════════════════════════════════════════════════════════════════════
def image_vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """
    VAE 训练损失 = 重建损失 + KL损失。

    ┌──────────────────────────────────────────────────────────────────┐
    │  数值示例:                                                      │
    │                                                                  │
    │  recon_loss (MSE — 均方误差):                                   │
    │    衡量重建图和原图的像素差异                                   │
    │    原图像素: [0.2, 0.8, 0.5, ...]                              │
    │    重建像素: [0.3, 0.7, 0.4, ...]                              │
    │    每个像素差的平方: (0.1², 0.1², 0.1², ...)                  │
    │    所有像素求平均: recon_loss ≈ 0.05                            │
    │    越小说明重建越像原图                                         │
    │                                                                  │
    │  kl_loss (KL散度):                                               │
    │    ≈ 0.135 (见上面 kl_divergence 的例子)                        │
    │    越小说明latent分布越接近标准正态                             │
    │                                                                  │
    │  total = recon_loss + kl_weight × kl_loss                       │
    │        = 0.05   + 0.000001 × 0.135                              │
    │        = 0.05   + 0.000000135                                   │
    │        ≈ 0.05                                                    │
    │                                                                  │
    │  为什么 KL 权重这么小 (1e-6)?                                    │
    │    如果KL权重太大:                                               │
    │      编码器会偷懒 → 所有图片都编码成μ≈0, σ²≈1                 │
    │      → latent全塌缩成标准正态 → 丢失所有图片信息              │
    │      → 解码不出原图 → 重建质量崩了                             │
    │                                                                  │
    │    KL权重极小:                                                   │
    │      重建是主要目标, KL只是轻轻推一下                           │
    │      "你尽量重建原图, 顺便让latent分布别太离谱就行"            │
    │                                                                  │
    │  实际 SD 还会加什么?                                             │
    │    LPIPS损失: 用预训练VGG网络比较"看起来像不像"               │
    │      → MSE关注像素值, LPIPS关注人眼感知                        │
    │    GAN损失:   加判别器, 让重建图更锐利                          │
    │      → MSE倾向输出模糊的平均图, GAN对抗这个问题               │
    │    本文件只实现了最基础的 MSE + KL                               │
    └──────────────────────────────────────────────────────────────────┘
    """
    recon_loss = F.mse_loss(recon, target)
    kl_loss = kl_divergence(mu, logvar)
    total = recon_loss + kl_weight * kl_loss
    return {"total": total, "recon_loss": recon_loss, "kl_loss": kl_loss}
