"""
Diffusion Transformer (DiT) 模块 —— 视频生成 / 文生视频 学习用实现
===================================================================

本文件实现了：
  1. TimestepEmbedding     — 将扩散时间步 t 编码为向量（正弦编码 + MLP）
  2. VideoPatchEmbedding   — 将视频切成 3D patch（时间×空间）并投影
  3. AdaLayerNorm          — 自适应层归一化（adaLN），用条件向量调制归一化参数
  4. DiTBlock              — DiT 核心块：adaLN-Zero + 自注意力 + 交叉注意力 + FFN
  5. FinalLayer            — 最终层：adaLN + 线性投影回 patch 像素空间
  6. DiT                   — 完整的 Diffusion Transformer（视频去噪网络）
  7. DDPMScheduler         — 简化版 DDPM 噪声调度器（加噪 / 去噪）
  8. TextToVideoPipeline   — 文生视频推理管线（把所有组件串起来）

=== 什么是 DiT？为什么用 DiT 做视频生成？===

  传统扩散模型（Stable Diffusion 1/2）使用 U-Net 做去噪网络。
  DiT 的核心想法是：把 U-Net 替换为 Transformer，用 patch 化的方式处理图像/视频。

  ┌────────────────────────────────────────────────────────────────────────┐
  │                        传统 vs DiT                                    │
  │                                                                        │
  │   U-Net (传统)              │    DiT (Transformer)                     │
  │  ─────────────────          │   ──────────────────                     │
  │  卷积 + 下采样/上采样       │   Patch 化 → Transformer blocks          │
  │  ResBlock + Cross-Attn      │   adaLN-Zero + Self-Attn + Cross-Attn   │
  │  固定分辨率                  │   灵活序列长度（可变分辨率/帧数）        │
  │  扩展性有限                  │   Scaling Law 更好（参数越多效果越好）   │
  └────────────────────────────────────────────────────────────────────────┘

  OpenAI 的 Sora、Google 的 Veo、字节的可灵 等都采用 DiT 架构。

=== 扩散模型核心思想（DDPM）===

  扩散模型分两个过程：

  前向过程（加噪）：逐步给数据加高斯噪声，直到变成纯噪声
    x_0 → x_1 → x_2 → ... → x_T ≈ N(0, I)

  反向过程（去噪）：学一个网络 ε_θ(x_t, t) 预测噪声，逐步去噪还原
    x_T → x_{T-1} → ... → x_1 → x_0

  训练目标：  L = E[‖ε - ε_θ(x_t, t)‖²]
  即让网络预测的噪声 ε_θ 接近真实添加的噪声 ε。

  ┌─────────────────────────────────────────────────────────────────┐
  │  前向加噪:  x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε            │
  │                                                                 │
  │  反向去噪(简化):                                                │
  │    x_{t-1} = (1/√α_t) · (x_t - (β_t/√(1-ᾱ_t)) · ε_θ) + σ·z │
  │                                                                 │
  │  其中 ᾱ_t = ∏_{s=1}^{t} α_s，α_t = 1 - β_t                   │
  └─────────────────────────────────────────────────────────────────┘

=== DiT 用于视频的整体架构 ===

  ┌──────────────────────────────────────────────────────────────────────┐
  │                                                                      │
  │  输入:  文本 prompt + 随机噪声 z_T ~ N(0,I)  (latent space)         │
  │                                                                      │
  │  ┌─────────────┐    ┌──────────────────┐                             │
  │  │ Text Encoder │    │ Timestep Embed   │                             │
  │  │ (CLIP/T5)   │    │ (sinusoidal+MLP) │                             │
  │  └──────┬──────┘    └────────┬─────────┘                             │
  │         │ text_emb           │ t_emb                                 │
  │         │                    │                                        │
  │  ┌──────▼────────────────────▼─────────────────────────────┐         │
  │  │                                                          │         │
  │  │  z_T (B, C_latent, T_frames, H_latent, W_latent)        │         │
  │  │      │                                                   │         │
  │  │      ▼                                                   │         │
  │  │  VideoPatchEmbedding  → (B, N_patches, d_model)          │         │
  │  │      │                                                   │         │
  │  │      ▼  + Position Embedding                             │         │
  │  │                                                          │         │
  │  │  ┌──────────────────────────────────────┐                │         │
  │  │  │  DiTBlock × N                        │                │         │
  │  │  │   ├─ adaLN (条件: t_emb)             │                │         │
  │  │  │   ├─ Self-Attention (patch 之间)     │                │         │
  │  │  │   ├─ Cross-Attention (文本条件注入)  │                │         │
  │  │  │   └─ FFN                              │                │         │
  │  │  └──────────────────────────────────────┘                │         │
  │  │      │                                                   │         │
  │  │      ▼                                                   │         │
  │  │  FinalLayer (adaLN → Linear → patch 像素)                │         │
  │  │      │                                                   │         │
  │  │      ▼                                                   │         │
  │  │  Unpatchify → (B, C_latent, T, H, W)   = 预测噪声 ε     │         │
  │  │                                                          │         │
  │  └──────────────────────────────────────────────────────────┘         │
  │                                                                      │
  │  DDPM 采样循环: z_{t-1} = denoise(z_t, ε_θ, t)                      │
  │  重复 T 步 → z_0                                                     │
  │                                                                      │
  │  z_0 经 VAE Decoder → 像素空间视频 (B, 3, T_frames, H, W)           │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
# ┌─────────────────“空间注意力”可视化───────────────────────┐
# │                                                        │
# │          输入 patch 排布:                               │
# │        （以一帧为例，68×120 共 8160 个 patch）          │
# │          ┌─────────── 宽度方向(nw=120) ────────────┐    │
# │      高  │  p  p  p  p  ...  p  p  p              │    │
# │      度  │  p  p  p  p  ...  p  p  p              │    │
# │     (nh) │  .  .  .  .       .  .  .              │    │
# │          │  p  p  p  p  ...  p  p  p              │    │
# │          └───────────────────────────────────────┘    │
# │                                                        │
# │  “空间注意力” —— 同一帧内所有 patch 之间做注意力          │
# │   即在每一个时间步（帧 t）上，patch 可以自由地"交流信息"   │
# │   这样模型能理解当前帧里"有什么东西、怎么排列"            │
# │                                                        │
# │  “当前是不是空间注意力？” → 是:                          │
# │    DiTBlock 的空间注意力步骤：                           │
# │     - patch 表示 reshape 为 (B*nt, S, d)（每一帧作为一个序列）  │
# │     - 在同一帧内所有 patch 间做多头自注意力                 │
# │     - 信息可在同一帧的空间内任意传递（二维 patch 网格）       │
# │                                                        │
# │  对比：                                                 │
# │    - “空间注意力”聚焦同一帧空间                       │
# │    - “时间注意力”则是在同一空间位置跨帧沟通             │
# │                                                        │
# └───────────────────────────────────────────────────────┘

参考论文/项目：
  - "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023) — DiT 原论文
  - "Sora: Creating video from text" (OpenAI, 2024) — DiT 用于视频生成
  - "Latte: Latent Diffusion Transformer for Video Generation" (Ma et al., 2024)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import MultiHeadAttention
from model.feedforward import PositionwiseFeedForward


# ═══════════════════════════════════════════════════════════════════════════
# 0. 配置
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class DiTConfig:
    """
    DiT 视频生成的超参数 —— 按 1920×1080 Full HD 视频配置。

    === 从原始视频到 DiT 输入的完整链路 ===

    ┌────────────────────────────────────────────────────────────────────────────┐
    │                                                                            │
    │  原始视频                                                                  │
    │    分辨率: 1920 × 1080 (Full HD)                                           │
    │    帧率:   24 fps                                                          │
    │    时长:   4 秒 → 96 帧                                                    │
    │    像素:   RGB 3 通道                                                      │
    │    张量:   (B, 3, 96, 1080, 1920)                                          │
    │                                                                            │
    │  ──── ↓ 预处理：Pad 到能被 8 整除 ────                                    │
    │                                                                            │
    │    1080 不能被 8 整除 → pad 到 1088 (1088 / 8 = 136)                      │
    │    1920 ÷ 8 = 240 ✓                                                       │
    │    张量:   (B, 3, 96, 1088, 1920)                                          │
    │                                                                            │
    │  ──── ↓ VAE Encoder（压缩到 latent space）────                            │
    │                                                                            │
    │    空间压缩:  8× (1088→136, 1920→240)                                     │
    │    时间压缩:  4× (96帧→24帧)                                              │
    │    通道变换:  RGB 3 → latent 4 通道                                       │
    │    张量:   (B, 4, 24, 136, 240)                                            │
    │                                                                            │
    │    为什么要压缩？ 直接在像素空间做扩散计算量太大！                          │
    │    原始: 3×96×1088×1920 ≈ 600M 像素                                       │
    │    压缩: 4×24×136×240  ≈ 3.1M 值  → 压缩了约 190 倍                      │
    │                                                                            │
    │  ──── ↓ Patch Embedding（切 3D patch）────                                │
    │                                                                            │
    │    时间 patch:  patch_t = 2 → 24/2 = 12 段                                │
    │    高度 patch:  patch_h = 2 → 136/2 = 68 段                               │
    │    宽度 patch:  patch_w = 2 → 240/2 = 120 段                              │
    │    每个 patch:  4通道 × 2 × 2 × 2 = 32 维                                 │
    │                                                                            │
    │    总 patch 数: 12 × 68 × 120 = 97,920 个 ← 这就是 Transformer 的序列长度│
    │    投影到: d_model = 1152                                                  │
    │    张量:  (B, 97920, 1152)                                                 │
    │                                                                            │
    │  ──── ↓ DiT Blocks × 28 层 ────                                           │
    │                                                                            │
    │    Self-Attention 复杂度: O(N² · d) = O(97920² × 1152)                    │
    │                         ≈ 1.1 × 10¹³ FLOPs/层                             │
    │    28 层总计:           ≈ 3 × 10¹⁴ FLOPs (仅注意力部分)                   │
    │                                                                            │
    │    这就是为什么 Sora 需要巨大 GPU 集群！                                   │
    │    实际中会用各种优化（如分块注意力、Flash Attention、                      │
    │    时空分离注意力等）来降低计算量。                                         │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘

    === 参数规模对比 ===

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  模型规格      │ d_model │ heads │ layers │ 参数量    │ 对应模型       │
    │ ───────────── │ ─────── │ ───── │ ────── │ ───────── │ ────────────── │
    │  DiT-S/2      │  384    │   6   │   12   │   33M     │ 实验 / 教学用  │
    │  DiT-B/2      │  768    │  12   │   12   │  130M     │ 小规模实验     │
    │  DiT-L/2      │ 1024    │  16   │   24   │  458M     │ 中等规模       │
    │  DiT-XL/2     │ 1152    │  16   │   28   │  675M     │ DiT 论文最大   │
    │  Sora 级别    │ ~2048+  │  32+  │   48+  │  ~3B+     │ 工业生产级     │
    │ ─────────── (本文件默认使用 DiT-XL/2 配置) ─────────────────────── │
    └─────────────────────────────────────────────────────────────────────────┘

    === 不同分辨率的 patch 数对比 ===

    ┌───────────────────────────────────────────────────────────────────────┐
    │  原始分辨率       │ latent 尺寸       │ 帧数 │ patch 数  │ 序列长度  │
    │ ──────────────── │ ──────────────── │ ──── │ ───────── │ ──────── │
    │  256×256 (教学)  │ 32×32, 4帧       │  4   │ 2×16×16   │     512  │
    │  512×512 (SD)    │ 64×64, 16帧      │ 16   │ 8×32×32   │   8,192  │
    │  720p 1280×720   │ 160×90, 24帧     │ 24   │ 12×45×80  │  43,200  │
    │  1080p 1920×1080 │ 240×136, 24帧    │ 24   │ 12×68×120 │  97,920  │ ← 本配置(4秒)
    │  4K 3840×2160    │ 480×270, 24帧    │ 24   │ 12×135×240│ 388,800  │
    └───────────────────────────────────────────────────────────────────────┘

    === 🎬 如何生成 1 分钟视频？ ===

    1 分钟 1080p@24fps = 1440 帧，如果按 4 秒配置的方式直接硬算：
      1440 帧 / 4(VAE时间压缩) = 360 时间帧 latent
      patch 数 = (360/2) × 68 × 120 = 1,468,800 个 patch！

    Self-Attention 是 O(N²)，1,468,800² ≈ 2.2 × 10¹² 次乘法/层
    → 根本不可能直接全局注意力！

    现实中的解决方案：

    ┌──────────────────────────────────────────────────────────────────────────┐
    │                                                                          │
    │  方案一：时间自回归分块生成（主流方案，Sora / 可灵 / Gen-3 等使用）      │
    │  ══════════════════════════════════════════════════════════════════════   │
    │                                                                          │
    │  核心思想：不一次生成整段视频，而是每次生成一小段（如 4 秒），             │
    │           用前一段的最后几帧作为下一段的条件，滚动生成。                   │
    │                                                                          │
    │   时间轴:                                                                │
    │   ┌──────────┐                                                           │
    │   │ 第1段 4s │ ← 从纯噪声生成                                          │
    │   │ (0~4s)   │                                                           │
    │   └────┬─────┘                                                           │
    │        │ 最后几帧作为条件                                                │
    │        ▼                                                                 │
    │   ┌──────────┐                                                           │
    │   │ 第2段 4s │ ← 条件生成（保证时间连贯）                              │
    │   │ (4~8s)   │                                                           │
    │   └────┬─────┘                                                           │
    │        │                                                                 │
    │        ▼  ...重复 15 次...                                               │
    │   ┌──────────┐                                                           │
    │   │ 第15段4s │                                                           │
    │   │ (56~60s) │                                                           │
    │   └──────────┘                                                           │
    │                                                                          │
    │   每段独立去噪，序列长度仍然是 ~97,920 ← 可控！                         │
    │   总计 15 段 × 50 步去噪 × 2(CFG) = 1500 次 DiT 前向                    │
    │                                                                          │
    │   挑战：段与段之间的过渡需要平滑处理，否则会有"跳帧"                    │
    │   解决：重叠几帧 + 渐变混合（overlap blending）                          │
    │                                                                          │
    │  方案二：时空分离注意力（几乎所有长视频模型都用）                        │
    │  ══════════════════════════════════════════════════════════════════════   │
    │                                                                          │
    │  核心思想：不做全局 N²注意力，把空间和时间拆开分别做。                   │
    │                                                                          │
    │  全局注意力:                                                             │
    │    每个 patch 看所有 97,920 个 patch → O(97920²) ≈ 10¹⁰                 │
    │                                                                          │
    │  时空分离:                                                               │
    │    Step A: Spatial Attention（空间注意力）                               │
    │      同一帧内的 68×120=8,160 个 patch 互相看                             │
    │      → O(8160²) ≈ 6.7 × 10⁷                                            │
    │                                                                          │
    │    Step B: Temporal Attention（时间注意力）                              │
    │      同一空间位置、不同帧的 12 个 patch 互相看                           │
    │      → O(12²) = 144                                                      │
    │                                                                          │
    │    总计: 8160² × 12 + 12² × 8160 ≈ 8 × 10⁸                            │
    │    对比全局: 97920² ≈ 10¹⁰                                              │
    │    → 节省约 12 倍计算！                                                  │
    │                                                                          │
    │  可视化:                                                                 │
    │                                                                          │
    │    ┌──────────────────┐    ┌──────────────────┐                         │
    │    │   Spatial Attn   │    │  Temporal Attn    │                         │
    │    │                  │    │                   │                         │
    │    │  帧 t:           │    │  位置(i,j):       │                         │
    │    │  ┌──┬──┬──┬──┐  │    │  t=0 ●            │                         │
    │    │  │● │⇆ │⇆ │⇆ │  │    │      ↕            │                         │
    │    │  ├──┼──┼──┼──┤  │    │  t=1 ●            │                         │
    │    │  │⇆ │⇆ │⇆ │⇆ │  │    │      ↕            │                         │
    │    │  ├──┼──┼──┼──┤  │    │  t=2 ●            │                         │
    │    │  │⇆ │⇆ │⇆ │⇆ │  │    │      ↕            │                         │
    │    │  └──┴──┴──┴──┘  │    │  ... ●            │                         │
    │    │ 同帧 patch 互看  │    │ 同位置跨帧互看    │                         │
    │    └──────────────────┘    └──────────────────┘                         │
    │                                                                          │
    │  方案三：多尺度 / 先粗后精（Pyramidal / Cascaded）                       │
    │  ══════════════════════════════════════════════════════════════════════   │
    │                                                                          │
    │  核心思想：先生成低分辨率+低帧率的版本，再逐级放大。                      │
    │                                                                          │
    │    Stage 1: 生成 480×270, 6fps, 60秒 → 360帧                            │
    │             latent: 60×34×34 → patch 数 ~35,000 (可控)                   │
    │                                                                          │
    │    Stage 2: 空间超分 480→1080p                                           │
    │             逐帧或分块超分，类似图像超分辨率                              │
    │                                                                          │
    │    Stage 3: 时间插帧 6fps→24fps                                          │
    │             用帧插值模型补帧                                              │
    │                                                                          │
    │    ┌────────┐     ┌────────────┐     ┌──────────────┐                   │
    │    │Stage 1 │ →   │  Stage 2   │ →   │   Stage 3    │                   │
    │    │低分低帧│     │ 空间超分   │     │  时间插帧    │                   │
    │    │480p 6f │     │→ 1080p 6f  │     │→ 1080p 24f   │                   │
    │    └────────┘     └────────────┘     └──────────────┘                   │
    │                                                                          │
    │  方案四：更激进的 VAE 压缩                                               │
    │  ══════════════════════════════════════════════════════════════════════   │
    │                                                                          │
    │  用更强力的 3D VAE，把视频压缩得更狠：                                    │
    │    - 时间压缩 8× 甚至 16×（而非 4×）                                    │
    │    - 空间压缩 16× 甚至 32×（而非 8×）                                   │
    │                                                                          │
    │  1分钟 1080p@24fps = 1440帧:                                             │
    │    VAE 4×时间 + 8×空间: → latent (4, 360, 136, 240) → 太大              │
    │    VAE 8×时间 + 16×空间: → latent (16, 180, 68, 120)                    │
    │      → patch(2×2×2): 90×34×60 = 183,600 → 还是大                       │
    │    VAE 16×时间 + 16×空间: → latent (16, 90, 68, 120)                    │
    │      → patch(2×2×2): 45×34×60 = 91,800 → 接近单段 4 秒的量级           │
    │                                                                          │
    │  代价：压缩太狠会丢失细节，VAE 重建质量下降                              │
    │                                                                          │
    │  === 实际产品的做法（综合使用以上方案）===                               │
    │                                                                          │
    │  ┌───────────────────────────────────────────────────────────────────┐   │
    │  │  模型          │ 最长时长 │ 主要策略                              │   │
    │  │ ─────────────  │ ─────── │ ─────────────────────────────────── │   │
    │  │  Sora (OpenAI) │  ~60s   │ 时空分离注意力 + 分块生成 + 超大模型│   │
    │  │  可灵 (快手)   │  ~120s  │ 自回归分块 + 时空分离 + 3D VAE      │   │
    │  │  Veo2 (Google) │  ~60s   │ 级联生成(先低分后超分) + 分块       │   │
    │  │  Gen-3 (Runway)│  ~10s   │ 时空分离注意力 + DiT                │   │
    │  │  CogVideoX     │  ~6s    │ 3D VAE(4×4×4压缩) + 全局注意力     │   │
    │  │  HunyuanVideo  │  ~5s    │ 双流DiT(文本流+视频流) + 全局注意力 │   │
    │  └───────────────────────────────────────────────────────────────────┘   │
    │                                                                          │
    │  结论：没有任何模型能一次性全局注意力处理 1 分钟 1080p 视频。            │
    │       都是通过 "拆分(时间/空间) + 压缩(VAE) + 级联(先粗后精)"           │
    │       的组合策略来实现的。                                                │
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘
    """

    # =====================================================================
    # 视频 latent 参数
    # =====================================================================
    # 假设原始视频已经过 VAE 编码到 latent space
    # VAE 空间压缩 8×，时间压缩 4×
    #
    # 原始视频: 1920×1080, 96帧(4秒×24fps)
    # → pad 到 1920×1088 (让高度能被 8 整除)
    # → VAE encode → latent: (B, 4, 24, 136, 240)

    num_frames: int = 24  # VAE 时间压缩后的帧数 (原始 96 帧 / 4 = 24)
    latent_h: int = 136  # latent 高度 (原始 1088px / 8 = 136)
    latent_w: int = 240  # latent 宽度 (原始 1920px / 8 = 240)
    in_channels: int = 4  # VAE latent 通道数（SD-VAE = 4, SD3-VAE = 16）

    # =====================================================================
    # Patch 参数
    # =====================================================================
    # 3D patch: 每个 patch 覆盖 (2帧 × 2高 × 2宽) 的 latent 区域
    # 总 patch 数 = (24/2) × (136/2) × (240/2) = 12 × 68 × 120 = 97,920
    #
    # 这 97,920 就是 Transformer 的序列长度——对比语言模型的 4K~128K token，
    # 视频生成的序列长度可以非常大，这是计算量的主要来源。

    patch_size_t: int = 2  # 时间 patch 大小（每个 patch 跨 2 帧）
    patch_size_h: int = 2  # 空间高度 patch 大小
    patch_size_w: int = 2  # 空间宽度 patch 大小

    # =====================================================================
    # Transformer 参数 (DiT-XL/2 配置)
    # =====================================================================
    # DiT 论文中最大的模型配置，也是性能最好的
    # 参数量约 675M（不含 VAE 和文本编码器）

    d_model: int = 1152  # 隐藏维度（DiT-XL = 1152, Sora 级别可能 2048+）
    n_heads: int = 16  # 注意力头数（d_k = 1152/16 = 72 per head）
    n_layers: int = 28  # DiT block 层数（DiT-XL = 28 层）
    d_ff: int = 4608  # FFN 中间层维度（= 4 × d_model = 4 × 1152）
    dropout: float = 0.0  # DiT 论文不使用 dropout（数据量够大时不需要）

    # =====================================================================
    # 文本条件
    # =====================================================================
    # 真实系统中用预训练文本编码器:
    #   - CLIP ViT-L: dim=768,  max_len=77   (Stable Diffusion 1/2 使用)
    #   - T5-XXL:     dim=4096, max_len=256  (Imagen, SD3, 视频模型常用)
    #   - CLIP + T5 双编码器组合 (SD3, Flux 使用)
    #
    # 这里按 T5-XXL 配置（视频生成需要更强的文本理解能力）

    text_embed_dim: int = 4096  # T5-XXL 输出维度
    max_text_len: int = 256  # 最大文本 token 数（视频描述通常比图像更长）

    # =====================================================================
    # 扩散参数
    # =====================================================================
    # 标准 DDPM 使用 1000 步，推理时可用 DDIM/DPM-Solver 加速到 20~50 步

    num_timesteps: int = 1000  # 扩散总步数 T

    # =====================================================================
    # 注意力模式
    # =====================================================================
    # True  → 时空分离注意力 (STDiTBlock): 空间 O(S²)×nt + 时间 O(nt²)×S
    #          1080p 可支持更长视频（~20s+ 单卡），商业模型都用这个
    # False → 全局注意力 (DiTBlock): O(N²) 所有 patch 互看
    #          1080p 约 4 秒就到显存极限，但信息传播更直接

    use_st_attn: bool = True  # 默认开启时空分离，大幅降低显存和计算量

    # =====================================================================
    # 计算属性
    # =====================================================================

    @property
    def num_patches_t(self) -> int:
        """时间维 patch 数: 24 / 2 = 12"""
        return self.num_frames // self.patch_size_t

    @property
    def num_patches_h(self) -> int:
        """高度 patch 数: 136 / 2 = 68"""
        return self.latent_h // self.patch_size_h

    @property
    def num_patches_w(self) -> int:
        """宽度 patch 数: 240 / 2 = 120"""
        return self.latent_w // self.patch_size_w

    @property
    def num_patches(self) -> int:
        """
        总 patch 数（= Transformer 序列长度）。

        1080p: 12 × 68 × 120 = 97,920 个 patch
        对比: GPT-4 上下文 128K token, 但 GPT 的 d_model 更大

        Self-Attention 复杂度 O(N²·d):
          97920² × 1152 ≈ 1.1 × 10¹³ FLOPs / 层
          × 28 层 ≈ 3.1 × 10¹⁴ FLOPs (仅注意力)

        这就是为什么实际系统需要:
          - Flash Attention (减少显存，不减计算)
          - 时空分离注意力 (先 spatial-only, 再 temporal-only, 大幅降低 N)
          - Sliding window / 分块注意力
          - 多卡并行 (Tensor Parallel + Sequence Parallel)
        """
        return self.num_patches_t * self.num_patches_h * self.num_patches_w

    @property
    def patch_dim(self) -> int:
        """
        每个 patch 展平后的维度 = C × pt × ph × pw。

        4 × 2 × 2 × 2 = 32 维 → 投影到 d_model=1152 维
        """
        return (
            self.in_channels * self.patch_size_t * self.patch_size_h * self.patch_size_w
        )

    @property
    def st_attn_speedup(self) -> float:
        """时空分离注意力相比全局注意力的加速比。"""
        N = self.num_patches
        S = self.num_patches_h * self.num_patches_w
        nt = self.num_patches_t
        full_cost = N * N
        st_cost = S * S * nt + nt * nt * S
        return full_cost / st_cost if st_cost > 0 else 1.0

    @property
    def original_video_shape(self) -> str:
        """还原到原始视频的尺寸（供参考）。"""
        raw_t = self.num_frames * 4  # VAE 时间压缩 4×
        raw_h = self.latent_h * 8  # VAE 空间压缩 8×
        raw_w = self.latent_w * 8
        return f"({raw_t} frames, {raw_h}×{raw_w} pixels) ≈ {raw_t / 24:.1f}s @24fps"

    @property
    def estimated_params_m(self) -> float:
        """粗略估算模型参数量（百万）。"""
        # 每层: self-attn(4d²) + cross-attn(4d²) + FFN(8d²) + adaLN ≈ 17d²
        per_layer = 17 * self.d_model**2
        total = self.n_layers * per_layer
        # 加上 patch_embed, pos_embed, final_layer, text_proj 等
        total += self.num_patches * self.d_model  # pos_embed
        total += self.d_model * self.patch_dim  # patch_embed (近似)
        total += self.d_model * self.patch_dim  # final_layer
        return total / 1e6


# ═══════════════════════════════════════════════════════════════════════════
# 1. Timestep Embedding — 时间步编码
# ═══════════════════════════════════════════════════════════════════════════
class TimestepEmbedding(nn.Module):
    """
    将扩散时间步 t ∈ {0, 1, ..., T-1} 编码为向量。

    为什么需要时间步编码？
      扩散模型在不同时间步 t 的噪声程度不同：
        - t 大 → 噪声多，模型需要做"粗粒度"去噪
        - t 小 → 噪声少，模型需要做"精细"修复
      所以网络必须知道当前是第几步，才能做出正确的去噪行为。

    实现方式（与 Transformer 位置编码相同的正弦编码 + 两层 MLP）：
      t → sinusoidal_encoding(t) → Linear → SiLU → Linear → t_emb

    ┌────────────────────────────────────────────┐
    │  t = 500                                    │
    │  │                                          │
    │  ▼                                          │
    │  sin/cos 正弦编码 → (d_model,)              │
    │  │                                          │
    │  ▼                                          │
    │  Linear(d_model → 4*d_model) → SiLU         │
    │  │                                          │
    │  ▼                                          │
    │  Linear(4*d_model → d_model) → t_emb        │
    └────────────────────────────────────────────┘

    参数:
        d_model (int): 输出嵌入维度
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),  # SiLU = x·sigmoid(x)，比 ReLU 更平滑
            nn.Linear(4 * d_model, d_model),
        )

    # 这里的t是一个张量，形状是(B,)，表示时间步的索引，范围是0到T-1。
    def sinusoidal_encoding(self, t: torch.Tensor) -> torch.Tensor:
        """
        正弦时间步编码，与 positional.py 中的位置编码原理相同。

        t: (B,) — 整数时间步
        返回: (B, d_model)
        """
        #d_model的值是1152
        #half_dim的值是1152/2=576
        half_dim = self.d_model // 2
        # 频率因子：exp(-ln(10000) * i / (d/2))
        #half_dim不是一个int吗？怎么还能是张量？   
        #freq是一个数组，形状是(half_dim,)其中half_dim是576是一个int，为什么还能是张量？
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / half_dim
        )
        # t 与频率做外积: (B, 1) * (half_dim,) → (B, half_dim)
        #unsqueeze函数是用来在指定维度上增加一个维度，这里在1维度上增加一个维度，所以形状是(B, 1, half_dim)
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)
        # 拼接 sin 和 cos: (B, d_model)
        #cat函数是用来拼接两个张量，这里拼接的是sin(args)和cos(args)，拼接后的形状是(B, d_model*2)
        #拼接方式是沿着最后一个维度拼接，所以拼接后的形状是(B, d_model*2)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) — 时间步（整数或浮点数）
        返回: (B, d_model) — 时间步嵌入向量
        """
        #现在的t的值是999到0的列表
        #t_emb的值是(B, d_model)
        t_emb = self.sinusoidal_encoding(t)  # (B, d_model)
        #t_emb的值是(B, d_model)
        #t_emb的值是(B, d_model)
        return self.mlp(t_emb)  # (B, d_model)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Video Patch Embedding — 3D patch 切分 + 投影
# ═══════════════════════════════════════════════════════════════════════════
class VideoPatchEmbedding(nn.Module):
    """
    将视频 latent 切成 3D patch（时间 × 高度 × 宽度），并投影到 d_model 维。

    与 ViT 的 PatchEmbedding 对比：
      - ViT:  2D — Conv2d(kernel=P×P)  — 图像 (B, C, H, W) → (B, N, d)
      - DiT:  3D — Conv3d(kernel=Pt×Ph×Pw) — 视频 (B, C, T, H, W) → (B, N, d)
    # 这里的24指的是经过VAE压缩后的latent视频帧数T（例如T=24）。
    # 也就是说，原始视频在送入Transformer前，先被VAE等模块处理，变成24帧的latent空间表示。
    # patch_t=2表示每个patch在时间轴上覆盖2帧。
    # 那么时间维度上可以切出的patch段数是T // patch_t = 24 // 2 = 12。
    # 所以后续你看到的nt实际上是“时间轴被patch切段后的段数”，对应12。
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │  输入视频 latent: (B, C=4, T=24, H=136, W=240)                      │
    │  (原始 1920×1080 视频经 VAE 压缩后)                                  │
    │                                                                      │
    │  时间维切分:  T=24 / patch_t=2 = 12 段                               │
    │  高度切分:    H=136 / patch_h=2 = 68 段                              │
    │  宽度切分:    W=240 / patch_w=2 = 120 段                             │
    │                                                                      │
    │  总 patch 数:  12 × 68 × 120 = 97,920 个 patch                      │
    │  每个 patch:   4 × 2 × 2 × 2 = 32 维  →  投影到 d_model=1152 维    │
    │                                                                      │
    │  输出: (B, 97920, 1152)                                              │
    │                                                                      │
    │  可视化（时间维展开）:                                               │
    │                                                                      │
    │   帧 0-1:              帧 2-3:              帧 22-23:                │
    │  ┌──┬──┬──┬───┬──┐   ┌──┬──┬──┬───┬──┐    ┌──┬──┬──┬───┬──┐       │
    │  │p │p │..│   │p │   │p │p │..│   │p │    │p │p │..│   │p │       │
    │  ├──┼──┼──┼───┼──┤   ├──┼──┼──┼───┼──┤ .. ├──┼──┼──┼───┼──┤       │
    │  │: │  │  │   │  │   │  │  │  │   │  │    │  │  │  │   │  │       │
    │  ├──┼──┼──┼───┼──┤   ├──┼──┼──┼───┼──┤    ├──┼──┼──┼───┼──┤       │
    │  │p │p │..│   │p │   │p │p │..│   │p │    │p │p │..│   │p │       │
    │  └──┴──┴──┴───┴──┘   └──┴──┴──┴───┴──┘    └──┴──┴──┴───┴──┘       │
    │   68×120=8160 patches  8160 patches         8160 patches             │
    │  (68行×120列 空间)     (68行×120列 空间)    (68行×120列 空间)        │
    │                                                                      │
    │  × 12 个时间段 = 97,920 patches total                                │
    │                                                                      │
    │  对比: ViT-Base 处理 224×224 图像只有 196 个 patch                    │
    │        1080p 视频是图像的 ~500 倍序列长度！                           │
    └──────────────────────────────────────────────────────────────────────┘

    参数:
        cfg (DiTConfig): 配置
    """

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg

        # Conv3d 同时完成 3D 切分 + 线性投影
        # kernel_size = stride = (patch_t, patch_h, patch_w) → 不重叠切分
        self.proj = nn.Conv3d(
            in_channels=cfg.in_channels,
            out_channels=cfg.d_model,
            kernel_size=(cfg.patch_size_t, cfg.patch_size_h, cfg.patch_size_w),
            stride=(cfg.patch_size_t, cfg.patch_size_h, cfg.patch_size_w),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W) — 视频 latent
        返回: (B, num_patches, d_model)
        """
        # Conv3d 输出: (B, d_model, T/pt, H/ph, W/pw)
        # 输入的 x 形状: (B, 4, 24, 136, 240)
        # 每个维度含义如下：
        #       B    — batch size，样本数
        #       4    — 通道数，来自 VAE latent 空间（SD-VAE 输出是 4 通道）
        #       24   — 时间帧数。原图像采样 4 秒 × 24fps = 96 帧，VAE 时间方向压缩 4×，所以 96 / 4 = 24
        #       136  — 高度（像素）。原图像 pad 到 1088，高度方向被 VAE 空间压缩 8×，所以 1088 / 8 = 136
        #       240  — 宽度（像素）。原图像宽度 1920，同样空间压缩 8×，1920 / 8 = 240
        # 这些数的来源都是前面 DiTConfig 参数说明和视频预处理/VAE 编码流程

        # 输出是 (B, d_model, T/pt, H/ph, W/pw)
        # 具体为 (B, 1152, 12, 68, 120)
        # 说明：
        #   B         — batch size，批尺寸
        #   d_model   — 投影后特征维度（来自 cfg.d_model = 1152）
        #   12        — 时间维分块数量，= num_frames // patch_size_t = 24 // 2 = 12
        #   68        — 高度分块数量，= latent_h // patch_size_h = 136 // 2 = 68
        #   120       — 宽度分块数量，= latent_w // patch_size_w = 240 // 2 = 120
        # 即经过 Conv3d 3D patch 切分 + 投影之后，得到 (B, 1152, 12, 68, 120)

        x = self.proj(x)
        # flatten之前的值 x= (B, 1152, 12, 68, 120)
        # flatten之后的值 x= (B, d_model, 12*68*120) = (B, d_model, 97920)
        x = x.flatten(2)

        # transpose: → (B, num_patches, d_model)
        # 交换1和2的维度，对齐多头注意力计算的形状要求
        x = x.transpose(1, 2)
        # 交换之后的值 x= (B, 97920, d_model)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# 3. Adaptive Layer Norm (adaLN) — DiT 的核心创新
# ═══════════════════════════════════════════════════════════════════════════
class AdaLayerNorm(nn.Module):
    """
    自适应层归一化 (Adaptive Layer Normalization, adaLN)。

    === 为什么需要 adaLN？===

    标准 LayerNorm：参数 γ(scale) 和 β(shift) 是固定学来的，对所有输入都一样。
    adaLN：γ 和 β 由条件向量（如时间步 t）*动态生成*，让归一化行为随条件变化。

    这是 DiT 与普通 ViT 最关键的区别之一！

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  标准 LayerNorm:                                                │
    │    y = γ · (x - μ) / σ + β        ← γ, β 是固定参数           │
    │                                                                 │
    │  adaLN:                                                         │
    │    [γ, β] = Linear(condition)      ← γ, β 由条件向量生成       │
    │    y = γ · LayerNorm(x) + β        ← 动态调制                  │
    │                                                                 │
    │  adaLN-Zero (DiT 使用):                                        │
    │    [γ, β, α] = Linear(condition)   ← 额外输出一个门控 α        │
    │    y = x + α · Attention(γ · LN(x) + β)                        │
    │                                                                 │
    │    初始化时 α = 0，这样训练初期每个 DiT Block 像恒等映射，      │
    │    有助于深层网络的训练稳定性。                                  │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

    参数:
        d_model (int):    特征维度
        cond_dim (int):   条件向量维度（通常等于 d_model）
    """

    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # 输出 6 个调制参数: (γ1, β1, α1, γ2, β2, α2)
        # 分别用于 注意力子层 和 FFN 子层
        # 共 6 × d_model 个参数
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model),
        )
        self._init_zero()

    def _init_zero(self):
        """将输出层初始化为零 → adaLN-Zero 策略"""
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        x:    (B, N, d_model)  — 输入特征
        cond: (B, d_model)     — 条件向量（如时间步嵌入）

        返回 6 个调制参数，每个形状为 (B, 1, d_model)：
          (γ1, β1, α1, γ2, β2, α2)

        调用方式（在 DiTBlock 中）：
          γ1, β1, α1, γ2, β2, α2 = ada_ln(x, t_emb)
          # 注意力子层
          h = self_attn(γ1 * norm(x) + β1)
          x = x + α1 * h
          # FFN 子层
          h = ffn(γ2 * norm(x) + β2)
          x = x + α2 * h
        """
        # (B, 6*d_model)
        params = self.adaLN_modulation(cond)
        # 拆分为 6 组，每组 (B, d_model) → unsqueeze → (B, 1, d_model)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = params.chunk(6, dim=-1)
        return (
            gamma1.unsqueeze(1),
            beta1.unsqueeze(1),
            alpha1.unsqueeze(1),
            gamma2.unsqueeze(1),
            beta2.unsqueeze(1),
            alpha2.unsqueeze(1),
        )

    def modulate(
        self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """
        执行 adaLN: γ · LayerNorm(x) + β

        x:     (B, N, d_model)
        gamma: (B, 1, d_model)
        beta:  (B, 1, d_model)
        返回:  (B, N, d_model)
        """
        return gamma * self.norm(x) + beta


# ═══════════════════════════════════════════════════════════════════════════
# 4. DiT Block — 核心 Transformer 块
# ═══════════════════════════════════════════════════════════════════════════
class DiTBlock(nn.Module):
    """
    DiT 的核心 Transformer 块。

    与 ViT 的 EncoderBlock 和 Decoder 的 DecoderBlock 的对比：

    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │  ViT EncoderBlock:        │  DecoderBlock:      │  DiTBlock:         │
    │  ──────────────────        │  ───────────────     │  ─────────────    │
    │  LN → Self-Attn → +       │  LN → Masked        │  adaLN → Self     │
    │  LN → FFN → +             │    Self-Attn → +     │    Attn → gate +  │
    │                            │  LN → FFN → +       │  adaLN → Cross    │
    │  归一化: 固定 LN           │  归一化: 固定 LN    │    Attn → +       │
    │  条件注入: 无              │  条件注入: 无       │  adaLN → FFN      │
    │  掩码: 无                  │  掩码: 因果         │    → gate +       │
    │                            │                     │  归一化: adaLN    │
    │                            │                     │  条件注入: t_emb  │
    │                            │                     │  掩码: 无         │
    │                            │                     │  + Cross-Attn     │
    │                            │                     │    (文本条件)     │
    └──────────────────────────────────────────────────────────────────────┘

    adaLN-Zero 的数据流：

      条件 c (timestep) ──→ adaLN ──→ (γ1, β1, α1, γ2, β2, α2)
                                          │         │
          ┌─────────────────────────────────┘         │
          ▼                                           │
      x → γ1·LN(x)+β1 → Self-Attn → ×α1 → +x       │
                                              │       │
          ┌───────────────────────────────────┘       │
          ▼                                           │
      x → LN(x) → Cross-Attn(Q=x, KV=text) → +x    │
                                                │     │
          ┌─────────────────────────────────────┘     │
          ▼                                           ▼
      x → γ2·LN(x)+β2 → FFN → ×α2 → +x → out

    参数:
        d_model   (int):  特征维度
        n_heads   (int):  注意力头数
        d_ff      (int):  FFN 中间层维度
        dropout   (float): dropout 比例
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # --- adaLN-Zero 参数生成 ---
        self.ada_ln = AdaLayerNorm(d_model, cond_dim=d_model)

        # --- 自注意力（patch 之间的全局关系）---
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # --- 交叉注意力（文本条件注入）---
        # Q 来自图像 patch，K/V 来自文本编码
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn_norm = nn.LayerNorm(d_model)

        # --- 前馈网络 ---
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        text_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:        (B, N, d_model)         — patch 序列
        t_emb:    (B, d_model)            — 时间步条件向量
        text_emb: (B, text_len, d_model)  — 文本编码（可选，无条件时为 None）

        返回:     (B, N, d_model)
        """
        # 1) 生成 adaLN-Zero 的 6 个调制参数
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.ada_ln(x, t_emb)

        # 2) 自注意力子层 (adaLN-Zero 风格)
        #    先用 γ1, β1 调制归一化，再过注意力，最后用 α1 做门控
        h = self.ada_ln.modulate(x, gamma1, beta1)
        h = self.self_attn(h, h, h, mask=None)  # 双向注意力，无 mask
        x = x + alpha1 * h  # α 门控 + 残差

        # 3) 交叉注意力子层（文本条件注入）
        if text_emb is not None:
            h = self.cross_attn_norm(x)
            # Q 来自 patch 特征，K/V 来自文本 → 每个 patch 都能关注所有文本 token
            h = self.cross_attn(h, text_emb, text_emb, mask=None)
            x = x + h

        # 4) FFN 子层 (adaLN-Zero 风格)
        h = self.ada_ln.modulate(x, gamma2, beta2)
        h = self.ffn(h)
        x = x + alpha2 * h

        return x


# ═══════════════════════════════════════════════════════════════════════════
# 4b. STDiTBlock — 时空分离注意力版本（商业模型使用的核心优化）
# ═══════════════════════════════════════════════════════════════════════════
class STDiTBlock(nn.Module):
    """
    Spatial-Temporal DiT Block — 时空分离注意力。

    这是 CogVideoX / Latte / Open-Sora 等模型的核心优化。
    把原来一次性处理所有 97,920 个 patch 的全局注意力，
    拆成 "先空间、再时间" 两步局部注意力。

    === 全局 vs 时空分离 ===

    全局注意力 (DiTBlock):
      每个 patch 看所有 N = nt × nh × nw 个 patch
      复杂度: O(N²) = O((nt·nh·nw)²)

      1080p: O(97920²) ≈ 9.6 × 10⁹

    时空分离注意力 (STDiTBlock):
      Step 1 — Spatial Attention: 同一帧内的 nh×nw 个 patch 互看
        复杂度: O((nh·nw)²) × nt 次

      Step 2 — Temporal Attention: 同一空间位置、不同帧的 nt 个 patch 互看
        复杂度: O(nt²) × (nh·nw) 次

      1080p: O(8160²)×12 + O(12²)×8160 ≈ 8.0 × 10⁸

      节省: 9.6×10⁹ / 8.0×10⁸ ≈ 12 倍！

    ┌──────────────────────────────────────────────────────────────────────────┐
    │                                                                          │
    │  输入 x: (B, nt×nh×nw, d)  例如 (B, 12×68×120, 1152)                   │
    │                                                                          │
    │  ─── Step 1: Spatial Attention（同帧内的 patch 互看）───                 │
    │                                                                          │
    │    reshape → (B×nt, nh×nw, d) = (B×12, 8160, 1152)                      │
    │                                                                          │
    │    帧 0:  ┌──┬──┬──┬──┐    每帧内部 8160 个 patch                       │
    │           │⇆ │⇆ │⇆ │⇆ │    做双向自注意力                               │
    │           ├──┼──┼──┼──┤    复杂度: O(8160²)                              │
    │           │⇆ │⇆ │⇆ │⇆ │                                                 │
    │           └──┴──┴──┴──┘                                                  │
    │    帧 1:  ┌──┬──┬──┬──┐    12 帧并行处理（batch 维合并）                │
    │           │⇆ │⇆ │⇆ │⇆ │                                                 │
    │           └──┴──┴──┴──┘                                                  │
    │    ...                                                                   │
    │                                                                          │
    │    reshape → (B, nt×nh×nw, d)                                            │
    │                                                                          │
    │  ─── Step 2: Temporal Attention（同位置跨帧互看）───                     │
    │                                                                          │
    │    reshape → (B×nh×nw, nt, d) = (B×8160, 12, 1152)                      │
    │                                                                          │
    │    位置(0,0):  t=0 ● ↕ t=1 ● ↕ t=2 ● ↕ ... ↕ t=11 ●                   │
    │    位置(0,1):  t=0 ● ↕ t=1 ● ↕ t=2 ● ↕ ... ↕ t=11 ●                   │
    │    ...         每个空间位置的 12 帧做注意力                               │
    │                复杂度: O(12²) = O(144) — 几乎免费！                      │
    │                                                                          │
    │    reshape → (B, nt×nh×nw, d)                                            │
    │                                                                          │
    │  ─── Step 3: Cross-Attention（文本条件注入）───                          │
    │  ─── Step 4: FFN ───                                                     │
    │                                                                          │
    │  输出: (B, nt×nh×nw, d)                                                  │
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘

    为什么这样拆分是合理的？
      - 空间注意力：学习"这一帧里有什么东西、怎么排列"
      - 时间注意力：学习"这个东西在不同帧之间怎么运动"
      - 两者交替，信息仍然能在时空之间传播（只是需要多层才能传远）

    参数:
        d_model   (int): 特征维度
        n_heads   (int): 注意力头数
        d_ff      (int): FFN 中间层维度
        dropout   (float): dropout 比例
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # --- adaLN-Zero 参数生成 ---
        self.ada_ln = AdaLayerNorm(d_model, cond_dim=d_model)

        # --- 空间注意力（同一帧内 patch 互看）---
        self.spatial_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # --- 时间注意力（同一位置跨帧互看）---
        self.temporal_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.temporal_norm = nn.LayerNorm(d_model)

        # --- 交叉注意力（文本条件注入）---
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn_norm = nn.LayerNorm(d_model)

        # --- 前馈网络 ---
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,                # 输入 patch 张量。形状 (B, N, d_model)，N=nt×nh×nw。由 patch embedding 得到，包含当前时刻的所有视频 patch 的特征。
        t_emb: torch.Tensor,            # 时间步条件嵌入向量。形状 (B, d_model)。由当前扩散步数 t 经 TimestepEmbedding 得到，为每个 batch 样本提供时间条件。
        text_emb: torch.Tensor | None = None,  # 文本编码。形状 (B, text_len, d_model)，可为 None。text_encoder 得到的文本特征，经过 Linear 投影对齐到 d_model 维度，用于 cross-attention。无文本条件则为 None。
        nt: int = 1,                    # 时间维 patch 数（每段视频有多少时间片 patch）。由配置 cfg.num_patches_t 得来，如 12。
        nh: int = 1,                    # 高度方向 patch 数。由配置 cfg.num_patches_h 得来，如 68。
        nw: int = 1,                    # 宽度方向 patch 数。由配置 cfg.num_patches_w 得来，如 120。
    ) -> torch.Tensor:
        """
        x:        (B, N, d_model)         — patch 序列, N = nt × nh × nw
        t_emb:    (B, d_model)            — 当前 timestep 的条件向量
        text_emb: (B, text_len, d_model)  — 文本编码，如果没有文本条件为 None
        nt, nh, nw: patch 网格形状 —— nt 表示时间方向 patch 数，nh、nw 分别是高度和宽度方向 patch 数
        返回:     (B, N, d_model)

        下面详细分解每一步的核心操作和目的，适合初学者理解。
        """

        B = x.size(0)           # Batch 大小
        S = nh * nw             # 每个时间帧上，空间 patch 的数量

        # 步骤 1：根据输入 patch 序列 x 和步数条件 t_emb，计算 adaLN-Zero 用的调制参数
        # 得到六个参数：gamma1, beta1, alpha1（控制空间注意力残差），gamma2, beta2, alpha2（控制 FFN 残差）
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.ada_ln(x, t_emb)

        # 步骤 2：空间注意力（spatial attention），只在同一帧内部做 patch 之间的信息交换
        #   2.1 先用第一组调制参数(gamma1, beta1)对所有 patch 特征做调制
        h = self.ada_ln.modulate(x, gamma1, beta1)   # h 形状 (B, nt×S, d)
        #   2.2 把 batch 维和时间帧维合并，方便在同一帧内部做注意力
        h = h.reshape(B * nt, S, -1)                 # 形状 (B*nt, S, d)
        #   2.3 在空间维做多头注意力（同一帧内 patch 自由交流信息）
        h = self.spatial_attn(h, h, h, mask=None)
        #   2.4 恢复原始的 (B, nt×S, d) 形状
        h = h.reshape(B, nt * S, -1)
        #   2.5 用 alpha1 控制残差比例，然后加回到原始 x
        x = x + alpha1 * h

        # 步骤 3：时间注意力（temporal attention），同一空间位置上跨越不同帧实现交流
        #   3.1 先做归一化，帮助模型训练稳定
        h = self.temporal_norm(x)
        #   3.2 调整维度，把 [nt, S] 两个维度分别分开，然后交换顺序，目的是同一空间位置上跨时间做注意力
        #       (B, nt×S, d) → (B, nt, S, d) → (B, S, nt, d) → (B*S, nt, d)
        h = h.reshape(B, nt, S, -1).permute(0, 2, 1, 3).reshape(B * S, nt, -1)
        #   3.3 在时间维做多头注意力（同一空间位置的 patch，跨所有时间帧互相交流信息）
        h = self.temporal_attn(h, h, h, mask=None)
        #   3.4 还原回原始形状：(B*S, nt, d) → (B, S, nt, d) → (B, nt, S, d) → (B, nt×S, d)
        h = h.reshape(B, S, nt, -1).permute(0, 2, 1, 3).reshape(B, nt * S, -1)
        #   3.5 直接加回去（temporal attention 没有 alpha 残差缩放，初学者可以理解为和 spatial attention 类似）
        x = x + h

        # 步骤 4：交叉注意力（cross attention），将文本特征注入到 patch 表示里
        if text_emb is not None:
            #   4.1 先对 x 做 LayerNorm，帮助数值稳定
            h = self.cross_attn_norm(x)
            #   4.2 用 patch 表示做 Query，文本特征做 Key/Value，做一次多头交叉注意力
            h = self.cross_attn(h, text_emb, text_emb, mask=None)
            #   4.3 把 cross-attention 输出特征加回 patch 表示（残差连接）
            x = x + h

        # 步骤 5：前馈网络（FFN）
        #   5.1 用第二组调制参数(gamma2, beta2)对 patch 表示再做一次调制
        h = self.ada_ln.modulate(x, gamma2, beta2)
        #   5.2 通过前馈神经网络提取和转换特征
        h = self.ffn(h)
        #   5.3 用 alpha2 缩放后与原 x 残差相加
        x = x + alpha2 * h

        return x


# ═══════════════════════════════════════════════════════════════════════════
# 5. Final Layer — 输出投影
# ═══════════════════════════════════════════════════════════════════════════
class FinalLayer(nn.Module):
    """
    DiT 的最终输出层。

    将 Transformer 的隐藏表示 (B, N, d_model) 投影回 patch 像素空间 (B, N, patch_dim)。
    使用 adaLN 做最后一次条件调制。

    patch_dim = C_latent × patch_t × patch_h × patch_w
    例如: 4 × 2 × 2 × 2 = 32

    参数:
        d_model   (int): 隐藏维度
        patch_dim (int): 每个 patch 展平后的像素维度
    """

    def __init__(self, d_model: int, patch_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),  # 只需 γ 和 β
        )
        self.linear = nn.Linear(d_model, patch_dim)

        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x:     (B, N, d_model)
        t_emb: (B, d_model)
        返回:  (B, N, patch_dim)
        """
        params = self.adaLN_modulation(t_emb)
        gamma, beta = params.chunk(2, dim=-1)  # 各 (B, d_model)
        x = gamma.unsqueeze(1) * self.norm(x) + beta.unsqueeze(1)
        x = self.linear(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# 6. DiT — 完整的 Diffusion Transformer
# ═══════════════════════════════════════════════════════════════════════════
class DiT(nn.Module):
    """
    完整的 Diffusion Transformer (DiT) 去噪网络。

    在扩散模型中，DiT 充当噪声预测器 ε_θ(x_t, t, text)：
      - 输入：加噪后的视频 latent x_t + 时间步 t + 文本条件
      - 输出：预测的噪声 ε
    # 问：t 的 shape 一般为 (B,)，那这个 B 代表什么？t 不就是一个整数吗？
    #
    # 答：这里的 B 指的是 batch size（批量大小）。
    # 在扩散模型的训练和推理中，通常会同时处理一批（B 个）样本。
    #
    # - t 的 shape 为 (B,) 时，表示每个样本可以有各自独立的 time step（扩散步）。
    #   例如，训练时常常为每个样本单独随机采样 t（比如 t = torch.randint(0, num_timesteps, (B,))）。
    #   这样就能让网络学习在不同步数下去噪，增强泛化性。
    #
    # - 如果 t 是一个整数（比如 t=500），也可以直接用同一个 t 处理所有样本，此时 shape 是 () 或 (1,)。
    #   这种用法见于推理/采样时，全 batch 用同一个 t 逐步去噪。
    #
    # 总结：
    #   - 训练时：建议 t.shape = (B,) —— 每个样本独立采样 time step。
    #   - 推理时：可以 t.shape = () 或 (B,) —— 所有样本用同一个 step，更高效。
    #
    # t 的含义与视频帧、patch 等空间信息无关，只是控制当前扩散/去噪的程度，取值范围通常为 [0, num_timesteps-1]。
    # 例如，DiT 的 forward 可以写为：
    #     def forward(self, x_t, t, text_tokens, ...):
    #         ...
    # 其中 t 是提前准备好的、和 batch 对应的时间步索引。
    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  x_t (B,4,24,136,240)   t (B,)      text_tokens (B,256,4096)          │
    │  (1080p latent)           │           (T5-XXL 编码)                    │
    │      │                    │               │                            │
    │      ▼                    ▼               ▼                            │
    │  VideoPatchEmbed      TimestepEmb      TextProj                       │
    │  (B, 97920, 1152)     (B, 1152)       (B, 256, 1152)                 │
    │      │                    │               │                            │
    │      + pos_embed          │               │                            │
    │  (1, 97920, 1152)        │               │                            │
    │      │                    │               │                            │
    │      ▼                    ▼               ▼                            │
    │   ┌─────────────────────────────────────────────────┐                 │
    │   │  DiTBlock/STDiTBlock × 28 (DiT-XL/2)            │                 │
    │   │                                                  │                 │
    │   │  use_st_attn=False (DiTBlock):                   │                 │
    │   │   ├─ adaLN-Zero                                  │                 │
    │   │   ├─ Full Self-Attention (97920², 16 heads)     │                 │
    │   │   ├─ Cross-Attention (Q: patches, KV: text)     │                 │
    │   │   └─ FFN                                         │                 │
    │   │                                                  │                 │
    │   │  use_st_attn=True (STDiTBlock, 默认):            │                 │
    │   │   ├─ adaLN-Zero                                  │                 │
    │   │   ├─ Spatial Attn (8160², 同帧内)               │                 │
    │   │   ├─ Temporal Attn (12², 跨帧)                  │                 │
    │   │   ├─ Cross-Attention (Q: patches, KV: text)     │                 │
    │   │   └─ FFN                                         │                 │
    │   │                                                  │                 │
    │   │  时空分离节省 ~12× 注意力计算量！                │                 │
    │   └─────────────────────────────────────────────────┘                 │
    │      │                                                                 │
    │      ▼                                                                 │
    │   FinalLayer → (B, 97920, 32)                                         │
    │      │                                                                 │
    │      ▼                                                                 │
    │   Unpatchify → (B, 4, 24, 136, 240)  = 预测噪声 ε                     │
    │      │                                                                 │
    │      ▼  (推理完成后)                                                   │
    │   VAE Decode → (B, 3, 96, 1088, 1920)  = 原始像素视频                  │
    │                                                                        │
    │   模型参数:  ~675M (不含 VAE 和 T5)                                    │
    │   显存需求:  推理约 40~80 GB (取决于精度和优化)                          │
    │   训练成本:  数百~数千 GPU·天 (1080p 视频数据)                          │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘

    参数:
        cfg (DiTConfig): 配置
    """

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg

        # --- 3D Patch Embedding ---
        self.patch_embed = VideoPatchEmbedding(cfg)

        # --- 可学习位置编码（覆盖所有 3D patch 位置）---
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.num_patches, cfg.d_model))

        # --- 时间步编码 ---
        self.time_embed = TimestepEmbedding(cfg.d_model)

        # --- 文本条件投影（把文本编码器维度对齐到 d_model）---
        self.text_proj = nn.Linear(cfg.text_embed_dim, cfg.d_model)

        # --- DiT Blocks ---
        # 根据配置选择全局注意力 (DiTBlock) 或时空分离注意力 (STDiTBlock)
        block_cls = STDiTBlock if cfg.use_st_attn else DiTBlock
        self.blocks = nn.ModuleList(
            [
                block_cls(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                for _ in range(cfg.n_layers)
            ]
        )

        # --- 最终输出层 ---
        self.final_layer = FinalLayer(cfg.d_model, cfg.patch_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 patch 序列还原为视频张量。

        这是 patch embedding 的逆操作：
          (B, N, patch_dim) → (B, C, T, H, W)

        具体地：
          (B, 2048, 32) → reshape → (B, 8, 16, 16, 4, 2, 2, 2) → permute → (B, 4, 16, 32, 32)
        """
        cfg = self.cfg
        nt = cfg.num_patches_t  # 8
        nh = cfg.num_patches_h  # 16
        nw = cfg.num_patches_w  # 16
        pt = cfg.patch_size_t  # 2
        ph = cfg.patch_size_h  # 2
        pw = cfg.patch_size_w  # 2
        c = cfg.in_channels  # 4

        # (B, nt*nh*nw, c*pt*ph*pw) → (B, nt, nh, nw, c, pt, ph, pw)
        x = x.reshape(-1, nt, nh, nw, c, pt, ph, pw)
        # 调整维度顺序: → (B, c, nt, pt, nh, ph, nw, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
        # 合并 patch 维度: → (B, c, nt*pt, nh*ph, nw*pw) = (B, C, T, H, W)
        x = x.reshape(-1, c, nt * pt, nh * ph, nw * pw)
        return x

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        DiT 前向传播：预测噪声。

        参数:
            x_t:      (B, C, T, H, W)          — 加噪后的视频 latent
            t:        (B,)                      — 扩散时间步
            text_emb: (B, text_len, text_dim)   — 文本编码（可选）

        返回:
            noise_pred: (B, C, T, H, W) — 预测的噪声 ε_θ，形状与输入相同
        """
        # 1) Patch embedding
        #图像编码之后的值是(B, 97920, 1152)
        x = self.patch_embed(x_t)  # (B, N, d_model)
        #x的值是(B, 97920, 1152)
        #pos_embed的值是(1, 97920, 1152)
        #x + pos_embed的值是(B, 97920, 1152)
        x = x + self.pos_embed  # + 位置编码
        #x的值是(B, 97920, 1152)

        # 2) 时间步编码
        # t现在是什么不说形状 就是说输入
        # 配置里这个t是24
        #t的值是999到0的列表
        #t_emb的值是(B, d_model)
        t_emb = self.time_embed(t)  # (B, d_model)

        # 3) 文本条件投影（如果有）
        text_cond = None
        if text_emb is not None:
            text_cond = self.text_proj(text_emb)  # (B, text_len, d_model)

        # 4) 通过 N 层 DiT Block
        #    STDiTBlock 需要知道 3D 网格尺寸来做 reshape
        st_kwargs = {}
        if self.cfg.use_st_attn:
            st_kwargs = dict(
                nt=self.cfg.num_patches_t,   # nt=12 (时间维 patch 数)
                nh=self.cfg.num_patches_h,   # nh=68 (高度 patch 数)
                nw=self.cfg.num_patches_w,   # nw=120 (宽度 patch 数)
            )
        for block in self.blocks:
            x = block(x, t_emb, text_cond, **st_kwargs)

        # 5) 最终层：投影回 patch 像素空间
        x = self.final_layer(x, t_emb)  # (B, N, patch_dim)

        # 6) Unpatchify: (B, N, patch_dim) → (B, C, T, H, W)
        noise_pred = self.unpatchify(x)

        return noise_pred


# ═══════════════════════════════════════════════════════════════════════════
# 7. DDPM Scheduler — 噪声调度（简化版）
# ═══════════════════════════════════════════════════════════════════════════
class DDPMScheduler:
    """
    简化版 DDPM 噪声调度器。

    负责：
      - 定义噪声 schedule（β_t 序列）
      - 前向加噪 q(x_t | x_0)
      - 反向去噪 p(x_{t-1} | x_t)

    === 噪声 Schedule ===

    β_t 从 β_start 线性增长到 β_end，控制每步加多少噪声：
      β_1=0.0001, β_2=0.0002, ..., β_T=0.02

    由 β 可以推导出：
      α_t = 1 - β_t                    — 每步保留信号的比例
      ᾱ_t = α_1 · α_2 · ... · α_t     — 累积保留比例（越晚越小）

    ┌──────────────────────────────────────────────────────────────┐
    │  t:      0    100    200    500    800    999                │
    │  ᾱ_t:   1.0  0.98   0.95   0.70   0.20   0.01              │
    │  噪声:   无    微弱   轻微   明显   很重   接近纯噪声        │
    └──────────────────────────────────────────────────────────────┘

    参数:
        num_timesteps (int):  扩散总步数 T（默认 1000）
        beta_start (float):   β 起始值
        beta_end   (float):   β 终止值
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.num_timesteps = num_timesteps

        # 线性 β schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

        # α_t = 1 - β_t
        self.alphas = 1.0 - self.betas

        # ᾱ_t = ∏_{s=1}^{t} α_s （累积乘积）
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # ᾱ_{t-1}，在 t=0 时定义为 1
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向加噪: q(x_t | x_0) = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε

        参数:
            x_0:   (B, C, T, H, W) — 原始干净数据
            noise: (B, C, T, H, W) — 标准正态噪声 ε ~ N(0,I)
            t:     (B,)            — 时间步

        返回:
            x_t:   (B, C, T, H, W) — 加噪后的数据
        """
        device = x_0.device
        sqrt_alpha_cumprod = self.alphas_cumprod[t].sqrt().to(device)
        sqrt_one_minus = (1.0 - self.alphas_cumprod[t]).sqrt().to(device)

        # 扩展维度以匹配 (B, C, T, H, W)
        while sqrt_alpha_cumprod.dim() < x_0.dim():
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        return sqrt_alpha_cumprod * x_0 + sqrt_one_minus * noise

    @torch.no_grad()
    def denoise_step(
        self,
        model: DiT,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        反向去噪一步: p_θ(x_{t-1} | x_t)

        公式:
          x_{t-1} = (1/√α_t) · (x_t - (β_t/√(1-ᾱ_t)) · ε_θ(x_t, t)) + σ_t · z

          其中 σ_t = √β_t，z ~ N(0,I)（t > 0 时加噪声，t = 0 时不加）

        参数:
            model:    DiT 模型
            x_t:      (B, C, T, H, W) — 当前时间步的噪声数据
            t:        (B,)            — 当前时间步（所有 batch 相同）
            text_emb: 文本条件

        返回:
            x_{t-1}: (B, C, T, H, W) — 去噪一步后的数据
        """
        device = x_t.device
        t_val = t[0].item()  # 假设同一 batch 时间步相同

        # 1) 模型预测噪声
        noise_pred = model(x_t, t, text_emb)

        # 2) 取出当前时间步对应的参数
        beta_t = self.betas[t_val].to(device)
        alpha_t = self.alphas[t_val].to(device)
        alpha_bar_t = self.alphas_cumprod[t_val].to(device)

        # 3) 计算去噪后的均值
        #    x_{t-1}_mean = (1/√α_t) · (x_t - β_t/√(1-ᾱ_t) · ε_θ)
        coeff_eps = beta_t / (1.0 - alpha_bar_t).sqrt()
        mean = (x_t - coeff_eps * noise_pred) / alpha_t.sqrt()

        # 4) 加入随机噪声（t > 0 时）
        if t_val > 0:
            sigma = beta_t.sqrt()
            z = torch.randn_like(x_t)
            return mean + sigma * z
        else:
            return mean


# ═══════════════════════════════════════════════════════════════════════════
# 8. Text-to-Video Pipeline — 文生视频推理管线
# ═══════════════════════════════════════════════════════════════════════════
class SimpleTextEncoder(nn.Module):
    """
    简化版文本编码器（学习用）。

    真实系统中这里会用预训练模型：
      ┌──────────────────────────────────────────────────────────────────┐
      │  编码器         │ dim   │ max_len │ 使用场景                    │
      │ ─────────────── │ ───── │ ─────── │ ────────────────────────── │
      │  CLIP ViT-L/14 │  768  │   77    │ SD 1.x / 2.x              │
      │  OpenCLIP bigG  │ 1280  │   77    │ SDXL                      │
      │  T5-XXL        │ 4096  │  256    │ Imagen, 视频模型 ← 本配置  │
      │  CLIP + T5 双  │ 768   │   77    │ SD3, Flux                  │
      │                │+4096  │ +256    │                            │
      └──────────────────────────────────────────────────────────────────┘

    这里用一个简单的 Embedding + Transformer 层来模拟 T5-XXL 的输出维度，
    让整个管线可以跑通。

    参数:
        vocab_size (int):  词表大小
        d_model    (int):  输出嵌入维度（T5-XXL = 4096）
        max_len    (int):  最大序列长度（256 for video descriptions）
        n_layers   (int):  Transformer 层数（简化版用 2 层，T5-XXL 实际有 24 层）
        n_heads    (int):  注意力头数
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 4096,
        max_len: int = 256,
        n_layers: int = 2,
        n_heads: int = 16,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, L) — 文本 token id
        返回: (B, L, d_model) — 文本编码
        """
        x = self.token_embed(input_ids) + self.pos_embed[:, : input_ids.size(1)]
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TextToVideoPipeline:
    """
    文生视频推理管线：把所有组件串起来。

    === 推理流程 ===

    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │  Step 1: 文本编码                                                    │
    │    "一只猫在弹钢琴" → T5-XXL → (1, 256, 4096)                      │
    │                                                                      │
    │  Step 2: 初始化纯噪声（latent space，不是像素空间！）               │
    │    z_T ~ N(0, I)  形状: (1, 4, 24, 136, 240)                        │
    │    (对应 1920×1080 视频，96 帧，经 VAE 压缩后)                      │
    │                                                                      │
    │  Step 3: 迭代去噪                                                    │
    │    全量: T=1000 步（太慢）                                            │
    │    加速: DDIM/DPM-Solver 减少到 20~50 步                              │
    │                                                                      │
    │    for t = T-1, T-2, ..., 1, 0:                                     │
    │      ε_cond   = DiT(z_t, t, text_emb)  ← 有文本条件                │
    │      ε_uncond = DiT(z_t, t, None)       ← 无条件                    │
    │      ε = ε_uncond + 7.5·(ε_cond - ε_uncond)  ← CFG 引导            │
    │      z_{t-1} = denoise(z_t, ε, t)                                   │
    │                                                                      │
    │    注意: CFG 每步需要跑 DiT 两次！1080p 每步处理 97920 个 patch     │
    │    50 步 × 2 次/步 = 100 次前向传播，工业级需要多卡并行              │
    │                                                                      │
    │  Step 4: VAE 解码 (本文件不实现)                                    │
    │    z_0 (1,4,24,136,240) → VAE Decoder → (1, 3, 96, 1088, 1920)     │
    │    → crop 回 1920×1080，得到 96 帧 Full HD 视频                     │
    │                                                                      │
    │  ┌────────┐   ┌────────┐   ┌────────┐       ┌────────┐             │
    │  │ z_1000 │ → │ z_980  │ → │ z_960  │ → ··· │  z_0   │             │
    │  │(纯噪声)│   │        │   │        │       │(干净)  │             │
    │  └────────┘   └────────┘   └────────┘       └────────┘             │
    │   4×24×136×240    逐步去噪 (50 步)      4×24×136×240                │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

    Classifier-Free Guidance (CFG):
      实际推理中通常使用 CFG 来增强文本控制力：
        ε_guided = ε_uncond + guidance_scale · (ε_cond - ε_uncond)
      即同时跑一次"有文本条件"和一次"无条件"，然后做插值。
      guidance_scale 越大，生成结果越忠实于文本（但多样性下降）。

    参数:
        dit       (DiT):             去噪网络
        scheduler (DDPMScheduler):   噪声调度器
        text_encoder (nn.Module):    文本编码器
        cfg       (DiTConfig):       配置
    """

    def __init__(
        self,
        dit: DiT,
        scheduler: DDPMScheduler,
        text_encoder: nn.Module,
        cfg: DiTConfig,
    ):
        self.dit = dit
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.cfg = cfg

    @torch.no_grad()
    def generate(
        self,
        text_ids: torch.Tensor,
        num_inference_steps: int | None = None,
        guidance_scale: float = 7.5,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """
        文生视频推理。

        参数:
            text_ids:            (B, L) — 文本 token id
            num_inference_steps: 去噪步数（None 则用全部 T 步）
            guidance_scale:      CFG 引导强度（1.0 = 不引导，7.5 = 常用值）
            device:              推理设备

        返回:
            z_0: (B, C, T, H, W) — 去噪后的视频 latent
                 （实际使用时还需要经过 VAE Decoder 才能得到像素空间的视频）

        使用示例:
            cfg = DiTConfig()  # 1080p 配置
            dit = DiT(cfg)     # ~675M 参数
            scheduler = DDPMScheduler(cfg.num_timesteps)
            text_enc = SimpleTextEncoder(d_model=cfg.text_embed_dim)
            pipe = TextToVideoPipeline(dit, scheduler, text_enc, cfg)

            # 模拟文本输入 "一只猫在弹钢琴"
            text_ids = torch.randint(0, 32000, (1, 20))
            video_latent = pipe.generate(text_ids, num_inference_steps=50)

            # video_latent.shape = (1, 4, 24, 136, 240) — 视频 latent
            # → 经 VAE Decoder 得到 (1, 3, 96, 1088, 1920) 的 1080p 视频
            # → crop 到 (1, 3, 96, 1080, 1920) → 保存为 4 秒 24fps 视频
            print(f"输出 latent: {video_latent.shape}")
            print(f"对应视频: {cfg.original_video_shape}")
        """
        self.dit.eval()
        self.text_encoder.eval()
        B = text_ids.size(0)

        # --- 1) 文本编码 ---
        text_emb = self.text_encoder(text_ids.to(device))  # (B, L, text_dim)

        # --- 2) 初始化纯高斯噪声 ---
        z = torch.randn(
            B,
            self.cfg.in_channels,
            self.cfg.num_frames,
            self.cfg.latent_h,
            self.cfg.latent_w,
            device=device,
        )

        # --- 3) 确定去噪时间步序列 ---
        total_steps = self.scheduler.num_timesteps
        if num_inference_steps is not None and num_inference_steps < total_steps:
            # 均匀跳步（简化版 DDIM 思路，减少采样步数）
            step_ratio = total_steps // num_inference_steps
            timesteps = list(range(total_steps - 1, -1, -step_ratio))[
                :num_inference_steps
            ]
        else:
            timesteps = list(range(total_steps - 1, -1, -1))

        # --- 4) 迭代去噪 ---
        #timesteps的值是是999到0的列表
        for t_val in timesteps:
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)

            if guidance_scale > 1.0:
                # === Classifier-Free Guidance (CFG) ===
                # 同时预测 有条件 和 无条件 的噪声，然后做线性组合
                noise_cond = self.dit(z, t_tensor, text_emb)
                noise_uncond = self.dit(z, t_tensor, None)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

                # 手动执行去噪步骤（因为 noise_pred 是合成的，不能直接用 scheduler.denoise_step）
                beta_t = self.scheduler.betas[t_val].to(device)
                alpha_t = self.scheduler.alphas[t_val].to(device)
                alpha_bar_t = self.scheduler.alphas_cumprod[t_val].to(device)
                coeff_eps = beta_t / (1.0 - alpha_bar_t).sqrt()
                mean = (z - coeff_eps * noise_pred) / alpha_t.sqrt()

                if t_val > 0:
                    z = mean + beta_t.sqrt() * torch.randn_like(z)
                else:
                    z = mean
            else:
                # 无 CFG，直接调用 scheduler
                z = self.scheduler.denoise_step(self.dit, z, t_tensor, text_emb)

        return z


# ═══════════════════════════════════════════════════════════════════════════
# 快捷构建函数
# ═══════════════════════════════════════════════════════════════════════════
def build_text_to_video_pipeline(
    dit_cfg: DiTConfig | None = None,
) -> TextToVideoPipeline:
    """
    快捷构建文生视频管线（学习/测试用）。

    使用示例（1080p 默认配置，仅供学习，实际无法在单卡运行）:
        pipe = build_text_to_video_pipeline()
        # DiT-XL/2: ~675M 参数, 序列长度 97920

        text_ids = torch.randint(0, 32000, (1, 20))  # 模拟 "一只猫在弹钢琴"
        # ⚠️ 1080p 配置的 pos_embed 有 97920×1152 ≈ 1.1 亿参数
        # 仅 pos_embed 就需要 ~430 MB 显存（fp32），实际无法在普通 GPU 运行
        # 如果要实际测试，请用小配置:
        #   small_cfg = DiTConfig(
        #       num_frames=4, latent_h=8, latent_w=8,
        #       n_layers=2, d_model=128, n_heads=4, d_ff=256, text_embed_dim=64,
        #   )
        #   pipe = build_text_to_video_pipeline(small_cfg)

        video_latent = pipe.generate(text_ids, num_inference_steps=50)
        print(video_latent.shape)
        # → (1, 4, 24, 136, 240) — 1080p 视频 latent
        # → VAE Decoder → (1, 3, 96, 1088, 1920) → crop → 1920×1080 视频

    训练伪代码:
        cfg = DiTConfig()
        dit = DiT(cfg)                                # ~675M params
        scheduler = DDPMScheduler(cfg.num_timesteps)
        optimizer = torch.optim.AdamW(dit.parameters(), lr=1e-4)

        for x_0, text_emb in dataloader:
            # x_0: VAE 编码后的干净视频 latent (B, 4, 24, 136, 240)
            # text_emb: T5-XXL 编码的文本 (B, 256, 4096)
            B = x_0.size(0)
            t = torch.randint(0, cfg.num_timesteps, (B,))
            noise = torch.randn_like(x_0)
            x_t = scheduler.add_noise(x_0, noise, t)  # 加噪

            noise_pred = dit(x_t, t, text_emb)         # 预测噪声
            loss = F.mse_loss(noise_pred, noise)        # 简单 MSE loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 训练一个 1080p 的 DiT-XL 模型需要:
        #   - 数据: 数百万条视频 + 文本描述
        #   - 硬件: 64~256 张 A100/H100 GPU
        #   - 时间: 数周到数月
        #   - 成本: 数十万~数百万美元
    """
    if dit_cfg is None:
        dit_cfg = DiTConfig()

    dit = DiT(dit_cfg)
    scheduler = DDPMScheduler(num_timesteps=dit_cfg.num_timesteps)
    text_encoder = SimpleTextEncoder(d_model=dit_cfg.text_embed_dim)

    return TextToVideoPipeline(
        dit=dit,
        scheduler=scheduler,
        text_encoder=text_encoder,
        cfg=dit_cfg,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#
#  附录：商业视频生成产品的完整技术栈分析
#  （可灵 Kling / 即梦 Jimeng / Sora / Veo 等）
#
# ═══════════════════════════════════════════════════════════════════════════════
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   商业视频生成系统：从用户输入到最终视频的完整链路                            ║
║                                                                              ║
║   本节以 可灵(Kling) / 即梦(Jimeng) / Sora 等产品为参考，                    ║
║   分析一个商业级文生视频系统的完整技术栈。                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 一、端到端架构总览
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

用户在手机上输入 "一只金毛犬在海边奔跑，夕阳，电影感" → 10 秒 1080p 视频

整个系统远不止一个 DiT 模型，而是一条复杂的 pipeline：

┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  ① 用户输入                                                                │
│     "一只金毛犬在海边奔跑，夕阳，电影感"                                   │
│         │                                                                  │
│         ▼                                                                  │
│  ② 文本理解与增强（LLM 驱动）                                              │
│     ┌──────────────────────────────────────────────┐                       │
│     │ 用户写的 prompt 通常很短、很模糊。                │                       │
│     │ 系统先用 LLM（如 GPT-4/自研大模型）做 prompt 改写│                       │
│     │                                                  │                       │
│     │ 输入: "一只金毛犬在海边奔跑，夕阳，电影感"         │                       │
│     │                                                  │                       │
│     │ 改写: "A golden retriever running on a sandy      │                       │
│     │  beach at sunset, waves crashing in the           │                       │
│     │  background, cinematic 4K, warm golden hour       │                       │
│     │  lighting, shallow depth of field, slow motion,   │                       │
│     │  the dog's fur blowing in the wind, ocean         │                       │
│     │  spray catching the sunlight, anamorphic lens"    │                       │
│     │                                                  │                       │
│     │ 为什么要改写？                                    │                       │
│     │  - 视频模型训练时的文本标注通常是英文、很详细的    │                       │
│     │  - 短 prompt 缺少空间/时间/风格信息，生成质量差  │                       │
│     │  - LLM 补充：画面构图、光照、运动方式、风格等     │                       │
│     └──────────────────────────────────────────────┘                       │
│         │                                                                  │
│         │ 增强后的长 prompt                                                │
│         ▼                                                                  │
│  ③ 安全审核（必须！）                                                      │
│     ┌──────────────────────────────────────────────┐                       │
│     │ - 文本安全：过滤违规、暴力、色情等内容            │                       │
│     │ - 人物保护：检测是否试图生成真实公众人物          │                       │
│     │ - 版权检测：检查是否涉及受保护的 IP/品牌          │                       │
│     │ - 通常用分类模型 + 规则引擎 + 人工审核            │                       │
│     └──────────────────────────────────────────────┘                       │
│         │                                                                  │
│         ▼                                                                  │
│  ④ 文本编码（Text Encoder）                                                │
│     ┌──────────────────────────────────────────────┐                       │
│     │ 将文本转化为模型能理解的向量表示                  │                       │
│     │                                                  │                       │
│     │ 常见方案:                                        │                       │
│     │  a) T5-XXL (dim=4096) — 语义理解能力最强         │                       │
│     │  b) CLIP ViT-bigG (dim=1280) — 视觉-语言对齐好  │                       │
│     │  c) 双编码器: T5 + CLIP 同时使用                 │                       │
│     │     → T5 提供深层语义，CLIP 提供视觉对齐         │                       │
│     │                                                  │                       │
│     │ 可灵: 据推测使用 T5 系列                         │                       │
│     │ 即梦: 字节内部多语言编码器 + CLIP                 │                       │
│     │ Sora: 可能是 T5 + CLIP 双编码器 或 GPT-4V 内部    │                       │
│     │                                                  │                       │
│     │ 输出: text_emb (1, ~256, 4096)                   │                       │
│     └──────────────────────────────────────────────┘                       │
│         │                                                                  │
│         ▼                                                                  │
│  ⑤ 核心：视频生成（DiT 去噪）                                             │
│     ┌──────────────────────────────────────────────────────────────┐       │
│     │                                                              │       │
│     │  这就是本文件实现的 DiT 部分！但商业版本复杂得多：           │       │
│     │                                                              │       │
│     │  5a. 噪声初始化                                              │       │
│     │      z_T ~ N(0, I)  形状: (1, 16, latent_T, latent_H, W)   │       │
│     │                                                              │       │
│     │  5b. 分段生成（长视频时）                                    │       │
│     │      每段 ~4 秒，自回归滚动                                  │       │
│     │                                                              │       │
│     │  5c. DiT 去噪（每段 20~50 步）                               │       │
│     │      - 时空分离注意力（降低计算量）                          │       │
│     │      - Flash Attention 2/3（加速 + 省显存）                  │       │
│     │      - CFG (guidance_scale=7~9)                              │       │
│     │      - 可能用 Flow Matching 替代 DDPM                       │       │
│     │                                                              │       │
│     │  5d. 模型规模                                                │       │
│     │      可灵/即梦: 估计 3B~10B 参数                             │       │
│     │      Sora: 估计 3B~30B 参数                                  │       │
│     │                                                              │       │
│     │  输出: video_latent (1, 16, T, H, W) — 还在 latent 空间    │       │
│     │                                                              │       │
│     └──────────────────────────────────────────────────────────────┘       │
│         │                                                                  │
│         ▼                                                                  │
│  ⑥ VAE 解码（latent → 像素）                                              │
│     ┌──────────────────────────────────────────────┐                       │
│     │ 3D VAE Decoder 将 latent 还原为像素视频          │                       │
│     │                                                  │                       │
│     │ latent (1, 16, 24, 136, 240)                     │                       │
│     │   → VAE Decode                                   │                       │
│     │   → video (1, 3, 96, 1088, 1920) — 约 4 秒      │                       │
│     │                                                  │                       │
│     │ VAE 也是关键组件！                                │                       │
│     │  - 商业模型的 3D VAE 是专门训练的                 │                       │
│     │  - 既要压缩率高，又要重建质量好                   │                       │
│     │  - CogVideoX 用 4×4×4 的 3D VAE (压缩 256 倍)   │                       │
│     │  - 可灵/即梦 的 VAE 规格未公开                    │                       │
│     └──────────────────────────────────────────────┘                       │
│         │                                                                  │
│         ▼                                                                  │
│  ⑦ 后处理 Pipeline                                                        │
│     ┌──────────────────────────────────────────────┐                       │
│     │ a) 超分辨率（可选）                              │                       │
│     │    - 有些模型先生成 720p，再超分到 1080p/4K      │                       │
│     │    - 用专门的视频超分模型（如 Real-ESRGAN 变体） │                       │
│     │                                                  │                       │
│     │ b) 帧插值（可选）                                │                       │
│     │    - 生成 12fps → 插帧到 24fps                   │                       │
│     │    - 常用 RIFE / AMT 等光流插帧模型              │                       │
│     │                                                  │                       │
│     │ c) 色彩校正 / 风格统一                            │                       │
│     │    - 确保分段生成的各段色调一致                   │                       │
│     │                                                  │                       │
│     │ d) 视频编码                                      │                       │
│     │    - 编码为 H.264/H.265 MP4                      │                       │
│     │    - 码率控制、音频添加等                         │                       │
│     └──────────────────────────────────────────────┘                       │
│         │                                                                  │
│         ▼                                                                  │
│  ⑧ 输出安全审核                                                           │
│     ┌──────────────────────────────────────────────┐                       │
│     │ - 逐帧检测生成的视频是否包含违规内容              │                       │
│     │ - 人脸检测：是否意外生成了真实人物                │                       │
│     │ - 水印添加：可灵/即梦都会加隐形水印 + 可见水印   │                       │
│     │ - 元数据：写入 AI 生成标记（C2PA 标准）          │                       │
│     └──────────────────────────────────────────────┘                       │
│         │                                                                  │
│         ▼                                                                  │
│  ⑨ 返回用户：1080p MP4 视频                                               │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 二、各产品技术对比（基于公开论文 / 技术博客推测）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  可灵 Kling（快手）                                                           │
│  ─────────────────                                                            │
│  公开论文: 无完整论文，但有技术博客和专利                                      │
│  核心架构: 3D VAE + DiT (时空分离注意力)                                      │
│  文本编码: T5 系列（多语言版本）                                              │
│  视频时长: 最长 2 分钟（分段自回归，约 5s/段）                                │
│  分辨率:   720p / 1080p                                                       │
│  特色功能: 运动笔刷（控制物体运动轨迹）                                       │
│  推理硬件: 推测 8×H100 / 单次生成                                             │
│  推理时间: ~3 分钟 / 10 秒视频（用户感知）                                    │
│                                                                               │
│  技术细节推测:                                                                │
│  - 3D VAE: 自研，空间 8× + 时间 4× 压缩                                     │
│  - DiT: ~5B 参数，可能 40+ 层                                                │
│  - 注意力: spatial-temporal 分离                                              │
│    先在同一帧内做空间注意力，再在同一位置跨帧做时间注意力                      │
│  - 训练数据: 快手 ~数亿条短视频，筛选后约数千万条高质量视频                    │
│  - 文本标注: 用 LLM 对视频做自动 caption（视频描述生成）                       │
│  - 长视频: 前一段最后 N 帧 concat 到下一段输入，做条件去噪                    │
│                                                                               │
│  ────────────────────────────────────────────────                             │
│                                                                               │
│  即梦 Jimeng（字节跳动 / 豆包）                                               │
│  ──────────────────────────────                                               │
│  公开论文: 部分技术来自 Seaweed (2025)                                        │
│  核心架构: DiT (可能是双流架构 / MM-DiT 风格)                                 │
│  文本编码: 多语言 CLIP + T5 (或字节自研 LLM Encoder)                          │
│  视频时长: 最长 ~10 秒（高质量模式）                                          │
│  分辨率:   720p / 1080p                                                       │
│  特色功能: 图生视频 / 角色一致性 / 多风格                                     │
│  推理时间: ~1~2 分钟 / 5 秒视频                                              │
│                                                                               │
│  技术细节推测:                                                                │
│  - 基于 MM-DiT (Multi-Modal DiT) 或类似 SD3 的双流架构：                     │
│    文本流和视频流各自有独立的 Transformer 参数，                               │
│    通过 Joint Attention 交互                                                  │
│  - 训练数据: 抖音海量视频（数十亿条，筛选后数千万~数亿）                      │
│  - 数据优势: 字节拥有全球最大的短视频数据集（抖音 + TikTok）                  │
│  - 可能使用 Rectified Flow（而非 DDPM）作为扩散框架                           │
│                                                                               │
│  ────────────────────────────────────────────────                             │
│                                                                               │
│  Sora（OpenAI）                                                               │
│  ─────────────                                                                │
│  公开论文: 技术报告 (2024.02)，无完整论文                                     │
│  核心架构: Spatial-Temporal DiT + 3D VAE（"patches" in spacetime）            │
│  文本编码: 推测 GPT-4 内部表征 或 T5/CLIP 组合                                │
│  视频时长: 最长 ~60 秒                                                        │
│  分辨率:   720p / 1080p                                                       │
│  特色功能: 可变分辨率/时长/宽高比                                             │
│                                                                               │
│  关键创新:                                                                    │
│  - "Spacetime patches": 视频是时空体的 patch 序列                            │
│  - 灵活尺寸: 不固定分辨率，训练时 bucket 不同尺寸，推理时可变                │
│  - 训练规模: 推测用了 ~数千万条高质量视频-文本对                              │
│  - 参数规模: 推测 ~3B+，可能更大                                             │
│  - 超大规模训练: 数千张 H100，训练数月                                       │
│                                                                               │
│  ────────────────────────────────────────────────                             │
│                                                                               │
│  HunyuanVideo（腾讯混元）                                                     │
│  ──────────────────────                                                       │
│  公开论文: HunyuanVideo (2024.12)，已开源                                     │
│  核心架构: "Dual-Stream DiT" (MMDIT) — 文本流 + 视频流                       │
│  文本编码: MLLM（多语言大语言模型）+ CLIP                                     │
│  视频时长: ~5 秒                                                              │
│  分辨率:   720p（开源版）                                                     │
│  模型大小: 13B 参数（已开源最大的视频 DiT）                                   │
│                                                                               │
│  架构特点:                                                                    │
│  - Dual-Stream: 文本和视频各有独立的 Transformer 层                           │
│    ┌─────────────────────────────────────────────┐                            │
│    │  视频 tokens ──→ Video Self-Attn ──┐         │                            │
│    │                                     ├→ Joint │                            │
│    │  文本 tokens ──→ Text Self-Attn ───┘  Attn   │                            │
│    │                      ↓                       │                            │
│    │               各自 FFN                        │                            │
│    └─────────────────────────────────────────────┘                            │
│  - 用 MLLM 替代 T5 做文本编码，多语言能力更强                                │
│  - 使用 3D VAE (空间 8×, 时间 4×, 16 通道)                                   │
│  - 使用 "Full Attention"（不做时空分离），靠堆算力                            │
│                                                                               │
│  ────────────────────────────────────────────────                             │
│                                                                               │
│  CogVideoX（智谱 AI / 清华）                                                  │
│  ────────────────────────                                                     │
│  公开论文: CogVideoX (2024)，已开源                                           │
│  核心架构: 3D VAE (4×4×4) + Expert Transformer                               │
│  文本编码: T5-XXL                                                             │
│  视频时长: ~6 秒                                                              │
│  分辨率:   720p（开源版）                                                     │
│  模型大小: 2B / 5B 参数                                                       │
│                                                                               │
│  架构特点:                                                                    │
│  - 3D VAE 压缩非常激进: 4×4×4 = 空间4× + 时间4×                             │
│    （对比其他模型空间 8×，这里只压了 4 倍）                                   │
│  - 但通道数更多 (16 通道 vs 常见的 4 通道)                                    │
│  - "Expert Transformer": 部分层用 3D full attention，部分层用分离             │
│  - 已开源，可以直接看代码学习                                                 │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 三、训练数据：商业模型的核心壁垒
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

模型架构是公开的（论文都有），但数据才是商业产品的最大壁垒。

┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  数据处理流水线:                                                              │
│                                                                               │
│  原始视频池                                                                   │
│  (抖音/快手/YouTube 等平台数十亿条视频)                                       │
│      │                                                                        │
│      ▼                                                                        │
│  ① 初筛（规则过滤）                                                          │
│      - 去除 < 720p、< 3 秒、模糊/黑屏                                       │
│      - 去除有大面积文字覆盖/水印的视频                                        │
│      - 去除静态画面（幻灯片/PPT 录屏）                                       │
│      - 去除违规内容                                                           │
│      │  → 留下约 10%~20%                                                     │
│      ▼                                                                        │
│  ② 美学评分                                                                  │
│      - 用图像美学评分模型（如 LAION-Aesthetics）逐帧打分                     │
│      - 只保留高美学分数的视频                                                 │
│      - 有些团队训练专门的视频美学模型                                         │
│      │  → 留下约 5%~10%                                                      │
│      ▼                                                                        │
│  ③ 运动质量评分                                                              │
│      - 检测运动是否流畅、自然                                                 │
│      - 过滤抖动、快切、转场过多的视频                                         │
│      - 计算光流评分，保留运动丰富但不混乱的                                   │
│      │  → 留下约 50%                                                         │
│      ▼                                                                        │
│  ④ 场景切割                                                                  │
│      - 用场景检测模型（TransNetV2 等）把长视频切成单场景片段                  │
│      - 每个片段 3~10 秒，包含一个连贯的场景                                  │
│      │                                                                        │
│      ▼                                                                        │
│  ⑤ 文本标注（Captioning）— 最关键的一步！                                    │
│      - 用多模态大模型（GPT-4V / Qwen-VL / InternVL 等）                      │
│        自动为每个视频片段生成详细描述                                          │
│      - 描述包括：                                                             │
│        · 主体/物体："一只金色的拉布拉多犬"                                   │
│        · 动作/运动："在沙滩上奔跑，溅起水花"                                 │
│        · 场景/环境："海边沙滩，远处有椰子树"                                 │
│        · 光照/时间："夕阳金色光线，黄昏时分"                                 │
│        · 风格/质感："电影感，浅景深"                                         │
│        · 镜头运动："镜头从左向右缓慢跟拍"                                    │
│      - 这一步的质量直接决定模型的文本跟随能力！                               │
│      │                                                                        │
│      ▼                                                                        │
│  ⑥ 去重                                                                      │
│      - 用视觉特征做近似去重（避免同一视频不同剪辑版本重复）                   │
│      │                                                                        │
│      ▼                                                                        │
│  最终训练集: 约 数千万 ~ 数亿 条 (视频, 文本) 对                             │
│                                                                               │
│  各公司的数据优势:                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐      │
│  │  快手(可灵):  快手平台日活数亿，海量 UGC 短视频                     │      │
│  │  字节(即梦):  抖音+TikTok，全球最大短视频平台                       │      │
│  │  OpenAI(Sora): 推测通过授权合作获取大量专业视频                     │      │
│  │  腾讯(混元):   微信视频号 + 腾讯视频                                │      │
│  │  Google(Veo):  YouTube 全球最大视频平台                             │      │
│  └─────────────────────────────────────────────────────────────────────┘      │
│                                                                               │
│  关键洞察：                                                                   │
│  快手和字节在文生视频领域领先，不仅因为技术好，更因为它们                     │
│  天然拥有全球最大的高质量短视频数据集。Google 有 YouTube 数据，               │
│  所以 Veo 也非常强。模型架构大同小异，数据才是核心差异。                      │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 四、训练流程：多阶段训练策略
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

商业模型不会直接在最高分辨率上从头训练，而是分阶段递进：

┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  阶段 1: 图像预训练（最便宜，数据最多）                                       │
│  ──────────────────────────────────                                           │
│  - 先当成图像生成模型训练（忽略时间维度）                                     │
│  - 数据: ~数亿张高质量图像-文本对（LAION-5B 等）                             │
│  - 分辨率: 256×256 → 512×512                                                │
│  - 目的: 让模型先学会空间内容生成（物体、场景、构图）                         │
│  - 硬件: 64~128 GPU, ~1~2 周                                                │
│  - 成本: ~$50K~100K                                                          │
│                                                                               │
│  阶段 2: 低分辨率视频训练                                                     │
│  ──────────────────────────                                                   │
│  - 开启时间维度，在低分辨率视频上训练                                         │
│  - 数据: ~数千万条视频，256×256 / 16帧                                       │
│  - 目的: 学习运动模式（走路、跑步、流水、光影变化）                           │
│  - 硬件: 128~256 GPU, ~2~4 周                                               │
│  - 成本: ~$200K~500K                                                         │
│                                                                               │
│  阶段 3: 高分辨率视频微调                                                     │
│  ──────────────────────────                                                   │
│  - 提升到 720p/1080p                                                         │
│  - 数据: 数百万条高质量视频（严格筛选）                                      │
│  - 目的: 学习高清细节（毛发、水花、光线等）                                  │
│  - 硬件: 256~1024 GPU, ~2~4 周                                              │
│  - 成本: ~$500K~2M                                                           │
│                                                                               │
│  阶段 4: 长视频训练（可选）                                                   │
│  ──────────────────────────                                                   │
│  - 训练分段生成 + 段间衔接能力                                               │
│  - 可能需要额外的 "chunk conditioning" 机制                                  │
│  - 成本: ~$200K+                                                             │
│                                                                               │
│  阶段 5: 人类反馈对齐（RLHF / DPO for Video）                                │
│  ──────────────────────────                                                   │
│  - 类似 LLM 的 RLHF，用人类偏好来优化生成质量                               │
│  - 标注员对比两个生成视频，选择更好的那个                                     │
│  - 用 DPO / ReFL 等方法优化                                                  │
│  - 优化目标: 运动自然度、文本跟随度、美学质量                                │
│  - 这一步对用户感知质量提升非常大！                                          │
│                                                                               │
│  总成本估算:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐      │
│  │  小团队复现 (720p, ~5s):                                            │      │
│  │    - 开源模型基础上微调: ~$10K~50K                                  │      │
│  │    - 从头训练: ~$100K~500K                                          │      │
│  │                                                                     │      │
│  │  商业竞品级别 (1080p, ~30s+):                                       │      │
│  │    - 模型训练: $1M~10M                                              │      │
│  │    - 数据标注: $500K~2M                                             │      │
│  │    - 推理基础设施: $1M+/月 (按用户规模)                             │      │
│  │                                                                     │      │
│  │  顶级水平 (Sora/可灵/Veo):                                          │      │
│  │    - 研发总投入: $10M~100M+                                         │      │
│  │    - 团队: 50~200 人                                                │      │
│  └─────────────────────────────────────────────────────────────────────┘      │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 五、推理部署：用户按下"生成"之后发生了什么
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  用户点击"生成" → API 请求进入后端                                           │
│                                                                               │
│  1. 请求调度                                                                  │
│     - 进入排队系统（高峰期可能等 1~5 分钟）                                  │
│     - 分配到有空闲 GPU 的推理节点                                            │
│     - 通常每个请求独占 4~8 张 GPU（Tensor Parallel）                         │
│                                                                               │
│  2. 推理过程（以 10 秒 1080p 视频为例）                                      │
│     ┌─────────────────────────────────────────────┐                           │
│     │  环节              │ 耗时     │ 硬件          │                           │
│     │ ─────────────────  │ ──────── │ ──────────── │                           │
│     │  Prompt 改写 (LLM) │ ~1 秒   │ LLM 推理集群  │                           │
│     │  文本编码 (T5)      │ ~0.5 秒 │ 1×GPU        │                           │
│     │  DiT 去噪 (50步)   │ ~60~120s│ 4~8×H100     │                           │
│     │  VAE 解码           │ ~5~10 秒│ 1~2×GPU      │                           │
│     │  后处理(超分/插帧) │ ~5~10 秒│ 1×GPU        │                           │
│     │  安全审核           │ ~2 秒   │ CPU + GPU    │                           │
│     │ ───────────────────────────────────────────  │                           │
│     │  总计              │ ~90~150s │              │                           │
│     └─────────────────────────────────────────────┘                           │
│                                                                               │
│  3. 推理优化技巧                                                              │
│     - FP16/BF16 推理（半精度，速度翻倍，质量几乎无损）                       │
│     - Flash Attention 2/3（注意力计算加速 2~4 倍）                           │
│     - Tensor Parallelism（模型参数切分到多卡）                               │
│     - 投机采样 / 步数优化：50步减到20步(DPM-Solver++)                        │
│     - 缓存优化：KV-Cache 复用（对 CFG 的两次前向传播）                       │
│     - 模型量化：INT8/FP8 量化（减少显存、加速推理）                          │
│                                                                               │
│  4. 基础设施规模（估算）                                                      │
│     ┌─────────────────────────────────────────────────────────────┐           │
│     │  产品        │ 推测 GPU 集群规模       │ 日活/月活           │           │
│     │ ──────────── │ ────────────────────── │ ─────────────────  │           │
│     │  可灵        │ ~500~2000 张 H100      │ 数百万月活          │           │
│     │  即梦        │ ~1000~5000 张 H100     │ 数百万~千万月活     │           │
│     │  Sora        │ ~数千张 H100           │ 未公开              │           │
│     │                                                             │           │
│     │  每张 H100: ~$30K 购买 / ~$3/小时 租赁                     │           │
│     │  1000 张 H100 × $3/h × 24h × 30天 = $2.16M/月 (纯算力)    │           │
│     └─────────────────────────────────────────────────────────────┘           │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 六、本项目的 DiT 实现 vs 商业系统
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│                     本文件实现              商业系统                           │
│  ─────────────────  ──────────────────────  ──────────────────────────        │
│  DiT 架构           ✅ 完整 adaLN-Zero     ✅ + 时空分离 / 双流DiT          │
│  3D Patch Embed     ✅ Conv3d              ✅ 相同                           │
│  Cross-Attention    ✅ 文本条件注入        ✅ + 多条件(图像/音频等)          │
│  DDPM 采样          ✅ 基础版              ✅ + DPM-Solver++ / Flow Match     │
│  CFG                ✅ 基础版              ✅ + 动态 guidance / 多尺度       │
│  文本编码器         ⚠️  简化版 Embedding    ✅ T5-XXL / CLIP / MLLM          │
│  VAE                ❌ 不实现              ✅ 自研 3D VAE                     │
│  超分/插帧          ❌ 不实现              ✅ 专门的后处理模型               │
│  分块长视频         ❌ 不实现              ✅ 自回归分段 + overlap            │
│  时空分离注意力     ❌ 用全局注意力        ✅ 大幅降低计算量                 │
│  Flash Attention    ❌ 标准实现            ✅ FA2/FA3                         │
│  模型并行           ❌ 单卡                ✅ TP/SP/PP 多卡并行              │
│  安全审核           ❌ 无                  ✅ 必须有                          │
│  Prompt 改写        ❌ 无                  ✅ LLM 驱动                        │
│  RLHF/DPO           ❌ 无                  ✅ 人类反馈优化                    │
│  参数规模           ~675M (XL/2)           ~3B~13B                           │
│  训练数据           无                      数千万~数亿条视频                │
│                                                                               │
│  本文件的价值: 理解 DiT 的核心原理和数据流                                   │
│  商业差距: 主要在数据、工程优化、多阶段训练、后处理 pipeline                 │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
"""
