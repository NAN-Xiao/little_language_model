"""
Video MM-DiT (Multi-Modal Diffusion Transformer for Video) —— 学习用实现
========================================================================

本文件实现视频生成版的 MM-DiT，是当前（2024-2025）视频生成的主流架构。
核心创新: 将 MM-DiT 的双向联合注意力与时空分离注意力结合。

代表模型: HunyuanVideo(腾讯), Wan/万相(阿里), Step-Video(阶跃星辰),
         CogVideoX-1.5, Sora(后期版本)

本文件实现了:
  1. VideoMMDiTConfig         — 视频 MM-DiT 配置
  2. VideoPatchEmbedding3D    — 3D 视频 patch 切分
  3. STJointAttention         — 时空联合注意力（核心创新）
  4. VideoMMDiTBlock          — 双流视频 MM-DiT 块
  5. VideoSingleStreamSTBlock — 单流时空注意力块
  6. VideoMMDiT               — 完整的视频 MM-DiT 模型

=== 从图像 MM-DiT 到视频 MM-DiT 的演进 ===

  ┌────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │  第一代: DiT + Cross-Attn + ST分离 (ditblock.py)                      │
  │    文本注入: Cross-Attention (单向, 文本不更新)                        │
  │    时空处理: 空间注意力 → 时间注意力 (分离)                           │
  │    代表: Open-Sora, Latte, 早期 CogVideoX                            │
  │                                                                        │
  │  第二代: 图像 MM-DiT (mmdit.py)                                       │
  │    文本注入: Joint Attention (双向, 文本也更新) ← 创新                │
  │    时空处理: 无 (只做图像, 无时间维)                                  │
  │    代表: SD3, FLUX                                                    │
  │                                                                        │
  │  第三代: 视频 MM-DiT (本文件) ← 合并前两代的优点                     │
  │    文本注入: Joint Attention (双向) ← 来自 MM-DiT                     │
  │    时空处理: 空间联合注意力 + 时间注意力 ← 来自 ST-DiT               │
  │    代表: HunyuanVideo, Wan, Step-Video                                │
  │                                                                        │
  │  对比:                                                                 │
  │  ┌─────────────┬────────────────┬──────────────┬──────────────────┐   │
  │  │             │ ditblock.py    │ mmdit.py     │ 本文件           │   │
  │  │ 输入        │ 视频 (T,H,W)  │ 图像 (H,W)  │ 视频 (T,H,W)    │   │
  │  │ 文本注入    │ Cross-Attn     │ Joint-Attn   │ Joint-Attn      │   │
  │  │ 文本更新    │ ✗ 不更新       │ ✓ 双向更新   │ ✓ 双向更新      │   │
  │  │ 时空处理    │ ST分离         │ 无           │ ST分离          │   │
  │  │ 文本×时空   │ 分开做         │ 无时空       │ 空间时联合!     │   │
  │  └─────────────┴────────────────┴──────────────┴──────────────────┘   │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘

=== 视频 MM-DiT 的核心: 时空联合注意力 (STJointAttention) ===

  ┌────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │  关键洞察:                                                             │
  │    文本描述的是"空间内容" (什么东西在哪里)                            │
  │    文本不描述"时间位置" (第几帧是什么)                                │
  │    所以:                                                               │
  │      空间注意力: 文本应该参与 → Joint Attention (双向)                │
  │      时间注意力: 文本不参与 → 只有视频 patch 之间做                   │
  │                                                                        │
  │  Step 1 — Spatial Joint Attention (空间 + 文本联合):                  │
  │                                                                        │
  │    对每个时间帧 t:                                                    │
  │      video_patches_t: 当前帧的 nh×nw 个空间 patch                    │
  │      text_tokens:     所有文本 token                                  │
  │      拼接 → [video_t ; text] → Joint Attention → split               │
  │                                                                        │
  │    帧0: [视频patch_0 | 文本] ←→ Joint Attn ← 文本和每帧空间双向交互 │
  │    帧1: [视频patch_1 | 文本] ←→ Joint Attn ← 每帧都能影响文本表示   │
  │    帧2: [视频patch_2 | 文本] ←→ Joint Attn                           │
  │    ...                                                                 │
  │                                                                        │
  │    复杂度: O((S+L)²) × nt                                            │
  │    S=空间patch数, L=文本长度, nt=时间帧数                             │
  │                                                                        │
  │  Step 2 — Temporal Attention (纯时间, 无文本):                        │
  │                                                                        │
  │    对每个空间位置 (i,j):                                              │
  │      同一位置在 nt 帧的 patch 互看                                    │
  │                                                                        │
  │    位置(0,0): t0 ↔ t1 ↔ t2 ↔ ... ↔ t_{nt-1}                        │
  │    位置(0,1): t0 ↔ t1 ↔ t2 ↔ ... ↔ t_{nt-1}                        │
  │    ...                                                                 │
  │                                                                        │
  │    复杂度: O(nt²) × S — 通常很小                                     │
  │                                                                        │
  │  vs 全局做法 (不可行):                                                │
  │    把所有帧的 patch + 文本全部拼成一个超长序列                        │
  │    O((nt×S + L)²) — 1080p 视频会爆炸                                 │
  │                                                                        │
  │  vs ditblock.py 的做法 (第一代):                                      │
  │    空间注意力: 只有视频 patch (无文本)                                │
  │    然后 Cross-Attn: 单向查询文本                                     │
  │    → 文本不更新, 交互不够深                                          │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘

参考:
  - "HunyuanVideo: A Systematic Framework For Large Video Generation Model"
    (Kong et al., 2024) — 腾讯, MM-DiT 用于视频
  - "Wan: Open and Advanced Large-Scale Video Generative Models"
    (Alibaba, 2025) — 阿里万相
  - "Step-Video-T2V" (Ma et al., 2025) — 阶跃星辰


═══════════════════════════════════════════════════════════════════════════════
【图生视频完整维度流转模拟】以 1920×1080, 4秒@24fps 为例
═══════════════════════════════════════════════════════════════════════════════

场景: 输入一张猫的照片 + 文本"猫在草地上从左走到右"

  输入图片:        (3, 1080, 1920)        一张 RGB 照片
      ↓
  3D VAE 编码      (4, 1, 136, 240)       空间压缩8×, 时间压缩后只有1帧
      │                              ↑ 4通道latent, 136×240≈1080p/8
      │
      ▼ 构造图生视频的输入 latent
  z_t 构造:        (1, 4, 24, 136, 240)   ← 第0帧=图片latent
      │                                          第1~23帧=随机噪声
      │
  扩散时间步 t:    (1,)                   ← 标量, 如 0.7="去噪70%"
      │
  T5 文本编码:     (1, 256, 4096)         ← "猫在草地上从左走到右"
      │                                          256个token, 每个4096维
      │
      ▼ 进入 VideoMMDiT
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Step 1: Embedding                                                   │
  │   PatchEmbed3D: (1, 4, 24, 136, 240) → (1, 97920, 1536)            │
  │     nt=12, nh=68, nw=120 → 12×68×120=97920 patch                   │
  │   TimestepEmbed: (1,) → (1, 1536)                                  │
  │   TextProj: (1, 256, 4096) → (1, 256, 1536)                        │
  │   + 位置编码                                                        │
  │                                                                     │
  │ Step 2: 双流层 × 8                                                  │
  │   VideoMMDiTBlock(vid, txt, t_emb) × 8                             │
  │   每层: Spatial Joint Attn + Temporal Attn + FFN                   │
  │                                                                     │
  │ Step 3: 单流层 × 8                                                  │
  │   x = [vid; txt] → VideoSingleStreamSTBlock × 8                    │
  │                                                                     │
  │ Step 4: 输出                                                        │
  │   FinalLayer → (1, 97920, 48)  [patch_dim=2×2×2×4=48]              │
  │   unpatchify3d → (1, 4, 24, 136, 240)                              │
  └─────────────────────────────────────────────────────────────────────┘
      │
  noise_pred:       (1, 4, 24, 136, 240)   ← 预测的噪声 ε_θ
      │
      ▼ 扩散去噪 (重复 N 步)
  z_0 (去噪完成):   (1, 4, 24, 136, 240)   ← 干净的视频 latent
      ↓
  3D VAE 解码:      (1, 3, 96, 1088, 1920) ← 96帧@24fps = 4秒视频
      │                              ↑ 空间上采样8×: 136→1088, 240→1920
      │                                  时间上采样4×: 24→96
      ▼
  输出视频:         (1, 3, 96, 1080, 1920)  约 4 秒 1080p 视频

关键数字关系:
  原始视频:  96帧 × 1080 × 1920 = 3.97 亿像素
  VAE压缩后: 24帧 × 136 × 240  = 78.3 万像素 (压缩比 ≈ 500×)
  Patch后:   12×68×120 = 97,920 个 3D patch
  每 patch:  2×2×2×4 = 48 个数字 (patch_t=2, patch_h=2, patch_w=2, C=4)

图生视频 vs 文生视频的区别:
  图生: z_t[0] = 图片latent(已知), z_t[1:] = 噪声(需生成)
  文生: z_t[所有帧] = 噪声(全部需生成)
"""

# ═══════════════════════════════════════════════════════════════════════════
# 【无 batch 简化版理解】—— 先忘掉 batch，只看一个样本怎么走
# ═══════════════════════════════════════════════════════════════════════════
#
# 完整版上面已经有了，但维度太多容易晕。下面去掉 batch，只看一条数据。
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 一、单帧图像（你已经会的）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 原图:     (3, 1080, 1920)          RGB 照片
#     ↓
# VAE编码:  (4, 136, 240)            空间压缩 8x，变成 4 通道 latent
#     ↓
# 2D Patch: kernel=2x2，stride=2
#           136/2=68，240/2=120
#           -> 68x120 = 8,160 个 patch
#     ↓
# 每个 patch 有: 4通道 x 2高 x 2宽 = 16 个数字
#     ↓
# Linear(16 -> 1536)
#     ↓
# 输出:     (8160, 1536)             8,160 个 token，每个 1,536 维
#
# 理解: 一张图变成了一篇"8,160 个字的文章"，每个字 1,536 维。
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 二、加时间——视频就是"多帧叠在一起切"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 视频有 24 帧 latent（原始 96 帧经 VAE 时间压缩 4x）:
#
# 帧 0:  (4, 136, 240) --┐
# 帧 1:  (4, 136, 240) --┤
# ...                    ├── 在时间维度"叠起来"
# 帧 23: (4, 136, 240) --┘
#          ↓
# 叠起来: (4, 24, 136, 240)
#           ↑   ↑   ↑    ↑
#          通道 帧   高    宽
#
# 现在不用 2D Patch，改用 3D Patch：
#   kernel = 2(帧) x 2(高) x 2(宽) = 一个 3D 小方块
#   stride = 2x2x2
#
# 切法:
#   时间: 24/2 = 12 块  <- nt=12
#   高度: 136/2 = 68 块 <- nh=68
#   宽度: 240/2 = 120 块<- nw=120
#
# 总 patch 数 = 12 x 68 x 120 = 97,920 个
#
# 每个 3D patch 里面是什么？
#   2帧 x 2高 x 2宽 x 4通道 = 32 个数字
#
#   -> Linear(32 -> 1536)
#   -> 1 个 token
#
# 最终输出: (97920, 1536)
#           97,920 个 token，每个 1,536 维
#
# 理解: 一段视频也变成了一篇"文章"，只不过这次有 97,920 个"字"。
#       每个字包含了"2 帧时间 + 2x2 空间"的像素信息。
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 三、时间是怎么"消失"的？
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 关键点: Transformer 不吃"3D 网格"，它只吃"一维序列"。
#
# 所以 (12, 68, 120) 这个三维网格会被"展平"成一条线:
#
#   帧0的(0,0) -> 帧0的(0,1) -> ... -> 帧0的(67,119)
#   -> 帧1的(0,0) -> 帧1的(0,1) -> ... -> 帧11的(67,119)
#
#   共 97,920 个 token 排成一排: (97920, 1536)
#
# 模型不知道"这是第几帧第几行"，它只知道"这是第 42,000 个 token"。
# 时间关系靠两样东西重建:
#   1. 时间注意力 (Temporal Attention):
#      让"同一个空间位置、不同帧"的 token 互相看
#      例: 所有帧的 (0,0) 位置聚一起讨论"这里应该怎么动"
#
#   2. 位置编码 (vid_pos_embed):
#      给每个 token 发一个"身份证号"，模型训练后会学到
#      "身份证号相邻的 token 在时间和空间上也相邻"
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 四、和文字怎么拼在一起？
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 文字: "一只橘猫在草地上跑" -> T5 编码 -> (256, 4096)
#                                     ↓ TextProj
#                                     (256, 1536)
#
# 视频: 97,920 个 token，每个 1,536 维 -> (97920, 1536)
#
# 双流层（前 8 层）:
#   视频和文本"各干各的"，但每层都开一次"联席会"(Joint Attention)
#   让橘猫的 token 和视频里猫的位置对上号
#
# 单流层（后 8 层）:
#   把视频和文本拼成一条长序列:
#     [视频97920个 | 文字256个] -> (98176, 1536)
#   坐到同一张桌子前，共享参数，深度对齐
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 五、一句话总结
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# "视频变成 token" = "把时空切成 3D 豆腐块，每块压扁成 1536 维向量，
#                     排成一排交给 Transformer 处理"
#
# 时间维度不是在 Transformer 里"处理"的，而是在切 patch 阶段
# 就被"打包进每个 token"了。Transformer 只看到一排数字，
# 时间关系全靠注意力机制自己学出来。
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# 0. 配置
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class VideoMMDiTConfig:
    """
    视频 MM-DiT 配置。

    === 商业视频模型参数对比 ===

    ┌──────────────────────────────────────────────────────────────────────┐
    │  模型             │ d_model │ heads │ 层数     │ 参数量  │ 时空处理│
    │ ───────────────── │ ─────── │ ───── │ ──────── │ ─────── │ ─────── │
    │  HunyuanVideo     │  3072   │  24   │ 20双+40单│  13B    │ Full    │
    │  Wan-14B          │  5120   │  40   │ 40       │  14B    │ Full    │
    │  Step-Video       │  ~3072  │  24   │ ~48      │  30B    │ ST+MoE  │
    │  CogVideoX-1.5    │  ~3072  │  24   │ ~42      │  5B     │ ST      │
    │  本配置(学习用)    │  1536   │  24   │ 8双+8单  │  ~2B    │ ST      │
    └──────────────────────────────────────────────────────────────────────┘
    """

    # --- 视频 latent ---
    in_channels: int = 4  # VAE latent 通道数（3D VAE 输出 4 通道）
    latent_t: int = 24  # latent 时间帧数（96 帧 / VAE 4× 压缩）
    latent_h: int = 136  # latent 高度（1088 / VAE 8× 压缩）
    latent_w: int = 240  # latent 宽度（1920 / VAE 8× 压缩）

    # --- 3D Patch ---
    patch_t: int = 2  # 时间 patch 大小
    patch_h: int = 2  # 空间高度 patch 大小
    patch_w: int = 2  # 空间宽度 patch 大小

    # --- Transformer ---
    d_model: int = 1536  # 隐藏维度
    n_heads: int = 24  # 注意力头数（1536/24=64 per head）
    d_ff: int = 1536 * 4  # FFN 中间维度
    n_double_layers: int = 8  # 双流块层数（文本+视频各自参数）
    n_single_layers: int = 8  # 单流块层数（文本+视频共享参数）
    dropout: float = 0.0

    # --- 文本 ---
    text_max_len: int = 256  # 文本最大长度
    text_d_model: int = 4096  # T5-XXL 输出维度

    # --- 扩散 ---
    num_timesteps: int = 1000

    @property
    def nt(self) -> int:
        """时间 patch 数。"""
        return self.latent_t // self.patch_t  # 24/2 = 12

    @property
    def nh(self) -> int:
        """高度 patch 数。"""
        return self.latent_h // self.patch_h  # 136/2 = 68

    @property
    def nw(self) -> int:
        """宽度 patch 数。"""
        return self.latent_w // self.patch_w  # 240/2 = 120

    @property
    def num_spatial(self) -> int:
        """每帧的空间 patch 数。"""
        return self.nh * self.nw  # 68 × 120 = 8160

    @property
    def num_patches(self) -> int:
        """视频总 patch 数。"""
        return self.nt * self.num_spatial  # 12 × 8160 = 97,920

    @property
    def patch_dim(self) -> int:
        """每个 3D patch 展平后的维度。"""
        return self.in_channels * self.patch_t * self.patch_h * self.patch_w

    @property
    def st_joint_attn_info(self) -> str:
        """时空联合注意力的复杂度分析。"""
        S = self.num_spatial
        L = self.text_max_len
        nt = self.nt
        spatial_cost = (S + L) ** 2 * nt
        temporal_cost = nt**2 * S
        total_st = spatial_cost + temporal_cost
        full_cost = (nt * S + L) ** 2
        speedup = full_cost / total_st if total_st > 0 else 1.0
        return (
            f"空间联合: O(({S}+{L})²×{nt}) = {spatial_cost:.2e}\n"
            f"时间: O({nt}²×{S}) = {temporal_cost:.2e}\n"
            f"总计: {total_st:.2e}\n"
            f"全局: O(({nt}×{S}+{L})²) = {full_cost:.2e}\n"
            f"加速比: {speedup:.1f}×"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 1. 基础组件
# ═══════════════════════════════════════════════════════════════════════════


class RMSNorm(nn.Module):
    """RMS Normalization — 比 LayerNorm 更快，不减均值只除 RMS。"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class TimestepEmbedding(nn.Module):
    """时间步编码: t → 正弦编码 → MLP → t_emb。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def sinusoidal_encoding(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.d_model // 2
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / half_dim
        )
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        参数:
            t (torch.Tensor): 整数时间步，形状 (B,)，值域 [0, T-1]。
        返回:
            torch.Tensor: 时间嵌入向量，形状 (B, out_dim)。
        """
        return self.mlp(self.sinusoidal_encoding(t))


class Modulation(nn.Module):
    """条件调制层: 从 t_emb 生成 adaLN-Zero 的 (γ, β, α) 参数。"""

    def __init__(self, d_model: int, n_modulations: int = 6):
        super().__init__()
        self.n_modulations = n_modulations
        self.linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, n_modulations * d_model),
        )
        nn.init.zeros_(self.linear[1].weight)
        nn.init.zeros_(self.linear[1].bias)

    def forward(self, cond: torch.Tensor) -> list[torch.Tensor]:
        """
        参数:
            cond (torch.Tensor): 文本嵌入，形状 (B, L, d_model)。
        返回:
            list[torch.Tensor]: 参数列表，形状 (B, 1, d_model)。
        """
        params = self.linear(cond)
        return [p.unsqueeze(1) for p in params.chunk(self.n_modulations, dim=-1)]


# ═══════════════════════════════════════════════════════════════════════════
# 2. VideoPatchEmbedding3D — 3D 视频 patch 切分
# ═══════════════════════════════════════════════════════════════════════════
class VideoPatchEmbedding3D(nn.Module):
    """
    3D Patch Embedding — 把视频 latent 切成 3D patch 并投影。

    与 mmdit.py 的 PatchEmbedding2D 对比:
      2D: Conv2d(kernel=P×P)       — 图像 (B, C, H, W) → (B, N, d)
      3D: Conv3d(kernel=Pt×Ph×Pw)  — 视频 (B, C, T, H, W) → (B, N, d)

    (B, 4, 24, 136, 240) → Conv3d(kernel=2×2×2, stride=2×2×2)
    → (B, 1536, 12, 68, 120) → flatten → (B, 97920, 1536)
    """

    def __init__(self, cfg: VideoMMDiTConfig):
        super().__init__()
        self.proj = nn.Conv3d(
            cfg.in_channels,
            cfg.d_model,
            kernel_size=(cfg.patch_t, cfg.patch_h, cfg.patch_w),
            stride=(cfg.patch_t, cfg.patch_h, cfg.patch_w),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W) → (B, nt*nh*nw, d_model)"""
        # 3D卷积: (2, 4, 24, 136, 240) → (2, 1536, 12, 68, 120)
        #   kernel=2×2×2, stride=2×2×2
        #   时间: 24/2=12, 高度: 136/2=68, 宽度: 240/2=120
        #   通道: 4→1536 (从patch像素空间投影到Transformer特征空间)
        x = self.proj(x)
        # flatten: (2, 1536, 12, 68, 120) → (2, 1536, 97920)
        #   把时空网格(12×68×120=97920)展平成序列维
        # transpose: (2, 1536, 97920) → (2, 97920, 1536)
        #   Transformer期望(N, d)格式: 97920个token, 每个1536维
        return x.flatten(2).transpose(1, 2)


# ═══════════════════════════════════════════════════════════════════════════
# 3. STJointAttention — 时空联合注意力（本文件的核心创新）
# ═══════════════════════════════════════════════════════════════════════════
class STJointAttention(nn.Module):
    """
    时空联合注意力 (STJointAttention) —— 视频 MM-DiT 的"心脏"
    ================================================================

    这个模块回答了视频生成中最核心的问题:
      "文本怎么影响视频? 视频的时间连续性怎么保证?"

    它把两个前沿创新合二为一:
      1. 来自 SD3/FLUX 的 Joint Attention —— 文本和视频双向交互
      2. 来自 CogVideoX 的 ST 分离 —— 空间和时间分开处理

    ┌────────────────────────────────────────────────────────────────────────┐
    │  生活类比: 导演 + 分镜师 + 动画师 的合作流程                            │
    │                                                                          │
    │  文本 = 导演剧本: "一只橘猫在草地上从左走到右"                         │
    │  视频帧 = 分镜画面: 每帧是静态图                                       │
    │                                                                          │
    │  Step 1 — 空间联合注意力 (导演指导每帧画面):                           │
    │    导演(文本)和每帧的分镜师(空间patch)坐在一起讨论:                    │
    │      "这帧的猫应该在左边" → 文本告诉视频                               │
    │      "这帧的草地颜色太暗了" → 视频反馈给文本                           │
    │    → 每帧都独立和文本交互，但帧之间不交流                              │
    │                                                                          │
    │  Step 2 — 时间注意力 (动画师让画面动起来):                             │
    │    动画师把同一位置的像素跨帧连起来，确保运动平滑:                       │
    │      位置(0,0): t0的像素 → t1的像素 → t2的像素 → ...                   │
    │      导演不参与这一步（导演不管中间帧怎么过渡）                          │
    │    → 只处理视频，文本不参与                                            │
    │                                                                          │
    │  为什么分两步?                                                         │
    │    如果一步到位: 所有帧的patch + 文本全部拼一起                        │
    │    → 序列长度 = nt×S + L = 12×8160 + 256 = 98,176                      │
    │    → 注意力矩阵 = 98K × 98K ≈ 96亿个分数 → 显存爆炸！                  │
    │                                                                          │
    │    分步做法:                                                           │
    │      空间: O((S+L)²) × nt = (8160+256)² × 12 ≈ 8.5亿                  │
    │      时间: O(nt²) × S = 12² × 8160 ≈ 117万                            │
    │      总计 ≈ 8.5亿 << 96亿 ✓                                           │
    └────────────────────────────────────────────────────────────────────────┘

    ┌────────────────────────────────────────────────────────────────────────┐
    │  完整维度流转 (B=2, nt=12帧, nh=68, nw=120, L=256, d=1536)            │
    │                                                                          │
    │  输入:                                                                   │
    │    vid: (2, 12×8160, 1536) = (2, 97920, 1536)                         │
    │    txt: (2, 256, 1536)                                                │
    │                                                                          │
    │  ═══ Step 1: 空间联合注意力 ═══                                        │
    │                                                                          │
    │  ① 视频重组 —— 把 batch 和帧合并:                                      │
    │     vid: (2, 97920, 1536)                                             │
    │       → reshape(2, 12, 8160, 1536)                                    │
    │       → reshape(24, 8160, 1536)    ← 24个"样本"，每个是一帧          │
    │     这样每帧可以独立做注意力                                           │
    │                                                                          │
    │  ② 文本扩展 —— 每帧都配一份同样的剧本:                                 │
    │     txt: (2, 256, 1536)                                               │
    │       → unsqueeze(1) → (2, 1, 256, 1536)                              │
    │       → expand → (2, 12, 256, 1536)   ← 复制12份                     │
    │       → reshape(24, 256, 1536)        ← 和24帧对齐                   │
    │                                                                          │
    │  ③ 生成 Q/K/V —— 两套独立投影:                                         │
    │     视频: vid_qkv(24, 8160, 1536) → (24, 8160, 3×1536)               │
    │            → chunk → vq, vk, vv: 各 (24, 8160, 1536)                  │
    │            → reshape_heads → (24, 24头, 8160, 64)                     │
    │                                                                          │
    │     文本: txt_qkv(24, 256, 1536) → (24, 256, 3×1536)                 │
    │            → chunk → tq, tk, tv: 各 (24, 256, 1536)                   │
    │            → reshape_heads → (24, 24头, 256, 64)                      │
    │                                                                          │
    │     注意: 视频和文本有各自独立的 qkv 投影层！                          │
    │           因为"像素语言"和"文字语言"需要不同的编码方式                 │
    │                                                                          │
    │  ④ 拼接 —— 把视频和文本 token 拼成一个长序列:                           │
    │     q = cat([vq, tq], dim=2) → (24, 24头, 8160+256, 64)              │
    │     k = cat([vk, tk], dim=2) → 同上                                   │
    │     v = cat([vv, tv], dim=2) → 同上                                   │
    │                                                                          │
    │     想象: 8416 个 token 坐在一个大教室里                               │
    │           前8160个是"像素学生"(每帧的空间patch)                        │
    │           后256个是"文字学生"(文本token)                               │
    │           所有人互相看，没有遮罩(双向注意力)                             │
    │                                                                          │
    │  ⑤ Attention 计算:                                                     │
    │     scores = q @ k.T / √64  → (24, 24头, 8416, 8416)                 │
    │     attn = softmax(scores)  → 每行和为1                                │
    │     out = attn @ v          → (24, 24头, 8416, 64)                   │
    │                                                                          │
    │     关键: 这是一个"大融合"操作！                                       │
    │       "橘猫"这个词的 v 向量 → 会被所有包含猫的 patch 关注               │
    │       猫的 patch → 也会回看"橘猫"这个词，确认自己的身份                 │
    │       → 双向交互，文本和视频互相理解                                   │
    │                                                                          │
    │  ⑥ 拆分输出:                                                           │
    │     out: (24, 8416, 1536)                                             │
    │       → 前8160 = vid_s_out → reshape回 (2, 12×8160, 1536)            │
    │       → 后256  = txt_s_out → reshape(2, 12, 256, 1536) → mean(1)     │
    │                              → (2, 256, 1536)                         │
    │                                                                          │
    │     文本为什么要在 nt 方向求平均?                                       │
    │       每帧都产出了一个"更新后的文本表示"                                │
    │       但这些表示应该是一致的(同一句话)                                  │
    │       取平均 = 汇总所有帧对文本的反馈                                   │
    │                                                                          │
    │  ═══ Step 2: 时间注意力 ═══                                            │
    │                                                                          │
    │  ① 空间残差加回:                                                       │
    │     vid_after = vid + vid_s_out  → (2, 97920, 1536)                   │
    │     (已经带有文本信息的空间特征)                                       │
    │                                                                          │
    │  ② 视频重组 —— 这次按空间位置合并:                                     │
    │     vid: (2, 97920, 1536)                                             │
    │       → reshape(2, 12, 8160, 1536)                                    │
    │       → permute(0, 2, 1, 3) → (2, 8160, 12, 1536)                    │
    │       → reshape(16320, 12, 1536)   ← 16320个"样本"                   │
    │                                       每个是同一位置跨12帧              │
    │                                                                          │
    │     想象: 把8160个空间位置"立起来"                                     │
    │           每个位置变成一根"时间轴"                                     │
    │           这根轴上有12个"时间节点"                                     │
    │                                                                          │
    │  ③ 生成 Q/K/V —— 只有视频参与:                                         │
    │     tq, tk, tv: 各 (16320, 12, 1536)                                  │
    │       → reshape_heads → (16320, 24头, 12, 64)                        │
    │                                                                          │
    │  ④ Attention 计算:                                                     │
    │     每根"时间轴"上的12个节点互相看                                     │
    │     → 建模运动连续性(猫从左到右的轨迹)                                 │
    │                                                                          │
    │  ⑤ 还原形状:                                                           │
    │     t_out: (16320, 12, 1536)                                          │
    │       → reshape(2, 8160, 12, 1536)                                    │
    │       → permute(0, 2, 1, 3) → (2, 12, 8160, 1536)                    │
    │       → reshape(2, 97920, 1536)                                       │
    │                                                                          │
    │  ═══ 最终输出 ═══                                                      │
    │                                                                          │
    │    vid_final = vid_s_out + t_out  → (2, 97920, 1536)                  │
    │    txt_out   = txt_s_out          → (2, 256, 1536)                    │
    │                                                                          │
    │    vid_s_out 是"空间更新"(每帧内部调整)                                │
    │    t_out      是"时间更新"(跨帧调整运动)                                │
    │    两者相加 = 同时考虑空间布局和时间连续性                               │
    └────────────────────────────────────────────────────────────────────────┘

    === 为什么文本只在空间注意力中参与，不在时间注意力中？ ===

    文本描述: "一只橘猫在草地上从左走到右"
      - "橘猫" "草地" → 空间内容 → 应该影响每帧的空间排布
      - "从左走到右" → 被 Transformer 编码为语义向量
        → 空间注意力已经把这个语义注入到了视频 patch 中
        → 时间注意力只需要让这些已被"文本引导"的 patch 在帧间对齐
      - 如果文本也参与时间注意力 → 序列从 nt 变成 nt+L → 不必要的开销

    参数:
        d_model (int): 特征维度
        n_heads (int): 注意力头数
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 视频流 QKV（空间+时间共享同一套投影）
        self.vid_qkv = nn.Linear(d_model, 3 * d_model)
        self.vid_spatial_out = nn.Linear(d_model, d_model)

        # 文本流 QKV（只参与空间注意力）
        self.txt_qkv = nn.Linear(d_model, 3 * d_model)
        self.txt_out = nn.Linear(d_model, d_model)

        # 时间注意力的独立投影（只有视频参与）
        self.temporal_qkv = nn.Linear(d_model, 3 * d_model)
        self.temporal_out = nn.Linear(d_model, d_model)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(..., N, d_model) → (..., n_heads, N, d_k)"""
        *batch, N, _ = x.shape
        return x.reshape(*batch, N, self.n_heads, self.d_k).transpose(-3, -2)

    def _attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """标准缩放点积注意力。"""
        scale = self.d_k**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

    def forward(
        self,
        vid: torch.Tensor,
        txt: torch.Tensor,
        nt: int,
        nh: int,
        nw: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        vid: (B, nt*S, d)  — 视频 patch, S = nh*nw
        txt: (B, L, d)     — 文本 token
        nt, nh, nw: 时空 patch 网格形状

        返回: (vid_out, txt_out)
        """
        B = vid.size(0)
        S = nh * nw
        L = txt.size(1)
        d = vid.size(2)

        # ========== Step 1: 空间联合注意力（Spatial Joint Attention） ==========
        # Step 1.1: 视频特征重组
        # 输入 vid: (B, nt*S, d)
        #   - B: batch size，批大小
        #   - nt: 时间上的patch数（帧数），如12
        #   - nh, nw: 空间上patch数（高、宽），S = nh * nw（如68*120=8160）
        #   - d: 特征维度
        #   - S: 每帧的空间patch总数
        # 将视频patch重组为 (B*nt, S, d)，即每一帧作为一个样本，方便每一帧与文本做联合注意力
        """
        # ---- 输入视频与 nt 的关系说明 ----
        # 输入视频数据通常来自连续的多帧图像（例如 24 帧视频片段），
        # 在进入模型前会先用 3D patch embedding 把原始视频按 patch_t（如2）做时间维的切分。
        # 例如原视频 shape (B, C, T, H, W)，T 为原始帧数，经过 patch_t 切分后，
        # 时序维变成 nt = T // patch_t。
        # 在本模型中，nt 表示当前视频 latent 表示中有多少个“时间 patch”（每个 patch 代表 patch_t 帧的内容压缩），
        # 也可以理解为模型视角下的“抽象帧”数目，并不是原始帧数。
        # 
        # 总结：输入一段视频（如24帧），经过 patch_t=2 切分后，nt=12，模型中每个 batch 样本
        # 会表示成 nt 个时间 patch，每个 patch 编码对应时间段的连续多帧内容（如patch_t=2即2帧合成1 patch）。
        """
        # Step 1.1: 视频重组 —— 把batch和帧合并
        # vid: (2, 97920, 1536) → reshape(2, 12, 8160, 1536) → reshape(24, 8160, 1536)
        # 现在24个"样本"，每个是1帧(8160个空间patch)，方便每帧独立做注意力
        vid_spatial = vid.reshape(B, nt, S, d).reshape(B * nt, S, d)

        # Step 1.2: 文本扩展 —— 每帧配一份同样的剧本
        # txt: (2, 256, 1536) → expand → (2, 12, 256, 1536) → reshape(24, 256, 1536)
        # 12帧各配一份256个token的文本，总共24=2×12个"样本"
        txt_spatial = txt.unsqueeze(
            1).expand(-1, nt, -1, -1).reshape(B * nt, L, d)

        # Step 1.3: 生成Q/K/V —— 视频和文本各自独立的投影
        # vid_qkv: Linear(1536 → 4608)，每个patch产出3个1536维向量(q,k,v)
        # 视频QKV: (24, 8160, 1536) → chunk → vq,vk,vv 各(24, 8160, 1536)
        #          → reshape_heads → (24, 24头, 8160, 64)
        vq, vk, vv = [self._reshape_heads(
            x) for x in self.vid_qkv(vid_spatial).chunk(3, dim=-1)]
        # 文本QKV: (24, 256, 1536) → chunk → tq,tk,tv 各(24, 256, 1536)
        #          → reshape_heads → (24, 24头, 256, 64)
        tq, tk, tv = [self._reshape_heads(
            x) for x in self.txt_qkv(txt_spatial).chunk(3, dim=-1)]

        # Step 1.4: 拼接 —— 视频patch和文本token坐进同一个教室
        # q = cat([vq,tq]) → (24, 24头, 8160+256, 64) = (24, 24, 8416, 64)
        # 8416 = 8160个像素学生 + 256个文字学生
        q = torch.cat([vq, tq], dim=2)
        k = torch.cat([vk, tk], dim=2)
        v = torch.cat([vv, tv], dim=2)

        # Step 1.5: Attention —— 8416个token互相打分、加权混合
        # scores: q@k.T / √64 → (24, 24头, 8416, 8416)
        #   每帧的注意力矩阵: 8416×8416 = 7083万个分数
        #   24帧总计: 24 × 7083万 ≈ 17亿个分数
        #   (如果用FlashAttention，不存完整矩阵，峰值显存O(8416))
        # attn = softmax(scores) → 每行和为1
        # out = attn @ v → (24, 24头, 8416, 64)
        out = self._attention(q, k, v)
        # 还原: (24, 24头, 8416, 64) → transpose → (24, 8416, 24, 64)
        #      → reshape → (24, 8416, 1536)
        out = out.transpose(1, 2).reshape(B * nt, S + L, -1)

        # Step 1.6: 拆分 —— 前8160个是视频输出，后256个是文本输出
        # vid_s_out: (24, 8160, 1536) → 过Linear(1536→1536) → 同shape
        vid_s_out = self.vid_spatial_out(out[:, :S])
        # txt_s_out: (24, 256, 1536) → 过Linear(1536→1536) → 同shape
        txt_s_out = self.txt_out(out[:, S:])

        # Step 1.7: 视频还原 —— 把24帧拼回原始batch
        # (24, 8160, 1536) → reshape(2, 12, 8160, 1536) → reshape(2, 97920, 1536)
        vid_s_out = vid_s_out.reshape(B, nt, S, d).reshape(B, nt * S, d)

        # Step 1.8: 文本聚合 —— 每帧都产出一个"更新后的文本"
        # (24, 256, 1536) → reshape(2, 12, 256, 1536) → mean(dim=1)
        # 12帧的文本输出取平均 → (2, 256, 1536)
        # "汇总所有帧对文本的反馈"——12帧一致认为"这是橘猫"
        txt_s_out = txt_s_out.reshape(B, nt, L, d).mean(dim=1)

        # ========== Step 2: 时间注意力（Temporal Attention） ==========
        # Step 2.1: 空间残差加回 —— vid已经带着文本信息了
        # vid: (2, 97920, 1536) + vid_s_out: (2, 97920, 1536)
        vid_after_spatial = vid + vid_s_out

        # Step 2.2: 重组 —— 这次不按帧分，而是按"空间位置"分
        # (2, 97920, 1536) → reshape(2, 12, 8160, 1536)
        #   → permute(0, 2, 1, 3) → (2, 8160, 12, 1536) "把帧维和时间维交换"
        #   → reshape(16320, 12, 1536) "16320=2×8160个时间轴"
        # 想象: 8160个空间位置"立起来"，每个位置是一根"时间轴"(12个时间节点)
        vid_temporal = (
            vid_after_spatial.reshape(B, nt, S, d)
            .permute(0, 2, 1, 3)
            .reshape(B * S, nt, d)
        )

        # Step 2.3: 时间QKV —— 只有视频参与，文本不掺和
        # temporal_qkv: Linear(1536 → 4608)
        # tq,tk,tv: 各(16320, 12, 1536) → reshape_heads → (16320, 24头, 12, 64)
        tq, tk, tv = [self._reshape_heads(x) for x in self.temporal_qkv(
            vid_temporal).chunk(3, dim=-1)]
        # 每根"时间轴"上的12个节点互相看 → 建模运动
        # t_out: (16320, 24头, 12, 64) → transpose → (16320, 12, 1536)
        t_out = self._attention(tq, tk, tv)
        t_out = t_out.transpose(1, 2).reshape(B * S, nt, d)

        # Step 2.4: 还原 —— 把时间轴"放倒"回帧的顺序
        # (16320, 12, 1536) → reshape(2, 8160, 12, 1536)
        #   → permute(0, 2, 1, 3) → (2, 12, 8160, 1536) "帧维放回来"
        #   → reshape(2, 97920, 1536)
        t_out = t_out.reshape(B, S, nt, d).permute(
            0, 2, 1, 3).reshape(B, nt * S, d)
        # 过输出投影: (2, 97920, 1536) → Linear(1536→1536)
        vid_out = self.temporal_out(t_out)

        # Step 3: 最终视频 = 空间更新 + 时间更新
        # vid_s_out: "每帧内部调整"(猫在画面左边还是右边)
        # vid_out:   "跨帧调整运动"(猫从左走到右的轨迹)
        vid_final = vid_s_out + vid_out

        # Step 4: 返回
        # vid_final: (2, 97920, 1536) — 同时有空间布局和时间连续性
        # txt_s_out: (2, 256, 1536)   — 吸收了所有帧的反馈
        return vid_final, txt_s_out


# ═══════════════════════════════════════════════════════════════════════════
# 4. VideoMMDiTBlock — 双流视频 MM-DiT 块
# ═══════════════════════════════════════════════════════════════════════════
class VideoMMDiTBlock(nn.Module):
    """
    双流视频 MM-DiT 块 —— "各干各的 + 定期交流"
    =============================================

    这个模块是整个模型的"前8层"（双流层），核心思想:
      视频和文本是两种不同的"语言"，先各自学好，再交换信息。

    ┌────────────────────────────────────────────────────────────────────────┐
    │  为什么需要"双流"?                                                   │
    │                                                                        │
    │  想象一个跨国团队:                                                     │
    │    视频 = 中国工程师（擅长像素、纹理、运动）                           │
    │    文本 = 美国产品经理（擅长语义、概念、描述）                         │
    │                                                                        │
    │  如果一开始就把他们混在一起工作:                                       │
    │    → 工程师被迫学英文语法，经理被迫理解卷积核                           │
    │    → 双方都学不好自己的专长                                           │
    │                                                                        │
    │  双流方案:                                                             │
    │    1. 各干各的: 工程师有自己的 Norm/FFN，经理有自己的 Norm/FFN         │
    │    2. 定期交流: STJointAttention 是"联合会议"，双方交换信息           │
    │    3. 保持独立: 交流完各自继续自己的工作                               │
    │                                                                        │
    │  与 mmdit.py 的 MMDiTBlock 对比:                                      │
    │    MMDiTBlock:      只做图像，没有时间维（更简单）                     │
    │    VideoMMDiTBlock: 加入时间注意力，处理视频（更复杂）                 │
    └────────────────────────────────────────────────────────────────────────┘

    ┌────────────────────────────────────────────────────────────────────────┐
    │  完整数据流 (B=2, nt=12, S=8160, L=256, d=1536)                       │
    │                                                                        │
    │  输入:                                                                 │
    │    vid:   (2, 97920, 1536)  — 视频 patch                              │
    │    txt:   (2, 256, 1536)    — 文本 token                              │
    │    t_emb: (2, 1536)         — 时间步条件("现在是去噪第几步")          │
    │                                                                        │
    │  ═══ 条件调制 (ADA-LN) ═══                                            │
    │                                                                        │
    │  时间步 t_emb 像"进度条":                                             │
    │    t=1.0 (纯噪声): "大胆猜，随便改"                                   │
    │    t=0.5 (半成品): "参考已有结构，精细调整"                           │
    │    t=0.0 (接近完成): "几乎别改，只做微调"                             │
    │                                                                        │
    │  vid_mod(t_emb) → [γ1, β1, α1, γ2, β2, α2]                          │
    │  txt_mod(t_emb) → [γ1, β1, α1, γ2, β2, α2]                          │
    │                                                                        │
    │  每组 6 个数的分工:                                                   │
    │    γ1, β1: 注意力前的"缩放+偏移" —— 调节输入特征的分布               │
    │    α1:     注意力输出的"强度" —— 控制残差连接的贡献                  │
    │    γ2, β2: FFN 前的"缩放+偏移"                                       │
    │    α2:     FFN 输出的"强度"                                          │
    │                                                                        │
    │  ═══ 时空联合注意力 ═══                                               │
    │                                                                        │
    │  vid_h = γ1_v * RMSNorm(vid) + β1_v   → (2, 97920, 1536)            │
    │  txt_h = γ1_t * RMSNorm(txt) + β1_t   → (2, 256, 1536)              │
    │                                                                        │
    │  ↓ STJointAttention                                                   │
    │    Step 1: 每帧空间 + 文本 → Joint Attn → vid_s_out, txt_s_out      │
    │    Step 2: 每位置跨帧 → Temporal Attn → t_out                        │
    │                                                                        │
    │  vid = vid + α1_v * (vid_s_out + t_out)  — 空间更新 + 时间更新       │
    │  txt = txt + α1_t * txt_s_out            — 文本更新                  │
    │                                                                        │
    │  ═══ 各自 FFN ═══                                                     │
    │                                                                        │
    │  vid_h = γ2_v * RMSNorm(vid) + β2_v                                   │
    │  vid = vid + α2_v * FFN(vid_h)                                        │
    │                                                                        │
    │  txt_h = γ2_t * RMSNorm(txt) + β2_t                                   │
    │  txt = txt + α2_t * FFN(txt_h)                                        │
    │                                                                        │
    │  注意: 视频和文本的 FFN 是独立的！参数不共享。                        │
    │        因为"像素处理"和"语义处理"需要不同的变换。                     │
    └────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()

        # 视频流
        self.vid_mod = Modulation(d_model, n_modulations=6)
        self.vid_norm1 = RMSNorm(d_model)
        self.vid_norm2 = RMSNorm(d_model)
        self.vid_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # 文本流
        self.txt_mod = Modulation(d_model, n_modulations=6)
        self.txt_norm1 = RMSNorm(d_model)
        self.txt_norm2 = RMSNorm(d_model)
        self.txt_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # 时空联合注意力
        self.st_joint_attn = STJointAttention(d_model, n_heads)

    def forward(
        self,
        vid: torch.Tensor,
        txt: torch.Tensor,
        t_emb: torch.Tensor,
        nt: int,
        nh: int,
        nw: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        vid:   (2, 97920, 1536)  — 视频 patch (2个样本, 97920=12×8160)
        txt:   (2, 256, 1536)    — 文本 token (256个token)
        t_emb: (2, 1536)         — 时间步条件 (如[0.7, 0.3]表示去噪进度)
        nt=12, nh=68, nw=120     — 时空网格: 12帧, 每帧68×120个空间patch

        返回: (vid, txt) 同输入shape — 这是残差更新，shape不变
        """
        # === 条件调制: 时间步控制"去噪进度" ===
        # t_emb: (2, 1536) → vid_mod → 6个(2, 1536)的调制参数
        # 例: t=0.7(早期) → γ≈1.0, β≈0, α≈0.3 → "大胆改"
        #     t=0.1(晚期) → γ≈0.9, β≈0.1, α≈0.05 → "微调"
        v_γ1, v_β1, v_α1, v_γ2, v_β2, v_α2 = self.vid_mod(t_emb)
        t_γ1, t_β1, t_α1, t_γ2, t_β2, t_α2 = self.txt_mod(t_emb)

        # === 时空联合注意力 ===
        # vid: (2, 97920, 1536) → vid_norm1 → 同shape
        #      → *γ1 + β1 → vid_h: (2, 97920, 1536)
        # γ1=1.0, β1=0.1 表示"把特征放大一点再偏移"
        vid_h = v_γ1 * self.vid_norm1(vid) + v_β1
        txt_h = t_γ1 * self.txt_norm1(txt) + t_β1
        # STJointAttention 内部:
        #   Step 1: 每帧+文本 → Joint Attn → vid_s_out(2,97920,1536), txt_s_out(2,256,1536)
        #   Step 2: 每位置跨帧 → Temporal Attn → t_out(2,97920,1536)
        #   vid_attn = vid_s_out + t_out → (2, 97920, 1536)
        vid_attn, txt_attn = self.st_joint_attn(vid_h, txt_h, nt, nh, nw)
        # 残差: 原始vid + α1×注意力输出
        # α1=0.3 → "注意力输出贡献30%"，原始特征保留70%
        vid = vid + v_α1 * vid_attn
        txt = txt + t_α1 * txt_attn

        # === 各自 FFN (视频和文本独立) ===
        # vid: (2, 97920, 1536) → vid_norm2 → *γ2 + β2 → vid_h: 同shape
        # → vid_ffn: Linear(1536→6144) → GELU → Linear(6144→1536)
        #   相当于"每个token过一个小MLP"，非线性变换特征
        vid_h = v_γ2 * self.vid_norm2(vid) + v_β2
        vid = vid + v_α2 * self.vid_ffn(vid_h)
        # 文本同理: (2, 256, 1536) → txt_ffn → 同shape
        # 注意: vid_ffn和txt_ffn参数完全独立！不共享权重

        txt_h = t_γ2 * self.txt_norm2(txt) + t_β2
        txt = txt + t_α2 * self.txt_ffn(txt_h)

        return vid, txt


# ═══════════════════════════════════════════════════════════════════════════
# 5. VideoSingleStreamSTBlock — 单流时空注意力块
# ═══════════════════════════════════════════════════════════════════════════
class VideoSingleStreamSTBlock(nn.Module):
    """
    单流视频 MM-DiT 块 —— "坐到同一张桌子前共同完成"
    ===================================================

    这是模型的"后8层"（单流层），核心思想:
      经过双流层的"各干各的+交流"后，视频和文本已经有了良好的基础表示。
      现在需要深度对齐——让"橘猫"这个词精确对应画面中的猫。

    ┌────────────────────────────────────────────────────────────────────────┐
    │  为什么需要"单流"?                                                   │
    │                                                                        │
    │  类比:                                                                 │
    │    双流层 = 两个人先各自做功课，再交流笔记                             │
    │    单流层 = 两人坐到同一张桌子前，一起完成最终方案                     │
    │                                                                        │
    │  双流层的局限:                                                         │
    │    虽然文本和视频交换了信息，但 Norm/FFN 参数是独立的。                │
    │    视频学的特征和文本学的特征可能"各说各话"——                         │
    │    视频认为"这个区域是猫"，文本认为"猫在左边"，                      │
    │    但两者的"猫"概念可能对不齐。                                        │
    │                                                                        │
    │  单流层的解决:                                                         │
    │    把视频 patch 和文本 token 拼成一个长序列。                          │
    │    共享同一套 Norm/FFN/Attention 参数。                                │
    │    强制两者学习统一的特征空间。                                        │
    │                                                                        │
    │  与双流层的本质区别:                                                   │
    │    双流: 视频和文本"各自有各自的老师"(独立的FFN参数)                  │
    │    单流: 视频和文本"同一个老师教"(共享FFN参数)                        │
    │                                                                        │
    │  为什么不全用单流?                                                     │
    │    前期视频(噪声)和文本(语义)差异太大，共享参数会导致一方"绑架"另一方。│
    │    先双流各自学好基础，再单流深度对齐，是最佳策略。                    │
    └────────────────────────────────────────────────────────────────────────┘

    ┌────────────────────────────────────────────────────────────────────────┐
    │  完整数据流 (B=2, nt=12, S=8160, L=256, d=1536)                       │
    │                                                                        │
    │  输入: x = [vid_tokens | txt_tokens]                                  │
    │    x: (2, 97920+256, 1536) = (2, 98176, 1536)                        │
    │                                                                        │
    │  ═══ 条件调制 ═══                                                     │
    │    h = γ * RMSNorm(x) + β                                             │
    │                                                                        │
    │  ═══ 分离视频和文本 ═══                                               │
    │    vid_h = h[:, :97920]  → (2, 97920, 1536)                          │
    │    txt_h = h[:, 97920:]  → (2, 256, 1536)                            │
    │                                                                        │
    │  ═══ 空间注意力 + 并行 FFN (FLUX 风格) ═══                            │
    │                                                                        │
    │  FLUX 的一个创新: Attention 和 FFN 同时计算，不串行。                 │
    │  传统: Attention → 残差 → Norm → FFN → 残差 (串行，慢)                │
    │  FLUX: Attention + FFN 并行 → 相加 → 残差 (并行，快)                  │
    │                                                                        │
    │  ① 视频按帧拆分，文本扩展:                                             │
    │     vid_per_frame: (24, 8160, 1536)   ← 24=2×12                      │
    │     txt_expanded:  (24, 256, 1536)    ← 每帧配一份文本               │
    │     frame_seq:     (24, 8416, 1536)   ← 拼接                         │
    │                                                                        │
    │  ② 并行计算 Attention 和 FFN:                                          │
    │     qkv_ffn_in(frame_seq) → (24, 8416, 3×1536 + d_ff)                │
    │     → split → qkv (给 Attention) + ffn_in (给 FFN)                   │
    │     → attn_out + ffn_out = spatial_out                               │
    │                                                                        │
    │  ③ 分回视频和文本:                                                     │
 │     vid_s_out: (2, 97920, 1536)                                       │
    │     txt_s_out: (2, 256, 1536)   ← 在 nt 方向求平均                   │
    │                                                                        │
    │  ═══ 时间注意力 (只对视频) ═══                                        │
    │                                                                        │
    │    vid_after = x[:, :97920] + α * vid_s_out   — 先做空间残差         │
    │    → 重组为 (16320, 12, 1536) — 每位置跨12帧                         │
    │    → Temporal Attention → t_out                                       │
    │    → 还原形状 → vid_final                                             │
    │                                                                        │
    │  文本更新:                                                             │
    │    txt_final = x[:, 97920:] + α * txt_s_out                           │
    │                                                                        │
    │  输出: [vid_final | txt_final] → (2, 98176, 1536)                    │
    └────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model
        self.d_ff = d_ff

        self.mod = Modulation(d_model, n_modulations=3)
        self.norm = RMSNorm(d_model)

        # 空间注意力 + FFN 并行
        self.qkv_ffn_in = nn.Linear(d_model, 3 * d_model + d_ff)
        self.ffn_act = nn.GELU(approximate="tanh")
        self.ffn_out = nn.Linear(d_ff, d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        # 时间注意力
        self.temporal_norm = RMSNorm(d_model)
        self.temporal_qkv = nn.Linear(d_model, 3 * d_model)
        self.temporal_out = nn.Linear(d_model, d_model)

    def _attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        scale = self.d_k**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        nt: int,
        nh: int,
        nw: int,
        n_txt: int,
    ) -> torch.Tensor:
        """
        x:     (B, nt*S + L, d)  — 视频+文本拼接序列
        t_emb: (B, d)
        nt, nh, nw: 视频 patch 网格
        n_txt: 文本 token 数

        返回: (B, nt*S + L, d)
        """
        B = x.size(0)          # 2
        S = nh * nw              # 68×120 = 8160
        N_vid = nt * S           # 12×8160 = 97920
        d = self.d_model         # 1536

        # 调制: (2, 1536) → 3个(2, 1536) —— 只有3个(γ,β,α)，比双流的6个少
        # 单流共享调制，因为视频和文本"同一个老师教"
        gamma, beta, alpha = self.mod(t_emb)
        # h: (2, 98176, 1536) → norm → *γ + β → 同shape
        h = gamma * self.norm(x) + beta

        # 分离视频和文本
        # vid_h: (2, 98176, 1536) → 前97920个 → (2, 97920, 1536)
        # txt_h: (2, 98176, 1536) → 后256个  → (2, 256, 1536)
        vid_h = h[:, :N_vid]
        txt_h = h[:, N_vid:]

        # ========== 空间注意力 + 并行 FFN (FLUX风格) ==========
        # 视频按帧拆分: (2, 97920, 1536) → reshape(2,12,8160,1536) → (24, 8160, 1536)
        vid_per_frame = vid_h.reshape(B, nt, S, d).reshape(B * nt, S, d)
        # 文本扩展: (2, 256, 1536) → expand → (2,12,256,1536) → (24, 256, 1536)
        txt_expanded = (
            txt_h.unsqueeze(1).expand(-1, nt, -1, -1).reshape(B * nt, n_txt, d)
        )
        # 拼接: cat([vid, txt]) → (24, 8160+256, 1536) = (24, 8416, 1536)
        frame_seq = torch.cat([vid_per_frame, txt_expanded], dim=1)

        # FLUX的并行创新: 一个Linear同时产出QKV和FFN输入
        # qkv_ffn_in: Linear(1536 → 3×1536 + 6144) = Linear(1536 → 10752)
        qkv_ffn = self.qkv_ffn_in(frame_seq)           # (24, 8416, 10752)
        # split: qkv(24, 8416, 4608) + ffn_in(24, 8416, 6144)
        qkv, ffn_in = qkv_ffn.split([3 * d, self.d_ff], dim=-1)
        q, k, v = qkv.chunk(3, dim=-1)                 # 各(24, 8416, 1536)

        # 拆多头: (24, 8416, 1536) → reshape → (24, 8416, 24, 64) → transpose
        #        → (24, 24头, 8416, 64)
        N_frame = S + n_txt        # 8416
        q = q.reshape(B * nt, N_frame, self.n_heads, self.d_k).transpose(1, 2)
        k = k.reshape(B * nt, N_frame, self.n_heads, self.d_k).transpose(1, 2)
        v = v.reshape(B * nt, N_frame, self.n_heads, self.d_k).transpose(1, 2)

        # Attention: (24, 24头, 8416, 64) @ (24, 24头, 64, 8416) → (24, 24, 8416, 8416)
        # → softmax → @v → (24, 24, 8416, 64)
        attn_out = self._attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B * nt, N_frame, d)  # (24, 8416, 1536)
        attn_out = self.attn_out(attn_out)      # 输出投影: Linear(1536→1536)

        # FFN分支(并行): (24, 8416, 6144) → GELU → Dropout → Linear(6144→1536)
        ffn_out = self.ffn_out(self.dropout_layer(self.ffn_act(ffn_in)))

        # FLUX创新: Attention输出 + FFN输出 = 空间输出
        # 传统是串行: Attn → 残差 → FFN → 残差
        # FLUX是并行: Attn和FFN同时算，结果相加 → 一次残差
        spatial_out = attn_out + ffn_out        # (24, 8416, 1536)

        # 分回视频和文本
        # vid: (24, 8416, 1536) → 前8160个 → reshape(2, 97920, 1536)
        vid_s_out = spatial_out[:, :S].reshape(B, nt * S, d)
        # txt: (24, 8416, 1536) → 后256个 → reshape(2,12,256,1536) → mean(1)
        txt_s_out = spatial_out[:, S:].reshape(B, nt, n_txt, d).mean(dim=1)

        # ========== 时间注意力（只对视频）==========
        # vid_after: 原始视频x + α×空间更新 → (2, 97920, 1536)
        vid_after = x[:, :N_vid] + alpha[:, :, :d] * vid_s_out
        vid_t = self.temporal_norm(vid_after)   # RMSNorm
        # 重组: (2, 97920, 1536) → reshape(2,12,8160,1536)
        #   → permute(0,2,1,3) → (2, 8160, 12, 1536)
        #   → reshape(16320, 12, 1536) —— 8160根"时间轴"
        vid_t = vid_t.reshape(B, nt, S, d).permute(
            0, 2, 1, 3).reshape(B * S, nt, d)

        # 时间QKV: temporal_qkv(1536→4608) → chunk → 各(16320, 12, 1536)
        tq, tk, tv = self.temporal_qkv(vid_t).chunk(3, dim=-1)
        tq = tq.reshape(B * S, nt, self.n_heads, self.d_k).transpose(1, 2)
        tk = tk.reshape(B * S, nt, self.n_heads, self.d_k).transpose(1, 2)
        tv = tv.reshape(B * S, nt, self.n_heads, self.d_k).transpose(1, 2)
        # 每根时间轴上的12个节点互相看 → (16320, 24头, 12, 64)
        t_out = self._attention(tq, tk, tv)
        t_out = t_out.transpose(1, 2).reshape(B * S, nt, d)  # (16320, 12, 1536)
        t_out = self.temporal_out(t_out)        # Linear(1536→1536)
        # 还原: (16320, 12, 1536) → reshape(2, 8160, 12, 1536)
        #   → permute(0,2,1,3) → (2, 12, 8160, 1536) → (2, 97920, 1536)
        t_out = t_out.reshape(B, S, nt, d).permute(
            0, 2, 1, 3).reshape(B, nt * S, d)

        # 视频最终 = 空间残差后的视频 + 时间更新
        vid_final = vid_after + t_out           # (2, 97920, 1536)

        # 文本更新
        txt_final = x[:, N_vid:] + alpha[:, :, :d] * txt_s_out

        return torch.cat([vid_final, txt_final], dim=1)


# ═══════════════════════════════════════════════════════════════════════════
# 6. FinalLayer — 输出头
# ═══════════════════════════════════════════════════════════════════════════
class FinalLayer(nn.Module):
    """输出投影: Transformer 特征 → patch 像素空间。"""

    def __init__(self, d_model: int, patch_dim: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mod = Modulation(d_model, n_modulations=2)
        self.linear = nn.Linear(d_model, patch_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x:     (2, 97920, 1536) — 单流层输出的视频特征
        t_emb: (2, 1536)         — 时间步条件
        返回:  (2, 97920, 48)    — 每个patch的像素预测值
        """
        # 最终调制: (2, 1536) → [γ, β] 两个参数
        # 时间步越接近0(去噪完成)，调制越强，让输出更"确定"
        gamma, beta = self.mod(t_emb)
        # x: (2, 97920, 1536) → norm → *γ + β → 同shape
        # 调制作用: t=1.0(纯噪声)时γ≈1,β≈0 → 不改; t=0.0(完成)时微调
        x = gamma * self.norm(x) + beta
        # linear: (2, 97920, 1536) → Linear(1536→48) → (2, 97920, 48)
        # 48 = patch_dim = 2×2×2×4 = pt×ph×pw×C
        # 每个token预测一个3D patch的48个像素值(噪声)
        return self.linear(x)


# ═══════════════════════════════════════════════════════════════════════════
# 7. VideoMMDiT — 完整的视频 MM-DiT 模型
# ═══════════════════════════════════════════════════════════════════════════
class VideoMMDiT(nn.Module):
    """
    完整的视频 MM-DiT 模型 — 当前视频生成的主流架构。

    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  输入:                                                                 │
    │    z_t:  (B, 4, 24, 136, 240) — 加噪的视频 latent (3D VAE 输出)     │
    │    t:    (B,)                  — 扩散时间步                           │
    │    text: (B, 256, 4096)       — T5 文本编码                          │
    │                                                                        │
    │  ── Embedding ──                                                      │
    │  z_t  → 3D PatchEmbed → (B, 97920, 1536)  vid_tokens                │
    │  t    → TimestepEmbed → (B, 1536)          t_emb                     │
    │  text → TextProj → (B, 256, 1536)          txt_tokens                │
    │  + 位置编码                                                           │
    │                                                                        │
    │  ── 双流层 × 8 ──                                                    │
    │  vid, txt = VideoMMDiTBlock(vid, txt, t_emb, nt, nh, nw) × 8        │
    │    每层内部:                                                          │
    │      Step 1: 每帧空间 + 文本 → Joint Attention (双向)                │
    │      Step 2: 每位置跨帧 → Temporal Attention                         │
    │      Step 3: 各自 FFN                                                 │
    │                                                                        │
    │  ── 拼接 → 单流层 × 8 ──                                             │
    │  x = [vid ; txt] → VideoSingleStreamSTBlock × 8                     │
    │    共享参数，空间注意力 + 时间注意力 + 并行 FFN                      │
    │                                                                        │
    │  ── 输出 ──                                                           │
    │  取视频部分 → FinalLayer → unpatchify3D                              │
    │  → (B, 4, 24, 136, 240) — 预测的噪声 ε_θ                           │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, cfg: VideoMMDiTConfig):
        super().__init__()
        self.cfg = cfg

        # ═══ Embedding 层 ═══
        # 3D Patch Embedding: (B,4,24,136,240) → (B,97920,1536)
        #   Conv3d(kernel=2×2×2, stride=2×2×2) 把时空切成patch再投影到d维
        self.patch_embed = VideoPatchEmbedding3D(cfg)

        # 时间步嵌入: (B,)标量 → (B,1536)向量
        #   正弦编码 + MLP，把"去噪进度"变成可学习的条件
        self.t_embed = TimestepEmbedding(cfg.d_model)

        # 文本投影: (B,256,4096) → (B,256,1536)
        #   T5-XXL输出4096维，需要映射到模型的1536维
        self.text_proj = nn.Linear(cfg.text_d_model, cfg.d_model)

        # 可学习位置编码 —— 让模型知道"这是第几个patch/token"
        # vid_pos_embed: (1, 97920, 1536) — 每个视频patch一个位置向量
        #   97920 = 12帧 × 68高 × 120宽 = nt×nh×nw
        # txt_pos_embed: (1, 256, 1536) — 每个文本token一个位置向量
        self.vid_pos_embed = nn.Parameter(
            torch.zeros(1, cfg.num_patches, cfg.d_model))
        self.txt_pos_embed = nn.Parameter(
            torch.zeros(1, cfg.text_max_len, cfg.d_model))

        # ═══ 双流层 × 8 ═══
        # 每层: 视频和文本"各干各的"，通过STJointAttention定期交流
        # 视频流: vid_norm1 → STJointAttn → vid_norm2 → vid_ffn
        # 文本流: txt_norm1 → STJointAttn → txt_norm2 → txt_ffn
        # 两套参数完全独立，防止模态互相"绑架"
        self.double_blocks = nn.ModuleList(
            [
                VideoMMDiTBlock(cfg.d_model, cfg.n_heads,
                                cfg.d_ff, cfg.dropout)
                for _ in range(cfg.n_double_layers)
            ]
        )

        # ═══ 单流层 × 8 ═══
        # 每层: 视频和文本拼成一个序列，共享同一套Norm/FFN/Attention
        # 输入: (B, 97920+256, 1536) = (B, 98176, 1536)
        # 强制两种模态学习统一的特征空间，实现深度语义-画面对齐
        self.single_blocks = nn.ModuleList(
            [
                VideoSingleStreamSTBlock(
                    cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout
                )
                for _ in range(cfg.n_single_layers)
            ]
        )

        # ═══ 输出头 ═══
        # FinalLayer: (B, 97920, 1536) → (B, 97920, 48)
        #   48 = patch_dim = 2×2×2×4 = 每个3D patch的像素数
        #   用t_emb做最终调制，时间步越接近0(完成)输出越"确定"
        self.final_layer = FinalLayer(cfg.d_model, cfg.patch_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.vid_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.txt_pos_embed, std=0.02)

    def unpatchify3d(self, x: torch.Tensor) -> torch.Tensor:
        """
        3D unpatchify: patch 序列 → 视频 latent。

        这是patch_embed的逆操作。把Transformer输出的"一维token序列"
        还原回"五维视频张量"(B, C, T, H, W)。

        完整维度流转 (B=2, nt=12, nh=68, nw=120, patch=2×2×2):
          x: (2, 97920, 48)
            → reshape(2, 12, 68, 120, 2, 2, 2, 4)
              把97920个token拆回时空网格: 12帧×68高×120宽
              每个位置包含2×2×2的空间块和4个通道
            → permute(0,7,1,4,2,5,3,6)
              重排维度: (B, C, nt, pt, nh, ph, nw, pw)
              例: (2, 4, 12, 2, 68, 2, 120, 2)
            → reshape(2, 4, 24, 136, 240)
              合并patch维度: 12×2=24帧, 68×2=136高, 120×2=240宽
        """
        c = self.cfg
        # x: (2, 97920, 48) → reshape → (2, 12, 68, 120, 2, 2, 2, 4)
        #   把token序列还原成"时空网格 + patch内部"
        x = x.reshape(
            -1, c.nt, c.nh, c.nw, c.patch_t, c.patch_h, c.patch_w, c.in_channels
        )
        # permute: (B, nt, nh, nw, pt, ph, pw, C)
        #       → (B, C, nt, pt, nh, ph, nw, pw)
        # 重排维度顺序，让通道在最前，相邻的patch维度相邻
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        # reshape: (B, C, nt, pt, nh, ph, nw, pw)
        #       → (B, C, nt*pt, nh*ph, nw*pw)
        # 例: (2, 4, 12, 2, 68, 2, 120, 2) → (2, 4, 24, 136, 240)
        # 合并patch边界，还原连续的视频latent
        x = x.reshape(-1, c.in_channels, c.latent_t, c.latent_h, c.latent_w)
        return x

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        视频 MM-DiT 前向传播 —— 从"噪声+文本"预测"应该去掉的噪声"

        这是扩散模型的核心: ε_θ(z_t, t, text) → 预测噪声
        训练时和真实噪声求 MSE loss；推理时一步步去噪生成视频。

        参数:
            z_t:      (B, C, T, H, W)          — 加噪的视频 latent (3D VAE 输出)
            t:        (B,)                       — 扩散时间步 (0=完成, 1=纯噪声)
            text_emb: (B, text_len, text_dim)   — T5/CLIP 文本编码

        返回:     (B, C, T, H, W)          — 预测的噪声 ε_θ

        ┌────────────────────────────────────────────────────────────────────────┐
        │  完整数据流 (B=2, C=4, T=24, H=136, W=240, L=256, d=1536)             │
        │                                                                        │
        │  输入:                                                                 │
        │    z_t:      (2, 4, 24, 136, 240)   — 加噪视频 (3D VAE latent)       │
        │    t:        (2,)                    — 时间步，如 [0.7, 0.3]         │
        │    text_emb: (2, 256, 4096)         — T5编码的文本                   │
        │                                                                        │
        │  ═══ Embedding ═══                                                    │
        │                                                                        │
        │  z_t → PatchEmbed3D → (2, 97920, 1536)  vid_tokens                  │
        │  t   → TimestepEmbed  → (2, 1536)       t_emb (时间步条件)          │
        │  text_emb → TextProj  → (2, 256, 1536)  txt_tokens                  │
        │  + 位置编码                                                           │
        │                                                                        │
        │  ═══ 双流层 × 8 —— "各干各的 + 交流" ═══                             │
        │                                                                        │
        │  for i in range(8):                                                   │
        │    vid, txt = VideoMMDiTBlock(vid, txt, t_emb, nt, nh, nw)          │
        │      每层:                                                            │
        │        ① 条件调制: γ·Norm(x)+β (根据时间步调节特征)                  │
        │        ② STJointAttention:                                          │
        │           Step 1: 每帧 + 文本 → Joint Attn (双向)                   │
        │           Step 2: 每位置跨帧 → Temporal Attn                         │
        │        ③ 残差: x = x + α·attn_out                                   │
        │        ④ 各自 FFN: x = x + α·FFN(Norm(x))                          │
        │                                                                        │
        │  双流层结束后:                                                        │
        │    vid: (2, 97920, 1536) — 已注入文本语义，有空间+时间特征          │
        │    txt: (2, 256, 1536)   — 已收到视频反馈，知道画面什么样           │
        │                                                                        │
        │  ═══ 拼接 → 单流层 × 8 —— "坐到同一张桌子" ═══                       │
        │                                                                        │
        │  x = cat([vid, txt], dim=1) → (2, 98176, 1536)                      │
        │                                                                        │
        │  for i in range(8):                                                   │
        │    x = VideoSingleStreamSTBlock(x, t_emb, nt, nh, nw, n_txt)        │
        │      每层:                                                            │
        │        ① 条件调制                                                   │
        │        ② 分离视频/文本 → 每帧做空间注意力 + 并行FFN                 │
        │        ③ 视频做时间注意力                                           │
        │        ④ 拼回 → 残差更新                                            │
        │                                                                        │
        │  单流层结束后:                                                        │
        │    x: (2, 98176, 1536) — 视频和文本深度对齐                         │
        │                                                                        │
        │  ═══ 输出头 ═══                                                       │
        │                                                                        │
        │  vid_out = x[:, :97920]        — 取视频部分                         │
        │  vid_out = FinalLayer(vid_out) — (2, 97920, patch_dim)              │
        │  noise_pred = unpatchify3d(vid_out)                                 │
        │             → (2, 4, 24, 136, 240) — 预测的噪声 ε_θ                │
        │                                                                        │
        │  这个 noise_pred 和训练时的真实噪声求 MSE loss:                       │
        │    loss = MSE(noise_pred, noise_true)                               │
        └────────────────────────────────────────────────────────────────────────┘
        """
        cfg = self.cfg

        # ═══ Step 1: Embedding ═══
        # 3D Patch Embedding: 把视频 latent 切成 3D patch
        # z_t: (B, C, T, H, W) → vid: (B, nt×nh×nw, d)
        # 例: (2, 4, 24, 136, 240) → (2, 97920, 1536)
        vid = self.patch_embed(z_t)

        # 时间步嵌入: 把标量 t 变成 d 维向量
        # t: (B,) → t_emb: (B, d)
        # 例: (2,) → (2, 1536)
        # 这个 t_emb 会传给每一层的 Modulation，控制"去噪进度"
        t_emb = self.t_embed(t)

        # 文本投影: 把 T5/CLIP 的文本特征映射到模型维度
        # text_emb: (B, L, text_d) → txt: (B, L, d)
        # 例: (2, 256, 4096) → (2, 256, 1536)
        txt = self.text_proj(text_emb)

        # 加可学习位置编码 —— 让模型知道每个 token 的位置
        # vid_pos_embed: (1, num_patches, d) — 预训练的视觉位置编码
        # txt_pos_embed: (1, text_max_len, d) — 预训练的文本位置编码
        vid = vid + self.vid_pos_embed[:, : vid.size(1)]
        txt = txt + self.txt_pos_embed[:, : txt.size(1)]

        n_txt = txt.size(1)

        # ═══ Step 2: 双流层（前 n_double_layers 层）═══
        # 核心思想: 视频和文本"各干各的"，但通过 STJointAttention 定期交流。
        # 视频有独立的 Norm/FFN/Modulation，文本也有独立的。
        # 这保证两种模态都能学好各自的特征，不被对方"带偏"。
        for block in self.double_blocks:
            vid, txt = block(vid, txt, t_emb, cfg.nt, cfg.nh, cfg.nw)

        # ═══ Step 3: 单流层（后 n_single_layers 层）═══
        # 核心思想: 把视频和文本拼成一个序列，共享参数做深度对齐。
        # 经过双流层后，视频和文本已经有了良好的基础表示。
        # 单流层让"橘猫"这个词精确对应画面中猫的位置。
        x = torch.cat([vid, txt], dim=1)  # (B, nt×S + L, d)
        for block in self.single_blocks:
            x = block(x, t_emb, cfg.nt, cfg.nh, cfg.nw, n_txt)

        # ═══ Step 4: 输出头 ═══
        # 只取视频部分（前 nt×S 个 token），文本部分不用于输出
        vid_out = x[:, : cfg.num_patches]  # (B, nt×S, d)

        # FinalLayer: 把 Transformer 特征投影回 patch 像素空间
        # 同时用 t_emb 做最终调制（时间步越接近0，调制越强）
        vid_out = self.final_layer(vid_out, t_emb)  # (B, nt×S, patch_dim)

        # 把 patch 序列还原回 3D 视频 latent
        # (B, nt×S, patch_dim) → (B, C, T, H, W)
        noise_pred = self.unpatchify3d(vid_out)

        return noise_pred
