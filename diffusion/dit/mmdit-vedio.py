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
"""

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
        x = self.proj(x)  # (B, d, nt, nh, nw)
        return x.flatten(2).transpose(1, 2)  # (B, N, d)


# ═══════════════════════════════════════════════════════════════════════════
# 3. STJointAttention — 时空联合注意力（本文件的核心创新）
# ═══════════════════════════════════════════════════════════════════════════
class STJointAttention(nn.Module):
    """
    Spatial-Temporal Joint Attention — 视频 MM-DiT 的核心。

    结合了两个来源的创新:
      来自 MM-DiT:  Joint Attention (文本和视频双向交互)
      来自 ST-DiT:  时空分离 (空间注意力 + 时间注意力)

    === 数据流 ===

    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  输入:                                                                 │
    │    vid: (B, nt×S, d)   — 视频 patch (S = nh×nw 每帧空间 patch 数)   │
    │    txt: (B, L, d)      — 文本 token                                  │
    │                                                                        │
    │  ── Step 1: Spatial Joint Attention ──                                │
    │     B×nt是视频的帧数，S是空间patch数                                                                   │
    │    将视频 reshape 成 (B×nt, S, d)  — 每帧独立                        │
    │    将文本 expand 成 (B×nt, L, d)   — 每帧都看到同样的文本            │
    │                                                                        │
    │    对每帧做 Joint Attention:                                          │
    │      Q = [vid_Q_frame ; txt_Q]                                        │
    │      K = [vid_K_frame ; txt_K]                                        │
    │      V = [vid_V_frame ; txt_V]                                        │
    │      out = softmax(QK^T/√d) · V                                      │
    │      vid_out_frame, txt_out = split(out)                              │
    │                                                                        │
    │    帧0: [■■■■ vid_patches + ◆◆ txt] ←→ Joint Attn                   │
    │    帧1: [■■■■ vid_patches + ◆◆ txt] ←→ Joint Attn                   │
    │    ...                                                                 │
    │    帧11:[■■■■ vid_patches + ◆◆ txt] ←→ Joint Attn                   │
    │                                                                        │
    │    文本和每帧空间内容双向交互 ✓                                       │
    │    复杂度: O((S+L)²) × nt                                            │
    │                                                                        │
    │  ── Step 2: Temporal Attention ──                                     │
    │                                                                        │
    │    将视频 reshape 成 (B×S, nt, d)  — 每个空间位置独立                 │
    │    文本不参与（文本没有时间结构）                                     │
    │                                                                        │
    │    位置(0,0): t0 ↔ t1 ↔ ... ↔ t11                                   │
    │    位置(0,1): t0 ↔ t1 ↔ ... ↔ t11                                   │
    │    ...                                                                 │
    │                                                                        │
    │    建模时间连续性 ✓                                                   │
    │    复杂度: O(nt²) × S — 很小                                         │
    │                                                                        │
    │  输出:                                                                 │
    │    vid_out: (B, nt×S, d)                                              │
    │    txt_out: (B, L, d)                                                  │
    │                                                                        │
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
        # vid_spatial意思是把视频patch重组为 (B*nt, S, d)，即每一帧作为一个样本，方便每一帧与文本做联合注意力
        vid_spatial = vid.reshape(B, nt, S, d).reshape(B * nt, S, d)

        # Step 1.2: 文本特征扩张
        # 输入 txt: (B, L, d)
        #   - L: 文本 token 数（如256）
        #   - 文本特征扩张为 (B*nt, L, d)，因为每一帧都需要与相同文本互动
        # 做法体现了「空间」在于，空间注意力是对每一帧（时间patch）上的所有空间位置（S=nh*nw）和文本token（L）联合进行的，
        # 因此要把文本特征复制成与每一帧对应，然后空间联合。每帧对应S个patch，所有帧文本都一样——文本先扩张到 (B, nt, L, d)，
        # 再展平成 (B*nt, L, d)，便于和每一帧的所有空间patch一起送入空间联合注意力。
        txt_spatial = txt.unsqueeze(
            1).expand(-1, nt, -1, -1).reshape(B * nt, L, d)

        # Step 1.3: 分别生成Q、K、V（注意力的查询/键/值）
        # 生成每一帧空间patch的QKV
        vq, vk, vv = [self._reshape_heads(
            x) for x in self.vid_qkv(vid_spatial).chunk(3, dim=-1)]
        # 生成文本token的QKV
        tq, tk, tv = [self._reshape_heads(
            x) for x in self.txt_qkv(txt_spatial).chunk(3, dim=-1)]

        # Step 1.4: 拼接视频和文本QKV，做空间+文本联合注意力
        # 维度说明
        #   vq: (B*nt, n_heads, S, d_k)
        #   tq: (B*nt, n_heads, L, d_k)
        # 合并后
        #   q/k/v: (B*nt, n_heads, S+L, d_k)
        q = torch.cat([vq, tq], dim=2)
        k = torch.cat([vk, tk], dim=2)
        v = torch.cat([vv, tv], dim=2)

        # Step 1.5: 进行 attention 运算
        # 输出为 (B*nt, n_heads, S+L, d_k)
        out = self._attention(q, k, v)
        # 还原回 (B*nt, S+L, d)
        out = out.transpose(1, 2).reshape(B * nt, S + L, -1)

        # Step 1.6: 拆分视频patch输出和文本输出
        # 前S为视频、后L为文本
        vid_s_out = self.vid_spatial_out(out[:, :S])   # (B*nt, S, d)
        txt_s_out = self.txt_out(out[:, S:])           # (B*nt, L, d)

        # Step 1.7: 视频输出还原
        # 把每帧的结果还原回原始batch和序列
        vid_s_out = vid_s_out.reshape(B, nt, S, d).reshape(B, nt * S, d)

        # Step 1.8: 文本输出聚合
        # 文本在每一帧都有输出，聚合成单一序列（在nt方向求平均）
        txt_s_out = txt_s_out.reshape(B, nt, L, d).mean(dim=1)

        # ========== Step 2: 时间注意力（Temporal Attention） ==========
        # Step 2.1: 空间残差加回原始输入
        vid_after_spatial = vid + vid_s_out

        # Step 2.2: 重组为(位置×batch)的时间序列
        # (B, nt*S, d) → (B, nt, S, d) → (B*S, nt, d)
        vid_temporal = (
            vid_after_spatial.reshape(B, nt, S, d)
            .permute(0, 2, 1, 3)                       # (B, S, nt, d)
            .reshape(B * S, nt, d)
        )

        # Step 2.3: 时间 self-attention 的QKV生成
        # 只对时间序列做注意力
        tq, tk, tv = [self._reshape_heads(x) for x in self.temporal_qkv(
            vid_temporal).chunk(3, dim=-1)]
        # 时间轴 attention
        t_out = self._attention(tq, tk, tv)    # (B*S, n_heads, nt, d_k)
        t_out = t_out.transpose(1, 2).reshape(B * S, nt, d)

        # Step 2.4: 还原时间信息至原始形状
        # (B*S, nt, d) → (B, S, nt, d) → (B, nt, S, d) → (B, nt*S, d)
        t_out = t_out.reshape(B, S, nt, d).permute(
            0, 2, 1, 3).reshape(B, nt * S, d)
        vid_out = self.temporal_out(t_out)

        # Step 3: 最终视频输出 = 空间分支残差 + 时间分支残差
        vid_final = vid_s_out + vid_out

        # Step 4: 返回最终的视频特征和文本特征（后续还会走残差、FFN等）
        return vid_final, txt_s_out


# ═══════════════════════════════════════════════════════════════════════════
# 4. VideoMMDiTBlock — 双流视频 MM-DiT 块
# ═══════════════════════════════════════════════════════════════════════════
class VideoMMDiTBlock(nn.Module):
    """
    双流视频 MM-DiT 块。

    与 mmdit.py 的 MMDiTBlock 对比:
      MMDiTBlock:      Joint Attention (全空间, 无时间)
      VideoMMDiTBlock: ST Joint Attention (空间联合 + 时间分离)

    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  输入: vid (B, nt×S, d), txt (B, L, d), t_emb (B, d)                │
    │                                                                        │
    │  ── 条件调制 ──                                                       │
    │  t_emb → vid_mod → [γ1, β1, α1, γ2, β2, α2]  (视频用)             │
    │  t_emb → txt_mod → [γ1, β1, α1, γ2, β2, α2]  (文本用)             │
    │                                                                        │
    │  ── 时空联合注意力 ──                                                 │
    │  vid_h = γ1_v · RMSNorm(vid) + β1_v                                  │
    │  txt_h = γ1_t · RMSNorm(txt) + β1_t                                  │
    │        ↓                                                               │
    │  STJointAttention(vid_h, txt_h, nt, nh, nw)                          │
    │    Step 1: 每帧空间 + 文本 → Joint Attn (双向)                       │
    │    Step 2: 每个位置跨帧 → Temporal Attn                              │
    │        ↓                                                               │
    │  vid = vid + α1_v · vid_attn_out                                      │
    │  txt = txt + α1_t · txt_attn_out                                      │
    │                                                                        │
    │  ── 各自 FFN ──                                                       │
    │  (与 mmdit.py 的 MMDiTBlock 完全相同)                                 │
    │                                                                        │
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
        vid:   (B, nt*S, d)  — 视频 patch
        txt:   (B, L, d)     — 文本 token
        t_emb: (B, d)        — 时间步条件
        nt, nh, nw: 时空 patch 网格

        返回: (vid, txt)
        """
        # 调制参数
        v_γ1, v_β1, v_α1, v_γ2, v_β2, v_α2 = self.vid_mod(t_emb)
        t_γ1, t_β1, t_α1, t_γ2, t_β2, t_α2 = self.txt_mod(t_emb)

        # === 时空联合注意力 ===
        vid_h = v_γ1 * self.vid_norm1(vid) + v_β1
        txt_h = t_γ1 * self.txt_norm1(txt) + t_β1
        vid_attn, txt_attn = self.st_joint_attn(vid_h, txt_h, nt, nh, nw)
        vid = vid + v_α1 * vid_attn
        txt = txt + t_α1 * txt_attn

        # === 各自 FFN ===
        vid_h = v_γ2 * self.vid_norm2(vid) + v_β2
        vid = vid + v_α2 * self.vid_ffn(vid_h)

        txt_h = t_γ2 * self.txt_norm2(txt) + t_β2
        txt = txt + t_α2 * self.txt_ffn(txt_h)

        return vid, txt


# ═══════════════════════════════════════════════════════════════════════════
# 5. VideoSingleStreamSTBlock — 单流时空注意力块
# ═══════════════════════════════════════════════════════════════════════════
class VideoSingleStreamSTBlock(nn.Module):
    """
    单流视频 MM-DiT 块 — FLUX 风格, 但加入时间注意力。

    与 mmdit.py 的 SingleStreamBlock 对比:
      SingleStreamBlock:        图文拼成一个序列做全局注意力
      VideoSingleStreamSTBlock: 图文拼接 → 空间注意力 → 视频做时间注意力

    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  输入: x = [vid_tokens | txt_tokens]  (B, nt*S + L, d)               │
    │                                                                        │
    │  Step 1: 空间注意力                                                   │
    │    每帧: [vid_frame_t | txt] 做 self-attention (共享参数)             │
    │    → 视频每帧和文本交互 + 帧内空间特征提取                            │
    │                                                                        │
    │  Step 2: 时间注意力                                                   │
    │    只对视频部分: 每个空间位置跨帧做 self-attention                    │
    │    → 时间连续性建模                                                   │
    │                                                                        │
    │  并行 FFN (FLUX 风格): 与注意力同时计算                               │
    │                                                                        │
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
        B = x.size(0)
        S = nh * nw
        N_vid = nt * S
        d = self.d_model

        gamma, beta, alpha = self.mod(t_emb)
        h = gamma * self.norm(x) + beta

        # 分离视频和文本
        vid_h = h[:, :N_vid]  # (B, nt*S, d)
        txt_h = h[:, N_vid:]  # (B, L, d)

        # ========== 空间注意力 + 并行 FFN ==========
        # 视频按帧拆分
        vid_per_frame = vid_h.reshape(B, nt, S, d).reshape(B * nt, S, d)
        txt_expanded = (
            txt_h.unsqueeze(1).expand(-1, nt, -1, -1).reshape(B * nt, n_txt, d)
        )
        frame_seq = torch.cat(
            [vid_per_frame, txt_expanded], dim=1)  # (B*nt, S+L, d)

        qkv_ffn = self.qkv_ffn_in(frame_seq)
        qkv, ffn_in = qkv_ffn.split([3 * d, self.d_ff], dim=-1)
        q, k, v = qkv.chunk(3, dim=-1)

        N_frame = S + n_txt
        q = q.reshape(B * nt, N_frame, self.n_heads, self.d_k).transpose(1, 2)
        k = k.reshape(B * nt, N_frame, self.n_heads, self.d_k).transpose(1, 2)
        v = v.reshape(B * nt, N_frame, self.n_heads, self.d_k).transpose(1, 2)

        attn_out = self._attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B * nt, N_frame, d)
        attn_out = self.attn_out(attn_out)

        ffn_out = self.ffn_out(self.dropout_layer(self.ffn_act(ffn_in)))

        spatial_out = attn_out + ffn_out  # (B*nt, S+L, d)

        # 分回视频和文本
        vid_s_out = spatial_out[:, :S].reshape(B, nt * S, d)
        txt_s_out = spatial_out[:, S:].reshape(B, nt, n_txt, d).mean(dim=1)

        # ========== 时间注意力（只对视频）==========
        vid_after = x[:, :N_vid] + alpha[:, :, :d] * vid_s_out  # 先做空间残差
        vid_t = self.temporal_norm(vid_after)
        vid_t = vid_t.reshape(B, nt, S, d).permute(
            0, 2, 1, 3).reshape(B * S, nt, d)

        tq, tk, tv = self.temporal_qkv(vid_t).chunk(3, dim=-1)
        tq = tq.reshape(B * S, nt, self.n_heads, self.d_k).transpose(1, 2)
        tk = tk.reshape(B * S, nt, self.n_heads, self.d_k).transpose(1, 2)
        tv = tv.reshape(B * S, nt, self.n_heads, self.d_k).transpose(1, 2)
        t_out = self._attention(tq, tk, tv)
        t_out = t_out.transpose(1, 2).reshape(B * S, nt, d)
        t_out = self.temporal_out(t_out)
        t_out = t_out.reshape(B, S, nt, d).permute(
            0, 2, 1, 3).reshape(B, nt * S, d)

        vid_final = vid_after + t_out

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
        gamma, beta = self.mod(t_emb)
        x = gamma * self.norm(x) + beta
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
    """
    这是来自 FLUX/SD3 的设计思路，后续被 HunyuanVideo、Wan 等采纳：

    双流层（前8层）— 先各自长本事

    视频和文本是两种完全不同的模态：

    视频：空间纹理、运动轨迹、色彩分布
    文本：语义概念、语法结构
    如果一开始就强行混在一起，两种模态会互相干扰，谁也学不好。双流层让视频和文本各有独立的 norm / FFN / modulation 参数，通过 STJointAttention 做信息交互，但各自保留自己的特征提取能力。

    类比：两个人先各自做功课，再交流笔记，而不是一开始就一个人替另一个人写作业。

    单流层（后8层）— 再深度融合

    经过双流层后，视频和文本已经有了各自良好的表示，这时需要深层对齐——让视频特征精确对应文本语义（"橘猫"这个词要精确对应画面中猫的位置）。单流层把两者拼接成一个序列、共享参数，强制做更紧密的特征融合。

    类比：两人各自研究完后，坐到同一张桌子前共同完成最后的方案。

    为什么不全用双流？ 参数太多，双流两套参数代价大。后8层模态差异已经缩小，没必要再维护两套。

    为什么不全用单流？ 前期视频和文本统计特性差异大，共享参数会导致一方被另一方"绑架"，学不好基础特征。
    推理时双流和单流的分工更直观：

双流层（前8层）— 理解"你要什么"和"现在有什么"

输入是纯噪声 z_t + 文本"一只橘猫在草地上从左走到右"。此时：

文本流：提取语义关键信息——"橘猫""草地""从左到右"
视频流：在噪声中找与文本对应的结构线索
两套参数各司其职，文本不用被迫去理解像素纹理，视频不用被迫去理解语法。通过 STJointAttention 交互，文本告诉视频"往猫的方向走"，视频告诉文本"我目前这个区域可能是猫的轮廓"。

单流层（后8层）— 精准对齐语义和画面

双流层已经让噪声大致成型（能看出猫的形状、草地的颜色），但细节可能不对——猫可能朝右但文本说"从左走到右"。单流层把视频和文本拼在一起共享参数，做精确对齐：

"橘猫"这个词必须精确锚定到画面中猫的位置
"从左到右"必须精确控制运动方向
"草地"必须精确对应绿色区域
推理时的直觉：


t=1.0 纯噪声  → 双流层: 文本指引大方向，视频找大致结构
                 单流层: 微调对齐

t=0.5 半成型  → 双流层: 文本和视频各自深化理解
                 单流层: 精确绑定词和区域

t=0.0 接近完成 → 双流层: 稳定整体结构
                 单流层: 最终语义-像素对齐
一句话总结：双流是"各干各的+交流"，保证基础特征学得好；单流是"一起干"，保证最终语义和画面对得上。
    """

    def __init__(self, cfg: VideoMMDiTConfig):
        super().__init__()
        self.cfg = cfg

        # --- Embedding ---
        self.patch_embed = VideoPatchEmbedding3D(cfg)
        self.t_embed = TimestepEmbedding(cfg.d_model)
        self.text_proj = nn.Linear(cfg.text_d_model, cfg.d_model)

        self.vid_pos_embed = nn.Parameter(
            torch.zeros(1, cfg.num_patches, cfg.d_model))
        self.txt_pos_embed = nn.Parameter(
            torch.zeros(1, cfg.text_max_len, cfg.d_model))

        # --- 双流层 ---
        self.double_blocks = nn.ModuleList(
            [
                VideoMMDiTBlock(cfg.d_model, cfg.n_heads,
                                cfg.d_ff, cfg.dropout)
                for _ in range(cfg.n_double_layers)
            ]
        )

        # --- 单流层 ---
        self.single_blocks = nn.ModuleList(
            [
                VideoSingleStreamSTBlock(
                    cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout
                )
                for _ in range(cfg.n_single_layers)
            ]
        )

        # --- 输出 ---
        self.final_layer = FinalLayer(cfg.d_model, cfg.patch_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.vid_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.txt_pos_embed, std=0.02)

    def unpatchify3d(self, x: torch.Tensor) -> torch.Tensor:
        """
        3D unpatchify: patch 序列 → 视频 latent。

        x: (B, nt*nh*nw, patch_dim) → (B, C, T, H, W)
        """
        c = self.cfg
        x = x.reshape(
            -1, c.nt, c.nh, c.nw, c.patch_t, c.patch_h, c.patch_w, c.in_channels
        )
        # (B, nt, nh, nw, pt, ph, pw, C)
        # → (B, C, nt, pt, nh, ph, nw, pw)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        # → (B, C, nt*pt, nh*ph, nw*pw) = (B, C, T, H, W)
        x = x.reshape(-1, c.in_channels, c.latent_t, c.latent_h, c.latent_w)
        return x

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        z_t:      (B, C, T, H, W)          — 加噪的视频 latent
        t:        (B,)                       — 扩散时间步
        text_emb: (B, text_len, text_dim)   — 文本编码

        返回:     (B, C, T, H, W)          — 预测的噪声 ε_θ
        """
        cfg = self.cfg

        # --- Embedding ---
        vid = self.patch_embed(z_t)  # (B, N_vid, d)
        t_emb = self.t_embed(t)  # (B, d)
        txt = self.text_proj(text_emb)  # (B, L, d)

        vid = vid + self.vid_pos_embed[:, : vid.size(1)]
        txt = txt + self.txt_pos_embed[:, : txt.size(1)]

        n_txt = txt.size(1)

        # --- 为什么有双流层（视频流+文本流）和单流层（合流） ---
        # 双流层：先分别处理视频patch和文本，让它们能各自提取适合自己的特征，并用联合注意力交互（如视频时空联合、文本联合）。
        #        这样能最大化保留各自 modality 的特殊结构，同时获得早期的信息交互。
        for block in self.double_blocks:
            vid, txt = block(vid, txt, t_emb, cfg.nt, cfg.nh, cfg.nw)

        # 单流层：将视频和文本 patch 序列拼接成一个“合流”序列，再用统一的 Transformer 层做更深层的信息融合。
        #        这样做可以强化 modality 间的高层关联，利于最终目标（如生成视频、对齐语义）。
        x = torch.cat([vid, txt], dim=1)
        for block in self.single_blocks:
            x = block(x, t_emb, cfg.nt, cfg.nh, cfg.nw, n_txt)

        # --- 取视频部分 → 输出 ---
        vid_out = x[:, : cfg.num_patches]
        vid_out = self.final_layer(vid_out, t_emb)
        noise_pred = self.unpatchify3d(vid_out)

        # mmdit在这里forward结束，返回预测的噪声；潜空间视频的输出就是noise_pred（(B, C, T, H, W)），即解码后的潜空间视频
        return noise_pred
