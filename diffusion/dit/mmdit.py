"""
MM-DiT (Multi-Modal Diffusion Transformer) 模块 —— 学习用实现
=============================================================

本文件实现了 Stable Diffusion 3 / FLUX 使用的 MM-DiT 架构。
MM-DiT 是 DiT 的下一代演进，核心创新是 **双流联合注意力**。

本文件实现了：
  1. MMDiTConfig           — 配置
  2. Modulation            — adaLN 调制层（为双流各自生成调制参数）
  3. JointAttention        — 联合注意力（MM-DiT 核心创新）
  4. MMDiTBlock            — 双流 MM-DiT 块（SD3 前 ~38 层使用）
  5. SingleStreamBlock     — 单流块（FLUX 后半段使用）
  6. MMDiT                 — 完整模型（双流块 + 单流块 + 去噪头）

=== MM-DiT vs DiT: 文本条件注入方式的进化 ===

  ┌────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │  DiT (ditblock.py):                                                    │
  │    文本通过 Cross-Attention 注入                                       │
  │    图像 patch 做 Q，文本做 KV → 单向：图像查询文本                    │
  │                                                                        │
  │    image ──→ Self-Attn ──→ Cross-Attn(Q=img, KV=text) ──→ FFN        │
  │                                 ↑                                      │
  │    text ─────────────────────────┘  (文本只被读取，不更新)             │
  │                                                                        │
  │    问题:                                                               │
  │      - 文本表示是固定的，不随图像特征变化而调整                        │
  │      - 图文交互是单向的（图→文），文本不能根据图像反馈调整             │
  │      - Cross-Attention 是额外的计算开销                                │
  │                                                                        │
  │  ──────────────────────────────────────────────────────────────────    │
  │                                                                        │
  │  MM-DiT (本文件):                                                      │
  │    文本和图像在同一个注意力中联合处理                                  │
  │    两个 modality 的 token 拼在一起做注意力，互相都能看到对方           │
  │                                                                        │
  │    image ──→ [img_Q, img_K, img_V] ─┐                                 │
  │                                      ├→ Concat → Joint Attention      │
  │    text  ──→ [txt_Q, txt_K, txt_V] ─┘       ↓                        │
  │                                        ┌─────┴─────┐                  │
  │                                        ↓           ↓                   │
  │                                    img_out     txt_out                 │
  │                                        ↓           ↓                   │
  │                                    img_FFN     txt_FFN                 │
  │                                                                        │
  │    优势:                                                               │
  │      ✓ 双向交互: 图像和文本互相影响，文本也会被更新                   │
  │      ✓ 无额外 Cross-Attn: 融合在一次注意力中完成                      │
  │      ✓ 更强的对齐: 文本 token 和图像 patch 在同一空间竞争注意力       │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘

=== 双流 vs 单流 ===

  ┌────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │  双流 (MMDiTBlock, SD3 / FLUX 前半段):                                 │
  │    图像和文本各有独立的 QKV 投影、归一化、FFN                          │
  │    只在 Attention 计算时把 KV 拼在一起                                 │
  │    → 每个 modality 保持自己的"风格"，但能互相看到对方                 │
  │                                                                        │
  │      img: ──→ img_norm ──→ img_QKV ──┐                                │
  │                                       ├→ Joint Attn → split           │
  │      txt: ──→ txt_norm ──→ txt_QKV ──┘         ↓                     │
  │                                          ┌──────┴──────┐              │
  │                                          ↓             ↓               │
  │                                      img_FFN       txt_FFN            │
  │                                                                        │
  │  单流 (SingleStreamBlock, FLUX 后半段):                                │
  │    图像和文本的 token 直接拼成一个序列                                 │
  │    共享同一套 QKV 投影、归一化、FFN                                    │
  │    → 更紧密的融合，参数更少                                           │
  │                                                                        │
  │      [img_tokens | txt_tokens]                                        │
  │        ──→ shared_norm ──→ shared_QKV ──→ Self-Attn ──→ shared_FFN   │
  │                                                                        │
  │  FLUX 的设计:                                                          │
  │    前 19 层: 双流 MMDiTBlock（图文有独立参数，保持各自特点）           │
  │    后 38 层: 单流 SingleStreamBlock（图文融合，共享参数，更高效）      │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘

=== 为什么 MM-DiT 比 DiT + Cross-Attn 效果更好？ ===

  1. 双向信息流:
     DiT: text→img 单向（文本给图像信息，但文本本身不变）
     MMDiT: text↔img 双向（文本表示也会根据图像特征调整）
     → 文本理解和图像生成更紧密协调

  2. 注意力竞争:
     所有 token（文本+图像）在同一个 softmax 里竞争注意力权重
     → 模型必须学会哪些文本 token 和哪些图像区域最相关
     → 更精确的文本-图像对齐

  3. 无额外开销:
     DiT: Self-Attn(图像) + Cross-Attn(图像×文本) = 两次注意力
     MMDiT: Joint-Attn(图像+文本) = 一次注意力（虽然序列更长）
     → 实际中 文本token很短(77~256)，图像token很长(4096+)
     → 拼在一起增加的计算量 << 单独做一次 Cross-Attn

参考:
  - "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
    (Esser et al., 2024) — SD3 论文，首次提出 MM-DiT
  - "FLUX" (Black Forest Labs, 2024) — 双流+单流混合架构
  - "Stable Diffusion 3 Medium" (Stability AI, 2024)
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
class MMDiTConfig:
    """
    MM-DiT 配置。

    默认按 SD3-Medium / FLUX 类似规格设置（图像生成，非视频）。

    === SD3 / FLUX 参数对比 ===

    ┌──────────────────────────────────────────────────────────────────────┐
    │  模型              │ d_model │ heads │ 双流层 │ 单流层 │ 参数量    │
    │ ────────────────── │ ─────── │ ───── │ ────── │ ────── │ ───────── │
    │  SD3-Medium        │  1536   │  24   │   24   │   0    │  2B       │
    │  FLUX.1-dev        │  3072   │  24   │   19   │   38   │  12B      │
    │  FLUX.1-schnell    │  3072   │  24   │   19   │   38   │  12B      │
    │  本配置(学习用)     │  1536   │  24   │   12   │   12   │  ~2B      │
    └──────────────────────────────────────────────────────────────────────┘
    """
    # --- 图像 ---
    in_channels: int = 16             # latent 通道数（SD3 的 VAE 输出 16 通道）
    image_size: int = 64              # latent 空间的尺寸（原图 512/8=64）
    patch_size: int = 2               # 2D patch 大小 → 64/2 = 32 → 32²=1024 个 patch

    # --- Transformer ---
    d_model: int = 1536               # 隐藏维度（SD3-Medium 用 1536）
    n_heads: int = 24                 # 注意力头数（1536/24=64 per head）
    d_ff: int = 1536 * 4              # FFN 中间维度
    n_double_layers: int = 12         # 双流 MMDiTBlock 层数
    n_single_layers: int = 12         # 单流 SingleStreamBlock 层数
    dropout: float = 0.0              # SD3/FLUX 不用 dropout

    # --- 文本 ---
    text_max_len: int = 256           # 文本 token 最大长度
    text_d_model: int = 4096          # 文本编码器的维度（T5-XXL = 4096）
    # SD3 用两个文本编码器:
    #   CLIP-L (768d) + CLIP-G (1280d) + T5-XXL (4096d)
    #   拼接后投影到 d_model
    # 这里简化为单个 T5-XXL

    # --- 扩散 ---
    num_timesteps: int = 1000         # 扩散步数

    @property
    def num_patches(self) -> int:
        """图像 patch 总数。"""
        return (self.image_size // self.patch_size) ** 2  # 32² = 1024

    @property
    def patch_dim(self) -> int:
        """每个 patch 展平后的维度。"""
        return self.in_channels * self.patch_size ** 2  # 16 * 4 = 64

    @property
    def total_layers(self) -> int:
        return self.n_double_layers + self.n_single_layers


# ═══════════════════════════════════════════════════════════════════════════
# 1. 基础组件
# ═══════════════════════════════════════════════════════════════════════════

class TimestepEmbedding(nn.Module):
    """
    时间步编码 — 与 ditblock.py 中的实现相同。

    t (整数) → 正弦编码 → MLP → t_emb (向量)
    """

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
        return self.mlp(self.sinusoidal_encoding(t))


class PatchEmbedding2D(nn.Module):
    """
    2D Patch Embedding — 把图像 latent 切成不重叠的 patch。

    与 ViT 的 PatchEmbedding 完全相同，只是输入是 latent（16 通道）而非 RGB（3 通道）。

    (B, 16, 64, 64) → Conv2d(kernel=2, stride=2) → (B, 1536, 32, 32)
    → flatten → (B, 1024, 1536)
    """

    def __init__(self, cfg: MMDiTConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            cfg.in_channels, cfg.d_model,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, num_patches, d_model)"""
        x = self.proj(x)             # (B, d_model, H/P, W/P)
        return x.flatten(2).transpose(1, 2)  # (B, N, d_model)


class RMSNorm(nn.Module):
    """
    RMS Normalization — SD3/FLUX 使用的归一化方式。

    === RMSNorm vs LayerNorm ===

    ┌────────────────────────────────────────────────────────────────┐
    │                                                                │
    │  LayerNorm:                                                    │
    │    y = (x - mean) / std · γ + β                               │
    │    需要计算均值和标准差，有 γ 和 β 两组参数                    │
    │                                                                │
    │  RMSNorm:                                                      │
    │    y = x / RMS(x) · γ                                         │
    │    RMS(x) = sqrt(mean(x²))                                    │
    │    只需计算 RMS，没有 β（偏移），更快更简洁                    │
    │                                                                │
    │  为什么 FLUX/SD3 用 RMSNorm:                                   │
    │    1. 计算更快（少一步减均值）                                  │
    │    2. Llama/Gemma 等 LLM 已验证 RMSNorm 效果不输 LayerNorm    │
    │    3. 与 adaLN 配合时，反正 γ/β 都由条件动态生成              │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ═══════════════════════════════════════════════════════════════════════════
# 2. Modulation — adaLN 调制参数生成（双流版本）
# ═══════════════════════════════════════════════════════════════════════════
class Modulation(nn.Module):
    """
    条件调制层 — 从时间步嵌入生成归一化调制参数。

    与 ditblock.py 中的 AdaLayerNorm 类似，但:
      - 这里只负责生成参数，不做归一化本身
      - 可生成不同数量的调制向量（双流块需要 6 个，单流块需要 3 个）

    === adaLN-Zero 回顾 ===

      condition → Linear → [γ, β, α]
      γ: scale — 缩放归一化后的特征
      β: shift — 平移归一化后的特征
      α: gate  — 门控残差连接（初始化为 0 → 训练初期为恒等映射）

    参数:
        d_model (int):  特征维度
        n_modulations (int): 输出的调制向量数量
            双流块: 6 个 (γ1, β1, α1 给注意力; γ2, β2, α2 给 FFN)
            单流块: 3 个 (γ, β, α 共用)
    """

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
        cond: (B, d_model)
        返回: n_modulations 个 (B, 1, d_model) 张量的列表
        """
        params = self.linear(cond)  # (B, n_mod * d_model)
        return [p.unsqueeze(1) for p in params.chunk(self.n_modulations, dim=-1)]


# ═══════════════════════════════════════════════════════════════════════════
# 3. JointAttention — 联合注意力（MM-DiT 核心创新）
# ═══════════════════════════════════════════════════════════════════════════
class JointAttention(nn.Module):
    """
    联合注意力 — MM-DiT 的核心创新。

    图像和文本各自生成 QKV，然后在 K/V 维度拼接，共享注意力计算。

    === 与 DiT Cross-Attention 的对比 ===

    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  DiT Cross-Attention:                                                  │
    │    Q = img_proj(img)            ← 只有图像做 query                    │
    │    K = text_proj_k(text)        ← 文本做 key                          │
    │    V = text_proj_v(text)        ← 文本做 value                        │
    │    out = softmax(QK^T/√d) · V   ← 图像单向查询文本                   │
    │    只更新图像，文本不变                                                │
    │                                                                        │
    │  MM-DiT Joint Attention:                                               │
    │    img_Q, img_K, img_V = img_proj(img)    ← 图像生成自己的 QKV       │
    │    txt_Q, txt_K, txt_V = txt_proj(text)   ← 文本生成自己的 QKV       │
    │                                                                        │
    │    Q = [img_Q ; txt_Q]   ← 拼接                                      │
    │    K = [img_K ; txt_K]   ← 拼接                                      │
    │    V = [img_V ; txt_V]   ← 拼接                                      │
    │                                                                        │
    │    out = softmax(QK^T/√d) · V                                         │
    │    img_out, txt_out = split(out)  ← 分回各自的部分                    │
    │                                                                        │
    │    图像和文本都被更新！双向交互！                                      │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘

    === 注意力矩阵的结构 ===

    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  假设 img 有 1024 个 token, text 有 256 个 token                      │
    │  拼接后序列长度 = 1280                                                │
    │                                                                        │
    │  注意力矩阵 (1280 × 1280):                                           │
    │                                                                        │
    │           K: img(1024)    txt(256)                                     │
    │          ┌──────────────┬──────────┐                                  │
    │  Q: img  │  img↔img     │ img→txt  │  ← 图像 token 能看到所有 token  │
    │   (1024) │  (自注意力)  │ (跨模态) │                                  │
    │          ├──────────────┼──────────┤                                  │
    │  Q: txt  │  txt→img     │ txt↔txt  │  ← 文本 token 也能看到所有 token│
    │   (256)  │  (跨模态)    │ (自注意力)│                                  │
    │          └──────────────┴──────────┘                                  │
    │                                                                        │
    │  与 DiT 的区别:                                                        │
    │    DiT 的 Cross-Attn 只有右上角的 "img→txt" 部分                     │
    │    MM-DiT 是完整的 1280×1280 矩阵，四个象限都有                       │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘

    参数:
        d_model (int): 特征维度
        n_heads (int): 注意力头数
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 图像流的 QKV 投影
        self.img_qkv = nn.Linear(d_model, 3 * d_model)
        self.img_out = nn.Linear(d_model, d_model)

        # 文本流的 QKV 投影（独立参数！）
        self.txt_qkv = nn.Linear(d_model, 3 * d_model)
        self.txt_out = nn.Linear(d_model, d_model)

    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, d_model) → (B, n_heads, N, d_k)"""
        B, N, _ = x.shape
        return x.reshape(B, N, self.n_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        img: (B, N_img, d_model)  — 图像 token
        txt: (B, N_txt, d_model)  — 文本 token

        返回: (img_out, txt_out)
            img_out: (B, N_img, d_model)
            txt_out: (B, N_txt, d_model)
        """
        B, N_img, _ = img.shape
        N_txt = txt.size(1)

        # 各自生成 QKV
        img_qkv = self.img_qkv(img).chunk(3, dim=-1)  # 3 × (B, N_img, d)
        txt_qkv = self.txt_qkv(txt).chunk(3, dim=-1)  # 3 × (B, N_txt, d)

        # reshape 为多头格式
        img_q, img_k, img_v = [self._reshape_for_heads(x) for x in img_qkv]
        txt_q, txt_k, txt_v = [self._reshape_for_heads(x) for x in txt_qkv]

        # === 核心: 在序列维拼接 QKV ===
        # (B, n_heads, N_img+N_txt, d_k)
        q = torch.cat([img_q, txt_q], dim=2)
        k = torch.cat([img_k, txt_k], dim=2)
        v = torch.cat([img_v, txt_v], dim=2)

        # 标准缩放点积注意力（所有 token 互相可见）
        scale = self.d_k ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, heads, N_total, N_total)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, heads, N_total, d_k)

        # reshape 回 (B, N_total, d_model)
        out = out.transpose(1, 2).reshape(B, N_img + N_txt, -1)

        # === 分回各自的部分 ===
        img_out = self.img_out(out[:, :N_img])
        txt_out = self.txt_out(out[:, N_img:])

        return img_out, txt_out


# ═══════════════════════════════════════════════════════════════════════════
# 4. MMDiTBlock — 双流 MM-DiT 块
# ═══════════════════════════════════════════════════════════════════════════
class MMDiTBlock(nn.Module):
    """
    双流 MM-DiT 块 — Stable Diffusion 3 / FLUX 的核心块。

    === 数据流 ===

    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  输入: img (B, N_img, d), txt (B, N_txt, d), t_emb (B, d)            │
    │                                                                        │
    │  ── 条件调制 ──                                                       │
    │  t_emb → img_mod → [γ1_i, β1_i, α1_i, γ2_i, β2_i, α2_i]  (图像用) │
    │  t_emb → txt_mod → [γ1_t, β1_t, α1_t, γ2_t, β2_t, α2_t]  (文本用) │
    │                                                                        │
    │  ── 联合注意力 ──                                                     │
    │  img_h = γ1_i · RMSNorm(img) + β1_i     ← 各自调制                  │
    │  txt_h = γ1_t · RMSNorm(txt) + β1_t     ← 各自调制                  │
    │              ↓                 ↓                                       │
    │         JointAttention(img_h, txt_h)     ← 联合注意力                │
    │              ↓                 ↓                                       │
    │  img = img + α1_i · img_attn_out         ← 各自门控残差              │
    │  txt = txt + α1_t · txt_attn_out         ← 各自门控残差              │
    │                                                                        │
    │  ── 各自 FFN ──                                                       │
    │  img_h = γ2_i · RMSNorm(img) + β2_i                                  │
    │  img = img + α2_i · FFN_img(img_h)                                    │
    │                                                                        │
    │  txt_h = γ2_t · RMSNorm(txt) + β2_t                                  │
    │  txt = txt + α2_t · FFN_txt(txt_h)                                    │
    │                                                                        │
    │  输出: img (B, N_img, d), txt (B, N_txt, d)                          │
    │                                                                        │
    │  关键: 图像和文本有独立的归一化、调制、FFN                            │
    │       只在注意力计算时"见面"                                          │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()

        # 图像流的调制和归一化
        self.img_mod = Modulation(d_model, n_modulations=6)
        self.img_norm1 = RMSNorm(d_model)
        self.img_norm2 = RMSNorm(d_model)
        self.img_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(approximate="tanh"),  # SD3/FLUX 用 GELU 而非 SiLU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # 文本流的调制和归一化（独立参数！）
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

        # 共享的联合注意力
        self.joint_attn = JointAttention(d_model, n_heads)

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        img:   (B, N_img, d_model) — 图像 token
        txt:   (B, N_txt, d_model) — 文本 token
        t_emb: (B, d_model)        — 时间步条件

        返回: (img, txt) — 更新后的图像和文本表示
        """
        # 生成调制参数
        img_γ1, img_β1, img_α1, img_γ2, img_β2, img_α2 = self.img_mod(t_emb)
        txt_γ1, txt_β1, txt_α1, txt_γ2, txt_β2, txt_α2 = self.txt_mod(t_emb)

        # === 联合注意力 ===
        # 各自调制
        img_h = img_γ1 * self.img_norm1(img) + img_β1
        txt_h = txt_γ1 * self.txt_norm1(txt) + txt_β1
        # 联合注意力计算
        img_attn, txt_attn = self.joint_attn(img_h, txt_h)
        # 门控残差
        img = img + img_α1 * img_attn
        txt = txt + txt_α1 * txt_attn

        # === 各自 FFN ===
        img_h = img_γ2 * self.img_norm2(img) + img_β2
        img = img + img_α2 * self.img_ffn(img_h)

        txt_h = txt_γ2 * self.txt_norm2(txt) + txt_β2
        txt = txt + txt_α2 * self.txt_ffn(txt_h)

        return img, txt


# ═══════════════════════════════════════════════════════════════════════════
# 5. SingleStreamBlock — 单流块（FLUX 后半段使用）
# ═══════════════════════════════════════════════════════════════════════════
class SingleStreamBlock(nn.Module):
    """
    单流块 — 图像和文本 token 拼成一个序列，共享所有参数。

    === 与 MMDiTBlock 的区别 ===

    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  MMDiTBlock (双流):                                                    │
    │    图像和文本有各自的: norm, modulation, FFN, QKV 投影                │
    │    只在注意力时拼接                                                    │
    │    参数量大（两套 FFN + 两套 QKV）                                    │
    │                                                                        │
    │  SingleStreamBlock (单流):                                             │
    │    先把 [img_tokens ; txt_tokens] 拼成一个序列                        │
    │    共享同一套: norm, modulation, QKV, FFN                             │
    │    更紧密融合，参数减半                                               │
    │                                                                        │
    │  FLUX 的混合策略:                                                      │
    │    前 19 层用双流 → 图文各自先建立好表示                              │
    │    后 38 层用单流 → 深度融合，共享参数更高效                          │
    │                                                                        │
    │  类比:                                                                 │
    │    双流 = 两个翻译专家各自阅读中文和英文，在讨论时交流                │
    │    单流 = 一个双语专家直接同时读两种语言                               │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘

    === FLUX 单流块的特殊设计: 并行注意力 ===

    普通 Transformer:  Attn → FFN  (串行)
    FLUX 单流块:       Attn + FFN  (并行！)

    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  串行 (普通):                                                          │
    │    x → norm → Attn → +x → norm → FFN → +x                            │
    │    两步串行，FFN 依赖 Attn 的输出                                     │
    │                                                                        │
    │  并行 (FLUX):                                                          │
    │    x → norm → ┬─ Attn ─┐                                              │
    │               └─ FFN  ─┤                                               │
    │                        +→ +x                                           │
    │    Attn 和 FFN 可以同时计算！GPU 利用率更高                           │
    │                                                                        │
    │  为什么可以并行？                                                      │
    │    实验发现在深层，FFN 和 Attn 的功能已经不那么依赖彼此               │
    │    并行不影响效果，但提升了硬件利用率                                  │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.mod = Modulation(d_model, n_modulations=3)
        self.norm = RMSNorm(d_model)

        # QKV + FFN 并行: 把 QKV 和 FFN 第一层打包成一个大 Linear
        # 输出: 3*d_model (QKV) + d_ff (FFN 第一层)
        self.qkv_ffn_in = nn.Linear(d_model, 3 * d_model + d_ff)
        self.ffn_act = nn.GELU(approximate="tanh")
        self.ffn_out = nn.Linear(d_ff, d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.d_ff = d_ff

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:     (B, N_img + N_txt, d_model) — 拼接的图文序列
        t_emb: (B, d_model)                — 时间步条件

        返回:  (B, N_img + N_txt, d_model)
        """
        B, N, _ = x.shape
        gamma, beta, alpha = self.mod(t_emb)

        h = gamma * self.norm(x) + beta

        # 一次性计算 QKV 和 FFN 输入
        qkv_ffn = self.qkv_ffn_in(h)
        qkv, ffn_in = qkv_ffn.split([3 * self.d_model, self.d_ff], dim=-1)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape 多头
        q = q.reshape(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = k.reshape(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = v.reshape(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # 注意力
        scale = self.d_k ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, -1)
        attn_out = self.attn_out(attn_out)

        # FFN（与注意力并行，输入都是同一个 h）
        ffn_out = self.ffn_out(self.dropout(self.ffn_act(ffn_in)))

        # 合并（并行的 Attn 和 FFN 结果直接相加）
        x = x + alpha * (attn_out + ffn_out)

        return x


# ═══════════════════════════════════════════════════════════════════════════
# 6. FinalLayer — 去噪输出头
# ═══════════════════════════════════════════════════════════════════════════
class FinalLayer(nn.Module):
    """
    最终输出层 — 将 Transformer 特征投影回 patch 像素空间。

    adaLN 调制 → 线性投影 → patch 维度
    """

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
# 7. MMDiT — 完整模型
# ═══════════════════════════════════════════════════════════════════════════
class MMDiT(nn.Module):
    """
    完整的 MM-DiT 模型（图像生成版本）。

    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  输入:                                                                 │
    │    z_t:  (B, 16, 64, 64)  — 加噪后的 latent                          │
    │    t:    (B,)              — 时间步                                    │
    │    text: (B, 256, 4096)   — T5 文本编码                               │
    │                                                                        │
    │  ── Embedding ──                                                      │
    │  z_t → PatchEmbed → (B, 1024, 1536)   img_tokens                     │
    │  t   → TimestepEmbed → (B, 1536)      t_emb                          │
    │  text → TextProj → (B, 256, 1536)     txt_tokens                     │
    │  + 位置编码                                                           │
    │                                                                        │
    │  ── 双流层 × 12 ──                                                    │
    │  img_tokens, txt_tokens = MMDiTBlock(img, txt, t_emb) × 12           │
    │  图文各自有独立参数，只在注意力时联合                                  │
    │                                                                        │
    │  ── 拼接 ──                                                           │
    │  x = [img_tokens ; txt_tokens]  → (B, 1024+256, 1536)               │
    │                                                                        │
    │  ── 单流层 × 12 ──                                                    │
    │  x = SingleStreamBlock(x, t_emb) × 12                                │
    │  图文共享所有参数                                                     │
    │                                                                        │
    │  ── 输出 ──                                                           │
    │  取图像部分 → FinalLayer → (B, 1024, 64)                             │
    │  → unpatchify → (B, 16, 64, 64) — 预测的噪声                        │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, cfg: MMDiTConfig):
        super().__init__()
        self.cfg = cfg

        # --- Embedding ---
        self.patch_embed = PatchEmbedding2D(cfg)
        self.t_embed = TimestepEmbedding(cfg.d_model)
        self.text_proj = nn.Linear(cfg.text_d_model, cfg.d_model)

        # 可学习位置编码
        self.img_pos_embed = nn.Parameter(
            torch.zeros(1, cfg.num_patches, cfg.d_model)
        )
        self.txt_pos_embed = nn.Parameter(
            torch.zeros(1, cfg.text_max_len, cfg.d_model)
        )

        # --- 双流层 ---
        self.double_blocks = nn.ModuleList([
            MMDiTBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_double_layers)
        ])

        # --- 单流层 ---
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_single_layers)
        ])

        # --- 输出 ---
        self.final_layer = FinalLayer(cfg.d_model, cfg.patch_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.img_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.txt_pos_embed, std=0.02)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        把 patch 序列还原为 2D latent。

        x: (B, N, patch_dim) → (B, C, H, W)
        """
        P = self.cfg.patch_size
        H = W = self.cfg.image_size // P
        C = self.cfg.in_channels
        # 这三个操作是把 patch 序列还原成原始 2D 格式的关键步骤：
        # 1. reshape：先将每个 patch 展开还原成 (H, W, P, P, C) 格式，
        #    - H, W: patch 格点数
        #    - P, P: patch 尺寸
        #    - C: 通道数
        x = x.reshape(-1, H, W, P, P, C)
        # 2. permute：交换维度，把 patch 结构排列到图像平面上
        #    - 调整成 (B, C, H, P, W, P)，方便最后一步合并空间维度
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, H, P, W, P)
        # 3. reshape：合并每个 patch 的空间尺寸，还原为标准 (B, C, H*P, W*P) 图像
        x = x.reshape(-1, C, H * P, W * P)
        return x

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        z_t:      (B, C, H, W)            — 加噪的 latent
        t:        (B,)                     — 扩散时间步
        text_emb: (B, text_len, text_dim)  — 文本编码 (如 T5 输出)

        返回:     (B, C, H, W)            — 预测的噪声 ε_θ
        """
        # --- Embedding ---
        img = self.patch_embed(z_t)   # (B, N_img, d)
        t_emb = self.t_embed(t)       # (B, d)
        txt = self.text_proj(text_emb)  # (B, N_txt, d)

        # 加位置编码
        img = img + self.img_pos_embed[:, :img.size(1)]
        txt = txt + self.txt_pos_embed[:, :txt.size(1)]

        # --- 双流层 ---
        for block in self.double_blocks:
            img, txt = block(img, txt, t_emb)

        # --- 拼接 → 单流层 ---
        # -------- 为什么要加一个单流层？--------
        # 在前面的双流（MM-DiT）块中，图像和文本各自归一化、调制和残差 FFN，只在注意力时融合交互，保持各自独立的表征。
        # 在模型后段（如 FLUX 架构），将两种模态拼接为一个 Token 流，经过单流层实现更深层次的语义融合、信息交互，提升模型表达和解码能力。
        # 单流层的本质与传统 Transformer Block 相同，对整个 token 序列（图文已拼接）做统一建模和推理。
        x = torch.cat([img, txt], dim=1)  # (B, N_img + N_txt, d)
        for block in self.single_blocks:
            x = block(x, t_emb)

        # --- 取图像部分 → 输出 ---
        img_out = x[:, :self.cfg.num_patches]
        img_out = self.final_layer(img_out, t_emb)  # (B, N_img, patch_dim)
        noise_pred = self.unpatchify(img_out)

        return noise_pred
