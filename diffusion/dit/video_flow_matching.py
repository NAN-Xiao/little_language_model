"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      VideoFlowMatching —— 视频生成的 Flow Matching 训练与推理框架             ║
╚══════════════════════════════════════════════════════════════════════════════╝

本文件将 Flow Matching 与 VideoMMDiT 对接，构成完整的视频生成管线：

  文字提示词                                            生成的视频
      │                                                     ▲
      ▼                                                     │
  T5-XXL 编码器                                        3D VAE 解码器
      │                                                     │
      │  text_emb (B, L, 4096)          z_0 (B, 4, T, H, W) ← latent 视频
      │                                      ▲
      │                    VideoFlowMatchingSampler
      │                    (Euler ODE 求解，20步)
      │                          │
      └──────────────────────────▼
                           VideoMMDiT
                      (本目录 mmdit-vedio.py)
                      输入: z_t + t + text_emb
                      输出: 速度场 v_θ(z_t, t, text)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【视频 Flow Matching 与图像 Flow Matching 的区别】

  图像 Flow Matching：
    - latent 形状: (B, C, H, W)  → 4维
    - 无文本或文本可选

  视频 Flow Matching：
    - latent 形状: (B, C, T, H, W) → 5维（多了时间轴 T）
    - 文本是必须的（text-to-video）
    - 必须实现 CFG（Classifier-Free Guidance），否则效果很差

  但核心公式完全一样（只是张量多了一维）：
    z_t = (1-t) * z_0  +  t * z_1         (z_0=干净视频, z_1=噪声)
    v_target = z_1 - z_0                  (目标速度，恒定)
    loss = MSE(v_θ(z_t, t, text), v_target)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【Classifier-Free Guidance（CFG）—— 文本控制强度的核心机制】

  问题：如何让生成的视频更贴近文字描述？

  解答：让模型同时学两件事：
    1. 有条件预测（给文字）：v_cond  = model(z_t, t, text_emb)
    2. 无条件预测（不给文字）：v_uncond = model(z_t, t, null_emb)

  推理时，把两者插值：
    v_cfg = v_uncond + cfg_scale × (v_cond - v_uncond)
            ↑ 无条件基础        ↑ 文本引导的额外方向 × 放大系数

  cfg_scale 的含义：
    cfg_scale = 1.0 → 不引导，等同于无条件生成
    cfg_scale = 7.0 → 中等引导，质量和多样性均衡（常用默认值）
    cfg_scale = 15+ → 强引导，严格贴合文字但多样性下降，可能过饱和

  训练时如何支持 CFG？
    随机用概率 p_uncond（如 10%）把文本替换成全零嵌入，
    让模型学会在没有文本时也能生成视频。
    推理时才用 CFG 插值公式。

  可视化（cfg_scale=7.0）：
    v_uncond: 视频会动，但不知道生成什么  ← 基础运动学
    v_cond:   视频按文字描述生成           ← 语义方向
    v_cfg:    = v_uncond + 7*(v_cond - v_uncond)  ← 强化语义，保留运动

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

参考论文:
  - Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
  - Kong et al., "HunyuanVideo", 2024 — 实际使用 Flow Matching + MM-DiT
  - Ho & Salimans, "Classifier-Free Diffusion Guidance", 2022
    https://arxiv.org/abs/2207.12598
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════
# 0. 配置
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class VideoFlowMatchingConfig:
    """视频 Flow Matching 的超参数配置。

    参数:
        num_train_timesteps (int): 训练用的离散时间步总数（用于将浮点 t 映射到
            VideoMMDiT 内部的整数时间嵌入索引）。
        t_sample_mode (str): 训练时 t 的采样分布：
            - "uniform":      t ~ U[0, 1]，均匀采样
            - "logit_normal": sigmoid(N(0,1))，偏重中间 t（SD3/HunyuanVideo 用）
              中间 t ≈ 0.5 最难预测（一半信号一半噪声），多采样对训练更有效
        p_uncond (float): 训练时随机丢弃文本的概率（用于支持 CFG）。
            典型值 0.1（10% 的 batch 不给文本，用全零嵌入替代）。
        cfg_scale (float): 推理时 Classifier-Free Guidance 的引导强度。
            典型值 7.0（HunyuanVideo 默认），越大越贴近文字但多样性下降。
        clip_output (bool): 推理完成后是否把 latent 值夹到 [-4, 4]（VAE latent
            的典型值域），防止极端值导致 VAE 解码失败。
    """

    num_train_timesteps: int = 1000
    t_sample_mode: str = "logit_normal"
    p_uncond: float = 0.1
    cfg_scale: float = 7.0
    clip_output: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# 1. 视频 Flow Matching 调度器
# ═══════════════════════════════════════════════════════════════════════════


class VideoFlowMatchingScheduler:
    """视频 Flow Matching 调度器：管理 5D latent 的直线路径插值。

    相比图像 Flow Matching（4D），唯一的区别是张量多了时间轴 T：
        图像: (B, C, H, W)    → reshape t 系数为 (B, 1, 1, 1)
        视频: (B, C, T, H, W) → reshape t 系数为 (B, 1, 1, 1, 1)

    参数:
        config (VideoFlowMatchingConfig): 配置对象。
        device (str | torch.device): 设备。
    """

    def __init__(
        self, config: VideoFlowMatchingConfig, device: str | torch.device = "cpu"
    ):
        self.config = config
        self.device = device

    def interpolate(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """直线插值构造加噪视频 latent z_t。

        公式：z_t = (1 - t) * z_0 + t * z_1

        参数:
            z0 (torch.Tensor): 干净视频的 latent，形状 (B, C, T, H, W)。
                               由 3D VAE 编码器对真实视频编码得到。
            z1 (torch.Tensor): 高斯噪声，形状 (B, C, T, H, W)，z1 ~ N(0, I)。
            t  (torch.Tensor): 时间步，形状 (B,)，值域 [0.0, 1.0]。
        返回:
            torch.Tensor: 插值后的 z_t，形状 (B, C, T, H, W)。
        """
        # t: (B,) → (B, 1, 1, 1, 1)，广播到 5D 视频 latent
        t_bc = t.reshape(-1, 1, 1, 1, 1)
        return (1.0 - t_bc) * z0 + t_bc * z1

    def get_velocity_target(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
    ) -> torch.Tensor:
        """计算直线路径的目标速度场。

        v_target = d(z_t)/dt = z_1 - z_0   （直线路径的斜率，常数）

        直觉：速度场就是"从干净视频指向噪声的方向向量"，恒定不变。
        模型要学会在任意 t 处，都能预测出这个方向。

        参数:
            z0 (torch.Tensor): 干净视频 latent，(B, C, T, H, W)。
            z1 (torch.Tensor): 高斯噪声，(B, C, T, H, W)。
        返回:
            torch.Tensor: 目标速度，形状 (B, C, T, H, W)。
        """
        return z1 - z0

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """训练时采样浮点时间步 t ∈ [0, 1]，形状 (B,)。

        "logit_normal" 模式（HunyuanVideo/SD3 默认）：
            t = sigmoid(u),  u ~ N(0, 1)
            分布峰值在 0.5，t=0.5 被最多采样。

        为什么偏重中间？
            t≈0（微量噪声）：z_t 几乎就是 z_0，预测很容易，
                            过度训练会导致模型只会"精修"，不会"理解结构"。
            t≈1（大量噪声）：z_t 几乎全是噪声，预测方向几乎随机，
                            梯度信号很弱，对学习贡献不大。
            t≈0.5（中间）：z_t 半信号半噪声，模型要真正理解视频内容，
                           这是最有价值的训练区域。
        """
        if self.config.t_sample_mode == "uniform":
            return torch.rand(batch_size, device=device)
        elif self.config.t_sample_mode == "logit_normal":
            u = torch.randn(batch_size, device=device)
            return torch.sigmoid(u)
        else:
            raise ValueError(f"未知采样模式: {self.config.t_sample_mode!r}")

    def t_to_int(self, t: torch.Tensor) -> torch.Tensor:
        """将浮点 t ∈ [0, 1] 转换为整数时间步索引，供 VideoMMDiT 时间嵌入使用。

        VideoMMDiT 内部的 TimestepEmbedding 接收整数 [0, T-1]，
        这里做线性缩放，模型本身不需要任何改动。

        例（T=1000）：t=0.0→0, t=0.5→499, t=1.0→999
        """
        T = self.config.num_train_timesteps
        return (t * (T - 1)).long().clamp(0, T - 1)


# ═══════════════════════════════════════════════════════════════════════════
# 2. 视频 Flow Matching 训练器
# ═══════════════════════════════════════════════════════════════════════════


class VideoFlowMatchingTrainer:
    """VideoMMDiT 的 Flow Matching 训练步骤封装。

    完整的训练步骤：
        1. 采样浮点时间步 t ~ logit_normal[0,1]
        2. 采样噪声 z_1 ~ N(0, I)，与 z_0 同形状
        3. 线性插值：z_t = (1-t)*z_0 + t*z_1
        4. CFG 丢弃：以概率 p_uncond 将 text_emb 替换为全零
        5. VideoMMDiT 预测速度：v_pred = model(z_t, t_int, text_emb)
        6. 损失：MSE(v_pred, z_1 - z_0)

    用法示例:
        >>> cfg = VideoFlowMatchingConfig()
        >>> scheduler = VideoFlowMatchingScheduler(cfg, device="cuda")
        >>> model = VideoMMDiT(VideoMMDiTConfig()).to("cuda")
        >>> trainer = VideoFlowMatchingTrainer(scheduler)
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>>
        >>> for z0, text_emb in dataloader:    # z0: (B,4,24,136,240)  text_emb: (B,256,4096)
        ...     loss = trainer.training_step(model, z0.cuda(), text_emb.cuda())
        ...     optimizer.zero_grad()
        ...     loss.backward()
        ...     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        ...     optimizer.step()
    """

    def __init__(self, scheduler: VideoFlowMatchingScheduler):
        self.scheduler = scheduler

    def training_step(
        self,
        model: nn.Module,
        z0: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """执行一步视频 Flow Matching 训练，返回 MSE 损失。

        详细流程：

            ┌────────────────────────────────────────────────────────────────┐
            │                                                                │
            │  Step 1: t ~ logit_normal[0,1]   每个视频独立采一个时间步      │
            │                                                                │
            │  Step 2: z_1 ~ N(0, I)           高斯噪声，与 z_0 同形状       │
            │                                                                │
            │  Step 3: z_t = (1-t)*z_0 + t*z_1  直线插值，一步到位          │
            │                                                                │
            │  Step 4: CFG dropout               p=10% 时用全零替换 text_emb │
            │          → 让模型学会无条件生成，推理时 CFG 才能工作           │
            │                                                                │
            │  Step 5: t_int = round(t * 999)    浮点→整数，给模型用        │
            │                                                                │
            │  Step 6: v_pred = model(z_t, t_int, text_emb)                 │
            │                                                                │
            │  Step 7: loss = MSE(v_pred, z_1 - z_0)                        │
            │                                                                │
            └────────────────────────────────────────────────────────────────┘

        参数:
            model    (nn.Module): VideoMMDiT（处于 train 模式）。
            z0       (torch.Tensor): 干净视频的 3D VAE latent，
                                     形状 (B, C, T, H, W)。
            text_emb (torch.Tensor): T5/CLIP 文本嵌入，
                                     形状 (B, L, text_dim)。
        返回:
            torch.Tensor: 标量 MSE 损失，用于反向传播。
        """
        B = z0.shape[0]
        device = z0.device

        # ── Step 1: 采样浮点时间步 t ──────────────────────────────────────
        t = self.scheduler.sample_t(B, device)  # (B,), 值域 [0.0, 1.0]

        # ── Step 2: 采样噪声 z_1 ~ N(0, I) ──────────────────────────────
        # 与视频 latent 完全同形状：(B, C, T, H, W)
        z1 = torch.randn_like(z0)

        # ── Step 3: 直线插值构造 z_t ─────────────────────────────────────
        # z_t = (1-t)*z_0 + t*z_1
        # t 系数被 reshape 成 (B,1,1,1,1)，广播到 5D 视频张量
        z_t = self.scheduler.interpolate(z0, z1, t)

        # ── Step 4: CFG dropout —— 随机丢弃文本 ─────────────────────────
        # 以概率 p_uncond 将该 batch 样本的文本嵌入替换为全零
        # 这样模型同时学会"有文本时如何生成"和"没有文本时如何生成"
        # 推理时 CFG 插值公式才能正确工作
        text_emb = self._apply_cfg_dropout(text_emb, device)

        # ── Step 5: 浮点 t → 整数索引 ───────────────────────────────────
        # VideoMMDiT 内部的时间嵌入接收整数，做一次线性缩放即可
        t_int = self.scheduler.t_to_int(t)  # (B,), 整数

        # ── Step 6: VideoMMDiT 预测速度场 ────────────────────────────────
        # 接口：model(z_t, t, text_emb) → v_pred，完全复用 mmdit-vedio.py
        v_pred = model(z_t, t_int, text_emb)  # (B, C, T, H, W)

        # ── Step 7: 计算目标速度并算损失 ─────────────────────────────────
        # v_target = z_1 - z_0（直线路径的导数，常数）
        v_target = self.scheduler.get_velocity_target(z0, z1)

        loss = nn.functional.mse_loss(v_pred, v_target)
        return loss

    def _apply_cfg_dropout(
        self,
        text_emb: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """随机将部分样本的文本嵌入替换为全零（CFG 训练 trick）。

        每个样本独立判断是否丢弃，不是整个 batch 一起。

        参数:
            text_emb (torch.Tensor): 文本嵌入，形状 (B, L, text_dim)。
        返回:
            torch.Tensor: 部分样本文本被清零的嵌入，形状不变。
        """
        B = text_emb.shape[0]
        p = self.config.p_uncond

        if p <= 0.0:
            return text_emb

        # 为每个样本生成一个 bool 掩码：True 表示丢弃文本
        # 形状 (B, 1, 1)，广播到 (B, L, text_dim)
        drop_mask = torch.rand(B, device=device) < p  # (B,)
        drop_mask = drop_mask.reshape(B, 1, 1)  # (B, 1, 1)

        # 丢弃的样本文本嵌入替换为 0（全零即"无条件"）
        return text_emb.masked_fill(drop_mask, 0.0)

    @property
    def config(self) -> VideoFlowMatchingConfig:
        return self.scheduler.config


# ═══════════════════════════════════════════════════════════════════════════
# 3. 视频 Flow Matching 采样器（推理）
# ═══════════════════════════════════════════════════════════════════════════


class VideoFlowMatchingSampler:
    """VideoMMDiT + Flow Matching 的推理采样器。

    从纯高斯噪声 z_1 出发，通过 ODE 求解逐步生成视频 latent z_0，
    最后由 3D VAE 解码为真实视频帧。

    实现了两种采样方式：
      1. Euler 法 + CFG（标准，步数 20~50）
      2. Midpoint 法 + CFG（二阶精度，步数 10~20，质量更好）

    完整的视频生成管线（本类负责中间环节）：

        文字 → T5编码 → text_emb
                                 ↘
        z_1 ~ N(0,I)  →  本类.sample() → z_0  →  3D VAE 解码  →  视频帧
                                 ↗
                          VideoMMDiT

    用法示例:
        >>> sampler = VideoFlowMatchingSampler(scheduler)
        >>> z0 = sampler.sample_euler_cfg(
        ...     model=video_mmdit,
        ...     text_emb=t5_encoder("一只猫在草地上奔跑"),
        ...     shape=(1, 4, 24, 136, 240),
        ...     num_steps=30,
        ...     cfg_scale=7.0,
        ...     device="cuda"
        ... )
        >>> video_frames = vae_3d.decode(z0)   # → (1, 3, 96, 1088, 1920)
    """

    def __init__(self, scheduler: VideoFlowMatchingScheduler):
        self.scheduler = scheduler

    @torch.no_grad()
    def sample_euler_cfg(
        self,
        model: nn.Module,
        text_emb: torch.Tensor,
        shape: tuple[int, ...],
        num_steps: int = 30,
        cfg_scale: float | None = None,
        null_text_emb: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Euler 法 + Classifier-Free Guidance 视频采样。

        推理流程（每步）：

            z_t (加噪视频 latent)
              │
              ├── model(z_t, t, text_emb)   → v_cond   (有文本的速度)
              │
              └── model(z_t, t, null_emb)   → v_uncond (无文本的速度)
              │
              ▼
            v_cfg = v_uncond + cfg_scale × (v_cond - v_uncond)
              │                    ↑ 放大文本方向
              ▼
            z_{t-dt} = z_t - dt × v_cfg    (Euler 步，反向走向 z_0)

        可视化（num_steps=5，cfg_scale=7.0）：
            t=1.0  z = 纯噪声                  ← 起点
            t=0.8  z = 微弱轮廓浮现
            t=0.6  z = 大致场景可辨
            t=0.4  z = 主体清晰
            t=0.2  z = 细节丰富
            t=0.0  z = 完整视频 latent         ← 终点 → VAE解码 → 视频

        参数:
            model         (nn.Module): 已训练好的 VideoMMDiT（eval 模式）。
            text_emb      (torch.Tensor): 文本嵌入，形状 (B, L, text_dim)。
            shape         (tuple): latent 形状 (B, C, T, H, W)。
            num_steps     (int): ODE 求解步数，越多越精细（20~50 通常足够）。
            cfg_scale     (float | None): CFG 引导强度，None 时使用 config 默认值。
            null_text_emb (torch.Tensor | None): 无条件文本嵌入（全零或空文本的
                           编码）。None 时自动生成全零张量。
            device        : 计算设备。
            show_progress (bool): 是否打印采样进度。
        返回:
            torch.Tensor: 生成的视频 latent，形状 (B, C, T, H, W)。
                          需要再经 3D VAE 解码才能得到像素级视频。
        """
        model.eval()
        cfg = cfg_scale if cfg_scale is not None else self.scheduler.config.cfg_scale
        B = shape[0]

        # ── 构造无条件文本嵌入（全零 = 无提示词）─────────────────────────
        if null_text_emb is None:
            # 形状与 text_emb 相同，内容全零
            null_text_emb = torch.zeros_like(text_emb)

        # ── 初始化：从高斯噪声出发（对应 t=1.0）─────────────────────────
        z = torch.randn(shape, device=device)

        # ── 构造时间步序列：t 从 1.0 → 0.0 ───────────────────────────────
        # num_steps+1 个端点，num_steps 个区间
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
        dt = 1.0 / num_steps

        # ── Euler ODE 求解循环（每步调用 VideoMMDiT 两次） ────────────────
        for i in range(num_steps):
            t_curr = timesteps[i].item()

            if show_progress:
                pct = int((i / num_steps) * 20)
                bar = "█" * pct + "░" * (20 - pct)
                print(
                    f"\r  [{bar}] {i + 1}/{num_steps}  t={t_curr:.2f}",
                    end="",
                    flush=True,
                )

            # 当前时间步整数化，广播给 batch
            t_float = torch.full((B,), t_curr, device=device)
            t_int = self.scheduler.t_to_int(t_float)

            # ── 有条件预测（给文本）─────────────────────────────────────
            # v_cond = VideoMMDiT(z_t, t, text_emb)
            v_cond = model(z, t_int, text_emb)

            # ── 无条件预测（给全零文本）──────────────────────────────────
            # v_uncond = VideoMMDiT(z_t, t, null_emb)
            # 这就是 CFG 的关键：同一个模型跑两次，方向不同
            v_uncond = model(z, t_int, null_text_emb)

            # ── CFG 插值：放大文本引导方向 ────────────────────────────────
            # v_cfg = v_uncond + cfg_scale × (v_cond - v_uncond)
            # 当 cfg_scale=1 时，v_cfg = v_cond（等同于有条件，无放大）
            # 当 cfg_scale=7 时，文本方向被放大 7 倍（强引导）
            v = v_uncond + cfg * (v_cond - v_uncond)

            # ── Euler 步：z_{t-dt} = z_t - dt × v ───────────────────────
            # 沿速度场反向走一步（从 t→0 方向是减法）
            z = z - dt * v

        if show_progress:
            print()  # 换行

        # ── 后处理：夹断 latent 极端值 ───────────────────────────────────
        if self.scheduler.config.clip_output:
            # 3D VAE latent 典型值域 [-4, 4]，超出范围会导致解码失真
            z = z.clamp(-4.0, 4.0)

        return z

    @torch.no_grad()
    def sample_midpoint_cfg(
        self,
        model: nn.Module,
        text_emb: torch.Tensor,
        shape: tuple[int, ...],
        num_steps: int = 15,
        cfg_scale: float | None = None,
        null_text_emb: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Midpoint 法（二阶 RK）+ CFG：更少步数，更高质量。

        Midpoint 法每步做两次 VideoMMDiT 推理：
          1. 用 Euler 半步估计中间点 z_mid（t - dt/2 处）
          2. 在中间点再做一次推理，用中点速度走完整步

        与 Euler 法对比（同样是 15 步）：
            Euler：误差 O(dt)，走弯了稍多，需要更多步数才能逼近直线
            Midpoint：误差 O(dt²)，更精确，同等步数下生成质量更好

        代价：每步调用 VideoMMDiT 4 次（Euler 2次，Midpoint 额外 2次）
        适合：想要少步数高质量时使用（如演示、快速预览）

        参数:
            model         (nn.Module): VideoMMDiT。
            text_emb      (torch.Tensor): 文本嵌入，(B, L, text_dim)。
            shape         (tuple): latent 形状 (B, C, T, H, W)。
            num_steps     (int): ODE 求解步数（Midpoint 需要步数更少，如 10~20）。
            cfg_scale     (float | None): CFG 强度。
            null_text_emb (torch.Tensor | None): 无条件文本嵌入。
            device        : 设备。
            show_progress (bool): 是否打印进度。
        返回:
            torch.Tensor: 生成的视频 latent，形状 (B, C, T, H, W)。
        """
        model.eval()#评估模式，不进行梯度计算
        cfg = cfg_scale if cfg_scale is not None else self.scheduler.config.cfg_scale
        B = shape[0]#shape0是batch size

        if null_text_emb is None:
            null_text_emb = torch.zeros_like(text_emb)

        z = torch.randn(shape, device=device)
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
        dt = 1.0 / num_steps
        #num_steps一般设置为15，因为Midpoint 法每步做两次 VideoMMDiT 推理：
        for i in range(num_steps):
            t_curr = timesteps[i].item()

            if show_progress:
                pct = int((i / num_steps) * 20)
                bar = "█" * pct + "░" * (20 - pct)
                print(
                    f"\r  [{bar}] {i + 1}/{num_steps}  t={t_curr:.2f}",
                    end="",
                    flush=True,
                )

            # ── 子步 1：在当前点 z_t 估计速度（有+无条件各一次）──────────
            t_f = torch.full((B,), t_curr, device=device)   # 构造当前步的浮点时间步 t，形状 (B,)  t_curr是当前时间步的浮点值
            t_i = self.scheduler.t_to_int(t_f)              # 转换为整数时间步索引，供模型时间嵌入使用

            v1_cond = model(z, t_i, text_emb)               # 有条件（有文本描述）下，模型预测的速度场
            v1_uncond = model(z, t_i, null_text_emb)        # 无条件（不给文本）下，模型预测的速度场
            v1 = v1_uncond + cfg * (v1_cond - v1_uncond)    # Classifier-Free Guidance 公式，插值两者得到总引导速度

            # Euler 半步，走到中间点
            z_mid = z - (dt / 2.0) * v1

            # ── 子步 2：在中间点 z_mid 估计速度（有+无条件各一次）────────
            t_mid_val = max(t_curr - dt / 2.0, 0.0)
            t_mid_f = torch.full((B,), t_mid_val, device=device)
            t_mid_i = self.scheduler.t_to_int(t_mid_f)

            # 这里的model是VideoMMDiT模型实例，它通常是一个基于Transformer的神经网络，
            # 用于预测给定噪声视频latent（z_mid）、时间步（t_mid_i）、文本特征（text_emb）条件下的速度场v_theta。
            # 其输入张量形状分别为：(B, C, T, H, W)、(B,)或其嵌入、以及文本embedding，输出与输入z_mid形状一致的速度张量。
            #这里是执行了model的forward方法，输入是z_mid, t_mid_i, text_emb，输出是v2_cond和v2_uncond，v2_cond和v2_uncond的形状是(B, C, T, H, W)
            v2_cond = model(z_mid, t_mid_i, text_emb)
            v2_uncond = model(z_mid, t_mid_i, null_text_emb)
            v2 = v2_uncond + cfg * (v2_cond - v2_uncond)

            # 用中点速度走完整步（比用起点速度更准确）  
    
            # z 是视频 latent，v2 是速度场，每一步更新 z 后，最后循环结束时的 z 就可以直接作为 VAE 的输入解码成视频了
            z = z - dt * v2

        if show_progress:
            print()

        if self.scheduler.config.clip_output:
            z = z.clamp(-4.0, 4.0)

        return z#z的形状是(B, C, T, H, W)
