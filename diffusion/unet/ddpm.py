"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           DDPM —— 去噪扩散概率模型的噪声调度、训练与推理                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

DDPM（Denoising Diffusion Probabilistic Models）是一种生成模型，
核心思想是：先学会"如何加噪"（前向过程），再学会"如何去噪"（后向过程）。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【前向过程（Forward Process）—— 训练时如何加噪声】

    q(x_0) ──加噪──▶ q(x_1) ──加噪──▶ ... ──加噪──▶ q(x_T)
         大约                                       纯高斯噪声
         原始图像                                   N(0, I)

    关键公式 1（单步加噪）：
        x_t = sqrt(α_t) * x_{t-1} + sqrt(1 - α_t) * ε,    ε ~ N(0, I)
        其中 α_t = 1 - β_t，β_t 是当前步的"噪声强度"（很小的正数）。

    关键公式 2（一步直达，训练时用这个！不需要逐步加）：
        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        其中 ᾱ_t = α_1 * α_2 * ... * α_t（累积乘积），
        ε ~ N(0, I) 是标准正态噪声。

        这个公式很重要：只需知道原始图像 x_0 和时间步 t，就能一步计算出 x_t！
        训练时不需要真的从 t=0 走到 t=T，直接"传送"到任意时间步。

    直觉理解：
        - 当 t=0 时，ᾱ_0 ≈ 1，x_t ≈ x_0（几乎没噪声）
        - 当 t=T 时，ᾱ_T ≈ 0，x_t ≈ ε（几乎全是噪声）
        - 噪声量随 t 单调增加

    β 的调度（Beta Schedule）：
        - Linear: β_t 从 β_start 线性增长到 β_end（Ho et al. 2020 原版）
        - Cosine: β_t 由余弦曲线决定（Nichol & Dhariwal 2021，更平滑）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【训练过程（Training）】

    训练目标：让 UNet 能准确预测被加入的噪声 ε。

    每个训练步骤（单张图片 x_0）：
        1. 随机采样时间步:  t ~ Uniform{0, 1, ..., T-1}
        2. 随机采样噪声:    ε ~ N(0, I)，形状与 x_0 相同
        3. 构造加噪图像:    x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        4. UNet 预测噪声:   ε_pred = UNet(x_t, t)
        5. 计算损失:        L = MSE(ε, ε_pred) = ||ε - ε_pred||²
        6. 反向传播，更新 UNet 的参数

    损失函数直觉：
        我们让 UNet 知道"当前图片里有多少噪声、噪声长什么样"，
        训练的目标就是让预测的噪声越来越接近真实加入的噪声。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【后向过程（Reverse Process）—— 推理时如何去噪声】

    推理目标：从纯噪声 x_T ~ N(0, I) 出发，逐步去噪，生成真实图像 x_0。

    每步去噪（从 x_t 得到 x_{t-1}）：
        1. 用 UNet 预测噪声:    ε_pred = UNet(x_t, t)

        2. 计算"干净图像"的估计:
           x_0_pred = (x_t - sqrt(1-ᾱ_t) * ε_pred) / sqrt(ᾱ_t)

        3. 计算均值（DDPM 公式）：
           μ_t = (1/sqrt(α_t)) * (x_t - (1-α_t)/sqrt(1-ᾱ_t) * ε_pred)

        4. 加入少量随机噪声（t > 0 时）：
           x_{t-1} = μ_t + sqrt(β_t) * z,    z ~ N(0, I)
           （最后一步 t=0 时不加噪声，直接用 μ_0 作为生成结果）

    为什么推理时还要加噪声？
        DDPM 的后向过程本质上是随机的（马尔可夫链），每步都加一点点噪声，
        这样生成的图像多样性更好，不会陷入单一的确定性结果。
        但加的噪声量（σ_t = sqrt(β_t)）远小于去掉的噪声量，整体仍然是去噪的。

    推理步数：
        原始 DDPM 需要 T=1000 步，较慢。
        DDIM（Song et al. 2020）可以用 ~50 步得到同等质量，本文件也实现了 DDIM。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【噪声"被减去"的直觉解释】

    x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε

    如果我们知道 ε（UNet 预测到了），可以反解 x_0：
        x_0_pred = (x_t - sqrt(1-ᾱ_t) * ε_pred) / sqrt(ᾱ_t)

    这就是"减去噪声"的数学本质：
        从混合信号 x_t 中，减去噪声分量 sqrt(1-ᾱ_t) * ε_pred，
        再除以信号系数 sqrt(ᾱ_t)，还原出干净图像。

    但实际推理不直接用这个，而是用更稳定的一步去噪公式（上面的 μ_t 公式），
    因为直接跳回 x_0 会有累积误差，逐步去噪（1000 步）效果更好。

参考论文:
  - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
    https://arxiv.org/abs/2006.11239
  - Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models",
    ICML 2021, https://arxiv.org/abs/2102.09672
  - Song et al., "Denoising Diffusion Implicit Models", ICLR 2021
    https://arxiv.org/abs/2010.02502
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════
# 0. 配置
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DDPMConfig:
    """DDPM 噪声调度超参数配置。

    参数:
        T            (int): 扩散步数（总时间步数），通常 1000。
        beta_start   (float): 第 0 步的 β 值（噪声强度下限），通常 1e-4。
        beta_end     (float): 第 T-1 步的 β 值（噪声强度上限），通常 0.02。
        beta_schedule (str): β 调度方式，可选 "linear" 或 "cosine"。
            - "linear": β_t 在 [beta_start, beta_end] 之间线性增长。
              简单直观，但图像质量略逊于 cosine。
            - "cosine": β_t 由余弦曲线推导，在开头和结尾变化更缓慢，
              生成质量通常更好（Nichol & Dhariwal 2021）。
        clip_denoised (bool): 推理时是否将 x_0 的估计值夹到 [-1, 1]，
            防止累积误差导致像素值爆炸。训练好的模型通常需要开启。
    """
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    clip_denoised: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# 1. 噪声调度器（DDPMNoiseScheduler）
# ═══════════════════════════════════════════════════════════════════════════

class DDPMNoiseScheduler:
    """DDPM 的核心：管理所有时间步的噪声水平，提供加噪和去噪操作。

    初始化时预计算所有 T 步的关键统计量，训练和推理时都从这里取。

    预计算量（形状均为 (T,)）：
        betas          : β_t，每步加入噪声的比例
        alphas         : α_t = 1 - β_t，每步保留信号的比例
        alphas_cumprod : ᾱ_t = ∏α_1...α_t，累积乘积（前向过程核心）
        sqrt_alphas_cumprod      : sqrt(ᾱ_t)，加噪时 x_0 的系数
        sqrt_one_minus_alphas_cumprod : sqrt(1-ᾱ_t)，加噪时 ε 的系数
        posterior_variance : β̃_t = β_t * (1-ᾱ_{t-1}) / (1-ᾱ_t)，后验方差

    参数:
        config (DDPMConfig): 调度配置，详见 DDPMConfig。
        device (str | torch.device): 存放预计算张量的设备。
    """

    def __init__(self, config: DDPMConfig, device: str | torch.device = "cpu"):
        self.config = config
        self.T = config.T
        self.device = device

        # ── 计算 β 调度表 ─────────────────────────────────────────────────
        betas = self._make_beta_schedule(config)
        betas = betas.to(device)

        # ── 从 β 推导所有需要的量 ─────────────────────────────────────────
        alphas = 1.0 - betas                              # α_t = 1 - β_t
        alphas_cumprod = torch.cumprod(alphas, dim=0)     # ᾱ_t = ∏α_1..α_t
        # ᾱ_{t-1}：用 torch.roll 右移一位，t=0 的位置填 1（ᾱ_0 = 1，无噪声）
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), alphas_cumprod[:-1]]
        )

        # ── 前向加噪时用到 ────────────────────────────────────────────────
        # x_t = sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*ε
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

        # ── 后向去噪时用到 ────────────────────────────────────────────────
        # μ_t = (1/sqrt(α_t)) * (x_t - β_t/sqrt(1-ᾱ_t) * ε_pred)
        self.sqrt_recip_alphas = (1.0 / alphas).sqrt()
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod

        # 后验方差：β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        # t=0 时分母为 0，截断到 1e-20 防止 NaN
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod).clamp(min=1e-20)
        )

    @staticmethod
    def _make_beta_schedule(config: DDPMConfig) -> torch.Tensor:
        """根据配置生成 β 序列，形状 (T,)。

        支持两种调度：
          1. linear: β_t 从 beta_start 到 beta_end 均匀线性递增。
             可视化（T=10）：[0.0001, 0.0023, 0.0045, ..., 0.02]

          2. cosine: 从ᾱ_t 的余弦曲线反推 β_t（Nichol & Dhariwal 2021）。
             ᾱ_t = cos((t/T + s)/(1+s) * π/2)²
             s = 0.008 是偏移量，防止 t=0 附近 β 太小。
             β_t 被截断到 0.999，防止极端值。
        """
        T = config.T
        if config.beta_schedule == "linear":
            return torch.linspace(config.beta_start, config.beta_end, T)

        elif config.beta_schedule == "cosine":
            s = 0.008
            steps = torch.arange(T + 1, dtype=torch.float64)
            # ᾱ_t 的余弦定义
            alphas_cumprod = torch.cos(
                ((steps / T) + s) / (1.0 + s) * math.pi / 2.0
            ) ** 2
            # 归一化，使 ᾱ_0 = 1
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            # 从 ᾱ 反推 β：β_t = 1 - ᾱ_t / ᾱ_{t-1}
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return betas.clamp(0.0, 0.999).float()

        else:
            raise ValueError(f"未知的 beta_schedule: {config.beta_schedule!r}，"
                             f"支持 'linear' 或 'cosine'。")

    # ──────────────────────────────────────────────────────────────────────
    # 前向过程：加噪（训练时使用）
    # ──────────────────────────────────────────────────────────────────────

    def add_noise(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向加噪：给定干净图像 x_0 和时间步 t，一步计算加噪后的 x_t。

        数学公式（重参数化技巧）：
            x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε,   ε ~ N(0, I)

        为什么可以"一步到位"而不需要逐步加噪？
            因为多个高斯分布的组合仍然是高斯分布：
            把 t 步的小噪声逐步叠加，等价于一步加一个大噪声。
            公式中的 sqrt(1-ᾱ_t) 就是这个"等价大噪声"的标准差。

        可视化（batch_size=2, C=3, H=W=64, t=[250, 750]）：
            t=250: ᾱ_250 ≈ 0.75 → 图像保留 75% 信号 + 25% 噪声（轻度加噪）
            t=750: ᾱ_750 ≈ 0.25 → 图像保留 25% 信号 + 75% 噪声（重度加噪）

        参数:
            x0    (torch.Tensor): 干净的原始图像，形状 (B, C, H, W)。
                                  像素值通常归一化到 [-1, 1]。
            t     (torch.Tensor): 时间步索引，形状 (B,)，整数，值域 [0, T-1]。
            noise (torch.Tensor | None): 可选的预设噪声，形状 (B, C, H, W)。
                                         若为 None，则自动采样标准正态噪声。
        返回:
            Tuple[torch.Tensor, torch.Tensor]:
                - x_t: 加噪后的图像，形状 (B, C, H, W)。
                - noise: 实际加入的噪声，形状 (B, C, H, W)。
                  训练时将其作为监督目标，让 UNet 来预测。
        """
        if noise is None:
            # 从标准正态分布采样噪声，形状与输入图像相同
            noise = torch.randn_like(x0)

        # ── 取出时间步 t 对应的系数 ──────────────────────────────────────
        # self.sqrt_alphas_cumprod 形状 (T,)，用 t 索引后形状 (B,)
        # 需要 reshape 成 (B, 1, 1, 1) 才能与 (B, C, H, W) 广播运算
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)

        # ── 加噪核心公式 ──────────────────────────────────────────────────
        # x_t = sqrt(ᾱ_t) * x_0  +  sqrt(1-ᾱ_t) * ε
        #         ↑ 信号分量                ↑ 噪声分量
        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

        return x_t, noise

    # ──────────────────────────────────────────────────────────────────────
    # 后向过程：单步去噪（推理时使用）
    # ──────────────────────────────────────────────────────────────────────

    def remove_noise_step(
        self,
        x_t: torch.Tensor,
        t: int,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        """DDPM 后向去噪单步：给定 x_t 和预测的噪声，计算 x_{t-1}。

        数学推导（DDPM 后验均值）：
            贝叶斯公式可以推出，在已知 ε_θ 的情况下，x_{t-1} 的后验分布为：
            p(x_{t-1} | x_t, ε_θ) = N(μ_θ(x_t, t), β̃_t * I)

            其中均值为：
            μ_θ(x_t, t) = (1/sqrt(α_t)) * (x_t - β_t/sqrt(1-ᾱ_t) * ε_θ)

            方差为：
            β̃_t = β_t * (1-ᾱ_{t-1}) / (1-ᾱ_t)

            采样公式（t > 0 时加随机噪声）：
            x_{t-1} = μ_θ(x_t, t) + sqrt(β̃_t) * z,   z ~ N(0, I)
            x_{t-1} = μ_θ(x_t, t)                         （t = 0 时不加噪声）

        为什么推理时（t > 0）还要加噪声 z？
            DDPM 的后验分布是随机的（不同的随机种子产生不同的图像），
            加入 z 保持了这种随机性，使生成多样化。
            但每步加的噪声量 sqrt(β̃_t) 很小，整体趋势仍然是去噪。

        可视化（去噪一步）：
            x_t (含大量噪声) → 预测噪声 ε_θ → 减去噪声分量 → x_{t-1} (稍干净一点)
                                                                       ↑
                                                             再加一点点随机噪声 z（保持随机性）

        参数:
            x_t             (torch.Tensor): 当前步加噪图像，形状 (B, C, H, W)。
            t               (int): 当前时间步（标量整数，值域 [0, T-1]）。
            predicted_noise (torch.Tensor): UNet 预测的噪声，形状 (B, C, H, W)。
        返回:
            torch.Tensor: 去噪一步后的图像 x_{t-1}，形状 (B, C, H, W)。
        """
        # ── 取出时间步 t 对应的系数（标量） ──────────────────────────────
        beta_t = self.betas[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]

        # ── 计算均值 μ_θ(x_t, t) ─────────────────────────────────────────
        # μ = (1/sqrt(α_t)) * (x_t - β_t/sqrt(1-ᾱ_t) * ε_θ)
        #       ↑ 系数            ↑ x_t 项     ↑ 减去预测噪声分量
        noise_coeff = beta_t / sqrt_one_minus_alpha_bar_t
        mean = sqrt_recip_alpha_t * (x_t - noise_coeff * predicted_noise)

        # ── 加入随机噪声（t > 0 时） ──────────────────────────────────────
        if t > 0:
            # 后验方差 β̃_t（取 ln 后再 exp 是为了数值稳定）
            variance = self.posterior_variance[t]
            # 从标准正态采样，乘以标准差
            z = torch.randn_like(x_t)
            x_prev = mean + variance.sqrt() * z
        else:
            # 最后一步（t=0）不加噪声，直接用均值作为最终生成结果
            x_prev = mean

        return x_prev

    def estimate_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        """从 x_t 和预测的噪声，直接估计干净图像 x_0。

        公式（由加噪公式反解）：
            x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
            ⟹  x_0 = (x_t - sqrt(1-ᾱ_t) * ε_pred) / sqrt(ᾱ_t)

        这个估计不直接用于去噪采样，但可用于：
            - 可视化训练过程中 x_0 的重建质量
            - 某些加速采样算法（如 DDIM）

        参数:
            x_t             (torch.Tensor): 加噪图像，形状 (B, C, H, W)。
            t               (torch.Tensor): 时间步，形状 (B,)。
            predicted_noise (torch.Tensor): UNet 预测噪声，形状 (B, C, H, W)。
        返回:
            torch.Tensor: x_0 的估计，形状 (B, C, H, W)。
        """
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)

        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

        if self.config.clip_denoised:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        return x0_pred


# ═══════════════════════════════════════════════════════════════════════════
# 2. 训练器（DiffusionTrainer）
# ═══════════════════════════════════════════════════════════════════════════

class DiffusionTrainer:
    """DDPM 的训练流程封装。

    封装了完整的单步训练逻辑：随机加噪 → UNet 预测 → 计算损失。
    外部只需提供模型、调度器和优化器，调用 training_step 即可。

    用法示例:
        >>> config = DDPMConfig(T=1000, beta_schedule="cosine")
        >>> scheduler = DDPMNoiseScheduler(config, device="cuda")
        >>> unet = UNet(UNetConfig()).to("cuda")
        >>> trainer = DiffusionTrainer(scheduler)
        >>> optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
        >>>
        >>> for x0 in dataloader:          # x0: (B, C, H, W), 归一化到 [-1,1]
        ...     loss = trainer.training_step(unet, x0.to("cuda"))
        ...     optimizer.zero_grad()
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(self, scheduler: DDPMNoiseScheduler):
        self.scheduler = scheduler

    def training_step(
        self,
        model: nn.Module,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """执行一步训练，返回 MSE 损失。

        详细流程（逐步注释）：

            Step 1: 随机采样时间步 t
                每张图片独立采样一个 t，让模型学会应对所有噪声水平。
                范围：[0, T-1]，均匀分布。

            Step 2: 随机采样噪声 ε
                从标准正态分布采样，与输入图像形状相同。
                这就是"真实噪声"，训练时作为监督目标。

            Step 3: 前向加噪，构造 x_t
                x_t = sqrt(ᾱ_t) * x0 + sqrt(1-ᾱ_t) * ε
                这一步不需要梯度，只是数据预处理。

            Step 4: UNet 预测噪声
                ε_pred = UNet(x_t, t)
                UNet 接受加噪图像和时间步，输出对噪声的预测。

            Step 5: 计算 MSE 损失
                L = mean((ε - ε_pred)²)
                目标是让 UNet 预测的噪声尽可能接近真实加入的噪声。

        参数:
            model (nn.Module): UNet 模型（需处于 train 模式）。
            x0    (torch.Tensor): 干净图像批次，形状 (B, C, H, W)，值域 [-1, 1]。
        返回:
            torch.Tensor: 标量损失值（MSE），用于反向传播。
        """
        B = x0.shape[0]
        device = x0.device

        # ── Step 1: 随机采样时间步 t ─────────────────────────────────────
        # 每张图片独立采样，形状 (B,)
        # 为什么随机采样 t 而不是顺序采样？
        #   因为我们希望模型在所有时间步上都有良好表现，
        #   随机采样保证每个 t 都有足够的训练机会。
        t = torch.randint(0, self.scheduler.T, (B,), device=device, dtype=torch.long)

        # ── Step 2: 随机采样噪声 ε ──────────────────────────────────────
        # 从 N(0, I) 采样，形状与 x0 完全相同
        noise = torch.randn_like(x0)

        # ── Step 3: 前向加噪，构造 x_t ──────────────────────────────────
        # x_t = sqrt(ᾱ_t) * x0 + sqrt(1-ᾱ_t) * noise
        # 注意：这步不需要计算梯度（scheduler 是纯计算，不是 nn.Module）
        x_t, noise = self.scheduler.add_noise(x0, t, noise)

        # ── Step 4: UNet 前向，预测噪声 ε_pred ───────────────────────────
        # UNet 同时接收加噪图像 x_t 和时间步 t
        # 时间步 t 通过时间嵌入注入网络，告诉网络"现在是哪个噪声水平"
        noise_pred = model(x_t, t)

        # ── Step 5: 计算 MSE 损失 ────────────────────────────────────────
        # 损失 = 真实噪声 ε 和预测噪声 ε_pred 之间的均方误差
        # mean() 在 B, C, H, W 所有维度上取平均
        loss = nn.functional.mse_loss(noise_pred, noise)

        return loss


# ═══════════════════════════════════════════════════════════════════════════
# 3. DDPM 采样器（DDPMSampler）—— 推理
# ═══════════════════════════════════════════════════════════════════════════

class DDPMSampler:
    """DDPM 推理采样器：从纯噪声出发，逐步去噪生成图像。

    实现了两种采样方式：
      1. DDPM 采样（标准随机采样，原论文方法，需要 T 步）
      2. DDIM 采样（确定性采样，Song et al. 2021，只需 ~50 步）

    用法示例（DDPM 标准采样）:
        >>> sampler = DDPMSampler(scheduler)
        >>> with torch.no_grad():
        ...     samples = sampler.sample_ddpm(unet, shape=(4, 3, 64, 64), device="cuda")
        >>> # samples: (4, 3, 64, 64)，像素值在 [-1, 1]

    用法示例（DDIM 加速采样，50 步）:
        >>> samples = sampler.sample_ddim(unet, shape=(4, 3, 64, 64),
        ...                               num_steps=50, device="cuda")
    """

    def __init__(self, scheduler: DDPMNoiseScheduler):
        self.scheduler = scheduler

    @torch.no_grad()
    def sample_ddpm(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        device: str | torch.device = "cpu",
        show_progress: bool = True,
    ) -> torch.Tensor:
        """DDPM 标准采样：从高斯噪声逐步去噪 T 步。

        推理流程可视化（T=1000，图像 64×64）：

            x_1000 ~ N(0, I)   ← 纯随机噪声，什么都看不出来
               ↓ 第999步去噪（UNet 预测噪声，减去一点）
            x_999              ← 依然是噪声，但有轻微变化
               ↓ 第998步去噪
            x_998
               ↓ ...（共 1000 步，每步减少一点噪声）
               ↓
            x_1                ← 已经能隐约看出图像轮廓
               ↓ 第0步去噪
            x_0                ← 清晰的生成图像！

        每步去噪（t 从 T-1 降到 0）：
            1. 调用 UNet：ε_pred = model(x_t, t)
            2. 计算 x_{t-1}：使用后验均值公式 + 加少量随机噪声（t>0 时）
            3. 更新：x_t ← x_{t-1}

        参数:
            model        (nn.Module): 已训练好的 UNet，处于 eval 模式。
            shape        (tuple): 生成图像的形状，如 (B, C, H, W)。
            device       : 计算设备。
            show_progress (bool): 是否打印进度（每100步）。
        返回:
            torch.Tensor: 生成的图像批次，形状 shape，值域近似 [-1, 1]。
        """
        model.eval()
        T = self.scheduler.T

        # ── 从标准正态分布初始化 x_T ──────────────────────────────────────
        # x_T ~ N(0, I)：纯高斯噪声，没有任何图像信息
        x_t = torch.randn(shape, device=device)

        # ── 从 t=T-1 逐步去噪到 t=0 ──────────────────────────────────────
        for t_idx in range(T - 1, -1, -1):

            if show_progress and t_idx % 100 == 0:
                print(f"  DDPM 采样中... 时间步 {t_idx}/{T-1}")

            # 当前时间步，广播给 batch 中的每个样本
            t_tensor = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)

            # ── UNet 前向：预测当前步的噪声 ──────────────────────────────
            # ε_pred = model(x_t, t)
            # 这是推理的核心：用训练好的 UNet"估计" x_t 中混入了多少噪声
            noise_pred = model(x_t, t_tensor)

            # ── 单步去噪：x_t → x_{t-1} ──────────────────────────────────
            x_t = self.scheduler.remove_noise_step(x_t, t_idx, noise_pred)

        return x_t   # 此时 x_t 就是 x_0（生成完成的图像）

    @torch.no_grad()
    def sample_ddim(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        num_steps: int = 50,
        eta: float = 0.0,
        device: str | torch.device = "cpu",
        show_progress: bool = True,
    ) -> torch.Tensor:
        """DDIM 加速采样：只需少量步数（如 50 步）生成高质量图像。

        DDIM（Denoising Diffusion Implicit Models）是对 DDPM 的改进：
          - DDPM：随机采样（每步加随机噪声），必须走完 T 步
          - DDIM：确定性采样（eta=0 时），可以跳步，50 步得到同等质量

        DDIM 的核心公式（eta=0 的确定性版本）：
            x_0_pred = (x_t - sqrt(1-ᾱ_t) * ε_pred) / sqrt(ᾱ_t)
            x_{t'} = sqrt(ᾱ_{t'}) * x_0_pred + sqrt(1-ᾱ_{t'}) * ε_pred

            其中 t' < t 是下一个时间步（可以跳多步）。

        直觉解释：
            1. 先用 UNet 预测噪声 ε_pred
            2. 从 x_t 和 ε_pred 估计 x_0（干净图像）
            3. 利用 x_0 和 ε_pred，"重构"到目标时间步 t'
            这个过程是确定性的（没有随机性），因此质量更稳定，步数更少。

        参数:
            model     (nn.Module): 已训练好的 UNet。
            shape     (tuple): 生成图像的形状，如 (B, C, H, W)。
            num_steps (int): 采样步数（远小于 T，如 50）。
            eta       (float): 随机性参数，eta=0 为完全确定性（DDIM），
                               eta=1 近似等价于 DDPM。
            device    : 计算设备。
            show_progress (bool): 是否打印进度。
        返回:
            torch.Tensor: 生成的图像批次，形状 shape。
        """
        model.eval()
        T = self.scheduler.T

        # ── 构造均匀间隔的时间步序列 ──────────────────────────────────────
        # 从 [0, T-1] 中均匀选 num_steps 个时间步（跳步）
        # 例如 T=1000, num_steps=50 → [0, 20, 40, ..., 980, 999]
        step_indices = torch.linspace(0, T - 1, num_steps, dtype=torch.long)
        # 逆序：从大到小（从 t=T-1 降到 t=0）
        step_indices = step_indices.flip(0)

        # ── 初始化 x_T ────────────────────────────────────────────────────
        x_t = torch.randn(shape, device=device)

        # ── DDIM 去噪循环 ────────────────────────────────────────────────
        for i, t_curr in enumerate(step_indices):
            t_curr = t_curr.item()

            if show_progress and i % 10 == 0:
                print(f"  DDIM 采样中... 步骤 {i}/{num_steps-1} (时间步 {t_curr})")

            t_tensor = torch.full((shape[0],), t_curr, device=device, dtype=torch.long)

            # ── UNet 预测噪声 ────────────────────────────────────────────
            noise_pred = model(x_t, t_tensor)

            # ── 取出当前步和下一步的 ᾱ ────────────────────────────────────
            alpha_bar_t = self.scheduler.alphas_cumprod[t_curr]
            # 下一个时间步（若已到最后一步，则 ᾱ_{t-1} = 1）
            if i + 1 < num_steps:
                t_next = step_indices[i + 1].item()
                alpha_bar_t_next = self.scheduler.alphas_cumprod[t_next]
            else:
                alpha_bar_t_next = torch.tensor(1.0, device=device)

            # ── 估计 x_0_pred ─────────────────────────────────────────────
            # x_0_pred = (x_t - sqrt(1-ᾱ_t) * ε_pred) / sqrt(ᾱ_t)
            x0_pred = (
                x_t - (1.0 - alpha_bar_t).sqrt() * noise_pred
            ) / alpha_bar_t.sqrt()

            if self.scheduler.config.clip_denoised:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            # ── DDIM 更新公式 ─────────────────────────────────────────────
            # 方向噪声（"预测方向"）
            direction = (1.0 - alpha_bar_t_next).sqrt() * noise_pred
            # 随机噪声（eta=0 时为 0，完全确定性）
            if eta > 0 and i + 1 < num_steps:
                sigma = eta * (
                    (1 - alpha_bar_t_next) / (1 - alpha_bar_t)
                    * (1 - alpha_bar_t / alpha_bar_t_next)
                ).sqrt()
                rand_noise = sigma * torch.randn_like(x_t)
            else:
                rand_noise = 0.0

            # x_{t'} = sqrt(ᾱ_{t'}) * x_0_pred + sqrt(1-ᾱ_{t'}) * ε_pred + σ*z
            x_t = alpha_bar_t_next.sqrt() * x0_pred + direction + rand_noise

        return x_t
