"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           Flow Matching —— 直线路径流匹配，现代生成模型的主流方案               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Flow Matching（流匹配）是 2022-2023 年提出的扩散模型替代方案，
被 Stable Diffusion 3、Flux、Sora 等现代顶级模型采用。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【与 DDPM/DDIM 最核心的区别：路径形状】

  DDPM（弯曲路径，由复杂的 β 调度表控制）：
      x_0 ──弯弯曲曲──→ x_T
      就像在山路上开车，路很弯，只能慢慢走，需要 1000 步

  Flow Matching（直线路径，最短距离）：
      x_0 ─────────→ x_1
      就像在直路上开车，可以大步迈进，只需 5~20 步

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【前向过程（训练时如何构造 x_t）】

  DDPM：  x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε    （复杂，依赖 β 表）
  Flow：   x_t = (1 - t) * x_0  +  t * x_1              （简单！线性插值）
               ↑ 信号系数随 t 线性减小   ↑ 噪声系数随 t 线性增大

  其中 t ∈ [0.0, 1.0]（连续浮点数，不是整数！）：
    - t=0.0 → x_t = x_0（纯干净图像）
    - t=0.5 → x_t = 0.5*x_0 + 0.5*x_1（一半图像一半噪声）
    - t=1.0 → x_t = x_1（纯高斯噪声）

  可视化（t 从 0 到 1 的直线插值）：
      t=0.0  [██████████] 100% 图像  + 0%  噪声
      t=0.25 [███████░░░]  75% 图像  + 25% 噪声
      t=0.5  [█████░░░░░]  50% 图像  + 50% 噪声
      t=0.75 [███░░░░░░░]  25% 图像  + 75% 噪声
      t=1.0  [░░░░░░░░░░]   0% 图像  + 100%噪声

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【模型学什么？速度场 v，而不是噪声 ε】

  DDPM：模型预测 ε（"图里有多少噪声、噪声长什么样"）
  Flow：模型预测 v（"从当前位置往哪个方向走能最快到达 x_0"）

  在直线路径下，速度 v 非常简单——它就是从噪声到图像的方向：
      v_target = dx_t/dt = x_1 - x_0    （对直线求导就是斜率，恒定！）

  训练目标：
      v_pred = UNet(x_t, t)
      loss   = MSE(v_pred, x_1 - x_0)

  直觉：给定任意时刻 t 的中间状态 x_t，
  模型要学会预测"从 x_t 出发，应该往哪个方向移动才能到达 x_0"。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【推理过程（ODE 求解，沿速度场走）】

  从 x_1 ~ N(0, I) 出发，解 ODE：dx/dt = -v_θ(x_t, t)
  使用 Euler 方法（最简单的 ODE 数值解法）：

      x_{t-Δt} = x_t - Δt * v_θ(x_t, t)
                         ↑ 每步沿速度方向走 Δt 距离

  例如 n_steps=10，Δt=0.1：
      x_1.0 → x_0.9 → x_0.8 → x_0.7 → ... → x_0.1 → x_0.0
      每步走 0.1，共 10 步，因为路径是直线所以质量不损失多少

  为什么步数少也没问题？
      路径是直线，速度场 v 几乎不变（不同 t 下预测值相近），
      所以大步长走也准确。DDPM 的弯曲路径必须小步走否则会偏离。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【UNet 需要改动吗？不需要！】

  UNet 本质上是：(特征图, 时间步) → 同尺寸特征图
  无论输出含义是"噪声 ε"还是"速度 v"，网络结构完全一样。

  唯一的适配：UNet 的时间嵌入接收整数 [0, T-1]，
  而 Flow Matching 的 t 是浮点 [0.0, 1.0]。
  解决方案：在传入前缩放一下，UNet 完全不用动。
      t_int = (t_float * (T - 1)).long()

参考论文:
  - Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
    https://arxiv.org/abs/2210.02747
  - Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer
    Data with Rectified Flow", ICLR 2023, https://arxiv.org/abs/2209.03003
  - Esser et al., "Scaling Rectified Flow Transformers for High-Resolution
    Image Synthesis" (Stable Diffusion 3), 2024, https://arxiv.org/abs/2403.03206
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════
# 0. 配置
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FlowMatchingConfig:
    """Flow Matching 超参数配置。

    参数:
        num_train_timesteps (int): 训练时离散化的步数，用于将浮点 t 映射到
            UNet 时间嵌入的整数索引。通常与 UNet 训练时的 T 一致（默认 1000）。
            推理步数不受此限制，可以任意指定（如 20 步）。
        sigma_min (float): 最小噪声水平。t=0 时理论上是纯图像，但加少量噪声
            可以提升训练稳定性。默认 0（不加）。
        clip_output (bool): 推理完成后是否将图像像素值夹到 [-1, 1]。
        t_sample_mode (str): 训练时 t 的采样方式：
            - "uniform": 均匀采样，t ~ U[0, 1]（标准做法）
            - "logit_normal": 偏重中间时间步（SD3 用的，中间步更难，多采样）
    """
    num_train_timesteps: int = 1000
    sigma_min: float = 0.0
    clip_output: bool = True
    t_sample_mode: str = "uniform"


# ═══════════════════════════════════════════════════════════════════════════
# 1. Flow Matching 调度器
# ═══════════════════════════════════════════════════════════════════════════

class FlowMatchingScheduler:
    """Flow Matching 的核心：管理直线路径的插值和速度场。

    不需要 DDPM 那样的 β 表！所有公式都是简单的线性插值。

    参数:
        config (FlowMatchingConfig): 配置对象。
        device (str | torch.device): 设备。
    """

    def __init__(self, config: FlowMatchingConfig,
                 device: str | torch.device = "cpu"):
        self.config = config
        self.device = device

    # ──────────────────────────────────────────────────────────────────────
    # 前向过程：构造 x_t（训练时使用）
    # ──────────────────────────────────────────────────────────────────────

    def interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """直线插值：从 x_0（图像）和 x_1（噪声）构造时刻 t 的中间状态 x_t。

        公式：
            x_t = (1 - t) * x_0 + t * x_1

        与 DDPM 加噪对比：
            DDPM: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε   ← 复杂系数
            Flow: x_t = (1 -  t ) * x_0 + t * x_1              ← 就是线性插值！

        参数:
            x0 (torch.Tensor): 干净图像，形状 (B, C, H, W)。
            x1 (torch.Tensor): 高斯噪声，形状 (B, C, H, W)，x1 ~ N(0, I)。
            t  (torch.Tensor): 时间步，形状 (B,)，值域 [0.0, 1.0]。
        返回:
            torch.Tensor: 插值后的中间状态 x_t，形状 (B, C, H, W)。
        """
        # t: (B,) → reshape 成 (B, 1, 1, 1) 才能与图像广播
        t_bc = t.reshape(-1, 1, 1, 1)
        return (1.0 - t_bc) * x0 + t_bc * x1

    def get_velocity_target(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """计算直线路径上的目标速度场 v_target。

        在直线路径上，速度 v = dx_t/dt，对 x_t = (1-t)*x_0 + t*x_1 求导：
            v = d/dt [(1-t)*x_0 + t*x_1]
              = -x_0 + x_1
              = x_1 - x_0

        关键：速度是常数！不随 t 变化。
        这是 Flow Matching 比 DDPM 简单的根本原因——
        DDPM 的"速度"随 t 非线性变化，Flow Matching 的速度是固定方向的箭头。

        可视化：
            x_0 ──── v ────▶ x_1      v = x_1 - x_0（方向从图像指向噪声）
            推理时反向走：   x_1 ──── -v ───▶ x_0

        参数:
            x0 (torch.Tensor): 干净图像，形状 (B, C, H, W)。
            x1 (torch.Tensor): 高斯噪声，形状 (B, C, H, W)。
        返回:
            torch.Tensor: 目标速度场，形状 (B, C, H, W)。
        """
        return x1 - x0

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """训练时采样时间步 t，形状 (B,)，值域 [0.0, 1.0]。

        支持两种模式：
          - "uniform": t ~ U[0, 1]，每个时间步被均等训练
          - "logit_normal": t 更多集中在中间（0.3~0.7），
            因为中间时间步预测最难（SD3 采用此方案）

        为什么中间步更难？
            - t 接近 0：x_t ≈ x_0，噪声很少，很容易还原
            - t 接近 1：x_t ≈ x_1，几乎全是噪声，预测方向任意都差不多
            - t ≈ 0.5：真正需要判断"这一半噪声一半信号的图是什么"，最难！
        """
        if self.config.t_sample_mode == "uniform":
            return torch.rand(batch_size, device=device)

        elif self.config.t_sample_mode == "logit_normal":
            # 从标准正态采样，再经 sigmoid 映射到 (0,1)
            # sigmoid(N(0,1)) 的分布峰值在 0.5 附近，符合"偏重中间"的需求
            u = torch.randn(batch_size, device=device)
            return torch.sigmoid(u)

        else:
            raise ValueError(f"未知采样模式: {self.config.t_sample_mode!r}")

    def t_to_int(self, t: torch.Tensor) -> torch.Tensor:
        """将浮点时间步 t ∈ [0, 1] 转换为整数索引，供 UNet 时间嵌入使用。

        UNet 的 SinusoidalTimeEmbedding 接收整数 [0, T-1]，
        而 Flow Matching 使用浮点 t ∈ [0.0, 1.0]。
        这里做一次线性缩放，UNet 本身完全不需要修改。

        例如（T=1000）：
            t=0.0   → 0
            t=0.5   → 499
            t=1.0   → 999
        """
        T = self.config.num_train_timesteps
        return (t * (T - 1)).long().clamp(0, T - 1)

    # ──────────────────────────────────────────────────────────────────────
    # 后向过程：ODE 单步（推理时使用）
    # ──────────────────────────────────────────────────────────────────────

    def euler_step(
        self,
        x_t: torch.Tensor,
        t: float,
        velocity: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Euler 法单步去噪：沿速度场反向走一步。

        ODE：dx/dt = v_θ(x_t, t)
        反向求解（从 t=1 走向 t=0）：
            x_{t - dt} = x_t - dt * v_θ(x_t, t)

        可视化（dt=0.1，从 t=1.0 走到 t=0.0）：
            t=1.0: x = 纯噪声
            t=0.9: x = x_1.0 - 0.1 * v   ← 往图像方向走了 10%
            t=0.8: x = x_0.9 - 0.1 * v   ← 再走 10%
            ...
            t=0.0: x = x_0.1 - 0.1 * v   ← 到达图像

        与 DDPM 去噪单步对比：
            DDPM: x_{t-1} = 复杂公式（涉及 β_t, ᾱ_t 等）+ 随机噪声
            Flow: x_{t-dt} = x_t - dt * v   ← 简单！确定性！

        参数:
            x_t      (torch.Tensor): 当前时刻的特征图，形状 (B, C, H, W)。
            t        (float): 当前时间步（浮点，值域 [0, 1]），仅用于记录，
                              实际计算只用 velocity 和 dt。
            velocity (torch.Tensor): 模型预测的速度场，形状 (B, C, H, W)。
            dt       (float): 步长（正数），例如 1/num_steps。
        返回:
            torch.Tensor: 下一时刻的特征图，形状 (B, C, H, W)。
        """
        # 反向走：从 t 走向 t-dt（靠近 x_0）
        return x_t - dt * velocity


# ═══════════════════════════════════════════════════════════════════════════
# 2. Flow Matching 训练器
# ═══════════════════════════════════════════════════════════════════════════

class FlowMatchingTrainer:
    """Flow Matching 训练流程封装。

    与 DiffusionTrainer 对比：
        DiffusionTrainer（DDPM）:
            1. 随机采样整数 t ~ U[0, T-1]
            2. 查 β 表算加噪系数，构造 x_t
            3. 预测噪声 ε，算 MSE

        FlowMatchingTrainer:
            1. 随机采样浮点 t ~ U[0, 1]（无需 β 表！）
            2. 线性插值构造 x_t = (1-t)*x0 + t*x1
            3. 预测速度 v，算 MSE（目标就是 x1-x0，很简单）

    用法示例:
        >>> config = FlowMatchingConfig()
        >>> scheduler = FlowMatchingScheduler(config, device="cuda")
        >>> unet = UNet(UNetConfig()).to("cuda")
        >>> trainer = FlowMatchingTrainer(scheduler)
        >>> optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
        >>>
        >>> for x0 in dataloader:          # x0: (B, C, H, W), 归一化到 [-1, 1]
        ...     loss = trainer.training_step(unet, x0.to("cuda"))
        ...     optimizer.zero_grad()
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(self, scheduler: FlowMatchingScheduler):
        self.scheduler = scheduler

    def training_step(
        self,
        model: nn.Module,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """执行一步 Flow Matching 训练，返回 MSE 损失。

        详细流程（与 DDPM 逐步对比）：

            ┌──────────────────┬────────────────────────────────────────────┐
            │ DDPM             │ Flow Matching                              │
            ├──────────────────┼────────────────────────────────────────────┤
            │ t ~ U[0, T-1]    │ t ~ U[0.0, 1.0]（浮点！）                  │
            │ ε ~ N(0, I)      │ x1 ~ N(0, I)（同样采样噪声）                │
            │ 查 β 表算系数     │ 不需要 β 表！直接线性插值                    │
            │ x_t = 复杂公式   │ x_t = (1-t)*x0 + t*x1（简单！）            │
            │ 预测目标 = ε      │ 预测目标 = x1 - x0（速度）                  │
            │ loss = MSE(ε, ε̂) │ loss = MSE(v_target, v_pred)              │
            └──────────────────┴────────────────────────────────────────────┘

        参数:
            model (nn.Module): UNet 模型（处于 train 模式，无需修改）。
            x0    (torch.Tensor): 干净图像批次，形状 (B, C, H, W)，值域 [-1, 1]。
        返回:
            torch.Tensor: 标量损失值（MSE），用于反向传播。
        """
        B = x0.shape[0]
        device = x0.device

        # ── Step 1: 采样时间步 t（浮点，[0.0, 1.0]） ──────────────────────
        # 注意：Flow Matching 用连续浮点 t，不是离散整数
        t = self.scheduler.sample_t(B, device)   # (B,)，每个样本独立

        # ── Step 2: 采样噪声 x_1 ~ N(0, I) ──────────────────────────────
        x1 = torch.randn_like(x0)

        # ── Step 3: 线性插值构造 x_t ─────────────────────────────────────
        # x_t = (1-t) * x_0 + t * x_1
        # 无需查任何调度表，就是简单的加权平均！
        x_t = self.scheduler.interpolate(x0, x1, t)

        # ── Step 4: 计算目标速度 v_target ────────────────────────────────
        # v_target = x_1 - x_0（直线路径的导数，恒定）
        # 比 DDPM 的噪声目标更直观：就是"从图像指向噪声的向量"
        v_target = self.scheduler.get_velocity_target(x0, x1)

        # ── Step 5: 将浮点 t 转为整数，传给 UNet（UNet 不需要改！） ────────
        # UNet 的时间嵌入接收整数 [0, T-1]，做一次线性缩放即可
        t_int = self.scheduler.t_to_int(t)       # (B,)，整数

        # ── Step 6: UNet 预测速度场 v_pred ────────────────────────────────
        # UNet 的输入/输出格式与 DDPM 完全相同！
        # 区别只是训练目标：DDPM 预测噪声 ε，Flow 预测速度 v
        v_pred = model(x_t, t_int)               # (B, C, H, W)

        # ── Step 7: MSE 损失 ─────────────────────────────────────────────
        loss = nn.functional.mse_loss(v_pred, v_target)

        return loss


# ═══════════════════════════════════════════════════════════════════════════
# 3. Flow Matching 采样器（推理）
# ═══════════════════════════════════════════════════════════════════════════

class FlowMatchingSampler:
    """Flow Matching 推理采样器：用 ODE 求解器从噪声生成图像。

    实现了两种 ODE 求解方法：
      1. Euler 法（最简单，线性精度，需要更多步）
      2. Midpoint 法（二阶精度，同等步数下质量更好）

    与 DDPM 采样对比：
        DDPM 采样（1000步）：
            for t in range(999, -1, -1):
                ε_pred = model(x_t, t)             ← 预测噪声
                x_t = 复杂公式(x_t, ε_pred, β_t)   ← 依赖 β 表

        Flow Matching 采样（20步）：
            for t in linspace(1.0, 0.0, 20):
                v_pred = model(x_t, t)             ← 预测速度
                x_t = x_t - dt * v_pred            ← 简单 Euler 步

    用法示例:
        >>> sampler = FlowMatchingSampler(scheduler)
        >>> images = sampler.sample_euler(unet, shape=(4, 3, 64, 64),
        ...                               num_steps=20, device="cuda")
    """

    def __init__(self, scheduler: FlowMatchingScheduler):
        self.scheduler = scheduler

    @torch.no_grad()
    def sample_euler(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        num_steps: int = 20,
        device: str | torch.device = "cpu",
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Euler 法采样：最简单的 ODE 求解，从噪声逐步走向图像。

        推理流程可视化（num_steps=5，图像 64×64）：

            t=1.0  x = 纯随机噪声 ~ N(0,I)        ← 起点
                   v = model(x, 1.0) → 速度指向 x_0
                   x = x - 0.2 * v
            t=0.8  x = 稍微像图像了一点
                   v = model(x, 0.8)
                   x = x - 0.2 * v
            t=0.6  ...
            t=0.4  ...
            t=0.2  x = 已经很清晰了
                   v = model(x, 0.2)
                   x = x - 0.2 * v
            t=0.0  x = 生成完成的图像！                 ← 终点

        与 DDPM 的本质区别：
            DDPM：路径弯曲，每步只能走 β_t 那么一小步（~0.0001），需要 1000 步
            Flow：路径是直线，速度近似恒定，可以大步走（0.2/步），只需 5~20 步

        参数:
            model     (nn.Module): 已训练好的 UNet（用 Flow Matching 训练）。
            shape     (tuple): 生成图像的形状，如 (B, C, H, W)。
            num_steps (int): ODE 求解步数（越多越精细，通常 10~50 就足够）。
            device    : 计算设备。
            show_progress (bool): 是否打印进度。
        返回:
            torch.Tensor: 生成的图像，形状 shape，值域近似 [-1, 1]。
        """
        model.eval()

        # ── 初始化：从标准正态分布采样 x_1 ─────────────────────────────────
        # 对应 t=1.0，纯高斯噪声
        x = torch.randn(shape, device=device)

        # ── 构造时间步序列：从 t=1.0 逐步降到 t=0.0 ─────────────────────────
        # 包含端点，共 num_steps+1 个点，num_steps 个区间
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
        # 步长（每步走多远）
        dt = 1.0 / num_steps

        # ── Euler ODE 求解循环 ────────────────────────────────────────────
        for i in range(num_steps):
            t_curr = timesteps[i].item()    # 当前时间步（浮点）

            if show_progress and i % max(1, num_steps // 5) == 0:
                print(f"  Flow Matching 采样中... {i}/{num_steps} (t={t_curr:.2f})")

            # 将浮点 t 转为整数，传给 UNet 时间嵌入
            t_float = torch.full((shape[0],), t_curr, device=device)
            t_int = self.scheduler.t_to_int(t_float)

            # UNet 预测速度场：v_pred = model(x_t, t)
            # 含义："从当前位置 x_t，应该往哪个方向走才能到 x_0"
            v_pred = model(x, t_int)

            # Euler 步：沿速度场反向走一步
            # x_{t-dt} = x_t - dt * v_pred
            x = self.scheduler.euler_step(x, t_curr, v_pred, dt)

        # ── 后处理 ─────────────────────────────────────────────────────────
        if self.scheduler.config.clip_output:
            x = x.clamp(-1.0, 1.0)

        return x

    @torch.no_grad()
    def sample_midpoint(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        num_steps: int = 10,
        device: str | torch.device = "cpu",
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Midpoint 法采样（二阶精度）：同等步数下比 Euler 法质量更好。

        Midpoint 法每步分两个子步：
            1. 用 Euler 法走半步到 t - dt/2（取中点）
            2. 在中点计算速度，用此速度走完整步

        数学上是二阶 Runge-Kutta（RK2），误差 O(dt²) 而非 Euler 的 O(dt)。
        意味着：同样的精度，步数可以减半；或者同样的步数，精度更高。

        示意图（单步从 t → t-dt）：
                          中点速度 v_mid
                         ↗ 更准确
            x_t ─────────────────────────▶ x_{t-dt}
                  ↗
               Euler 预估 x_mid（仅用于估计速度）

        参数:
            model     (nn.Module): 已训练好的 UNet。
            shape     (tuple): 生成图像的形状。
            num_steps (int): ODE 求解步数（Midpoint 法可以比 Euler 少一半步数）。
            device    : 计算设备。
            show_progress (bool): 是否打印进度。
        返回:
            torch.Tensor: 生成的图像，形状 shape。
        """
        model.eval()

        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_curr = timesteps[i].item()

            if show_progress and i % max(1, num_steps // 5) == 0:
                print(f"  Flow Matching（Midpoint）采样中... {i}/{num_steps} (t={t_curr:.2f})")

            # ── 子步 1：Euler 半步，到中点 ────────────────────────────────
            t_float = torch.full((shape[0],), t_curr, device=device)
            t_int = self.scheduler.t_to_int(t_float)
            v1 = model(x, t_int)
            x_mid = x - (dt / 2.0) * v1       # 走到 t_curr - dt/2

            # ── 子步 2：在中点计算速度，用它走完整步 ──────────────────────
            t_mid = t_curr - dt / 2.0
            t_mid_float = torch.full((shape[0],), max(t_mid, 0.0), device=device)
            t_mid_int = self.scheduler.t_to_int(t_mid_float)
            v2 = model(x_mid, t_mid_int)
            x = x - dt * v2                    # 用中点速度走完整步

        if self.scheduler.config.clip_output:
            x = x.clamp(-1.0, 1.0)

        return x
