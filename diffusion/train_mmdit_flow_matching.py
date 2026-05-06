"""
VideoMMDiT + Flow Matching 训练脚本 —— 学习用简化版
=====================================================

本脚本展示如何把 VideoMMDiT、3D VAE、Flow Matching 串成完整训练管线。

完整流程:
    原始视频 → 3D VAE 编码 → z_0
        ↓
    Flow Matching 训练:
        1. 采样噪声 z_1 ~ N(0, I)
        2. 采样时间步 t ~ [0, 1]
        3. 插值: z_t = (1-t)*z_0 + t*z_1
        4. MMDiT 预测: v_pred = model(z_t, t, text_emb)
        5. 目标: v_target = z_1 - z_0
        6. Loss: MSE(v_pred, v_target)
        7. 反向传播

用法:
    python diffusion/train_mmdit_flow_matching.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 假设这些模块已经实现
from diffusion.vae.video_vae_minimal import SimpleVideoVAE, vae_loss
from diffusion.dit.mmdit_vedio import VideoMMDiT, VideoMMDiTConfig
from diffusion.dit.video_flow_matching import (
    VideoFlowMatchingConfig,
    VideoFlowMatchingScheduler,
    VideoFlowMatchingTrainer,
    VideoFlowMatchingSampler,
)


def pretrain_vae(vae: SimpleVideoVAE, dataloader: DataLoader, epochs: int = 10):
    """阶段 1: 预训练 3D VAE（独立训练）。"""
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    vae.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (video,) in enumerate(dataloader):
            optimizer.zero_grad()

            # VAE 前向: 视频 → 重建视频
            out = vae(video)

            # 计算损失: 重建 + KL
            losses = vae_loss(
                out["recon"], video, out["mu"], out["logvar"], kl_weight=1e-4
            )

            losses["total"].backward()
            optimizer.step()

            total_loss += losses["total"].item()

        avg_loss = total_loss / len(dataloader)
        print(f"  [VAE Epoch {epoch+1}/{epochs}] loss={avg_loss:.4f}")

    print("✓ VAE 预训练完成\n")
    return vae


def train_mmdit(
    model: VideoMMDiT,
    vae: SimpleVideoVAE,
    dataloader: DataLoader,
    trainer: VideoFlowMatchingTrainer,
    epochs: int = 100,
):
    """阶段 2: 训练 VideoMMDiT（VAE 冻结）。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    vae.eval()  # VAE 冻结，只做编码

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (video, text_emb) in enumerate(dataloader):
            optimizer.zero_grad()

            # Step 1: VAE 编码视频 → z_0（确定性，不采样）
            with torch.no_grad():
                z0 = vae.encode_deterministic(video)

            # Step 2: Flow Matching 训练步骤
            # 内部完成: 采样噪声 → 插值 → CFG dropout → MMDiT 预测 → MSE loss
            loss = trainer.training_step(model, z0, text_emb)

            loss.backward()

            # 梯度裁剪（防止 Transformer 训练时梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            # 每 10 个 batch 打印一次
            if batch_idx % 10 == 0:
                print(f"    [Epoch {epoch+1} Batch {batch_idx}] loss={loss.item():.6f}")

        avg_loss = total_loss / len(dataloader)
        print(f"  [MMDiT Epoch {epoch+1}/{epochs}] avg_loss={avg_loss:.6f}\n")

    print("✓ MMDiT 训练完成\n")
    return model


@torch.no_grad()
def generate_video(
    model: VideoMMDiT,
    vae: SimpleVideoVAE,
    sampler: VideoFlowMatchingSampler,
    text_emb: torch.Tensor,
    latent_shape: tuple[int, ...],
    device: str = "cpu",
):
    """推理：从文本生成视频。"""
    model.eval()
    vae.eval()

    # Step 1: Flow Matching 采样 → z_0
    z0 = sampler.sample_euler_cfg(
        model=model,
        text_emb=text_emb,
        shape=latent_shape,
        num_steps=20,      # 20步采样
        cfg_scale=7.0,     # CFG 引导强度
        device=device,
        show_progress=True,
    )

    # Step 2: VAE 解码 → 视频像素
    video = vae.decode(z0)

    return video


def main():
    """主函数：完整训练 + 推理演示。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}\n")

    # ═══════════════════════════════════════════════════════════════════════
    # 构造假数据（真实场景应换成你的视频数据集）
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("准备数据")
    print("=" * 60)

    # 假视频数据: (N, 3, 8, 64, 64)
    num_samples = 32
    fake_videos = torch.randn(num_samples, 3, 8, 64, 64)
    # 假文本嵌入: (N, 256, 4096) —— 真实场景用 T5/CLIP 编码
    fake_texts = torch.randn(num_samples, 256, 4096)

    dataset = TensorDataset(fake_videos, fake_texts)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"  样本数: {num_samples}, Batch size: 4\n")

    # ═══════════════════════════════════════════════════════════════════════
    # 阶段 1: 预训练 VAE
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("阶段 1: 预训练 3D VAE")
    print("=" * 60)

    vae = SimpleVideoVAE().to(device)
    vae = pretrain_vae(vae, dataloader, epochs=5)

    # ═══════════════════════════════════════════════════════════════════════
    # 阶段 2: 训练 VideoMMDiT
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("阶段 2: 训练 VideoMMDiT + Flow Matching")
    print("=" * 60)

    # 创建模型
    mmdit_cfg = VideoMMDiTConfig(
        latent_t=8,
        latent_h=64,
        latent_w=64,
        d_model=256,          # 小模型，学习用
        n_heads=8,
        n_double_layers=4,
        n_single_layers=4,
    )
    model = VideoMMDiT(mmdit_cfg).to(device)

    # 创建 Flow Matching 训练器
    fm_cfg = VideoFlowMatchingConfig(
        num_train_timesteps=1000,
        t_sample_mode="logit_normal",
        p_uncond=0.1,         # 10% 概率丢弃文本（CFG 训练）
        cfg_scale=7.0,
    )
    scheduler = VideoFlowMatchingScheduler(fm_cfg, device=device)
    trainer = VideoFlowMatchingTrainer(scheduler)

    model = train_mmdit(model, vae, dataloader, trainer, epochs=10)

    # ═══════════════════════════════════════════════════════════════════════
    # 阶段 3: 推理生成
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("阶段 3: 推理生成视频")
    print("=" * 60)

    sampler = VideoFlowMatchingSampler(scheduler)

    # 构造一个文本提示（假数据，真实场景用 T5 编码）
    prompt_text_emb = torch.randn(1, 256, 4096).to(device)

    # latent 形状: (B=1, C=4, T=8, H=64, W=64)
    latent_shape = (1, 4, 8, 64, 64)

    video = generate_video(
        model=model,
        vae=vae,
        sampler=sampler,
        text_emb=prompt_text_emb,
        latent_shape=latent_shape,
        device=device,
    )

    print(f"\n✓ 生成完成！视频形状: {video.shape}")
    print(f"  即 (B={video.shape[0]}, C={video.shape[1]}, "
          f"T={video.shape[2]}, H={video.shape[3]}, W={video.shape[4]})")


if __name__ == "__main__":
    main()
