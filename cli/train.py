"""
Transformer 语言模型训练脚本 —— 从"小白"到"看懂日志"的完整指南
===============================================================

本文件是模型的"健身房教练"，负责：
  1. 准备数据（食材）
  2. 搭建模型（健身者）
  3. 设计训练计划（学习率、批次大小等）
  4. 执行训练（反复练习）
  5. 评估效果（考试打分）
  6. 保存最好的模型（颁发证书）

═══════════════════════════════════════════════════════════════════
【核心概念速查】训练前必看！
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│  Epoch（轮/周期）                                               │
│    把整个训练数据集完整过一遍 = 1 个 epoch                       │
│    例: 数据集有 10000 条文本, batch_size=4                      │
│        → 每轮有 10000/4 = 2500 个 batch（步）                   │
│        → 训练 10 轮 = 模型看了 10 遍全部数据                     │
│                                                                 │
│    类比: 学生复习课本, 第1轮粗略看, 第10轮烂熟于心              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Batch（批次）和 Batch Size                                     │
│    模型一次处理多少条数据。不是越大越好！                        │
│                                                                 │
│    batch_size=1:  一次学1个例子, 更新快但噪声大                 │
│    batch_size=4:  一次学4个例子, 平均后梯度更稳（推荐起步）     │
│    batch_size=32: 显存占用大, 但梯度更稳定                      │
│                                                                 │
│    显存不足? → 减小 batch_size 或增大 grad_accum_steps          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Gradient Accumulation（梯度累积）                              │
│    显存不够放 batch_size=8? 但可以放 batch_size=2?              │
│    → 每步算 batch_size=2, 累积4次后再更新参数                    │
│    → 等效 batch_size = 2 × 4 = 8                                │
│                                                                 │
│    好处: 用小显存模拟大 batch                                    │
│    代价: 每4步才更新一次, 训练稍慢                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Loss（损失）—— 模型"错得多离谱"                                 │
│                                                                 │
│    想象你在做选择题, 模型对每个词的预测是一个概率分布。          │
│    Loss 衡量: 模型给正确答案的概率有多低?                        │
│                                                                 │
│    交叉熵公式:  loss = -log(模型给正确答案的概率)               │
│                                                                 │
│    数值含义:                                                    │
│      loss=0.1   → 模型对正确答案很有信心 (~90%概率) → 很好     │
│      loss=1.0   → 模型有点拿不准 (~37%概率) → 一般             │
│      loss=3.0   → 模型基本在瞎猜 (~5%概率) → 很差              │
│      loss=4.16  → 完全随机猜（词表大小约64的均匀分布）         │
│                                                                 │
│    注意: loss 是"负数对数概率", 越低越好！                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Train Loss vs Val Loss（训练损失 vs 验证损失）                 │
│                                                                 │
│    Train Loss: 模型在"练习题"上的表现                           │
│    Val Loss:   模型在"模拟考"上的表现（没见过的新题）            │
│                                                                 │
│    ┌────────────────────────────────────────────────────┐       │
│    │  正常情况:                                          │       │
│    │    Train Loss ↓  Val Loss ↓  两者差距小            │       │
│    │    → 模型在学习和泛化, 继续训练                     │       │
│    │                                                     │       │
│    │  过拟合警告:                                        │       │
│    │    Train Loss ↓↓  Val Loss ↑ 或持平                │       │
│    │    → 模型在"背答案"而不是"学规律"                 │       │
│    │    → 立刻停！保存上一轮模型                         │       │
│    │                                                     │       │
│    │  欠拟合:                                            │       │
│    │    Train Loss 和 Val Loss 都居高不下                │       │
│    │    → 模型容量不够或训练不够                         │       │
│    │    → 增加层数/维度 或 延长训练时间                  │       │
│    └────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Learning Rate（学习率）—— 步长                                 │
│                                                                 │
│    模型更新参数时"迈多大步"。                                    │
│    太大: 一步迈过头, 来回震荡, 甚至发散                         │
│    太小: 走得慢, 半天到不了最优解                               │
│                                                                 │
│    三种调度策略:                                                │
│      1. const:        固定不变 (适合微调/短训)                  │
│      2. warmup_const: 先小步走(预热), 然后大步走(恒速)          │
│                        推荐！最稳                               │
│      3. transformer:  预热 → 峰值 → 衰减 (原论文设计)           │
│                        适合超大规模训练                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Perplexity（困惑度, PPL）—— 更直观的指标                       │
│                                                                 │
│    PPL = exp(loss), 表示模型"平均面对多少种等概率选择"          │
│                                                                 │
│    PPL=64  → 模型每次选词像从64个等概率词里猜 → 还行           │
│    PPL=100 → 像从100个里猜 → 一般                              │
│    PPL=20  → 像从20个里猜 → 不错                               │
│    PPL=5   → 像从5个里猜 → 很好                                │
│                                                                 │
│    计算公式: PPL = e^loss                                       │
│    例: loss=2.0 → PPL ≈ 7.4                                    │
│        loss=3.5 → PPL ≈ 33.1                                   │
│                                                                 │
│    PPL 的好处: 比 loss 更直观, 数值范围更好理解                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  混合精度训练（Mixed Precision）                                │
│                                                                 │
│    正常计算用 float32（4字节/数）                               │
│    混合精度用 float16 或 bfloat16（2字节/数）                   │
│                                                                 │
│    好处: 显存减半 + 计算更快（Tensor Core 加速）                │
│    注意:                                                        │
│      - fp16:  范围小, 大数可能溢出, 需要 GradScaler 保护       │
│      - bf16:  范围和 fp32 一样, 更稳, 推荐！                   │
│      - 只在 NVIDIA GPU 上有效                                   │
└─────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
【如何看懂训练日志】
═══════════════════════════════════════════════════════════════════

每轮结束后你会看到:
  第3/10轮 | 训练损失=2.3456 | 验证损失=2.6789 | 用时=45.2s

解读:
  "第3/10轮"       → 总共训练10轮, 当前是第3轮
  "训练损失=2.3456" → 模型在练习题上的平均loss
  "验证损失=2.6789" → 模型在模拟考上的平均loss
  "用时=45.2s"      → 这轮训练花了45秒

中间还会打印更细粒度的日志（每 log_interval 步）:
  第3轮 | step=150 | batch=500/2500 | loss=2.1234 | 平均损失=2.4567 | 学习率=3.00e-04

解读:
  "step=150"      → 全局第150步（所有epoch累加）
  "batch=500/2500"→ 当前epoch的第500个batch, 共2500个
  "loss=2.1234"   → 当前这个batch的loss
  "平均损失=2.4567"→ 本epoch到目前为止的平均loss
  "学习率=3.00e-04"→ 当前学习率 = 0.0003

═══════════════════════════════════════════════════════════════════
【判断训练好坏的黄金法则】
═══════════════════════════════════════════════════════════════════

✅ 好现象:
   • 训练损失和验证损失都在稳定下降
   • 两者差距不大（< 0.5 左右）
   • 每轮验证损失都在刷新最低记录
   • 生成的文本越来越通顺、相关

⚠️ 过拟合（立刻停！）:
   • 训练损失继续下降, 验证损失开始上升
   • 两者差距越来越大
   • 生成文本像复读机, 直接复制训练数据

⚠️ 欠拟合:
   • 训练损失和验证损失都很高（> 3.0）
   • 多轮之后下降非常缓慢
   • 生成文本完全是胡言乱语

🔧 调节方向:
   • 过拟合 → 减小模型（层数/维度）、加dropout、早停
   • 欠拟合 → 增大模型、延长训练、调高学习率

"""

from __future__ import annotations

import argparse
import contextlib
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

from config import ModelConfig, TrainConfig
from data.dataset import (
    create_dataloaders,
    create_qa_dataloaders,
    download_tiny_shakespeare,
)
from data.tokenizer import ensure_tokenizer
from model import Transformer
from model.moe_feedforward import collect_moe_load_balance_loss
from utils import (
    configure_runtime,
    count_parameters,
    get_amp_dtype,
    get_device,
    get_logger,
    load_checkpoint,
    recommend_batch_size,
    save_checkpoint,
    save_moe_experts,
)

log = get_logger()


def _dataloader_len(loader) -> int | None:
    """获取 DataLoader 的长度，流式数据集返回 None。"""
    try:
        return len(loader)
    except TypeError:
        return None


def get_lr(step: int, d_model: int, warmup_steps: int) -> float:
    """
    Transformer 原论文的学习率调度公式（Vaswani et al., 2017）。

    公式: lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))

    理解这个公式:
    ─────────────────────────────────────────────────────────────────
    分两段:
      1. 预热期 (step < warmup_steps):
         lr ∝ step        → 学习率线性增长
         就像开车: 先慢慢加速, 避免冲出轨道

      2. 衰减期 (step >= warmup_steps):
         lr ∝ step^(-0.5) → 学习率按平方根衰减
         就像快到目的地: 慢慢减速, 精细调整

    为什么除以 d_model^0.5?
      模型越大(d_model越大), 梯度越小, 需要更大的学习率补偿。
      这个因子让不同规模的模型用相似的学习率范围。

    例: d_model=512, warmup=4000
        step=100:   lr ∝ 100/4000^1.5      → 很小, 预热中
        step=4000:  lr ∝ 1/4000^0.5        → 峰值
        step=16000: lr ∝ 1/16000^0.5       → 峰值的一半
    ─────────────────────────────────────────────────────────────────
    """
    step = max(step, 1)
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def compute_lr(step: int, train_cfg: TrainConfig, model_cfg: ModelConfig) -> float:
    """
    根据配置计算当前步的学习率。

    三种策略对比:
    ┌─────────────────────────────────────────────────────────────────┐
    │  const (固定):                                                  │
    │    lr = learning_rate （从头到尾不变）                          │
    │    适用: 微调、短训、或你已经知道最佳学习率                     │
    │                                                                 │
    │  warmup_const (推荐):                                          │
    │    step < warmup:  lr 从 0 线性增长到 learning_rate           │
    │    step >= warmup: lr = learning_rate （恒速）                  │
    │    适用: 大多数训练场景, 最稳定                                 │
    │                                                                 │
    │  transformer (原论文):                                         │
    │    预热增长 → 达到峰值 → 平方根衰减                            │
    │    适用: 大规模预训练, 训练步数极多                             │
    └─────────────────────────────────────────────────────────────────┘
    """
    schedule = train_cfg.lr_schedule.lower()
    if schedule in {"const", "constant"}:
        lr = train_cfg.learning_rate
    elif schedule in {"warmup_const", "warmup-const", "warmup_constant"}:
        warmup = max(int(train_cfg.warmup_steps), 1)
        lr = train_cfg.learning_rate * min(step / warmup, 1.0)
    elif schedule == "transformer":
        lr = train_cfg.learning_rate * get_lr(
            step, model_cfg.d_model, train_cfg.warmup_steps
        )
    else:
        raise ValueError(f"Unknown lr_schedule: {train_cfg.lr_schedule}")
    # 学习率不能低于设定的最小值（防止衰减到0导致训练停滞）
    return max(float(train_cfg.min_lr), float(lr))


# ═══════════════════════════════════════════════════════════════════
# 训练一个周期 (Epoch)
# ═══════════════════════════════════════════════════════════════════
def train_one_epoch(
    model: Transformer,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    global_step: int,
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    scaler: torch.amp.GradScaler | None,
    amp_dtype: torch.dtype | None,
) -> tuple[float, int]:
    """
    训练一个 epoch 的完整流程。

    返回:
        (平均训练损失, 更新后的全局步数)
    """
    model.train()  # 设为训练模式（启用 dropout 等）
    total_loss = 0.0
    n_batches = 0

    # ==========================================================
    # Step 0: 清空梯度（epoch 开始时）
    #
    # 为什么? 梯度是累加的。上一步的残留梯度会干扰当前计算。
    # set_to_none=True: 比填0更省显存
    # ==========================================================
    optimizer.zero_grad(set_to_none=True)
    use_amp = device.type == "cuda" and amp_dtype is not None
    # non_blocking=True: GPU 异步数据传输, 不阻塞 CPU
    transfer_kwargs = {"non_blocking": True} if device.type == "cuda" else {}

    for batch_idx, (input_ids, labels) in enumerate(loader):
        # ==========================================================
        # Step 1: 数据搬运（CPU → GPU）
        #
        # input_ids: (B, seq)   例: (4, 128)  4个样本, 每个128个token
        # labels:    (B, seq)   和 input_ids 形状相同
        #
        # 注意: input_ids 和 labels 的关系
        #   input_ids: [今天, 天气, 真, 好]
        #   labels:    [天气, 真, 好, </s>]
        #   模型根据"今天"预测"天气", 根据"天气"预测"真"...
        # ==========================================================
        input_ids = input_ids.to(device, **transfer_kwargs)
        labels = labels.to(device, **transfer_kwargs)

        # ==========================================================
        # Step 2: 更新全局步数和学习率
        #
        # global_step: 累计步数, 用于学习率调度
        # 例: epoch=3, batch_idx=100, 每轮2500步
        #     → global_step = 3×2500 + 100 = 7600
        # ==========================================================
        global_step += 1
        lr = compute_lr(global_step, train_cfg, model_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        try:
            # ==========================================================
            # Step 3: 前向传播（Forward）
            #
            # input_ids (4, 128) → model → logits (4, 128, vocab_size)
            #
            # logits 含义: 每个位置对每个词的"得分"
            #   logits[0, 5, 123] = 2.5
            #   → 第0个样本第5个位置, 词表第123号词的得分是2.5
            #
            # autocast: 混合精度自动转换（仅 CUDA 启用时）
            #   fp16/bf16 计算更快更省显存, 但结果精度稍低
            # ==========================================================
            with (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp
                else contextlib.nullcontext()
            ):
                logits = model(input_ids)

                # ==========================================================
                # Step 4: 计算损失（Loss）
                #
                # reshape 原因:
                #   logits: (4, 128, vocab_size) → (512, vocab_size)
                #   labels: (4, 128)             → (512,)
                #   CrossEntropyLoss 期望: (N, C) 和 (N,)
                #   N=512 表示 512 个独立的"预测-答案"对
                #
                # ignore_index=-100:
                #   QA 训练时, 问题部分的 label 设为 -100, 不计入损失
                #   只让模型学习"回答"部分
                #
                # MoE 额外损失:
                #   如果启用了 MoE, 加上负载均衡损失
                #   目的: 让每个专家都被用到, 不要所有 token 挤到一个专家
                # ==========================================================
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                )
                if model_cfg.use_moe:
                    loss = (
                        loss
                        + model_cfg.moe_lb_coeff * collect_moe_load_balance_loss(model)
                    )
        except RuntimeError as exc:
            if device.type == "cuda" and "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "训练阶段发生 CUDA OOM。请减小 --batch-size，或增大 --grad-accum-steps，"
                    "也可以继续使用默认的安全模式。"
                ) from exc
            raise

        # ==========================================================
        # Step 5: 损失归一化（梯度累积用）
        #
        # 为什么除以 grad_accum_steps?
        #   假设等效 batch_size = 4 × 4 = 16
        #   每步算 loss, 但4步后才更新参数
        #   如果不除4, 梯度就是正常值的4倍, 导致步长过大
        #   除4后, 4步累加的梯度 = 正常一次16样本的梯度
        # ==========================================================
        scaled_loss = loss / train_cfg.grad_accum_steps

        # ==========================================================
        # Step 6: 反向传播（Backward）—— 计算梯度
        #
        # scaler.scale(): fp16 训练时, 先放大损失再反向传播
        #   防止 fp16 精度不够导致梯度下溢为0
        # .backward(): 计算每个参数的梯度, 存到 param.grad
        # ==========================================================
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # ==========================================================
        # Step 7: 参数更新（Optimizer Step）—— 梯度累积触发
        #
        # 只有每 grad_accum_steps 个 batch 才更新参数
        # 例: grad_accum_steps=4
        #   batch 0: 算梯度, 不累加 → grad += grad_0
        #   batch 1: 算梯度, 不累加 → grad += grad_1
        #   batch 2: 算梯度, 不累加 → grad += grad_2
        #   batch 3: 算梯度, 累加 → grad += grad_3
        #            → unscale + clip + step + zero_grad
        #
        # clip_grad_norm_(max_grad_norm=1.0):
        #   梯度裁剪。防止某些异常 batch 导致梯度爆炸
        #   如果梯度范数 > 1.0, 就整体缩放到1.0
        #   就像给车装限速器
        # ==========================================================
        should_step = (batch_idx + 1) % train_cfg.grad_accum_steps == 0
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)  # 把缩放的梯度还原
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)  # 如果梯度不是 NaN, 执行更新
                scaler.update()          # 更新缩放因子
            else:
                optimizer.step()         # 更新参数: w = w - lr × grad
            optimizer.zero_grad(set_to_none=True)

        # ==========================================================
        # Step 8: 记录损失
        # ==========================================================
        total_loss += loss.item()
        n_batches += 1

        # ==========================================================
        # Step 9: 打印日志（按 log_interval 间隔）
        #
        # 输出示例:
        #   第3轮 | step=150 | batch=500/2500 | loss=2.1234 | 平均损失=2.4567 | 学习率=3.00e-04
        #
        # loss.item(): 当前 batch 的原始损失（不归一化）
        # avg = total_loss / n_batches: 本 epoch 到目前为止的平均损失
        # ==========================================================
        if global_step % train_cfg.log_interval == 0:
            avg = total_loss / n_batches
            ntot = _dataloader_len(loader)
            batch_info = (
                f"{batch_idx + 1}/{ntot}" if ntot is not None else str(batch_idx + 1)
            )
            log.info(
                f"第{epoch}轮 | step={global_step} | batch={batch_info} | "
                f"loss={loss.item():.4f} | 平均损失={avg:.4f} | 学习率={lr:.2e}"
            )

    # ==========================================================
    # Step 10: 处理残余梯度
    #
    # 如果最后一个 epoch 的 batch 数不是 grad_accum_steps 的整数倍,
    # 会剩下一些梯度没有更新。这里补一次更新。
    # ==========================================================
    if n_batches > 0 and n_batches % train_cfg.grad_accum_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # 返回本 epoch 的平均损失和更新后的全局步数
    return total_loss / max(n_batches, 1), global_step


# ═══════════════════════════════════════════════════════════════════
# 验证（Validation）—— "模拟考试"
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()  # 不计算梯度, 节省显存和计算
def evaluate(
    model: Transformer,
    loader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> float:
    """
    在验证集上评估模型。

    和训练的区别:
      1. model.eval() —— 关闭 dropout, 不要"蒙眼答题"
      2. torch.no_grad() —— 不计算梯度, 只推理
      3. 不做参数更新

    返回: 验证集平均损失（越低越好）
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    use_amp = device.type == "cuda" and amp_dtype is not None
    transfer_kwargs = {"non_blocking": True} if device.type == "cuda" else {}

    for input_ids, labels in loader:
        input_ids = input_ids.to(device, **transfer_kwargs)
        labels = labels.to(device, **transfer_kwargs)

        with (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if use_amp
            else contextlib.nullcontext()
        ):
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def detect_mode(data_path: str | None) -> str:
    """根据文件扩展名自动检测数据模式：.jsonl/.tsv为'qa'，否则为'text'。"""
    if data_path is None:
        return "text"
    p = Path(data_path)
    if p.is_file() and p.suffix in (".jsonl", ".tsv"):
        return "qa"
    if p.is_dir():
        has_qa = list(p.glob("*.jsonl")) + list(p.glob("*.tsv"))
        if has_qa:
            return "qa"
    return "text"


# ═══════════════════════════════════════════════════════════════════
# 命令行参数解析
# ═══════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练 decoder-only Transformer 语言模型",
        epilog="""
使用示例:
  # 基础训练（自动下载 Tiny Shakespeare 数据）
  python __main__.py train

  # 用自己的文本训练
  python __main__.py train --data-path data/my_corpus.txt --seq-len 256

  # QA 模式训练
  python __main__.py train --data-path data/qa.jsonl --mode qa

  # 启用 MoE + 自定义专家数
  python __main__.py train --use-moe --moe-experts 8 --d-model 512

  # 从 checkpoint 继续训练
  python __main__.py train --resume-from checkpoints/run1/best_model.pt --num-epochs 10

  # 安全模式（自动限制 batch_size, 适合显存小的 GPU）
  python __main__.py train --safe-mode
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ──────────────────────────────────────────────────────────────
    # 数据相关参数
    # ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="训练数据路径。text模式: .txt 或目录；含 title/text 的维基 JSONL（.json/.jsonl）会自动流式读取；qa模式: .jsonl 或 .tsv。",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["text", "qa"],
        help="数据模式：'text'（连贯文本, 学语言规律）或'qa'（问答对, 学对话）。未填写时自动检测。",
    )

    # ──────────────────────────────────────────────────────────────
    # 训练超参数 —— 控制"训练计划"
    # ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="每步处理的样本数。显存够就调大（4→8→16）, 训练更稳定。默认从 config.py 读取。",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="训练总轮数。数据集大就少设几轮（如3-5）, 小就多设（如20-50）。",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="初始学习率。默认3e-4。如果 loss 震荡 → 调小（如1e-4）; 如果 loss 不降 → 调大（如1e-3）。",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="预热步数。学习率从0线性增长到初始值的步数。推荐: 总步数的5%-10%。防止初期步长太大。",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=None,
        help="学习率下限。防止学习率衰减到0导致训练停滞。默认1e-6。",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default=None,
        choices=["transformer", "warmup_const", "const"],
        help="""
学习率调度策略:
  const:        固定不变。适合微调。
  warmup_const: 先预热再恒定。最稳, 推荐！
  transformer:  预热→峰值→衰减。适合大规模预训练。
        """.strip(),
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="训练序列长度。每个样本包含多少个 token。越长模型能学的关系越远, 但显存占用越大。",
    )

    # ──────────────────────────────────────────────────────────────
    # 模型结构参数 —— 控制"模型大小"
    # ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--d-model",
        type=int,
        default=None,
        help="模型维度。每个 token 的向量维度。越大模型越强, 但越慢越吃显存。默认768。",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=None,
        help="注意力头数。必须能整除 d_model。越多注意力越丰富, 但计算量越大。默认12。",
    )
    parser.add_argument(
        "--dec-layers",
        type=int,
        default=None,
        help="Decoder 层数。模型深度。越深抽象能力越强, 但训练越慢。默认10。",
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        default=None,
        help="FFN 中间层维度。通常是 d_model 的 4 倍（768→3072）。越大 FFN 容量越大。",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout 比例。防止过拟合的正则化手段。训练时随机丢弃一部分神经元。默认0.1。",
    )

    # ──────────────────────────────────────────────────────────────
    # 输出与恢复参数
    # ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="训练产物输出目录, 包含 checkpoint 和 tokenizer。",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="从指定 checkpoint 继续训练。--num-epochs 表示在此基础上再训练多少轮。",
    )

    # ──────────────────────────────────────────────────────────────
    # 分词器参数
    # ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="已训练好的分词器模型文件路径。如果不存在会报错, 请先训练分词器。",
    )
    parser.add_argument(
        "--tokenizer-vocab-size",
        type=int,
        default=None,
        help="分词器词表大小。越大表示用更多子词, 越精细。默认32000。",
    )
    parser.add_argument(
        "--tokenizer-model-type",
        type=str,
        default=None,
        choices=["unigram", "bpe", "char", "word"],
        help="分词算法类型。bpe=字节对编码（最常用）, unigram=更灵活, char=字符级。",
    )
    parser.add_argument(
        "--retrain-tokenizer",
        action="store_true",
        help="强制重新训练分词器（即使已有分词器文件）。",
    )

    # ──────────────────────────────────────────────────────────────
    # 高级训练参数
    # ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="""
梯度累积步数。等效 batch_size = batch_size × grad_accum_steps。
显存不够时用这个模拟大 batch。
例: --batch-size 2 --grad-accum-steps 4 → 等效 batch_size=8
        """.strip(),
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default=None,
        choices=["auto", "fp16", "bf16", "off"],
        help="""
混合精度策略（仅 CUDA）:
  auto: 自动选择 bf16（推荐, 最稳）
  fp16: 半精度浮点, 快但需 GradScaler
  bf16: 脑浮点, 范围大更稳定, 推荐！
  off:  关闭, 用 float32
        """.strip(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="训练设备。auto=自动检测 GPU, 没有就用 CPU。",
    )
    parser.add_argument(
        "--safe-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="安全模式。自动限制 batch_size, 禁用 cuDNN benchmark, 适合显存小或驱动不稳定的场景。",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="训练结束后用来生成样本的起始文本。例: --prompt '今天天气'",
    )

    # ──────────────────────────────────────────────────────────────
    # MoE 参数
    # ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--use-moe",
        action="store_true",
        help="启用稀疏 MoE FFN。每个 token 只过 1 个专家, 总参数增加但计算量不变。",
    )
    parser.add_argument(
        "--moe-experts",
        type=int,
        default=None,
        help="MoE 专家数量。默认4。越多容量越大, 但负载均衡越难。",
    )
    parser.add_argument(
        "--moe-lb-coeff",
        type=float,
        default=None,
        help="负载均衡损失系数。越大越强制均匀分配, 但可能干扰主任务学习。默认0.01。",
    )

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════
# 主函数 —— 训练流程的"总指挥"
# ═══════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    # 加载默认配置
    model_cfg = ModelConfig()   # 模型结构配置
    train_cfg = TrainConfig()   # 训练超参数配置

    # ==========================================================
    # 命令行参数覆盖默认配置
    # ==========================================================
    if args.data_path is not None:
        train_cfg.data_path = args.data_path
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.num_epochs is not None:
        train_cfg.num_epochs = args.num_epochs
    if args.learning_rate is not None:
        train_cfg.learning_rate = args.learning_rate
    if args.warmup_steps is not None:
        train_cfg.warmup_steps = args.warmup_steps
    if args.min_lr is not None:
        train_cfg.min_lr = args.min_lr
    if args.lr_schedule is not None:
        train_cfg.lr_schedule = args.lr_schedule
    if args.seq_len is not None:
        train_cfg.seq_len = args.seq_len
    if args.d_model is not None:
        model_cfg.d_model = args.d_model
    if args.n_heads is not None:
        model_cfg.n_heads = args.n_heads
    if args.dec_layers is not None:
        model_cfg.n_decoder_layers = args.dec_layers
    if args.d_ff is not None:
        model_cfg.d_ff = args.d_ff
    if args.dropout is not None:
        model_cfg.dropout = args.dropout
    if args.output_dir is not None:
        train_cfg.checkpoint_dir = args.output_dir
    if args.tokenizer_path is not None:
        train_cfg.tokenizer_path = args.tokenizer_path
    if args.tokenizer_vocab_size is not None:
        train_cfg.tokenizer_vocab_size = args.tokenizer_vocab_size
    if args.tokenizer_model_type is not None:
        train_cfg.tokenizer_model_type = args.tokenizer_model_type
    if args.grad_accum_steps is not None:
        train_cfg.grad_accum_steps = args.grad_accum_steps
    if args.mixed_precision is not None:
        train_cfg.mixed_precision = args.mixed_precision
    if args.safe_mode is not None:
        train_cfg.safe_mode = args.safe_mode
    if args.use_moe:
        model_cfg.use_moe = True
    if args.moe_experts is not None:
        model_cfg.moe_num_experts = args.moe_experts
    if args.moe_lb_coeff is not None:
        model_cfg.moe_lb_coeff = args.moe_lb_coeff

    # ═══════════════════════════════════════════════════════════
    # Phase 1: 基础配置与初始化
    # ═══════════════════════════════════════════════════════════

    # 1.1 梯度累积步数至少为1
    train_cfg.grad_accum_steps = max(train_cfg.grad_accum_steps, 1)

    # 1.2 d_model 必须能被 n_heads 整除（否则 d_k 不是整数）
    if model_cfg.d_model % model_cfg.n_heads != 0:
        raise ValueError(
            f"d_model ({model_cfg.d_model}) 必须能被 n_heads ({model_cfg.n_heads}) 整除"
        )

    # 1.3 设置随机种子（保证实验可复现）
    # 同样的种子 → 同样的初始化 → 同样的训练结果
    torch.manual_seed(train_cfg.seed)
    random.seed(train_cfg.seed)

    # 1.4 获取训练设备
    device = get_device(args.device)
    configure_runtime(device, safe_mode=train_cfg.safe_mode)
    amp_dtype = get_amp_dtype(device, train_cfg.mixed_precision)

    # 1.5 安全模式: 根据显存自动限制 batch_size
    if train_cfg.safe_mode:
        safe_batch_size = recommend_batch_size(device, train_cfg.batch_size)
        if safe_batch_size != train_cfg.batch_size:
            log.warning(
                f"安全模式已将 batch_size 从 {train_cfg.batch_size} 调整为 {safe_batch_size}，"
                "以降低显存与驱动压力。"
            )
            train_cfg.batch_size = safe_batch_size

    # 1.6 打印训练配置概览
    log.info(f"当前使用设备: {device}")
    log.info(
        f"训练设置: batch_size={train_cfg.batch_size}, "
        f"grad_accum_steps={train_cfg.grad_accum_steps}, "
        f"等效 batch_size={train_cfg.batch_size * train_cfg.grad_accum_steps}, "
        f"mixed_precision={train_cfg.mixed_precision}, safe_mode={train_cfg.safe_mode}"
    )
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        log.info(
            f"CUDA 设备: {props.name} | 显存 {props.total_memory / 1024**3:.1f} GB | "
            f"AMP={'off' if amp_dtype is None else str(amp_dtype).replace('torch.', '')}"
        )

    # ═══════════════════════════════════════════════════════════
    # Phase 2: 数据与分词器准备
    # ═══════════════════════════════════════════════════════════

    # 2.1 检测数据模式（text 或 qa）
    mode = args.mode or detect_mode(train_cfg.data_path)
    log.info(f"数据模式: {mode}")

    # 2.2 创建输出目录
    output_dir = Path(train_cfg.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2.3 处理分词器路径
    if args.tokenizer_path is None and not train_cfg.tokenizer_path:
        train_cfg.tokenizer_path = str(output_dir / "tokenizer.model")
    tokenizer_path = Path(train_cfg.tokenizer_path)
    tokenizer_source = train_cfg.data_path

    # 2.4 检查分词器文件是否存在
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"未找到 tokenizer 文件: {tokenizer_path}。请通过 --tokenizer-path 指定已经训练好的分词器。"
        )
    log.info(
        f"加载 tokenizer: path={tokenizer_path}, vocab_size={train_cfg.tokenizer_vocab_size}, "
        f"model_type={train_cfg.tokenizer_model_type}"
    )

    # 2.5 加载分词器
    tokenizer = ensure_tokenizer(
        data_path=tokenizer_source,
        tokenizer_path=tokenizer_path,
        vocab_size=train_cfg.tokenizer_vocab_size,
        model_type=train_cfg.tokenizer_model_type,
        retrain=False,
    )
    log.info(f"tokenizer 已就绪: {tokenizer_path}")

    # 2.6 构建数据集和 DataLoader
    log.info("准备数据...")
    if mode == "qa":
        if train_cfg.data_path is None:
            raise ValueError("--data-path 参数在 QA 模式下是必须的")
        train_loader, val_loader, tokenizer = create_qa_dataloaders(
            data_path=train_cfg.data_path,
            max_len=train_cfg.seq_len,
            batch_size=train_cfg.batch_size,
            val_split=train_cfg.val_split,
            tokenizer=tokenizer,
        )
    else:  # text 模式
        train_loader, val_loader, tokenizer = create_dataloaders(
            seq_len=train_cfg.seq_len,
            batch_size=train_cfg.batch_size,
            val_split=train_cfg.val_split,
            data_path=train_cfg.data_path,
            tokenizer=tokenizer,
        )

    # 2.7 保存分词器并打印信息
    model_cfg.vocab_size = tokenizer.vocab_size
    log.info(f"词表大小: {model_cfg.vocab_size}")
    if model_cfg.use_moe:
        log.info(
            f"MoE 已启用: 专家数={model_cfg.moe_num_experts}, "
            f"负载系数={model_cfg.moe_lb_coeff}"
        )
    tokenizer.save(tokenizer_path)
    log.info(f"分词器已保存到: {tokenizer_path}")
    log.info(f"训练产物目录: {output_dir}")

    # ═══════════════════════════════════════════════════════════
    # Phase 3: 模型、优化器、损失函数准备
    # ═══════════════════════════════════════════════════════════

    # 3.1 构建模型
    model = Transformer(model_cfg).to(device)
    n_params = count_parameters(model)
    log.info(f"模型参数总量: {n_params:,} ({n_params / 1e6:.1f}M)")

    # 3.2 创建 Adam 优化器
    # betas=(0.9, 0.98): 一阶动量衰减0.9, 二阶动量衰减0.98
    #   二阶动量衰减慢 → 对学习率的波动更敏感, 需要小心
    # eps=1e-9: 防止除以0
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.learning_rate, betas=(0.9, 0.98), eps=1e-9
    )

    # 3.3 混合精度 GradScaler
    # 只在 CUDA + fp16 时启用。bf16 不需要 scaler。
    scaler = (
        torch.amp.GradScaler(
            "cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16)
        )
        if device.type == "cuda"
        else None
    )

    # 3.4 损失函数: 交叉熵
    # ignore_index=-100: QA 训练时用来 mask 问题部分
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 3.5 训练状态变量
    global_step = 0          # 全局步数（跨 epoch 累计）
    best_val_loss = float("inf")  # 最佳验证损失（越小越好）
    start_epoch = 1          # 起始轮数
    end_epoch = train_cfg.num_epochs  # 结束轮数

    # 3.6 断点恢复
    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
        ckpt = load_checkpoint(resume_path, model, optimizer=optimizer, device=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        end_epoch = start_epoch + train_cfg.num_epochs - 1
        global_step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("loss", float("inf")))
        log.info(
            f"已从 checkpoint 恢复: {resume_path} | 上次 epoch={ckpt.get('epoch', 0)} | "
            f"step={global_step} | best_val_loss={best_val_loss:.4f}"
        )

    # ═══════════════════════════════════════════════════════════
    # Phase 4: 正式训练主循环
    # ═══════════════════════════════════════════════════════════

    log.info(
        f"开始训练，执行 epoch {start_epoch} 到 {end_epoch} "
        f"(本次新增 {train_cfg.num_epochs} 轮)..."
    )
    ntr = _dataloader_len(train_loader)
    nva = _dataloader_len(val_loader)
    if ntr is not None and nva is not None:
        log.info(f"  训练批次数: {ntr}，验证批次数: {nva}")
    else:
        log.info(
            "  数据为 IterableDataset（如本地维基 JSONL），DataLoader 无固定 len；"
            "每 epoch 步数由语料切块数量决定。"
        )

    total_epochs_display = end_epoch
    current_epoch = start_epoch

    try:
        for epoch in range(start_epoch, end_epoch + 1):
            current_epoch = epoch
            t0 = time.time()

            # Step 4.1: 训练一个 epoch
            train_loss, global_step = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                epoch, global_step, train_cfg, model_cfg, scaler, amp_dtype,
            )

            # Step 4.2: 验证
            val_loss = evaluate(model, val_loader, criterion, device, amp_dtype)
            elapsed = time.time() - t0

            if device.type == "cuda":
                torch.cuda.empty_cache()  # 清理显存碎片

            # Step 4.3: 打印本轮总结
            # 同时输出 PPL（困惑度）, 更直观
            train_ppl = math.exp(min(train_loss, 10))  # 防止 overflow
            val_ppl = math.exp(min(val_loss, 10))
            log.info(
                f"第{epoch}/{total_epochs_display}轮 | "
                f"训练损失={train_loss:.4f}(PPL={train_ppl:.1f}) | "
                f"验证损失={val_loss:.4f}(PPL={val_ppl:.1f}) | "
                f"用时={elapsed:.1f}s"
            )

            # Step 4.4: 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, global_step, val_loss,
                    output_dir / "best_model.pt",
                )
                save_moe_experts(model, output_dir / "moe_weights")
                log.info(f"  ✓ 新的最佳模型已保存 (val_loss={val_loss:.4f}, PPL={val_ppl:.1f})")
            else:
                # 验证损失没有改善, 提示过拟合风险
                gap = val_loss - best_val_loss
                if gap > 0.5:
                    log.warning(
                        f"  ⚠ 验证损失已连续未改善, 与最佳差距 {gap:.4f}。"
                        "可能过拟合, 建议考虑提前停止。"
                    )

            # Step 4.5: 定期保存 checkpoint
            if epoch % train_cfg.save_interval == 0:
                save_checkpoint(
                    model, optimizer, epoch, global_step, val_loss,
                    output_dir / f"checkpoint_epoch{epoch}.pt",
                )
                log.info(f"  已保存周期 checkpoint (epoch {epoch})")

    except KeyboardInterrupt:
        # Step 4.6: Ctrl+C 安全中断
        interrupted_path = output_dir / "interrupted.pt"
        log.warning("收到 Ctrl+C，正在保存中断检查点并退出...")
        save_checkpoint(
            model, optimizer, current_epoch, global_step, best_val_loss,
            interrupted_path,
        )
        log.warning(f"已保存中断检查点: {interrupted_path}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return

    log.info(f"训练结束，最佳验证损失: {best_val_loss:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Phase 5: 训练后样本文本生成（可选）
    # ═══════════════════════════════════════════════════════════
    sample_text = args.prompt
    if sample_text is None:
        sample_text = "ROMEO:" if train_cfg.data_path is None else ""
    if sample_text:
        log.info("生成样本文本...")
        prompt_ids = [tokenizer.bos_token_id] + tokenizer.encode(
            sample_text, add_special=False
        )
        src_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(src_tensor, max_len=200, temperature=0.8, top_k=40)
        output_text = tokenizer.decode(generated[0].tolist())
        log.info(f"输入:  {sample_text}")
        log.info(f"输出: {output_text}")
    else:
        log.info("未指定 prompt，跳过样本生成。可通过 --prompt 参数提供。")


if __name__ == "__main__":
    main()
