"""Decoder-only Transformer语言模型的训练脚本。"""

from __future__ import annotations

import argparse
import contextlib
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
)

log = get_logger()


def _dataloader_len(loader) -> int | None:
    try:
        return len(loader)
    except TypeError:
        return None


def get_lr(step: int, d_model: int, warmup_steps: int) -> float:
    """Transformer 的学习率调度 scale（Vaswani 等人）。"""
    step = max(step, 1)
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def compute_lr(step: int, train_cfg: TrainConfig, model_cfg: ModelConfig) -> float:
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
    return max(float(train_cfg.min_lr), float(lr))


# 训练一个周期
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
    scaler: torch.amp.GradScaler | None,  # 混合精度训练时使用，用于动态调整学习率
    amp_dtype: torch.dtype | None,
) -> tuple[float, int]:
    model.train()  # 设置模型为训练模式
    total_loss = 0.0
    n_batches = 0

    # ==========================================================
    # ********* 1. 初始化：清空梯度（epoch开始） *********
    optimizer.zero_grad(set_to_none=True)
    use_amp = device.type == "cuda" and amp_dtype is not None
    transfer_kwargs = {"non_blocking": True} if device.type == "cuda" else {}
    # ==========================================================

    for batch_idx, (input_ids, labels) in enumerate(loader):
        # ==========================================================
        # ********* 2. 数据搬到设备上（每个batch）*********
        input_ids = input_ids.to(device, **transfer_kwargs)
        labels = labels.to(device, **transfer_kwargs)
        # ==========================================================

        # ==========================================================
        # ********* 3. 更新全局步数，并动态调整学习率 *********
        global_step += 1
        lr = compute_lr(global_step, train_cfg, model_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr  # 更新学习率
        # ==========================================================

        try:
            # ==========================================================
            # ********* 4. 前向传播步骤（计算模型输出） *********
            with (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp
                else contextlib.nullcontext()
            ):
                logits = model(input_ids)

                # ==========================================================
                # ********* 5. 计算损失 *********
                # 用criterion比较logits和labels
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                )
                if model_cfg.use_moe:
                    loss = (
                        loss
                        + model_cfg.moe_lb_coeff * collect_moe_load_balance_loss(model)
                    )
                # ==========================================================
        except RuntimeError as exc:
            if device.type == "cuda" and "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "训练阶段发生 CUDA OOM。请减小 --batch-size，或增大 --grad-accum-steps，"
                    "也可以继续使用默认的安全模式。"
                ) from exc
            raise

        # ==========================================================
        # ********* 6. 损失归一化（用于梯度累积） *********
        scaled_loss = loss / train_cfg.grad_accum_steps
        # ==========================================================

        # ==========================================================
        # ********* 7. 反向传播（累计梯度） *********
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        # ==========================================================

        # ==========================================================
        # ********* 8. 判断是否需要更新参数（做step），采用梯度累积 *********
        # 只有每 grad_accum_steps 个 batch 才真正更新一次参数
        should_step = (batch_idx + 1) % train_cfg.grad_accum_steps == 0
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        # ==========================================================

        # ==========================================================
        # ********* 9. 记录损失和批次数 *********
        total_loss += loss.item()
        n_batches += 1
        # ==========================================================

        # ==========================================================
        # ********* 10. 日志输出 *********
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

    # ==========================================================
    # ********* 11. epoch最后的残余梯度做一次参数更新（若有需要）*********
    # 假如剩下的batch数不足grad_accum_steps，也要确保做一次优化
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
    # ==========================================================

    # ==========================================================
    # ********* 12. 返回该轮的平均损失和新的global_step *********
    return total_loss / max(n_batches, 1), global_step
    # ==========================================================


@torch.no_grad()
def evaluate(
    model: Transformer,
    loader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> float:
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

    # 返回验证集平均损失
    return total_loss / max(n_batches, 1)


def detect_mode(data_path: str | None) -> str:
    """根据文件扩展名自动检测数据模式：.jsonl/.tsv为‘qa’，否则为‘text’。"""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练decoder-only Transformer语言模型")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="数据路径。text模式：.txt 或目录；含 title/text 的维基 JSONL（.json/.jsonl）会自动流式读取；qa模式：.jsonl 或 .tsv。",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["text", "qa"],
        help="数据模式：'text'（连贯文本）或'qa'（问答对）。未填写时自动检测。",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default=None,
        choices=["transformer", "warmup_const", "const"],
        help="学习率调度策略。transformer 会随 step 衰减；长训练建议用 warmup_const 或设置 --min-lr。",
    )
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None, help="模型维度 d_model")
    parser.add_argument("--n-heads", type=int, default=None, help="多头注意力头数")
    parser.add_argument("--dec-layers", type=int, default=None, help="Decoder 层数")
    parser.add_argument("--d-ff", type=int, default=None, help="FFN 隐藏维度 d_ff")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout 比例")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="本次训练产物输出目录，包含 checkpoint 和 tokenizer。",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="从指定 checkpoint 继续训练；--num-epochs 表示在此基础上再训练多少轮。",
    )
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--tokenizer-vocab-size", type=int, default=None)
    parser.add_argument(
        "--tokenizer-model-type",
        type=str,
        default=None,
        choices=["unigram", "bpe", "char", "word"],
    )
    parser.add_argument("--retrain-tokenizer", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default=None,
        choices=["auto", "fp16", "bf16", "off"],
        help="CUDA 下的混合精度策略。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="训练设备。",
    )
    parser.add_argument(
        "--safe-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用更保守的训练参数与 CUDA 运行时设置。",
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="训练后生成样本时的起始prompt文本"
    )
    parser.add_argument(
        "--use-moe",
        action="store_true",
        help="启用稀疏 MoE FFN（每 token top-1 专家），需配合负载均衡项训练",
    )
    parser.add_argument(
        "--moe-experts",
        type=int,
        default=None,
        help="专家数量（默认读取 ModelConfig.moe_num_experts）",
    )
    parser.add_argument(
        "--moe-lb-coeff",
        type=float,
        default=None,
        help="MoE 负载均衡损失系数（默认读取 ModelConfig.moe_lb_coeff）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_cfg = ModelConfig()  # 模型配置参数
    train_cfg = TrainConfig()  # 训练配置参数

    # 命令行参数优先覆盖默认训练配置
    # 优先用命令行参数覆盖默认训练配置，下列每个参数都加有注释

    if args.data_path is not None:
        train_cfg.data_path = args.data_path  # 训练数据路径
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size  # 批大小
    if args.num_epochs is not None:
        train_cfg.num_epochs = args.num_epochs  # 训练轮数
    if args.learning_rate is not None:
        train_cfg.learning_rate = args.learning_rate  # 初始学习率
    if args.warmup_steps is not None:
        train_cfg.warmup_steps = args.warmup_steps  # 预热步数
    if args.min_lr is not None:
        train_cfg.min_lr = args.min_lr  # 最小学习率
    if args.lr_schedule is not None:
        train_cfg.lr_schedule = args.lr_schedule  # 学习率调度策略
    if args.seq_len is not None:
        train_cfg.seq_len = args.seq_len  # 训练序列长度
    if args.d_model is not None:
        model_cfg.d_model = args.d_model  # 模型隐层维度
    if args.n_heads is not None:
        model_cfg.n_heads = args.n_heads  # 注意力头数
    if args.dec_layers is not None:
        model_cfg.n_decoder_layers = args.dec_layers  # 解码器层数
    if args.d_ff is not None:
        model_cfg.d_ff = args.d_ff  # 前馈层维度
    if args.dropout is not None:
        model_cfg.dropout = args.dropout  # dropout 概率
    if args.output_dir is not None:
        train_cfg.checkpoint_dir = args.output_dir  # 模型保存目录
    if args.tokenizer_path is not None:
        train_cfg.tokenizer_path = args.tokenizer_path  # 分词器模型文件路径
    if args.tokenizer_vocab_size is not None:
        train_cfg.tokenizer_vocab_size = args.tokenizer_vocab_size  # 分词器词表大小
    if args.tokenizer_model_type is not None:
        train_cfg.tokenizer_model_type = args.tokenizer_model_type  # 分词器类型
    if args.grad_accum_steps is not None:
        train_cfg.grad_accum_steps = args.grad_accum_steps  # 梯度累积步数
    if args.mixed_precision is not None:
        train_cfg.mixed_precision = args.mixed_precision  # 混合精度策略
    if args.safe_mode is not None:
        train_cfg.safe_mode = args.safe_mode  # 是否启用安全模式
    if args.use_moe:
        model_cfg.use_moe = True  # 启用 MoE 稀疏专家结构
    if args.moe_experts is not None:
        model_cfg.moe_num_experts = args.moe_experts  # MoE 专家数量
    if args.moe_lb_coeff is not None:
        model_cfg.moe_lb_coeff = args.moe_lb_coeff  # MoE 负载均衡损失系数

    """
    =========================================
    1. 基础配置与初始化
    =========================================
    - 验证和修正部分关键配置（如梯度累积步数最小为1，d_model必须能被n_heads整除）
    - 设置随机种子，保证实验可复现
    - 获取训练设备（CPU/GPU/MPS等）
    - 配置运行时环境，包括混合精度等
    - 根据安全模式调整batch_size
    """
    # 1.1 梯度累积步数最小为1
    train_cfg.grad_accum_steps = max(train_cfg.grad_accum_steps, 1)
    # 1.2 d_model必须能被n_heads整除
    if model_cfg.d_model % model_cfg.n_heads != 0:
        raise ValueError(
            f"d_model ({model_cfg.d_model}) 必须能被 n_heads ({model_cfg.n_heads}) 整除"
        )
    # 1.3 设置随机种子
    torch.manual_seed(train_cfg.seed)
    random.seed(train_cfg.seed)
    # 1.4 获取设备和配置运行环境
    device = get_device(args.device)
    configure_runtime(device, safe_mode=train_cfg.safe_mode)
    amp_dtype = get_amp_dtype(device, train_cfg.mixed_precision)
    # 1.5 安全模式下根据设备情况自动调整batch_size
    if train_cfg.safe_mode:
        safe_batch_size = recommend_batch_size(device, train_cfg.batch_size)
        if safe_batch_size != train_cfg.batch_size:
            log.warning(
                f"安全模式已将 batch_size 从 {train_cfg.batch_size} 调整为 {safe_batch_size}，"
                "以降低显存与驱动压力。"
            )
            train_cfg.batch_size = safe_batch_size
    # 1.6 记录当前训练的基本设置，包括设备信息
    log.info(f"当前使用设备: {device}")
    log.info(
        f"训练设置: batch_size={train_cfg.batch_size}, grad_accum_steps={train_cfg.grad_accum_steps}, "
        f"mixed_precision={train_cfg.mixed_precision}, safe_mode={train_cfg.safe_mode}"
    )
    if device.type == "cuda":
        # props是cuda设备的属性，包括name，total_memory，major，minor，major_rev，minor_rev，multi_processor_count
        props = torch.cuda.get_device_properties(device)
        log.info(
            f"CUDA 设备: {props.name} | 显存 {props.total_memory / 1024**3:.1f} GB | "
            f"AMP={'off' if amp_dtype is None else str(amp_dtype).replace('torch.', '')}"
        )

    """
    =========================================
    2. 数据与分词器准备
    =========================================
    - 检测与设定数据模式（text/qa）
    - 创建输出目录
    - 确保分词器存在，不存在则训练或下载；如无数据自动下载demo语料
    - 构建数据集与dataloader（根据模式分别处理qa或文本）
    """
    # 2.1 检测数据模式（自动判断或根据用户指定）
    mode = args.mode or detect_mode(train_cfg.data_path)
    log.info(f"数据模式: {mode}")
    # 2.2 创建输出目录
    output_dir = Path(train_cfg.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 2.3 检查并处理分词器路径
    if args.tokenizer_path is None and not train_cfg.tokenizer_path:
        train_cfg.tokenizer_path = str(output_dir / "tokenizer.model")
    # 如果tokenizer_path为空，则使用output_dir/tokenizer.model作为分词器路径
    tokenizer_path = Path(train_cfg.tokenizer_path)
    tokenizer_source = train_cfg.data_path
    # 2.4 必须指定分词器路径且分词器文件必须存在，否则报错
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"未找到 tokenizer 文件: {tokenizer_path}。请通过 --tokenizer-path 指定已经训练好的分词器。"
        )
    log.info(
        f"加载 tokenizer: path={tokenizer_path}, vocab_size={train_cfg.tokenizer_vocab_size}, "
        f"model_type={train_cfg.tokenizer_model_type}"
    )
    # 2.5 确保分词器（不存在则训练或加载）
    tokenizer = ensure_tokenizer(
        data_path=tokenizer_source,
        tokenizer_path=tokenizer_path,
        vocab_size=train_cfg.tokenizer_vocab_size,
        model_type=train_cfg.tokenizer_model_type,
        retrain=False,
    )
    log.info(f"tokenizer 已就绪: {tokenizer_path}")

    # 2.6 构建数据集并获得DataLoader，自动支持text和qa两种模式
    log.info("准备数据...")
    if mode == "qa":
        if train_cfg.data_path is None:
            raise ValueError("--data-path 参数在QA模式下是必须的")
        train_loader, val_loader, tokenizer = create_qa_dataloaders(
            data_path=train_cfg.data_path,
            max_len=train_cfg.seq_len,
            batch_size=train_cfg.batch_size,
            val_split=train_cfg.val_split,
            tokenizer=tokenizer,
        )
    else:  # 文本模式text
        train_loader, val_loader, tokenizer = create_dataloaders(
            seq_len=train_cfg.seq_len,
            batch_size=train_cfg.batch_size,
            val_split=train_cfg.val_split,
            data_path=train_cfg.data_path,
            tokenizer=tokenizer,
        )

    # 2.7 记录词表大小等信息，保存分词器到输出目录
    model_cfg.vocab_size = tokenizer.vocab_size
    log.info(f"词表大小: {model_cfg.vocab_size}")
    if model_cfg.use_moe:
        log.info(
            f"MoE 已启用: 专家数={model_cfg.moe_num_experts}, "
            f"负载系数={model_cfg.moe_lb_coeff}"
        )
    tokenizer_path = Path(train_cfg.tokenizer_path)
    tokenizer.save(tokenizer_path)
    log.info(f"分词器已保存到: {tokenizer_path}")
    log.info(f"训练产物目录: {output_dir}")

    """
    =========================================
    3. 模型、优化器、损失函数、scaler准备
    =========================================
    - 构建模型，并统计参数量
    - 创建Adam优化器
    - 配置混合精度梯度缩放（如需）
    - 设置交叉熵损失函数
    - 训练断点恢复功能
    """
    # 3.1 构建模型并输出参数数量
    model = Transformer(model_cfg).to(device)
    n_params = count_parameters(model)
    log.info(f"模型参数总量: {n_params:,} ({n_params / 1e6:.1f}M)")
    # 3.2 构造Adam优化器
    # 参数是：
    # model.parameters()是模型中所有可训练参数的迭代器
    # lr是学习率
    # betas是Adam优化器的两个超参数，分别是动量和第二动量
    # eps是Adam优化器的平滑参数
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.learning_rate, betas=(0.9, 0.98), eps=1e-9
    )
    # 3.3 配置混合精度Scaler（仅在GPU+FP16情况下启用）
    scaler = (
        torch.amp.GradScaler(
            "cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16)
        )
        if device.type == "cuda"
        else None
    )
    # 3.4 设置损失函数为交叉熵，忽略-100标签
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    # 3.5 训练变量初始化与断点恢复
    global_step = 0
    best_val_loss = float("inf")
    start_epoch = 1
    end_epoch = train_cfg.num_epochs
    """ 继续训练 走这里"""
    if args.resume_from is not None:
        # args.resume_from是checkpoint的路径
        resume_path = Path(args.resume_from)
        # load_checkpoint是加载checkpoint的函数，
        # 参数是：
        # resume_path是checkpoint的路径
        # model是模型
        # optimizer是优化器
        # device是设备
        ckpt = load_checkpoint(resume_path, model, optimizer=optimizer, device=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        end_epoch = start_epoch + train_cfg.num_epochs - 1
        global_step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("loss", float("inf")))
        log.info(
            f"已从 checkpoint 恢复: {resume_path} | 上次 epoch={ckpt.get('epoch', 0)} | "
            f"step={global_step} | best_val_loss={best_val_loss:.4f}"
        )

    """
    =========================================
    4. 正式训练主循环
    =========================================
    - 循环每个epoch，分别执行训练和验证
    - 记录耗时与每步日志
    - 保存最佳模型和周期性checkpoint
    - 支持Ctrl+C安全中断
    =========================================
    """
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
            "  数据为 IterableDataset（如本地维基 JSONL），DataLoader 无固定 len；每 epoch 步数由语料切块数量决定。"
        )

    total_epochs_display = end_epoch
    current_epoch = start_epoch

    try:
        for epoch in range(start_epoch, end_epoch + 1):
            current_epoch = epoch
            t0 = time.time()

            # Step 4.1: 执行一个epoch的训练
            train_loss, global_step = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                epoch,
                global_step,
                train_cfg,
                model_cfg,
                scaler,
                amp_dtype,
            )
            # Step 4.2: 在验证集上评估
            val_loss = evaluate(model, val_loader, criterion, device, amp_dtype)
            elapsed = time.time() - t0
            if device.type == "cuda":
                torch.cuda.empty_cache()
            # Step 4.3: 记录日志
            log.info(
                f"第{epoch}/{total_epochs_display}轮 | "
                f"训练损失={train_loss:.4f} | 验证损失={val_loss:.4f} | "
                f"用时={elapsed:.1f}s"
            )
            # Step 4.4: 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    val_loss,
                    output_dir / "best_model.pt",
                )
                log.info(f"  新的最佳模型已保存 (val_loss={val_loss:.4f})")
            # Step 4.5: 定期保存checkpoint
            if epoch % train_cfg.save_interval == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    val_loss,
                    output_dir / f"checkpoint_epoch{epoch}.pt",
                )
    except KeyboardInterrupt:
        """
        Step 4.6: 支持Ctrl+C安全中断并保存中断状态
        """
        interrupted_path = output_dir / "interrupted.pt"
        log.warning("收到 Ctrl+C，正在保存中断检查点并退出...")
        save_checkpoint(
            model,
            optimizer,
            current_epoch,
            global_step,
            best_val_loss,
            interrupted_path,
        )
        log.warning(f"已保存中断检查点: {interrupted_path}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return

    log.info(f"训练结束，最佳验证损失: {best_val_loss:.4f}")

    """
    =========================================
    5. 训练后样本文本生成环节（可选）
    - 自动选取或由参数传入prompt，生成一段文本并打印
    =========================================
    """
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
        log.info("未指定prompt，跳过样本生成。可通过--prompt参数提供。")


if __name__ == "__main__":
    main()
