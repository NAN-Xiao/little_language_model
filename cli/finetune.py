"""LoRA / 全参数微调脚本。

用法示例:
    # 使用 LoRA 对预训练模型在新数据上进行微调
    python finetune.py --data-path my_data.txt --base-checkpoint checkpoints/pretrain/base-decoder-only/best_model.pt

    # 合并 LoRA 权重到基础模型，便于部署
    python finetune.py merge --lora-path checkpoints/sft/lora/qa-decoder-only/best_lora.pt
"""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import torch
import torch.nn as nn

from config import ModelConfig, FinetuneConfig, LoRAConfig
from data.dataset import create_dataloaders, create_qa_dataloaders
from data.tokenizer import SentencePieceTokenizer
from model import Transformer
from model.lora import (
    apply_lora, get_lora_parameters, save_lora, load_lora,
    merge_lora, count_lora_parameters,
)
from model.moe_feedforward import collect_moe_load_balance_loss
from utils import (
    configure_runtime,
    get_amp_dtype,
    get_device,
    get_logger,
    recommend_batch_size,
    save_checkpoint,
)

log = get_logger()


def get_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    """线性预热然后转为固定学习率。"""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    return base_lr


def load_base_model(
    checkpoint_path: str, tokenizer_path: str, device: torch.device
) -> tuple[Transformer, SentencePieceTokenizer]:
    # 加载分词器
    tokenizer = SentencePieceTokenizer.load(tokenizer_path)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = Transformer(cfg)

    # 加载基础模型权重
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    log.info(f"已从 {checkpoint_path} 加载基础模型 "
             f"(epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})")
    return model, tokenizer


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    global_step: int,
    ft_cfg: FinetuneConfig,
    scaler: torch.amp.GradScaler | None,
    amp_dtype: torch.dtype | None,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)
    use_amp = device.type == "cuda" and amp_dtype is not None
    transfer_kwargs = {"non_blocking": True} if device.type == "cuda" else {}

    for batch_idx, (input_ids, labels) in enumerate(loader):
        input_ids = input_ids.to(device, **transfer_kwargs)
        labels = labels.to(device, **transfer_kwargs)

        global_step += 1

        # 更新学习率（带预热）
        lr = get_lr(global_step, ft_cfg.warmup_steps, ft_cfg.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ==================== LoRA权重在此处参与前向与反向传播计算 ==========================
        # LoRA实际的参数在 model.forward 调用时生效。LoRA已在apply_lora处集成进模型相应模块，
        # 所以此处 model(input_ids) 的前向传播时会自动带上LoRA的增量权重去影响原有全连接层等权重输出。
        # 具体融合细节可以查看 model/lora.py 中的 apply_lora 实现。
        # 下面的loss.backward()时只会累积LoRA权重的梯度，因为优化器只传入了LoRA参数。
        try:
            with (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp
                else contextlib.nullcontext()
            ):
                logits = model(input_ids)
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                if model.cfg.use_moe:
                    loss = loss + model.cfg.moe_lb_coeff * collect_moe_load_balance_loss(model)
        except RuntimeError as exc:
            if device.type == "cuda" and "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "LoRA 微调阶段发生 CUDA OOM。请减小 --batch-size，或增大 --grad-accum-steps。"
                ) from exc
            raise

        scaled_loss = loss / ft_cfg.grad_accum_steps
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        should_step = (batch_idx + 1) % ft_cfg.grad_accum_steps == 0
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), ft_cfg.max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        n_batches += 1

        # 日志输出
        if global_step % ft_cfg.log_interval == 0:
            avg = total_loss / n_batches
            log.info(
                f"第{epoch}轮 | 步骤 {global_step} | 批次 {batch_idx+1}/{len(loader)} | "
                f"损失 {loss.item():.4f} | 均值 {avg:.4f} | 学习率 {lr:.2e}"
            )

    if n_batches > 0 and n_batches % ft_cfg.grad_accum_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), ft_cfg.max_grad_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(n_batches, 1), global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> float:
    # 验证集评估
    model.eval()
    total_loss = 0.0
    n_batches = 0
    use_amp = device.type == "cuda" and amp_dtype is not None
    transfer_kwargs = {"non_blocking": True} if device.type == "cuda" else {}
    for input_ids, labels in loader:
        input_ids = input_ids.to(device, **transfer_kwargs)
        labels = labels.to(device, **transfer_kwargs)
        # LoRA推理阶段同样在前向计算作用于模型
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

"""
这个函数的作用是：
1. 加载基础模型
2. 处理微调数据
3. 应用 LoRA
4. 优化器 - 只优化 LoRA 参数
5. 开始 LoRA 微调
6. 保存最佳模型
7. 按间隔保存
8. 微调完成，最佳验证损失: {best_val_loss:.4f}
"""
def do_finetune(args: argparse.Namespace):
    """""
    1. 解析和处理命令行参数，将参数映射到配置对象中
    """""
    ft_cfg = FinetuneConfig()
    lora_cfg = LoRAConfig()
    tuning_mode = getattr(args, "tuning_mode", "lora")

    if args.data_path is not None:
        ft_cfg.data_path = args.data_path
    if args.base_checkpoint is not None:
        ft_cfg.base_checkpoint = args.base_checkpoint
    if args.base_tokenizer is not None:
        ft_cfg.base_tokenizer = args.base_tokenizer
    if args.output_dir is not None:
        ft_cfg.output_dir = args.output_dir
    if args.batch_size is not None:
        ft_cfg.batch_size = args.batch_size
    if args.num_epochs is not None:
        ft_cfg.num_epochs = args.num_epochs
    if args.learning_rate is not None:
        ft_cfg.learning_rate = args.learning_rate
    if args.seq_len is not None:
        ft_cfg.seq_len = args.seq_len
    if args.grad_accum_steps is not None:
        ft_cfg.grad_accum_steps = args.grad_accum_steps
    if args.mixed_precision is not None:
        ft_cfg.mixed_precision = args.mixed_precision
    if args.safe_mode is not None:
        ft_cfg.safe_mode = args.safe_mode
    if args.rank is not None:
        lora_cfg.rank = args.rank
    if args.alpha is not None:
        lora_cfg.alpha = args.alpha
    ft_cfg.grad_accum_steps = max(ft_cfg.grad_accum_steps, 1)

    """""
    2. 配置运行环境，包括种子、设备、混合精度和Batch调整
    """""
    torch.manual_seed(ft_cfg.seed)
    device = get_device(args.device)
    configure_runtime(device, safe_mode=ft_cfg.safe_mode)
    amp_dtype = get_amp_dtype(device, ft_cfg.mixed_precision)
    if ft_cfg.safe_mode:
        safe_batch_size = recommend_batch_size(device, ft_cfg.batch_size)
        if safe_batch_size != ft_cfg.batch_size:
            log.warning(
                f"安全模式已将 batch_size 从 {ft_cfg.batch_size} 调整为 {safe_batch_size}，"
                "以降低显存与驱动压力。"
            )
            ft_cfg.batch_size = safe_batch_size
    log.info(f"使用设备: {device}")
    log.info(
        f"微调设置: batch_size={ft_cfg.batch_size}, grad_accum_steps={ft_cfg.grad_accum_steps}, "
        f"mixed_precision={ft_cfg.mixed_precision}, safe_mode={ft_cfg.safe_mode}"
    )

    """""
    3. 加载基础模型及其分词器
    """""
    model, tokenizer = load_base_model(ft_cfg.base_checkpoint, ft_cfg.base_tokenizer, device)

    """""
    4. 检测数据模式并准备微调数据，生成DataLoader和数据分词器
    """""
    mode = getattr(args, "mode", None)
    if mode is None and ft_cfg.data_path:
        from .train import detect_mode
        mode = detect_mode(ft_cfg.data_path)
    mode = mode or "text"
    log.info(f"数据模式: {mode}")

    log.info("准备微调数据...")
    if mode == "qa":
        if ft_cfg.data_path is None:
            raise ValueError("--data-path 参数在 QA 模式下是必须的")
        train_loader, val_loader, ft_tokenizer = create_qa_dataloaders(
            data_path=ft_cfg.data_path,
            max_len=ft_cfg.seq_len,
            batch_size=ft_cfg.batch_size,
            val_split=ft_cfg.val_split,
            tokenizer=tokenizer,
        )
    else:
        train_loader, val_loader, ft_tokenizer = create_dataloaders(
            seq_len=ft_cfg.seq_len,
            batch_size=ft_cfg.batch_size,
            val_split=ft_cfg.val_split,
            data_path=ft_cfg.data_path,
            tokenizer=tokenizer,
        )

    """""
    5. 校验微调数据分词器词表大小是否与基础分词器一致，并报警告
    """""
    if ft_tokenizer.vocab_size != tokenizer.vocab_size:
        log.warning(
            f"微调数据词表 ({ft_tokenizer.vocab_size}) 与基础模型词表 ({tokenizer.vocab_size}) 不同。"
            "将使用基础分词器，未知字符会被映射为 <pad>。"
        )

    """""
    6. 构建微调参数，选择LoRA或全参数，统计可训练参数量并制定保存策略
    """""
    output_dir = Path(ft_cfg.output_dir)
    if tuning_mode == "lora":
        """""
        6.1 LoRA方式插入LoRA适配器，仅LoRA参数可训练与保存
        """""
        target_modules = set(lora_cfg.target_modules)
        replaced = apply_lora(
            model, rank=lora_cfg.rank, alpha=lora_cfg.alpha,
            dropout=lora_cfg.dropout, target_modules=target_modules,
        )
        trainable, total = count_lora_parameters(model)
        log.info(f"LoRA 已应用于 {len(replaced)} 层: {replaced[:6]}{'...' if len(replaced) > 6 else ''}")
        log.info(f"可训练参数: {trainable:,} / 总参数: {total:,} ({100*trainable/total:.2f}%)")
        # 获取可训练的LoRA参数
        trainable_params = get_lora_parameters(model)
        best_artifact_path = output_dir / "best_lora.pt"
        periodic_artifact = lambda epoch: output_dir / f"lora_epoch{epoch}.pt"
    else:
        """""
        6.2 全参数微调，全部模型参数可更新且保存
        """""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"全参数微调已启用: 可训练参数 {trainable:,} / 总参数 {total:,} ({100*trainable/total:.2f}%)")
        trainable_params = model.parameters()
        best_artifact_path = output_dir / "best_model.pt"
        periodic_artifact = lambda epoch: output_dir / f"checkpoint_epoch{epoch}.pt"

    """""
    7. 创建优化器、损失函数以及AMP混合精度Scaler
    """""
    optimizer = torch.optim.AdamW(trainable_params, lr=ft_cfg.learning_rate, weight_decay=0.01)
    scaler = (
        torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))
        if device.type == "cuda"
        else None
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    """""
    8. 主训练循环：逐轮训练，每轮完成训练、验证和模型保存
    """""
    global_step = 0
    best_val_loss = float("inf")
    if tuning_mode == "lora":
        log.info(
            f"开始 LoRA 微调，共 {ft_cfg.num_epochs} 轮 "
            f"(rank={lora_cfg.rank}, alpha={lora_cfg.alpha})..."
        )
    else:
        log.info(f"开始全参数微调，共 {ft_cfg.num_epochs} 轮...")

    current_epoch = 1
    try:
        for epoch in range(1, ft_cfg.num_epochs + 1):
            current_epoch = epoch
            t0 = time.time()

            """""
            8.1 训练一个epoch并更新global_step，验证集计算损失
            """""
            train_loss, global_step = train_one_epoch(
                model, train_loader, optimizer, criterion,
                device, epoch, global_step, ft_cfg, scaler, amp_dtype,
            )
            val_loss = evaluate(model, val_loader, criterion, device, amp_dtype)
            elapsed = time.time() - t0
            if device.type == "cuda":
                torch.cuda.empty_cache()

            log.info(
                f"第 {epoch}/{ft_cfg.num_epochs} 轮 | "
                f"训练损失 {train_loss:.4f} | 验证损失 {val_loss:.4f} | 用时 {elapsed:.1f}s"
            )

            """""
            8.2 若验证损失更优，则保存当前最佳模型或LoRA参数
            """""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if tuning_mode == "lora":
                    save_lora(model, best_artifact_path)
                    log.info(f"  新的最佳 LoRA 适配器已保存 (val_loss={val_loss:.4f})")
                else:
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        global_step,
                        val_loss,
                        best_artifact_path,
                    )
                    log.info(f"  新的最佳全量模型已保存 (val_loss={val_loss:.4f})")

            """""
            8.3 按间隔(epoch)保存中间检查点，方便断点恢复或备份
            """""
            if epoch % ft_cfg.save_interval == 0:
                if tuning_mode == "lora":
                    save_lora(model, periodic_artifact(epoch))
                else:
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        global_step,
                        val_loss,
                        periodic_artifact(epoch),
                    )
    except KeyboardInterrupt:
        """""
        8.4 捕捉中断信号，保存中断检查点以支持训练恢复
        """""
        log.warning("收到 Ctrl+C，正在保存中断检查点并退出...")
        if tuning_mode == "lora":
            interrupted_path = output_dir / "interrupted_lora.pt"
            save_lora(model, interrupted_path)
        else:
            interrupted_path = output_dir / "interrupted.pt"
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

    """""
    9. 训练流程结束后，打印最佳损失和最终权重路径
    """""
    log.info(f"微调完成，最佳验证损失: {best_val_loss:.4f}")
    if tuning_mode == "lora":
        log.info(f"LoRA 适配器保存于 {best_artifact_path}")
    else:
        log.info(f"全量微调模型保存于 {best_artifact_path}")

    """""
    10. 支持按需对用户自定义prompt生成样本文本，体验当前模型效果
    """""
    prompt = args.prompt
    if prompt:
        log.info("生成样本文本...")
        prompt_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special=False)
        src_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        # ================ LoRA生成阶段也是在模型forward参与 ==================
        generated = model.generate(src_tensor, max_len=200, temperature=0.8, top_k=40)
        log.info(f"输入:  {prompt}")
        log.info(f"输出: {tokenizer.decode(generated[0].tolist())}")


def do_merge(args: argparse.Namespace):
    """将 LoRA 适配器合并进基础模型，并保存为独立权重文件。"""
    device = get_device()

    tokenizer_path = args.base_tokenizer or "checkpoints/tokenizers/wiki-pretrain-v1/tokenizer.model"
    base_ckpt_path = args.base_checkpoint or "checkpoints/pretrain/base-decoder-only/best_model.pt"
    lora_path = args.lora_path
    lora_rank = args.rank or 8
    lora_alpha = args.alpha or 16.0

    log.info("加载基础模型...")
    model, tokenizer = load_base_model(base_ckpt_path, tokenizer_path, device)

    # 先插入LoRA结构，再加载LoRA参数
    target_modules = set(LoRAConfig().target_modules)
    apply_lora(model, rank=lora_rank, alpha=lora_alpha, target_modules=target_modules)
    # ==================== LoRA结构插入相关，后续调用 merge_lora 进行权重融合 ====================

    log.info(f"加载 LoRA 适配器: {lora_path} ...")
    load_lora(model, lora_path, device)

    log.info("将 LoRA 权重合并进基础模型...")
    merge_lora(model)
    # ==================== merge_lora 在此将 LoRA 增量权重合成到原始权重，之后即可仅保存合并后的基础模型权重 ====================

    out_path = Path(args.merge_output or "checkpoints/sft/full/merged-model.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": 0, "step": 0, "model_state_dict": model.state_dict(), "loss": 0.0},
        out_path,
    )
    log.info(f"合并后的模型已保存到 {out_path}")


def main():
    
    """
    LoRA 微调和权重合并主入口
    1. 解析命令行参数
    2. 判断要做 LoRA/全量微调还是 LoRA 合并
    3. 分别调用对应流程
    """
    
    # ========== 1. 创建 ArgumentParser 并添加子命令 ==========
    """
    1. 创建命令行参数解析器。支持两种模式:
        (1) train: 进行 LoRA 或全参数微调（Recommended LoRA 微调方式）
        (2) merge: 将 LoRA 适配器合并到基础模型，得到单独权重（部署推荐）
    """
    parser = argparse.ArgumentParser(description="Transformer LM 的 LoRA / 全参数微调")
    # 子命令集合
    sub = parser.add_subparsers(dest="command")

    # --- (1) LoRA/全参数微调
    """
    2. 设置 train 子命令相关参数, 支持 LoRA、全量微调流程自定义。
       包括微调数据、基础模型、batch、lr、LoRA 超参、推理采样等。
    """
    ft = sub.add_parser("train", help="进行 LoRA 或全参数微调")
    ft.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="微调数据 (.txt/.jsonl/.tsv 文件或目录)"
    )  # 微调语料数据路径（文件或目录）
    ft.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["text", "qa"],
        help="数据模式（如不填写自动检测）"
    )  # 输入数据模式，支持"text"（纯文本）和"qa"（问答），默认自动检测
    ft.add_argument(
        "--base-checkpoint",
        type=str,
        default=None
    )  # 基础模型 checkpoint 路径（预训练模型权重）
    ft.add_argument(
        "--base-tokenizer",
        type=str,
        default=None
    )  # 分词器路径
    ft.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="LoRA 产物输出目录"
    )  # LoRA (或全参) 微调后产物的保存目录
    ft.add_argument(
        "--batch-size",
        type=int,
        default=None
    )  # 微调 batch size（每步多少样本）
    ft.add_argument(
        "--num-epochs",
        type=int,
        default=None
    )  # 微调 epoch 数（训练多少轮全量数据集）
    ft.add_argument(
        "--learning-rate",
        type=float,
        default=None
    )  # 学习率（optimizer 的 lr）
    ft.add_argument(
        "--seq-len",
        type=int,
        default=None
    )  # 最大序列长度（最大支持的 token 数）
    ft.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None
    )  # 梯度累积步数（几步累加再反向传播）
    ft.add_argument(
        "--mixed-precision",
        type=str,
        default=None,
        choices=["auto", "fp16", "bf16", "off"],
        help="CUDA 下的混合精度策略。",
    )  # 是否开启混合精度训练（"fp16"/"bf16"/"auto"/"off"）
    ft.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="微调设备。",
    )  # 设备类型，可选 "auto"、"cpu"、"cuda"
    ft.add_argument(
        "--safe-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用更保守的微调参数与 CUDA 运行时设置。",
    )  # 启用安全模式（更保守的训练参数与 CUDA 环境设置）
    ft.add_argument(
        "--rank",
        type=int,
        default=None,
        help="LoRA rank（默认: 8）"
    )  # LoRA 的秩（分解矩阵的维度，默认 8）
    ft.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="LoRA alpha（默认: 16）"
    )  # LoRA 缩放参数 alpha，默认 16
    ft.add_argument(
        "--tuning-mode",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="微调模式：lora 或 full（全参数微调）。",
    )  # 微调模式，"lora"为增量微调，"full"为全参数微调
    ft.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="微调后用于采样生成的提示语"
    )  # 微调完成后采样生成的 prompt 提示文本

    # --- (2) LoRA 权重合并
    """""
    3. 设置 merge 子命令, 用于将 LoRA 权重融合进基础模型, 适合部署/推理。
    """""
    mg = sub.add_parser("merge", help="将 LoRA 适配器合并进基础模型")
    mg.add_argument("--lora-path", type=str, required=True, help="LoRA 适配器权重 .pt 文件路径")
    mg.add_argument("--base-checkpoint", type=str, default=None)
    mg.add_argument("--base-tokenizer", type=str, default=None)
    mg.add_argument("--rank", type=int, default=None)
    mg.add_argument("--alpha", type=float, default=None)
    mg.add_argument("--merge-output", type=str, default=None, help="合并后模型的输出路径")

    # ========== 2. 解析参数并根据命令模式分派 ==========
    args = parser.parse_args()

    if args.command == "train":
        """
        4. 当命令为 train, 进行 LoRA 或全参数微调流程。
            包含:
            - 数据加载
            - 模型加载与 LoRA 模块插入
            - 训练主循环
            - 模型权重保存
            - 可选推理采样
        """""
        do_finetune(args)
    elif args.command == "merge":
        """
        5. 当命令为 merge, 进行 LoRA 参数合并进原始模型。适合部署推理。
            包含:
            - 基础模型加载
            - 插入/加载 LoRA 结构
            - 合并权重到原始模型
            - 保存新权重
        """""
        do_merge(args)
    else:
        """
        6. 未输入命令时，打印帮助信息
        """""
        parser.print_help()


if __name__ == "__main__":
    main()
