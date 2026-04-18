from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch


def get_logger(name: str = "lit-lm", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    path: str | Path,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict:

    # 加载模型断点（checkpoint），包含权重、优化器状态等
    # 1. 用torch.load加载checkpoint文件到指定设备
    #torch.load返回一个字典
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # 2. 恢复模型权重
    #ckpt["model_state_dict"]是checkpoint的字典中的model_state_dict键的值
    #ckpt还有epoch，step，loss等键
    model.load_state_dict(ckpt["model_state_dict"])
    # 3. 如提供优化器，恢复优化器状态（如动量等参数）
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    # 4. 返回整个checkpoint的dict（包含自定义字段如epoch、loss等）
    return ckpt


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(preferred: str | None = None) -> torch.device:
    if preferred in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preferred == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前 PyTorch 环境未检测到可用的 CUDA 设备。")
    return torch.device(preferred)


def configure_runtime(device: torch.device, safe_mode: bool = True) -> None:
    if device.type != "cuda":
        return
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if safe_mode and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def get_amp_dtype(device: torch.device, precision: str) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    precision = precision.lower()
    if precision == "off":
        return None
    if precision == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if precision == "fp16":
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def recommend_batch_size(device: torch.device, requested_batch_size: int) -> int:
    if device.type != "cuda":
        return requested_batch_size
    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
    if total_vram_gb <= 8:
        return min(requested_batch_size, 8)
    if total_vram_gb <= 12:
        return min(requested_batch_size, 12)
    if total_vram_gb <= 16:
        return min(requested_batch_size, 16)
    return min(requested_batch_size, 24)
