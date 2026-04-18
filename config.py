from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 68
    d_model: int = 768
    n_heads: int = 12
    n_decoder_layers: int = 10
    d_ff: int = 3072
    dropout: float = 0.1
    max_seq_len: int = 256
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    # 位置编码：为 True 时使用旋转位置编码 RoPE (Llama/Qwen 风格)，
    # 为 False 时使用加法正弦位置编码 (原始 Transformer 风格)
    use_rope: bool = False
    # MoE：为 True 时每个 Decoder Block 的 FFN 换为稀疏专家（见 model/moe_feedforward.py）
    use_moe: bool = False
    moe_num_experts: int = 4
    # 训练时总损失 += moe_lb_coeff * collect_moe_load_balance_loss(model)
    moe_lb_coeff: float = 0.01


@dataclass
class TrainConfig:
    data_path: str | None = "data/pretrain/wikipedia-zh-cn-20240820.json"  # 数据文件路径，str 或 None
    batch_size: int = 4                   # 每个批次的样本数量
    num_epochs: int = 3                   # 训练总轮数
    learning_rate: float = 3e-4           # 初始学习率
    warmup_steps: int = 800               # 学习率预热步数
    min_lr: float = 0.0                  # 学习率下限（防止长训练后 lr 衰减到几乎为 0）
    lr_schedule: str = "warmup_const"    # transformer / warmup_const / const
    max_grad_norm: float = 1.0            # 梯度裁剪的最大范数
    val_split: float = 0.1                # 验证集划分比例
    checkpoint_dir: str = "checkpoints/pretrain/base-decoder-only"   # 检查点保存目录
    log_interval: int = 50                # 日志记录的间隔步数
    save_interval: int = 1                # 检查点保存的间隔（以 epoch 为单位）
    seed: int = 42                        # 随机种子
    seq_len: int = 256                    # 输入序列最大长度 (128k，一般大模型设置) deepseek-r1 设置为 128k
    grad_accum_steps: int = 8             # 梯度累积步数，用于在较小显存上模拟更大 batch
    mixed_precision: str = "auto"         # auto / fp16 / bf16 / off
    safe_mode: bool = True                # 启用更稳的 CUDA 训练策略
    tokenizer_path: str = "checkpoints/tokenizers/wiki-pretrain-v1/tokenizer.model"
    tokenizer_vocab_size: int = 16000
    tokenizer_model_type: str = "unigram"


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["w_q", "w_v"])


@dataclass
class FinetuneConfig:
    base_checkpoint: str = "checkpoints/pretrain/base-decoder-only/best_model.pt"
    data_path: str | None = None
    output_dir: str = "checkpoints/sft/full/qa-decoder-only"
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    val_split: float = 0.1
    log_interval: int = 20
    save_interval: int = 2
    seed: int = 42
    seq_len: int = 128
    grad_accum_steps: int = 1
    mixed_precision: str = "auto"
    safe_mode: bool = True
    base_tokenizer: str = "checkpoints/tokenizers/wiki-pretrain-v1/tokenizer.model"
