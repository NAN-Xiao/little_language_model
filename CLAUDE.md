# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A from-scratch implementation of a decoder-only Transformer language model with training, LoRA fine-tuning, text generation, and an HTTP inference server. The project also includes experimental diffusion (DiT, UNet) and vision (ViT, VAE) modules. Code comments and logs are primarily in Chinese.

## Commands

```bash
# Unified entry point
python __main__.py train                   # Pretrain from scratch
python __main__.py finetune                # LoRA/full fine-tuning (has subcommands)
python __main__.py generate                # Text generation
python __main__.py server                  # FastAPI inference server
python __main__.py train-tokenizer         # Train SentencePiece tokenizer
python __main__.py sample-corpus           # Sample corpus for tokenizer training
python __main__.py eval-geo-prompts        # Evaluate on geo prompts

# Training with resume
python __main__.py train --resume-from checkpoints/.../best_model.pt  # Continue training

# Fine-tuning subcommands
python __main__.py finetune train --data-path ... --base-checkpoint ...
python __main__.py finetune train --tuning-mode full ...              # Full parameter fine-tuning (default: lora)
python __main__.py finetune merge --lora-path ...                     # Merge LoRA weights into base model

# Generation (supports LoRA and interactive mode)
python __main__.py generate --checkpoint ... --tokenizer ... --prompt "text"
python __main__.py generate --lora lora.pt --interactive              # Interactive Q&A with LoRA
python __main__.py generate --min-new-tokens 50 --repetition-penalty 1.2 --no-repeat-ngram-size 3

# HTTP server (supports LoRA)
python __main__.py server --checkpoint ... --tokenizer ... --lora lora.pt --host 0.0.0.0 --port 8000

# Direct module execution
python -m cli.train --help
python -m scripts.data_prep.train_tokenizer --help
```

## Dependencies

`torch>=2.0`, `numpy>=1.24`, `fastapi>=0.116`, `uvicorn>=0.35`, `sentencepiece>=0.2`. No test framework configured — no tests exist yet.

## Architecture

### Core Model (`model/`)

- **`Transformer`** (`transformer.py`): Top-level decoder-only model. Composes `Decoder` + output projection. Handles causal mask construction and text generation with multiple sampling strategies (top-k, top-p, temperature, repetition penalty, n-gram blocking, min-new-tokens).
- **`Decoder`** (`decoder.py`): Token embedding + optional positional encoding + stack of `DecoderBlock` layers + final LayerNorm. Two positional encoding modes controlled by `ModelConfig.use_rope`.
- **`DecoderBlock`** (`decoder.py`): Pre-norm residual block: LayerNorm → masked self-attention → residual → LayerNorm → FFN/MoE → residual.
- **`MultiHeadAttention`** (`attention.py`): Multi-head scaled dot-product attention. Supports both self-attention and cross-attention. Optional RoPE rotation on Q/K.
- **`PositionwiseFeedForward`** (`feedforward.py`): 2-layer FFN: Linear → ReLU → Dropout → Linear → Dropout.
- **`MoEFeedForward`** (`moe_feedforward.py`): Switch-style top-1 sparse MoE replacing dense FFN. Includes load-balance auxiliary loss. Collected via `collect_moe_load_balance_loss()`.
- **`LoRALinear`** (`lora.py`): LoRA wrapper for nn.Linear. `apply_lora()` replaces target modules (default: `w_q`, `w_v`) with LoRALinear. Supports `merge_and_unload()` for deployment. Module also exports `save_lora`, `load_lora`, `merge_lora`, `count_lora_parameters`.
- **Positional encodings** (`positional.py`): `SinusoidalPositionalEncoding` (additive, GPT-2 style) and `RotaryPositionEmbedding` (RoPE, LLaMA style). Selected via `ModelConfig.use_rope`.

### Configuration (`config.py`)

- **`ModelConfig`**: Model hyperparameters (d_model=768, n_heads=12, n_decoder_layers=10, use_rope, use_moe, moe_num_experts=4, etc.)
- **`TrainConfig`**: Training hyperparameters (batch_size=4, lr=3e-4, warmup, grad accumulation, mixed precision, safe mode, tokenizer settings)
- **`FinetuneConfig`**: Fine-tuning specific settings
- **`LoRAConfig`**: LoRA rank=8, alpha=16, target_modules=["w_q", "w_v"]

### Data Pipeline (`data/`)

- **`SentencePieceTokenizer`** (`tokenizer.py`): Wraps sentencepiece. Supports train/load/save/encode/decode. `ensure_tokenizer()` loads existing or trains new. Alias `CharTokenizer = SentencePieceTokenizer` for backward compat.
- **`CausalLMDataset`** (`dataset.py`): Chunks continuous text into next-token prediction samples.
- **`QADataset`** (`dataset.py`): QA pairs with prompt masking (labels set to -100 for question portion).
- **`WikipediaJsonlBlockIterable`** (`dataset.py`): Streaming IterableDataset for large Wikipedia JSONL files using byte-offset indexing.
- **`create_dataloaders()`**: Auto-detects Wikipedia JSONL vs plain text; returns (train_loader, val_loader, tokenizer).
- **`create_qa_dataloaders()`**: For .jsonl/.tsv QA data.
- **Data auto-detection**: `detect_mode()` in `cli/train.py` infers "qa" vs "text" from file extension (.jsonl/.tsv → qa, else text). When no data path given, auto-downloads Tiny Shakespeare.

### Utilities (`utils.py`)

Shared helpers: `save_checkpoint`, `load_checkpoint`, `count_parameters`, `get_device`, `configure_runtime`, `get_amp_dtype`, `recommend_batch_size`, `get_logger`.

### CLI (`cli/`)

Each subcommand is a standalone module with `main()` entry point. CLI args override config defaults. Training loop follows: forward → loss (with optional MoE aux loss) → gradient accumulation → clip → step. Supports Ctrl+C graceful checkpoint save.

### Server (`cli/server.py`)

FastAPI app with `/health` and `/generate` endpoints. Loads model at startup via lifespan handler. Supports LoRA adapter loading.

### Diffusion & Vision (`diffusion/`, `vision/`)

Experimental modules not yet integrated into the main training pipeline:
- **Diffusion**: DiT (MMDiT, 3D VAE, video flow matching), UNet (DDPM, flow matching)
- **Vision**: ViT blocks, VAE

## Key Design Decisions

- Position encoding is a **runtime switch** (`use_rope`): False = additive sinusoidal at Decoder input; True = RoPE rotation per-layer in attention. Both codepaths coexist in `Decoder.forward()` and `MultiHeadAttention.forward()`.
- MoE is controlled by `ModelConfig.use_moe`. When enabled, each DecoderBlock's FFN is replaced by `MoEFeedForward`. Training loss includes `moe_lb_coeff * collect_moe_load_balance_loss(model)`.
- LoRA targets `w_q` and `w_v` by default (configurable via `LoRAConfig.target_modules`). Base weights are frozen; only A/B matrices are trained. Fine-tune also supports `--tuning-mode full` for full-parameter training.
- Safe mode (`TrainConfig.safe_mode`) adjusts batch size based on GPU VRAM and disables cuDNN benchmark for stability.
- Checkpoint format: dict with keys `epoch`, `step`, `model_state_dict`, `optimizer_state_dict`, `loss`. Naming convention: `best_model.pt` (best val loss), `checkpoint_epoch{N}.pt` (periodic), `interrupted.pt` (Ctrl+C). LoRA saves as `best_lora.pt` / `lora_epoch{N}.pt` / `interrupted_lora.pt`.
- Learning rate schedules: `transformer` (Vaswani decay), `warmup_const` (linear warmup then constant, default), `const` (fixed). All respect `min_lr` floor.
- Mixed precision: `auto` prefers bf16 if supported, falls back to fp16. GradScaler only enabled for fp16 on CUDA.
