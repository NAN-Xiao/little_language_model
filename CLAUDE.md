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

## Text Generation & Sampling Parameters

The `Transformer.generate()` method in [model/transformer.py](model/transformer.py) implements the full decoding pipeline. Below is how each parameter transforms the raw model logits into the next token.

### Pipeline Overview

```
raw logits → temperature → repetition_penalty → no_repeat_ngram → top_k → top_p → softmax → multinomial sampling
```

Each step modifies the `logits` array; setting a position to `-inf` makes its probability zero after softmax.

### Parameter Details

**`temperature`**
Divides all logits. Scales the distribution sharpness without changing the ranking.
- `< 1` (e.g. 0.7): gaps widen → softmax is sharper → high-scoring tokens dominate → more deterministic
- `= 1`: no change
- `> 1` (e.g. 1.5): gaps shrink → softmax is flatter → more random/creative
- `→ 0` approaches argmax; `→ ∞` approaches uniform

**`repetition_penalty`**
Reduces probability of already-generated tokens. Applied to every token in `generated`.
- Positive logit: `logit / penalty` (moves toward zero from above)
- Negative logit: `logit * penalty` (moves toward zero from below)
- In both cases the absolute value shrinks, so after softmax the probability drops.

**`no_repeat_ngram_size`**
Prevents repeating n-grams. If `n=3` and `"天气真"` already appeared, the next token cannot be `"好"` if that would recreate the exact 3-gram. Works by scanning history and banning matching continuation tokens (set to `-inf`).

**`top_k`**
Fixed-count filtering. Keeps only the `k` highest-scoring tokens; everything else becomes `-inf`.
- Simple and fast, but rigid: always keeps exactly `k` tokens even if the distribution is very flat or very peaked.

**`top_p` (nucleus sampling)**
Dynamic-count filtering based on cumulative probability.
1. Sort tokens by score descending
2. Compute softmax probabilities
3. Accumulate probabilities from the top down
4. Keep the smallest set whose cumulative probability ≥ `p`
5. Discard the rest (`-inf`)
- More adaptive than `top_k`: if the top few tokens already account for 90% of probability, only those are kept; if probability is spread out, many more are retained.

**`min_new_tokens`**
Forces the model to generate at least this many tokens before it is allowed to emit `eos`. Implemented by setting `logits[:, eos_token_id] = -inf` for the first `min_new_tokens` steps.

### Softmax & Sampling

After all filters, `F.softmax(logits)` converts scores to probabilities. Tokens previously set to `-inf` get probability 0 and can never be selected.

`torch.multinomial(probs, 1)` draws one token according to the probability distribution. Unlike `argmax`, this is stochastic: high-probability tokens are likely to win, but low-probability tokens still have a chance.

### Common Combinations

| Scenario | Suggested Settings |
|----------|-------------------|
| Creative writing | `temperature=1.2`, `top_p=0.9` |
| Factual Q&A | `temperature=0.7`, `top_p=0.95` |
| Code generation | `temperature=0.2`, `top_p=0.95`, `repetition_penalty=1.1` |
| Prevent repetition | `repetition_penalty=1.2`, `no_repeat_ngram_size=3` |
