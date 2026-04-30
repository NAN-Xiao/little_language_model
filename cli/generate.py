"""Text generation script for the trained Transformer model.

Supports plain checkpoints, LoRA-adapted models, and interactive Q&A mode.
"""

from __future__ import annotations

import argparse

import torch

from config import ModelConfig, LoRAConfig
from tokenizer.tokenizer import SentencePieceTokenizer
from model import Transformer
from model.lora import apply_lora, load_lora
from utils import get_device, get_logger

log = get_logger()


def load_model(
    checkpoint_path: str,
    tokenizer_path: str,
    device: torch.device,
    lora_path: str | None = None,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
) -> tuple[Transformer, SentencePieceTokenizer]:
    tokenizer = SentencePieceTokenizer.load(tokenizer_path)

    cfg = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = Transformer(cfg).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info(f"Loaded model from {checkpoint_path} (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})")

    if lora_path is not None:
        target_modules = set(LoRAConfig().target_modules)
        apply_lora(model, rank=lora_rank, alpha=lora_alpha, target_modules=target_modules)
        load_lora(model, lora_path, device)
        log.info(f"Loaded LoRA adapter from {lora_path}")

    model.eval()
    return model, tokenizer


def generate_text(
    model: Transformer,
    tokenizer: SentencePieceTokenizer,
    prompt: str,
    device: torch.device,
    max_len: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    min_new_tokens: int = 0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> str:
    prompt_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special=False)
    src_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated = model.generate(
        src_tensor,
        max_len=max_len,
        min_new_tokens=min_new_tokens,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    
    return tokenizer.decode(generated[0].tolist())


def interactive_qa(
    model: Transformer,
    tokenizer: SentencePieceTokenizer,
    device: torch.device,
    max_len: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    min_new_tokens: int = 0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
):
    """Interactive Q&A loop: type a question, get an answer. Type 'quit' to exit."""
    print("\n" + "=" * 60)
    print("Interactive Q&A Mode")
    print("Type your question and press Enter. Type 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            question = input("\nQ: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        answer = generate_text(
            model, tokenizer, question, device,
            max_len=max_len, temperature=temperature, top_k=top_k, top_p=top_p,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        print(f"A: {answer}")


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sft/full/qa-decoder-only/best_model.pt", help="Path to base model checkpoint")  # 基础模型权重文件路径
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizers/wiki-pretrain-v1/tokenizer.model", help="Path to tokenizer file")  # 分词器文件路径
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA adapter file (optional)")  # LoRA 增量权重路径（可选）
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (must match adapter)")  # LoRA 低秩矩阵维度，需与适配器匹配
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha (must match adapter)")  # LoRA scaling系数，需与适配器匹配
    parser.add_argument("--interactive", action="store_true", help="Enter interactive Q&A mode")  # 是否进入交互式问答模式
    parser.add_argument("--prompt", type=str, default="ROMEO:", help="Input prompt text")  # 输入的prompt文本
    parser.add_argument("--max-len", type=int, default=200, help="Max tokens to generate")  # 最大生成token数
    parser.add_argument("--min-new-tokens", type=int, default=0, help="Minimum new tokens before allowing EOS")  # 允许EOS前必须生成的新token数
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help=">1.0 reduces repetition")  # 重复惩罚系数，大于1可以减少重复
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0, help="If >1, bans repeating ngrams of this size")  # 防止重复n-gram的大小（大于1生效）
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")  # 采样温度
    parser.add_argument("--top-k", type=int, default=40, help="Top-k filtering (0 to disable)")  # top-k采样，0为不启用
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")  # nucleus采样阈值（top-p）
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate")  # 生成样本数量
    args = parser.parse_args()

    device = get_device()
    log.info(f"Using device: {device}")

    model, tokenizer = load_model(
        args.checkpoint, args.tokenizer, device,
        lora_path=args.lora, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
    )
    #interactive是交互模式，如果交互模式为True，则进入交互模式。
    #交互模式的意思是，用户可以输入问题，模型会给出回答。
    #如果交互模式为False，差别是交互模式和非交互模式。交互模式会一直循环，直到用户输入quit。
    if args.interactive:
        interactive_qa(
            model, tokenizer, device,
            max_len=args.max_len, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p,
            min_new_tokens=args.min_new_tokens,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
    else:
        for i in range(args.num_samples):
            output = generate_text(
                model, tokenizer, args.prompt, device,
                max_len=args.max_len, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p,
                min_new_tokens=args.min_new_tokens,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            print(f"\n{'='*60}")
            print(f"Sample {i+1}")
            print(f"{'='*60}")
            print(f"Prompt: {args.prompt}")
            print(f"Generated:\n{output}")


if __name__ == "__main__":
    main()
