from __future__ import annotations

import argparse
import json
from pathlib import Path

from cli.generate import load_model, generate_text
from utils import get_device, get_logger

log = get_logger()


def normalize_prompt(line: str) -> str:
    prompt = line.strip()
    if not prompt:
        return ""
    if "\n答案：" in prompt:
        return prompt
    return f"{prompt}\n答案："


def main():
    parser = argparse.ArgumentParser(description="Batch-evaluate QA prompts from a text file")
    parser.add_argument("--input", type=str, required=True, help="Prompt file path, one question per line")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer model path")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    parser.add_argument("--max-len", type=int, default=20, help="Max generation length")
    parser.add_argument("--min-new-tokens", type=int, default=1, help="Minimum new tokens")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.3, help="Top-p sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.02, help="Repetition penalty")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=2, help="No-repeat ngram size")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = [normalize_prompt(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
    prompts = [prompt for prompt in prompts if prompt]

    device = get_device(args.device)
    log.info(f"Using device: {device}")

    model, tokenizer = load_model(args.checkpoint, args.tokenizer, device)

    with output_path.open("w", encoding="utf-8") as f:
        for index, prompt in enumerate(prompts, 1):
            answer = generate_text(
                model,
                tokenizer,
                prompt,
                device,
                max_len=args.max_len,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                min_new_tokens=args.min_new_tokens,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            ).strip()
            record = {
                "index": index,
                "prompt": prompt,
                "answer": answer,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.info(f"[{index}/{len(prompts)}] {prompt.splitlines()[0]} -> {answer}")

    log.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
