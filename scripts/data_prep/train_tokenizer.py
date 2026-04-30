from __future__ import annotations

import argparse

from tokenizer.tokenizer import SentencePieceTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer for the project")
    parser.add_argument("--data-path", type=str, required=True, help="Corpus path: .txt/.json/.jsonl/.tsv or directory")
    parser.add_argument("--output", type=str, default="checkpoints/tokenizers/wiki-pretrain-v1/tokenizer.model", help="Output tokenizer model path")
    parser.add_argument("--vocab-size", type=int, default=16000, help="SentencePiece vocabulary size")
    parser.add_argument("--model-type", type=str, default="unigram", choices=["unigram", "bpe", "char", "word"])
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="仅使用前 N 行抽样训练 tokenizer，适合先做小样本验证。",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="按概率随机抽样语料片段，取值范围 (0, 1]，例如 0.1 表示保留约 10%%。",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="随机抽样种子，保证 sample-rate 结果可复现。",
    )
    return parser.parse_args()


def main() -> None:
    # 该 main() 仅负责加载参数、训练 tokenizer 并保存，不需要循环
    args = parse_args()
    tokenizer = SentencePieceTokenizer.train(
        data_path=args.data_path,                # 训练语料文件路径（可以是文本、json、tsv等或目录）
        model_prefix=args.output.removesuffix(".model"),  # 分词器模型保存前缀（不包含扩展名.model）
        vocab_size=args.vocab_size,              # 词表大小（token数量）
        model_type=args.model_type,              # 分词子模型类型：unigram, bpe, char, word
        max_lines=args.max_lines,                # 最多读取语料的行数，仅部分训练（可选）
        sample_rate=args.sample_rate,            # 对语料随机抽样比例 (0, 1]，降低训练资源消耗
        sample_seed=args.sample_seed,            # 抽样的随机种子，保证可复现
    )
    tokenizer.save(args.output)
    print(f"Tokenizer saved to {args.output}")
    print(f"Vocab size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
