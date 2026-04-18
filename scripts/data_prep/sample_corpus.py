from __future__ import annotations

import argparse
import random
from pathlib import Path


def reservoir_sample_lines(
    input_path: Path,
    output_path: Path,
    num_lines: int,
    seed: int = 42,
) -> int:
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0

    with input_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            seen += 1
            if len(reservoir) < num_lines:
                reservoir.append(line)
                continue
            j = rng.randrange(seen)
            if j < num_lines:
                reservoir[j] = line

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for line in reservoir:
            out.write(line)
            out.write("\n")

    return len(reservoir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly sample lines from a large text/JSONL file (streaming).")
    parser.add_argument("--input", type=str, required=True, help="Input file path (one JSON per line or plain text).")
    parser.add_argument("--output", type=str, required=True, help="Output file path.")
    parser.add_argument("--num-lines", type=int, required=True, help="Number of lines to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if args.num_lines <= 0:
        raise ValueError("--num-lines must be > 0")
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    n = reservoir_sample_lines(input_path, output_path, num_lines=args.num_lines, seed=args.seed)
    print(f"Wrote {n:,} sampled lines to {output_path.resolve()}")


if __name__ == "__main__":
    main()
