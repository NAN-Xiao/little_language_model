from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PREFIXES = (
    "图片中的",
    "图片中展示的",
    "图片中显示的",
    "图中的",
    "图中展示的",
    "图中显示的",
)


def normalize_question(question: str) -> str:
    normalized = question.strip()
    for prefix in PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    replacements = [
        ("城市的气候类型是什么？", "气候类型是什么？"),
        ("建筑是在什么年份开工兴建的？", "开工兴建于哪一年？"),
        ("建筑群始建于哪一年？", "始建于哪一年？"),
        ("遗址是哪个文明的防御城市遗址？", "是哪个文明的防御城市遗址？"),
    ]
    for old, new in replacements:
        normalized = normalized.replace(old, new)
    return normalized


def build_variants(name: str, normalized_question: str, topic: str | None = None) -> list[str]:
    question_core = normalized_question
    if question_core.startswith("的"):
        question_core = question_core[1:]

    variants = {
        f"{name}{question_core}",
        f"关于{name}，{question_core}",
        f"请回答：{name}{question_core}",
        f"已知对象是{name}。{question_core}",
    }

    if "气候类型是什么" in question_core:
        variants.update(
            {
                f"{name}属于什么气候类型？",
                f"请问{name}的气候类型是什么？",
            }
        )
    if "哪一年" in question_core or "什么年份" in question_core:
        variants.update(
            {
                f"{name}是哪一年开始的？",
                f"{name}始于哪一年？",
            }
        )
    if "哪个文明" in question_core:
        variants.update(
            {
                f"{name}属于哪个文明？",
                f"{name}是哪个文明的遗址？",
            }
        )
    if topic:
        variants.add(f"主题是{topic}。关于{name}，{question_core}")
    return list(variants)


def augment_file(
    input_path: Path,
    output_path: Path,
    seed: int = 42,
    variants_per_item: int = 3,
) -> int:
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with input_path.open("r", encoding="utf-8", errors="replace") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            recognition_answer = str(obj.get("recognition_answer", "")).strip()
            final_question = str(obj.get("final_question", "")).strip()
            final_answer = str(obj.get("final_answer", "")).strip()
            topic = str(obj.get("Topic", "")).strip() or None
            if not (recognition_answer and final_question and final_answer):
                continue

            normalized_question = normalize_question(final_question)
            variants = build_variants(recognition_answer, normalized_question, topic=topic)
            rng.shuffle(variants)
            selected = variants[: max(1, min(variants_per_item, len(variants)))]

            for question in selected:
                dst.write(
                    json.dumps(
                        {
                            "question": question,
                            "answer": final_answer,
                            "source": "chinese_simplevqa_text_natural",
                        },
                        ensure_ascii=False,
                    )
                )
                dst.write("\n")
                written += 1

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create more natural text-only QA from Chinese SimpleVQA.")
    parser.add_argument("--input", type=str, required=True, help="Input chinese_simplevqa.jsonl path")
    parser.add_argument(
        "--output",
        type=str,
        default="data/sft/chinese_simplevqa_text_natural.jsonl",
        help="Output augmented QA jsonl path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--variants-per-item", type=int, default=3, help="How many paraphrase variants to keep per item")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = augment_file(
        Path(args.input),
        Path(args.output),
        seed=args.seed,
        variants_per_item=args.variants_per_item,
    )
    print(f"Wrote {count:,} natural QA pairs to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
