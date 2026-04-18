from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_text_only_question(
    recognition_answer: str,
    final_question: str,
    topic: str | None = None,
) -> str:
    parts: list[str] = []
    if topic:
        parts.append(f"主题：{topic}。")
    parts.append(f"已知对象是“{recognition_answer}”。")

    normalized = final_question.strip()
    replacements = [
        "图片中的",
        "图片中展示的",
        "图片中显示的",
        "图中的",
        "图中展示的",
        "图中显示的",
    ]
    for prefix in replacements:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    parts.append(normalized)
    return "".join(parts)


def convert_file(input_path: Path, output_path: Path) -> int:
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

            question = build_text_only_question(recognition_answer, final_question, topic)
            dst.write(
                json.dumps(
                    {
                        "question": question,
                        "answer": final_answer,
                        "source": "chinese_simplevqa_text_only",
                    },
                    ensure_ascii=False,
                )
            )
            dst.write("\n")
            written += 1
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Chinese SimpleVQA into text-only QA pairs.")
    parser.add_argument("--input", type=str, required=True, help="Input chinese_simplevqa.jsonl path")
    parser.add_argument(
        "--output",
        type=str,
        default="data/sft/chinese_simplevqa_textonly.jsonl",
        help="Output QA jsonl path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = convert_file(Path(args.input), Path(args.output))
    print(f"Wrote {count:,} text-only QA pairs to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
