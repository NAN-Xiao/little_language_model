from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from ..tokenizer.tokenizer import SentencePieceTokenizer

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
DATA_DIR = Path(__file__).parent / "raw"


def download_tiny_shakespeare(force: bool = False) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / "tiny_shakespeare.txt"
    if not filepath.exists() or force:
        print(f"Downloading Tiny Shakespeare to {filepath} ...")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, filepath)
        print("Done.")
    return filepath.read_text(encoding="utf-8")


def load_text(data_path: str | Path | None = None) -> str:
    if data_path is None:
        return download_tiny_shakespeare()
    p = Path(data_path)
    if p.is_file():
        print(f"Loading text from file: {p}")
        return p.read_text(encoding="utf-8")
    if p.is_dir():
        txt_files = sorted(p.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in directory: {p}")
        print(f"Loading {len(txt_files)} text file(s) from directory: {p}")
        return "\n".join(f.read_text(encoding="utf-8") for f in txt_files)
    raise FileNotFoundError(f"Data path does not exist: {p}")


def is_wikipedia_title_text_jsonl(path: str | Path) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    if p.suffix.lower() not in (".json", ".jsonl"):
        return False
    with p.open("r", encoding="utf-8", errors="replace") as f:
        line = f.readline()
    if not line.strip():
        return False
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return False
    return isinstance(obj, dict) and "title" in obj and "text" in obj


def build_line_byte_offsets(path: str | Path) -> list[int]:
    offsets: list[int] = []
    with Path(path).open("rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(pos)
    return offsets


def collect_chars_wikipedia_jsonl(
    path: str | Path,
    byte_offsets: list[int],
    line_start: int,
    line_end: int,
) -> set[str]:
    chars: set[str] = set()
    p = Path(path)
    with p.open("rb") as f:
        for li in range(line_start, line_end):
            if li >= len(byte_offsets):
                break
            f.seek(byte_offsets[li])
            raw = f.readline()
            try:
                obj = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(obj, dict):
                continue
            title = obj.get("title") or ""
            text = obj.get("text") or ""
            chars.update(str(title))
            chars.update(str(text))
            chars.update("\n")
    return chars


class WikipediaJsonlBlockIterable(IterableDataset):
    """按行读取维基 JSONL，流式切成 decoder-only 训练样本。"""

    def __init__(
        self,
        path: str | Path,
        tokenizer: SentencePieceTokenizer,
        seq_len: int,
        byte_offsets: list[int],
        line_start: int,
        line_end: int,
        shuffle_lines: bool = False,
    ):
        super().__init__()
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.byte_offsets = byte_offsets
        self.line_start = line_start
        self.line_end = line_end
        self.shuffle_lines = shuffle_lines
        self.block_len = seq_len + 1

    def __iter__(self):
        from torch.utils.data import get_worker_info

        worker = get_worker_info()
        idxs = list(range(self.line_start, self.line_end))
        if self.shuffle_lines:
            random.shuffle(idxs)
        if worker is not None:
            idxs = idxs[worker.id :: worker.num_workers]

        buffer: list[int] = []
        with self.path.open("rb") as f:
            for li in idxs:
                if li < 0 or li >= len(self.byte_offsets):
                    continue
                f.seek(self.byte_offsets[li])
                raw = f.readline()
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(obj, dict):
                    continue
                title = obj.get("title") or ""
                text = obj.get("text") or ""
                piece = f"{title}\n{text}\n\n"
                buffer.extend(self.tokenizer.encode(piece, add_special=False))
                while len(buffer) >= self.block_len:
                    block = buffer[: self.block_len]
                    del buffer[: self.block_len]
                    yield (
                        torch.tensor(block[:-1], dtype=torch.long),
                        torch.tensor(block[1:], dtype=torch.long),
                    )


def load_qa_pairs(data_path: str | Path) -> list[tuple[str, str]]:
    p = Path(data_path)

    if p.is_dir():
        files = sorted(p.glob("*.jsonl")) + sorted(p.glob("*.tsv"))
        if not files:
            raise FileNotFoundError(f"No .jsonl or .tsv files found in: {p}")
        pairs: list[tuple[str, str]] = []
        for f in files:
            pairs.extend(_load_single_qa_file(f))
        print(f"Loaded {len(pairs)} QA pairs from {len(files)} file(s) in {p}")
        return pairs

    if p.is_file():
        pairs = _load_single_qa_file(p)
        print(f"Loaded {len(pairs)} QA pairs from {p}")
        return pairs

    raise FileNotFoundError(f"QA data path does not exist: {p}")


def _load_single_qa_file(path: Path) -> list[tuple[str, str]]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    pairs: list[tuple[str, str]] = []

    if path.suffix == ".jsonl":
        for line in lines:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("question", obj.get("q", ""))
            a = obj.get("answer", obj.get("a", ""))
            if q and a:
                pairs.append((q, a))
            recog_q = obj.get("recognition_question", "")
            recog_a = obj.get("recognition_answer", "")
            if recog_q and recog_a:
                pairs.append((recog_q, recog_a))
            final_q = obj.get("final_question", "")
            final_a = obj.get("final_answer", "")
            if final_q and final_a:
                pairs.append((final_q, final_a))
    else:
        for line in lines:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                pairs.append((parts[0], parts[1]))

    return pairs


class CausalLMDataset(Dataset):
    """连续文本切块为 decoder-only next-token 样本。"""

    def __init__(self, token_ids: list[int], seq_len: int):
        block_len = seq_len + 1
        n_blocks = len(token_ids) // block_len
        trimmed = token_ids[: n_blocks * block_len]
        self.blocks = [
            trimmed[i : i + block_len] for i in range(0, len(trimmed), block_len)
        ]

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int):
        block = self.blocks[idx]
        return (
            torch.tensor(block[:-1], dtype=torch.long),
            torch.tensor(block[1:], dtype=torch.long),
        )


class QADataset(Dataset):
    """问答 SFT 数据，问题部分仅作为上下文，loss 只作用于答案。"""

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        tokenizer: SentencePieceTokenizer,
        max_len: int = 128,
    ):
        self.samples: list[tuple[list[int], list[int]]] = []

        for question, answer in pairs:
            prompt = f"问题：{question}\n答案："
            prompt_ids = tokenizer.encode(prompt, add_special=False)
            answer_ids = tokenizer.encode(answer, add_special=False)
            full = [tokenizer.bos_token_id] + prompt_ids + answer_ids + [tokenizer.eos_token_id]
            if len(full) < 2:
                continue
            input_ids = full[:-1][:max_len]
            labels = full[1:][:max_len]
            ignore_len = min(len(prompt_ids), len(labels))
            labels[:ignore_len] = [-100] * ignore_len
            self.samples.append((input_ids, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        input_ids, labels = self.samples[idx]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def collate_fn(batch):
    input_ids, labels = zip(*batch)
    max_len = max(x.size(0) for x in input_ids)

    inputs_padded = torch.zeros(len(input_ids), max_len, dtype=torch.long)
    labels_padded = torch.full((len(labels), max_len), -100, dtype=torch.long)

    for i, (inp, lab) in enumerate(zip(input_ids, labels)):
        inputs_padded[i, : inp.size(0)] = inp
        labels_padded[i, : lab.size(0)] = lab

    return inputs_padded, labels_padded


def create_dataloaders(
    seq_len: int = 128,
    batch_size: int = 64,
    val_split: float = 0.1,
    data_path: str | Path | None = None,
    tokenizer: SentencePieceTokenizer | None = None,
) -> tuple[DataLoader, DataLoader, SentencePieceTokenizer]:
    if data_path is not None:
        p = Path(data_path)
        if not p.exists():
            raise FileNotFoundError(
                f"数据路径不存在: {p}\n"
                f"当前工作目录: {Path.cwd()}\n"
                f"请使用相对或绝对路径；维基语料若在仓库内常见为 data/raw/wikipedia-zh-cn-*.json"
            )
        if p.is_file() and is_wikipedia_title_text_jsonl(p):
            if tokenizer is None:
                raise ValueError("Wikipedia JSONL 训练需要先准备好 SentencePiece tokenizer。")
            byte_offsets = build_line_byte_offsets(p)
            n = len(byte_offsets)
            if n < 2:
                raise ValueError("JSONL 至少需要 2 行才能划分 train/val。")
            split_idx = max(1, int(n * (1 - val_split)))
            split_idx = min(split_idx, n - 1)
            train_ds = WikipediaJsonlBlockIterable(
                p, tokenizer, seq_len, byte_offsets, 0, split_idx, shuffle_lines=True
            )
            val_ds = WikipediaJsonlBlockIterable(
                p, tokenizer, seq_len, byte_offsets, split_idx, n, shuffle_lines=False
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                drop_last=True,
                num_workers=0,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                drop_last=False,
                num_workers=0,
            )
            return train_loader, val_loader, tokenizer

    text = load_text(data_path)
    if tokenizer is None:
        raise ValueError("文本训练需要先准备好 SentencePiece tokenizer。")

    all_ids = tokenizer.encode(text, add_special=False)
    split_idx = int(len(all_ids) * (1 - val_split))
    train_ds = CausalLMDataset(all_ids[:split_idx], seq_len)
    val_ds = CausalLMDataset(all_ids[split_idx:], seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return train_loader, val_loader, tokenizer


def create_qa_dataloaders(
    data_path: str | Path,
    max_len: int = 128,
    batch_size: int = 64,
    val_split: float = 0.1,
    tokenizer: SentencePieceTokenizer | None = None,
) -> tuple[DataLoader, DataLoader, SentencePieceTokenizer]:
    pairs = load_qa_pairs(data_path)

    if tokenizer is None:
        raise ValueError("QA 训练需要先准备好 SentencePiece tokenizer。")

    print(f"QA pairs: {len(pairs):,}, Vocab size: {tokenizer.vocab_size}")

    random.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - val_split))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    train_ds = QADataset(train_pairs, tokenizer, max_len)
    val_ds = QADataset(val_pairs, tokenizer, max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return train_loader, val_loader, tokenizer
