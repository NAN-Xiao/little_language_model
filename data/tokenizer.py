from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import sentencepiece as spm


class SentencePieceTokenizer:
    """SentencePiece tokenizer with the same interface as the old CharTokenizer."""

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.processor = spm.SentencePieceProcessor(model_file=str(self.model_path))
        self.vocab_size = self.processor.vocab_size()

    @property
    def pad_token_id(self) -> int:
        return int(self.processor.pad_id())

    @property
    def bos_token_id(self) -> int:
        return int(self.processor.bos_id())

    @property
    def eos_token_id(self) -> int:
        return int(self.processor.eos_id())

    @property
    def unk_token_id(self) -> int:
        return int(self.processor.unk_id())

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = list(self.processor.encode(text, out_type=int))
        if add_special:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        if skip_special:
            special_ids = {
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
            }
            ids = [token_id for token_id in ids if token_id not in special_ids]
        return self.processor.decode(ids)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        target = path if path.suffix == ".model" else path.with_suffix(".model")
        target.write_bytes(self.model_path.read_bytes())

        meta_path = target.with_suffix(".json")
        meta_path.write_text(
            json.dumps(
                {
                    "type": "sentencepiece",
                    "model_path": str(target.name),
                    "vocab_size": self.vocab_size,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> SentencePieceTokenizer:
        path = Path(path)
        if path.suffix == ".json":
            meta = json.loads(path.read_text(encoding="utf-8"))
            model_path = path.with_name(meta["model_path"])
            return cls(model_path)
        if path.suffix != ".model":
            model_path = path.with_suffix(".model")
            if model_path.exists():
                return cls(model_path)
        return cls(path)

    @classmethod
    def train(
        cls,
        data_path: str | Path,           # 语料数据路径，可以是文件(.txt/.json/.jsonl/.tsv)或目录，作为分词器训练数据
        model_prefix: str | Path,        # SentencePiece模型保存前缀（不带扩展名.model），决定模型保存的目标路径和文件名
        vocab_size: int = 16000,         # 分词器词表大小（token数量），决定最终模型支持多少个token
        model_type: str = "unigram",     # 分词模型类型，支持 "unigram"、"bpe"、"char"、"word"
        character_coverage: float = 0.9995, # 覆盖多少独特字符，影响所保留的字符集比例，较低会舍弃稀有字符
        max_lines: int | None = None,    # 仅加载前N行语料用于训练，None则使用全部语料，适合小样本快速验证
        sample_rate: float = 1.0,        # 从语料中随机采样的概率，减小可降低资源消耗，取值范围(0,1]
        sample_seed: int = 42,           # 随机采样种子，保证采样实验可复现
    ) -> SentencePieceTokenizer:
        """""
        1. 创建模型保存前缀相关的目录，确保分词器模型可以正确写入目标位置
        """""
        model_prefix = Path(model_prefix)
        model_prefix.parent.mkdir(parents=True, exist_ok=True)

        """""
        2. 构建临时训练语料文件，并按设定随机采样/截断，将清洗后的训练文本写入spm_corpus.txt
           通过write_corpus_text支持从多种格式与数据源读取
        """""
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = Path(tmpdir) / "spm_corpus.txt"
            _, written_lines = write_corpus_text(
                data_path,
                corpus_path,
                max_lines=max_lines,
                sample_rate=sample_rate,
                sample_seed=sample_seed,
            )
            print(
                f"Prepared tokenizer corpus: {written_lines:,} lines "
                f"(sample_rate={sample_rate}, sample_seed={sample_seed})"
            )
            """""
            3. 用 SentencePieceTrainer 加载语料并训练分词器模型；参数里详细指定模型类型、词表大小、特殊符号、字符覆盖率、
               归一化规则等，避免中文或特殊数据训练异常；模型最终存储到模型前缀位置
                    那整个 spm.SentencePieceTrainer.train是黑盒的，我们只需要知道它需要什么参数，然后给它就可以了
            """""
            spm.SentencePieceTrainer.train(
                input=str(corpus_path),
                model_prefix=str(model_prefix),
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=character_coverage,
                pad_id=0,
                bos_id=1,
                eos_id=2,
                unk_id=3,
                pad_piece=cls.PAD_TOKEN,
                bos_piece=cls.BOS_TOKEN,
                eos_piece=cls.EOS_TOKEN,
                unk_piece=cls.UNK_TOKEN,
                normalization_rule_name="nmt_nfkc",
                train_extremely_large_corpus=True,
                max_sentence_length=16384,
                hard_vocab_limit=False,
            )
        """""
        4. 返回已训练的SentencePieceTokenizer对象，加载生成的.model模型权重
        """""
        return cls(model_prefix.with_suffix(".model"))


def iter_corpus_text(data_path: str | Path):
    """""
    1. 遍历数据源，支持多种格式(.txt/.json/.jsonl/.tsv)，自动递归处理目录
    """""
    path = Path(data_path)
    if path.is_dir():
        for file_path in path.glob("*"):
            yield from iter_corpus_text(file_path)
        return
    """""
    3. 处理单个文件，支持文本(.txt)、JSON(.json/.jsonl)、TSV(.tsv)格式
    """""
    if path.suffix.lower() == ".txt":
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield line
        return

    if path.suffix.lower() in {".json", ".jsonl"}:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                title = obj.get("title")
                text = obj.get("text")
                if isinstance(title, str) and isinstance(text, str):
                    yield title
                    yield text
                    continue
                question = obj.get("question", obj.get("q"))
                answer = obj.get("answer", obj.get("a"))
                if isinstance(question, str):
                    yield question
                if isinstance(answer, str):
                    yield answer
        return

    if path.suffix.lower() == ".tsv":
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2:
                    continue
                if parts[0].strip():
                    yield parts[0].strip()
                if parts[1].strip():
                    yield parts[1].strip()
        return

    raise ValueError(f"Unsupported tokenizer corpus file: {path}")


def write_corpus_text(
    data_path: str | Path,
    output_path: str | Path,
    max_lines: int | None = None,
    sample_rate: float = 1.0,
    sample_seed: int = 42,
) -> tuple[Path, int]:
    """""
    1. 确保输出路径目录存在，并创建临时训练语料文件，按设定随机采样/截断，将清洗后的训练文本写入spm_corpus.txt
    """""
    if not (0.0 < sample_rate <= 1.0):
        raise ValueError("sample_rate must be in the range (0, 1].")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    rng = random.Random(sample_seed)
    with output_path.open("w", encoding="utf-8") as handle:
        for line in iter_corpus_text(data_path):
            if sample_rate < 1.0 and rng.random() > sample_rate:
                continue
            handle.write(line)
            handle.write("\n")
            written += 1
            if max_lines is not None and written >= max_lines:
                break
    return output_path, written


def ensure_tokenizer(
    data_path: str | Path,         # 训练语料路径（文本/JSON/TSV文件或目录）
    tokenizer_path: str | Path,    # tokenizer模型保存/加载路径（通常为 .model 文件）
    vocab_size: int = 16000,       # 词表大小，决定分词器能识别的token数量
    model_type: str = "unigram",   # 分词模型类型：unigram, bpe, char, word
    retrain: bool = False,         # 是否强制重新训练分词器，即使文件已存在也会覆盖
    max_lines: int | None = None,  # 最多用于训练的语料行数（None为全部）
    sample_rate: float = 1.0,      # 训练时对语料采样的概率，(0, 1]，减少数据量可加快速度
    sample_seed: int = 42,         # 采样随机种子，保证采样复现性
) -> SentencePieceTokenizer:
    """""
    1. 如果需要重新训练分词器，则调用SentencePieceTokenizer.train()，否则调用SentencePieceTokenizer.load()
    """""
    tokenizer_path = Path(tokenizer_path)
    model_path = tokenizer_path if tokenizer_path.suffix == ".model" else tokenizer_path.with_suffix(".model")
    if retrain or not model_path.exists():
        return SentencePieceTokenizer.train(
            data_path=data_path,
            model_prefix=model_path.with_suffix(""),
            vocab_size=vocab_size,
            model_type=model_type,
            max_lines=max_lines,
            sample_rate=sample_rate,
            sample_seed=sample_seed,
        )
    return SentencePieceTokenizer.load(model_path)


CharTokenizer = SentencePieceTokenizer
