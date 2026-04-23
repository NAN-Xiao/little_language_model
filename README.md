# little-language-model

## 目录结构

```
model/                 # 语言模型核心（Decoder-Only Transformer、LoRA、MoE）
diffusion/             # 扩散模型（DiT、UNet、DDPM、Flow Matching）
  dit/
  unet/
vision/                # 视觉模型（ViT、VAE）
cli/                   # 命令行入口
data/                  # 数据集与 Tokenizer + 训练数据
  pretrain/
  sft/
  eval/
scripts/               # 数据预处理与评测脚本
  data_prep/
  eval/
config.py
utils.py
```

## 使用方式

```bash
# 统一入口
python __main__.py train
python __main__.py finetune
python __main__.py generate
python __main__.py server
python __main__.py train-tokenizer
python __main__.py sample-corpus
python __main__.py eval-geo-prompts

# 也可以直接运行子模块
python -m cli.train --help
python -m scripts.data_prep.train_tokenizer --help
```
大规模预训练

悟道 Wudao — 3TB 中文，最主流的选择
维基百科中文 dump — 你的项目已经有 Wikipedia JSONL 的 IterableDataset 支持
Common Crawl 中文子集 — 网页文本，需要清洗
SkyPile-150B — HuggingFace 上直接下载


小规模练手（推荐先跑通流程）

nlp_chinese_corpus — 维基+百科+新闻，几十GB，够用
CLUECorpus2020 — 清洗过的中文语料
QA 微调数据

BELLE — 中文指令微调数据
alpaca-chinese — 中文 Alpaca 格式
Firefly — 中文多任务对话
建议路径：先用项目自带的 Tiny Shakespeare 跑通训练流程，确认没报错，然后换维基百科中文（你项目已有 Wikipedia JSONL 的流式加载支持），最后再上大规模语料。