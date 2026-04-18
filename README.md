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
