"""
挖洞训练（Span Corruption / FIM）的实现原理

两种"挖洞"方式：
1. FIM (Fill-In-the-Middle): 挖掉代码中间一段，填中间
2. Span Corruption: 随机挖掉多个片段，填这些片段
"""

import random
import torch
import torch.nn.functional as F


# ============================================================================
# 方式1: FIM (Fill-In-the-Middle) —— 代码补全场景
# ============================================================================

def fim_transform(code_text, fim_rate=0.5):
    """
    把一段代码变成 FIM 格式。

    输入: 完整代码字符串
    输出: (prefix, suffix, middle) 或原样返回（按概率 fim_rate）

    示例:
        输入: "def add(a, b):\n    return a + b"
        输出 prefix:  "def add(a, b):\n    "
        输出 middle:  "return a + b"
        输出 suffix:  ""（空）
    """
    if random.random() > fim_rate:
        # 不做 FIM，原样返回（保持普通预训练）
        return None, None, code_text

    # 按行分割，随机选一行作为"分割点"
    lines = code_text.split('\n')
    if len(lines) < 2:
        return None, None, code_text

    # 随机选一个分割位置
    split_idx = random.randint(1, len(lines) - 1)

    # prefix = 分割点之前的所有行
    prefix = '\n'.join(lines[:split_idx])
    # middle = 分割点所在行
    middle = lines[split_idx]
    # suffix = 分割点之后的所有行
    suffix = '\n'.join(lines[split_idx + 1:])

    return prefix, suffix, middle


def fim_to_training_sample(prefix, suffix, middle, tokenizer):
    """
    把 (prefix, suffix, middle) 拼接成模型输入。

    格式: <|fim_prefix|> + prefix + <|fim_suffix|> + suffix + <|fim_middle|> + middle

    模型要预测的是 middle 部分。

    示例:
        输入: prefix="def add(a, b):\n    ", suffix="", middle="return a + b"
        拼接: "<|fim_prefix|>def add(a, b):\n    <|fim_suffix|><|fim_middle|>return a + b"

        模型看到 "<|fim_prefix|>..." → 预测 "return a + b"
    """
    # 特殊 token（需要在 tokenizer 中预先定义）
    FIM_PREFIX = "<|fim_prefix|>"
    FIM_SUFFIX = "<|fim_suffix|>"
    FIM_MIDDLE = "<|fim_middle|>"

    # 拼接: prefix 部分 + suffix 部分 + middle 部分
    # 注意：模型看到的是 prefix + suffix（上下文），要预测的是 middle
    input_text = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}"

    # Tokenize
    tokens = tokenizer.encode(input_text)

    # 找到 FIM_MIDDLE 的位置，只有它后面的 token 才计算 loss
    middle_start = tokens.index(tokenizer.encode(FIM_MIDDLE)[0])

    # 构造 labels：middle 之前的设为 -100（忽略），middle 及之后保留
    labels = [-100] * len(tokens)
    for i in range(middle_start + len(tokenizer.encode(FIM_MIDDLE)), len(tokens)):
        labels[i] = tokens[i]

    return tokens, labels


# ============================================================================
# 方式2: Span Corruption —— 通用预训练增强
# ============================================================================

def span_corruption_transform(tokens, mask_token_id, vocab_size,
                               mask_ratio=0.15, mean_span_length=3):
    """
    随机挖掉连续的 token 片段（spans），让模型填补。

    这是 T5 的训练方式，在代码预训练中也很有效。

    参数:
        tokens: 原始 token id 列表
        mask_ratio: 总共要挖掉多少比例的 token（默认 15%）
        mean_span_length: 每个被挖掉的片段平均多长（默认 3 个 token）

    示例:
        原始: [10, 20, 30, 40, 50, 60, 70, 80, 90]
        挖掉: [30, 40] 和 [70, 80, 90]
        输入: [10, 20, <MASK>, 50, 60, <MASK>]
        目标: [30, 40, 70, 80, 90]
    """
    n_tokens = len(tokens)
    n_to_mask = int(n_tokens * mask_ratio)

    if n_to_mask == 0:
        return tokens, tokens.copy()

    # 计算需要挖掉几个 span
    n_spans = max(1, n_to_mask // mean_span_length)

    # 随机选 span 的起始位置
    span_starts = sorted(random.sample(range(n_tokens), n_spans))

    # 构造输入（带 mask）和目标
    input_tokens = []
    target_tokens = []
    current_pos = 0

    for start in span_starts:
        # span 长度：泊松分布或固定长度
        span_len = random.randint(1, mean_span_length * 2)
        end = min(start + span_len, n_tokens)

        # 把 start 之前的 token 加入输入
        input_tokens.extend(tokens[current_pos:start])

        # 加入一个特殊的 mask token
        input_tokens.append(mask_token_id)

        # 把挖掉的 token 加入目标
        target_tokens.extend(tokens[start:end])

        current_pos = end

    # 加入剩余的 token
    input_tokens.extend(tokens[current_pos:])

    # 目标也要补全长度（和输入对齐，方便计算 loss）
    # 实际上更常见的做法是：目标和输入一样长，非 mask 位置设为 -100
    labels = [-100] * len(input_tokens)

    # 找到所有 mask_token 的位置，把对应的目标 token 填进去
    target_idx = 0
    for i, tok in enumerate(input_tokens):
        if tok == mask_token_id and target_idx < len(target_tokens):
            labels[i] = target_tokens[target_idx]
            target_idx += 1

    return input_tokens, labels


# ============================================================================
# 方式3: 结合 FIM 和 Span Corruption 的混合训练
# ============================================================================

def mixed_corruption_batch(codes, tokenizer, fim_rate=0.3, span_rate=0.3):
    """
    一个 batch 中混合多种"挖洞"方式：
    - 30% 用 FIM（代码补全场景）
    - 30% 用 Span Corruption（通用预训练增强）
    - 40% 不做挖洞（普通因果语言模型）
    """
    batch_tokens = []
    batch_labels = []

    for code in codes:
        r = random.random()

        if r < fim_rate:
            # FIM
            prefix, suffix, middle = fim_transform(code, fim_rate=1.0)
            if prefix is not None:
                tokens, labels = fim_to_training_sample(prefix, suffix, middle, tokenizer)
            else:
                tokens = tokenizer.encode(code)
                labels = tokens.copy()

        elif r < fim_rate + span_rate:
            # Span Corruption
            tokens = tokenizer.encode(code)
            tokens, labels = span_corruption_transform(
                tokens,
                mask_token_id=tokenizer.encode("<|mask|>")[0],
                vocab_size=tokenizer.vocab_size
            )

        else:
            # 普通因果语言模型（不挖洞）
            tokens = tokenizer.encode(code)
            labels = tokens.copy()

        batch_tokens.append(tokens)
        batch_labels.append(labels)

    return batch_tokens, batch_labels


# ============================================================================
# 在训练循环中的使用
# ============================================================================

def training_step_example(model, batch_codes, tokenizer, optimizer):
    """
    训练一步的完整示例
    """
    # 1. 做"挖洞"数据增强
    batch_tokens, batch_labels = mixed_corruption_batch(
        batch_codes, tokenizer,
        fim_rate=0.3,      # 30% FIM
        span_rate=0.3      # 30% Span Corruption
    )

    # 2. Padding（补到相同长度）
    max_len = max(len(t) for t in batch_tokens)
    padded_tokens = []
    padded_labels = []
    for tokens, labels in zip(batch_tokens, batch_labels):
        pad_len = max_len - len(tokens)
        padded_tokens.append(tokens + [0] * pad_len)          # 0 = pad_token
        padded_labels.append(labels + [-100] * pad_len)       # -100 = ignore_index

    input_ids = torch.tensor(padded_tokens)
    labels = torch.tensor(padded_labels)

    # 3. 前向传播
    logits = model(input_ids)  # (batch, seq, vocab)

    # 4. 计算 loss（只有 labels != -100 的位置才计算）
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )

    # 5. 反向传播
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


# ============================================================================
# 核心要点总结
# ============================================================================

"""
【FIM vs Span Corruption 的核心区别】

                    FIM                          Span Corruption
挖洞位置            代码的"中间一段"                随机任意位置
挖洞数量            1个                            多个
输入格式            特殊token标记(prefix/suffix)    特殊mask token
适用场景            IDE代码补全、中间插入           通用预训练、局部修复
模型看到的          prefix + suffix → 预测 middle   带mask的完整文本 → 预测mask处

【为什么有效？】

1. 数据利用率翻倍：
   一段代码 "def f():\n    return 1" 可以产生：
   - 原始: 预测 "return 1"（看后文）
   - FIM: 给定前后文，预测中间（双向利用）
   - Span: 挖掉 "def"，给定其余预测 "def"（强制学习关键词）

2. 和下游任务对齐：
   IDE 的代码补全就是 FIM 场景（光标在中间，前后都有代码）
   Bug 修复就是 Span Corruption 场景（改某几行，其余不变）

3. 不增加模型复杂度：
   不需要改模型结构！只改数据预处理。
   模型还是因果语言模型，只是输入的构造方式变了。

【实现关键点】

- 特殊 token（<|fim_prefix|> 等）需要在 tokenizer 训练时加入词表
- labels 中 -100 表示"这个位置不计算 loss"（PyTorch 的 ignore_index）
- FIM 的比例通常 30%~50%，太高会导致模型"忘记"普通续写能力
- Span Corruption 的 mean_span_length 通常 3~10，太短学不到结构，太长难预测
"""
