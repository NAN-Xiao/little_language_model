# LLM 训练方法大全

> 从预训练到后训练，从通用能力到专项能力，系统梳理大语言模型的各种训练方法。标注 **【必要】** 的为工业界标准做法，**【可选】** 为锦上添花的增强手段。

---

## 目录

- [一、预训练阶段 (Pre-training)](#一预训练阶段-pre-training)
- [二、指令微调阶段 (SFT)](#二指令微调阶段-sft)
- [三、对齐阶段 (Alignment)](#三对齐阶段-alignment)
- [四、长上下文扩展](#四长上下文扩展)
- [五、专项能力训练](#五专项能力训练)
- [六、数据工程方法](#六数据工程方法)
- [七、训练技巧与优化](#七训练技巧与优化)
- [八、推荐组合](#八推荐组合)

---

## 一、预训练阶段 (Pre-training)

预训练的目标：让模型"学会语言"。在数万亿 token 上训练，学习语法、知识、推理基础。

---

### 1.1 因果语言模型 (Causal Language Modeling, CLM)【必要】

**原理**：看到前面的 token，预测下一个 token。

```
输入:  "今天天气"
目标:  "很好"

输入:  "今天天气很好"
目标:  "。"
```

**实现**：

```python
# Transformer decoder 的标准训练方式
logits = model(input_ids)  # (batch, seq, vocab)
loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size),
                       input_ids[:, 1:].reshape(-1))
```

**为什么必要**：所有 decoder-only 模型的基础训练方式，GPT、LLaMA、Qwen 都采用。

---

### 1.2 全词掩码 (Whole Word Masking)【可选】

**原理**：不把单个 token 作为掩码单位，而是把整个词（由多个 token 组成）一起掩码。

**示例**：

```
普通掩码："我爱<MASK><MASK>"（掩码 2 个 token）
全词掩码："我爱<MASK><MASK>"（"北京"由 2 个 token 组成，一起掩码）
```

**作用**：让模型学习"词级别"的语义关系，而不是 token 级别的表面关联。

**适用**：中文预训练尤其重要（因为中文一个字可能对应多个 token）。

---

### 1.3 前缀语言模型 (Prefix LM)【可选】

**原理**：输入的前半部分用双向 attention，后半部分用因果 attention。

```
输入:  "[双向] 今天天气 [因果] 很好，适合出门。"
        ↑ 这部分双向可见          ↑ 这部分只能向左看
```

**代表模型**：T5、U-PaLM

**适用场景**：需要同时理解上下文和生成后续内容的任务（如摘要、翻译）。

**缺点**：实现复杂，性能不如纯 causal decoder，现在较少使用。

---

### 1.4 去噪自编码 (Denoising Autoencoding)【可选】

**原理**：在输入中加入噪声（删除、替换、打乱 token），让模型恢复原文。

**噪声类型**：

| 噪声类型 | 示例 | 恢复目标 |
|---------|------|---------|
| Token 删除 | "今天 天气 很好" → "今天 <MASK> 很好" | "天气" |
| Token 替换 | "今天 天气 很好" → "今天 食物 很好" | "天气" |
| Span 打乱 | "A B C D" → "A <MASK> <MASK> D" | "B C" |
| 文档旋转 | 把文档从中间切断，前后互换 | 恢复原始顺序 |

**代表模型**：T5、BART

**适用场景**：Encoder-Decoder 架构的模型，现在 decoder-only 模型较少使用。

---

### 1.5 多任务预训练 (Multi-task Pre-training)【可选】

**原理**：在预训练阶段就引入多种任务，不只是预测下一个 token。

```
任务1: 因果语言建模（预测下一个token）
任务2: 句子排序（给乱序的句子排正确顺序）
任务3: 文档分类（判断两段文字是否来自同一文档）
任务4: 常识推理（选择最合理的下一句）
```

**代表模型**：GLM、T5

**优势**：预训练阶段就学习多种能力，减少后续微调需求。

---

## 二、指令微调阶段 (SFT)

SFT 的目标：让模型"听懂指令"。用 `(指令, 回答)` 成对数据训练。

---

### 2.1 标准指令微调 (Standard SFT)【必要】

**数据格式**：

```json
{
  "instruction": "解释一下量子力学",
  "input": "",
  "output": "量子力学是研究微观粒子运动规律的物理学分支..."
}
```

**关键设置**：

```python
# 只计算 output 部分的 loss，instruction 部分忽略
text = f"用户：{instruction}\n助手：{output}"
labels = [-100] * len(instruction_tokens) + output_tokens
```

**为什么必要**：没有 SFT，模型只会"续写"，不会"回答问题"。

---

### 2.2 多轮对话微调 (Multi-turn SFT)【必要】

**数据格式**：

```json
{
  "messages": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮你的？"},
    {"role": "user", "content": "讲个笑话"},
    {"role": "assistant", "content": "为什么程序员总把万圣节和圣诞节搞混？因为 Oct 31 = Dec 25！"}
  ]
}
```

**Loss Mask**：

```
tokens:   [用户, 你好, 助手, 你好！, 用户, 讲个笑话, 助手, 为什...]
labels:   [-100, -100, 你好！, ... , -100, -100, -100, 为什...]
           ↑ 用户说的话不算loss    ↑ 助手说的话算loss
```

**为什么必要**：单轮 SFT 训练出的模型不会"记住之前的对话"。

---

### 2.3 角色扮演微调 (Role-play SFT)【可选】

**原理**：给模型设定特定角色，训练它在不同角色下的回答风格。

```json
{
  "system": "你是一位资深 Python 程序员，回答要简洁，多用代码示例。",
  "messages": [
    {"role": "user", "content": "怎么写快排？"},
    {"role": "assistant", "content": "```python\ndef quicksort(arr):..."}
  ]
}
```

**作用**：让模型在特定场景下有更专业的表现。

---

### 2.4 自我指令 (Self-Instruct)【可选】

**原理**：用模型自己生成训练数据。

```
步骤1: 准备 100 条人工写的种子指令
步骤2: 让模型基于种子指令生成类似的 1000 条新指令
步骤3: 让模型回答这 1000 条指令
步骤4: 用生成的 (指令, 回答) 对训练模型
```

**代表工作**：Alpaca、WizardLM

**优势**：低成本获取大量指令数据。

**风险**：模型自我强化，可能放大错误。

---

### 2.5 拒绝采样微调 (Rejection Sampling Fine-Tuning, RFT)【可选】

**原理**：让模型生成多个回答，选最好的用于训练。

```python
# 对同一个问题生成 10 个回答
answers = [model.generate(question) for _ in range(10)]

# 用规则/模型评分选出最好的
best = select_best(answers, criteria="流畅度 + 准确性 + 安全性")

# 用最好的回答做 SFT
sft_train(question, best)
```

**作用**：提升回答质量的上限。

---

## 三、对齐阶段 (Alignment)

对齐的目标：让模型"回答得更符合人类偏好"（有用、无害、诚实）。

---

### 3.1 RLHF (Reinforcement Learning from Human Feedback)【必要】

**三阶段流程**：

```
阶段1: 训练奖励模型 (RM)
        人工标注 (回答A, 回答B) 哪个更好
        训练一个模型来预测"人类偏好分数"

阶段2: 用 PPO 优化策略
        模型生成回答 → 奖励模型打分 → PPO 更新模型参数
        目标: 最大化奖励分数，同时不要偏离原始模型太远 (KL散度约束)
```

**核心代码**：

```python
# PPO 训练循环
for batch in dataloader:
    # 旧模型生成回答
    old_responses = old_model.generate(batch.prompts)

    # 新模型生成回答
    new_responses = new_model.generate(batch.prompts)

    # 奖励模型打分
    rewards = reward_model(new_responses)

    # PPO 更新（带 KL 惩罚）
    kl_penalty = KL(new_model, old_model)
    loss = -rewards + beta * kl_penalty
    loss.backward()
    optimizer.step()
```

**代表模型**：InstructGPT、ChatGPT、Claude

**为什么必要**：SFT 只能让模型"回答问题"，RLHF 让模型"回答得好"。

---

### 3.2 DPO (Direct Preference Optimization)【可选】

**原理**：直接用偏好数据优化模型，不需要训练奖励模型。

```python
# DPO 损失函数
# 给定 (prompt, chosen, rejected)
loss = -log(sigmoid(
    beta * (log_p_chosen - log_p_rejected)
))
```

**优势**：
- 不需要训练奖励模型（省一个阶段）
- 不需要 PPO（省复杂实现）
- 训练更稳定

**缺点**：对数据质量要求高，数据噪音会直接影响模型。

**代表模型**：Zephyr、Llama-2-Chat（部分使用）

---

### 3.3 KTO (Kahneman-Tversky Optimization)【可选】

**原理**：不需要成对偏好数据，只需要"这个回答好/不好"的标注。

```python
# KTO 损失
# 给定 (prompt, response, label)  label = 1(好) 或 0(不好)
loss = label * log(sigmoid(reward)) + (1-label) * log(sigmoid(-reward))
```

**优势**：
- 数据获取成本更低（不需要 A/B 对比）
- 适合标注资源有限的场景

---

### 3.4 ORPO (Odds Ratio Preference Optimization)【可选】

**原理**：把 SFT 和对齐合二为一，在一个阶段完成。

```python
# ORPO 损失 = SFT_loss + Preference_loss
# 同时训练模型"能回答"和"回答得好"
```

**优势**：省一个训练阶段，效率更高。

**缺点**：效果略逊于分阶段训练（RLHF 或 DPO）。

---

### 3.5 RLAIF (RL from AI Feedback)【可选】

**原理**：不用人类标注，用更强模型（如 GPT-4）来评判回答好坏。

```python
# 用 GPT-4 替代人类标注者
for response_a, response_b in pairs:
    preference = gpt4_judge(response_a, response_b)
    # "回答A比回答B更好，因为..."
```

**优势**：低成本获取大量偏好数据。

**风险**：
- AI 评判可能有偏见
- 弱模型容易被强模型的偏好"带偏"

**代表模型**：Claude 3、Llama 2（部分使用）

---

### 3.6 各种对齐方法对比

| 方法 | 需要奖励模型 | 需要成对数据 | 训练阶段 | 稳定性 | 效果上限 |
|------|------------|------------|---------|--------|---------|
| **RLHF/PPO** | ✅ 需要 | ✅ 需要 | 3 阶段 | 中等 | 高 |
| **DPO** | ❌ 不需要 | ✅ 需要 | 2 阶段 | 高 | 中高 |
| **KTO** | ❌ 不需要 | ❌ 不需要 | 2 阶段 | 高 | 中 |
| **ORPO** | ❌ 不需要 | ✅ 需要 | 1 阶段 | 高 | 中 |
| **RLAIF** | ✅ 需要 | ✅ 需要 | 3 阶段 | 中等 | 高 |

---

## 四、长上下文扩展

---

### 4.1 位置编码外推 (Positional Extrapolation)【必要】

**原理**：调整 RoPE 的旋转角度，让模型适应更长的序列。

```python
# NTK-aware 扩展
def extend_rope(model, original=4096, target=131072):
    scale = target / original
    model.rope_theta *= scale ** (2 / model.head_dim)
    return model
```

**为什么必要**：基础模型训练时最多 4K/8K，要支持 128K 必须做扩展。

---

### 4.2 渐进式长度扩展 (Progressive Length Extension)【可选】

**原理**：逐步增加训练长度，而不是一步跳到目标长度。

```
阶段1: 在 4K 长度上训练
阶段2: 扩展到 8K，训练 1000 步
阶段3: 扩展到 16K，训练 1000 步
阶段4: 扩展到 32K，训练 1000 步
...
```

**优势**：模型更容易适应长度变化，减少训练不稳定性。

---

### 4.3 长短文本混合训练 (Mixed Length Training)【可选】

**原理**：一个 batch 中同时包含短文本和长文本。

```python
# 避免所有样本都是长文本导致训练效率低
batch = [
    short_text_1k,   # 20%
    medium_text_8k,  # 30%
    long_text_32k,   # 30%
    very_long_128k,  # 20%
]
```

**作用**：保持模型对短文本的处理能力，同时学习长文本。

---

## 五、专项能力训练

---

### 5.1 思维链训练 (Chain-of-Thought, CoT)【必要】

**原理**：训练模型先输出思考过程，再给出答案。

```json
{
  "question": "3 + 5 * 2 = ?",
  "answer": "先算乘法：5 * 2 = 10\n再算加法：3 + 10 = 13\n答案是 13"
}
```

**为什么必要**：没有 CoT，模型做复杂推理时容易出错。

---

### 5.2 工具使用训练 (Tool Use)【可选】

**原理**：训练模型生成结构化工具调用。

```json
{
  "messages": [
    {"role": "user", "content": "北京今天天气怎么样？"},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"weather_api\", \"parameters\": {\"city\": \"北京\"}}\n</tool_call>"},
    {"role": "tool", "content": "{\"temperature\": 25, \"weather\": \"晴\"}"},
    {"role": "assistant", "content": "北京今天 25 度，晴天。"}
  ]
}
```

---

### 5.3 数学专项训练 (Math Specialization)【可选】

**数据类型**：
- 数学问题 + 分步解答
- 形式化证明（Lean、Coq）
- 代码验证（用 Python 验证答案）

```json
{
  "problem": "证明：对于任意正整数 n，1+2+...+n = n(n+1)/2",
  "solution": "用数学归纳法：\n基础情况：n=1 时，左边=1，右边=1*2/2=1，成立。\n归纳假设：假设 n=k 时成立。\n归纳步骤：n=k+1 时，左边=1+...+k+(k+1)=k(k+1)/2+(k+1)=(k+1)(k+2)/2，成立。"
}
```

---

### 5.4 多语言训练 (Multilingual Training)【可选】

**策略**：
- **平衡采样**：每种语言按相同比例采样（避免英语主导）
- **语言标识**：在输入前加语言标记（`<|zh|>`、`<|en|>`）
- **翻译对训练**：平行语料增强跨语言理解

---

## 六、数据工程方法

---

### 6.1 数据清洗 (Data Cleaning)【必要】

**清洗内容**：

| 问题 | 处理方式 |
|------|---------|
| 重复内容 | MinHash/LSH 去重 |
| 低质量文本 | 困惑度过滤（PPL > 阈值丢弃） |
| 垃圾信息 | 规则过滤（广告、乱码、过度重复） |
| 个人隐私 | 正则匹配删除邮箱、手机号、身份证号 |
| 有害内容 | 分类器过滤（暴力、色情、歧视） |

**为什么必要**：Garbage in, garbage out。数据质量决定模型上限。

---

### 6.2 数据混合 (Data Mixing)【必要】

**原理**：不同来源的数据按特定比例混合。

```python
data_mix = {
    "web_pages": 0.60,      # 网页（通用知识）
    "code": 0.15,           # 代码（推理能力）
    "books": 0.10,          # 书籍（叙事能力）
    "academic": 0.08,       # 论文（专业知识）
    "conversational": 0.05, # 对话（交互能力）
    "math": 0.02,           # 数学（推理能力）
}
```

**为什么必要**：单一数据源会导致模型能力偏科。

---

### 6.3 课程学习 (Curriculum Learning)【可选】

**原理**：按难度排序数据，从简单到难训练。

```
阶段1（前10%步数）: 短文本（<512 token）、简单语法
阶段2（10%-30%）: 中等文本（512-2048）、标准内容
阶段3（30%-70%）: 长文本（2048-4096）、复杂推理
阶段4（70%-100%）: 超长文本（>4096）、专业领域
```

**作用**：提升训练稳定性，加速收敛。

---

### 6.4 打包 (Packing)【可选】

**原理**：把多个短文本拼接成一个长序列，减少 padding 浪费。

```
普通方式: [文本A(100)] + [padding(3900)] → 利用率 2.5%
打包方式: [文本A(100)|EOS|文本B(200)|EOS|文本C(300)...] → 利用率 95%
```

**注意**：打包时需要特殊 attention mask，防止文本之间互相看到。

---

### 6.5 上采样/下采样 (Up/Down-sampling)【可选】

**原理**：调整某些数据的采样频率。

```python
# 高质量数据多采样（上采样）
high_quality_weight = 3.0   # 采样概率 ×3

# 低质量数据少采样（下采样）
low_quality_weight = 0.3    # 采样概率 ×0.3
```

**作用**：让模型多学习高质量内容，少学习低质量内容。

---

## 七、训练技巧与优化

---

### 7.1 学习率调度 (Learning Rate Scheduling)【必要】

**常用策略**：

| 策略 | 公式 | 适用阶段 |
|------|------|---------|
| **Warmup + Cosine Decay** | 先线性增长到峰值，再余弦衰减到最小值 | 预训练 |
| **Warmup + Constant** | 先 warmup，然后保持不变 | 微调 |
| **Warmup + Linear Decay** | 先 warmup，然后线性衰减到 0 | 短训练 |

```python
# Warmup + Cosine Decay
lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * step / total_steps))
```

---

### 7.2 梯度累积 (Gradient Accumulation)【必要】

**原理**：多次 forward 累积梯度，一次 backward，模拟大 batch。

```python
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**为什么必要**：GPU 显存有限，gradient accumulation 让小 GPU 也能训练大模型。

---

### 7.3 混合精度训练 (Mixed Precision)【必要】

**原理**：大部分计算用 fp16/bf16，关键部分（loss、norm）保留 fp32。

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast(dtype=torch.bfloat16):
    logits = model(input_ids)
    loss = F.cross_entropy(logits, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**为什么必要**：速度提升 2-3 倍，显存减少一半。

---

### 7.4 梯度裁剪 (Gradient Clipping)【必要】

**原理**：限制梯度大小，防止梯度爆炸。

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**为什么必要**：Transformer 训练不稳定，不裁剪容易 loss 突增。

---

### 7.5 Flash Attention【可选但强烈推荐】

**原理**：用 GPU 内存层次优化 attention 计算，减少 HBM 读写。

```python
# 替换标准 attention
from flash_attn import flash_attn_func

# 标准: O(N^2) 内存，频繁 HBM 读写
# Flash: O(N) 内存，分块计算，少 5-10 倍内存
```

**效果**：训练速度提升 2-4 倍，支持更长序列。

---

### 7.6 模型并行 (Model Parallelism)【可选】

**类型**：

| 类型 | 分割方式 | 适用场景 |
|------|---------|---------|
| **Tensor Parallelism** | 每层切分到多个 GPU | 单节点多卡 |
| **Pipeline Parallelism** | 不同层放到不同 GPU | 多节点 |
| **Sequence Parallelism** | 序列维度切分 | 超长序列 |
| **ZeRO** | 优化器状态/梯度/参数分片 | 显存优化 |

---

## 八、推荐组合

### 8.1 通用 LLM 训练流程（推荐）

```
【阶段1: 预训练】                【必要】
├── 因果语言模型 (CLM)
├── 数据清洗 + 数据混合
├── 学习率: Warmup + Cosine Decay
├── 混合精度 (bf16)
├── Flash Attention
└── 数据量: 1-3T tokens

【阶段2: 指令微调 (SFT)】         【必要】
├── 多轮对话格式
├── Loss mask: 只算 assistant 部分
├── 学习率: 2e-5，3 epochs
├── 数据量: 100K-1M 样本
└── 包含: 通用指令 + 代码 + 数学 + 安全

【阶段3: 对齐 (Alignment)】        【必要】
├── 推荐: DPO（简单稳定）
├── 或: RLHF/PPO（效果上限高）
├── 学习率: 1e-6
└── 数据量: 10K-100K 偏好对
```

### 8.2 各阶段"必要 vs 可选"总结

| 方法 | 预训练 | SFT | 对齐 | 说明 |
|------|--------|-----|------|------|
| **因果语言模型** | ✅ 必要 | — | — | 所有 decoder 的基础 |
| **数据清洗** | ✅ 必要 | ✅ 必要 | ✅ 必要 | Garbage in, garbage out |
| **数据混合** | ✅ 必要 | ✅ 必要 | — | 防止能力偏科 |
| **多轮对话 SFT** | — | ✅ 必要 | — | 对话能力必备 |
| **DPO/RLHF** | — | — | ✅ 必要 | 对齐人类偏好 |
| **FIM/Span** | 🔄 可选 | — | — | 代码模型建议加 |
| **CoT 数据** | — | 🔄 可选 | — | 推理任务建议加 |
| **工具使用** | — | 🔄 可选 | — | Agent 能力必备 |
| **Flash Attention** | 🔄 可选 | 🔄 可选 | 🔄 可选 | 强烈推荐 |
| **课程学习** | 🔄 可选 | — | — | 训练不稳定时加 |
| **打包** | 🔄 可选 | — | — | 短文本多时加 |
| **全词掩码** | 🔄 可选 | — | — | 中文模型建议加 |

---

## 总结

```
┌─────────────────────────────────────────────┐
│              LLM 训练 = 数据 + 算法 + 工程      │
├─────────────────────────────────────────────┤
│                                             │
│  【必须做的】                                │
│  1. 预训练: CLM + 数据清洗 + 数据混合        │
│  2. SFT: 多轮对话 + 只算 output loss        │
│  3. 对齐: DPO 或 RLHF                       │
│  4. 工程: 混合精度 + 梯度裁剪 + LR 调度      │
│                                             │
│  【强烈建议做的】                            │
│  5. Flash Attention（速度 ×2-4）            │
│  6. CoT 数据（推理能力提升）                │
│  7. 代码数据（编程能力提升）                │
│                                             │
│  【根据场景选择】                            │
│  8. FIM/Span（代码模型）                    │
│  9. 工具使用（Agent 模型）                  │
│  10. 长上下文扩展（长文本场景）              │
│  11. 课程学习（训练不稳定时）                │
│                                             │
└─────────────────────────────────────────────┘
```
