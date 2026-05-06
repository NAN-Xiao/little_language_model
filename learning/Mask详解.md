# Causal Mask 详解 —— 从训练到推理

## 一句话理解

Causal Mask 是一个**下三角矩阵**，让模型每个位置只能看自己和前面的内容，不能偷看后面的字。训练时一次性构造，推理时动态变化。

---

## 一、训练时的 Mask

### 1.1 输入示例

```
文本:   [BOS]  猫    抓    老    鼠   [EOS]
id:      1     45    23    67    12     2
```

输入张量: `[1, 45, 23, 67, 12, 2]`，形状 `(1, 6)`

### 1.2 Mask 怎么构造

```python
causal_mask = torch.tril(torch.ones(6, 6, dtype=torch.bool))
```

**这个函数分三步执行：**

```python
# Step 1: torch.ones(6, 6) —— 创建 6×6 全1矩阵
# [[1, 1, 1, 1, 1, 1],
#  [1, 1, 1, 1, 1, 1],
#  [1, 1, 1, 1, 1, 1],
#  [1, 1, 1, 1, 1, 1],
#  [1, 1, 1, 1, 1, 1],
#  [1, 1, 1, 1, 1, 1]]

# Step 2: dtype=torch.bool —— 类型改成布尔 (1=True, 0=False)
# 目前全是 True

# Step 3: torch.tril —— 取下三角 (Triangular Lower)
#   保留对角线及下方，上方变成 0
# [[1, 0, 0, 0, 0, 0],
#  [1, 1, 0, 0, 0, 0],
#  [1, 1, 1, 0, 0, 0],
#  [1, 1, 1, 1, 0, 0],
#  [1, 1, 1, 1, 1, 0],
#  [1, 1, 1, 1, 1, 1]]
```

```
        BOS  猫   抓   老   鼠  EOS
BOS   [ 1,   0,   0,   0,   0,   0 ]   ← BOS 只看自己
猫    [ 1,   1,   0,   0,   0,   0 ]   ← 猫 看 BOS、猫
抓    [ 1,   1,   1,   0,   0,   0 ]   ← 抓 看 BOS、猫、抓
老    [ 1,   1,   1,   1,   0,   0 ]   ← 老 看前4个
鼠    [ 1,   1,   1,   1,   1,   0 ]   ← 鼠 看前5个
EOS   [ 1,   1,   1,   1,   1,   1 ]   ← EOS 看全部
```

### 1.3 每个位置在预测什么

| 位置 | 能看到 | 要预测 | 说明 |
|------|--------|--------|------|
| BOS | BOS | **猫** | 看到 BOS，猜下一个字是"猫" |
| 猫 | BOS, 猫 | **抓** | 看到"猫"，猜下一个字是"抓" |
| 抓 | BOS, 猫, 抓 | **老** | 看到"猫抓"，猜下一个字是"老" |
| 老 | BOS, 猫, 抓, 老 | **鼠** | 看到"猫抓老"，猜下一个字是"鼠" |
| 鼠 | BOS, 猫, 抓, 老, 鼠 | **EOS** | 看到"猫抓老鼠"，猜句子结束 |
| EOS | 全部 | **(不管)** | 已经结束，不计算 loss |

### 1.4 Loss 计算

```python
# 模型输出: (batch=1, seq=6, vocab=68)
logits = model(input_ids)   # (1, 6, 68)

# 目标: 每个位置的正确答案是"下一个字"
targets = [45, 23, 67, 12, 2, -100]
#            ↑   ↑   ↑   ↑   ↑   ↑
#           猫  抓  老  鼠  EOS  忽略

loss = F.cross_entropy(
    logits.view(-1, 68),
    torch.tensor(targets),
    ignore_index=-100,   # -100 的位置不算 loss
)
```

**关键点**：
- 6 个位置同时计算，但 mask 保证每个位置只能看前面的内容
- Loss 只算前 5 个位置（预测下一个字），EOS 后面不计算
- 一次性反向传播，更新所有参数

---

## 二、推理时的 Mask

### 2.1 第1步：处理完整 prompt

和训练完全一样！

```python
prompt = [1, 45, 23]   # [BOS, "猫", "抓"]
mask = make_causal_mask(prompt)
# (1, 3, 3) 下三角
```

模型输出预测下一个字 → **"老"**

### 2.2 第2步起：只输入1个新 token

```python
# 已生成: [BOS, "猫", "抓", "老"]
# 只输入新 token: ["老"]
step_input = generated[:, -1:]   # (1, 1)

# mask 变成全1！
seq_k = 4   # 缓存里有4个历史 token
mask = torch.ones(1, 1, 4, dtype=torch.bool)
# (1, 1, 4) = [[[True, True, True, True]]]
```

**为什么全1？**

```
输入只有 ["老"] 1个token:
  q形状: (1, d)     ← 只有1个query
  K缓存: (4, d)     ← 4个历史key
  
  scores = q @ K^T = (1, d) @ (d, 4) = (1, 4)
  
  scores只有1行: [?, ?, ?, ?]
  
  → 没有"未来行"需要遮！未来还没生成呢！
```

---

## 三、核心对比

| | 训练 / 推理第1步 | 推理后续步 |
|--|-----------------|-----------|
| 输入 | `[t0,t1,t2]` (多个token) | `[t2]` (1个token) |
| q形状 | (seq, d) | (1, d) |
| scores | (seq, seq) — 多行多列 | (1, seq) — 只有1行 |
| mask | 下三角 | 全1 |
| 原因 | 多行矩阵，后面的行不能偷看前面的列 | 只有1行，不存在"后面的行" |

---

## 四、可视化：Mask 作用于 Attention 分数

### 训练时（3个token）

```
Scores (q @ k^T):
        BOS   猫    抓
BOS   [ 5.0,  3.0,  1.0 ]
猫    [ 4.0,  6.0,  2.0 ]
抓    [ 3.0,  5.0,  7.0 ]

Mask (下三角):
        BOS   猫    抓
BOS   [  1 ,   0 ,   0  ]
猫    [  1 ,   1 ,   0  ]
抓    [  1 ,   1 ,   1  ]

Masked Scores:
        BOS   猫    抓
BOS   [ 5.0, -inf, -inf ]   ← "BOS"只能看自己
猫    [ 4.0,  6.0, -inf ]   ← "猫"不能看"抓"
抓    [ 3.0,  5.0,  7.0 ]   ← "抓"看全部

Softmax后 (概率):
        BOS   猫    抓
BOS   [1.00, 0.00, 0.00]
猫    [0.12, 0.88, 0.00]   ← "猫"的注意力: 12%看BOS, 88%看自己
抓    [0.02, 0.10, 0.88]   ← "抓"的注意力: 2%看BOS, 10%看猫, 88%看自己
```

### 推理后续步（只输入"抓"）

```
Scores:
        BOS   猫    抓
抓    [ 3.0,  5.0,  7.0 ]   ← 只有1行！

Mask (全1):
        BOS   猫    抓
抓    [  1 ,   1 ,   1  ]

Softmax后:
        BOS   猫    抓
抓    [0.02, 0.10, 0.88]   ← 和训练时"抓"那行完全一样！
```

---

## 五、关键代码位置

- **Mask 构造**: `model/transformer.py:104` (`make_causal_mask`)
- **Mask 使用**: `model/attention.py:228` (`scores.masked_fill`)
- **推理 mask**: `model/transformer.py:357` (`torch.ones(batch_size, 1, seq_k)`)
- **维度调整**: `model/attention.py:448` (`mask.unsqueeze(1)`)

---

## 六、一句话总结

> 训练时：多个 token 一起算，下三角 mask 防止偷看未来。
> 推理时：只算1个新 token，全1 mask 就行——未来还不存在，偷看不了。

---

## 附录：从文本到 Loss 的完整训练流程

### Step 0: 原始文本

```
"今天天气真好"
```

### Step 1: Tokenizer 分词

```python
token_ids = tokenizer.encode("今天天气真好")
# 假设分词结果: [45, 23, 67, 12, 8]  (5个token)
```

### Step 2: 数据集切块

```python
seq_len = 4  # 模型最大序列长度

# 切成 seq_len+1 = 5 的块
block = [45, 23, 67, 12, 8]  # 5个token

input_ids = block[:-1]   # [45, 23, 67, 12]  ← 前4个
labels    = block[1:]    # [23, 67, 12, 8]   ← 后4个（向右移1位）
```

**为什么这样切？**

```
input_ids:  [45,  23,  67,  12]
            ↓    ↓    ↓    ↓
labels:     [23,  67,  12,  8 ]
            
位置0: 看到45 → 预测23（"今"→"天"）
位置1: 看到23 → 预测67（"天"→"气"）
位置2: 看到67 → 预测12（"气"→"真"）
位置3: 看到12 → 预测8  （"真"→"好"）
```

### Step 3: 打包成 batch

```python
# 一个 batch 有4个样本
batch_input_ids = torch.tensor([
    [45, 23, 67, 12],    # 样本0
    [3,  12, 45, 7],     # 样本1
    [8,  3,  12, 0],     # 样本2（最后一个0是pad）
    [1,  45, 23, 67],    # 样本3
])  # 形状: (4, 4)

batch_labels = torch.tensor([
    [23, 67, 12, 8],     # 样本0
    [12, 45, 7,  2],     # 样本1
    [3,  12, 0,  -100],  # 样本2（-100表示不算loss）
    [45, 23, 67, 12],    # 样本3
])  # 形状: (4, 4)
```

### Step 4: 模型前向传播

```python
logits = model(batch_input_ids)  # (4, 4, 68)
# 每个位置对每个词的预测分数
```

内部流程：

```
input_ids: (4, 4)
  → Embedding:       (4, 4, 768)
  → + 位置编码:      (4, 4, 768)
  → + causal mask:   (1, 1, 4, 4)
  → Decoder × 10:    (4, 4, 768)
  → Output projection: (4, 4, 68)  ← logits!
```

### Step 5: 计算 Loss

```python
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# logits reshape: (4, 4, 68) → (16, 68)  [batch*seq, vocab]
# labels reshape: (4, 4)     → (16,)      [batch*seq]

loss = criterion(
    logits.reshape(-1, logits.size(-1)),   # (16, 68)
    batch_labels.reshape(-1),               # (16,)
)
```

**CrossEntropyLoss 内部做了什么？**

对每个位置，取正确标签对应的 logits 分数，算 softmax，取负对数：

```
位置0 (样本0): logits=[2.3, -1.5, 3.1, ...], 正确标签=23
  → softmax → 给23的概率=0.15
  → loss = -log(0.15) = 1.90

位置1 (样本0): logits=[1.2, 0.5, -0.3, ...], 正确标签=67
  → softmax → 给67的概率=0.42
  → loss = -log(0.42) = 0.87

... 所有位置算完取平均
```

### Step 6: 反向传播更新参数

```python
loss.backward()   # 计算每个参数的梯度
optimizer.step()  # 用梯度更新参数
```

### 一句话总结

文本切成块 → `input_ids` 是前N个token，`labels` 是后N个token（向右移1位）→ 模型输出 `(batch, seq, vocab)` 的 logits → CrossEntropyLoss 比较预测和正确答案 → 反向传播更新参数。

---

## 附录二：长文本怎么训练（100个字怎么办）

### 问题

如果一篇文章有 100 个字，而模型的 `seq_len=4`，怎么训练？

### 答案：切成多个块

代码在 `data/dataset.py:219`：

```python
block_len = seq_len + 1          # seq_len=4 → block_len=5
n_blocks = len(token_ids) // block_len  # 100 // 5 = 20 个块
```

**100 个字的文本，seq_len=4 时切成 20 个样本：**

```
原始文本: "今天天气真好啊..." (100个token)

切成 20 个块，每块 5 个token:
  block 0:  [t0,  t1,  t2,  t3,  t4]   → input=[t0,t1,t2,t3],  label=[t1,t2,t3,t4]
  block 1:  [t5,  t6,  t7,  t8,  t9]   → input=[t5,t6,t7,t8],  label=[t6,t7,t8,t9]
  block 2:  [t10, t11, t12, t13, t14]  → ...
  ...
  block 19: [t95, t96, t97, t98, t99]  → input=[t95,t96,t97,t98], label=[t96,t97,t98,t99]
```

**如果 seq_len=256（大模型常用）：**

```
100 < 256，不够一个完整块！
→ 直接作为一个样本
→ input = 前99个token, label = 后99个token
→ 末尾 pad 到 256
```

### 块与块之间有关系吗？

**没有。** 每个块是独立样本，模型不知道 block 1 的 t5 前面是 block 0 的 t4。

这是 GPT 风格预训练的标准做法：
- **优点**：简单高效，可以并行处理海量文本
- **缺点**：长文本被切碎，块之间丢失联系

### 如果要保留长距离联系怎么办？

| 方法 | 说明 |
|------|------|
| **更长的 seq_len** | 如 4k, 8k, 128k，直接装下整篇文章 |
| **滑动窗口** | 块之间重叠一部分 token |
| **Ring Attention** | 把序列环起来，支持超长上下文 |

### 一句话总结

> 长文本按 `seq_len` 切成多个块，每块是独立样本。seq_len 越大，越能保留上下文联系。
