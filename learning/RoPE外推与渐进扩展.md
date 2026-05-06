# RoPE 外推与渐进扩展详解

> 本文档说明：为什么基础模型只能处理 4K/8K 长度，以及如何通过 RoPE 外推和渐进扩展让模型支持 128K+ 长上下文。

---

## 目录

- [一、为什么模型有长度限制](#一为什么模型有长度限制)
- [二、RoPE 是什么](#二rope-是什么)
- [三、RoPE 外推（Extrapolation）](#三rope-外推extrapolation)
  - [3.1 问题：直接外推会崩溃](#31-问题直接外推会崩溃)
  - [3.2 位置插值（Positional Interpolation, PI）](#32-位置插值positional-interpolation-pi)
  - [3.3 NTK-aware 扩展](#33-ntk-aware-扩展)
  - [3.4 YaRN（最优方案）](#34-yarn最优方案)
- [四、渐进扩展（Progressive Extension）](#四渐进扩展progressive-extension)
- [五、代码实现](#五代码实现)
- [六、各种方法对比](#六各种方法对比)

---

## 一、为什么模型有长度限制

Transformer 模型在预训练时，**所有训练样本的长度都不超过一个上限**（比如 4096 token）。模型"学会"了如何处理 0~4095 位置的信息，但**从没见过 4096 以上的位置**。

```
预训练时看到的：
位置 0, 1, 2, ... 4095

推理时用户给的：
位置 0, 1, 2, ... 4095, 4096, 4097, ... 100000
                             ↑ 模型从没学过这些位置！
```

**结果**：超出训练长度的位置，模型完全不知道如何处理，输出变成乱码。

---

## 二、RoPE 是什么

RoPE（Rotary Position Embedding，旋转位置编码）是目前主流的位置编码方式（LLaMA、Qwen、ChatGLM 都用）。

### 2.1 核心思想

不给每个位置一个"位置向量"（像正弦位置编码那样），而是**旋转 Query 和 Key 向量**。

```python
# 标准 attention: q @ k.T
# RoPE attention: rotate(q, pos_i) @ rotate(k, pos_j).T

# rotate(x, pos) = 把向量 x 在二维平面上旋转 pos * angle 度
# angle 由频率决定：angle = 1 / (base ^ (2i/d))
```

### 2.2 为什么用旋转？

**关键性质**：旋转后的内积只和**相对位置**有关。

```
rotate(q, pos_i) @ rotate(k, pos_j).T = f(q, k, pos_i - pos_j)
                                          ↑ 只和相对距离有关！
```

这意味着：模型学到的是"距离我 3 个位置的 token 有什么特征"，而不是"位置 100 的 token 有什么特征"。

### 2.3 RoPE 的频率

RoPE 用一组**频率**控制旋转角度：

```python
# 频率公式
freq = 1.0 / (base ^ (dim_index / head_dim))

# 默认 base = 10000
# dim_index = 0, 1, 2, ... head_dim-1
# head_dim = 64 (每个 attention head 的维度)

# 高频（dim_index 小）: 旋转快，感知短距离变化
# 低频（dim_index 大）: 旋转慢，感知长距离变化
```

**关键洞察**：低频维度负责"长距离关系"。如果低频旋转得太快，模型就分不清"远"和"更远"的区别。

---

## 三、RoPE 外推（Extrapolation）

### 3.1 问题：直接外推会崩溃

假设模型训练时最长 4096，现在要给 8192 长度的输入：

```python
# 位置 4096 的频率
freq = 1.0 / (10000 ^ (dim_index / 64))

# 位置 4096 的旋转角度
angle_4096 = 4096 * freq

# 位置 8192 的旋转角度
angle_8192 = 8192 * freq = 2 * angle_4096
```

**问题**：位置 8192 的旋转角度是 4096 的 2 倍。对于高频维度，这个角度可能已经超过 360° 很多圈了，模型完全"看不懂"这个位置。

```
训练时：位置 0~4095 的角度范围是 0° ~ 360°（刚好一圈）
外推时：位置 4096~8191 的角度范围是 360° ~ 720°（第二圈）

模型只学过"第一圈"，第二圈对它来说是全新的！
```

### 3.2 位置插值（Positional Interpolation, PI）

**核心思想**：把长位置"压缩"到模型学过的范围内。

```python
# 原始：位置 8192 的角度 = 8192 * freq
# PI：  位置 8192 的角度 = (8192 * 0.5) * freq = 4096 * freq
#                              ↑ 缩放因子 = 训练长度 / 目标长度 = 4096/8192 = 0.5

scale = train_length / target_length  # 0.5
angle = pos * scale * freq            # 压缩后的角度
```

**效果**：
- ✅ 位置 8192 的角度 = 原来位置 4096 的角度，模型认识！
- ❌ 但相邻位置的差距变小了（分辨率降低），短距离精度下降

```
原来：位置 0 和 1 的角度差 = 1 * freq
PI后：位置 0 和 1 的角度差 = 0.5 * freq（变小了！）
```

### 3.3 NTK-aware 扩展

**核心思想**：不要所有频率都同样缩放，**高频少缩、低频多缩**。

```python
# NTK-aware 缩放
# 高频（短距离）: 缩放少，保持精度
# 低频（长距离）: 缩放多，扩展范围

def ntk_scaling(freq, scale, dim_index, head_dim):
    """
    非线性缩放：不同维度用不同的缩放因子
    """
    # 基础缩放
    base_scale = scale

    # 高频维度（dim_index 小）：缩放因子接近 1（几乎不缩）
    # 低频维度（dim_index 大）：缩放因子接近 base_scale
    adjusted_scale = (
        base_scale * dim_index / (head_dim - 1) +
        1 * (head_dim - 1 - dim_index) / (head_dim - 1)
    )

    return freq / adjusted_scale
```

**效果**：
- 短距离精度保持（高频不缩）
- 长距离可以扩展（低频多缩）

### 3.4 YaRN（最优方案）

YaRN = Yet another RoPE extensioN，是目前效果最好的 RoPE 扩展方法。

**核心思想**：结合 PI 和 NTK，再加一个**温度系数**修正 attention 分布。

```python
def yarn_scaling(pos, dim_index, head_dim,
                 train_length=4096, target_length=131072,
                 base=10000, beta=32):
    """
    YaRN 扩展
    """
    scale = target_length / train_length  # 32

    # 1. 频率计算（和 RoPE 一样）
    freq = 1.0 / (base ** (dim_index / head_dim))

    # 2. 低频维度（负责长距离）用 NTK 缩放
    # 高频维度（负责短距离）保持原样
    if dim_index < head_dim // 2:
        # 高频：PI 缩放
        freq = freq / scale
    else:
        # 低频：NTK-aware 缩放
        freq = freq / (scale ** (dim_index / head_dim))

    # 3. 温度系数：修正长序列的 attention 分布
    # 长序列中 attention 分数会变小，需要放大
    temperature = 0.1 * math.log(scale) + 1.0

    # 4. 计算旋转角度
    angle = pos * freq

    return angle, temperature
```

**YaRN 的三个改进**：

| 改进 | 作用 |
|------|------|
| **NT-aware 频率缩放** | 高频保持精度，低频扩展范围 |
| **温度系数** | 修正长序列的 attention 分数衰减 |
| **混合策略** | 部分 head 用扩展频率，部分保持原频率 |

---

## 四、渐进扩展（Progressive Extension）

### 4.1 为什么需要渐进？

直接一步从 4K 扩展到 128K 会有很多问题：

1. **训练不稳定**：模型突然面对 32 倍长的序列，loss 会剧烈波动
2. **计算成本高**：128K 的 attention 计算量是 4K 的 1024 倍
3. **效果差**：一步到位扩展的效果不如逐步适应

### 4.2 渐进扩展流程

```
阶段0: 基础模型（训练长度 4096）
    │
    ▼ 加载模型，调整 RoPE 参数
阶段1: 扩展到 8192，训练 500 步
    │
    ▼ 加载上一步模型，调整 RoPE 参数
阶段2: 扩展到 16384，训练 500 步
    │
    ▼ 加载上一步模型，调整 RoPE 参数
阶段3: 扩展到 32768，训练 300 步
    │
    ▼ 加载上一步模型，调整 RoPE 参数
阶段4: 扩展到 65536，训练 200 步
    │
    ▼ 加载上一步模型，调整 RoPE 参数
阶段5: 扩展到 131072，训练 200 步
    │
    ▼
最终模型（支持 128K 上下文）
```

### 4.3 每一步的具体操作

```python
def progressive_extension(model, stages):
    """
    渐进扩展主函数

    stages = [
        {"length": 8192,   "steps": 500, "lr": 1e-5},
        {"length": 16384,  "steps": 500, "lr": 1e-5},
        {"length": 32768,  "steps": 300, "lr": 5e-6},
        {"length": 65536,  "steps": 200, "lr": 5e-6},
        {"length": 131072, "steps": 200, "lr": 2e-6},
    ]
    """
    for stage in stages:
        target_length = stage["length"]

        # 1. 调整 RoPE 参数
        scale = target_length / model.train_length
        model.rope_theta = model.base_rope_theta * (scale ** (2 / model.head_dim))

        # 2. 加载长文本数据
        long_data = load_long_documents(min_length=target_length // 2)

        # 3. 训练
        trainer = Trainer(
            model=model,
            data=long_data,
            max_seq_length=target_length,
            lr=stage["lr"],
            steps=stage["steps"],
        )
        trainer.train()

        # 4. 保存 checkpoint
        save_checkpoint(model, f"checkpoint_length_{target_length}")

        # 更新当前长度
        model.train_length = target_length

    return model
```

### 4.4 关键细节

**学习率递减**：

| 阶段 | 目标长度 | 学习率 | 步数 |
|------|---------|--------|------|
| 1 | 8K | 1e-5 | 500 |
| 2 | 16K | 1e-5 | 500 |
| 3 | 32K | 5e-6 | 300 |
| 4 | 64K | 5e-6 | 200 |
| 5 | 128K | 2e-6 | 200 |

**为什么学习率递减？**
- 早期阶段（8K→16K）：变化相对小，可以用较高学习率
- 后期阶段（64K→128K）：模型已经适应较长序列，只需微调，学习率要低

**数据长度要求**：

```python
# 每个阶段的数据应该比目标长度略短
# 让模型学会"在目标长度内处理信息"
min_length = target_length // 2   # 目标长度的一半
max_length = target_length        # 不超过目标长度
```

---

## 五、代码实现

### 5.1 RoPE 外推的核心代码

```python
import torch
import math

class RotaryPositionEmbedding:
    """RoPE 位置编码 + 外推扩展"""

    def __init__(self, dim, max_seq_len=2048, base=10000):
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        # 预计算频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def extend(self, target_length, method="yarn"):
        """扩展位置编码以支持更长的序列"""
        scale = target_length / self.max_seq_len

        if method == "pi":
            # 位置插值：所有频率统一缩放
            self.inv_freq = self.inv_freq / scale

        elif method == "ntk":
            # NTK-aware：高频少缩、低频多缩
            for i in range(len(self.inv_freq)):
                dim_ratio = i / (len(self.inv_freq) - 1)
                # 高频（i小）：缩放接近 1
                # 低频（i大）：缩放接近 scale
                adjusted_scale = 1 + (scale - 1) * dim_ratio
                self.inv_freq[i] = self.inv_freq[i] / adjusted_scale

        elif method == "yarn":
            # YaRN：最优方案
            for i in range(len(self.inv_freq)):
                if i < len(self.inv_freq) // 2:
                    # 高频：PI 缩放
                    self.inv_freq[i] = self.inv_freq[i] / scale
                else:
                    # 低频：NTK 缩放
                    dim_ratio = i / len(self.inv_freq)
                    self.inv_freq[i] = self.inv_freq[i] / (scale ** dim_ratio)

            # 温度系数
            self.temperature = 0.1 * math.log(scale) + 1.0

        self.max_seq_len = target_length

    def forward(self, x, seq_len):
        """
        x: (batch, seq_len, dim)
        返回旋转后的 x
        """
        # 生成位置索引
        positions = torch.arange(seq_len, device=x.device)

        # 计算角度：angle = pos * freq
        angles = torch.outer(positions, self.inv_freq)  # (seq_len, dim//2)

        # 旋转
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # 应用旋转到 x
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1).flatten(-2)

        return rotated
```

### 5.2 渐进扩展的完整流程

```python
def train_with_progressive_extension(base_model, tokenizer):
    """渐进扩展训练完整流程"""

    stages = [
        # (目标长度, 训练步数, 学习率, batch_size)
        (8192,   500, 1e-5, 8),
        (16384,  500, 1e-5, 4),
        (32768,  300, 5e-6, 2),
        (65536,  200, 5e-6, 1),
        (131072, 200, 2e-6, 1),
    ]

    model = base_model
    current_length = 4096  # 模型原始训练长度

    for target_length, steps, lr, batch_size in stages:
        print(f"\n{'='*60}")
        print(f"阶段: {current_length} → {target_length}")
        print(f"{'='*60}")

        # 1. 扩展 RoPE
        scale = target_length / current_length
        model.rope_theta *= scale ** (2 / model.config.head_dim)

        # 2. 准备长文本数据
        # 数据长度在 [target_length//2, target_length] 之间
        dataset = LongContextDataset(
            min_length=target_length // 2,
            max_length=target_length,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size)

        # 3. 训练
        optimizer = AdamW(model.parameters(), lr=lr)

        for step in range(steps):
            batch = next(iter(dataloader))

            # 前向传播
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = outputs.loss

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            if step % 50 == 0:
                print(f"  Step {step}/{steps}, Loss: {loss.item():.4f}")

        # 4. 保存
        save_path = f"checkpoint_length_{target_length}"
        model.save_pretrained(save_path)
        print(f"  已保存到: {save_path}")

        # 更新当前长度
        current_length = target_length

    return model
```

---

## 六、各种方法对比

| 方法 | 原理 | 短距离精度 | 长距离效果 | 实现复杂度 | 推荐度 |
|------|------|-----------|-----------|-----------|--------|
| **直接外推** | 不做任何修改 | ❌ 极差 | ❌ 不能用 | 零 | ⭐ |
| **位置插值 (PI)** | 所有频率统一缩放 | ⚠️ 下降 | ✅ 可用 | 低 | ⭐⭐⭐ |
| **NTK-aware** | 高频少缩、低频多缩 | ✅ 保持 | ✅ 好 | 中 | ⭐⭐⭐⭐ |
| **YaRN** | NTK + 温度系数 + 混合 | ✅ 好 | ✅ 最好 | 中 | ⭐⭐⭐⭐⭐ |
| **直接训练** | 直接用 128K 数据训练 | ✅ 最好 | ✅ 最好 | 高 | ⭐⭐⭐⭐⭐ |

**推荐方案**：

```
预算充足：直接训练 128K（效果最好，但成本最高）
预算中等：YaRN 外推 + 渐进扩展 8K→128K（性价比最高）
预算有限：NTK-aware 外推 + 微调（快速见效）
```

---

## 总结

```
┌─────────────────────────────────────────────┐
│         RoPE 外推 = 调整旋转频率              │
│         渐进扩展 = 逐步适应长度               │
├─────────────────────────────────────────────┤
│                                             │
│  1. 问题：模型只学过 0~4095 位置              │
│     超出后旋转角度超出范围，模型崩溃          │
│                                             │
│  2. 解法：调整频率（inv_freq）                │
│     - PI：所有频率统一除以一个缩放因子        │
│     - NTK：高频少除、低频多除                 │
│     - YaRN：NTK + 温度修正（最优）            │
│                                             │
│  3. 渐进：不要一步跳到 128K                   │
│     4K → 8K → 16K → 32K → 64K → 128K        │
│     每步训练 200-500 步，逐步适应            │
│                                             │
│  4. 关键参数                                 │
│     rope_theta *= (scale ^ (2/head_dim))     │
│     scale = 目标长度 / 原始长度               │
│                                             │
└─────────────────────────────────────────────┘
```
