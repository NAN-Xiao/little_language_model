# GQA (Grouped-Query Attention) — 从 MHA 到 GQA 的演进

## 一、为什么要演进？

### 核心矛盾：效果好 ↔ 推理快

Transformer 推理时的瓶颈不是计算，而是 **KV Cache 显存**。

每生成一个新 token，都要缓存前面所有 token 的 K 和 V：

```
生成第 100 个 token 时：
  需要 K[0..99], V[0..99]  — 100 个 token 的历史
  显存占用 = 2 × n_heads × seq_len × d_k
```

| 配置 | KV Cache 每层 | 10 层总显存 (seq=1024) |
|------|--------------|------------------------|
| MHA (n_heads=12) | 2 × 12 × 1024 × 64 = 1.57 MB | 15.7 MB |
| GQA (n_kv_heads=3) | 2 × 3 × 1024 × 64 = 0.39 MB | 3.9 MB |
| MQA (n_kv_heads=1) | 2 × 1 × 1024 × 64 = 0.13 MB | 1.3 MB |

**GQA 把显存降到 1/4，推理快 30%，效果几乎不下降。**

---

## 二、三种 Attention 对比

### MHA (Multi-Head Attention) — 标准多头

```python
n_heads=12, d_model=768, d_k=64

Q: 12 个投影 → (B, 12, seq, 64)  ← 每个头独立的查询
K: 12 个投影 → (B, 12, seq, 64)  ← 每个头独立的键
V: 12 个投影 → (B, 12, seq, 64)  ← 每个头独立的值

每个头独立做 attention → 12 个结果拼接 → 投影回 768 维
```

**优点**：每个头有自己的 K/V，表达能力最强  
**缺点**：KV Cache 太大，长文本推理时显存爆炸

---

### GQA (Grouped-Query Attention) — 推荐平衡方案

```python
n_heads=12, n_kv_heads=3, d_k=64

Q: 12 个投影 → (B, 12, seq, 64)      ← 不变，每个头独立的查询策略
K: 3 个投影  → (B, 3, seq, 64)       ← 变少了！
V: 3 个投影  → (B, 3, seq, 64)       ← 变少了！

关键操作: K/V 在 heads 维度重复 4 次
  K: (B, 3, seq, 64) → repeat → (B, 12, seq, 64)
  这样 Q 的 12 个头都能和 K 做运算
```

**为什么效果没崩？**

- Q 投影还是独立的（12 个），每个头仍能学不同的"查询策略"
- K/V 被共享，相当于多个 Query 查同一个"知识库"
- LLaMA-2 论文：效果接近 MHA，推理快 30%

**KV Cache**：`2 × n_kv_heads × seq × d_k` = 2 × 3 × seq × 64 = **原来的 1/4**

---

### MQA (Multi-Query Attention) — 极端压缩

```python
n_heads=12, n_kv_heads=1

Q: 12 个投影
K: 1 个投影（所有 Q 共享）
V: 1 个投影（所有 Q 共享）
```

**KV Cache**：`2 × 1 × seq × d_k` = **原来的 1/12**

**代价**：所有头被迫共享同一个"记忆库"，表达能力受限。PaLM 用过，但后续模型大多转向 GQA。

---

## 三、可视化对比

### MHA — 12 个独立专家

```
头 0:  Q0 ──┐
头 1:  Q1 ──┤
头 2:  Q2 ──┤
...         ├─→ 各自独立的 K/V ──→ Attention ──→ 拼接
头 11: Q11 ─┘

KV Cache: 12 套 K/V
```

### GQA — 12 个 Query 查 3 个知识库

```
头 0-3:   Q0,Q1,Q2,Q3 ──┐
头 4-7:   Q4,Q5,Q6,Q7 ──┼─→ 每组共享 1 套 K/V ──→ Attention
头 8-11:  Q8,Q9,Q10,Q11─┘

         K0,V0    K1,V1    K2,V2
         ↑↑↑↑     ↑↑↑↑     ↑↑↑↑
         4个头    4个头    4个头

KV Cache: 3 套 K/V（每套被 4 个头共享）
```

### MQA — 12 个 Query 查 1 个知识库

```
头 0-11:  Q0..Q11 ──→ 全部共享 1 套 K/V ──→ Attention

KV Cache: 1 套 K/V
```

---

## 四、代码实现思路

### 核心改动：K/V 投影和重复

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None, ...):
        # n_kv_heads=None → 标准 MHA
        # n_kv_heads=3   → GQA (n_heads 必须是 n_kv_heads 的倍数)
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads  # 默认等于 n_heads (MHA)
        self.n_rep = n_heads // self.n_kv_heads   # 重复次数

        # Q 投影: 保持 n_heads 个
        self.w_q = nn.Linear(d_model, n_heads * d_k)

        # K/V 投影: 只有 n_kv_heads 个
        self.w_k = nn.Linear(d_model, self.n_kv_heads * d_k)
        self.w_v = nn.Linear(d_model, self.n_kv_heads * d_k)

    def forward(self, q, k, v, ...):
        # Q: (B, seq, d_model) → (B, n_heads, seq, d_k)
        q = self.w_q(q).view(B, seq, self.n_heads, d_k).transpose(1, 2)

        # K/V: (B, seq, d_model) → (B, n_kv_heads, seq, d_k)
        k = self.w_k(k).view(B, seq, self.n_kv_heads, d_k).transpose(1, 2)
        v = self.w_v(v).view(B, seq, self.n_kv_heads, d_k).transpose(1, 2)

        # GQA 关键: 把 K/V 重复 n_rep 次，让 Q 的每个头都能匹配
        if self.n_rep > 1:
            # (B, 3, seq, 64) → (B, 12, seq, 64)
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # 现在 Q/K/V 都是 (B, 12, seq, 64)，标准 attention 计算
        attn = softmax(q @ k^T / sqrt(d_k)) @ v
```

### repeat_interleave 详解

```python
k = (B, 3, seq, 64)
      ↓ repeat_interleave(n_rep=4, dim=1)
k = (B, 12, seq, 64)

具体:
  输入: [K0, K1, K2]  ← 3 个头
  输出: [K0,K0,K0,K0, K1,K1,K1,K1, K2,K2,K2,K2]  ← 每个重复 4 次

这样 Q0..Q3 都查 K0，Q4..Q7 都查 K1，Q8..Q11 都查 K2
```

---

## 五、配置建议

| 模型规模 | n_heads | n_kv_heads | 类型 | KV Cache |
|---------|---------|-----------|------|---------|
| 小模型 (< 1B) | 8 | 8 | MHA | 100% |
| 中模型 (1-7B) | 12 | 3-4 | GQA | 25-33% |
| 大模型 (7B+) | 16-32 | 4-8 | GQA | 12-25% |
| 超大模型 (70B+) | 64 | 8 | GQA | 12.5% |

**推荐**：本项目 `n_heads=12`，`n_kv_heads=3` 是甜点。

---

## 六、一句话总结

| 类型 | KV Cache | 效果 | 推荐 |
|------|----------|------|------|
| MHA | 100% | 最好 | 小模型 |
| **GQA** | **25%** | **接近 MHA** | **⭐ 强烈推荐** |
| MQA | 8% | 明显下降 | 不推荐 |
