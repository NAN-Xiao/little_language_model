"""
演示: 为什么矩阵乘法能"一次处理所有行"
运行: python learning/why_parallel_demo.py
"""

import torch

torch.set_printoptions(precision=2, sci_mode=False)

print("=" * 60)
print("核心问题: 4个位置怎么同时算?")
print("=" * 60)

# 简化: 2个位置, 每个3维, 投影到5维
seq = 2
d_model = 3
vocab = 5

# 输入: 2个位置的向量
x = torch.tensor([
    [1.0, 2.0, 3.0],   # 位置0的向量
    [4.0, 5.0, 6.0],   # 位置1的向量
])
print(f"\n输入 x:\n{x}")
print(f"shape: {tuple(x.shape)} = (seq={seq}, d_model={d_model})")

# 投影矩阵 W: (3, 5)
W = torch.tensor([
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.5, 0.4, 0.3, 0.2, 0.1],
    [0.2, 0.2, 0.2, 0.2, 0.2],
])
print(f"\n投影矩阵 W:\n{W}")
print(f"shape: {tuple(W.shape)} = (d_model={d_model}, vocab={vocab})")

# ============================================
# 方法1: for循环逐个算（你理解的方式）
# ============================================
print("\n" + "=" * 60)
print("【方法1】for循环逐个位置算（你的直觉）")
print("=" * 60)

result_loop = []
for i in range(seq):
    # x[i]: (3,) @ W: (3, 5) = (5,)
    y_i = x[i] @ W
    result_loop.append(y_i)
    print(f"  位置{i}: x[{i}] @ W = {y_i.tolist()}")

result_loop = torch.stack(result_loop)
print(f"\nfor循环结果:\n{result_loop}")

# ============================================
# 方法2: 矩阵乘法一次性算（Transformer的方式）
# ============================================
print("\n" + "=" * 60)
print("【方法2】矩阵乘法一步全算（Transformer的方式）")
print("=" * 60)

result_matmul = x @ W
print(f"x @ W =\n{result_matmul}")

# ============================================
# 对比
# ============================================
print("\n" + "=" * 60)
print("【对比】两种方法结果一样吗?")
print("=" * 60)

print(f"for循环结果:\n{result_loop}")
print(f"矩阵乘法结果:\n{result_matmul}")
print(f"完全一样: {torch.allclose(result_loop, result_matmul)}")

# ============================================
# 核心洞察
# ============================================
print("\n" + "=" * 60)
print("【核心洞察】")
print("=" * 60)
print("""
方法1(for循环):
  位置0: x[0] @ W → 结果0
  位置1: x[1] @ W → 结果1
  ... 串行执行, 一个一个算

方法2(矩阵乘法):
  x @ W
    其中 x[0] @ W → 结果0  (和其他行无关!)
         x[1] @ W → 结果1  (和其他行无关!)
  ... 并行执行, 同时算!

关键: 输出第i行只依赖输入第i行, 不依赖其他行!
  result[0] = x[0] @ W  ← 只用x[0], 不用x[1]
  result[1] = x[1] @ W  ← 只用x[1], 不用x[0]

因为互相独立, 所以可以同时算!
这就是 GPU 并行计算的基础!
""")

# ============================================
# 扩展到Transformer完整流程
# ============================================
print("=" * 60)
print("【扩展到Transformer】")
print("=" * 60)

print("""
输入 x: (batch=1, seq=4, d_model=8)

每一层的操作:
  Attention: q = x @ W_q   → (1, 4, 8)
             k = x @ W_k   → (1, 4, 8)
             scores = q @ k.T → (1, 4, 4)
  FFN:       ffn = x @ W_ffn → (1, 4, 8)
  Output:    logits = ffn @ W_proj → (1, 4, vocab)

规律: 每一行(seq维度)的计算都和其他行无关!
      所以4行同时算, 128000行也同时算!

不是"循环4次", 是"一次算4行"!
""")
