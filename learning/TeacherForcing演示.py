"""
演示: Teacher Forcing —— 训练时为什么能一次算所有位置
运行: python learning/teacher_forcing_demo.py
"""

import torch
import torch.nn.functional as F

torch.set_printoptions(precision=3, sci_mode=False)

print("=" * 65)
print("核心问题: 位置2怎么能'看到'位置0和1的内容?")
print("=" * 65)

# 简化: 3个token, 每个4维
seq = 3
d = 4

# 输入: [BOS, 今, 天]
# 假设embedding后:
x = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],   # e_BOS
    [0.0, 1.0, 0.0, 0.0],   # e_今
    [0.0, 0.0, 1.0, 0.0],   # e_天
])
print(f"\n输入 x (3个token的向量):\n{x}")
print(f"shape: {tuple(x.shape)} = (seq={seq}, d={d})")

# 简化的W_q, W_k, W_v (用单位矩阵简化)
W_q = torch.eye(d)
W_k = torch.eye(d)
W_v = torch.eye(d)

# Step 1: 算 q, k, v (所有位置同时算!)
q = x @ W_q.T   # (3, 4)
k = x @ W_k.T   # (3, 4)
v = x @ W_v.T   # (3, 4)

print(f"\nq = x @ W_q (所有位置同时算):\n{q}")
print(f"k = x @ W_k (所有位置同时算):\n{k}")
print(f"v = x @ W_v (所有位置同时算):\n{v}")

# Step 2: 算 scores (3×3矩阵, 一步全算!)
scores = q @ k.T   # (3, 3)
print(f"\nscores = q @ k.T (3×3矩阵, 一步全算):\n{scores}")

# Step 3: causal mask
mask = torch.tril(torch.ones(seq, seq))
scores_masked = scores.masked_fill(mask == 0, float("-inf"))
print(f"\nmask (下三角):\n{mask}")
print(f"masked scores (右上角变-inf):\n{scores_masked}")

# Step 4: softmax
attn = F.softmax(scores_masked, dim=-1)
print(f"\nsoftmax后 (注意力权重):\n{attn}")

# Step 5: attn @ v (输出!)
out = attn @ v   # (3, 3) @ (3, 4) = (3, 4)
print(f"\nout = attn @ v (输出):\n{out}")

# ============================================
# 关键解释
# ============================================
print("\n" + "=" * 65)
print("【关键】看每一行的输出是怎么组成的")
print("=" * 65)

print(f"\nout[0] = {out[0].tolist()}")
print(f"  = {attn[0,0]:.3f} * v_BOS + {attn[0,1]:.3f} * v_今 + {attn[0,2]:.3f} * v_天")
print(f"  = {attn[0,0]:.3f} * [1,0,0,0] + 0 * [0,1,0,0] + 0 * [0,0,1,0]")
print(f"  ← 只混合了 v_BOS!")

print(f"\nout[1] = {out[1].tolist()}")
print(f"  = {attn[1,0]:.3f} * v_BOS + {attn[1,1]:.3f} * v_今 + {attn[1,2]:.3f} * v_天")
print(f"  ← 混合了 v_BOS 和 v_今!")

print(f"\nout[2] = {out[2].tolist()}")
print(f"  = {attn[2,0]:.3f} * v_BOS + {attn[2,1]:.3f} * v_今 + {attn[2,2]:.3f} * v_天")
print(f"  ← 混合了 v_BOS, v_今, v_天!")

print("\n" + "=" * 65)
print("【核心洞察】")
print("=" * 65)
print("""
out[1] 包含了 v_BOS 和 v_今，但它需要等 out[0] 算完吗?

不需要! 因为:
  - v_BOS 是从 e_BOS 直接算出来的 (x[0] @ W_v)
  - v_今 是从 e_今 直接算出来的 (x[1] @ W_v)
  - 它们都是"已知量"，一开始就在 x 里了!

out[1] = attn[1,0]*v_BOS + attn[1,1]*v_今
         ↑↑↑↑↑↑   ↑↑↑↑      ↑↑↑↑↑↑   ↑↑↑↑
         已知     已知      已知     已知

公式里的每个数都是已知的，可以直接算!
不需要先算 out[0]!

这就是为什么 3 个位置能同时算:
  - v_0, v_1, v_2 同时从 x 算出来
  - scores 矩阵同时算出来
  - softmax 同时做
  - out[0], out[1], out[2] 同时算出来

不是"先算位置0，再用结果算位置1"!
是"3个位置独立计算，同时出结果"!
""")
