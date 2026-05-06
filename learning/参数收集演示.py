"""
model.parameters() 详解 —— Transformer 有哪些参数？
运行: python learning/parameters_demo.py
"""

import torch
import torch.nn as nn

print("=" * 70)
print("model.parameters() 是什么？")
print("=" * 70)
print("""
model.parameters() 返回模型中所有"可训练参数"的迭代器。
每个参数是一个 torch.nn.Parameter 对象，包含:
  - 参数值 (weight 或 bias 的数据)
  - 梯度 (grad，backward后自动填充)
  - requires_grad=True (表示这个参数需要计算梯度并更新)
""")

# ============ 极简模型示例 ============
print("\n" + "=" * 70)
print("【示例1】极简模型: y = w*x + b")
print("=" * 70)

simple_model = nn.Linear(2, 3)  # 输入2维, 输出3维
# 内部参数:
#   weight: (3, 2) -- 3行2列的矩阵
#   bias:   (3,)   -- 3个偏置项

print(f"\nsimple_model 参数列表:")
for name, param in simple_model.named_parameters():
    print(f"  {name}: shape={tuple(param.shape)}, 参数量={param.numel()}")

# weight: (3, 2) = 3行2列 = 6个数
# bias:   (3,)   = 3个数
# 总参数量: 6 + 3 = 9

print(f"\n总参数量: {sum(p.numel() for p in simple_model.parameters())}")

# ============ Transformer 参数拆解 ============
print("\n" + "=" * 70)
print("【示例2】Transformer 参数拆解")
print("=" * 70)

vocab_size = 8
d_model = 4
n_heads = 2
n_layers = 2
seq_len = 6

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Embedding 层
        self.embedding = nn.Embedding(vocab_size, d_model)
        #    参数: weight (vocab_size, d_model) = (8, 4) = 32个参数
        #    含义: 词表里8个词, 每个词对应一个4维向量

        # 2. Attention 层
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        #    每个Linear参数: weight (d_model, d_model) = (4, 4) = 16个参数
        #    4个Linear共: 16 * 4 = 64个参数

        # 3. FFN 层
        d_ff = d_model * 2
        self.W_ff1 = nn.Linear(d_model, d_ff, bias=False)
        self.W_ff2 = nn.Linear(d_ff, d_model, bias=False)
        #    W_ff1: weight (d_model, d_ff) = (4, 8) = 32个参数
        #    W_ff2: weight (d_ff, d_model) = (8, 4) = 32个参数
        #    共: 32 + 32 = 64个参数

        # 4. Output Projection
        self.W_proj = nn.Linear(d_model, vocab_size, bias=False)
        #    参数: weight (d_model, vocab_size) = (4, 8) = 32个参数

    def forward(self, x):
        x = self.embedding(x)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        scores = q @ k.transpose(-2, -1)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        out = self.W_o(out)
        x = x + out
        ffn = torch.relu(self.W_ff1(x))
        ffn = self.W_ff2(ffn)
        x = x + ffn
        logits = self.W_proj(x)
        return logits

model = TinyTransformer()

print(f"\nTransformer 各层参数:")
print(f"{'层名':<20} {'shape':<15} {'参数量':<8}")
print("-" * 50)

total = 0
for name, param in model.named_parameters():
    n = param.numel()
    total += n
    print(f"{name:<20} {str(tuple(param.shape)):<15} {n:<8}")

print("-" * 50)
print(f"{'总参数量':<20} {'':<15} {total:<8}")

# ============ optimizer 怎么管理这些参数 ============
print("\n" + "=" * 70)
print("【优化器怎么管理参数？】")
print("=" * 70)

# 创建优化器时, 把 model.parameters() 传进去
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print(f"\noptimizer.param_groups 中有 {len(optimizer.param_groups)} 个参数组")
print(f"这个参数组里有 {len(optimizer.param_groups[0]['params'])} 个参数张量")

print(f"\n优化器管理的参数列表:")
for i, p in enumerate(optimizer.param_groups[0]['params']):
    print(f"  参数{i}: shape={tuple(p.shape)}, numel={p.numel()}")

print(f"\n总参数量: {sum(p.numel() for p in optimizer.param_groups[0]['params'])}")

print("""
========== 核心结论 ==========

model.parameters() 收集所有 nn.Parameter 对象:
  - nn.Embedding.weight       -- 词嵌入表
  - nn.Linear.weight          -- 线性变换矩阵
  - nn.Linear.bias            -- 偏置项(如果有)
  - nn.LayerNorm.weight/bias  -- 归一化参数(如果有)

optimizer 持有这些参数的引用:
  - step() 时遍历所有参数: p = p - lr * p.grad
  - zero_grad() 时清空所有参数的 grad

参数数量计算:
  Embedding:    vocab_size × d_model
  Linear:       in_features × out_features (+ out_features 如果有bias)
  Attention(Q/K/V/O): 4 × d_model × d_model
  FFN:          d_model × d_ff + d_ff × d_model
  Output Proj:  d_model × vocab_size
""")
