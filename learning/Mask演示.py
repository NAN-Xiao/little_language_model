"""
极简演示: mask 到底是什么、怎么变
=================================
运行: python learning/mask_demo.py
"""

import torch
import torch.nn.functional as F

print("=" * 60)
print("【演示1】训练时: 3个token一起输入，mask是下三角")
print("=" * 60)

# 假设3个token: "我" "爱" "你"
seq_len = 3

# mask怎么构造的:
# torch.tril(torch.ones(3,3)) 取下三角
causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
print(f"\n输入: 3个token")
print(f"make_causal_mask 输出:")
print(f"形状: {tuple(causal_mask.shape)}")
print(causal_mask.int())

# 模拟注意力分数 (3个token互相打分)
scores = torch.tensor([
    [5.0, 3.0, 1.0],   # "我"对["我","爱","你"]的分数
    [4.0, 6.0, 2.0],   # "爱"对["我","爱","你"]的分数
    [3.0, 5.0, 7.0],   # "你"对["我","爱","你"]的分数
])

print(f"\n注意力分数 (scores):")
print(scores)

# mask作用: 把0的位置填-inf
masked_scores = scores.masked_fill(causal_mask == 0, float("-inf"))
print(f"\nmask后 (0的位置变-inf):")
print(masked_scores)

# softmax -> 概率
probs = F.softmax(masked_scores, dim=-1)
print(f"\nsoftmax后 (概率):")
print(probs)

print("\n解读:")
print("  '我'只看自己: [1.0, 0, 0]")
print("  '爱'看'我'和自己: [0.12, 0.88, 0]")
print("  '你'看全部: [0.02, 0.10, 0.88]")
print("  → 因为mask是下三角，'爱'的'你'分数变成-inf，softmax后概率=0")


print("\n" + "=" * 60)
print("【演示2】推理第1步: 和训练一样，完整prompt一起算")
print("=" * 60)

prompt = [1, 3, 12]  # [BOS, "我", "爱"]
seq_len = len(prompt)

causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
print(f"\n输入prompt: {prompt} (3个token)")
print(f"mask 形状: {tuple(causal_mask.shape)}")
print(causal_mask.int())
print("→ 下三角，和训练时一模一样")


print("\n" + "=" * 60)
print("【演示3】推理第2步: 只输入1个新token，mask是全1")
print("=" * 60)

# 前面已经缓存了 [BOS, "我", "爱"] 的K和V
# 现在只输入新token "你"
# 当前总序列长度: 4 (BOS + 我 + 爱 + 你)
# 但只输入最后1个！

seq_k = 4  # 缓存里有4个历史token
mask = torch.ones(1, seq_k, dtype=torch.bool)  # (1, 4)

print(f"\n当前序列: [BOS, '我', '爱', '你'] (共4个)")
print(f"但只输入最后1个: ['你']")
print(f"mask: torch.ones(1, {seq_k})")
print(f"形状: {tuple(mask.shape)}")
print(mask.int())

print("\n→ 全1！为什么？")
print("  因为只输入了1个token ('你')，它作为query只有1行")
print("  它和历史K算注意力: q(1,d) @ K^T(d,4) -> scores(1,4)")
print("  scores只有1行: [?,?,?,?] —— 没有'未来行'需要遮！")
print("  未来根本不存在（还没生成），所以全1就行")


print("\n" + "=" * 60)
print("【核心对比】")
print("=" * 60)
print("""
训练/推理第1步:
  输入: [t0, t1, t2] (多个token)
  q形状: (3, d)  — 3个query
  k形状: (3, d)  — 3个key
  scores: (3, 3) — 3行3列的矩阵
           ┌─────┐
           │ ? ? ? │  ← 第0行: t0看全部
           │ ? ? ? │  ← 第1行: t1看全部
           │ ? ? ? │  ← 第2行: t2看全部
           └─────┘
  → 必须用下三角mask！否则t1会偷看t2

推理后续步:
  输入: [t2] (只有1个token)
  q形状: (1, d)  — 1个query
  k形状: (3, d)  — 3个历史key (来自缓存)
  scores: (1, 3) — 1行3列的矩阵
           ┌───────┐
           │ ? ? ? │  ← 只有1行！没有"未来行"
           └───────┘
  → 全1就行！因为只有1行，不存在偷看问题
""")
