"""
运行这个脚本，看注意力计算中张量形状的变化。
比看注释直观 100 倍。
"""
import torch

B, seq, n_heads, d_k = 2, 4, 3, 5  # 2个句子, 4个词, 3个头, 每头5维(比10好画)

# 模拟 q, k (transpose 后的形状)
q = torch.randn(B, n_heads, seq, d_k)
k = torch.randn(B, n_heads, seq, d_k)

print("=" * 50)
print("【transpose 后】")
print(f"q 形状:    {q.shape}    (B=2, heads=3, seq=4, d_k=5)")
print(f"k 形状:    {k.shape}    (B=2, heads=3, seq=4, d_k=5)")

k_t = k.transpose(-2, -1)
print(f"k^T 形状:  {k_t.shape}    (最后两维交换: 4,5 → 5,4)")

scores = torch.matmul(q, k_t)
print(f"q @ k^T:   {scores.shape}    (4,5) @ (5,4) = (4,4) ← 词×词!")
print()
print("scores[0,0] 是第一个句子的第一个头的 4×4 注意力矩阵:")
print(scores[0, 0].round(decimals=2))
print()

print("=" * 50)
print("【对比: 不 transpose 会怎样】")
q_bad = torch.randn(B, seq, n_heads, d_k)
k_bad = torch.randn(B, seq, n_heads, d_k)
k_bad_t = k_bad.transpose(-2, -1)
print(f"q 形状:    {q_bad.shape}    (B=2, seq=4, heads=3, d_k=5)")
print(f"k^T 形状:  {k_bad_t.shape}    (最后两维交换: 3,5 → 5,3)")
scores_bad = torch.matmul(q_bad, k_bad_t)
print(f"q @ k^T:   {scores_bad.shape}    (3,5) @ (5,3) = (3,3) ← 头×头! 错了!")
