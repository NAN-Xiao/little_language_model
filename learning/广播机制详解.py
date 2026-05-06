"""
广播机制详解 —— 形状对齐后，输出值是什么？
运行: python learning/broadcasting_demo.py
"""

import torch

print("=" * 60)
print("例1: A(2,3) + B(3,) → 结果是什么？")
print("=" * 60)

A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])     # shape: (2, 3)

B = torch.tensor([10, 20, 30])    # shape: (3,)

print(f"\nA =\n{A}")
print(f"A.shape = {tuple(A.shape)}")

print(f"\nB = {B}")
print(f"B.shape = {tuple(B.shape)}")

# 广播过程:
# B: (3,) → 补齐为 (1, 3) → 扩展为 (2, 3)
# B 广播后 = [[10, 20, 30],
#             [10, 20, 30]]

print("\n【广播过程】")
print("B: (3,) → 补齐 (1, 3) → 扩展 (2, 3)")
print("B 广播后 = [[10, 20, 30],")
print("           [10, 20, 30]]")

result = A + B
print(f"\n【计算结果】A + B =\n{result}")
print(f"结果.shape = {tuple(result.shape)}")

print("\n【每个值怎么来的】")
print(f"  result[0,0] = A[0,0] + B[0] = 1 + 10 = 11")
print(f"  result[0,1] = A[0,1] + B[1] = 2 + 20 = 22")
print(f"  result[0,2] = A[0,2] + B[2] = 3 + 30 = 33")
print(f"  result[1,0] = A[1,0] + B[0] = 4 + 10 = 14")
print(f"  result[1,1] = A[1,1] + B[1] = 5 + 20 = 25")
print(f"  result[1,2] = A[1,2] + B[2] = 6 + 30 = 36")

print("\n" + "=" * 60)
print("例2: A(3,1) + B(1,4) → 结果是什么？")
print("=" * 60)

A = torch.tensor([[1],
                  [2],
                  [3]])           # shape: (3, 1)

B = torch.tensor([[10, 20, 30, 40]])  # shape: (1, 4)

print(f"\nA =\n{A}")
print(f"A.shape = {tuple(A.shape)}")

print(f"\nB =\n{B}")
print(f"B.shape = {tuple(B.shape)}")

# 广播过程:
# A: (3, 1) → 扩展为 (3, 4)
# B: (1, 4) → 扩展为 (3, 4)

print("\n【广播过程】")
print("A: (3, 1) → 扩展 (3, 4)")
print("  [[1, 1, 1, 1],    ← 第0列复制4份")
print("   [2, 2, 2, 2],")
print("   [3, 3, 3, 3]]")
print("")
print("B: (1, 4) → 扩展 (3, 4)")
print("  [[10, 20, 30, 40],  ← 第0行复制3份")
print("   [10, 20, 30, 40],")
print("   [10, 20, 30, 40]]")

result = A + B
print(f"\n【计算结果】A + B =\n{result}")
print(f"结果.shape = {tuple(result.shape)}")

print("\n【每个值怎么来的】")
print(f"  result[0,0] = A[0,0] + B[0,0] = 1 + 10 = 11")
print(f"  result[0,1] = A[0,0] + B[0,1] = 1 + 20 = 21")
print(f"  result[0,2] = A[0,0] + B[0,2] = 1 + 30 = 31")
print(f"  result[0,3] = A[0,0] + B[0,3] = 1 + 40 = 41")
print(f"  result[1,0] = A[1,0] + B[0,0] = 2 + 10 = 12")
print(f"  result[2,3] = A[2,0] + B[0,3] = 3 + 40 = 43")

print("\n" + "=" * 60)
print("例3: 标量广播")
print("=" * 60)

A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])     # shape: (2, 3)

B = 10                            # 标量，shape: ()

print(f"\nA =\n{A}")
print(f"A.shape = {tuple(A.shape)}")
print(f"\nB = {B}")
print(f"B.shape = ()")

print("\n【广播过程】")
print("B: () → 补齐 (1, 1) → 扩展 (2, 3)")
print("B 广播后 = [[10, 10, 10],")
print("           [10, 10, 10]]")

result = A + B
print(f"\n【计算结果】A + B =\n{result}")

print("\n" + "=" * 60)
print("例4: 不能广播的情况")
print("=" * 60)

A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])     # shape: (2, 3)

B = torch.tensor([10, 20])        # shape: (2,)

print(f"\nA =\n{A}")
print(f"A.shape = {tuple(A.shape)}")
print(f"\nB = {B}")
print(f"B.shape = {tuple(B.shape)}")

print("\n【形状对齐】")
print("A: (2, 3)")
print("B: (2,) → 补齐 (1, 2)")
print("")
print("对齐后:")
print("  A: (2, 3)")
print("  B: (1, 2)")
print("      ↑")
print("      最后一维: 3 ≠ 2，而且都不是1！")
print("")

try:
    result = A + B
except RuntimeError as e:
    print(f"【结果】报错！\n{e}")

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
广播后的输出值 = 对应位置运算

  A(2,3) + B(3,)  →  B广播为(2,3)  →  A[i,j] + B[j]
  A(3,1) + B(1,4) →  A扩为(3,4), B扩为(3,4) → A[i,0] + B[0,j]
  A(2,3) + 标量   →  标量扩为(2,3) → A[i,j] + 标量

核心: 广播后形状相同，然后对应位置一对一运算。
""")
