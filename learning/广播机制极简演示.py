"""
广播机制极简演示 —— 具体值是怎么算的
运行: python learning/broadcasting_simple_demo.py
"""

import torch

# ========== 最简单的例子 ==========
print("=" * 50)
print("例1: 最简单的广播 A(2,3) + B(3,)")
print("=" * 50)

A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

B = torch.tensor([10, 20, 30])

print(f"\nA =\n{A}")
print(f"A.shape = {tuple(A.shape)}  <-- 2行3列")

print(f"\nB = {B}")
print(f"B.shape = {tuple(B.shape)}  <-- 1行3列")

print("\n【广播】B 从 (3,) 扩展为 (2, 3):")
print("  B 原来:  [10, 20, 30]")
print("  B 广播后:[[10, 20, 30],")
print("           [10, 20, 30]]")

result = A + B
print(f"\n【结果】A + B =\n{result}")

print("\n【每个值怎么算】")
print("  [0,0] = 1 + 10 = 11")
print("  [0,1] = 2 + 20 = 22")
print("  [0,2] = 3 + 30 = 33")
print("  [1,0] = 4 + 10 = 14")
print("  [1,1] = 5 + 20 = 25")
print("  [1,2] = 6 + 30 = 36")


# ========== 两个方向都广播 ==========
print("\n" + "=" * 50)
print("例2: 两个方向都广播 A(3,1) + B(1,4)")
print("=" * 50)

A = torch.tensor([[1],
                  [2],
                  [3]])

B = torch.tensor([[10, 20, 30, 40]])

print(f"\nA =\n{A}")
print(f"A.shape = {tuple(A.shape)}  <-- 3行1列")

print(f"\nB =\n{B}")
print(f"B.shape = {tuple(B.shape)}  <-- 1行4列")

print("\n【广播】")
print("  A 从 (3,1) 扩展为 (3, 4):")
print("    [[1, 1, 1, 1],")
print("     [2, 2, 2, 2],")
print("     [3, 3, 3, 3]]")
print("")
print("  B 从 (1,4) 扩展为 (3, 4):")
print("    [[10, 20, 30, 40],")
print("     [10, 20, 30, 40],")
print("     [10, 20, 30, 40]]")

result = A + B
print(f"\n【结果】A + B =\n{result}")

print("\n【每个值怎么算】")
print("  [0,0] = 1 + 10 = 11")
print("  [0,1] = 1 + 20 = 21")
print("  [0,2] = 1 + 30 = 31")
print("  [0,3] = 1 + 40 = 41")
print("  [1,0] = 2 + 10 = 12")
print("  [2,3] = 3 + 40 = 43")


# ========== 标量广播 ==========
print("\n" + "=" * 50)
print("例3: 标量广播 A(2,3) + 10")
print("=" * 50)

A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(f"\nA =\n{A}")
print(f"A.shape = {tuple(A.shape)}")

print(f"\nB = 10")
print(f"B.shape = ()  <-- 标量")

print("\n【广播】B 从 () 扩展为 (2, 3):")
print("  B 广播后:[[10, 10, 10],")
print("           [10, 10, 10]]")

result = A + 10
print(f"\n【结果】A + 10 =\n{result}")

print("\n【每个值怎么算】")
print("  [0,0] = 1 + 10 = 11")
print("  [0,1] = 2 + 10 = 12")
print("  [1,2] = 6 + 10 = 16")


# ========== 报错的情况 ==========
print("\n" + "=" * 50)
print("例4: 不能广播的情况 A(2,3) + B(2,)")
print("=" * 50)

A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

B = torch.tensor([10, 20])

print(f"\nA =\n{A}")
print(f"A.shape = {tuple(A.shape)}")

print(f"\nB = {B}")
print(f"B.shape = {tuple(B.shape)}")

print("\n【对齐】")
print("  A: (2, 3)")
print("  B: (2,) → 补齐 (1, 2)")
print("")
print("  最后一维: A=3, B=2")
print("  3 ≠ 2，而且都不是1！")
print("  → 不能广播！报错！")

try:
    result = A + B
except RuntimeError as e:
    print(f"\n【结果】\n{e}")
