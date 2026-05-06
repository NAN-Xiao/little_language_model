"""
model.parameters() 收集规则详解
运行: python learning/parameters_rules_demo.py

========== 核心规则 ==========
PyTorch 的 nn.Module 在创建时，会自动扫描 self 的所有属性，
收集其中的 nn.Parameter 和子 nn.Module 的参数。

规则1: self.xxx = nn.Parameter(...)  → 被收集
规则2: self.xxx = nn.Linear(...)      → 被收集 (递归收集子模块的参数)
规则3: self.xxx = nn.Embedding(...)   → 被收集 (递归收集子模块的参数)
规则4: self.xxx = torch.tensor(...)   → 不被收集 (普通tensor，不是Parameter)
规则5: self.xxx = 3.14                → 不被收集 (普通Python数值)
规则6: self.xxx = [nn.Linear(...)]    → 不被收集 (放在list里，Module发现不了)
       解决: 用 nn.ModuleList([...])  → 被收集
规则7: self.xxx = {'a': nn.Linear(...)} → 不被收集 (放在dict里，Module发现不了)
       解决: 用 nn.ModuleDict({...})  → 被收集
"""

import torch
import torch.nn as nn


class DemoModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ===== 规则1: nn.Parameter 会被收集 =====
        self.manual_weight = nn.Parameter(torch.randn(3, 3))
        # 这是一个手动创建的参数，会被 model.parameters() 收集
        # 参数量: 3 × 3 = 9

        # ===== 规则2: nn.Linear 会被收集 =====
        self.linear = nn.Linear(4, 5)
        # nn.Linear 内部有两个 Parameter:
        #   - weight: shape (5, 4), 参数量 20
        #   - bias:   shape (5,),   参数量 5
        # 总参数量: 25

        # ===== 规则3: nn.Embedding 会被收集 =====
        self.embedding = nn.Embedding(10, 8)
        # nn.Embedding 内部有一个 Parameter:
        #   - weight: shape (10, 8), 参数量 80

        # ===== 规则4: 普通 torch.tensor 不会被收集 =====
        self.fixed_tensor = torch.randn(3, 3)
        # 这是普通的 torch.Tensor，不是 nn.Parameter
        # model.parameters() 不会收集它！
        # 因为它没有 requires_grad=True，也不会被优化器更新

        # ===== 规则5: Python 数值不会被收集 =====
        self.scale = 2.0
        # 这是 Python float，不是 tensor
        # model.parameters() 不会收集它！

        # ===== 规则6: 放在 list 里的子模块不会被收集 =====
        self.bad_layers = [nn.Linear(2, 2), nn.Linear(2, 2)]
        # 放在 Python list 里，nn.Module 的 __setattr__ 发现不了！
        # model.parameters() 不会收集这两个 Linear 的参数！
        # 参数量丢失: 2 × (2×2 + 2) = 12 个参数不会被训练！

        # 正确做法: 用 nn.ModuleList
        self.good_layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        # nn.ModuleList 是 PyTorch 专门设计的容器
        # 里面的每个模块都会被 model.parameters() 收集
        # 参数量: 2 × (2×2 + 2) = 12

        # ===== 规则7: 放在 dict 里的子模块不会被收集 =====
        self.bad_dict = {'a': nn.Linear(3, 3), 'b': nn.Linear(3, 3)}
        # 放在 Python dict 里，nn.Module 发现不了！
        # 参数量丢失: 2 × (3×3 + 3) = 24

        # 正确做法: 用 nn.ModuleDict
        self.good_dict = nn.ModuleDict({
            'a': nn.Linear(3, 3),
            'b': nn.Linear(3, 3)
        })
        # nn.ModuleDict 会被 model.parameters() 收集
        # 参数量: 2 × (3×3 + 3) = 24


# ============ 创建模型并查看参数 ============
model = DemoModel()

print("=" * 70)
print("model.parameters() 收集的参数:")
print("=" * 70)

total = 0
for name, param in model.named_parameters():
    n = param.numel()
    total += n
    print(f"  {name:<30} shape={str(tuple(param.shape)):<12} numel={n}")

print(f"\n总参数量: {total}")

# ============ 对比: 哪些参数"丢失"了 ============
print("\n" + "=" * 70)
print("注意: 以下参数没有被 model.parameters() 收集！")
print("=" * 70)

print(f"\n  self.fixed_tensor (普通tensor):")
print(f"    type={type(model.fixed_tensor)}, shape={tuple(model.fixed_tensor.shape)}")
print(f"    → 不是 nn.Parameter，不会被收集，不会被训练")

print(f"\n  self.scale (Python float):")
print(f"    value={model.scale}, type={type(model.scale)}")
print(f"    → 不是 tensor，不会被收集")

print(f"\n  self.bad_layers[0] (放在list里的Linear):")
print(f"    这个Linear有 {sum(p.numel() for p in model.bad_layers[0].parameters())} 个参数")
print(f"    → 但放在list里，model.parameters() 发现不了！")
print(f"    → 这 {sum(p.numel() for p in model.bad_layers[0].parameters()) * 2} 个参数永远不会被训练！")

print(f"\n  self.bad_dict['a'] (放在dict里的Linear):")
print(f"    这个Linear有 {sum(p.numel() for p in model.bad_dict['a'].parameters())} 个参数")
print(f"    → 但放在dict里，model.parameters() 发现不了！")
print(f"    → 这 {sum(p.numel() for p in model.bad_dict['a'].parameters()) * 2} 个参数永远不会被训练！")

# ============ 正确vs错误对比 ============
print("\n" + "=" * 70)
print("正确 vs 错误 对比:")
print("=" * 70)

print("""
  错误写法 (参数丢失!)                正确写法
  ──────────────────                  ──────────────────
  self.layers = [                     self.layers = nn.ModuleList([
      nn.Linear(2, 2),                    nn.Linear(2, 2),
      nn.Linear(2, 2)                     nn.Linear(2, 2)
  ]                                   ])
  → 参数不会被训练!                    → 参数会被正确收集和训练

  self.blocks = {                     self.blocks = nn.ModuleDict({
      'a': nn.Linear(3, 3),               'a': nn.Linear(3, 3),
      'b': nn.Linear(3, 3)                'b': nn.Linear(3, 3)
  }                                   })
  → 参数不会被训练!                    → 参数会被正确收集和训练
""")

# ============ 底层原理 ============
print("=" * 70)
print("底层原理: nn.Module.__setattr__")
print("=" * 70)
print("""
当你写 self.xxx = ... 时，PyTorch 会调用 nn.Module.__setattr__()。

这个方法的逻辑:
  1. 如果 value 是 nn.Parameter 类型 → 加入 _parameters 字典
  2. 如果 value 是 nn.Module 类型    → 加入 _modules 字典
  3. 其他情况 → 作为普通属性存储

model.parameters() 就是遍历:
  - self._parameters 中的参数
  - self._modules 中每个子模块的 _parameters (递归)

所以:
  - self.manual_weight = nn.Parameter(...)  → 加入 _parameters [对]
  - self.linear = nn.Linear(...)            → 加入 _modules [对]
  - self.fixed_tensor = torch.randn(...)    → 普通属性 [错]
  - self.bad_layers = [nn.Linear(...)]      → 普通属性(list) [错]
  - self.good_layers = nn.ModuleList(...)   → 加入 _modules [对]
""")

# ============ 验证: 手动对比 ============
print("=" * 70)
print("验证: 手动对比收集到的参数")
print("=" * 70)

print("\n收集到的参数名:")
for name, _ in model.named_parameters():
    print(f"  - {name}")

print("\n注意: 以下名称不在列表中:")
print("  - fixed_tensor          (普通tensor)")
print("  - scale                 (Python数值)")
print("  - bad_layers.0.weight   (放在list里的Linear)")
print("  - bad_layers.1.weight   (放在list里的Linear)")
print("  - bad_dict.a.weight     (放在dict里的Linear)")
print("  - bad_dict.b.weight     (放在dict里的Linear)")

expected_total = (
    9 +           # manual_weight (3×3)
    25 +          # linear (5×4 + 5)
    80 +          # embedding (10×8)
    12 +          # good_layers[0] (2×2 + 2)
    12 +          # good_layers[1] (2×2 + 2)
    12 +          # good_dict['a'] (3×3 + 3)
    12            # good_dict['b'] (3×3 + 3)
)
print(f"\n预期总参数量: {expected_total}")
print(f"实际总参数量: {total}")
print(f"匹配: {expected_total == total}")

"""
========== 常见问题 ==========

Q: self.xxx 都会收集吗？
A: 不是！取决于赋值的是什么类型：
   - self.xxx = nn.Parameter(...)  → 收集
   - self.xxx = nn.Linear(...)     → 收集
   - self.xxx = torch.tensor(...)  → 不收集
   - self.xxx = 3.14               → 不收集

Q: 在 forward 中没用到，只是声明了，会收集吗？
A: 会！收集发生在 __init__ 赋值时，不是在 forward 使用时。
   只要类型对了（nn.Parameter 或 nn.Module），就会被收集。

   但如果 forward 中没用到：
   - 参数仍然会被收集
   - 梯度仍然会计算
   - 优化器仍然会更新
   - 只是浪费计算，不会报错

Q: forward 中临时创建的 tensor 会被收集吗？
A: 不会！只有 self.xxx = ... 这种形式才会被收集。
   forward 中的局部变量（如 x = torch.randn(...)）不会被收集。

========== 总结 ==========

model.parameters() 收集规则:

  [对] 被收集:
    - nn.Parameter(...)
    - nn.Module 子类 (nn.Linear, nn.Embedding, nn.Conv2d 等)
    - nn.ModuleList([...])
    - nn.ModuleDict({...})

  [错] 不被收集:
    - 普通 torch.Tensor
    - Python 数值 (int, float, str)
    - Python list (即使里面放的是 nn.Module)
    - Python dict (即使里面放的是 nn.Module)

  常见错误:
    - 用 list 存放层 → 改用 nn.ModuleList
    - 用 dict 存放层 → 改用 nn.ModuleDict
    - 忘记把 tensor 包装成 nn.Parameter
"""