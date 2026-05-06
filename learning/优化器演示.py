"""
Optimizer 优化器使用流程演示
运行: python learning/optimizer_demo.py

========== 核心概念 ==========
优化器(Optimizer)的作用: 根据计算出的梯度(grad)，更新模型的参数，让loss逐渐减小。

核心4步流程:
  1. optimizer.zero_grad()  -- 清空旧梯度
  2. loss.backward()        -- 计算新梯度
  3. optimizer.step()       -- 用梯度更新参数
  4. (可选) 调整学习率       -- 控制步长
"""

import torch
import torch.nn as nn

# ============ 创建一个极简模型 ============
# 模型: y = w * x + b
# 目标: 让模型学会 w=2.0, b=1.0 (真实关系: y = 2x + 1)
model = nn.Linear(1, 1)  # 1个输入, 1个输出

# 初始参数(随机):
#   w = model.weight    (初始约 -0.5)
#   b = model.bias      (初始约 0.3)
print("初始参数:")
print(f"  w = {model.weight.item():.4f}")
print(f"  b = {model.bias.item():.4f}")
print("  目标: w=2.0, b=1.0")

# ============ 训练数据 ============
# 输入 x: [1, 2, 3, 4]
# 输出 y: [3, 5, 7, 9]  (因为 y = 2x + 1)
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_true = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

# ============ 损失函数 ============
# MSELoss: 均方误差, 计算预测值和真实值的差距
criterion = nn.MSELoss()

# ============ 优化器 ============
# SGD: 随机梯度下降
#   lr=0.01: 学习率(步长), 每次更新参数的步幅大小
#   参数 = 参数 - lr * 梯度
#
# 对比 Adam(更常用):
#   optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#   Adam会自动调整每个参数的步长, 通常收敛更快
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print(f"\n{'='*60}")
print("【训练开始】")
print(f"{'='*60}")

# ============ 训练10轮 ============
for epoch in range(10):
    # ===== 第1步: 清空旧梯度 =====
    # 为什么: PyTorch的梯度是累加的。如果不清理, 上一步的残差会干扰当前计算
    optimizer.zero_grad()

    # ===== Forward: 前向传播 =====
    y_pred = model(x)  # 预测: y_pred = w*x + b

    # ===== 计算 Loss =====
    loss = criterion(y_pred, y_true)
    # loss = mean((y_pred - y_true)²)
    # loss越大, 预测越不准; loss=0, 预测完全正确

    # ===== 第2步: 计算梯度 =====
    # backward() 从loss往回走, 计算每个参数(w和b)的梯度
    # 梯度告诉参数应该往哪个方向调整, 才能让loss减小
    loss.backward()

    # backward() 后, 每个参数的 .grad 被填充:
    #   model.weight.grad = d(loss)/d(w)  -- loss对w的偏导数
    #   model.bias.grad   = d(loss)/d(b)  -- loss对b的偏导数
    #
    # 梯度为正: 增加这个参数会让loss增大 → 应该减小这个参数
    # 梯度为负: 增加这个参数会让loss减小 → 应该增大这个参数

    # ===== 第3步: 更新参数 =====
    # step() 执行: w = w - lr * w.grad
    #              b = b - lr * b.grad
    #
    # 例: 如果 w=0.5, w.grad=10.0, lr=0.01
    #     新 w = 0.5 - 0.01*10.0 = 0.4  (梯度为正, 减小w)
    optimizer.step()

    # ===== 打印结果 =====
    w = model.weight.item()
    b = model.bias.item()
    print(f"Epoch {epoch+1:2d} | loss={loss.item():8.4f} | w={w:7.4f} | b={b:7.4f}")

print(f"\n{'='*60}")
print("【训练结束】")
print(f"{'='*60}")
print(f"最终参数:")
print(f"  w = {model.weight.item():.4f} (目标: 2.0)")
print(f"  b = {model.bias.item():.4f} (目标: 1.0)")

# ============ 验证 ============
print(f"\n{'='*60}")
print("【验证】")
print(f"{'='*60}")
test_x = torch.tensor([[5.0]])  # x=5, 真实y=11
pred_y = model(test_x)
print(f"  输入 x=5, 预测 y={pred_y.item():.4f}, 真实 y=11.0")

"""
========== 总结 ==========

优化器4步流程:
  1. zero_grad()  -- 清空旧梯度(防止累加干扰)
  2. backward()   -- 计算新梯度(告诉参数该往哪走)
  3. step()       -- 更新参数(沿着梯度方向走一步)
  4. (循环)        -- 重复以上步骤, loss逐渐减小

学习率(lr)的作用:
  - lr 太大: 步子太大, 在最优解附近来回震荡, 甚至发散
  - lr 太小: 步子太小, 收敛极慢, 训练时间太长
  - lr 合适: 稳定收敛到最优解

SGD vs Adam:
  - SGD: 简单直接, 所有参数用同一个学习率
  - Adam: 自适应, 每个参数有自己的"步长", 通常收敛更快更稳定
  - 训练Transformer通常用Adam (lr=3e-4, betas=(0.9, 0.98))
"""
