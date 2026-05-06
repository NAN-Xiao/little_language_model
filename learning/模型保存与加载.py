"""
模型参数保存与加载详解
运行: python learning/checkpoint_demo.py

========== 核心概念 ==========
PyTorch 保存模型有两种方式:
  方式1: 保存整个模型 (model)           → 文件大, 加载快, 但依赖代码结构
  方式2: 只保存参数 (state_dict)        → 文件小, 推荐做法, 需要重新定义模型

推荐做法: 只保存 state_dict + 其他训练信息(epoch, loss, optimizer)
"""

import torch
import torch.nn as nn
import os

# ============ 创建一个极简模型 ============
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 8)
        self.linear = nn.Linear(8, 4)

    def forward(self, x):
        x = self.embedding(x)
        return self.linear(x)

model = TinyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 模拟训练一步
x = torch.tensor([[1, 2, 3]])
loss = model(x).sum()
loss.backward()
optimizer.step()

# ============ 方式1: 保存整个模型 (不推荐) ============
# torch.save(model, ...) 保存整个模型对象
#
# 缺点:
#   - 文件大 (包含模型结构和参数)
#   - 加载时需要相同的类定义 (如果类变了会报错)
#   - 不保存 optimizer 状态
#
# 结果:
#   文件: checkpoint_full_model.pt
#   大小: 3871 bytes (3.8 KB)
#   加载后类型: <class '__main__.TinyModel'>
#   参数相同: True
#
torch.save(model, "learning/checkpoint_full_model.pt")

# ============ 方式2: 只保存参数 state_dict (推荐) ============
# state_dict 是什么?
#   - 一个普通的 Python OrderedDict
#   - key: 参数名 (如 'linear.weight')
#   - value: 参数的 tensor 值
#
# 结果:
#   state_dict 类型: <class 'collections.OrderedDict'>
#   包含 3 个参数:
#     embedding.weight: shape=(10, 8)
#     linear.weight:    shape=(4, 8)
#     linear.bias:      shape=(4,)
#
# 保存后:
#   文件: checkpoint_state_dict.pt
#   大小: 2847 bytes (2.8 KB)
#   比方式1小了: 1024 bytes
#
state_dict = model.state_dict()

torch.save(state_dict, "learning/checkpoint_state_dict.pt")

# 加载 state_dict
# 步骤1: 重新定义模型 (代码必须一样)
new_model = TinyModel()
# 步骤2: 加载参数
new_model.load_state_dict(torch.load("learning/checkpoint_state_dict.pt", weights_only=True))
# 加载后参数相同: True

# ============ 方式3: 保存完整的训练状态 (最完整) ============
# 训练中通常需要保存更多信息:
#   - epoch: 当前训练轮数
#   - model_state_dict: 模型参数
#   - optimizer_state_dict: 优化器状态(动量等)
#   - loss: 当前loss
#   - best_loss: 最佳loss
#
# 结果:
#   文件: checkpoint_complete.pt
#   大小: 6423 bytes (6.3 KB)
#   包含内容:
#     - epoch: 5
#     - model_state_dict: 3 个参数
#     - optimizer_state_dict: 2 个状态
#     - loss: 1.2345
#     - best_loss: 0.9876
#
checkpoint = {
    "epoch": 5,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": 1.2345,
    "best_loss": 0.9876,
}

torch.save(checkpoint, "learning/checkpoint_complete.pt")

# 加载完整状态
loaded_checkpoint = torch.load("learning/checkpoint_complete.pt", weights_only=False)

# 恢复模型
resume_model = TinyModel()
resume_model.load_state_dict(loaded_checkpoint["model_state_dict"])

# 恢复优化器
resume_optimizer = torch.optim.Adam(resume_model.parameters(), lr=0.001)
resume_optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])

# 恢复后:
#   epoch: 5
#   loss: 1.2345
#   参数相同: True

# ============ 三种方式对比 ============
# 方式              文件大小    保存内容                    推荐度
# ─────────────────────────────────────────────────────────────
# 完整模型          3.8 KB     模型结构 + 参数               不推荐
# 仅参数            2.8 KB     参数权重                     推荐
# 完整状态          6.3 KB     参数 + 优化器 + epoch + loss   最推荐
#
# 最佳实践:
#   1. 保存: torch.save(checkpoint, 'model.pt')
#      checkpoint = {
#          'epoch': epoch,
#          'model_state_dict': model.state_dict(),
#          'optimizer_state_dict': optimizer.state_dict(),
#          'loss': loss,
#      }
#
#   2. 加载:
#      checkpoint = torch.load('model.pt')
#      model.load_state_dict(checkpoint['model_state_dict'])
#      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#
#   3. state_dict 本质:
#      - 普通 Python OrderedDict
#      - key: 参数名 (如 'embedding.weight')
#      - value: torch.Tensor (参数值)
#      - 可以修改、可以打印、可以部分加载

# 清理临时文件
for f in ["learning/checkpoint_full_model.pt", "learning/checkpoint_state_dict.pt", "learning/checkpoint_complete.pt"]:
    if os.path.exists(f):
        os.remove(f)
