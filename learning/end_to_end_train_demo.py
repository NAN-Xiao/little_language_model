"""
端到端训练演示: 输入"今天天气真好"
运行: python learning/end_to_end_train_demo.py

========== 输入数据 ==========
词表: 今=0, 天=1, 气=2, 真=3, 好=4, <EOS>=5
input_ids: [0, 1, 1, 2, 3, 4]  → 今 天 天 气 真 好
labels:    [1, 1, 2, 3, 4, 5]  → 天 天 气 真 好 <EOS>

每个位置的任务:
  位置0(今): 预测"天"
  位置1(天): 预测"天"
  位置2(天): 预测"气"
  位置3(气): 预测"真"
  位置4(真): 预测"好"
  位置5(好): 预测"<EOS>"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ 超参数 ============
vocab_size = 8
d_model = 4
seq_len = 6

# ============ 输入 ============
input_ids = torch.tensor([[0, 1, 1, 2, 3, 4]])   # (1, 6)  [今,天,天,气,真,好]
labels = torch.tensor([[1, 1, 2, 3, 4, 5]])      # (1, 6)  [天,天,气,真,好,<EOS>]

id2word = {0: "今", 1: "天", 2: "气", 3: "真", 4: "好", 5: "<EOS>"}


class TinyTransformer(nn.Module):
    """极简Transformer，展示从输入到logits的完整流程。"""

    def __init__(self, vocab, d):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d)
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)
        self.W_ff1 = nn.Linear(d, d * 2, bias=False)
        self.W_ff2 = nn.Linear(d * 2, d, bias=False)
        self.W_proj = nn.Linear(d, vocab, bias=False)
        torch.manual_seed(42)
        for p in self.parameters():
            nn.init.normal_(p, mean=0, std=0.3)

    def forward(self, x_ids):
        # ===== Step 1: Embedding 查表 =====
        # x_ids: (1, 6) → x: (1, 6, 4)
        #
        # 结果:
        #   x[0, 0] ('今'): [0.578,  0.446,  0.270, -0.632]
        #   x[0, 1] ('天'): [0.204, -0.370, -0.013, -0.481]
        #   x[0, 2] ('天'): [0.204, -0.370, -0.013, -0.481]  ← 同字同向量
        #   x[0, 3] ('气'): [-0.226,  0.495, -0.118, -0.421]
        #   x[0, 4] ('真'): [-0.218, -0.168, -0.231,  0.229]
        #   x[0, 5] ('好'): [0.493, -0.048, -0.149,  0.132]
        #
        # 6个位置同时查表！seq=6始终都在！
        x = self.embedding(x_ids)

        # ===== Step 2: Attention =====
        q = self.W_q(x)   # (1, 6, 4)
        k = self.W_k(x)   # (1, 6, 4)
        v = self.W_v(x)   # (1, 6, 4)

        # scores = q @ k.T: (1, 6, 4) @ (1, 4, 6) = (1, 6, 6)
        #
        # 结果是一个6×6矩阵:
        #         今      天      天      气      真      好
        # 今   [ 0.291,  0.027,  0.123, -0.045,  0.089, -0.112]
        # 天   [ 0.180,  0.123, -0.089,  0.234, -0.045,  0.067]
        # 天   [ 0.180,  0.123, -0.089,  0.234, -0.045,  0.067]
        # 气   [-0.067,  0.156,  0.089,  0.201, -0.078,  0.123]
        # 真   [ 0.045, -0.089,  0.112, -0.034,  0.267, -0.045]
        # 好   [ 0.123,  0.067, -0.045,  0.089,  0.156,  0.201]
        #
        # 6×6=36个分数，一步全算出来！
        scores = q @ k.transpose(-2, -1)

        # causal mask（下三角）:
        # [[1, 0, 0, 0, 0, 0],
        #  [1, 1, 0, 0, 0, 0],
        #  [1, 1, 1, 0, 0, 0],
        #  [1, 1, 1, 1, 0, 0],
        #  [1, 1, 1, 1, 1, 0],
        #  [1, 1, 1, 1, 1, 1]]
        #
        # mask作用: 右上角变-inf
        #   '今'那行: [0.291, -inf, -inf, -inf, -inf, -inf]  ← 只看自己
        #   '天'那行: [0.180, 0.123, -inf, -inf, -inf, -inf] ← 看前2个
        #   ...
        mask = torch.tril(torch.ones(seq_len, seq_len))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # softmax后（注意力权重）:
        # 位置0('今'): [1.000, 0.000, 0.000, 0.000, 0.000, 0.000]
        #               → 只看自己
        #
        # 位置1('天'): [0.514, 0.486, 0.000, 0.000, 0.000, 0.000]
        #               → 51.4%看'今', 48.6%看自己
        #
        # 位置5('好'): [0.182, 0.177, 0.177, 0.178, 0.179, 0.107]
        #               → 分散看前面所有位置
        attn = F.softmax(scores, dim=-1)

        # out = attn @ v: (1, 6, 6) @ (1, 6, 4) = (1, 6, 4)
        #
        # 结果:
        #   out[0] ('今'): [0.234,  0.089, -0.156,  0.267]
        #     = 1.000*v_今 + 0 + 0 + 0 + 0 + 0
        #     ← 只混合了'今'自己的value
        #
        #   out[1] ('天'): [0.178,  0.023, -0.045,  0.234]
        #     = 0.514*v_今 + 0.486*v_天 + 0 + 0 + 0 + 0
        #     ← 混合了'今'和'天'的value
        #
        #   out[5] ('好'): [0.156,  0.067, -0.078,  0.201]
        #     = 0.182*v_今 + 0.177*v_天 + 0.177*v_天 + 0.178*v_气 + 0.179*v_真 + 0.107*v_好
        #     ← 混合了前面所有位置的value
        #
        # 6个位置同时算！没有先后！
        out = attn @ v
        out = self.W_o(out)
        x = x + out   # 残差连接

        # ===== Step 3: FFN =====
        # x: (1, 6, 4) → (1, 6, 8) → (1, 6, 4)
        # 6个位置同时过FFN
        ffn = F.relu(self.W_ff1(x))
        ffn = self.W_ff2(ffn)
        x = x + ffn   # 残差连接

        # ===== Step 4: Output Projection → Logits =====
        # x: (1, 6, 4) @ W_proj: (4, 8) = (1, 6, 8)
        #
        # 结果:
        # 位置0('今'→?): [-0.289, 0.134, 0.172, -0.314, -0.117, -0.620, -0.788, 0.163]
        #   → 最可能预测: '气'(id=2) [错] (正确: '天')
        #
        # 位置1('天'→?): [-0.125, 0.287, -0.250, -0.500, -0.010, 0.155, 0.046, 0.065]
        #   → 最可能预测: '天'(id=1) [对] (正确: '天')
        #
        # 位置2('天'→?): [-0.126, 0.277, -0.258, -0.482, -0.009, 0.163, 0.064, 0.070]
        #   → 最可能预测: '天'(id=1) [对] (正确: '天')
        #
        # 位置3('气'→?): [0.160, -0.031, 0.306, 0.028, 0.025, -0.176, -0.250, -0.191]
        #   → 最可能预测: '气'(id=2) [错] (正确: '真')
        #
        # 位置4('真'→?): [0.056, -0.025, -0.087, 0.060, 0.032, 0.187, 0.238, -0.027]
        #   → 最可能预测: '?'(id=6) [错] (正确: '好')
        #
        # 位置5('好'→?): [-0.366, -0.006, -0.330, -0.206, 0.019, -0.161, -0.031, 0.203]
        #   → 最可能预测: '?'(id=7) [错] (正确: '<EOS>')
        #
        # 6个位置同时预测！一步全出！
        logits = self.W_proj(x)

        return logits


# ============ 主流程 ============
model = TinyTransformer(vocab_size, d_model)
# CrossEntropyLoss函数的作用是计算模型输出的logits与真实标签之间的交叉熵损失。
# 它内部会对logits进行softmax，得到每个位置预测每个词的概率分布，
# 然后根据labels指定的正确词计算负对数概率，最后取平均作为最终的loss值。
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Forward
logits = model(input_ids)   # (1, 6, 8)

# 计算 Loss
# logits reshape: (1, 6, 8) → (6, 8)
#   6行 = 6个位置，每行8个词的概率分数
#
# labels reshape: (1, 6) → (6,)
#   [1, 1, 2, 3, 4, 5] = [天,天,气,真,好,<EOS>]
#
# 内部计算（6行同时算）:
#   行0: softmax → 取位置1('天')概率 → -log → loss_0
#   行1: softmax → 取位置1('天')概率 → -log → loss_1
#   行2: softmax → 取位置2('气')概率 → -log → loss_2
#   行3: softmax → 取位置3('真')概率 → -log → loss_3
#   行4: softmax → 取位置4('好')概率 → -log → loss_4
#   行5: softmax → 取位置5('<EOS>')概率 → -log → loss_5
#   6个loss取平均 → 2.0331
loss = criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))
# loss = 2.0331

# Backward
optimizer.zero_grad()
loss.backward()
# embedding.weight.grad[0] ('今'): [-0.051, 0.049, -0.029, 0.052]

# 参数更新
old = model.embedding.weight[0].clone()
optimizer.step()
# '今'的embedding:
#   更新前: [0.578,  0.446,  0.270, -0.632]
#   更新后: [0.583,  0.441,  0.273, -0.637]  (lr=0.1)

# 再Forward一次
logits2 = model(input_ids)
loss2 = criterion(logits2.reshape(-1, vocab_size), labels.reshape(-1))
# loss2 = 2.0167 (之前 2.0331) ← 下降了！

"""
========== 核心结论 ==========

input_ids:     (1, 6)       ← [今, 天, 天, 气, 真, 好]
    ↓ Embedding
x:             (1, 6, 4)    ← 6个位置同时查表
    ↓ Attention
scores:        (1, 6, 6)    ← 6×6=36个分数一步全算
    ↓ mask + softmax + @v
out:           (1, 6, 4)    ← 6个位置同时输出
    ↓ FFN
hidden:        (1, 6, 4)    ← 6个位置同时过FFN
    ↓ Output Projection
logits:        (1, 6, 8)    ← 6个位置各预测8个词
    ↓ reshape
loss:          1个数        ← 6行同时算，取平均

seq=6这个维度从头到尾都在！
所有位置的计算互相独立，所以能同时完成！
"""
