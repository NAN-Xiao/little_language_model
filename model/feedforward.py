import torch.nn as nn

#PositionwiseFeedForward的中文含义是 位置独立的前馈网络。
#位置独立的前馈网络，意味着每个位置的输出只依赖于该位置的输入，与其他位置的输入无关。
#是旋转位置编码的，不是位置编码的。
class PositionwiseFeedForward(nn.Module):
    """
    两层前馈全连接网络（Position-wise FeedForward Network, FFN）：
      FFN(x) = Linear2(ReLU(Linear1(x)))
      结构为：线性层 → 激活 → 线性层，实际是输入-升维-激活-降维-输出。
      
    这里的“两层”指的是有两组线性变换（nn.Linear），分别是升维和再降维，激活函数通常夹在中间。
      - 第1层：输入维度 d_model → 升维到 d_ff
      - 第2层：d_ff → 再降维回 d_model
    隐藏层通常指的是这两次线性之间的那个更高维度（d_ff），实际的参数（权重矩阵）在每一层都独立。

    业界主流 Transformer（如Qwen、GPT、BERT等）也都是2层前馈网络（2个线性层，中间激活），少数模型尝试过3层结构，但主流实现仍为2层。
      - 例：Qwen2、Llama2、GPT-3、BERT、T5 フィードフォワード全都是2层结构。
      - 3层情况极少见，1层更不常见，2层最常用。

    nn.Linear 就是一组权重+bias，实现了 y = x @ W + b 的全连接变换。

    参数:
        d_model (int): 输入与输出的特征维度（等于模型主通道数）
        d_ff (int): 隐藏层（升维后）的特征维度（通常是 d_model 的4倍）
        dropout (float): dropout 比例（默认0.1）
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        """
        两层全连接网络: FFN(x) = ReLU(xW1 + b1)W2 + b2
        nn.Linear是一个线性层，它是一个矩阵，xW1 + b1就是矩阵乘法和加法。
        参数:
            d_model (int): 输入的维度
            d_ff (int): 中间层的维度
            dropout (float): dropout 的比例 (默认值为 0.1)
        """
        #Sequential是PyTorch中的一个模块，它是一个有序的容器，用于按顺序执行一系列的模块。
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),  # 第一层线性变换（输入维度到中间维度）
            nn.ReLU(),  # 激活函数 ReLU
            nn.Dropout(dropout),  # dropout 随机丢弃部分神经元，防止过拟合
            nn.Linear(d_ff, d_model),  # 第二层线性变换（中间维度到输出维度）
            nn.Dropout(dropout),  # 再次应用 dropout
        )

    # 前向传播
    # 参数:
    #   x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)
    # 返回值:
    #   torch.Tensor: 输出张量，形状为 (batch_size, seq_len, d_model)
    # 在哪调用了这个forward？在DecoderBlock中调用了。
    def forward(self, x):
        return self.net(x)
