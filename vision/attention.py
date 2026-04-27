"""
Vision Transformer 注意力 —— 用"主流算法"让图像 patch 互相理解
===========================================================

这个文件和 model/attention.py 的区别：
  • model/attention.py: 为文字 LLM 设计，手写 attention，支持 causal mask、RoPE、KV-Cache
  • vision/attention.py: 为图像 ViT 设计，用 PyTorch 官方的 scaled_dot_product_attention

为什么单独写一份?
  ViT 的 attention 和文字 LLM 有本质区别：
    - 图像 patch 之间是"双向"的（没有因果限制）
    - 不需要 RoPE（用 2D 位置编码代替）
    - 不需要 KV-Cache（ViT 是编码器，不是自回归生成器）
    - 可以用 PyTorch 2.0 的 SDPA，自动调用 FlashAttention，更快更省显存

主流的注意力计算方式
═══════════════════════

PyTorch 2.0+ 提供了 torch.nn.functional.scaled_dot_product_attention，
这是目前工业界最主流的实现。它的好处:

  1. 自动选择最优后端:
       • FlashAttention-2: 最快、最省显存（需要 Ampere 及以上 GPU）
       • Memory-Efficient Attention: 中等速度，显存友好
       • Math (兜底): 纯 PyTorch 实现，兼容性最好

  2. 一句话调用，省去手写 10 行代码:
       out = F.scaled_dot_product_attention(q, k, v)
       ← 内部自动完成: q@k.T / sqrt(d) → softmax → dropout → @v

  3. 和手写公式数学等价，但速度可能快 2~8 倍

FlashAttention 为什么快?
────────────────────────

普通 attention 的瓶颈不是计算，而是"显存读写"。

想象你有一本 1000 页的书：
  普通做法: 把整本书复印一份（attention 矩阵），然后在复印件上勾画
  FlashAttention: 一次只看 1 页，算完直接写结果，不存中间复印件

具体来说：
  标准 attention: 显式构造 (seq, seq) 的 attention 矩阵
                  例: seq=1024 → 矩阵大小 = 1024×1024 = 1M 个数
                  如果 batch=8, 头数=12 → 8×12×1M = 9600 万个数
                  这些数要从显存读、写、再读，非常慢

  FlashAttention: 把序列分小块，算一块写一块，不存完整矩阵
                  显存复杂度从 O(seq²) 降到 O(seq)
                  → 长序列时优势巨大（如 4096 patch 的图像）

默认配置: d_model=768, n_heads=12, d_k=64
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTMultiHeadAttention(nn.Module):
    """
    ViT 多头注意力 —— 用 PyTorch 官方 SDPA 实现。

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  和文字 LLM Attention 的核心区别                                         │
    │                                                                          │
    │  文字 LLM (model/attention.py):                                          │
    │    • 有 causal mask（下三角遮罩）                                        │
    │    • 有 RoPE 旋转位置编码                                                │
    │    • 有 KV-Cache（推理时缓存已生成的 K/V）                               │
    │    • 手写 matmul → softmax → dropout → matmul                           │
    │                                                                          │
    │  ViT Attention (本类):                                                   │
    │    • 无 causal mask（双向可见）                                          │
    │    • 无 RoPE（用 2D 位置编码代替）                                       │
    │    • 无 KV-Cache（ViT 是一次性编码，不是逐 token 生成）                  │
    │    • 用 F.scaled_dot_product_attention（FlashAttention 加速）            │
    │                                                                          │
    │  类比:                                                                   │
    │    文字 LLM 像写作文——只能从左到右，不能修改前面的字。                     │
    │    ViT 像看照片——一眼扫过去，所有 patch 同时被感知。                       │
    └──────────────────────────────────────────────────────────────────────────┘

    维度流转 (B=2, n_patches=196, d_model=768, n_heads=12):
    ─────────────────────────────────────────

      输入 x:           (2, 196, 768)     ← 196 个图像 patch
          ↓
      w_q/w_k/w_v:      (2, 196, 768)     ← 投影到 Q/K/V
          ↓
      拆多头:            (2, 196, 12, 64)  ← 12 个头，每头 64 维
          ↓
      transpose:         (2, 12, 196, 64)  ← 把头数放前面，方便并行
          ↓
      F.scaled_dot_product_attention:
                         (2, 12, 196, 64)  ← 每个头输出新表示
          ↓
      拼回:              (2, 196, 768)     ← 12 个头拼回 768 维
          ↓
      w_o:               (2, 196, 768)     ← 融合多头信息
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        window_size: int = 0,
        shift_size: int = 0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model      # 768, 每个 patch 的维度
        self.n_heads = n_heads      # 12, 注意力头数
        self.d_k = d_model // n_heads  # 64, 每个头处理的维度
        self.dropout_p = dropout    # SDPA 内部 dropout 概率
        self.window_size = window_size  # 0=全局注意力, >0=窗口大小（如7）
        self.shift_size = shift_size    # 0=正常窗口, >0=移位窗口（如3）

        # Q/K/V/O 四个投影层 —— 和文字模型一样
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def _make_window_mask(
        self,
        seq_len: int,
        grid_size: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        构造窗口注意力掩码 —— 支持正常窗口 和 Shifted Window。

        ┌──────────────────────────────────────────────────────────────────────┐
        │  什么是窗口注意力?                                                   │
        │                                                                      │
        │  普通注意力: 每个 patch 看所有 patch（全局）                         │
        │    196 个 patch → 注意力矩阵 196×196 = 38,416 个分数               │
        │    复杂度 O(N²)，N 是 patch 数                                       │
        │                                                                      │
        │  窗口注意力: 每个 patch 只看"邻居"（局部窗口）                       │
        │    把 14×14 的 patch 网格分成 2×2 个 7×7 的窗口                     │
        │    每个 patch 只看自己窗口内的 49 个 patch                           │
        │    复杂度 O(N × window_size²)，大幅降低                             │
        │                                                                      │
        │  生活类比:                                                           │
        │    全局注意力像开全体员工大会——每个人都要和所有人握手。               │
        │    窗口注意力像分组讨论——每组内部交流，组间不交流。                   │
        │                                                                      │
        │  为什么能降低复杂度?                                                 │
        │    全局: 196 个 patch 每个算 196 次 → 196×196 = 38,416              │
        │    窗口: 196 个 patch 每个算 49 次  → 196×49  = 9,604               │
        │    → 速度提升约 4 倍！                                               │
        │                                                                      │
        │    如果是 512×512 图像 → 1024 个 patch:                              │
        │    全局: 1024×1024 = 1,048,576                                       │
        │    窗口: 1024×49   = 50,176                                          │
        │    → 速度提升约 20 倍！                                              │
        └──────────────────────────────────────────────────────────────────────┘

        ┌──────────────────────────────────────────────────────────────────────┐
        │  Shifted Window —— 解决窗口注意力的"盲区"                           │
        │                                                                      │
        │  问题: 窗口 A 的猫耳朵 和 窗口 B 的猫尾巴 永远看不到彼此！           │
        │                                                                      │
        │  解决方案 (Swin Transformer 的核心创新):                             │
        │                                                                      │
        │    第 0, 2, 4, ... 层（偶数层）: 正常窗口划分                        │
        │      窗口边界: 0, 7, 14, 21...                                       │
        │      ┌───┬───┐                                                       │
        │      │ A │ B │   A 的 patch 只能看 A                                │
        │      ├───┼───┤   B 的 patch 只能看 B                                │
        │      │ C │ D │   C 和 D 同理                                       │
        │      └───┴───┘                                                       │
        │                                                                      │
        │    第 1, 3, 5, ... 层（奇数层）: 窗口偏移 window_size//2            │
        │      窗口边界: 3, 10, 17...（偏移了 3 个 patch）                    │
        │      ┌───┬───┐                                                       │
        │      │ A'│   │   A' 包含了原来 A 和 B 的部分 patch                  │
        │      ├───┤ B'│   这些 patch 现在可以在同一窗口内交互！               │
        │      │   ├───┤                                                      │
        │      │ C'│   │                                                       │
        │      └───┴───┘                                                       │
        │                                                                      │
        │  为什么这样有效?                                                     │
        │    经过两层后，任何相邻 patch 至少有一次机会在同一窗口内。            │
        │    信息可以像涟漪一样，通过"奇数层跨窗口 + 偶数层窗口内聚合"        │
        │    传遍整张图。                                                      │
        │                                                                      │
        │  类比:                                                               │
        │    正常窗口像固定座位的课堂讨论——只和同桌交流。                     │
        │    Shifted 窗口像每节课换座位——今天和后桌聊，明天和同桌聊，         │
        │    最终全班都认识了。                                                │
        │                                                                      │
        │  【技术细节】循环移位 + 区域 mask                                    │
        │                                                                      │
        │    Swin 不是真的把 patch 物理移动，而是用"循环坐标"重新划分窗口。   │
        │                                                                      │
        │    例如 shift_size=3, window_size=7, grid=14×14:                   │
        │                                                                      │
        │    位置 (0,0) 的移位后坐标 = ((0+3)%14, (0+3)%14) = (3,3)         │
        │    属于窗口 (3//7, 3//7) = (0,0)                                    │
        │                                                                      │
        │    位置 (13,13) 的移位后坐标 = ((13+3)%14, (13+3)%14) = (2,2)     │
        │    属于窗口 (2//7, 2//7) = (0,0)                                    │
        │                                                                      │
        │    看！位置 (0,0) 和 (13,13) 被分到了同一个移位窗口！               │
        │    但它们在原图中是对角，根本不相邻。                                │
        │                                                                      │
        │    解决办法: 区域 mask                                               │
        │    把每个移位窗口内的 patch 按原始位置分成最多 4 个区域:              │
        │      区域 0: row < h-shift_size, col < w-shift_size (原图左上)      │
        │      区域 1: row < h-shift_size, col >= w-shift_size (原图右上)     │
        │      区域 2: row >= h-shift_size, col < w-shift_size (原图左下)     │
        │      区域 3: row >= h-shift_size, col >= w-shift_size (原图右下)    │
        │                                                                      │
        │    只有同一原始区域的 patch 才能互相看！                             │
        │    这样 (0,0) 和 (13,13) 虽然在一个移位窗口，但区域不同，被遮住。   │
        └──────────────────────────────────────────────────────────────────────┘

        掩码构造逻辑:
        ─────────────────

        1. 无偏移 (shift_size=0): 正常窗口 mask
           mask[i,j] = True 当且仅当 i 和 j 属于同一窗口

        2. 有偏移 (shift_size>0): Shifted Window mask
           a. 计算每个位置的"循环移位后坐标"
              shifted_row = (row + shift_size) % h
              shifted_col = (col + shift_size) % w

           b. 用移位后坐标计算窗口索引
              win_row = shifted_row // window_size
              win_col = shifted_col // window_size

           c. 基础 mask: 同一移位窗口
              base_mask = (win_row[i] == win_row[j]) & (win_col[i] == win_col[j])

           d. 区域 mask: 同一原始区域（防止循环移位带来的虚假连接）
              region = (row >= h-shift_size) * 2 + (col >= w-shift_size)
              region_mask = (region[i] == region[j])

           e. 最终 mask = base_mask & region_mask
        """
        h, w = grid_size
        assert h * w == seq_len, f"grid_size {grid_size} 和 seq_len {seq_len} 不匹配"

        # 每个位置在 2D 网格中的行列坐标
        # row: [0,0,...,0, 1,1,...,1, ..., h-1,...,h-1]  每行 w 个，共 seq_len 个
        row = torch.arange(h, device=device).unsqueeze(1).expand(h, w).reshape(-1)
        # col: [0,1,...,w-1, 0,1,...,w-1, ..., 0,1,...,w-1] 重复 h 次
        col = torch.arange(w, device=device).unsqueeze(0).expand(h, w).reshape(-1)

        # ═══ 计算窗口索引 ═══
        if self.shift_size == 0:
            # 正常窗口: 直接用原始坐标
            win_row = row // self.window_size
            win_col = col // self.window_size
        else:
            # Shifted Window: 用循环移位后的坐标
            # (row + shift) % h 实现循环移位
            shifted_row = (row + self.shift_size) % h
            shifted_col = (col + self.shift_size) % w
            win_row = shifted_row // self.window_size
            win_col = shifted_col // self.window_size

        # mask[i, j] = True 当且仅当位置 i 和 j 属于同一窗口
        # unsqueeze 让两个 (seq_len,) 变成 (1, seq_len) 和 (seq_len, 1)，广播比较
        mask = (win_row.unsqueeze(0) == win_row.unsqueeze(1)) & \
               (win_col.unsqueeze(0) == win_col.unsqueeze(1))

        # ═══ Shifted Window 的区域 mask ═══
        if self.shift_size > 0:
            # 把每个位置按原始坐标分成 4 个区域
            # 区域分界线: h - shift_size 和 w - shift_size
            #
            #  原始图:
            #  ┌───────────────┬───────────────┐
            #  │  区域 0       │  区域 1       │
            #  │  (左上)       │  (右上)       │
            #  │  row < 边界   │  row < 边界   │
            #  │  col < 边界   │  col >= 边界  │
            #  ├───────────────┼───────────────┤
            #  │  区域 2       │  区域 3       │
            #  │  (左下)       │  (右下)       │
            #  │  row >= 边界  │  row >= 边界  │
            #  │  col < 边界   │  col >= 边界  │
            #  └───────────────┴───────────────┘
            #  其中 边界 = (h - shift_size) 或 (w - shift_size)

            region_row = (row >= h - self.shift_size).long()  # 0 or 1
            region_col = (col >= w - self.shift_size).long()  # 0 or 1
            region = region_row * 2 + region_col  # 0, 1, 2, 3

            # 只有同一原始区域的 patch 才能交互
            same_region = region.unsqueeze(0) == region.unsqueeze(1)
            mask = mask & same_region

        return mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        grid_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        ViT 注意力前向传播。

        参数:
            query:  (B, seq_len, d_model)  —— "提问者"
            key:    (B, seq_len, d_model)  —— "被查询者"
            value:  (B, seq_len, d_model)  —— "实际内容"
            mask:   可选的 attention mask。ViT 通常不传（双向无遮罩）。
                    如果传了，形状需兼容 SDPA 要求。

        返回:
            (B, seq_len, d_model) —— 融合了上下文信息的新表示

        ┌──────────────────────────────────────────────────────────────────────┐
        │  数值示例: 处理 2 张图，每张 196 个 patch                            │
        │                                                                      │
        │  Step 1: 线性投影                                                    │
        │    q = w_q(query):  (2, 196, 768) → (2, 196, 768)                   │
        │                                                                      │
        │  Step 2: 拆多头                                                      │
        │    q.view(2, 196, 12, 64).transpose(1, 2)                          │
        │      → (2, 12, 196, 64)                                             │
        │                                                                      │
        │    现在每个头独立处理 196 个 patch 的 64 维向量                     │
        │                                                                      │
        │  Step 3: F.scaled_dot_product_attention                              │
        │    输入: q=(2,12,196,64), k=(2,12,196,64), v=(2,12,196,64)         │
        │    输出: (2, 12, 196, 64)                                           │
        │                                                                      │
        │    内部发生了什么? (以 1 个头、3 个 patch 为例):                     │
        │                                                                      │
        │      q[0] = "背景"的查询向量                                         │
        │      q[1] = "猫耳朵"的查询向量                                       │
        │      q[2] = "猫尾巴"的查询向量                                       │
        │                                                                      │
        │      ① 打分: q @ k.T                                                │
        │         猫耳朵 和 猫耳朵 最相关 (分数高)                             │
        │         猫耳朵 和 猫尾巴 也有点相关 (同属一只猫)                     │
        │         猫耳朵 和 背景 不太相关                                      │
        │                                                                      │
        │      ② 缩放 + softmax → 概率                                        │
        │         [0.1,  0.6,  0.3]  ← "猫耳朵"关注分布                      │
        │                                                                      │
        │      ③ 加权混合 v                                                   │
        │         out[1] = 0.1×v[背景] + 0.6×v[猫耳朵] + 0.3×v[猫尾巴]      │
        │         → "猫耳朵"的新表示融合了"猫尾巴"的信息                      │
        │           模型知道"耳朵和尾巴属于同一只猫"                           │
        │                                                                      │
        │    关键点: 所有 patch 互相可见（没有 causal mask）                   │
        │             猫耳朵可以看猫尾巴，猫尾巴也可以看猫耳朵                 │
        │                                                                      │
        │  Step 4: 拼回 + 融合                                                 │
        │    transpose: (2, 12, 196, 64) → (2, 196, 12, 64)                  │
        │    view:      (2, 196, 12, 64) → (2, 196, 768)                     │
        │    w_o:       (2, 196, 768) → (2, 196, 768)                         │
        │                                                                      │
        │    12 个头的信息通过 w_o 融合，每个 patch 的最终表示                  │
        │    同时包含了"局部细节"和"全局上下文"。                              │
        └──────────────────────────────────────────────────────────────────────┘
        """
        batch_size = query.size(0)

        # ═══ Step 1: 线性投影 —— 把输入变成 Q/K/V ═══
        # 三个 Linear(768→768)，各自学习不同的语义空间
        q = self.w_q(query)   # (B, seq, 768)
        k = self.w_k(key)     # (B, seq, 768)
        v = self.w_v(value)   # (B, seq, 768)

        # ═══ Step 2: 拆成多头 ═══
        # 把 768 维拆成 (n_heads, d_k) = (12, 64)
        # view 不改变数据，只改变"看待方式"
        #
        # 形状变化:
        #   (B, seq, 768) → view → (B, seq, 12, 64)
        #   → transpose(1, 2) → (B, 12, seq, 64)
        #
        # 为什么 transpose?
        #   SDPA 期望 (B, n_heads, seq, d_k)，这样可以在 n_heads 维度并行
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # 现在都是 (B, n_heads, seq, d_k)

        # ═══ Step 3: 窗口 mask（可选）═══
        # 如果配置了 window_size 且提供了 grid_size，构造窗口注意力掩码
        if self.window_size > 0 and grid_size is not None:
            seq_len = query.size(1)  # 在拆多头前的序列长度
            window_mask = self._make_window_mask(seq_len, grid_size, query.device)
            # window_mask: (seq_len, seq_len) — 同一窗口内为 True
            #
            # 与外部传入的 mask 合并（如图文混合的 causal mask）
            if mask is not None:
                mask = mask & window_mask  # 两者都为 True 才可见
            else:
                mask = window_mask

        # ═══ Step 4: 主流注意力计算 —— SDPA ═══
        #
        # F.scaled_dot_product_attention 是 PyTorch 2.0+ 提供的官方函数。
        # 它内部自动完成: q @ k.T / sqrt(d_k) → softmax → dropout → @v
        #
        # 参数说明:
        #   query, key, value: (B, n_heads, seq, d_k)
        #   attn_mask:         可选。ViT 通常不传（双向无遮罩）。
        #   dropout_p:         训练时启用 dropout，推理时自动关闭。
        #   is_causal=False:   ViT 不需要因果掩码！这是和 LLM 的根本区别。
        #                      False 表示"所有位置互相可见"。
        #
        # 为什么 is_causal=False?
        #   图像 patch 没有"前后顺序"的限制——左上角的 patch 和右下角的 patch
        #   是同时被感知的，不需要"只能看前面的"。
        #
        # 自动后端选择 (PyTorch 自动判断):
        #   1. FlashAttention-2: 最快最省显存，需 Ampere GPU (A100/3090/4090 等)
        #      ⚠️ 注意: 如果传了非 causal 的 attn_mask，FlashAttention 会被跳过，
        #         自动 fallback 到 Memory-Efficient Attention。这仍然比手写实现快。
        #   2. Memory-Efficient Attention: 显存友好，速度中等
        #   3. Math: 纯 PyTorch 实现，兼容性最好（兜底）
        #
        # 你可以用以下代码查看实际用的后端:
        #   >>> with torch.backends.cuda.sdp_kernel():
        #   ...     print(torch.backends.cuda.flash_sdp_enabled())

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,  # ← ViT 的关键：双向注意力，没有因果限制
        )
        # out: (B, n_heads, seq, d_k)

        # ═══ Step 4: 拼回多头 + 融合 ═══
        # transpose 回去: (B, n_heads, seq, d_k) → (B, seq, n_heads, d_k)
        out = out.transpose(1, 2).contiguous()
        # view 拼回: (B, seq, n_heads, d_k) → (B, seq, d_model)
        out = out.view(batch_size, -1, self.d_model)

        # w_o: 把 12 个头的信息融合回统一的 768 维表示
        return self.w_o(out)
