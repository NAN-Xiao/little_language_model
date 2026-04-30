"""
Transformer — 顶层模型: 把"理解的向量"变成"下一个词是什么"
=============================================================

如果说 Decoder 是一个"阅读理解高手"，
那 Transformer 就是"阅读理解 + 猜词游戏"的完整玩法。

Decoder 做完 10 层思考后，每个词变成一个 768 维的"理解向量"。
但模型最终要回答的是:"下一个词应该是什么?"

Transformer 多做一步: 用一个 Linear(768→68) 把"理解"投影到"词表选择"。
就像考试做选择题: 你理解了文章(768维)，然后在 68 个选项里选一个。

完整流程:
  token_ids → Decoder(10层理解) → hidden(768维) → output_proj(768→68) → logits
  → softmax → 概率 → 采样 → 新 token_id

默认配置: d_model=768, n_heads=12, n_layers=10, vocab_size=68
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from .attention import KVCache
from .decoder import Decoder


class Transformer(nn.Module):
    """
    Decoder-Only Transformer —— 从"一句话"到"下一个词预测"的完整系统。

    ┌──────────────────────────────────────────────────────────────────┐
    │  生活类比: 你写作文时的思考过程                                  │
    │                                                                  │
    │  已经写了: "今天天气"                                           │
    │  脑子里( Decoder 10层思考后):                                    │
    │    "今天" → [0.2, -0.5, ...768个数...]  ← "我理解'今天'了"    │
    │    "天气" → [0.8,  0.3, ...768个数...]  ← "我理解'天气'了"    │
    │                                                                  │
    │  但 768 个数不是词，不能直接说出来!                              │
    │  需要"输出投影"就像查字典:                                       │
    │    768维理解 → 查 68 个词哪个最匹配 → "真"的分数最高!         │
    │                                                                  │
    │  所以 "今天天气" 后面最可能接 "真"，组成 "今天天气真"         │
    └──────────────────────────────────────────────────────────────────┘

    参数 cfg (ModelConfig):
        vocab_size=68:  词表大小，就像字典有 68 个字
        d_model=768:    每个字的"理解深度"，768个数字的向量
        n_heads=12:     注意力头数
        n_decoder_layers=10: DecoderBlock 层数
        max_seq_len=256: 最大句子长度
    """

    def __init__(self, cfg: ModelConfig):
        """cfg: 模型配置，包含所有超参数 (词表大小、维度、层数等)。"""
        super().__init__()
        self.cfg = cfg  # 保存配置，后续生成时要用 (如 eos_token_id, max_seq_len)

        # Decoder: 10 层深度理解，输入输出都是 (B, seq, 768)
        self.decoder = Decoder(
            vocab_size=cfg.vocab_size,      # 词表大小 (68)
            d_model=cfg.d_model,            # 向量维度 (768)
            n_heads=cfg.n_heads,            # 注意力头数 (12)
            n_layers=cfg.n_decoder_layers,  # 层数 (10)
            d_ff=cfg.d_ff,                  # FFN 中间维度 (3072=768×4)
            max_seq_len=cfg.max_seq_len,    # 最大序列长度 (256)
            dropout=cfg.dropout,            # 随机丢弃比例 (0.1)
            pad_token_id=cfg.pad_token_id,  # 填充标记 id (0)
            use_moe=cfg.use_moe,            # 是否用 MoE (默认False)
            moe_num_experts=cfg.moe_num_experts,  # MoE 专家数 (默认4)
            use_rope=cfg.use_rope,          # 是否用 RoPE 位置编码 (默认False)
            use_rmsnorm=cfg.use_rmsnorm,    # 是否用 RMSNorm (默认True)
            use_swiglu=cfg.use_swiglu,      # 是否用 SwiGLU (默认True)
            n_kv_heads=cfg.n_kv_heads,      # None=MHA, 3=GQA, 1=MQA
        )

        # output_proj: 把"理解向量"映射到"词表选择"
        # Linear(768 → 68): 就像从 768 维的"内心活动"压缩成 68 个选项的"选择题分数"
        # 每个词输出 68 个分数，分数最高的就是模型认为最可能的下一个词
        self.output_proj = nn.Linear(cfg.d_model, cfg.vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化 —— 让神经网络起跑线公平。

        为什么要专门初始化?
          默认随机初始化可能让某些层的输出特别大或特别小，
          导致梯度爆炸或消失，模型训练不动。

        Xavier 的思想:
          让每层输出的方差 ≈ 输入的方差，信号不会越传越弱/越强。
          就像传话游戏，每个传话人都要保证音量一致，最后一个人才能听清。
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_causal_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        构造因果掩码 —— "只能看过去，不能偷看未来"。

        用具体例子理解 (B=2, seq=5, vocab=68):

        ┌──────────────────────────────────────────────────────────────────┐
        │  输入 input_ids:                                                 │
        │    样本0: [3, 12, 45,  7,  2]  ← "今天天气真好"                 │
        │    样本1: [8,  3, 12,  0,  0]  ← "我喜欢__" (后2个是pad)      │
        │                                                                  │
        │  掩码有两层含义，用 AND(与) 合并:                               │
        │                                                                  │
        │  ① pad_mask —— 忽略填充位 (id=0 的位置是假的)                  │
        │    形状: (2, 1, 5)                                               │
        │    样本0: [[T, T, T, T, T]]     ← 全是真词                     │
        │    样本1: [[T, T, T, F, F]]     ← 后两个是填充，忽略           │
        │                                                                  │
        │  ② causal_mask —— 因果约束 (下三角)                            │
        │    形状: (1, 5, 5)                                               │
        │         位0  位1  位2  位3  位4                                  │
        │    位0 [ T,  F,  F,  F,  F]  ← "今"只能看"今"                 │
        │    位1 [ T,  T,  F,  F,  F]  ← "天"能看"今、天"               │
        │    位2 [ T,  T,  T,  F,  F]  ← "气"能看"今、天、气"          │
        │    位3 [ T,  T,  T,  T,  F]  ← "真"能看前4个                  │
        │    位4 [ T,  T,  T,  T,  T]  ← "好"能看全部                   │
        │                                                                  │
        │    T=True(能看到), F=False(遮掉)                               │
        │    F 的位置在注意力分数里会被填成 -inf，softmax 后变 0         │
        │                                                                  │
        │  ③ final = pad_mask & causal_mask:                             │
        │    形状: (2, 5, 5)                                               │
        │    样本0 (无pad): 就是纯下三角                                  │
        │    样本1 (有pad): 下三角 + pad 行全 F                           │
        │         位0  位1  位2  位3  位4                                  │
        │    位0 [ T,  F,  F,  F,  F]                                     │
        │    位1 [ T,  T,  F,  F,  F]                                     │
        │    位2 [ T,  T,  T,  F,  F]                                     │
        │    位3 [ F,  F,  F,  F,  F]  ← pad位置，谁都不看               │
        │    位4 [ F,  F,  F,  F,  F]  ← pad位置，谁都不看               │
        │                                                                  │
        │    为什么 pad 也要遮?                                          │
        │      pad 不是真实文字，是填充占位符。                           │
        │      如果不遮，模型会从 pad 学东西，就乱套了。                  │
        └──────────────────────────────────────────────────────────────────┘
        """
        _, seq_len = input_ids.shape  # 序列长度 (如 5)

        # pad_mask: 找出哪些位置不是填充 (id != 0)
        # (B, seq) → (B, 1, seq)，中间加一维以便和 causal_mask 广播
        pad_mask = (input_ids != self.cfg.pad_token_id).unsqueeze(1)

        # causal_mask: 下三角矩阵，True 表示"能看到"
        # torch.tril: 取下三角 (对角线及下方为 1，上方为 0)
        # (seq, seq) → (1, seq, seq)，第 0 维为 1 以便广播到所有样本
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len,
                       device=input_ids.device, dtype=torch.bool)
        ).unsqueeze(0)

        # 合并: 两者都是 True 才算 True (既非 pad，又在因果范围内)
        return pad_mask & causal_mask

    def forward(
        self,
        input_ids: torch.Tensor,      # 输入 token id: (B, seq)，如 [[3,12,45,7,2]]
        kv_cache: KVCache | None = None,  # KV-Cache: 推理时缓存历史 K/V，训练时为 None
        rope_offset: int = 0,         # RoPE 旋转偏移: 新 token 在序列中的位置
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        """
        前向传播 —— 从 token id 到"每个位置下一个词的分数"。

        ┌──────────────────────────────────────────────────────────────────┐
        │  维度流转 (训练时, B=2, seq=5, 默认配置):                       │
        │                                                                  │
        │  ① 输入                                                        │
        │     input_ids: (2, 5)                                          │
        │       [[ 3, 12, 45,  7,  2],                                    │
        │        [ 8,  3, 12,  0,  0]]                                    │
        │       样本0: "今天天气真好" (5个token)                          │
        │       样本1: "我喜欢__" (3个token + 2个pad)                   │
        │                                                                  │
        │  ② 构造掩码                                                    │
        │     tgt_mask: (2, 5, 5)                                        │
        │       因果掩码 + pad掩码 (见 make_causal_mask 注释)            │
        │                                                                  │
        │  ③ Decoder 深度理解                                            │
        │     hidden, kv_cache = self.decoder(...)                       │
        │     hidden: (2, 5, 768)                                        │
        │       每个 token 变成融合了上下文的 768 维向量                 │
        │       "天"的向量里混合了"今"的信息                             │
        │       "好"的向量里混合了"今天天气真"的全部信息                 │
        │                                                                  │
        │  ④ 输出投影 —— "理解→选择"                                    │
        │     logits = self.output_proj(hidden)                          │
        │     logits: (2, 5, 68)                                         │
        │                                                                  │
        │     这是什么意思?                                              │
        │       位置0 ("今"): 68个分数 → 预测位置1应该是什么词          │
        │       位置1 ("天"): 68个分数 → 预测位置2应该是什么词          │
        │       ...                                                      │
        │       位置4 ("好"): 68个分数 → 预测位置5应该是什么词          │
        │                                                                  │
        │     为什么每个位置都预测"下一个词"?                            │
        │       因为训练时我们知道正确答案，可以并行计算所有位置的 loss  │
        │       "今"→"天", "天"→"气", "气"→"真"... 同时学    │
        │                                                                  │
        │  ⑤ 训练时: logits → CrossEntropyLoss                           │
        │       把 68 维分数和真实标签比较，算 loss，反向传播             │
        │                                                                  │
        │  ⑥ 生成时: logits → softmax → 采样                             │
        │       只看最后一个位置的分数，选概率最高的词                   │
        │       (因为最后一个位置已经"看完"了整句话)                     │
        └──────────────────────────────────────────────────────────────────┘
        """
        # 构造因果掩码: 每个词只能看自己及之前的词，形状 (B, seq, seq)
        tgt_mask = self.make_causal_mask(input_ids)

        # Decoder: 10层深度理解，输出 hidden (B, seq, 768)
        # 每个 token 的 768 维向量已经融合了所有它能看到的上下文
        hidden, new_kv_cache = self.decoder(
            input_ids, tgt_mask=tgt_mask,
            rope_offset=rope_offset, kv_cache=kv_cache,
        )

        # output_proj: (B, seq, 768) → (B, seq, vocab_size=68)
        # 把每个位置的"理解向量"变成"68个候选词的分数"
        logits = self.output_proj(hidden)

        # 生成模式 (kv_cache 不为 None): 同时返回新的缓存，用于下一步推理
        if kv_cache is not None:
            return logits, new_kv_cache
        return logits

    @torch.no_grad()
    def generate(
        self,
        # 起始 token，如 "今天" 的 [3, 12]，形状 (B, prompt_len)
        input_ids: torch.Tensor,
        max_len: int = 128,           # 最多生成多少个字 (含 prompt)
        eos_token_id: int | None = None,  # 结束标记 id，生成这个就停 (如 0)
        min_new_tokens: int = 0,      # 至少生成多少字 (0=不限制，防止过早输出 eos)
        repetition_penalty: float = 1.0,  # 重复惩罚 (>1 降低已出现词的概率)
        no_repeat_ngram_size: int = 0,    # 禁止重复 n-gram 长度 (0=不禁止)
        temperature: float = 1.0,     # 温度: >1 更随机有创意，<1 更保守确定
        top_k: int = 0,              # 只从概率最高的 k 个词里选 (0=关闭，用全部)
        top_p: float = 1.0,          # 核采样: 从累计概率达 p 的最小集合选 (1=关闭)
    ) -> torch.Tensor:
        """
        自回归文本生成 —— 像人类写作文一样，一字一字地"想"出来。

        ┌──────────────────────────────────────────────────────────────────┐
        │  核心思想: 每次只预测"下一个字"，然后把这个字拼回去继续预测    │
        │                                                                  │
        │  用具体例子: 输入 "今天" → 希望输出 "今天天气真好"            │
        │                                                                  │
        │  初始: generated = [3, 12]  ← "今天"                          │
        │                                                                  │
        │  step 0:                                                         │
        │    输入 [3, 12] → Transformer → logits (1, 2, 68)              │
        │    只看最后一个位置 logits[:, -1, :] = 68维分数                │
        │      → "天"这个位置已经看完了"今天"，预测下一个词             │
        │    分数排序: "气"=8.5, "是"=6.2, "的"=4.1, ...               │
        │    采样 → token 45 ("气")                                      │
        │    generated = [3, 12, 45]  ← "今天气"                        │
        │                                                                  │
        │  step 1:                                                         │
        │    输入 [3, 12, 45] → Transformer → logits                     │
        │    最后一个位置("气")看完了"今天气"                           │
        │    采样 → token 7 ("真")                                       │
        │    generated = [3, 12, 45, 7]  ← "今天气真"                   │
        │                                                                  │
        │  step 2:                                                         │
        │    采样 → token 2 ("好")                                       │
        │    generated = [3, 12, 45, 7, 2]  ← "今天气真好"              │
        │                                                                  │
        │  step 3:                                                         │
        │    采样 → token 0 (eos, 结束符)                                │
        │    检测到 eos → 停止生成                                       │
        │                                                                  │
        │  输出: [3, 12, 45, 7, 2, 0]                                    │
        └──────────────────────────────────────────────────────────────────┘

        ┌──────────────────────────────────────────────────────────────────┐
        │  KV-Cache 推理加速 (关键优化!)                                  │
        │                                                                  │
        │  不用 KV-Cache 的问题:                                          │
        │    step 1 输入 [3,12,45] → attention 要算 3×3 的矩阵           │
        │    step 2 输入 [3,12,45,7] → attention 要算 4×4 的矩阵         │
        │    step N → 算 N×N，越来越慢!                                  │
        │                                                                  │
        │  用 KV-Cache:                                                   │
        │    第1步: 输入完整 prompt [3,12] → 算 2×2，同时缓存 K,V       │
        │            K_cache=[k0,k1], V_cache=[v0,v1]                    │
        │                                                                  │
        │    第2步: 只输入新 token [45] → 只算 k2,v2                     │
        │            K=[k0,k1,k2], V=[v0,v1,v2] (缓存+新的)              │
        │            attention: Q只有1个，K/V有3个 → 1×3 矩阵            │
        │            而不是 3×3!                                         │
        │                                                                  │
        │    第3步: 只输入 [7] → attention 1×4                           │
        │                                                                  │
        │    复杂度从 O(N²) → O(N)，长文本生成快 10-100 倍              │
        │                                                                  │
        │    为什么可以只输入新 token?                                    │
        │      因为之前的 K,V 已经缓存了，新 token 的注意力只需要         │
        │      "新 Q" 和 "全部 K/V(缓存+新)" 算一次就行                  │
        │      因果性由掩码保证: 新 token 本来就可以看所有之前的 token    │
        └──────────────────────────────────────────────────────────────────┘

        参数说明:
            input_ids:  起始 token，如 "今天" 的 [3, 12]
            max_len:    最多生成多少个字
            eos_token_id: 结束标记，生成这个就停
            min_new_tokens: 至少生成多少字(防止过早结束)
            temperature: 温度，>1 更随机(有创意)，<1 更保守(确定)
            top_k:      只从概率最高的 k 个词里选 (0=关闭)
            top_p:      只从累计概率达 p 的最小词集里选 (1=关闭)
            repetition_penalty: 重复惩罚，>1 降低已出现词的概率
            no_repeat_ngram_size: 禁止重复的 n-gram 长度
        """
        eos_token_id = eos_token_id or self.cfg.eos_token_id  # 没传就用配置里的
        self.eval()  # 评估模式: 关闭 dropout，行为确定

        generated = input_ids.clone()           # 已生成的序列，逐步追加新 token
        batch_size = generated.size(0)          # B: 几个样本同时生成
        # finished: 每个样本是否已结束 (生成了 eos)，初始全 False
        finished = torch.zeros(
            batch_size, dtype=torch.bool, device=generated.device)
        kv_cache: KVCache | None = None         # 初始无缓存，第1步建立
        first_step = True                       # 标记是否是第1步 (处理完整 prompt)

        for step_idx in range(max_len):
            # 序列长度保护: 不能超过模型最大长度
            if generated.size(1) >= self.cfg.max_seq_len:
                break

            if first_step:
                # ═══ 第1步: 处理完整 prompt，建立 KV-Cache ═══
                step_input = generated            # 完整输入: (B, prompt_len)
                tgt_mask = self.make_causal_mask(step_input)
                rope_offset = 0                   # 第1步从位置 0 开始
                # 传空列表让 decoder 返回 kv_cache (训练模式传 None)
                kv_cache_arg: KVCache | None = []
                first_step = False
            else:
                # ═══ 后续步: 只输入最后1个新 token，复用缓存 ═══
                # generated[:, -1:] 取最后一列，形状 (B, 1)，只有1个新 token
                step_input = generated[:, -1:]

                # mask: 新 token 可以看所有已生成的 token (seq_k 个)
                # 因果性已由 KV-Cache 保证(缓存里只有历史，没有未来)
                seq_k = generated.size(1)
                tgt_mask = torch.ones(
                    batch_size, 1, seq_k,
                    dtype=torch.bool, device=generated.device,
                )
                # rope_offset: 新 token 在完整序列中的位置，用于 RoPE 旋转
                rope_offset = generated.size(1) - 1
                kv_cache_arg = kv_cache           # 复用之前缓存的 K/V

            # 前向传播，返回 logits (B, seq, 68) 和更新后的缓存
            logits, kv_cache = self(
                step_input, kv_cache=kv_cache_arg, rope_offset=rope_offset)

            # 只取最后一个位置的分数: (B, seq, 68) → (B, 68)
            # 最后一个位置已经"看完"了所有历史，最适合预测下一个
            logits = logits[:, -1, :]

            # repetition_penalty: 降低已出现词的重复概率
            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    if finished[b]:
                        continue
                    for token_id in generated[b].unique():
                        token_id = int(token_id)
                        if logits[b, token_id] > 0:
                            logits[b, token_id] /= repetition_penalty
                        else:
                            logits[b, token_id] *= repetition_penalty

            # no_repeat_ngram_size: 禁止重复 n-gram
            if no_repeat_ngram_size > 0:
                n = no_repeat_ngram_size
                for b in range(batch_size):
                    if finished[b]:
                        continue
                    seq = generated[b].tolist()
                    banned = set()
                    prefix = seq[-(n - 1):] if len(seq) >= n - 1 else seq
                    limit = len(seq) - (n - 1)
                    for i in range(max(0, limit)):
                        if seq[i:i + (n - 1)] == prefix:
                            banned.add(seq[i + (n - 1)])
                    if banned:
                        logits[b, list(banned)] = float("-inf")

            # top_k: 固定数量筛选。只保留分数最高的 k 个词，其余强制排除。
            # 实现：找到第 k 高的分数作为阈值，低于它的全部设为 -inf（softmax 后概率为 0）
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[:, -1:]] = float("-inf")

            # top_p (nucleus sampling): 动态数量筛选。按概率从高到低累加，
            # 取累计概率达到 p 的最小词集，其余排除。比 top_k 更自适应。
            # 例如 top_p=0.9: 若前3个词概率和已达0.9，则只留这3个；若很分散则可能留30个。
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # 累计概率超过 top_p 的位置（及之后的所有词）设为 -inf
                sorted_mask = (cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p)
                sorted_logits[sorted_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            # softmax: 将 logits 转换为概率分布。此前被设为 -inf 的词，e^(-inf)=0，概率为 0。
            probs = F.softmax(logits, dim=-1)

            # 多项式采样：按概率加权随机抽取一个词，不是直接取 argmax。
            # 这样高分词大概率被选中，但低分词也有小概率"爆冷"。
            next_token = torch.multinomial(probs, num_samples=1)

            # 已完成的序列继续输出 pad (而不是 eos)，保持 batch 形状一致

            next_token = next_token.masked_fill(
                finished.unsqueeze(1), self.cfg.pad_token_id
            )

            # 拼到序列末尾: (B, seq+1)
            # cat的函数原型是 torch.cat(tensors, dim=0)，
            # 其中 tensors 是一个张量列表，dim 是要连接的维度。
            generated = torch.cat([generated, next_token], dim=1)

            # 检查是否生成了 eos: 更新 finished 标记
            finished = finished | (next_token.squeeze(1) == eos_token_id)
            if finished.all():    # 所有样本都结束了，提前退出
                break
        #generated是token id 的序列，形状 (B, seq)，包含了输入的 prompt 和生成的新 token。
        return generated
