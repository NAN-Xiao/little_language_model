"""
Transformer — 顶层模型: 解码器 + 输出投影
=========================================

把 token 序列变成概率分布, 支持训练和自回归生成。

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
    Decoder-Only Transformer — 从 token id 到 logits。

    ┌──────────────────────────────────────────────────────────────────┐
    │  训练流程 (B=2, seq=5, vocab=68):                               │
    │                                                                  │
    │  ① 输入: token_ids (2, 5) — 如 [[3,12,45,7,2], [8,3,12,0,0]]  │
    │                                                                  │
    │  ② 构造因果掩码:                                                │
    │     pad_mask: (2,1,5) — pad位置(id=0)为False, 其余True         │
    │       [[T,T,T,T,T], [T,T,T,F,F]]                               │
    │     causal_mask: (1,5,5) — 下三角, 只看过去                    │
    │       [[T,F,F,F,F],                                              │
    │        [T,T,F,F,F],                                              │
    │        [T,T,T,F,F],                                              │
    │        [T,T,T,T,F],                                              │
    │        [T,T,T,T,T]]                                              │
    │     final: (2,5,5) — 两者 AND, pad位置也mask掉                  │
    │                                                                  │
    │  ③ Decoder:                                                     │
    │     token_ids → Embedding → 位置编码 → 10层DecoderBlock → Norm │
    │     → hidden (2, 5, 768)                                        │
    │                                                                  │
    │  ④ 输出投影: Linear(768 → 68)                                   │
    │     (2, 5, 768) → (2, 5, 68)                                    │
    │     每个位置的 68 维向量就是词表里 68 个 token 的分数 (logits) │
    │                                                                  │
    │  ⑤ 训练时: logits → CrossEntropy → loss                         │
    │     预测每个位置的下一个 token, 和真实标签比较                  │
    │     位置0预测位置1的token, 位置1预测位置2, ...                  │
    │                                                                  │
    ├──────────────────────────────────────────────────────────────────┤
    │  生成流程 (自回归, 逐步预测):                                   │
    │                                                                  │
    │  输入: "今天" → [3, 12]                                          │
    │                                                                  │
    │  step 0:                                                         │
    │    [3, 12] → Transformer → logits[:,-1,:] = 68维概率            │
    │    → 采样得 token 45 → "今天天"                                 │
    │                                                                  │
    │  step 1:                                                         │
    │    [3, 12, 45] → Transformer → 采样得 7 → "今天天气"            │
    │                                                                  │
    │  step 2:                                                         │
    │    [3, 12, 45, 7] → 采样得 2(eos) → 停止                      │
    │                                                                  │
    │  每次只取最后一个位置的 logits 来预测下一个 token               │
    │  因为因果掩码, 最后一个位置已经看到了前面所有的 token          │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.decoder = Decoder(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_decoder_layers,
            d_ff=cfg.d_ff,
            max_seq_len=cfg.max_seq_len,
            dropout=cfg.dropout,
            pad_token_id=cfg.pad_token_id,
            use_moe=cfg.use_moe,
            moe_num_experts=cfg.moe_num_experts,
            use_rope=cfg.use_rope,
        )

        # (768 → 68) 把隐藏状态投影回词表大小
        self.output_proj = nn.Linear(cfg.d_model, cfg.vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化 — 让各层的输出方差大致一致, 训练更稳定。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_causal_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        构造因果掩码: 每个 token 只看自己和之前的, 并屏蔽 pad。

        ┌──────────────────────────────────────────────────────────────┐
        │  input_ids: (2, 5) = [[3,12,45,7,2], [8,3,12,0,0]]         │
        │    第2个样本末尾有2个 pad (id=0)                             │
        │                                                              │
        │  pad_mask: (2, 1, 5)                                         │
        │    [[T, T, T, T, T],    ← 第1个样本没有 pad                 │
        │     [T, T, T, F, F]]   ← 第2个样本后2个是 pad               │
        │                                                              │
        │  causal_mask: (1, 5, 5) — 下三角                            │
        │    [[T,F,F,F,F],                                              │
        │     [T,T,F,F,F],                                              │
        │     [T,T,T,F,F],                                              │
        │     [T,T,T,T,F],                                              │
        │     [T,T,T,T,T]]                                              │
        │                                                              │
        │  final: pad_mask & causal_mask = (2, 5, 5)                   │
        │    第1个样本: 纯因果, 没有pad                                │
        │    第2个样本: 因果 + pad位置全False                          │
        │      [[T,F,F,F,F],                                           │
        │       [T,T,F,F,F],                                           │
        │       [T,T,T,F,F],  ← 第3行: 可以看位置0,1,2, 但3,4是pad  │
        │       [F,F,F,F,F],  ← pad位置: 谁都不能看                  │
        │       [F,F,F,F,F]]                                           │
        └──────────────────────────────────────────────────────────────┘
        """
        _, seq_len = input_ids.shape
        pad_mask = (input_ids != self.cfg.pad_token_id).unsqueeze(1)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool)
        ).unsqueeze(0)
        return pad_mask & causal_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache | None = None,
        rope_offset: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        """
        (B, seq) → (B, seq, vocab_size)
        当 kv_cache 不为 None 时, 返回 (logits, new_kv_cache)
        """
        tgt_mask = self.make_causal_mask(input_ids)
        hidden, new_kv_cache = self.decoder(
            input_ids, tgt_mask=tgt_mask,
            rope_offset=rope_offset, kv_cache=kv_cache,
        )
        logits = self.output_proj(hidden)
        if kv_cache is not None:
            return logits, new_kv_cache
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_len: int = 128,
        eos_token_id: int | None = None,
        min_new_tokens: int = 0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        自回归文本生成 — 逐步预测下一个 token, 拼接到序列末尾, 直到 eos 或 max_len。
        使用 KV-Cache: 第1步处理完整 prompt, 之后每步只处理 1 个新 token。

        ┌──────────────────────────────────────────────────────────────┐
        │  KV-Cache 生成流程:                                          │
        │                                                              │
        │  第1步: 输入完整 prompt [3,12,45,7] → forward → logits     │
        │         同时缓存每层的 K,V → kv_cache 建立起来             │
        │                                                              │
        │  第2步: 只输入新 token [45] → forward(kv_cache) → logits   │
        │         Q 只有1个token, K/V = 缓存+新 = [旧4个+新1个]     │
        │         注意力: 1×5 而非 5×5, 省了 80% 计算                │
        │                                                              │
        │  第3步: 只输入 [7] → K/V = [旧5个+新1个] = 6个             │
        │         注意力: 1×6, 仍然只需算1行                          │
        │                                                              │
        │  → 复杂度从 O(N²) 降到 O(N), 长序列生成大幅加速           │
        └──────────────────────────────────────────────────────────────┘
        """
        eos_token_id = eos_token_id or self.cfg.eos_token_id
        self.eval()

        generated = input_ids.clone()
        batch_size = generated.size(0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)
        kv_cache: KVCache | None = None
        first_step = True

        for step_idx in range(max_len):
            # 序列长度保护: 超过 max_seq_len 就停止
            if generated.size(1) >= self.cfg.max_seq_len:
                break

            if first_step:
                # 第1步: 处理完整 prompt, 建立缓存
                # 传空列表而非None, 让forward返回(logits, kv_cache)元组
                step_input = generated
                tgt_mask = self.make_causal_mask(step_input)
                rope_offset = 0
                kv_cache_arg: KVCache | None = []
                first_step = False
            else:
                # 后续步: 只输入新 token, 使用缓存
                step_input = generated[:, -1:]
                # 新 token 可以看所有已生成的 token (因果性已由缓存保证)
                seq_k = generated.size(1)
                tgt_mask = torch.ones(
                    batch_size, 1, seq_k,
                    dtype=torch.bool, device=generated.device,
                )
                rope_offset = generated.size(1) - 1
                kv_cache_arg = kv_cache

            # 带缓存的 forward
            logits, kv_cache = self(step_input, kv_cache=kv_cache_arg, rope_offset=rope_offset)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if step_idx < min_new_tokens:
                logits[:, eos_token_id] = float("-inf")

            if repetition_penalty and repetition_penalty != 1.0:
                token_ids = generated
                gathered = logits.gather(1, token_ids)
                adjusted = torch.where(
                    gathered < 0,
                    gathered * repetition_penalty,
                    gathered / repetition_penalty,
                )
                logits.scatter_(1, token_ids, adjusted)

            if no_repeat_ngram_size and no_repeat_ngram_size > 1:
                n = int(no_repeat_ngram_size)
                if generated.size(1) >= n - 1:
                    prefix = generated[:, -(n - 1) :].tolist()
                    full = generated.tolist()
                    for b in range(batch_size):
                        if finished[b]:
                            continue
                        banned: set[int] = set()
                        seq = full[b]
                        pre = prefix[b]
                        limit = len(seq) - (n - 1)
                        for i in range(max(0, limit)):
                            if seq[i : i + (n - 1)] == pre:
                                banned.add(seq[i + (n - 1)])
                        if banned:
                            logits[b, list(banned)] = float("-inf")

            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[:, -1:]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_mask = (
                    cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                )
                sorted_logits[sorted_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = next_token.masked_fill(
                finished.unsqueeze(1), self.cfg.pad_token_id
            )
            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == eos_token_id)
            if finished.all():
                break

        return generated
