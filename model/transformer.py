from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from .decoder import Decoder

# =====================================================================
# Transformer 解码器模块
# 这是一个“仅解码器（decoder-only）”的 Transformer——只能左到右地生成文本，
# 用于自回归语言建模任务，是 GPT、LLaMA、ChatGLM 等架构的基础。
# =====================================================================
class Transformer(nn.Module):
    """
    Decoder-only Transformer for autoregressive language modeling.

    这是一个典型的 GPT 类模型的实现核心——只包含解码器部分（没有编码器）。
    支持普通和 MoE（Mixture of Experts）结构。
    """

    def __init__(self, cfg: ModelConfig):
        """
        初始化 Transformer 模型（构建网络结构）

        Args:
            cfg (ModelConfig): 包含模型超参数的配置对象
        """
        super().__init__()
        self.cfg = cfg  # 保存配置参数，便于后续调用

        # 构造实际的解码器堆叠
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

        # 输出投影层: 把 d_model 的隐藏状态投影回词表大小（产生 token 概率分布）
        self.output_proj = nn.Linear(cfg.d_model, cfg.vocab_size)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """
        权重初始化方法：对所有权重参数（维度大于1的参数）用Xavier均匀分布初始化

        实际意义：
        Xavier初始化可以帮助深层神经网络更容易收敛，防止梯度消失/爆炸。
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_causal_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        构造自回归（causal）mask，使得每个位置只能看到自己以及之前的 token，
        并结合 pad_mask 实现“正确遮挡”。
        """
        _, seq_len = input_ids.shape  # batch_size, seq_len
        # pad_mask: mask 对应到 pad_token 的地方为 False，其余为 True
        pad_mask = (input_ids != self.cfg.pad_token_id).unsqueeze(1)
        # causal_mask: 下三角全1（只能看见自己和之前的 token）
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool)
        ).unsqueeze(0)
        # 两者与运算，获得 final mask
        return pad_mask & causal_mask

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播（即模型推理/训练时的主函数）。

        Args:
            input_ids (torch.Tensor): 输入的 token id （形状：[batch, seq_len]）

        Returns:
            torch.Tensor: 输出：每个 token 的 logits（[batch, seq_len, vocab_size]）

        步骤详细注释如下：
        """
        # ========= 1. 构造掩码（causal mask） =========
        # 目的：确保自回归语言模型不能“偷看”当前位置之后的 token
        # 原理：只允许当前位置访问自己和之前的 token，等价于下三角全1的 attention mask
        # 同时结合 pad_mask，把 pad 的位置置 0
        tgt_mask = self.make_causal_mask(input_ids)

        # ========= 2. 送入解码器堆叠 =========
        # 目的：特征提取与上下文交互
        # 过程：多层 Transformer 解码器叠加（每层含多头自注意力和前馈网络）
        # mask 会传递给解码器的自注意力，让它只能注意到历史 token
        hidden = self.decoder(input_ids, tgt_mask=tgt_mask)

        # ========= 3. 输出层投影（到词表）=========
        # 目的：把每个时刻的隐藏状态投影为词表大小的 logits
        # （每个位置预测下一个 token 的概率分布）
        logits = self.output_proj(hidden)

        # ========= 4. 返回 logits =========
        # 作用：供训练时计算损失，或推理时 softmax 采样生成 token
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
        文本生成函数（多种 sampling 策略支持），batch 版。
        
        支持常用的采样方法与解码约束，适用于推理。
        """
        eos_token_id = eos_token_id or self.cfg.eos_token_id
        self.eval()  # 切到评估模式

        # 初始化已生成序列，复制输入（不能直接 inplace 写入）
        generated = input_ids.clone()
        batch_size = generated.size(0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)

        for step_idx in range(max_len):
            # 预测下一个 token 的分布
            logits = self(generated)[:, -1, :] / max(temperature, 1e-5)
            # 前 min_new_tokens 步不允许生成 <eos>
            if step_idx < min_new_tokens:
                logits[:, eos_token_id] = float("-inf")

            # 重复惩罚（提升生成的多样性）
            if repetition_penalty and repetition_penalty != 1.0:
                token_ids = generated
                gathered = logits.gather(1, token_ids)
                adjusted = torch.where(
                    gathered < 0,
                    gathered * repetition_penalty,
                    gathered / repetition_penalty,
                )
                logits.scatter_(1, token_ids, adjusted)

            # ngram 重复惩罚（防止生成重复片段）
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

            # Top-k 策略：仅保留 top_k 概率最大的 token
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[:, -1:]] = float("-inf")

            # Top-p 策略（nucleus sampling）
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

            # 采样下一个 token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # 结束时填充 pad token
            next_token = next_token.masked_fill(
                finished.unsqueeze(1), self.cfg.pad_token_id
            )
            generated = torch.cat([generated, next_token], dim=1)
            # 检查是否已经遇到 <eos>
            finished = finished | (next_token.squeeze(1) == eos_token_id)
            if finished.all():
                break

        return generated
