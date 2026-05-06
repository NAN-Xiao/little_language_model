"""
VideoAudioLanguageModel —— 音视频多模态理解模型
====================================================

输入一段带声音的视频，输出文字描述（如"一只猫在喵喵叫"）。

架构: 视频编码器 + 音频编码器 + 模态融合 + 文本 Decoder（自回归生成）

┌─────────────────────────────────────────────────────────────────────────────┐
│  整体流程                                                                    │
│                                                                             │
│  视频 (B, 3, T, H, W)     音频 (B, n_mels, T_a)      文本 token ids (B, L)  │
│      │                         │                         │                  │
│      ▼                         ▼                         ▼                  │
│  VideoPatchEmbed3D         AudioPatchEmbed           Text Embedding         │
│  (B, N_v, d_model)         (B, N_a, d_model)         (B, L, d_model)        │
│      │                         │                         │                  │
│      └───────────┬─────────────┘                         │                  │
│                  ▼                                       │                  │
│           + Modality Embedding                           │                  │
│                  │                                       │                  │
│                  └──────────────────┬────────────────────┘                  │
│                                     ▼                                       │
│                              拼接序列                                       │
│                  [VID] video_tokens [AUD] audio_tokens text_tokens          │
│                              ↓                                              │
│                        + 位置编码                                           │
│                              ↓                                              │
│                        因果 Mask                                            │
│                              ↓                                              │
│                        Decoder Blocks × N                                   │
│                              ↓                                              │
│                        Output Projection                                    │
│                              ↓                                              │
│                        logits → 文字                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

关键设计
────────
1. 视频用 3D PatchEmbedding（时间×空间），捕捉动作时序
2. 音频用 2D PatchEmbedding（频率×时间），捕捉音频频谱特征
3. 模态类型嵌入（Modality Embedding）：让模型知道每个 token 来自视频/音频/文本
4. 特殊 token [VID]/[AUD] 标记各模态的起始位置
5. 因果 Mask：视频/音频 token 之间全双向可见，文本只能因果看自己，但能看到所有音视频

与图像多模态的区别
───────────────────
图像多模态: 只有 [img0..imgN] + [txt0..txtM]
音视频多模态: [VID][vid0..vidN][AUD][aud0..audM][txt0..txtL]
            视频和音频可以互相 attention（声音对应画面）

维度示例
────────
视频: (2, 3, 8, 224, 224) → patch=(2,16,16) → (2, 392, 768)
      T=8帧, H=W=224, patch_t=2, patch_h=patch_w=16
      N_v = (8/2) × (224/16) × (224/16) = 4 × 14 × 14 = 392

音频: (2, 80, 256) → patch=(16,16) → (2, 80, 768)
      n_mels=80, T_a=256, patch_f=16, patch_t=16
      N_a = (80/16) × (256/16) = 5 × 16 = 80

文本: (2, 20) → embedding → (2, 20, 768)

总序列: 1 + 392 + 1 + 80 + 20 = 494 个 token（含 [VID] [AUD] 标记）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 视频 3D Patch Embedding
# ═══════════════════════════════════════════════════════════════════════════════

class VideoPatchEmbed3D(nn.Module):
    """
    把视频切成 3D patch（时间×高×宽），同时投影到 d_model 维。

    用 Conv3d 一步完成：kernel_size=(Pt, Ph, Pw), stride=(Pt, Ph, Pw)
    输出的每个"体素"对应原视频一个 Pt×Ph×Pw 的时空块。

    维度流转:
      输入:  (B, 3, T, H, W)     例: (2, 3, 8, 224, 224)
      Conv:  (B, d_model, T/Pt, H/Ph, W/Pw)  例: (2, 768, 4, 14, 14)
      Flat:  (B, d_model, N_v)   例: (2, 768, 392)
      Trans: (B, N_v, d_model)   例: (2, 392, 768)
    """

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (2, 16, 16),
        in_channels: int = 3,
        d_model: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size  # (Pt, Ph, Pw)
        self.in_channels = in_channels
        self.d_model = d_model

        # 3D 卷积: 同时切时空 patch 并投影
        self.proj = nn.Conv3d(
            in_channels, d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W)
        返回: (B, N_v, d_model)
        """
        # Conv3d: (B, 3, 8, 224, 224) → (B, 768, 4, 14, 14)
        x = self.proj(x)
        # 压平后两维空间: (B, 768, 4, 14, 14) → (B, 768, 4*14*14=392)
        x = x.flatten(2)
        # 转置: (B, 392, 768)
        x = x.transpose(1, 2)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 音频 Patch Embedding
# ═══════════════════════════════════════════════════════════════════════════════

class AudioPatchEmbed(nn.Module):
    """
    把音频频谱图切成 2D patch（频率×时间），投影到 d_model 维。

    输入是预处理好的梅尔频谱图 (B, n_mels, T_a)。
    实际音频预处理（wav → mel spectrogram）在模型外完成，
    这里只负责把频谱图转成 token 序列。

    维度流转:
      输入:  (B, n_mels, T_a)     例: (2, 80, 256)
      Conv:  (B, d_model, n_mels/Pf, T_a/Pt)  例: (2, 768, 5, 16)
      Flat:  (B, d_model, N_a)   例: (2, 768, 80)
      Trans: (B, N_a, d_model)   例: (2, 80, 768)
    """

    def __init__(
        self,
        patch_size: tuple[int, int] = (16, 16),
        d_model: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size  # (Pf, Pt) 频率方向, 时间方向
        self.d_model = d_model

        # 2D 卷积切 patch + 投影
        # in_channels=1: 频谱图是单通道（幅度值）
        self.proj = nn.Conv2d(
            1, d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_mels, T_a) — 梅尔频谱图
        返回: (B, N_a, d_model)
        """
        # 加通道维: (B, 80, 256) → (B, 1, 80, 256)
        x = x.unsqueeze(1)
        # Conv2d: (B, 1, 80, 256) → (B, 768, 5, 16)
        x = self.proj(x)
        # 压平: (B, 768, 80)
        x = x.flatten(2)
        # 转置: (B, 80, 768)
        x = x.transpose(1, 2)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 模态类型嵌入 —— 让模型知道 token 来自哪个模态
# ═══════════════════════════════════════════════════════════════════════════════

class ModalityTypeEmbed(nn.Module):
    """
    为不同模态的 token 加上类型标识。

    三种模态: video(0), audio(1), text(2)
    类似 NLP 中的 token type embedding（BERT 区分句子 A/B）。

    为什么需要？
      视频 patch 和音频 patch 可能有相似数值范围，
      模型需要额外信号区分"这是画面信息"还是"这是声音信息"。
    """

    def __init__(self, d_model: int = 768):
        super().__init__()
        # 3 种模态: video=0, audio=1, text=2
        self.embed = nn.Embedding(3, d_model)

    def forward(
        self,
        x: torch.Tensor,
        modality_type: int,  # 0=video, 1=audio, 2=text
    ) -> torch.Tensor:
        """
        x: (B, seq_len, d_model)
        返回: x + modality_embedding
        """
        B, seq_len, _ = x.shape
        ids = torch.full((B, seq_len), modality_type, dtype=torch.long, device=x.device)
        return x + self.embed(ids)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 音视频多模态理解模型
# ═══════════════════════════════════════════════════════════════════════════════

class VideoAudioLanguageModel(nn.Module):
    """
    音视频多模态理解模型 —— 看视频+听声音，生成文字描述。

    ════════════════════════════════════════════════════════════════════════════
    【核心问题: 视频、音频、文字三种 token 怎么排？】
    ════════════════════════════════════════════════════════════════════════════

    排列方案:
      [VID] [vid0] [vid1] ... [vidN] [AUD] [aud0] [aud1] ... [audM] [txt0] [txt1] ...
       ↑     ↑ 视频 patch          ↑     ↑ 音频 patch          ↑ 文字 token
      特殊   (B, N_v, d)          特殊   (B, N_a, d)           (B, L, d)
      token

    [VID] 和 [AUD] 是可学习的特殊 token，标记各模态的起始位置。

    ════════════════════════════════════════════════════════════════════════════
    【因果 Mask 设计 —— 三模态混合】
    ════════════════════════════════════════════════════════════════════════════

    关键规则:
      1. 视频 patch 之间: 全双向可见（同一帧内 + 跨帧）
      2. 音频 patch 之间: 全双向可见
      3. 视频看音频: 可见（声音对应画面，互相辅助）
      4. 音频看视频: 可见（画面辅助声音理解）
      5. 文字看所有: 可见（生成时有完整音视频上下文）
      6. 文字看文字: 因果（自回归）
      7. 视频/音频看文字: 不可见（不需要看未来的文字）

    展开示例 (1视频 + 1音频 + 2文字):

              VID  vid0  AUD  aud0  txt0  txt1
            ┌────┬────┬────┬────┬────┬────┐
      VID   │ 1  │ 1  │ 1  │ 1  │ 0  │ 0  │   ← 特殊token看所有音视频
            ├────┼────┼────┼────┼────┼────┤
      vid0  │ 1  │ 1  │ 1  │ 1  │ 0  │ 0  │   ← 视频看音视频全可见
            ├────┼────┼────┼────┼────┼────┤
      AUD   │ 1  │ 1  │ 1  │ 1  │ 0  │ 0  │   ← 特殊token看所有音视频
            ├────┼────┼────┼────┼────┼────┤
      aud0  │ 1  │ 1  │ 1  │ 1  │ 0  │ 0  │   ← 音频看音视频全可见
            ├────┼────┼────┼────┼────┼────┼────┤
      txt0  │ 1  │ 1  │ 1  │ 1  │ 1  │ 0  │   ← 文字看所有音视频 + 看自己因果
            ├────┼────┼────┼────┼────┼────┤
      txt1  │ 1  │ 1  │ 1  │ 1  │ 1  │ 1  │   ← 文字看所有音视频 + 看所有文字
            └────┴────┴────┴────┴────┴────┘

    四块区域总结:
              音视频部分      文字部分
            ┌────────────┬────────────┐
      音视频 │   全 1     │    全 0    │   ← 音视频互相可见，不看文字
            ├────────────┼────────────┤
      文字  │   全 1     │   下三角   │   ← 文字看所有音视频，文字因果
            └────────────┴────────────┘
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 10,
        d_ff: int = 3072,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        # 视频配置
        video_patch_size: tuple[int, int, int] = (2, 16, 16),
        video_in_channels: int = 3,
        # 音频配置
        audio_patch_size: tuple[int, int] = (16, 16),
        # 模态标记
        vid_token_id: int = 3,   # [VID] 特殊 token id
        aud_token_id: int = 4,   # [AUD] 特殊 token id
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.vid_token_id = vid_token_id
        self.aud_token_id = aud_token_id

        # ─── 模态编码器 ───
        self.video_embed = VideoPatchEmbed3D(
            patch_size=video_patch_size,
            in_channels=video_in_channels,
            d_model=d_model,
        )
        self.audio_embed = AudioPatchEmbed(
            patch_size=audio_patch_size,
            d_model=d_model,
        )

        # ─── 文本嵌入 ───
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        # ─── 特殊模态标记的嵌入 ───
        # [VID] 和 [AUD] 是可学习的向量，作为各模态的"开头标记"
        self.vid_token_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.aud_token_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ─── 模态类型嵌入 ───
        self.modality_embed = ModalityTypeEmbed(d_model)

        # ─── 位置编码（1D 可学习，覆盖所有模态）───
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.embed_dropout = nn.Dropout(dropout)

        # ─── Transformer Decoder Blocks ───
        # 用标准 DecoderBlock（和语言模型一样），但 mask 我们自己构造
        from model.decoder import DecoderBlock
        self.layers = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_rmsnorm=True,
                use_swiglu=True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # ─── 输出投影 ───
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_multimodal_mask(
        self,
        total_len: int,
        n_video: int,      # 含 [VID] 标记
        n_audio: int,      # 含 [AUD] 标记
        device: torch.device,
    ) -> torch.Tensor:
        """
        构建音视频三模态因果掩码。

        音视频部分全双向可见，文字部分因果可见，文字能看到所有音视频。

        参数:
            total_len: 总序列长度
            n_video: 视频 token 数量（含 [VID]）
            n_audio: 音频 token 数量（含 [AUD]）
            device: 目标设备

        返回: (1, 1, total_len, total_len) — 可广播到 (B, n_heads, total_len, total_len)
        """
        mask = torch.zeros(total_len, total_len, device=device, dtype=torch.bool)

        # 音视频分界点
        av_end = n_video + n_audio  # 音视频 token 结束位置

        # 左上 + 右上 的左半部分：音视频看音视频（全可见）
        mask[:av_end, :av_end] = True

        # 左下：文字看音视频（全可见）
        if av_end < total_len:
            mask[av_end:, :av_end] = True

        # 右下：文字看文字（因果，下三角）
        if av_end < total_len:
            text_len = total_len - av_end
            causal = torch.tril(torch.ones(text_len, text_len, device=device, dtype=torch.bool))
            mask[av_end:, av_end:] = causal

        return mask.unsqueeze(0).unsqueeze(0)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        编码视频。
        video: (B, C, T, H, W)
        返回: (B, N_v, d_model)
        """
        x = self.video_embed(video)  # (B, N_v, d_model)
        x = x + self.modality_embed(x, modality_type=0)  # video=0
        return x

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        编码音频。
        audio: (B, n_mels, T_a) — 梅尔频谱图
        返回: (B, N_a, d_model)
        """
        x = self.audio_embed(audio)  # (B, N_a, d_model)
        x = x + self.modality_embed(x, modality_type=1)  # audio=1
        return x

    def forward(
        self,
        video: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        前向传播。

        参数:
            video:     (B, C, T, H, W) 或 None
            audio:     (B, n_mels, T_a) 或 None
            input_ids: (B, L) 文本 token ids 或 None

        返回:
            logits: (B, total_seq_len, vocab_size)

        维度流转示例 (B=2, 视频8帧224x224, 音频80x256, 文本10个token):

          video:           (2, 3, 8, 224, 224)
              ↓
          video_embed:     (2, 392, 768)          ← 8/2 × 224/16 × 224/16 = 392
              ↓
          + [VID] token    (2, 1, 768)
              ↓
          audio:           (2, 80, 256)
              ↓
          audio_embed:     (2, 80, 768)           ← 80/16 × 256/16 = 80
              ↓
          + [AUD] token    (2, 1, 768)
              ↓
          input_ids:       (2, 10)
              ↓
          text_embed:      (2, 10, 768)
              ↓
          拼接:            (2, 484, 768)          ← 1+392+1+80+10 = 484
              ↓
          + 位置编码
              ↓
          + embed_dropout
              ↓
          mask:            (1, 1, 484, 484)
              ↓
          Decoder × 10:    (2, 484, 768)
              ↓
          output_proj:     (2, 484, vocab_size)
        """
        B = None
        parts = []

        # ─── 视频部分 ───
        if video is not None:
            B = video.size(0)
            video_tokens = self.encode_video(video)  # (B, N_v, d_model)
            # 加上 [VID] 特殊标记
            vid_marker = self.vid_token_embed.expand(B, -1, -1)  # (B, 1, d_model)
            parts.append(vid_marker)
            parts.append(video_tokens)

        # ─── 音频部分 ───
        if audio is not None:
            B = audio.size(0) if B is None else B
            audio_tokens = self.encode_audio(audio)  # (B, N_a, d_model)
            # 加上 [AUD] 特殊标记
            aud_marker = self.aud_token_embed.expand(B, -1, -1)  # (B, 1, d_model)
            parts.append(aud_marker)
            parts.append(audio_tokens)

        # ─── 文本部分 ───
        if input_ids is not None:
            B = input_ids.size(0) if B is None else B
            text_tokens = self.token_embedding(input_ids) * (self.d_model ** 0.5)
            text_tokens = text_tokens + self.modality_embed(text_tokens, modality_type=2)
            parts.append(text_tokens)

        if not parts:
            raise ValueError("至少需要提供 video、audio 或 input_ids 之一")

        # ─── 拼接所有模态 ───
        x = torch.cat(parts, dim=1)  # (B, total_len, d_model)
        total_len = x.size(1)

        # ─── 位置编码 ───
        x = x + self.pos_embed[:, :total_len, :]
        x = self.embed_dropout(x)

        # ─── 计算各模态长度（用于 mask）───
        n_video = 0
        n_audio = 0
        if video is not None:
            n_video = 1 + video_tokens.size(1)  # [VID] + 视频 patch
        if audio is not None:
            n_audio = 1 + audio_tokens.size(1)  # [AUD] + 音频 patch

        # ─── 因果掩码 ───
        mask = self._build_multimodal_mask(total_len, n_video, n_audio, x.device)

        # ─── Decoder Blocks ───
        for layer in self.layers:
            x, _ = layer(x, tgt_mask=mask)
        x = self.norm(x)

        # ─── 输出投影 ───
        logits = self.output_proj(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        video: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """
        自回归生成文字描述。

        步骤:
          1. 先编码视频和音频（只做一次）
          2. 从 [BOS] 开始，逐个生成 token
          3. 每次把新生成的 token 拼到序列末尾，继续预测下一个

        参数:
            video:           (B, C, T, H, W) 或 None
            audio:           (B, n_mels, T_a) 或 None
            max_new_tokens:  最多生成多少个字
            temperature:     温度（<1更确定，>1更随机）
            top_k:           只从 top-k 个候选中采样
            bos_token_id:    开头标记
            eos_token_id:    结束标记

        返回:
            generated: (B, num_generated) 生成的 token ids
        """
        self.eval()
        device = next(self.parameters()).device

        B = 1
        if video is not None:
            B = video.size(0)
        elif audio is not None:
            B = audio.size(0)

        # ─── 1. 编码音视频（固定前缀，只做一次）───
        prefix_parts = []

        if video is not None:
            video = video.to(device)
            video_tokens = self.encode_video(video)
            vid_marker = self.vid_token_embed.expand(B, -1, -1)
            prefix_parts.extend([vid_marker, video_tokens])

        if audio is not None:
            audio = audio.to(device)
            audio_tokens = self.encode_audio(audio)
            aud_marker = self.aud_token_embed.expand(B, -1, -1)
            prefix_parts.extend([aud_marker, audio_tokens])

        if not prefix_parts:
            raise ValueError("生成时至少需要提供 video 或 audio")

        # 计算 mask 需要的参数
        n_video = 0
        n_audio = 0
        if video is not None:
            n_video = 1 + video_tokens.size(1)
        if audio is not None:
            n_audio = 1 + audio_tokens.size(1)
        # ─── 2. 自回归生成 ───
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            # 当前完整序列 = 音视频前缀 + 已生成的文字
            text_tokens = self.token_embedding(generated) * (self.d_model ** 0.5)
            text_tokens = text_tokens + self.modality_embed(text_tokens, modality_type=2)
            x = torch.cat(prefix_parts + [text_tokens], dim=1)
            total_len = x.size(1)

            # 位置编码
            x = x + self.pos_embed[:, :total_len, :]

            # mask
            mask = self._build_multimodal_mask(total_len, n_video, n_audio, device)

            # 前向传播
            for layer in self.layers:
                x, _ = layer(x, tgt_mask=mask)
            x = self.norm(x)
            logits = self.output_proj(x)

            # 取最后一个位置的 logits
            next_logits = logits[:, -1, :]  # (B, vocab_size)

            # temperature
            next_logits = next_logits / temperature

            # top-k
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')

            # softmax + 采样
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 拼接到生成序列
            generated = torch.cat([generated, next_token], dim=1)

            # 检查是否生成了 EOS
            if (next_token == eos_token_id).all():
                break

        # 去掉开头的 BOS
        return generated[:, 1:]
