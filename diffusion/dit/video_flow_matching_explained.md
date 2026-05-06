# Video Flow Matching 浅显讲解

> 本文用大白话解释 `video_flow_matching.py` 到底在干什么。
> 你不需要懂微积分，只需要理解"从 A 走到 B"的直觉。

---

## 一、Flow Matching 是什么？

### 先看 DDPM（老方法）

想象你要从北京走到上海：

```
DDPM 的路线：弯弯曲曲的山路

北京 ──╮──╮──╮──╮──╮──╮──╮──╮──╮──╮──→ 上海
       ╰──╯  ╰──╯  ╰──╯  ╰──╯  ╰──╯
       山路十八弯，必须慢慢走（1000步）
```

DDPM 从"干净视频"到"纯噪声"走一条弯曲的路径，每一步都要查复杂的"时刻表"（β 调度表）。

### 再看 Flow Matching（新方法）

```
Flow Matching 的路线：笔直的高速公路

北京 ────────────────────────────────→ 上海
       直路！可以大步走（20步就够）
```

Flow Matching 走的是**直线**。从"干净视频"到"纯噪声"，就是一条直线插值：

```
t=0.0:  [██████████] 100% 视频  +  0% 噪声   ← 起点（干净视频 z_0）
t=0.25: [██████░░░░]  75% 视频  + 25% 噪声
t=0.5:  [████░░░░░░]  50% 视频  + 50% 噪声   ← 最难猜的中间点
t=0.75: [██░░░░░░░░]  25% 视频  + 75% 噪声
t=1.0:  [░░░░░░░░░░]   0% 视频  + 100%噪声  ← 终点（纯噪声 z_1）
```

`t` 就是"进度条"：0=全是视频，1=全是噪声，中间就是视频和噪声的混合。

---

## 二、训练时：模型学什么？

### 直观理解

模型像一个"导航员"，站在路上的某个点，告诉你：**"往哪个方向走能最快到达上海（干净视频）"**。

```
你在 t=0.7 的位置（30%视频 + 70%噪声）

        ↑
        │ 模型说："往这个方向走！"
        │
   你在这里 ───────────────────────→ 上海（z_0）
   (z_t)                              (干净视频)

方向 = 速度 v_θ
```

### 数学上超级简单

```
构造训练样本：
  z_t = (1-t) × z_0 + t × z_1        ← 直线插值，一步搞定！

目标速度：
  v_target = z_1 - z_0                 ← 从视频指向噪声的方向（恒定！）

模型预测：
  v_pred = MMDiT(z_t, t, "一只猫在跑")  ← 模型猜的方向

损失：
  Loss = MSE(v_pred, v_target)         ← 猜的方向和真实方向有多像
```

**对比 DDPM：**

| | DDPM | Flow Matching |
|---|---|---|
| 加噪公式 | `x_t = √ᾱ·x_0 + √(1-ᾱ)·ε`（复杂） | `z_t = (1-t)·z_0 + t·z_1`（直线） |
| 模型预测 | ε（噪声长什么样） | v（往哪走） |
| 目标 | 需要查表算 | `z_1 - z_0`（直接减） |
| 训练步数 | 1000步 | 20~50步 |

---

## 三、训练代码的 7 步走（`VideoFlowMatchingTrainer`）

```python
def training_step(model, z_0, text_emb):
    # z_0: (B, 4, T, H, W)  —— 干净视频的 latent
    # text_emb: (B, 256, 4096) —— "一只猫在草地上跑"
```

### Step 1: 随机选一个"进度"t

```
t ~ [0.0, 1.0] 的随机数

不是均匀随机！而是用 logit_normal 分布：
  t=0.5 附近采样最多（中间点最难，多练）
  t=0.0 和 t=1.0 采样少（太简单，少练）
```

### Step 2: 造噪声

```
z_1 = randn_like(z_0)  ← 和 z_0 一样形状的高斯噪声
```

### Step 3: 直线插值

```
z_t = (1-t) × z_0 + t × z_1

例：t=0.3 → 70% 视频 + 30% 噪声
```

### Step 4: CFG Dropout（10%概率丢文本）

```
以 10% 概率把 text_emb 换成全零

为什么？训练时让模型学会两种情况：
  - 有文本时怎么生成（有条件）
  - 没文本时怎么生成（无条件）

推理时才能做 CFG（后面讲）
```

### Step 5: 把时间转成整数

```
t 是浮点数（如 0.732），模型要整数（如 732）

t_int = round(t × 999)  ← 映射到 [0, 999]
```

### Step 6: 模型预测

```
v_pred = MMDiT(z_t, t_int, text_emb)
         ↑
    调用 VideoMMDiT.forward()
    输入: 加噪视频 + 时间 + 文本
    输出: 预测的速度方向 (B, 4, T, H, W)
```

### Step 7: 算损失

```
v_target = z_1 - z_0          ← 真实方向（恒定不变）
loss = MSE(v_pred, v_target)  ← 猜的方向有多准
```

**结束！** 返回 loss，外面做 `loss.backward()` 和 `optimizer.step()`。

---

## 四、推理时：从噪声生成视频（`VideoFlowMatchingSampler`）

### 直观理解

训练好的模型是个"导航员"。推理时，你站在终点（纯噪声），让它一步步带你走回起点（干净视频）。

```
起点（推理）: 纯噪声 z_1 ~ N(0, I)
                ↓
           导航员说："往左走 0.05 公里"
                ↓
           新位置 z_{0.95}
                ↓
           导航员说："往左走 0.05 公里"
                ↓
           ... 重复 20 次 ...
                ↓
终点（推理）: 干净视频 z_0
                ↓
           VAE.decode(z_0) → 像素视频
```

### Euler 法（最简单）

每步走固定距离：

```
步数 = 20，每步 dt = 1/20 = 0.05

for i in range(20):
    t = 1.0 - i × 0.05   ← 当前进度

    # 问模型："我现在在哪？该往哪走？"
    v = MMDiT(z, t, text_emb)

    # 往反方向走一小步（从噪声走向视频）
    z = z - dt × v
```

### CFG（Classifier-Free Guidance）—— 文本控制强度的秘诀

**问题**：普通生成时，模型可能不听文本的话，随便生成。

**解决办法**：让模型跑两次，然后"放大差异"。

```
第一次：给文本    → v_cond   = MMDiT(z, t, "一只猫在跑")
第二次：不给文本  → v_uncond = MMDiT(z, t, 全零)

差异 = v_cond - v_uncond  ← "文本带来的额外方向"

最终方向 = v_uncond + cfg_scale × 差异
                              ↑
                         放大倍数！
```

**cfg_scale 的作用：**

| cfg_scale | 效果 |
|---|---|
| 1.0 | 不放大 = 普通有条件生成 |
| 7.0 | 默认，文本方向放大 7 倍，视频更贴描述 |
| 15+ | 极强引导，视频严格按文本但可能不自然 |

```
可视化（cfg_scale=7.0）:

v_uncond:  →→→  视频会动，但不知道是什么
              ↓
         + 7 × ↓  文本方向的额外推力
              ↓
v_cfg:     ↓↓↓   视频严格按"一只猫在跑"生成
```

### Midpoint 法（更准，但更慢）

Euler 法的问题：用起点的速度走整步，可能走歪。

Midpoint 法的改进：
1. 先走半步，看看中点在哪
2. 在中点重新测速度
3. 用中点的速度走整步

```
Step 1: 起点测速 → v1
        走半步: z_mid = z - (dt/2) × v1

Step 2: 中点测速 → v2（重新调用模型）
        走整步: z_new = z - dt × v2

误差: Euler O(dt)  vs  Midpoint O(dt²)
      粗糙          vs  精细
      2次模型调用    vs  4次模型调用
```

---

## 五、整体代码结构

```
video_flow_matching.py
├── VideoFlowMatchingConfig        ← 配置（时间步数、CFG强度等）
│
├── VideoFlowMatchingScheduler     ← 调度器
│   ├── interpolate(z0, z1, t)     ← 直线插值 z_t = (1-t)z_0 + t·z_1
│   ├── get_velocity_target(z0,z1) ← v_target = z_1 - z_0
│   ├── sample_t(batch_size)       ← 采样随机时间步
│   └── t_to_int(t)                ← 浮点→整数（给模型用）
│
├── VideoFlowMatchingTrainer       ← 训练器（训练时用）
│   └── training_step(model, z0, text_emb)
│       内部: 采样→插值→CFG dropout→模型预测→MSE loss
│
└── VideoFlowMatchingSampler       ← 采样器（推理时用）
    ├── sample_euler_cfg()         ← Euler法 + CFG（推荐，20步）
    └── sample_midpoint_cfg()      ← Midpoint法 + CFG（更准，10~20步）
```

---

## 六、和 MMDiT、VAE 的关系

```
训练阶段:

    视频 → VAE.encode() → z_0
                             ↓
              ┌──────────────────────────────┐
              │ video_flow_matching.py        │
              │  (构造 z_t → 调用 MMDiT → loss)│
              │       ↓                      │
              │  MMDiT.forward(z_t, t, text)  │ ← mmdit-vedio.py
              │       ↓                      │
              │  v_pred (速度预测)            │
              └──────────────────────────────┘
                             ↓
                         loss.backward()

推理阶段:

    噪声 z_1 ~ N(0, I)
        ↓
    ┌──────────────────────────────┐
    │ video_flow_matching.py        │
    │  (Euler迭代 → 调用 MMDiT)     │
    │       ↓                      │
    │  MMDiT.forward(z, t, text)    │ ← mmdit-vedio.py
    │       ↓                      │
    │  z_0 (干净latent)            │
    └──────────────────────────────┘
        ↓
    VAE.decode() → 视频像素
```

**总结**：
- `video_flow_matching.py` 是**教练/调度员**，负责安排训练/推理流程
- `mmdit-vedio.py` 是**运动员**，负责做预测
- `video_vae.py` 是**翻译官**，负责视频↔latent 的转换

---

## 七、核心公式速查

| 公式 | 含义 |
|---|---|
| `z_t = (1-t)·z_0 + t·z_1` | 直线插值，构造训练样本 |
| `v_target = z_1 - z_0` | 目标速度（恒定） |
| `Loss = MSE(v_pred, v_target)` | 训练损失 |
| `v_cfg = v_uncond + w×(v_cond - v_uncond)` | CFG 引导 |
| `z_new = z - dt × v_cfg` | Euler 去噪步 |

---

> **一句话总结**：Flow Matching 就是走直线。训练时模型学"方向感"（从任意点指出回起点的方向），推理时从噪声出发，一步步沿着模型指的方向走回干净视频。视频版比图像版只多了一件事：张量从 4D 变成 5D（多了时间轴 T）。
