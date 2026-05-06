# DDPM vs Flow Matching：对比详解

> 一句话总结：DDPM 走**弯曲的山路**（1000步），Flow Matching 走**笔直的高速公路**（20~50步）。

---

## 一、核心直觉差异

### DDPM：山路十八弯

```
干净视频 ──╮──╮──╮──╮──╮──╮──╮──╮──╮──╮──→ 纯噪声
           ╰──╯  ╰──╯  ╰──╯  ╰──╯  ╰──╯
              1000步，必须一步步走
```

DDPM 从"干净视频"到"纯噪声"走一条**预先设计好的弯曲路径**。每一步的"弯曲程度"由一张 `β` 调度表控制，模型必须死记硬背这张表。

### Flow Matching：笔直高速

```
干净视频 ────────────────────────────────→ 纯噪声
              直路！20步就够
```

Flow Matching 走的是**直线**。从 A 到 B 就是线性插值，不需要任何预计算表。

---

## 二、公式对比

| | DDPM | Flow Matching |
|---|---|---|
| **加噪/插值** | `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε` | `z_t = (1-t) · z_0 + t · z_1` |
| **预计算表** | 需要 `β_1...β_1000` 和 `ᾱ_t` | **不需要任何表** |
| **模型预测** | `ε_θ(x_t, t)` —— 噪声长什么样 | `v_θ(z_t, t)` —— 往哪走 |
| **目标** | `ε`（采样的噪声本身） | `z_1 - z_0`（方向向量） |
| **t 的含义** | 离散时刻 `t ∈ {1,2,...,1000}` | 连续进度 `t ∈ [0.0, 1.0]` |

### DDPM 的"查表"操作

```python
# DDPM 训练前必须预计算
betas = torch.linspace(1e-4, 0.02, 1000)   # β 调度表
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)  # ᾱ_t 表

# 训练时查表
sqrt_alpha_bar = alphas_cumprod[t].sqrt()
sqrt_one_minus = (1 - alphas_cumprod[t]).sqrt()
x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus * noise  # 查表计算
```

### Flow Matching 的"一步搞定"

```python
# Flow Matching：没有任何预计算，t 是什么就是什么
z_t = (1 - t) * z_0 + t * z_1   # t=0.732 → 直接算，不需要查表
```

---

## 三、训练流程对比

### DDPM 训练（单步）

```
x_0 (干净视频)
  ↓
随机选 t ∈ {1, 2, ..., 1000}
  ↓
查表得 ᾱ_t
  ↓
x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε   ← 加噪
  ↓
模型预测 ε_pred = UNet(x_t, t)
  ↓
Loss = MSE(ε_pred, ε)          ← 猜噪声长什么样
```

### Flow Matching 训练（单步）

```
z_0 (干净视频的 VAE latent)
  ↓
随机选 t ∈ [0.0, 1.0]（连续）
  ↓
采样噪声 z_1 ~ N(0, I)
  ↓
z_t = (1-t)·z_0 + t·z_1        ← 直线插值，无需查表
  ↓
模型预测 v_pred = MMDiT(z_t, t, text)
  ↓
Loss = MSE(v_pred, z_1 - z_0)  ← 猜方向对不对
```

**关键差异**：
- DDPM：`t=500` 和 `t=501` 是两个完全不同的查表结果，模型必须分别记忆。
- Flow Matching：`t=0.5` 和 `t=0.5001` 只是插值系数稍有不同，模型通过连续学习泛化到所有 `t`。

---

## 四、训练模拟流程（带具体数字）

用一个 batch 走一遍，看看每一步的张量形状和数值怎么变。

**假设条件**（图像生成场景）：
- Batch size: 2
- 图像: `(B, 3, 64, 64)` = `(2, 3, 64, 64)`
- 文本嵌入: `(B, 77, 768)`（CLIP 编码的标准形状）

### DDPM 训练模拟

```python
# 输入
x_0 = torch.randn(2, 3, 64, 64)   # 2张随机"干净图像"，值域约 [-3, 3]

# Step 1: 随机采样时间步 t
t = torch.randint(0, 1000, (2,))   # → t = [500, 750]
                                   # 图1走一半，图2走3/4

# Step 2: 采样噪声
noise = torch.randn_like(x_0)      # randn_like: 生成和 x_0 同形状的标准正态随机数
                                   # → (2, 3, 64, 64)，每个元素 ~ N(0,1)，值域约 [-3, 3]

# Step 3: 查表加噪（核心！）
# 查表得系数（从预计算的 1000 个元素数组中索引）
sqrt_alpha_bar = scheduler.sqrt_alphas_cumprod[t]
#   → t=500: 0.71,  t=750: 0.50
sqrt_one_minus = scheduler.sqrt_one_minus_alphas_cumprod[t]
#   → t=500: 0.70,  t=750: 0.87

# 广播到图像形状 (2,1,1,1)
sqrt_alpha_bar = sqrt_alpha_bar.reshape(2, 1, 1, 1)   # [0.71, 0.50]
sqrt_one_minus = sqrt_one_minus.reshape(2, 1, 1, 1)   # [0.70, 0.87]

# 加噪公式: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus * noise
#     [0.71, 0.50] * x_0   +   [0.70, 0.87] * noise
#     图1: 71%原图 + 70%噪声   图2: 50%原图 + 87%噪声
# x_t 形状: (2, 3, 64, 64)

# Step 4: UNet 预测噪声
noise_pred = unet(x_t, t)          # → (2, 3, 64, 64)
# UNet 接收加噪图 x_t 和时间步 t，输出"我猜这张图里混了多少噪声"

# Step 5: 算损失
loss = F.mse_loss(noise_pred, noise)
# = mean((noise_pred - noise)²)
# 在 (B, C, H, W) 所有维度上求平均，得到标量

# Step 6: 反向传播（只更新 UNet，表不动）
loss.backward()
optimizer.step()

# 一次训练迭代完成！只更新了 UNet 的权重，那张 β 表永远不变。
```

**DDPM 训练的数字总结**：

| 步骤 | 操作 | 输入形状 | 输出形状 | 关键数字 |
|---|---|---|---|---|
| 采样 t | `randint(0,1000)` | - | `(2,)` | t=[500, 750] |
| 采样噪声 | `randn_like` | `(2,3,64,64)` | `(2,3,64,64)` | ε ~ N(0,I) |
| **查表** | `table[t]` | `(2,)` | `(2,)` | √ᾱ=[0.71, 0.50] |
| 加噪 | `√ᾱ·x_0 + √(1-ᾱ)·ε` | `(2,3,64,64)` | `(2,3,64,64)` | 图1含71%信号 |
| UNet 预测 | `unet(x_t, t)` | `(2,3,64,64)` | `(2,3,64,64)` | ε_pred |
| 损失 | `MSE(ε_pred, ε)` | `(2,3,64,64)` | 标量 | 越小越好 |

---

### Flow Matching 训练模拟

```python
# 输入（注意：这里是 VAE 编码后的 latent，不是原始像素！）
z_0 = torch.randn(2, 4, 8, 64, 64)   # 2段视频的 VAE latent
                                     # (B, C_latent, T, H, W)
text_emb = torch.randn(2, 77, 768)   # 2段文本描述（CLIP/T5 编码）

# Step 1: 随机采样时间步 t（连续！不是整数）
t = torch.rand(2)                  # → t = [0.35, 0.72]
                                   # 图1: 35%进度，图2: 72%进度

# Step 2: 采样噪声 z_1
z_1 = torch.randn_like(z_0)        # randn_like: 生成和 z_0 同形状的标准正态随机数
                                   # → (2, 4, 8, 64, 64)，和 z_0 形状相同、设备相同

# Step 3: 直线插值（无需查表！）
# t: (2,) → reshape 为 (2, 1, 1, 1, 1) 广播到 5D
t_bc = t.reshape(2, 1, 1, 1, 1)    # [[0.35], [0.72]]

z_t = (1 - t_bc) * z_0 + t_bc * z_1
#     [[0.65], [0.28]] * z_0  +  [[0.35], [0.72]] * z_1
#     图1: 65%视频 + 35%噪声
#     图2: 28%视频 + 72%噪声
# z_t 形状: (2, 4, 8, 64, 64)

# Step 4: CFG dropout（10%概率丢文本）
# 假设 batch 中第1个样本的文本被丢弃
text_emb[0] = 0.0                  # → 第1个样本变成无条件

# Step 5: 浮点 t → 整数（给模型的时间嵌入用）
t_int = (t * 999).long()           # → [349, 719]
# 这只是把 [0,1] 映射到 [0,999] 的离散索引

# Step 6: MMDiT 预测速度
v_pred = mmdit(z_t, t_int, text_emb)   # → (2, 4, 8, 64, 64)
# MMDiT 接收: 加噪视频 z_t + 时间 t_int + 文本 text_emb
# 输出: "从当前点走向干净视频的速度方向"

# Step 7: 计算目标速度（恒定！）
v_target = z_1 - z_0               # → (2, 4, 8, 64, 64)
# 注意：目标速度对 batch 中所有样本都一样，
#       都是 "z_1 - z_0"，不依赖 t！

# Step 8: 算损失
loss = F.mse_loss(v_pred, v_target)
# = mean((v_pred - (z_1 - z_0))²)
# 在 (B, C, T, H, W) 所有维度上求平均，得到标量

# Step 9: 反向传播
loss.backward()
optimizer.step()

# 一次训练迭代完成！
```

**Flow Matching 训练的数字总结**：

| 步骤 | 操作 | 输入形状 | 输出形状 | 关键数字 |
|---|---|---|---|---|
| 采样 t | `rand(2)` | - | `(2,)` | t=[0.35, 0.72] |
| 采样噪声 | `randn_like` | `(2,4,8,64,64)` | `(2,4,8,64,64)` | z_1 ~ N(0,I) |
| **插值** | `(1-t)·z_0 + t·z_1` | `(2,4,8,64,64)` | `(2,4,8,64,64)` | 图1: 65%视频+35%噪声 |
| CFG dropout | `text_emb[0]=0` | `(2,77,768)` | `(2,77,768)` | 第1个样本无条件 |
| t→int | `round(t*999)` | `(2,)` | `(2,)` | [349, 719] |
| MMDiT 预测 | `mmdit(z_t, t, text)` | 5D+文本 | `(2,4,8,64,64)` | v_pred |
| 目标速度 | `z_1 - z_0` | 5D | 5D | **恒定不变** |
| 损失 | `MSE(v_pred, v_target)` | 5D | 标量 | 越小越好 |

---

### 两者训练流程的直观对比

```
DDPM 训练（单步）:
  x_0 ──→ 随机选 t=500 ──→ 查表得(0.71, 0.70) ──→ x_t = 0.71·x_0 + 0.70·ε
    ↓                                                      ↓
  UNet(x_t, 500) ───────────────────────────────────────→ ε_pred
    ↓                                                      ↓
  Loss = MSE(ε_pred, ε)  ← "猜噪声长什么样"

Flow Matching 训练（单步）:
  z_0 ──→ 随机选 t=0.35 ──→ 直接算(0.65, 0.35) ──→ z_t = 0.65·z_0 + 0.35·z_1
    ↓                                                       ↓
  MMDiT(z_t, 349, text) ─────────────────────────────────→ v_pred
    ↓                                                       ↓
  Loss = MSE(v_pred, z_1 - z_0)  ← "猜方向对不对"
```

**最大的区别**：
- DDPM：**查表**得到加噪系数，目标是**噪声本身**（ε），每步的系数都不同。
- Flow Matching：**直算**插值系数，目标是**方向向量**（z_1 - z_0），恒定不变。

---

## 五、推理（生成）对比

### DDPM 推理：必须1000步

```python
x = 纯噪声  # t=1000
for t in reversed(range(1000)):   # 1000次循环！
    noise_pred = model(x, t)
    x = denoise_step(x, noise_pred, t)  # 每步都要查表
    # 不能跳过，不能少步
```

### Flow Matching 推理：20~50步任选

```python
z = 纯噪声  # t=1.0
num_steps = 20    # ← 可以改成 10、20、50，灵活！
dt = 1.0 / num_steps

for i in range(num_steps):
    v = model(z, t, text)   # 问方向
    z = z - dt * v          # 走一步
```

| | DDPM | Flow Matching |
|---|---|---|
| **最少步数** | 1000（固定） | 10~20（可少） |
| **推荐步数** | 1000 | 20~50 |
| **能否灵活调整** | ❌ 不能 | ✅ 可以 |
| **每步计算量** | 1次 UNet | 2次 MMDiT（CFG） |
| **总计算量** | 1000 × UNet | 20 × 2 × MMDiT ≈ 40 × MMDiT |

---

## 六、为什么 Flow Matching 不能一步生成？

既然 Flow Matching 走直线、不需要查表，那为什么推理时还要20步，不能直接一步从噪声生成干净视频？

### 核心原因：模型是"近视眼"

模型 `MMDiT(z_t, t, text)` 预测的速度 `v_θ` 只在**当前这个点**精确。虽然数学上的"目标速度" `v_target = z_1 - z_0` 是恒定的，但**神经网络学出来的速度场**在整个路径上是有误差的：

```
Flow Matching 路径（直线）:
  z_0 ←─────────────────────────────────────── z_1
         ↑  模型只在"当前站立点"的预测精确
         ↓  离开这个点，预测就逐渐不准了

模型是一个"近视眼导航员"：
  - 站在 z_1（t=1.0）时，它说"往东走"
  - 但它只看清了前50米，不是整条1000米的路
  - 直接走1000米会跑偏

正确做法：每走50米，停下来重新问导航员
  z_1 → 问 → 走50米 → z_0.95 → 问 → 走50米 → ... → z_0
```

### 数学解释：数值积分

Flow Matching 的精确解是一个常微分方程（ODE）：

```
z(0) = z(1) + ∫_1^0 v_θ(z(t), t) dt
```

但这个积分**没有解析解**（因为 `v_θ` 是神经网络），只能用数值方法近似：

```python
# 20步 Euler 法：把 [0,1] 切成20段，每段 dt=0.05
z = z_1
for i in range(20):
    t = 1.0 - i * 0.05
    v = model(z, t, text)   # 在当前点重新采样速度
    z = z - 0.05 * v        # 沿速度走一小步
# 最后 z ≈ z_0（近似值，步数越多越接近真实解）
```

### 步数与质量的权衡

| 步数 | 本质 | 误差 | 效果 |
|---|---|---|---|
| **1步** | `z_0 ≈ z_1 - 1.0 * v_θ(z_1, 1.0)` | 很大 | 模糊/失真，不可用 |
| **5步** | 粗略数值积分 | 中等 | 能辨认内容但不精细 |
| **10步** | 较粗的积分 | 较小 | 基本可用，细节略差 |
| **20步** | 较细的积分 | 小 | 质量好，**常用默认值** |
| **50步** | 精细积分 | 很小 | 最佳质量，但速度慢 |
| **∞步** | 精确积分 | 0 | 理论极限，不可达 |

### 训练1步 vs 推理多步

```
训练时：
  随机采一个 t=0.732，让模型学"在这个点指出正确方向"
  → 只需要1步预测 + 和恒定目标比 loss

推理时：
  要把"每个局部方向"连起来走完整条路
  → 需要多步迭代，因为模型只有局部视野，没有全局先知能力
```

> **一句话**：Flow Matching 的"快"是相对 DDPM 的1000步而言。从数学上，任何 ODE 都需要多步数值积分来逼近真实解，20步已经是很少了。

---

## 七、速度差异的根本原因

### 为什么 DDPM 需要1000步？

因为 DDPM 的路径是**弯曲的**。每步只能沿当前点的切线方向走一小段，走多了就会偏离曲线。就像在山路上开车，必须慢慢拐。

```
DDPM 路径（弯曲）:
  x_0 ──→ x_1 ──→ x_2 ──→ ... ──→ x_999 ──→ x_1000
  每步只能走固定距离，走歪了无法回头
```

### 为什么 Flow Matching 只要20步？

因为 Flow Matching 的路径是**直线**。模型预测的是整条直线的方向（恒定不变），每步可以走很远。

```
Flow Matching 路径（直线）:
  z_0 ────────────────────────────────────────→ z_1
       ↑←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←↑
       └──────── 方向恒定，大步走就行 ─────────┘
```

用导航比喻：
- DDPM：山路弯弯，每步只能看眼前10米，需要1000步才能到。
- Flow Matching：笔直高速，导航员告诉你"一直往东"，大步走20步就到。

---

## 八、CFG（文本引导）对比

两者都支持 CFG，但实现方式一样：

```python
# DDPM 和 Flow Matching 的 CFG 公式完全相同
v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
```

差异在于 `v_cond` 和 `v_uncond` 是什么：

| | DDPM | Flow Matching |
|---|---|---|
| `v_cond` | `ε_cond = UNet(x_t, t, text)` | `v_cond = MMDiT(z_t, t, text)` |
| `v_uncond` | `ε_uncond = UNet(x_t, t, null)` | `v_uncond = MMDiT(z_t, t, null)` |
| 物理意义 | 文本让噪声变成什么样 | 文本让方向偏向哪里 |

---

## 九、优缺点总结

| 维度 | DDPM | Flow Matching |
|---|---|---|
| **训练复杂度** | 需要预计算 β/α 表 | **无需任何预计算** |
| **训练速度** | 单步1次模型调用 | 单步1次模型调用（相同） |
| **推理速度** | 慢（1000步 × 1次） | **快**（20步 × 2次 = 40次） |
| **灵活性** | 步数固定，不能改 | **步数任意可调** |
| **数学简洁度** | 复杂（需要理解马尔可夫链） | **简单**（就是直线插值） |
| **稳定性** | 步数少时质量差 | 步数少时仍能工作 |
| **与 VAE 配合** | 在像素空间或 latent 空间 | **必须在 VAE latent 空间** |
| **工业应用** | Stable Diffusion 1.x/2.x | SD3、HunyuanVideo、Flux |

---

## 十、在这个项目中的对应代码

| 组件 | DDPM | Flow Matching |
|---|---|---|
| 配置文件 | [`DDPMConfig`](diffusion/unet/ddpm.py) | [`VideoFlowMatchingConfig`](diffusion/dit/video_flow_matching.py#L94-L117) |
| 调度器 | [`DDPMNoiseScheduler`](diffusion/unet/ddpm.py) | [`VideoFlowMatchingScheduler`](diffusion/dit/video_flow_matching.py#L125-L217) |
| 训练器 | [`DiffusionTrainer`](diffusion/unet/ddpm.py) | [`VideoFlowMatchingTrainer`](diffusion/dit/video_flow_matching.py#L225-L358) |
| 采样器 | [`DDPMSampler`](diffusion/unet/ddpm.py) | [`VideoFlowMatchingSampler`](diffusion/dit/video_flow_matching.py#L366-L608) |
| 模型 | [`UNet`](diffusion/unet/unet.py) | [`VideoMMDiT`](diffusion/dit/mmdit-vedio.py) |

---

## 十一、UNet vs MMDiT（模型架构对比）

DDPM 和 Flow Matching 是**训练/推理框架**，UNet 和 MMDiT 是**神经网络模型**。两者正交：

```
框架（怎么走）        模型（谁来预测）
─────────────       ───────────────
DDPM        ──────→ UNet（图像用）
Flow Matching ─────→ MMDiT（视频用）
```

理论上可以互换：DDPM + MMDiT 或 Flow Matching + UNet 也是可行的，只是工业界形成了上面的搭配惯例。

### 架构差异

| | **UNet** | **MMDiT（VideoMMDiT）** |
|---|---|---|
| **基础单元** | 卷积层（Conv2d） | 注意力层（Self-Attention / Cross-Attention） |
| **空间建模** | 局部感受野（卷积核 3×3） | 全局感受野（注意力看所有位置） |
| **时间建模** | 需额外加 3D 卷积或时序模块 | 把时间轴 T 当作序列维度，用注意力自然处理 |
| **下采样** | MaxPool / Stride Conv（显式降分辨率） | Patch Embedding（把图像切成小块） |
| **上采样** | TransposeConv / Interpolate（显式升分辨率） | 没有上采样，直接预测完整分辨率输出 |
| **文本融合** | Cross-Attention 或通道拼接 | **双流设计**：文本和视频先各自处理，再交叉注意力融合 |
| **参数量** | 相对少（卷积共享权重） | 相对多（注意力需要 Q/K/V 投影） |
| **适合数据** | 图像（2D: H×W） | **视频+文本**（3D/5D: T×H×W + text） |

### 数据流对比

**UNet（DDPM 用）：**

```
输入 x_t (B, 3, 64, 64) + 时间步 t
  ↓
[编码器] Conv ↓ Pool → 特征图越来越小（64→32→16→8）
  ↓
[瓶颈] 最深层处理
  ↓
[解码器] Conv ↑ Upsample → 特征图越来越大（8→16→32→64）
  ↓         ↑ 跳跃连接（把编码器的特征直接传到解码器）
输出 ε_pred (B, 3, 64, 64)

直观：像一个"U"字形，先压扁再放大，中间用跳跃连接保留细节。
```

**MMDiT（Flow Matching 用）：**

```
视频 z_t (B, 4, T, H, W) → PatchEmbedding → 视频token序列
文本 text_emb (B, L, D)  ─────────────────→ 文本token序列
                                               ↓
                    ┌──────────────────────────┐
                    │    双流层（前8层）        │
                    │  文本流 ──→ Self-Attn    │
                    │  视频流 ──→ Self-Attn    │
                    │     ↓         ↓          │
                    │    交叉注意力（Q视频 K/V文本）│
                    └──────────────────────────┘
                               ↓
                    ┌──────────────────────────┐
                    │    单流层（后8层）        │
                    │  文本+视频 拼接在一起     │
                    │  统一做 Self-Attention   │
                    └──────────────────────────┘
                               ↓
输出 v_pred (B, 4, T, H, W)

直观：不像 UNet 那样压扁放大，而是把所有信息变成"token序列"，
      用注意力"互相看"来交换信息。双流层先各自理解，单流层再深度融合。
```

### 为什么视频生成用 MMDiT 而不是 UNet？

1. **全局建模能力**：视频帧之间的时间关系需要长距离依赖，注意力天然适合；卷积感受野有限，要堆很多层才能"看到" distant frames。

2. **文本融合更优雅**：MMDiT 的双流设计让文本和视频有独立的表示空间，通过交叉注意力精确控制"哪段文本影响哪块视频区域"。UNet 的文本注入通常是全局的，控制粒度粗。

3. **Scale 更好**：Transformer 架构在大模型时代验证了她的扩展性（GPT、LLaMA 都是 Transformer）。视频生成模型参数量动辄几十亿，Transformer 比 UNet 更容易 scaling。

4. **没有下采样的信息丢失**：UNet 的下采样会丢失精细的空间细节，对高分辨率视频不利。MMDiT 通过 Patch Embedding 直接分块，不做分辨率压缩。

### 一句话区分

```
UNet:   "卷积网络，U字形压扁放大，适合图像"
MMDiT:  "Transformer，token互看，适合视频+文本"
```

---

## 十二、一句话记忆法

```
框架层面:
  DDPM:        "猜噪声长什么样"  →  山路1000步  →  查表
  Flow Matching: "指方向往哪走"  →  直路20步   →  心算

模型层面:
  UNet:        "卷积U字形，压扁放大"    →  图像生成
  MMDiT:       "注意力互看，双流融合"   →  视频+文本生成
```

```
DDPM:        x_t = √ᾱ·x_0 + √(1-ᾱ)·ε    （复杂，查表）
Flow Matching: z_t = (1-t)·z_0 + t·z_1    （简单，直算）
```

> **核心差异不是"预测噪声 vs 预测速度"，而是"弯曲路径 vs 直线路径"。**
> 直线路径让 Flow Matching 可以大步走，所以更快。
