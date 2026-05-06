# 第二部分：平台实操指南

## 4. Meta (Facebook/Instagram) 广告投放

### 4.1 开户与准备

**开户方式：**
1. 自己开户：facebook.com/business，需企业资质
2. 代理开户：找 Meta 官方代理（推荐，有返点和支持）

**需要准备的材料：**
- 企业营业执照
- Facebook 个人账号（需实名）
- Business Manager（BM）
- 企业验证（Domain 验证或电话验证）
- 支付方式（信用卡或 PayPal）

**BM 设置步骤：**

```
1. 访问 business.facebook.com
2. 创建 Business Manager
3. 添加页面（Page）
4. 添加广告账户（Ad Account）
5. 添加支付方式
6. 设置像素（Pixel）或应用事件（App Events）
7. 验证域名（Domain Verification）
```

### 4.2 应用事件配置（关键！）

Meta 需要知道用户在应用内的行为，才能优化投放。

**需要追踪的事件：**

| 事件 | 优先级 | 说明 |
|------|--------|------|
| **Install** | 必须 | 应用安装 |
| **Launch** | 必须 | 应用启动 |
| **Registration** | 必须 | 用户注册/完成新手引导 |
| **Purchase** | 必须 | 用户付费 |
| **Level Achieved** | 推荐 | 完成关卡 |
| **Achievement Unlocked** | 推荐 | 解锁成就 |
| **Spent Credits** | 推荐 | 消耗游戏币 |

**配置方式（以 Adjust 为例）：**

```
1. 在 Adjust 后台创建 App
2. 获取 App Token
3. 在 Meta Events Manager 中：
   - Settings → Partner Integrations → Adjust
   - 输入 Adjust App Token
   - 映射事件（Adjust Event → Meta Event）
4. 测试事件回传（用 Test Events 工具）
```

### 4.3 创建广告系列（Campaign）

**登录 Ads Manager：adsmanager.facebook.com**

**步骤1：点击"Create"创建广告系列**

```
Campaign Objective（广告目标）:
  选择 "App Promotion"（应用推广）
  或 "Sales" → 优化购买事件
```

**步骤2：设置 Campaign 层级**

```
Campaign Name: 建议格式 "[游戏名]_[地区]_[目标]_[日期]"
  例："KingdomWar_US_CPI_20240101"

Buying Type:
  - Advantage+ Campaign Budget（推荐，自动分配预算）
  - Manual Budget（手动分配，需在每个 Ad Set 设预算）

Campaign Budget:
  测试期: $100~$500/天
  放量期: $2,000+/天

Bid Strategy:
  - Lowest Cost（最低成本，推荐测试期用）
  - Cost Cap（成本上限，设置 CPI 上限）
  - Minimum ROAS（最低 ROAS，放量期用）
```

**步骤3：设置 Ad Set（受众和版位）**

```
Ad Set Name: 建议包含定向信息
  例："US_Female_25-44_iOS_Lookalike1%"

App:
  选择你的应用（iOS 或 Android）

Budget & Schedule:
  - Daily Budget: $100（测试期）
  - Schedule: 长期投放（除非有特定活动）

Audience:
  Locations: United States
  Age: 25-44
  Gender: Women（如果目标女性用户）
  Languages: English
  
  Detailed Targeting:
    - Interests: Mobile games, Strategy games, Clash of Clans, Rise of Kingdoms
    - Behaviors: Game players, Engaged shoppers
    
  Custom Audiences:
    - 上传种子用户（高价值用户列表）
    
  Lookalike Audiences:
    - 1% Lookalike（最像，量小但精准）
    - 3% Lookalike（平衡）
    - 5% Lookalike（量大但泛）

Placements（版位）:
  - Advantage+ Placements（推荐，自动优化版位）
  - 或手动选择：
    - Facebook Feed（信息流）
    - Facebook Reels
    - Instagram Feed
    - Instagram Reels
    - Audience Network（第三方应用内广告）

Optimization & Delivery:
  - Optimization for Ad Delivery:
    - App Installs（优化安装）
    - App Events（优化事件，如 Purchase）
    - Value（优化付费金额）
  
  - Cost Control:
    - Cost Per Result Goal: $2.5（设置期望 CPI）
```

**步骤4：创建 Ad（素材）**

```
Identity:
  - 选择 Facebook Page 或 Instagram Account

Ad Format:
  - Single Image or Video（单图或视频）
  - Carousel（轮播）
  - Collection（集合）

Media:
  - 上传视频或图片
  - 视频时长：15~30秒（最佳）
  - 文件大小：
    - 视频 < 4GB
    - 图片 < 30MB
  - 分辨率：
    - Feed: 1080×1080 或 1080×1920
    - Reels: 1080×1920（竖屏9:16）

Primary Text（文案）:
  - 前3秒抓住注意力
  - 突出核心玩法
  - 避免过度承诺（避免虚假宣传）
  - 例："Can you solve this puzzle? 99% fail!"

Headline（标题）:
  - 简短有力
  - 例："Download Now!"

Call to Action（按钮）:
  - Download
  - Play Now
  - Install Now
```

### 4.4 A/B 测试设置

**在 Meta 中创建 A/B 测试：**

```
方法1：Ad Set 层级 A/B 测试
  1. 创建 Campaign
  2. 创建多个 Ad Set，其他设置相同，只变一个变量：
     - Ad Set A: 定向 25-34岁
     - Ad Set B: 定向 35-44岁
  3. 每个 Ad Set 放相同素材
  4. 跑3~7天后对比 CPI 和 ROAS

方法2：Ad 层级 A/B 测试
  1. 一个 Ad Set
  2. 创建多个 Ad，只变素材：
     - Ad A: 玩法视频
     - Ad B: 真人实拍
     - Ad C: UE4渲染
  3. 跑3~7天后对比 CTR 和 IPM

方法3：Dynamic Creative（动态创意）
  1. Ad Set 中勾选 "Dynamic Creative"
  2. 上传多个素材（5个视频）和多个文案（5条）
  3. Meta 自动组合测试所有组合
  4. 自动优化，展示效果最好的组合
```

### 4.5 数据查看与分析

**查看路径：Ads Manager → Campaigns/Ad Sets/Ads**

**核心列设置（Customize Columns）：**

```
必须添加的列：
  - Results（结果数，如安装数）
  - Cost per Result（CPI）
  - Amount Spent（花费）
  - CTR (Link Click-Through Rate)
  - Link Clicks（点击数）
  - CPC (Cost Per Link Click)
  - Impressions（曝光数）
  - CPM (Cost Per 1,000 Impressions)
  - Frequency（频次）
  - Mobile App Installs（应用安装数）
  - Cost Per Mobile App Install（CPI）
  - App Events（应用事件数）
  - Purchases（购买数）
  - Cost Per Purchase（CPP）
  - Purchase ROAS（购买ROAS）
  - Unique CTR（独立用户点击率）
```

** breakdown 分析：**

```
点击 "Breakdown" 可以按维度拆分数据：

By Delivery:
  - Age（年龄分布）
  - Gender（性别）
  - Country（国家）
  - Platform（iOS/Android）
  - Placement（版位）
  - Device（设备）

By Action:
  - Conversion Device（转化设备）
  - Post Reaction（互动类型）
```

### 4.6 优化操作

**何时关停广告：**

```
关停标准（测试期）：
  - 花费 > $50 且 0 安装 → 关停
  - CPI > 目标 CPI × 2 且花费 > $100 → 关停
  - Frequency > 5 且 CTR 下降 > 30% → 关停（素材疲劳）
  - Day3 ROAS < 20% → 预警
  - Day7 ROAS < 30% → 考虑关停

放量标准：
  - CPI < 目标 CPI
  - Day3 ROAS > 40%
  - 留存率达标（Day1 > 40%）
```

**如何放量：**

```
1. 增加预算（每次增加不超过 20~30%）
   错误：$100 → $1,000（一步跳10倍）
   正确：$100 → $130 → $170 → $220 → ...

2. 复制表现好的 Ad Set
   - 原 Ad Set 不动
   - Duplicate 一个，稍微扩大定向或换素材

3. 扩展 Lookalike
   - 从 1% → 3% → 5%

4. 扩展到新地区
   - 美国验证后 → 加拿大 → 英国 → 澳大利亚
```

---

