## 6. TikTok Ads 广告投放

### 6.1 开户

**TikTok for Business：** ads.tiktok.com

**开户方式：**
- 自助开户（需企业资质）
- 找代理开户（推荐，有返点和支持）

**需要准备：**
- 企业营业执照
- TikTok 账号
- 应用商店链接
- 支付方式

### 6.2 像素/事件配置

**TikTok Pixel 或 SDK：**

```
1. TikTok Events Manager
2. 创建 Pixel（网页）或 App Event（应用）
3. 集成 TikTok SDK
   - iOS: CocoaPods 集成
   - Android: Gradle 集成

4. 追踪事件：
   ```swift
   // iOS
   TTTracker.trackEvent("LaunchApp")
   TTTracker.trackEvent("Registration")
   TTTracker.trackEvent("Purchase", withValue: 9.99, currency: "USD")
   ```
```

**必须追踪的事件：**

| 事件 | 说明 |
|------|------|
| LaunchApp | 启动应用 |
| Registration | 完成注册 |
| Purchase | 完成购买 |
| AddToCart | 加入购物车 |
| Checkout | 结账 |
| ViewContent | 浏览内容 |

### 6.3 创建 Campaign

**步骤1：Campaign 层级**

```
Advertising Objective:
  - App Promotion（应用推广）

Campaign Name:
  "[游戏名]_TT_[地区]_[日期]"

Budget:
  - Daily Budget: $100~$500
  - 或 Lifetime Budget

Bidding & Optimization:
  - Optimization Goal:
    - App Install（优化安装）
    - In-App Event（优化应用内事件）
    - Value（优化付费金额）
  
  - Bid Strategy:
    - Lowest Cost（最低成本）
    - Cost Cap（成本上限）
    - Lowest Cost with Bid Cap（带上限的最低成本）
```

**步骤2：Ad Group 层级**

```
Ad Group Name:
  "[定向]_[版位]"

Placement:
  - TikTok（主站）
  - Pangle（TikTok  Audience Network，第三方应用）
  - 或 Automatic Placement（自动版位）

Targeting:
  - Location: United States
  - Gender: All / Female / Male
  - Age: 18-24, 25-34, 35-44, 45+
  - Language: English
  - Interests:
    - Gaming（游戏）
    - Strategy games（策略游戏）
    - War games（战争游戏）
    - MMO games（大型多人在线）
  - Behaviors:
    - Game app installers（游戏应用安装者）
    - Paid gamers（付费玩家）
  - Device:
    - iOS / Android
    - Connection: WiFi（推荐，付费意愿更高）

Budget & Schedule:
  - Daily Budget: $50~$200
  - Schedule: All Day 或指定时段

Bidding:
  - Bid Type:
    - oCPM（优化千次展示，推荐）
    - CPC（按点击付费）
  
  - Bid Price:
    - 如果选 Cost Cap，设定期望 CPI
    - 例：$2.5
```

**步骤3：Ad 层级（素材）**

```
TikTok 素材特点：
  - 竖屏视频为主（9:16）
  - 时长：9~15秒（最佳）
  - 前3秒必须抓住注意力
  - 真人出镜效果好
  - 音乐很重要（TikTok 是音乐平台）

素材规格：
  - 视频分辨率：1080×1920
  - 文件大小：< 500MB
  - 时长：5~60秒（推荐 9~15秒）
  - 格式：MP4、MOV、MPEG

文案：
  - Ad Text: ≤ 100字符
  - 例："Can you beat level 100? 🧩 Download now!"
  
Call to Action:
  - Download Now
  - Install Now
  - Play Now
  - Learn More
```

### 6.4 TikTok 素材最佳实践

```
✅ 做这些：
  - 真人出镜试玩（Authentic）
  - 展示失败瞬间 + "你能做得更好吗？"
  - 快节奏剪辑（每2~3秒一个镜头）
  - 使用热门音乐（TikTok Commercial Music Library）
  - 加字幕（80%用户静音观看）
  - 明确的 CTA 在结尾

❌ 不要做：
  - 横屏视频（TikTok 用户不习惯）
  - 过度 CG 包装（用户觉得假）
  - 慢节奏（3秒没重点就划走了）
  - 太长的视频（>30秒）
```

### 6.5 Spark Ads（原生广告）

```
Spark Ads = 把普通 TikTok 视频变成广告

优势：
  - 看起来像普通内容，用户接受度高
  - 可以借用 KOL/UGC 视频
  - 互动率通常比传统广告高 30~50%

操作方式：
  1. 自己发布视频到 TikTok 账号
  2. 在 Ads Manager 中选择 "Spark Ads"
  3. 选择要推广的视频
  4. 设置投放参数

或：
  1. 找 KOL 制作视频
  2. KOL 授权你使用其视频投放广告
  3. 通过 Spark Ads 推广
```

---

## 7. Apple Search Ads (ASA)

### 7.1 为什么投 ASA

- iOS 用户质量通常比 Android 高（付费意愿强）
- 搜索意图明确（用户主动搜索关键词）
- 竞争相对 Meta/Google 较小
- 是 iOS 产品的"必投"渠道

### 7.2 开户

```
1. 访问 appstoreconnect.apple.com
2. 需要有 App Store 开发者账号
3. 在 App Store Connect 中启用 Search Ads
4. 访问 searchads.apple.com
5. 设置支付方式
```

### 7.3 Campaign 结构

```
Apple Search Ads 有两种模式：

Basic（基础版）:
  - 设置月预算和 CPI 目标
  - Apple 自动优化
  - 适合新手

Advanced（高级版）:
  - 可以设置关键词
  - 可以分组管理
  - 可以查看详细数据
  - 推荐用 Advanced
```

**Advanced 模式结构：**

```
Campaign（广告系列）
  └── Ad Group（广告组）
        └── Keywords（关键词）
        └── Creative Sets（素材组）
        └── Negative Keywords（否定关键词）
```

### 7.4 关键词策略

```
关键词类型：

品牌词（Brand）:
  - 你的游戏名
  - 例："puzzle king", "puzzleking"
  - 竞争小，CPI 低
  - 必须投，防止竞品抢

竞品词（Competitor）:
  - 竞争对手的游戏名
  - 例："candy crush", "homescapes"
  - 竞争大，CPI 高
  - 可以投，测试效果

品类词（Category）:
  - 游戏品类
  - 例："puzzle game", "match 3", "brain game"
  - 量大但泛，CPI 中等

通用词（Generic）:
  - 泛词
  - 例："game", "free game", "fun game"
  - 量最大但最不精准，CPI 高
  - 通常不建议投
```

**关键词匹配类型：**

```
Exact Match（精确匹配）:
  - 只匹配完全一样的词
  - 例："strategy game" 只匹配 "strategy game"
  - 最精准，CPI 最低

Broad Match（广泛匹配）:
  - 匹配相关变体
  - 例："strategy game" 也匹配 "strategy games", "war strategy game"
  - 量大但可能不精准

Search Match（搜索匹配）:
  - Apple 自动匹配相关搜索
  - 适合发现新关键词
```

### 7.5 创建 Campaign 步骤

```
步骤1：创建 Campaign
  Campaign Name: "[游戏名]_ASA_[目标]"
  
  Campaign Goal:
    - Install（优化安装）
    - Re-engagement（召回老用户）
  
  Countries/Regions: United States
  
  Budget:
    - Daily Budget: $100~$500
    - 或 Lifetime Budget
  
  CPA Goal: $2.0（设置目标 CPI）

步骤2：创建 Ad Group
  Ad Group Name: "Brand" / "Competitor" / "Category"
  
  Audience:
    - New Users（新用户）
    - Returning Users（回访用户）
    - All Users（所有用户）
  
  Keywords:
    - 添加关键词列表
    - 设置 Match Type（Exact/Broad）
    - 设置 Max CPT（最高每次点击出价）
  
  Creative Sets:
    - 默认用 App Store 截图
    - 可以上传自定义素材

步骤3：设置 Negative Keywords
  - 排除不相关的词
  - 例：如果你的游戏没有多人模式，排除 "multiplayer"
```

### 7.6 ASA 数据分析

```
核心指标：
  - Impressions（展示数）
  - Taps（点击数）
  - Conversions（转化数，即安装数）
  - Spend（花费）
  - CPA（每次转化成本，即 CPI）
  - CPT（每次点击成本）
  - TTR（Tap-Through Rate，点击率）
  - CR（Conversion Rate，转化率）

按关键词查看：
  - 哪些词带来最多安装？
  - 哪些词 CPI 最低？
  - 哪些词花费高但安装少？→ 降低出价或暂停
```

---

