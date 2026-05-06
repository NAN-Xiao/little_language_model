## 5. Google Ads (UAC) 广告投放

### 5.1 开户与设置

**开户：**
- ads.google.com
- 或找 Google 代理开户（推荐，有返点）

**需要准备：**
- Google 账号
- 支付方式
- Google Play 开发者账号 或 App Store 链接
- Firebase 项目（用于事件追踪）

### 5.2 Firebase 事件配置

Google 的 App Campaign 依赖 Firebase 事件进行优化。

**配置步骤：**

```
1. 在 Firebase Console 创建项目
   console.firebase.google.com

2. 添加应用（iOS/Android）
   - 输入 Bundle ID 或 Package Name
   - 下载配置文件（GoogleService-Info.plist 或 google-services.json）
   - 集成到应用中

3. 集成 Firebase SDK
   - iOS: CocoaPods 或 Swift Package Manager
   - Android: Gradle 依赖

4. 记录事件
   ```swift
   // iOS 示例
   import FirebaseAnalytics
   
   // 记录自定义事件
   Analytics.logEvent("level_complete", parameters: [
     "level": 5,
     "score": 1000
   ])
   
   // 记录购买
   Analytics.logEvent(AnalyticsEventPurchase, parameters: [
     AnalyticsParameterValue: 9.99,
     AnalyticsParameterCurrency: "USD"
   ])
   ```

5. 关联 Google Ads
   - Google Ads → Tools → Linked Accounts → Firebase
   - 选择 Firebase 项目 → 关联

6. 导入事件
   - Google Ads → Tools → Conversions
   - 导入 Firebase 事件作为转化事件
```

**必须导入的事件：**

| 事件 | 事件名 | 作用 |
|------|--------|------|
| first_open | 首次打开 | 计算安装 |
| in_app_purchase | 应用内购买 | 计算付费 |
| session_start | 会话开始 | 计算活跃 |
| level_up | 升级 | 用户参与度 |
| achievement_unlocked | 解锁成就 | 用户参与度 |

### 5.3 创建 App Campaign

**步骤1：新建 Campaign**

```
Campaign Type: App Promotion

Campaign Subtype:
  - App Installs（优化安装）
  - App Engagement（优化互动，适合召回老用户）
  - App Pre-Registration（预注册，新游戏）

选择应用：
  - 从 Google Play 或 App Store 选择
```

**步骤2：Campaign 设置**

```
Campaign Name: "[游戏名]_Google_UAC_[地区]_[日期]"

Locations: United States
Languages: English

Budget:
  - Daily Budget: $100（测试期）
  - Campaign Total Budget: 可设总预算上限

Bid Strategy:
  - Target CPI: $2.5（设置目标 CPI）
  - 或 Target ROAS: 150%（设置目标 ROAS）
  
注意：Google UAC 没有 Ad Set 层级，直接到 Ad Group
```

**步骤3：Ad Group 设置**

```
Ad Group Name: "[定向]_[素材类型]"

Ad Assets（素材资源）：
  Google 会自动组合你的素材，需要上传：

  视频素材（必须）：
    - 最少2个，最多20个
    - 时长：10~30秒（最佳15秒）
    - 尺寸：
      - 横屏 16:9（1920×1080）
      - 竖屏 9:16（1080×1920）
      - 方形 1:1（1080×1080）
    - 文件大小：< 100MB
    - 格式：MP4、MOV、AVI

  图片素材（可选）：
    - 最少2个，最多20个
    - 尺寸：
      - 横屏 16:9（1200×628）
      - 竖屏 9:16（1080×1920）
      - 方形 1:1（1200×1200）
    - 格式：JPG、PNG

  文案（Headlines & Descriptions）：
    - Headlines: 最多5条，每条 ≤ 30字符
    - Descriptions: 最多5条，每条 ≤ 90字符
    - 例 Headline: "Kingdom War", "Build Your Empire", "Free to Play"
    - 例 Description: "Solve challenging puzzles", "Play offline anytime"
```

**步骤4：素材最佳实践**

```
视频素材要求：
  ✅ 前3秒必须有核心玩法展示
  ✅ 加入字幕（很多用户静音观看）
  ✅ 结尾有明确的 CTA（"Download Now"）
  ✅ 展示游戏真实画面（不要过度包装）
  ❌ 不要过长（>30秒会跳过）
  ❌ 不要太多文字覆盖
  ❌ 不要使用受版权保护的音乐
```

### 5.4 Google UAC 的特殊性

**Google UAC 和 Meta 的区别：**

| 维度 | Meta | Google UAC |
|------|------|-----------|
| 定向 | 可以精细设置 | 几乎全自动，无法精细定向 |
| 版位 | 可选 | 全自动（Search/YouTube/Display/Play） |
| 素材 | 需要为每个版位单独制作 | 自动适配所有版位 |
| 优化 | 可以选择优化目标 | 可以选择优化安装/事件/价值 |
| 可控性 | 高 | 低（黑盒） |

**Google UAC 的优化逻辑：**

```
Google 的算法会自动：
  1. 选择最佳版位（Search/YouTube/Display/Play Store）
  2. 选择最佳受众（基于你的应用类型）
  3. 组合最佳素材和文案
  4. 优化出价以获得目标 CPI/ROAS

你需要做的：
  1. 提供尽可能多的素材（让算法有更多选择）
  2. 设置合理的 CPI/ROAS 目标
  3. 定期更换效果差的素材
  4. 耐心等待学习期（7~14天）
```

### 5.5 数据查看

**路径：Google Ads → Campaigns → 点击 Campaign 名称**

**核心列：**

```
必须关注的列：
  - Conversions（转化数，即安装数）
  - Cost/conv.（每次转化成本，即 CPI）
  - Conv. rate（转化率）
  - Impressions（展示数）
  - CTR（点击率）
  - Cost（花费）
  - Conv. value/cost（转化价值/成本，即 ROAS）
  - View-through conv.（浏览转化，用户看到广告后未点击但安装了）

Assets 标签页：
  - 可以看到每个素材的表现
  - 视频/图片的 "Performance" 评级：
    - Learning（学习中）
    - Low（差）→ 考虑替换
    - Good（好）
    - Best（最好）→ 制作类似素材
```

---

