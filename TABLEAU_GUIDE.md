# Tableau 搭建指南 — 两个必交工作簿

本指南带你用 `week8_tableau_prep.py` 产出的四个 CSV，在 Tableau 里搭出
Handbook（Week 8–10）要求的 **两个工作簿**：

1. **`market_analysis.twbx`** — 市场走势（5 张必做图 + 1 张自选仪表盘）
2. **`competitive_analysis.twbx`** — 竞争格局（4 张必做图 + 1 张自选仪表盘）

时间范围统一：**2024-01 至最新月**。

字段名全部保留英文（Tableau 里显示的就是 CSV 的英文列名），操作说明用中文。

---

## ⚠️ 发布前必读：机密数据警告

这是机密 CRMLS MLS 数据。**Tableau Public 的默认"保存"是把数据连同工作簿一起上传到
公网、任何人可搜索可下载。** 在得到数据负责人（Aidan）明确许可前，**不要** 用
`File → Save to Tableau Public`。

替代方案二选一：
- **本地保存**：`File → Save As…`，存成本地 `.twbx`（打包工作簿，数据留在本机）。
  新版 Tableau Public 桌面版支持本地另存；旧版若只给 "Save to Tableau Public"，
  就改用 Tableau Desktop。
- **Tableau Desktop**：功能与 Public 一致，但默认本地保存，适合机密数据。

`.twb / .twbx / .hyper / .tde` 已在 `.gitignore` 里，不会误提交到 git。

> Handbook 的 Week 11–12 才要求把成品发布到 Tableau Public。**那一步发布前务必先找
> Aidan 确认可公开**，本指南只做到本地 `.twbx`。

---

## 0. 两个工作簿要交什么（Handbook 对照表）

### 工作簿 1：`market_analysis.twbx`
| # | 必做视图 | 度量 | 数据源 |
|---|---|---|---|
| 1 | Monthly median close price | `MEDIAN(ClosePrice)` 按月 | `tableau_sold` |
| 2 | Average days on market | `AVG(DaysOnMarket)` 按月 | `tableau_sold` |
| 3 | Average close-to-original-list ratio | `AVG(close_to_original_list_ratio)` 按月 | `tableau_sold` |
| 4 | New listings | 挂牌记录计数按月 | `tableau_listed` |
| 5 | Closed sales | 成交记录计数按月 | `tableau_sold` |
| + | 自选仪表盘 ×1 | 自由发挥 | — |

**全部要求可按 `City` / `CountyOrParish` / `PostalCode` / `PropertySubType` 筛选。**

### 工作簿 2：`competitive_analysis.twbx`
| # | 必做视图 | 度量 | 数据源 |
|---|---|---|---|
| 1 | Top 100 listing agents（成交额 & 套数） | `SUM(ClosePrice)` + 记录计数 | `tableau_sold` |
| 2 | Top 100 listing offices（成交额 & 套数） | 同上 | `tableau_sold` |
| 3 | 邮编中位成交价热力图 | `MEDIAN(ClosePrice)` by zip | `tableau_sold` |
| 4 | 邮编成交套数热力图 | 记录计数 by zip | `tableau_sold` |
| + | 自选仪表盘 ×1 | 自由发挥 | — |

视图 1–2 要可按 `City`/`CountyOrParish`/`PostalCode`/`PropertySubType` 筛选；
视图 3–4 还要再加一个 **月份** 筛选。

> **关键结论：所有"可筛选"的图都必须连行级 CSV（`tableau_sold` / `tableau_listed`），
> 不能用月度预聚合表 `monthly_*.csv`。** 因为月度表已经把地域维度聚合掉了，
> 选某个 county/zip 时它无法重算。月度表只留作"核对数字对不对"的对照，不进仪表盘。

---

## 1. 数据源（来自 `data/processed/tableau/`）

| 文件 | 粒度 | 在本指南里的用途 |
|---|---|---|
| `tableau_sold.csv` | 每行一笔成交 | **两个工作簿的主力数据源**：价格/DOM/比率/成交计数、Top 100、热力图 |
| `tableau_listed.csv` | 每行一条挂牌 | 工作簿 1 的 "New listings" 计数 |
| `monthly_market.csv` | 每月一行（成交） | **仅作对照校验**，不进仪表盘 |
| `monthly_new_listings.csv` | 每月一行（挂牌） | 仅作对照校验 |

`tableau_sold.csv` 里搭这两个工作簿会用到的列：
`CountyOrParish, City, PostalCode, PropertySubType, ListOfficeName, ListAgentFullName,
CloseDate, yr_mo, ClosePrice, OriginalListPrice, DaysOnMarket,
close_to_original_list_ratio, Latitude, Longitude`。

`tableau_listed.csv` 用到：`City, CountyOrParish, PostalCode, PropertySubType,
ListingContractDate, yr_mo`（计数即可，不需要金额列）。

---

## 2. 连接数据 & 字段准备

1. 打开 Tableau，`Connect → To a File → Text file`，先选 `tableau_sold.csv`。
2. 再 `Add → Text file` 把 `tableau_listed.csv` 也连进来，作为**同一工作簿里的独立
   数据源**（**不要 join** —— 两表粒度不同，分别用即可）。
3. 在 **Data Source** 页确认/设置字段类型与地理角色：
   - `CloseDate`（sold）、`ListingContractDate`（listed）设为 **Date**。
   - `ClosePrice`、`OriginalListPrice`、`close_to_original_list_ratio` 设为 **Number (decimal)**；
     `DaysOnMarket` 设为 **Number (whole)**。
   - `PostalCode`：右键 → `Geographic Role → ZIP Code/Postcode`（热力图要用）。
   - `Latitude`/`Longitude`：右键 → `Geographic Role → Latitude / Longitude`
     （作为热力图的备选画法）。
   - `City`、`CountyOrParish`、`PropertySubType` 保留 String（维度）。

> **建议建两个独立 `.twbx`**，分别对应两个工作簿，保持交付物干净。也可以放一个
> 工作簿里用两个 Dashboard，但 Handbook 是按两个 `.twbx` 命名交付的，分开更稳。

---

# 工作簿 1：`market_analysis.twbx`

5 张折线趋势 + 1 张自选仪表盘，全部行级、全部可按地域/物业类型筛选。

## 3. 五张必做趋势图（连 `tableau_sold` / `tableau_listed`）

通用做法（每张图都一样的骨架）：
- 把日期拖到 **Columns**，右键那颗胶囊 → 选 **Month**（连续的绿色胶囊），得到按月的横轴。
- 把度量拖到 **Rows**，Mark 选 **Line**。
- 限定时间范围：把同一个日期字段拖到 **Filters** → `Range of dates` → 起点设 `2024-01-01`。

### 3.1 Monthly median close price（`tableau_sold`）
1. `CloseDate`（Month, 连续）→ Columns。
2. `ClosePrice` → Rows，右键该胶囊 → `Measure → Median`。
3. 格式化为货币：右键轴 → `Format → Numbers → Currency (Custom)`，小数 0、前缀 `$`。
4. Title：`Monthly Median Close Price`。

### 3.2 Average days on market（`tableau_sold`）
1. `CloseDate`（Month）→ Columns。
2. `DaysOnMarket` → Rows，聚合保持 **Average**。
3. Title：`Average Days on Market`。这条能看出 2025 下半年 DOM 走高、2026 春回落的节奏。

### 3.3 Average close-to-original-list ratio（`tableau_sold`）
1. `CloseDate`（Month）→ Columns。
2. `close_to_original_list_ratio` → Rows，聚合 **Average**。
3. 格式化为百分比或 2 位小数（这是修过的 KPI，正常应在 0.98–1.02 之间，可加一条
   `Reference Line = 1.0` 作基准）。
4. Title：`Avg Close-to-Original-List Price Ratio`。

### 3.4 New listings（连 `tableau_listed`）
1. 数据源切到 `tableau_listed`。
2. `ListingContractDate`（Month）→ Columns。
3. Rows 放**记录数**：把度量区底部的 `tableau_listed (Count)` 拖到 Rows
   （或建计算字段 `COUNT([PostalCode])` 这类对每行计数）。Mark 选 Line。
4. 日期范围过滤同样从 `2024-01-01` 起。
5. Title：`New Listings`。

### 3.5 Closed sales（`tableau_sold`）
1. 回到 `tableau_sold`。
2. `CloseDate`（Month）→ Columns。
3. Rows 放 `tableau_sold (Count)`（成交记录数）。Mark 选 Line（或 Bar）。
4. Title：`Closed Sales`。

> 想核对数字：把每张图的结果和 `monthly_market.csv` / `monthly_new_listings.csv`
> 对应列比一下，一致就说明行级聚合没搭错。

## 4. 工作簿 1 的全局筛选器

要求：可按 `City` / `CountyOrParish` / `PostalCode` / `PropertySubType` 联动筛选。

1. 在任一 `tableau_sold` 的 worksheet 上，把这四个字段分别拖到 **Filters** → 右键
   每个 → `Show Filter`。
2. 每个筛选器右键 → `Apply to Worksheets → All Using This Data Source`
   —— 这样 3.1/3.2/3.3/3.5（都连 `tableau_sold`）一键联动。
3. **New listings（3.4，连 `tableau_listed`）要单独配一套**：在它上面把同名的四个
   字段（`City`/`CountyOrParish`/`PostalCode`/`PropertySubType`）也拖进 Filters、
   `Show Filter`、`Apply to All Using This Data Source`。

> 跨数据源的小坑：Tableau 的筛选默认只作用于**同一数据源**。所以两套源各配一套
> 同名筛选器。组装仪表盘时，可以把两套筛选器叠放在同一个筛选区，用户体感上是
> 一组；若想做到"动一个、两源都动"，需要在数据模型里建关系/混合，初版不必上，
> 留到 v2。

## 5. 工作簿 1 的自选仪表盘（+1）

Handbook 要求自带一张"自由设计"的图。建议二选一（都用行级、能被上面筛选器联动）：
- **价格 vs 单价**：`CloseDate`(Month) → Columns，`MEDIAN(ClosePrice)` 与
  `MEDIAN(price_per_sqft)` 双轴（`Dual Axis` + 视情况 `Synchronize`），看总价和单价是否同步。
- **供需对照**：成交量（`tableau_sold` count）做柱、新增挂牌（`tableau_listed` count）
  做线，叠在一张图（用 `yr_mo` 做 blend，右上角 blend 链接图标点亮即生效）。

## 6. 组装工作簿 1 的 Dashboard

1. `Dashboard → New Dashboard`，尺寸 `Automatic` 或固定 1366×768。
2. 顶部一行放 3.1 / 3.2 / 3.5（价、DOM、成交量）。
3. 中部放 3.3（比率）+ 3.4（新增挂牌）。
4. 底部放第 5 节的自选图。
5. 把四个 `Show Filter` 拖到右侧成一栏。
6. 标题 `Market Analysis — CRMLS Residential`，角落标注数据截止月份与 `Confidential MLS data`。
7. `File → Save As…` 存本地 **`market_analysis.twbx`**。

---

# 工作簿 2：`competitive_analysis.twbx`

4 张图全部连 `tableau_sold` 行级。

## 7. Top 100 listing agents（成交额 & 套数）

1. 新 worksheet，数据源 `tableau_sold`。
2. `ListAgentFullName` → Rows。
3. `ClosePrice` → Columns，右键胶囊 → `Measure → Sum`（这是**成交额 volume**），
   得到横向条形图；点轴上的排序按钮**降序**。
4. **套数 units**：把 `tableau_sold (Count)` 拖到 **Label**（条形上直接显示套数），
   并拖一份到 **Tooltip**。也可拖到 **Color** 让颜色深浅表示套数。
5. **只留前 100**：右键 `ListAgentFullName` → `Filter → Top 标签页 →
   By field → Top 100 by SUM(ClosePrice)`。
6. Title：`Top 100 Listing Agents by Sales Volume`。

> **Top N 与地域筛选的先后**：要让"先按 county/zip 筛、再在筛后范围里取前 100"，
> 必须把地域筛选器设成 **Context Filter**（右键筛选器 → `Add to Context`），
> 否则 Top 100 会在全量上算完再被地域筛，结果不对。

## 8. Top 100 listing offices（成交额 & 套数）

把第 7 节整张复制（worksheet 右键 → Duplicate），把 `ListAgentFullName` 换成
`ListOfficeName`，Top 100 改成 `Top 100 by SUM(ClosePrice)`（仍按 office）。
Title：`Top 100 Listing Offices by Sales Volume`。

## 9. 邮编中位成交价热力图（`tableau_sold`）

1. 新 worksheet。确认 `PostalCode` 已设地理角色 `ZIP Code`（第 2 节做过）。
2. 双击 `PostalCode` → Tableau 出地图；Marks 把 Mark 类型改成 **Map**（填充/choropleth）。
3. `ClosePrice` → **Color**，右键 → `Measure → Median`。调色板选 diverging/sequential。
4. `PostalCode` 同时在 Detail，使每个 zip 是一个填充块。
5. Title：`Median Close Price by ZIP`。

> 若填充地图覆盖不全或太卡：改用 **符号地图** —— 双击 `Latitude`、`Longitude`，
> `PostalCode` 拖 Detail、`MEDIAN(ClosePrice)` 拖 Color、Size 适当；数据量大时
> 先按 zip 聚合或抽样，避免上百万点拖慢渲染。

## 10. 邮编成交套数热力图（`tableau_sold`）

复制第 9 节，把 Color 从 `MEDIAN(ClosePrice)` 换成 `tableau_sold (Count)`
（成交套数 / homes sold）。Title：`Homes Sold by ZIP`。

## 11. 工作簿 2 的全局筛选器

1. 四个地域/物业筛选：`City`/`CountyOrParish`/`PostalCode`/`PropertySubType`
   拖 Filters → `Show Filter` → `Apply to All Using This Data Source`（都连
   `tableau_sold`，一套就够）。
2. **月份筛选（热力图要求）**：把 `CloseDate` 拖 Filters → 选 `Month / Year` 离散，
   `Show Filter`（下拉或滑块），同样 Apply to All。
3. 别忘了第 7 节的提醒：把这些筛选器对 Top 100 两张图 `Add to Context`。

## 12. 工作簿 2 的自选仪表盘（+1）+ 组装

- **自选图建议**：`avg_close_to_original_list_ratio by ListOfficeName`（哪些经纪公司
  更容易溢价成交），或 `AVG(DaysOnMarket) by City` 横向条形，看哪个城市卖得快。
- `Dashboard → New Dashboard`：左上 Top 100 agents、右上 Top 100 offices，
  下方左右并排两张邮编热力图，右侧一栏放 5 个筛选器（4 地域 + 1 月份）。
- 标题 `Competitive Analysis — CRMLS Residential`，角落标 `Confidential MLS data`。
- `File → Save As…` 存本地 **`competitive_analysis.twbx`**。

---

## 13. 保存 / 交付清单

- 两个文件：`market_analysis.twbx`、`competitive_analysis.twbx`，**都用 `Save As…`
  存本地**（机密前提，见顶部警告）。交付 `.twbx` 或截图。
- 数据刷新后（跑过 `python3 run_pipeline.py`），Tableau 里 `Data → Refresh` 即可
  更新，无需重搭视图。
- Week 11–12 要发布到 Tableau Public 时，**先经 Aidan 许可**再 `Save to Tableau Public`。

---

## 附：字段速查

**`tableau_sold.csv`**（行级成交，两个工作簿的主表）：
`CountyOrParish, City, PostalCode, MLSAreaMajor, PropertyType, PropertySubType,
ListOfficeName, BuyerOfficeName, ListAgentFullName, BuyerAgentFullName,
CloseDate, yr_mo, close_year, close_month, ClosePrice, ListPrice, OriginalListPrice,
LivingArea, BedroomsTotal, BathroomsTotalInteger, DaysOnMarket, price_ratio,
close_to_original_list_ratio, price_per_sqft, listing_to_contract_days,
contract_to_close_days, Latitude, Longitude`

**`tableau_listed.csv`**（行级挂牌，工作簿 1 的 New listings 用）：
`... MlsStatus, ListingContractDate, list_year, list_month, list_price_per_sqft,
price_reduction_ratio, close_to_list_ratio ...`（地域/物业列同上）

**`monthly_market.csv`** / **`monthly_new_listings.csv`**（仅作数字核对，不进仪表盘）：
`yr_mo, closed_sales, median_close_price, avg_close_price, median_price_per_sqft,
avg_dom, median_dom, avg_close_to_orig_ratio` / `yr_mo, new_listings,
median_list_price, median_list_per_sqft`
