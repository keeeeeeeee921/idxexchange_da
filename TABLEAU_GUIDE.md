# Market Analysis 仪表盘 — Tableau 搭建指南

本指南带你用 `week8_tableau_prep.py` 产出的四个 CSV，在 Tableau 里搭出
**Market Analysis** 仪表盘：KPI 卡片 + 趋势线 + 成交量/新增挂牌 + 地域/物业细分，
可按 city / county / zip / PropertySubType 筛选。

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

---

## 0. 数据源（来自 `data/processed/tableau/`）

| 文件 | 粒度 | 用途 |
|---|---|---|
| `tableau_sold.csv` | 每行一笔成交 | **主数据源**，做地域/物业细分、地图、明细筛选 |
| `tableau_listed.csv` | 每行一条挂牌 | 新增挂牌 / 库存视图 |
| `monthly_market.csv` | 每月一行（成交） | KPI 卡片 + 成交趋势线（预聚合，搭起来最省事） |
| `monthly_new_listings.csv` | 每月一行（挂牌） | 新增挂牌趋势 |

**建议**：KPI 卡片和趋势线优先连预聚合的 `monthly_*.csv`——数字已经算好，
不用在 Tableau 里重复聚合，也能当作明细视图的对照校验。地域/物业细分、地图、
明细筛选才用行级的 `tableau_sold.csv` / `tableau_listed.csv`。

`monthly_market.csv` 关键列：`yr_mo`（如 `2026-04`）、`closed_sales`、
`median_close_price`、`avg_close_price`、`median_price_per_sqft`、`avg_dom`、
`median_dom`、`avg_close_to_orig_ratio`、`avg_price_ratio`。

`monthly_new_listings.csv` 关键列：`yr_mo`、`new_listings`、`median_list_price`、
`median_list_per_sqft`。

---

## 1. 连接数据

1. 打开 Tableau Public，`Connect → To a File → Text file`，选 `monthly_market.csv`。
2. 在 **Data Source** 页确认每列类型：
   - `yr_mo` 设为 **Date**（或保留 String 也行，排序按字典序即可，因为 `YYYY-MM` 天然有序）。
   - 金额/比率列应为 **Number (decimal)**，计数列（`closed_sales`、`new_listings`）为 **Number (whole)**。
3. 再 `Add` 连入另外三个 CSV，作为同一工作簿里的独立数据源
   （**不要** join——粒度不同，分别用即可）。

> 若把 `yr_mo` 设成 Date：Tableau 可能把 `2026-04` 解析成某天，趋势线按月排序没问题。
> 想更稳妥，可建计算字段 `DATE(yr_mo + "-01")` 显式转成每月第一天。

---

## 2. KPI 卡片（连 `monthly_market.csv`）

做 5 张卡片，每张是一个单值 worksheet。以"最新月份中位成交价"为例：

1. 新建 worksheet，数据源选 `monthly_market`。
2. 只看最新月：把 `yr_mo` 拖到 **Filters**，选 `2026-04`（或用相对/Top 1 筛选取最新）。
   - 想自动跟最新月：`Filters → yr_mo → Top → By field → Top 1 by MAX(yr_mo)`。
3. 把 `median_close_price` 拖到 **Text**（Marks 卡）。聚合选 `MEDIAN` 或 `SUM`
   都行——预聚合后每月只有 1 行，结果一样。
4. 格式化成货币：右键该 measure → `Format → Numbers → Currency (Custom)`，
   小数位 0，前缀 `$`。
5. 调大字号，`Title` 写"中位成交价（最新月）"。

按同样套路再做 4 张：
- **closed_sales** → "成交套数（最新月）"
- **avg_dom** → "平均在售天数 DOM"（0 位小数）
- **avg_close_to_orig_ratio** → "成交/原始挂牌价比"（百分比或 2 位小数；这是上面修过的 KPI，应在 0.98–1.02 之间）
- **new_listings**（来自 `monthly_new_listings`）→ "新增挂牌（最新月）"

> 想在卡片上加"环比上月"箭头：用 `LOOKUP(SUM([median_close_price]), -1)` 之类的
> 表计算，或先做趋势线、把环比留到 v2，避免一开始过度设计。

---

## 3. 成交趋势线（连 `monthly_market.csv`）

1. 新 worksheet，`yr_mo` 拖到 **Columns**（若是 String，右键 → Sort 升序）。
2. `median_close_price` 拖到 **Rows** → 折线图。
3. 想叠加均值参照：再把 `avg_close_price` 拖到 Rows，右键 → `Dual Axis`，
   再 `Synchronize Axis`。
4. 复制此 worksheet 改 measure，得到第二条趋势线："平均 DOM 走势"（`avg_dom`），
   这条能清楚看出 2025 下半年 DOM 走高、2026 春季回落的市场节奏。

---

## 4. 成交量 vs 新增挂牌（双数据源对比）

目标：一张图看供需——柱子是月度成交套数，线是月度新增挂牌。

1. 新 worksheet，主数据源 `monthly_market`，`yr_mo` → Columns，`closed_sales` → Rows（柱状）。
2. 切到 `monthly_new_listings` 数据源，把 `new_listings` 拖到 Rows
   （Tableau 会用 `yr_mo` 做 **blend**，注意右上角 blend 链接图标点亮）。
3. 右键第二个轴 → `Dual Axis`。两个量级接近可 `Synchronize Axis`。
4. 颜色：成交柱用中性灰、新增挂牌线用强调色，便于区分供需两侧。

> blend 的前提是两个源都有 `yr_mo` 且命名一致——本 prep 脚本已保证这点。

---

## 5. 地域 / 物业细分（连 `tableau_sold.csv` 行级）

预聚合表没有地域维度，这部分必须用行级主数据源。

1. 新 worksheet，数据源 `tableau_sold`。
2. `CountyOrParish` → Rows，`ClosePrice` → Columns，聚合改 **MEDIAN**
   （右键 measure → Measure → Median）→ 横向条形图。
3. 排序：点轴上的排序按钮降序。
4. 想下钻到 city/zip：把 `City`、`PostalCode` 也加到 Rows 形成层级，
   或做成单独的 worksheet。
5. **物业类型对比**：复制此表，把 `CountyOrParish` 换成 `PropertySubType`。

**地图**（行级数据自带经纬度）：
- 新 worksheet，双击 `Latitude`、`Longitude`（Tableau 识别为地理角色后会画地图）。
- `PostalCode` 拖到 Detail，`median(ClosePrice)` 拖到 Color，得到分邮编的价格热力。
- 数据量大时先抽样或按 zip 聚合，避免上百万点拖慢渲染。

---

## 6. 全局筛选器

让整个仪表盘可按 city / county / zip / PropertySubType 联动筛选：

1. 在任一**行级**（`tableau_sold`）worksheet 上，把 `CountyOrParish`、`City`、
   `PostalCode`、`PropertySubType` 分别拖到 **Filters**。
2. 每个筛选器右键 → `Show Filter`。
3. 仪表盘里右键筛选器 → `Apply to Worksheets → All Using This Data Source`，
   让它们对所有行级视图统一生效。

> 注意：**预聚合的 KPI/趋势线（`monthly_*`）不带地域维度，无法被这些筛选器联动。**
> 如果你需要"选了某个 county 后 KPI 也跟着变"，那张 KPI 卡片就得改用行级
> `tableau_sold` 现算（如 `MEDIAN(ClosePrice)` + 同样的地域筛选），代价是放弃
> 预聚合的便利。两种做法各有取舍，按需求取一种，别混。

---

## 7. 组装仪表盘

1. `Dashboard → New Dashboard`，尺寸设为 `Automatic` 或固定 1366×768。
2. 顶部一行放 5 张 KPI 卡片（横向平铺）。
3. 中部放成交趋势线 + 成交量/新增挂牌对比。
4. 下部放地域条形图、物业细分、地图。
5. 把 `Show Filter` 出来的筛选器拖到右侧做一栏。
6. 加标题"Market Analysis — CRMLS Residential"，角落标注数据截止月份与
   "Confidential MLS data"。

---

## 8. 保存 / 交付

- **机密前提下**：`File → Save As…` 存本地 `.twbx`（见顶部警告），交付该文件或截图。
- 经 Aidan 许可可公开后，才用 `File → Save to Tableau Public` 发布。
- 数据刷新后（跑过 `run_pipeline.py`），Tableau 里 `Data → Refresh` 即可更新，
  无需重搭视图。

---

## 附：字段速查

**`tableau_sold.csv`**（行级成交，主表）：
`CountyOrParish, City, PostalCode, MLSAreaMajor, PropertyType, PropertySubType,
ListOfficeName, BuyerOfficeName, ListAgentFullName, BuyerAgentFullName,
CloseDate, yr_mo, close_year, close_month, ClosePrice, ListPrice, OriginalListPrice,
LivingArea, BedroomsTotal, BathroomsTotalInteger, DaysOnMarket, price_ratio,
close_to_original_list_ratio, price_per_sqft, listing_to_contract_days,
contract_to_close_days, Latitude, Longitude`

**`tableau_listed.csv`**（行级挂牌）：
`... MlsStatus, ListingContractDate, list_year, list_month, list_price_per_sqft,
price_reduction_ratio, close_to_list_ratio ...`（地域/物业列同上）
