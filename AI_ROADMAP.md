# AI Roadmap — 把 IDX Exchange 项目重定位为「AI-Augmented Analytics」

> 目标岗位：**Data Analyst / Operations**
> 定位转变：从「CRMLS 数据分析 (BI/EDA + Tableau)」 → 「**AI-Augmented Real Estate Market Intelligence**」
> 本文是**路线图 / 实施计划**，不含代码。看完后再决定从哪个模块开工。

---

## 0. 核心定位原则（先读这段）

**DA 项目「描述世界」，AI 项目「预测 / 生成 / 决策」。** 要让招聘方把它读成 AI 项目，
AI 必须是**承重墙**——产出一个业务真正会用的结果，而不是在 Tableau 旁边贴一个模型。

但你的目标是 **Data Analyst / Operations**，不是 ML Engineer。所以方向要校准成：

> **用 AI 放大分析能力、自动化运营流程。**
> AI 是分析师的「力量倍增器」：自动写报告、让业务自助查数、用可解释的模型做决策支持、自动做数据质量。

**刻意不做**（这些是 ML Engineer 信号，DA/Ops 面试不加分、反而难辩护）：
Docker / Kubernetes、MLflow 模型注册、FastAPI 模型服务、复杂超参调优、特征商店、模型漂移监控基建。
👉 部署层用 **Streamlit** 就够了（一个 App 把所有模块串起来演示），这才是 DA/Ops 的正确「海拔」。

---

## 1. 项目现状盘点（已有的底子）

| 资产 | 内容 | 价值 |
|---|---|---|
| 原始数据 | 28 个月 CRMLS Sold + Listed CSV（2024-01 ~ 2026-04，~700MB） | 真实、规模够大 |
| 清洗后数据 | `data/processed/week7_sold_clean.csv`（**96 列**）含价格/面积/卧浴/年份/经纬度/`rate_30yr_fixed` | 直接能喂模型 |
| 月度聚合 | `data/processed/tableau/monthly_market.csv` 等 | 时间序列预测现成输入 |
| 流水线 | `run_pipeline.py`（12 阶段编排） | 可重定位为「自动化数据管线」 |
| EDA 报告 | `eda_report.py`（叙述文字**硬编码**） | M1 的改造对象 |
| 可视化 | `market_analysis.twbx` / `competitive_analysis.twbx` | 保留，作为「AI + BI」组合卖点 |
| 文档 | Trestle 元数据 / handbook PDF | 可选 RAG 语料 |

**结论**：底子很好。不需要重做数据工程，重点是在上面加「AI 产出层」。

---

## 2. 模块清单（按 DA/Ops 简历 ROI 排序）

### 🥇 Tier A — 先做这两个，性价比最高（直接把项目「翻」成 AI）

**M1 · AI 自动市场报告生成器**（LLM Insight Generator）
- 做什么：把 `eda_report.py` 里硬编码的叙述，换成 **Claude API** 读当月指标 → 自动生成市场评论 + 「值得深挖的异常」段落。
- 技术：Claude API、prompt engineering、结构化输出 (JSON schema)、Jinja 模板。
- DA/Ops 故事：**「把每月手工写报告的流程自动化，从 ~X 小时压到几分钟」**——这是纯正的运营效率叙事。
- 工作量：小（1 周）。**改造现有脚本即可，收益最大。**

**M2 · Chat-with-your-Data 分析助手**（Text-to-SQL）
- 做什么：Streamlit App，业务用自然语言提问（「Orange County 近半年中位价趋势？」）→ LLM 生成 SQL/pandas 查询 → 返回图表 + 文字答案。
- 技术：LLM tool-use / function-calling、Text-to-SQL、Streamlit、Plotly。
- DA/Ops 故事：**「让非技术同事自助查数，减少临时取数请求」**——典型分析师价值点。
- 工作量：中（1-1.5 周）。

### 🥈 Tier B — 加分析「分量」，证明你不只是调 API

**M3 · AVM 房价估值模型 + SHAP**（决策支持模型）
- 做什么：回归预测 `ClosePrice` / `price_per_sqft`；**SHAP** 解释「哪些因素驱动房价」。
- 技术：scikit-learn / XGBoost、特征工程、交叉验证、**SHAP** 可解释性。
- DA/Ops 故事：**「定价决策支持；SHAP 向业务揭示 Top 价值驱动因素」**——强调**解释**而非建模炫技。
- 注意：**一个讲得清楚的模型 >> 五个调过参的黑盒**。重点是可解释、能讲故事。
- 工作量：中（1-1.5 周）。

**M4 · 市场预测 + 自动预警**（Forecasting + Alerting）
- 做什么：用 `monthly_market.csv` + 利率，预测各 ZIP 下季度中位价/成交量；偏离预期时自动预警。
- 技术：Prophet / statsmodels SARIMAX、异常检测。
- DA/Ops 故事：**「市场拐点自动预警，支持运营提前决策」**。
- 工作量：中（1 周）。

### 🥉 Tier C — 锦上添花（时间够再做）

**M5 · 自动数据质量 QA**
- 做什么：你已有 IQR 离群标记 (`outlier_*_flag`)，升级成自动数据质量报告 + IsolationForest。
- DA/Ops 故事：「跨 28 个月 / 700MB 自动化数据质检」。

**M6 · Comp Finder 相似房源推荐**
- 做什么：kNN / 向量相似度「找 5 套最相似的成交房」。非常对口 IDX 主业，但优先级最低。

---

## 3. 技术栈（DA/Ops 合适的「海拔」）

| 层 | 选型 | 说明 |
|---|---|---|
| 数据处理 | pandas, numpy, DuckDB（可选，查大 CSV 很快） | 已有 |
| 经典 ML | scikit-learn, XGBoost, **SHAP** | 只用一个主模型 (AVM) |
| 时间序列 | Prophet 或 statsmodels | M4 |
| GenAI | **anthropic (Claude API)**, prompt engineering, 结构化输出 | M1/M2 核心 |
| NL 查询 | Text-to-SQL（LLM tool-use）+ DuckDB/SQLite | M2 |
| 前端/演示 | **Streamlit**（一个 App 串起全部） | 不用 FastAPI/Docker |
| 可视化 | 保留 Tableau + Plotly | 「AI + BI」组合卖点 |
| LLM 评测 | 轻量 groundedness 检查 / LLM-as-judge | 提一句即可，别做重 |

---

## 4. 建议目录结构（在现有项目上增量）

```
IDX Exchange DA/
├── (现有的 week*.py / run_pipeline.py / *.twbx 保留)
├── ai/
│   ├── shared/            # llm 客户端封装、prompt 模板、config
│   ├── reporting/         # M1 LLM 自动报告
│   ├── assistant/         # M2 Chat-with-data (text-to-SQL)
│   ├── models/
│   │   ├── avm/           # M3 房价模型 + SHAP
│   │   └── forecast/      # M4 预测 + 预警
│   └── dataqa/            # M5 数据质量 (可选)
├── app.py                 # Streamlit 入口，串起 M1/M2/M3/M4 做演示
├── requirements.txt       # 加: anthropic, streamlit, scikit-learn, xgboost, shap, prophet, plotly, duckdb
└── AI_ROADMAP.md          # 本文件
```

---

## 5. 里程碑（单人节奏，约 5-6 周；做完前 2-3 个就够翻定位）

| 周 | 里程碑 | 产出 |
|---|---|---|
| W0 | 重构 & 重写 README，把项目重命名定位 | 新 README + 目录骨架 + requirements |
| W1 | **M1 完成** | LLM 自动报告替换 `eda_report.py` 叙述 |
| W2 | **M2 完成** | Chat-with-data Streamlit App |
| W3 | **M3 完成** | AVM 模型 + SHAP 解释图 |
| W4 | M4 完成 | 预测 + 预警 |
| W5 | M5 + Streamlit 整合 + 录演示 | 一个能跑的端到端 Demo |

> **最小可行版本**：W0 + M1 + M2 三步做完，就足以把简历从「DA 项目」改写成「AI 项目」。M3/M4 是加分。

---

## 6. 简历写法

**项目名**：`AI-Augmented Real Estate Market Intelligence Platform`（替换原来的「CRMLS 数据分析」）

**Bullet 模板**（每条 = 一个 AI 能力 × 一个分析/运营结果 × 量化）：
- *Built an **LLM-powered reporting pipeline** (Claude API) that auto-generates monthly market commentary from 28 months of MLS data, cutting manual report time from ~X hrs to <Y min.*
- *Developed a **natural-language analytics assistant** (text-to-SQL + Streamlit) enabling non-technical stakeholders to self-serve queries, reducing ad-hoc data requests by ~Z%.*
- *Built an **explainable home-valuation model** (XGBoost + SHAP) surfacing top price drivers for pricing decisions, MAPE ≈ N%.*
- *Automated a **12-stage data pipeline** processing ~700MB of CRMLS data with built-in data-quality checks.*

**Skills 关键词**（DA/Ops 口味）：
`Python · SQL · Tableau · pandas · scikit-learn · XGBoost · SHAP · Prophet · Claude/LLM API · Prompt Engineering · Text-to-SQL · RAG · Streamlit · Automated Reporting · Data Pipeline · Data Quality`

---

## 7. 面试可辩护性（重要提醒）

- DA/Ops 面试官会**追问**。每个模块你都要能讲清楚「为什么这么做、局限是什么」。
- **不要**写你不能解释的 ML Engineer 术语（如「模型漂移监控」「特征商店」）——会反噬。
- 把 AI 定位成**工具**：「我用 AI 把分析/报告做得更快更自助」，而不是「我是做 AI 的」。这才符合 DA/Ops 人设，也最可信。
- 真实数据 (CRMLS) + 真实业务 (IDX Exchange) 是你最大的优势，反复强调「真实场景」。

---

## 下一步

看完路线图后，选一个开工点（建议 **M1**，改造 `eda_report.py`，收益最大、工作量最小）。
我可以帮你搭该模块的代码骨架（含 requirements、目录、可运行的 baseline）。
