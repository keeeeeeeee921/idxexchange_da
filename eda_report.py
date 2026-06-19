"""
EDA Report Generator — Weeks 1-7 Review
=======================================
Builds a single self-contained HTML report (charts embedded as base64 + a
Chinese narrative) summarising the Week 1-7 analysis: data quality, price
distributions, market tempo, volume/new-listings, geography/segments, the
mortgage-rate relationship, and a data-quality / improvement section.

Inputs (read-only):
    data/processed/week7_sold_flagged.csv      all sold rows + flags + features
    data/processed/week7_listed_flagged.csv    all listed rows + flags + features
    data/processed/tableau/monthly_market.csv       monthly sold KPIs (cleaned)
    data/processed/tableau/monthly_new_listings.csv  monthly new-listing counts

Output:
    outputs/eda_report.html

Note: all chart text is English (matplotlib lacks a CJK font by default); the
narrative around the charts is Chinese and rendered by the browser.
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 110})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE_DIR, "data", "processed")
TAB = os.path.join(PROC, "tableau")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_HTML = os.path.join(OUT_DIR, "eda_report.html")

SOLD_PATH = os.path.join(PROC, "week7_sold_flagged.csv")
LISTED_PATH = os.path.join(PROC, "week7_listed_flagged.csv")
MONTHLY_PATH = os.path.join(TAB, "monthly_market.csv")
NEWLIST_PATH = os.path.join(TAB, "monthly_new_listings.csv")

# M1: LLM market-narrative module (project root on path so `ai`/`connectors` resolve)
sys.path.insert(0, BASE_DIR)
from ai.reporting.market_narrative import (
    build_market_metrics, generate_narrative, narrative_to_html,
)
from ai.shared.reporting import fig_to_b64 as fig_b64

print("=" * 70)
print("EDA REPORT GENERATOR — WEEKS 1-7")
print("=" * 70)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SECTIONS = []  # list of (html string)
CHART_COUNT = 0


def add_html(html):
    SECTIONS.append(html)


def add_chart(fig, caption=""):
    global CHART_COUNT
    CHART_COUNT += 1
    b64 = fig_b64(fig)
    cap = f'<p class="cap">图 {CHART_COUNT}：{caption}</p>' if caption else ""
    add_html(f'<div class="chart"><img src="data:image/png;base64,{b64}"/>{cap}</div>')
    print(f"  chart {CHART_COUNT}: {caption}")


def fmt(n, money=False):
    if pd.isna(n):
        return "—"
    if money:
        return f"${n:,.0f}"
    return f"{n:,.0f}"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("\nLoading datasets...")
sold = pd.read_csv(SOLD_PATH, low_memory=False)
monthly = pd.read_csv(MONTHLY_PATH)
newlist = pd.read_csv(NEWLIST_PATH)

for col in ["CloseDate"]:
    if col in sold.columns:
        sold[col] = pd.to_datetime(sold[col], errors="coerce")

# The listed dataset is only needed for its row/column count in the overview
# table, so count cheaply instead of loading the full ~300 MB flagged file.
listed_cols = len(pd.read_csv(LISTED_PATH, nrows=0).columns)
with open(LISTED_PATH) as _f:
    listed_rows = sum(1 for _ in _f) - 1  # minus header

print(f"  sold   : {sold.shape[0]:,} x {sold.shape[1]}")
print(f"  listed : {listed_rows:,} x {listed_cols}")

sold_min, sold_max = monthly["yr_mo"].min(), monthly["yr_mo"].max()
list_min, list_max = newlist["yr_mo"].min(), newlist["yr_mo"].max()

# ===========================================================================
# SECTION 0: Header + overview
# ===========================================================================
gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")

# --- M1: AI market narrative (real LLM if ANTHROPIC_API_KEY is set, else stub) ---
print("\n[Section] AI market narrative")
ai_metrics = build_market_metrics(monthly, newlist)
ai_narrative = generate_narrative(ai_metrics)
print(f"  narrative source: {ai_narrative['source']}")
ai_summary_html = narrative_to_html(ai_narrative)

add_html(f"""
<h1>CRMLS 加州住宅市场 — EDA 报告（Week 1–7）</h1>
<p class="meta">生成时间：{gen_time} ｜ 数据：CRMLS Sold &amp; Listing 月度抽取（仅 Residential）</p>
{ai_summary_html}
<h2>1. 概览</h2>
<p>本报告汇总 Week 1–7 流水线产出的探索性分析。数据是加州 CRMLS 的成交（Sold）与挂牌（Listing）
记录，按月抽取后拼接、清洗、做特征工程与异常值检测。下表是两条数据线的规模与覆盖区间：</p>
<table>
  <tr><th>数据集</th><th>记录数</th><th>字段数</th><th>时间覆盖（月）</th></tr>
  <tr><td>Sold（成交，含 flag）</td><td>{sold.shape[0]:,}</td><td>{sold.shape[1]}</td><td>{sold_min} → {sold_max}</td></tr>
  <tr><td>Listing（挂牌，含 flag）</td><td>{listed_rows:,}</td><td>{listed_cols}</td><td>{list_min} → {list_max}</td></tr>
</table>
<p class="flow"><b>流水线：</b>
Week1 拼接 27 个月文件 + 筛选 Residential →
Week2-3 EDA + 缺失分析 + FRED 房贷利率富化 →
Week4-5 日期/数值清洗 + 业务规则 flag + 地理校验 →
Week6 特征工程（价格比率、$/sqft、时间线指标）→
Week7 IQR 异常值检测（flagged + clean 两份）。</p>
""")

# ===========================================================================
# SECTION 1: Data quality
# ===========================================================================
print("\n[Section] Data quality")
add_html("<h2>2. 数据质量</h2>")

# 2.1 Missing values (top columns in sold)
miss = (sold.isnull().mean() * 100).sort_values(ascending=False)
miss = miss[miss > 0].head(20)
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(miss.index[::-1], miss.values[::-1], color="#c44e52")
ax.set_xlabel("Missing %")
ax.set_title("Sold — Top 20 columns by missing rate")
for i, v in enumerate(miss.values[::-1]):
    ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=8)
add_chart(fig, "Sold 数据集缺失率最高的 20 个字段。Week4-5 会删掉缺失 >90% 的列。")

add_html(f"""<p>核心字段（ClosePrice / ListPrice / CloseDate / 经纬度等）完整度高；缺失集中在补充属性列上，
Week4-5 已按 &gt;90% 阈值删列。下面看业务规则与异常值标记。</p>""")

# 2.2 Business-rule + quality flags
flag_specs = [
    ("invalid_price_flag", "价格≤0"),
    ("invalid_area_flag", "面积≤0"),
    ("invalid_dom_flag", "DOM<0"),
    ("listing_after_close_flag", "挂牌晚于成交"),
    ("purchase_after_close_flag", "签约晚于成交"),
    ("coord_missing_flag", "经纬度缺失"),
    ("coord_zero_flag", "经纬度为0"),
    ("coord_out_of_state_flag", "坐标出界(非CA)"),
    ("outlier_price_flag", "价格IQR异常"),
    ("outlier_area_flag", "面积IQR异常"),
    ("outlier_dom_flag", "DOM IQR异常"),
]
present = [(c, lab) for c, lab in flag_specs if c in sold.columns]
counts = [int(sold[c].sum()) for c, _ in present]
labels = [lab for _, lab in present]
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(len(counts)), counts, color="#dd8452")
ax.set_xticks(range(len(counts)))
ax.set_xticklabels([c for c, _ in present], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Records flagged")
ax.set_title("Sold — data-quality flag counts")
for b, v in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=7)
add_chart(fig, "Sold 各类数据质量 flag 的命中量（Week4-5 业务规则 + Week7 IQR 异常值）。")

# Quantify IQR removal impact
n_all = len(sold)
n_out = int(sold["outlier_any_flag"].sum()) if "outlier_any_flag" in sold.columns else 0
add_html(f"""<p><b>异常值处理的影响：</b>Week7 用全局 IQR（Q1−1.5·IQR ~ Q3+1.5·IQR）标记了
<b>{n_out:,}</b> 条记录（{n_out/n_all*100:.1f}%），clean 版本把这些行整体删除。由于价格、面积是右偏分布，
这种做法会过度切除高端/大户型样本——做市场分析时建议<b>优先用中位数</b>，或改用 winsorize 截尾，
而不是直接删行（详见末尾改进建议）。</p>""")

# ===========================================================================
# SECTION 2: Price analysis
# ===========================================================================
print("\n[Section] Price analysis")
add_html("<h2>3. 价格分析</h2>")

# 3.1 ClosePrice distribution (clipped at p99)
cp = sold["ClosePrice"].dropna()
p99 = cp.quantile(0.99)
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(cp[cp <= p99], bins=60, color="#4c72b0", edgecolor="white", alpha=0.85)
ax.axvline(cp.median(), color="#c44e52", ls="--", lw=2, label=f"Median ${cp.median():,.0f}")
ax.axvline(cp.mean(), color="black", ls=":", lw=2, label=f"Mean ${cp.mean():,.0f}")
ax.set_xlabel("ClosePrice (USD, clipped at p99)")
ax.set_ylabel("Frequency")
ax.set_title("Sold — ClosePrice distribution")
ax.legend()
add_chart(fig, f"成交价分布（截到 99 分位 ${p99:,.0f} 以避免长尾压缩）。均值 &gt; 中位数，典型右偏。")

# 3.2 Monthly median close price trend
m = monthly.copy()
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(m["yr_mo"], m["median_close_price"], marker="o", color="#4c72b0", lw=2)
ax.set_title("Monthly median close price")
ax.set_ylabel("Median ClosePrice (USD)")
ax.set_xticks(range(0, len(m), 2))
ax.set_xticklabels(m["yr_mo"][::2], rotation=45, ha="right", fontsize=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
add_chart(fig, "月度中位成交价。呈现明显的季节性（春夏走高、冬季回落）。")

# 3.3 price_per_sqft distribution
if "price_per_sqft" in sold.columns:
    pps = sold["price_per_sqft"].dropna()
    pps = pps[(pps > 0) & (pps <= pps.quantile(0.99))]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pps, bins=60, color="#55a868", edgecolor="white", alpha=0.85)
    ax.axvline(pps.median(), color="#c44e52", ls="--", lw=2, label=f"Median ${pps.median():,.0f}")
    ax.set_xlabel("Price per SqFt (USD, clipped at p99)")
    ax.set_ylabel("Frequency")
    ax.set_title("Sold — price per square foot")
    ax.legend()
    add_chart(fig, "单位面积成交价（$/sqft），对不同户型规模做了归一化。")

# 3.4 close-to-original-list ratio trend (cleaned)
if "avg_close_to_orig_ratio" in monthly.columns:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(m["yr_mo"], m["avg_close_to_orig_ratio"], marker="o", color="#8172b3", lw=2)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_title("Monthly avg close-to-original-list ratio (cleaned)")
    ax.set_ylabel("ClosePrice / OriginalListPrice")
    ax.set_xticks(range(0, len(m), 2))
    ax.set_xticklabels(m["yr_mo"][::2], rotation=45, ha="right", fontsize=8)
    add_chart(fig, "成交价/原始挂牌价（已清洗录入错误）。>1 表示卖家市场，<1 表示买家议价空间大。")

# ===========================================================================
# SECTION 3: Market tempo
# ===========================================================================
print("\n[Section] Market tempo")
add_html("<h2>4. 市场节奏（速度）</h2>")

dom = sold["DaysOnMarket"].dropna()
dom = dom[(dom >= 0) & (dom <= dom.quantile(0.99))]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].hist(dom, bins=50, color="#937860", edgecolor="white", alpha=0.85)
axes[0].axvline(dom.median(), color="#c44e52", ls="--", lw=2, label=f"Median {dom.median():.0f}d")
axes[0].set_xlabel("Days on market (clipped p99)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Sold — DaysOnMarket distribution")
axes[0].legend()
axes[1].plot(m["yr_mo"], m["median_dom"], marker="o", color="#937860", lw=2)
axes[1].set_title("Monthly median DOM")
axes[1].set_ylabel("Median days on market")
axes[1].set_xticks(range(0, len(m), 3))
axes[1].set_xticklabels(m["yr_mo"][::3], rotation=45, ha="right", fontsize=8)
add_chart(fig, "在售天数（DOM）分布与月度中位 DOM。冬季成交更慢（中位 DOM 走高）。")

# ===========================================================================
# SECTION 4: Volume & new listings
# ===========================================================================
print("\n[Section] Volume & new listings")
add_html("<h2>5. 成交量与新增挂牌</h2>")

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.bar(m["yr_mo"], m["closed_sales"], color="#4c72b0", alpha=0.85, label="Closed sales")
nl = newlist.set_index("yr_mo")["new_listings"].reindex(m["yr_mo"]).values
ax.plot(m["yr_mo"], nl, color="#dd8452", marker="o", lw=2, label="New listings")
ax.set_title("Monthly closed sales vs new listings")
ax.set_ylabel("Count")
ax.set_xticks(range(0, len(m), 2))
ax.set_xticklabels(m["yr_mo"][::2], rotation=45, ha="right", fontsize=8)
ax.legend()
add_chart(fig, "月度成交量（柱）对比新增挂牌（线）。新增挂牌领先成交，二者差值反映供给压力。")

# ===========================================================================
# SECTION 5: Geography & segments
# ===========================================================================
print("\n[Section] Geography & segments")
add_html("<h2>6. 地域与物业细分</h2>")

# Top counties by volume, show median price
if "CountyOrParish" in sold.columns:
    cty = (sold.groupby("CountyOrParish")
           .agg(n=("ClosePrice", "count"), med=("ClosePrice", "median"))
           .sort_values("n", ascending=False).head(12).sort_values("med"))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(cty.index, cty["med"], color="#4c72b0")
    ax.set_xlabel("Median ClosePrice (USD)")
    ax.set_title("Median close price — top 12 counties by volume")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    for i, (med, n) in enumerate(zip(cty["med"], cty["n"])):
        ax.text(med, i, f"  ${med/1000:.0f}k (n={n:,})", va="center", fontsize=8)
    add_chart(fig, "成交量前 12 的县，按中位成交价排序。沿海县（如湾区/橙县）显著高于内陆。")

# By PropertySubType
if "PropertySubType" in sold.columns:
    st = (sold.groupby("PropertySubType")
          .agg(n=("ClosePrice", "count"), med=("ClosePrice", "median"))
          .sort_values("n", ascending=False).head(8).sort_values("med"))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(st.index, st["med"], color="#55a868")
    ax.set_xlabel("Median ClosePrice (USD)")
    ax.set_title("Median close price by PropertySubType (top 8 by volume)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    for i, (med, n) in enumerate(zip(st["med"], st["n"])):
        ax.text(med, i, f"  ${med/1000:.0f}k (n={n:,})", va="center", fontsize=8)
    add_chart(fig, "按物业细分类型的中位成交价（取成交量前 8）。")

# ===========================================================================
# SECTION 6: Mortgage rate relationship
# ===========================================================================
if "rate_30yr_fixed" in sold.columns and "yr_mo" in sold.columns:
    print("\n[Section] Mortgage rate")
    add_html("<h2>7. 房贷利率与价格</h2>")
    rate_m = sold.groupby("yr_mo")["rate_30yr_fixed"].mean().reindex(m["yr_mo"])
    fig, ax1 = plt.subplots(figsize=(11, 4.8))
    ax1.plot(m["yr_mo"], m["median_close_price"], color="#4c72b0", marker="o", lw=2)
    ax1.set_ylabel("Median close price", color="#4c72b0")
    ax1.tick_params(axis="y", labelcolor="#4c72b0")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax1.set_xticks(range(0, len(m), 2))
    ax1.set_xticklabels(m["yr_mo"][::2], rotation=45, ha="right", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(m["yr_mo"], rate_m.values, color="#c44e52", marker="s", lw=2)
    ax2.set_ylabel("30yr fixed rate (%)", color="#c44e52")
    ax2.tick_params(axis="y", labelcolor="#c44e52")
    ax2.grid(False)
    ax1.set_title("Median close price vs 30-year fixed mortgage rate")
    add_chart(fig, "月度中位成交价（蓝）对比 FRED 30 年固定房贷利率（红，来自 Week2-3 富化）。")

# ===========================================================================
# SECTION 7: Findings & recommendations
# ===========================================================================
print("\n[Section] Findings")
add_html(f"""
<h2>8. 数据质量发现与改进建议</h2>
<p>对 Week 1–7 全流程审查后，按优先级列出可改进点（已用数据验证）：</p>

<div class="finding p0"><h3>P0 — 管线状态需保持一致</h3>
<p>审查时发现 Listed 管线一度落后于 Sold（week6 之后缺最新月份、少约 3.6 万行），原因是新月份数据加入后
<code>week6→week7</code> 未随 <code>week1→week4_5</code> 一起重跑。本次报告生成前已重跑刷新到一致。
<b>建议：</b>增加一个编排脚本（<code>run_pipeline.py</code> / Makefile）按序运行全部 stage，杜绝"部分刷新"漂移。</p></div>

<div class="finding p1"><h3>P1 — 异常值处理偏激进，且未覆盖工程特征</h3>
<p>Sold 的 IQR clean 删除了约 {n_out/n_all*100:.0f}% 记录，对右偏的价格/面积会过度切除高端样本，使市场分析低估高端市场。
工程比率（<code>close_to_original_list_ratio</code>、<code>price_per_sqft</code>）未纳入异常值检测——这正是
<code>close_to_orig</code> 比率被 $450 这类录入错误污染的根因。<b>建议：</b>仪表盘优先用 median；改用 winsorize 截尾而非删行；
价格类用 log 变换后再 IQR，或按 county / PropertySubType 分段；把比率特征也纳入质量校验。</p></div>

<div class="finding p2"><h3>P2 — 可复现性 / 工程化</h3>
<p>FRED 房贷利率每次实时联网拉取、失败静默回退 NaN → 不可复现，建议缓存一份到 <code>data/</code>。
中间产物用 CSV 导致每个 week 都要重新 <code>to_datetime</code>，建议改 <b>Parquet</b>（保留 dtype、更小更快）。</p></div>

<div class="finding p3"><h3>P3 — 数据卫生</h3>
<p>原始拼接里 Sold/Listed 各有约 590 / 159 条 <code>ListingKey</code> 完全重复，建议 week1 加 <code>drop_duplicates</code>。
<code>ListOfficeName</code> 等文本字段确含内嵌换行符，但经核验 pandas 默认引号可<b>无损 round-trip</b>（raw 与各阶段行数一致、无丢行），非缺陷。
<code>year_month</code> 与 <code>yr_mo</code> 的双月份列冗余已在 week2-3 移除（仅保留 <code>yr_mo</code>）。</p></div>
""")

# ===========================================================================
# Assemble HTML
# ===========================================================================
CSS = """
<style>
  body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
         max-width: 980px; margin: 0 auto; padding: 32px 24px; color: #1a1a1a;
         line-height: 1.7; background: #fafafa; }
  h1 { font-size: 26px; border-bottom: 3px solid #4c72b0; padding-bottom: 10px; }
  h2 { font-size: 20px; margin-top: 40px; color: #2a2a2a; border-left: 4px solid #4c72b0;
       padding-left: 10px; }
  h3 { font-size: 16px; margin: 6px 0; }
  .meta { color: #888; font-size: 13px; }
  .flow { background: #eef2f8; padding: 12px 16px; border-radius: 6px; font-size: 14px; }
  .ai-summary { background: linear-gradient(135deg,#eef4ff,#f7fbff); border: 1px solid #cfe0ff;
                border-left: 5px solid #4c72b0; border-radius: 8px; padding: 16px 20px; margin: 20px 0; }
  .ai-summary .ai-headline { font-size: 16px; font-weight: 600; color: #2a3a55; margin: 4px 0 10px; }
  .ai-badge { font-size: 11px; font-weight: 500; color: #4c72b0; background: #e3edff;
              padding: 2px 8px; border-radius: 10px; vertical-align: middle; margin-left: 8px; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 14px; }
  th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
  th { background: #4c72b0; color: white; }
  tr:nth-child(even) { background: #f4f4f4; }
  .chart { text-align: center; margin: 24px 0; }
  .chart img { max-width: 100%; border: 1px solid #e2e2e2; border-radius: 6px; background: white; }
  .cap { font-size: 13px; color: #666; margin-top: 6px; }
  code { background: #eee; padding: 1px 5px; border-radius: 3px; font-size: 13px; }
  .finding { padding: 12px 16px; margin: 14px 0; border-radius: 6px; border-left: 5px solid; }
  .finding.p0 { background: #fdecea; border-color: #c0392b; }
  .finding.p1 { background: #fef5e7; border-color: #e67e22; }
  .finding.p2 { background: #eef7ed; border-color: #27ae60; }
  .finding.p3 { background: #eef2f8; border-color: #4c72b0; }
</style>
"""

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>CRMLS EDA 报告 — Week 1-7</title>{CSS}</head>
<body>
{''.join(SECTIONS)}
<hr style="margin-top:40px"><p class="meta">由 eda_report.py 自动生成 · 图表 {CHART_COUNT} 张 · 数据为机密 MLS，请勿公开分享。</p>
</body></html>"""

with open(OUT_HTML, "w") as f:
    f.write(html)

print(f"\n{'='*70}")
print(f"  Report written: {OUT_HTML}")
print(f"  Charts: {CHART_COUNT}  |  size: {os.path.getsize(OUT_HTML)/1024:.0f} KB")
print(f"{'='*70}")
print("EDA Report — Complete!")
