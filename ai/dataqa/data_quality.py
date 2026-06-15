"""
M5 — Automated data-quality agent
=================================
A standalone data-quality report over the full (flagged) sold dataset:

  1. Missing-value analysis      — null rate per column, flag >90% empty
  2. Rule-based flag summary     — the business-rule / IQR flags from the pipeline
  3. ML anomaly detection (NEW)  — IsolationForest finds MULTIVARIATE anomalies:
     records that look fine column-by-column but are weird in combination
     (e.g. a high price on a tiny lot). Crucially it surfaces issues the rules
     miss — reported as "novel" anomalies.
  4. Overall quality score

Rules catch what you thought to check for; the model catches what you didn't.

Run:  .venv/bin/python ai/dataqa/data_quality.py
Output: outputs/data_quality.html
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE_DIR, "data", "processed", "week7_sold_flagged.csv")
OUT = os.path.join(BASE_DIR, "outputs", "data_quality.html")

FLAG_COLS = [
    ("invalid_price_flag", "价格 ≤ 0"), ("invalid_area_flag", "面积 ≤ 0"),
    ("invalid_dom_flag", "DOM < 0"), ("invalid_beds_flag", "卧室数无效"),
    ("invalid_baths_flag", "卫浴数无效"), ("listing_after_close_flag", "挂牌晚于成交"),
    ("purchase_after_close_flag", "签约晚于成交"), ("negative_timeline_flag", "时间线为负"),
    ("coord_missing_flag", "经纬度缺失"), ("coord_zero_flag", "经纬度为 0"),
    ("coord_positive_lon_flag", "经度为正(应为负)"), ("coord_out_of_state_flag", "坐标出界(非 CA)"),
    ("outlier_price_flag", "价格 IQR 异常"), ("outlier_area_flag", "面积 IQR 异常"),
    ("outlier_dom_flag", "DOM IQR 异常"),
]
IF_FEATURES = ["ClosePrice", "LivingArea", "BedroomsTotal", "BathroomsTotalInteger",
               "DaysOnMarket", "price_per_sqft", "LotSizeSquareFeet", "YearBuilt"]
CONTEXT = ["City", "CountyOrParish", "PropertySubType"]


def _flag_int(s):
    return s.map({True: 1, False: 0, "True": 1, "False": 0,
                  1: 1, 0: 0, "1": 1, "0": 0}).fillna(0).astype(int)


def main():
    print("Loading flagged dataset ...")
    df = pd.read_csv(DATA, low_memory=False)
    n = len(df)
    print(f"  {n:,} rows x {df.shape[1]} columns")

    # 1. Missing values
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    high_miss = miss[miss > 90]

    # 2. Rule-based flags
    flag_rows = []
    for col, label in FLAG_COLS:
        if col in df.columns:
            c = int(_flag_int(df[col]).sum())
            flag_rows.append((label, col, c, c / n * 100))
    flag_rows.sort(key=lambda r: -r[2])
    rule_flagged = (_flag_int(df["quality_issue_flag"]).astype(bool)
                    if "quality_issue_flag" in df.columns
                    else pd.Series(False, index=df.index))
    clean_rate = (1 - rule_flagged.mean()) * 100

    # 3. IsolationForest multivariate anomaly detection
    print("Running IsolationForest ...")
    feats = [c for c in IF_FEATURES if c in df.columns]
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42, n_jobs=-1)
    pred = iso.fit_predict(X)
    df["_if_anom"] = pred == -1
    df["_if_score"] = iso.score_samples(X)
    n_anom = int(df["_if_anom"].sum())
    novel = df["_if_anom"] & ~rule_flagged          # anomalies the rules did NOT catch
    n_novel = int(novel.sum())
    print(f"  {n_anom:,} anomalies ({n_novel:,} novel — missed by rules)")

    examples = df[novel].nsmallest(8, "_if_score")  # most anomalous novel records

    # ---- HTML ----
    miss_rows = "".join(
        f"<tr><td class='l'>{c}</td><td>{v:.1f}%</td></tr>" for c, v in miss.head(15).items()
    ) or "<tr><td colspan=2>无缺失</td></tr>"
    flag_html = "".join(
        f"<tr><td class='l'>{lab}</td><td><code>{col}</code></td><td>{c:,}</td><td>{p:.2f}%</td></tr>"
        for lab, col, c, p in flag_rows
    )
    ex_cols = [c for c in CONTEXT + ["ClosePrice", "LivingArea", "BedroomsTotal",
                                     "BathroomsTotalInteger", "price_per_sqft", "YearBuilt"] if c in df.columns]
    ex_head = "".join(f"<th>{c}</th>" for c in ex_cols)
    ex_rows = ""
    for _, r in examples.iterrows():
        cells = ""
        for c in ex_cols:
            v = r[c]
            if isinstance(v, float):
                v = f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2f}"
            cells += f"<td>{v}</td>"
        ex_rows += f"<tr>{cells}</tr>"

    gen = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>CRMLS 数据质量报告</title>{CSS}</head><body>
<div class="hero"><h1>CRMLS Sold · 数据质量报告 <span class="badge">质量分 {clean_rate:.1f}%</span></h1>
<div class="meta">生成 {gen} ｜ 数据 week7_sold_flagged.csv ｜ 规则校验 + IsolationForest 多变量异常检测（机密数据，仅供内部）</div></div>

<div class="kpis">
  <div class="kpi"><span>记录数</span><b>{n:,}</b></div>
  <div class="kpi"><span>规则判定洁净率</span><b>{clean_rate:.1f}%</b></div>
  <div class="kpi"><span>规则命中(任一)</span><b>{int(rule_flagged.sum()):,}</b></div>
  <div class="kpi"><span>IF 异常</span><b>{n_anom:,}</b></div>
  <div class="kpi accent"><span>新发现异常(规则漏掉)</span><b>{n_novel:,}</b></div>
</div>

<h2>1. 缺失值（Top 15）</h2>
<p class="sub">超过 90% 缺失的列：<b>{len(high_miss)}</b> 个（建议在清洗阶段删除）。</p>
<table><tr><th class="l">列</th><th>缺失率</th></tr>{miss_rows}</table>

<h2>2. 规则校验标记</h2>
<p class="sub">来自管线的业务规则与 IQR 异常标记（每条记录可命中多个）。</p>
<table><tr><th class="l">含义</th><th class="l">字段</th><th>命中数</th><th>占比</th></tr>{flag_html}</table>

<h2>3. IsolationForest 多变量异常 <span class="newtag">ML</span></h2>
<p class="sub">模型在 {len(feats)} 个数值特征上学习"正常"组合，标出最异常的 1%。其中
<b class="accent-t">{n_novel:,}</b> 条<b>未被任何规则命中</b>——这正是 ML 相对规则的增量价值
（单列都正常、但组合起来不合理）。下表为最异常的几条：</p>
<table class="ex"><tr>{ex_head}</tr>{ex_rows}</table>

<p class="note">由 data_quality.py 自动生成 · 规则 + ML 双层数据质量体检 · 全程只读。</p>
</body></html>"""

    with open(OUT, "w") as f:
        f.write(html)
    print(f"\nReport -> {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)")


CSS = """
<style>
 body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;max-width:980px;
      margin:0 auto;padding:32px 24px;background:#f6f7f9;color:#1c2330;line-height:1.6;}
 .hero{background:linear-gradient(135deg,#3a2a55,#7155a8);color:#fff;padding:22px 26px;border-radius:14px;}
 .hero h1{margin:0 0 6px;font-size:22px;} .hero .meta{opacity:.85;font-size:13px;}
 .badge{background:rgba(255,255,255,.2);padding:2px 10px;border-radius:10px;font-size:13px;margin-left:8px;}
 .kpis{display:flex;gap:12px;margin:18px 0;flex-wrap:wrap;}
 .kpi{flex:1;min-width:130px;background:#fff;border-radius:10px;padding:12px 14px;box-shadow:0 1px 3px rgba(0,0,0,.06);}
 .kpi span{color:#6b7686;font-size:12px;display:block;} .kpi b{font-size:21px;}
 .kpi.accent{background:#fff4e8;border:1px solid #f0c089;} .accent-t{color:#d2691e;}
 h2{font-size:18px;margin:28px 0 6px;border-left:4px solid #7155a8;padding-left:10px;}
 .sub{color:#5a6472;font-size:13px;margin:4px 0 10px;}
 .newtag{background:#7155a8;color:#fff;font-size:11px;padding:1px 7px;border-radius:8px;vertical-align:middle;}
 table{border-collapse:collapse;width:100%;background:#fff;border-radius:10px;overflow:hidden;
       box-shadow:0 1px 3px rgba(0,0,0,.06);font-size:13px;}
 th,td{padding:8px 11px;text-align:right;border-bottom:1px solid #eef0f3;}
 th{background:#3a2a55;color:#fff;font-weight:600;} th.l,td.l{text-align:left;}
 tr:last-child td{border-bottom:none;} code{background:#eee;padding:1px 5px;border-radius:3px;font-size:12px;}
 table.ex{font-size:12px;} .note{color:#8a93a0;font-size:12px;margin-top:22px;}
</style>"""


if __name__ == "__main__":
    main()
