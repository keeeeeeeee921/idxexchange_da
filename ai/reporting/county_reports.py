"""
M1+ — Per-county market report fan-out
======================================
Scales the M1 LLM narrative from one statewide report to one report PER COUNTY.
This is the real leverage of LLM-augmented reporting: a human can hand-write one
market summary, but not 60 of them every month — an LLM can.

For the top-N counties by volume it computes per-county monthly KPIs, reuses
M1's metric builder + narrative generator, and renders a polished HTML report
(overview table + per-county cards with a price sparkline and the AI write-up).

Run:  .venv/bin/python ai/reporting/county_reports.py
Output: outputs/county_reports.html
"""

import os
import sys
import base64
from io import BytesIO
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
from ai.reporting.market_narrative import build_market_metrics, generate_narrative
from ai.shared import llm

TAB = os.path.join(BASE_DIR, "data", "processed", "tableau")
OUT = os.path.join(BASE_DIR, "outputs", "county_reports.html")
TOP_N = 8


# --------------------------------------------------------------------------- #
# Per-county monthly aggregation
# --------------------------------------------------------------------------- #
def per_county_frames():
    sold = pd.read_csv(os.path.join(TAB, "tableau_sold.csv"), usecols=[
        "CountyOrParish", "yr_mo", "ClosePrice", "DaysOnMarket",
        "price_per_sqft", "close_to_original_list_ratio"])
    listed = pd.read_csv(os.path.join(TAB, "tableau_listed.csv"),
                         usecols=["CountyOrParish", "yr_mo"])

    sold_agg = (sold.groupby(["CountyOrParish", "yr_mo"]).agg(
        closed_sales=("ClosePrice", "size"),
        median_close_price=("ClosePrice", "median"),
        median_dom=("DaysOnMarket", "median"),
        median_price_per_sqft=("price_per_sqft", "median"),
        avg_close_to_orig_ratio=("close_to_original_list_ratio", "mean"),
    ).reset_index())
    list_agg = (listed.groupby(["CountyOrParish", "yr_mo"])
                .size().reset_index(name="new_listings"))

    top = (sold_agg.groupby("CountyOrParish")["closed_sales"].sum()
           .sort_values(ascending=False).head(TOP_N).index.tolist())
    return sold_agg, list_agg, top


def sparkline_b64(series):
    fig, ax = plt.subplots(figsize=(3.2, 0.6))
    ax.plot(range(len(series)), series, color="#4c72b0", lw=1.5)
    ax.fill_between(range(len(series)), series, series.min(), color="#4c72b0", alpha=0.12)
    ax.axis("off")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True, dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def _pct_html(x):
    if x is None:
        return '<span class="flat">—</span>'
    cls = "up" if x > 0 else ("down" if x < 0 else "flat")
    sign = "+" if x > 0 else ""
    return f'<span class="{cls}">{sign}{x}%</span>'


def card_html(county, metrics, narrative, spark):
    lat = metrics["latest"]
    mom = metrics["mom_change_pct"]
    yoy = metrics.get("yoy_change_pct") or {}
    paras = "".join(f"<p>{p}</p>" for p in str(narrative.get("summary", "")).split("\n\n") if p.strip())
    watch = "".join(f"<li>{w}</li>" for w in narrative.get("watch", []))
    return f"""
    <div class="card">
      <div class="card-head">
        <h3>{county}</h3>
        <img class="spark" src="data:image/png;base64,{spark}"/>
      </div>
      <div class="chips">
        <div class="chip"><span>中位成交价</span><b>${lat['median_close_price']:,}</b>
            <em>环比 {_pct_html(mom.get('median_close_price'))} · 同比 {_pct_html(yoy.get('median_close_price'))}</em></div>
        <div class="chip"><span>成交量</span><b>{lat['closed_sales']:,}</b>
            <em>环比 {_pct_html(mom.get('closed_sales'))}</em></div>
        <div class="chip"><span>中位 DOM</span><b>{lat['median_dom']:.0f} 天</b></div>
        <div class="chip"><span>成交/原始挂牌</span><b>{lat['avg_close_to_orig_ratio']}</b></div>
      </div>
      <p class="headline">{narrative.get('headline','')}</p>
      {paras}
      <p class="watch-label">值得深挖</p>
      <ul class="watch">{watch}</ul>
    </div>"""


def overview_row(county, metrics):
    lat = metrics["latest"]
    mom = metrics["mom_change_pct"]
    yoy = metrics.get("yoy_change_pct") or {}
    return (f"<tr><td class='cty'>{county}</td>"
            f"<td>${lat['median_close_price']:,}</td>"
            f"<td>{_pct_html(mom.get('median_close_price'))}</td>"
            f"<td>{_pct_html(yoy.get('median_close_price'))}</td>"
            f"<td>{lat['closed_sales']:,}</td>"
            f"<td>{lat['median_dom']:.0f}</td></tr>")


CSS = """
<style>
 body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;max-width:1080px;
      margin:0 auto;padding:32px 24px;background:#f6f7f9;color:#1c2330;line-height:1.65;}
 .hero{background:linear-gradient(135deg,#2a3a55,#4c72b0);color:#fff;padding:22px 26px;border-radius:14px;}
 .hero h1{margin:0 0 6px;font-size:23px;} .hero .meta{opacity:.85;font-size:13px;}
 .badge{display:inline-block;background:rgba(255,255,255,.18);padding:2px 10px;border-radius:10px;font-size:12px;margin-left:8px;}
 h2{font-size:18px;margin:30px 0 12px;border-left:4px solid #4c72b0;padding-left:10px;}
 table{border-collapse:collapse;width:100%;background:#fff;border-radius:10px;overflow:hidden;
       box-shadow:0 1px 3px rgba(0,0,0,.06);font-size:14px;}
 th,td{padding:9px 12px;text-align:right;border-bottom:1px solid #eef0f3;}
 th{background:#2a3a55;color:#fff;font-weight:600;} th:first-child,td.cty{text-align:left;font-weight:600;}
 tr:last-child td{border-bottom:none;}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:6px;}
 .card{background:#fff;border-radius:12px;padding:16px 18px;box-shadow:0 1px 3px rgba(0,0,0,.07);
       border-top:3px solid #4c72b0;}
 .card-head{display:flex;justify-content:space-between;align-items:center;}
 .card-head h3{margin:0;font-size:17px;color:#2a3a55;} .spark{height:34px;opacity:.9;}
 .chips{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 4px;}
 .chip{background:#f3f6fb;border-radius:8px;padding:6px 10px;font-size:12px;flex:1;min-width:120px;}
 .chip span{color:#6b7686;display:block;font-size:11px;} .chip b{font-size:15px;color:#1c2330;}
 .chip em{display:block;font-style:normal;color:#6b7686;font-size:11px;margin-top:2px;}
 .headline{font-weight:600;color:#2a3a55;margin:10px 0 6px;}
 .watch-label{font-weight:600;margin:8px 0 2px;font-size:13px;color:#4c72b0;}
 ul.watch{margin:0;padding-left:18px;font-size:13px;color:#3a4250;} ul.watch li{margin:2px 0;}
 .up{color:#27ae60;font-weight:600;} .down{color:#c0392b;font-weight:600;} .flat{color:#8a93a0;}
 .note{color:#8a93a0;font-size:12px;margin-top:24px;}
 @media(max-width:760px){.grid{grid-template-columns:1fr;}}
</style>"""


def main():
    print("Building per-county frames ...")
    sold_agg, list_agg, counties = per_county_frames()
    provider, model = llm.resolve_provider(), None
    model = llm.resolve_model(provider)
    print(f"Top {len(counties)} counties: {', '.join(counties)}")
    print(f"LLM: {provider}:{model}\n")

    cards, rows = [], []
    for i, c in enumerate(counties, 1):
        m = sold_agg[sold_agg.CountyOrParish == c].drop(columns="CountyOrParish").sort_values("yr_mo")
        nl = list_agg[list_agg.CountyOrParish == c].drop(columns="CountyOrParish").sort_values("yr_mo")
        metrics = build_market_metrics(m, nl)
        print(f"  [{i}/{len(counties)}] {c} -> generating narrative ...")
        narrative = generate_narrative(metrics)
        spark = sparkline_b64(m["median_close_price"].reset_index(drop=True))
        cards.append(card_html(c, metrics, narrative, spark))
        rows.append(overview_row(c, metrics))

    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    badge = f"🤖 {provider}:{model}"
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>CRMLS 分县市场报告</title>{CSS}</head><body>
<div class="hero"><h1>CRMLS 加州住宅 · 分县市场报告 <span class="badge">{badge}</span></h1>
<div class="meta">生成时间 {gen_time} ｜ 按成交量 Top {len(counties)} 县 ｜ 每县叙述由 LLM 自动生成（数据为机密 MLS，仅供内部）</div></div>
<h2>概览</h2>
<table><tr><th>县</th><th>中位成交价</th><th>环比</th><th>同比</th><th>成交量(最新月)</th><th>中位DOM</th></tr>
{''.join(rows)}</table>
<h2>分县详情</h2>
<div class="grid">{''.join(cards)}</div>
<p class="note">由 county_reports.py 自动生成 · {len(counties)} 份 AI 叙述 · 演示「把市场叙述规模化到每个县」。</p>
</body></html>"""

    with open(OUT, "w") as f:
        f.write(html)
    print(f"\nReport -> {OUT}")
    print(f"  {len(counties)} county narratives, {os.path.getsize(OUT)/1024:.0f} KB")


if __name__ == "__main__":
    main()
