"""
Capstone — 1-page Market Intelligence Report
============================================
Auto-composes the internship's final deliverable (handbook Weeks 11-12) for a
chosen county: Market Overview, Pricing Trends, Market Activity, Competitive
Landscape, and LLM-written Key Takeaways. It ties the whole project together —
the cleaned data, the engineered metrics, the M1 narrative engine, and the
competitive analysis — into one stakeholder-ready page.

Run:  .venv/bin/python ai/reporting/market_brief.py "Orange"
Output: outputs/market_brief_<county>.html
"""

import os
import sys
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
from ai.reporting.market_narrative import (
    build_market_metrics, generate_narrative, monthly_sold_frame,
)
from ai.shared import llm
from ai.shared.reporting import fig_to_b64, fmt_pct, summary_to_paras, watch_to_li

TAB = os.path.join(BASE_DIR, "data", "processed", "tableau")


def load_county(county):
    sold = pd.read_csv(os.path.join(TAB, "tableau_sold.csv"), usecols=[
        "CountyOrParish", "yr_mo", "ClosePrice", "DaysOnMarket", "price_per_sqft",
        "close_to_original_list_ratio", "ListAgentFullName", "ListOfficeName"])
    listed = pd.read_csv(os.path.join(TAB, "tableau_listed.csv"),
                         usecols=["CountyOrParish", "yr_mo"])
    s = sold[sold["CountyOrParish"] == county]
    if s.empty:
        avail = ", ".join(sorted(sold["CountyOrParish"].dropna().unique())[:20])
        raise SystemExit(f"County '{county}' not found. Try one of: {avail}")
    lst = listed[listed["CountyOrParish"] == county]
    return s, lst


def monthly_frames(sold, listed):
    sold_m = monthly_sold_frame(sold, by="yr_mo").sort_values("yr_mo")
    list_m = listed.groupby("yr_mo").size().reset_index(name="new_listings")
    return sold_m, list_m


def trend_chart(sold_m, list_m):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3))
    a1.plot(range(len(sold_m)), sold_m["median_close_price"], color="#4c72b0", lw=2)
    a1.set_title("Median close price", fontsize=10)
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    a1.set_xticks([])
    nl = list_m.set_index("yr_mo")["new_listings"].reindex(sold_m["yr_mo"]).values
    a2.bar(range(len(sold_m)), sold_m["closed_sales"], color="#4c72b0", alpha=.7, label="Closed sales")
    a2.plot(range(len(sold_m)), nl, color="#dd8452", lw=2, label="New listings")
    a2.set_title("Activity: closed sales vs new listings", fontsize=10)
    a2.legend(fontsize=8)
    a2.set_xticks([])
    return fig_to_b64(fig, dpi=110)


def competitive(sold, n=5):
    def top(col):
        g = (sold.dropna(subset=[col]).groupby(col)["ClosePrice"]
             .agg(volume="sum", units="size").sort_values("volume", ascending=False).head(n))
        return [(name, row.volume, int(row.units)) for name, row in g.iterrows()]
    return top("ListAgentFullName"), top("ListOfficeName")


def main():
    county = sys.argv[1] if len(sys.argv) > 1 else "Orange"
    print(f"Building market brief for {county} ...")
    sold, listed = load_county(county)
    sold_m, list_m = monthly_frames(sold, listed)
    metrics = build_market_metrics(sold_m, list_m)
    narrative = generate_narrative(metrics)
    chart = trend_chart(sold_m, list_m)
    agents, offices = competitive(sold)

    lat, mom = metrics["latest"], metrics["mom_change_pct"]
    yoy = metrics.get("yoy_change_pct") or {}

    pct = fmt_pct   # shared signed-percent formatter (None -> "—")

    def comp_rows(rows):
        return "".join(
            f"<tr><td class='l'>{nm}</td><td>${v/1e6:.1f}M</td><td>{u:,}</td></tr>"
            for nm, v, u in rows)

    overview = summary_to_paras(narrative["summary"])
    takeaways = watch_to_li(narrative.get("watch", []))
    gen = datetime.now().strftime("%Y-%m-%d")
    cov = metrics["coverage"]
    provider = llm.resolve_provider()
    badge = f"{provider}:{llm.resolve_model(provider)}"

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>{county} Market Intelligence Brief</title>{CSS}</head><body>
<div class="page">
  <div class="hd">
    <div><h1>{county} County — Market Intelligence Brief</h1>
      <div class="sub">CRMLS Residential ｜ {cov['first_month']} – {cov['last_month']} ｜ Generated {gen}</div></div>
    <div class="ai">🤖 {badge}</div>
  </div>

  <div class="kpis">
    <div class="k"><span>Median Close Price</span><b>${lat['median_close_price']:,}</b><em>MoM {pct(mom['median_close_price'])} · YoY {pct(yoy.get('median_close_price'))}</em></div>
    <div class="k"><span>Closed Sales (mo)</span><b>{lat['closed_sales']:,}</b><em>MoM {pct(mom['closed_sales'])}</em></div>
    <div class="k"><span>Median DOM</span><b>{lat['median_dom']:.0f} d</b></div>
    <div class="k"><span>$ / SqFt</span><b>${lat['median_price_per_sqft']:.0f}</b></div>
    <div class="k"><span>Sold/List Ratio</span><b>{lat['avg_close_to_orig_ratio']}</b></div>
  </div>

  <div class="cols">
    <div class="main">
      <h2>Market Overview</h2>{overview}
      <img class="chart" src="data:image/png;base64,{chart}"/>
      <h2>Key Takeaways</h2><ul class="take">{takeaways}</ul>
    </div>
    <div class="side">
      <h2>Competitive Landscape</h2>
      <h3>Top Listing Agents</h3>
      <table><tr><th class="l">Agent</th><th>Vol</th><th>Units</th></tr>{comp_rows(agents)}</table>
      <h3>Top Listing Offices</h3>
      <table><tr><th class="l">Office</th><th>Vol</th><th>Units</th></tr>{comp_rows(offices)}</table>
    </div>
  </div>
  <div class="ft">Auto-generated by market_brief.py · narrative grounded in the metrics above · confidential MLS data — internal use only.</div>
</div>
</body></html>"""

    out = os.path.join(BASE_DIR, "outputs", f"market_brief_{county.replace(' ', '_')}.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"  narrative source: {narrative['source']}")
    print(f"Report -> {out}")


CSS = """
<style>
 body{background:#e9ebee;margin:0;font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;color:#1c2330;}
 .page{max-width:920px;margin:24px auto;background:#fff;padding:28px 32px;box-shadow:0 2px 10px rgba(0,0,0,.1);}
 .hd{display:flex;justify-content:space-between;align-items:flex-start;border-bottom:3px solid #4c72b0;padding-bottom:10px;}
 .hd h1{margin:0;font-size:22px;color:#2a3a55;} .sub{color:#6b7686;font-size:12px;margin-top:3px;}
 .ai{font-size:11px;color:#4c72b0;background:#eef3fb;padding:3px 9px;border-radius:10px;white-space:nowrap;}
 .kpis{display:flex;gap:10px;margin:16px 0;}
 .k{flex:1;background:#f5f7fa;border-radius:8px;padding:9px 11px;text-align:center;}
 .k span{display:block;color:#6b7686;font-size:10.5px;} .k b{font-size:17px;display:block;margin:1px 0;}
 .k em{font-style:normal;color:#6b7686;font-size:10px;}
 .cols{display:flex;gap:22px;margin-top:8px;} .main{flex:2;} .side{flex:1;}
 h2{font-size:15px;color:#2a3a55;border-left:4px solid #4c72b0;padding-left:8px;margin:16px 0 6px;}
 h3{font-size:12.5px;color:#4c72b0;margin:12px 0 4px;}
 p{font-size:13px;margin:5px 0;} .chart{width:100%;margin:10px 0;border:1px solid #eee;border-radius:6px;}
 ul.take{font-size:12.5px;padding-left:18px;margin:4px 0;} ul.take li{margin:3px 0;}
 table{border-collapse:collapse;width:100%;font-size:11.5px;margin-bottom:4px;}
 th,td{padding:4px 6px;text-align:right;border-bottom:1px solid #eef0f3;} th{background:#2a3a55;color:#fff;}
 th.l,td.l{text-align:left;} .ft{margin-top:18px;border-top:1px solid #eee;padding-top:8px;color:#98a0ab;font-size:10.5px;}
</style>"""


if __name__ == "__main__":
    main()
