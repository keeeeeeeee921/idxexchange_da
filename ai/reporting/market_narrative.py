"""
LLM market-narrative generator (M1)
===================================
Turns the monthly market KPIs into an auto-written market commentary, replacing
the hand-coded narrative paragraphs in eda_report.py.

Pipeline:  monthly KPI frames  ->  compact metrics dict  ->  prompt  ->  Claude
           (Anthropic Messages API, REST/JSON, key from .env)  ->  {headline,
           summary, watch[]}  ->  HTML block embedded in the report.

Runs WITHOUT a key today: if ANTHROPIC_API_KEY is missing it returns a
deterministic stub narrative built from the same metrics, clearly labelled, so
the report still builds. Add the key later to light up the real LLM — no other
code change needed.

Stdlib only (urllib + json) — no SDK install required.
"""

import os
import json

import pandas as pd

from ai.shared import llm  # shared provider-agnostic LLM client (anthropic|ollama|stub)


# --------------------------------------------------------------------------- #
# 1. Metrics extraction (deterministic — always runs, no LLM)
# --------------------------------------------------------------------------- #
def _pct_change(curr, prev):
    if prev in (0, None) or pd.isna(prev) or pd.isna(curr):
        return None
    return round((curr - prev) / prev * 100, 1)


def build_market_metrics(monthly, newlist):
    """Compute a compact, LLM-friendly dict of market KPIs from the two
    monthly frames (monthly_market.csv + monthly_new_listings.csv)."""
    m = monthly.drop_duplicates("yr_mo", keep="last").sort_values("yr_mo").reset_index(drop=True)
    nl = newlist.drop_duplicates("yr_mo", keep="last").sort_values("yr_mo").set_index("yr_mo")

    last = m.iloc[-1]
    # Match prior month / prior year by ACTUAL calendar month, not by row
    # position — robust to gaps (a county missing a month would otherwise make
    # iloc[-13] the wrong "same month last year").
    by_ym = m.set_index("yr_mo")
    latest_p = pd.Period(last["yr_mo"], freq="M")

    def _row(period):
        key = str(period)
        return by_ym.loc[key] if key in by_ym.index else None

    prev = _row(latest_p - 1)   # true previous calendar month (None if missing)
    yoy = _row(latest_p - 12)   # same month a year earlier (None if missing)

    def _mom(col):  # month-over-month only against the immediately prior month
        return None if prev is None else _pct_change(last[col], prev[col])

    latest_ym = last["yr_mo"]
    new_listings_latest = float(nl["new_listings"].get(latest_ym, float("nan")))

    # peak / trough of the median-price series
    price = m.set_index("yr_mo")["median_close_price"]
    peak_ym, peak_val = price.idxmax(), price.max()
    trough_ym, trough_val = price.idxmin(), price.min()

    metrics = {
        "coverage": {
            "first_month": m["yr_mo"].iloc[0],
            "last_month": latest_ym,
            "n_months": int(len(m)),
        },
        "latest": {
            "month": latest_ym,
            "median_close_price": round(float(last["median_close_price"])),
            "closed_sales": int(last["closed_sales"]),
            "median_dom": round(float(last["median_dom"]), 1),
            "median_price_per_sqft": round(float(last["median_price_per_sqft"]), 1),
            "avg_close_to_orig_ratio": round(float(last["avg_close_to_orig_ratio"]), 4),
            "new_listings": None if pd.isna(new_listings_latest) else int(new_listings_latest),
        },
        "mom_change_pct": {
            "median_close_price": _mom("median_close_price"),
            "closed_sales": _mom("closed_sales"),
            "median_dom": _mom("median_dom"),
        },
        "yoy_change_pct": None if yoy is None else {
            "median_close_price": _pct_change(last["median_close_price"], yoy["median_close_price"]),
            "closed_sales": _pct_change(last["closed_sales"], yoy["closed_sales"]),
        },
        "series": {
            "peak_median_price": {"month": peak_ym, "value": round(float(peak_val))},
            "trough_median_price": {"month": trough_ym, "value": round(float(trough_val))},
            "latest_vs_peak_pct": _pct_change(last["median_close_price"], peak_val),
        },
        "supply_demand": {
            # >1 means new supply is outpacing closings this month
            "new_listings_to_sales_ratio": (
                None if pd.isna(new_listings_latest)
                else round(new_listings_latest / last["closed_sales"], 2)
            ),
        },
    }
    return metrics


# --------------------------------------------------------------------------- #
# 2. Prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a real-estate market analyst writing for an internal MLS analytics "
    "team. Write in concise, professional Simplified Chinese. Ground every claim "
    "ONLY in the numbers provided — never invent figures. Return STRICT JSON only."
)


def _fmt_pct(x):
    if x is None:
        return "数据不足"
    return f"+{x}%" if x >= 0 else f"{x}%"


def _facts_block(metrics):
    """Pre-labelled, unambiguous fact sheet. Small models mis-narrate raw JSON
    keys (MoM vs YoY, sign, decimals); spelling out each number's 口径 removes
    that interpretation burden and sharply cuts numeric errors."""
    cov, lat = metrics["coverage"], metrics["latest"]
    mom = metrics["mom_change_pct"]
    yoy = metrics.get("yoy_change_pct") or {}
    sr, sd = metrics["series"], metrics["supply_demand"]
    nl = "数据不足" if lat["new_listings"] is None else f"{lat['new_listings']:,} 套"
    ratio = "数据不足" if sd["new_listings_to_sales_ratio"] is None else sd["new_listings_to_sales_ratio"]
    return "\n".join([
        f"- 数据覆盖：{cov['first_month']} 至 {cov['last_month']}，共 {cov['n_months']} 个月",
        f"- 最新月份：{lat['month']}",
        f"- 中位成交价：${lat['median_close_price']:,}"
        f"（环比 {_fmt_pct(mom.get('median_close_price'))}；同比 {_fmt_pct(yoy.get('median_close_price'))}）",
        f"- 成交量：{lat['closed_sales']:,} 套"
        f"（环比 {_fmt_pct(mom.get('closed_sales'))}；同比 {_fmt_pct(yoy.get('closed_sales'))}）",
        f"- 中位在售天数(DOM)：{lat['median_dom']} 天（环比 {_fmt_pct(mom.get('median_dom'))}）",
        f"- 中位单价：${lat['median_price_per_sqft']}/sqft",
        f"- 成交价/原始挂牌价：{lat['avg_close_to_orig_ratio']}（>1 偏卖方市场，<1 偏买方议价）",
        f"- 当月新增挂牌：{nl}；新增挂牌/成交比：{ratio}",
        f"- 历史峰值中位价：{sr['peak_median_price']['month']} 的 ${sr['peak_median_price']['value']:,}"
        f"；当前较峰值 {_fmt_pct(sr['latest_vs_peak_pct'])}",
        f"- 历史最低中位价：{sr['trough_median_price']['month']} 的 ${sr['trough_median_price']['value']:,}",
    ])


def render_prompt(metrics):
    schema = (
        '{"headline": "<一句话标题>", '
        '"summary": "<2-3 段市场综述，用 \\n\\n 分段>", '
        '"watch": ["<值得深挖/异常关注点>", "..."]}'
    )
    return (
        "你是房地产市场分析师。下面是 CRMLS 加州住宅市场的『事实清单』（已标注口径）。\n\n"
        f"事实清单：\n{_facts_block(metrics)}\n\n"
        "硬性规则：\n"
        "- 只能引用事实清单里给出的数字与方向；禁止自行换算、推断或编造任何新数字。\n"
        "- 禁止编造事实清单中没有的定性结论（例如不得说『连续 N 个月增长』『创历史新高』"
        "『上个月为 X』等未给出的判断或数值）。\n"
        "- 『环比』=与上一个月相比；『同比』=与去年同一个月相比；切勿混淆二者。\n"
        "- 严格照搬正负号与小数位（例如 -1.5% 不可写成 -15%）。\n\n"
        "写作要求：\n"
        "1) headline：一句话点出当前市场状态。\n"
        "2) summary：2-3 段，覆盖价格(环比/同比)、市场速度(DOM)、成交量与供需、相对峰值位置。\n"
        "3) watch：2-4 条「值得深挖 / 异常关注」。\n\n"
        f"严格只输出如下结构的 JSON（不要解释、不要 markdown 代码块）：\n{schema}"
    )


# --------------------------------------------------------------------------- #
# 3. Narrative generation (LLM calls delegated to ai.shared.llm)
# --------------------------------------------------------------------------- #
def _parse_json_narrative(text):
    """Tolerant parse of the LLM's JSON reply (shared brace-matched extractor)."""
    return llm.extract_json(text)


def _stub_narrative(metrics):
    """Deterministic placeholder so the report builds with no API key."""
    lat = metrics["latest"]
    mom = metrics["mom_change_pct"]
    yoy = metrics.get("yoy_change_pct") or {}
    sd = metrics["supply_demand"]["new_listings_to_sales_ratio"]
    sd_txt = "数据不足" if sd is None else sd
    price_mom = mom.get("median_close_price")
    price_yoy = yoy.get("median_close_price")

    def arrow(x):
        if x is None:
            return "—"
        return f"+{x}%" if x >= 0 else f"{x}%"

    summary = (
        f"截至 {lat['month']}，中位成交价 ${lat['median_close_price']:,}"
        f"（环比 {arrow(price_mom)}，同比 {arrow(price_yoy)}），"
        f"成交量 {lat['closed_sales']:,} 套，中位在售天数 {lat['median_dom']} 天。\n\n"
        f"成交价/原始挂牌价比为 {lat['avg_close_to_orig_ratio']}，"
        f"当月新增挂牌/成交比约 {sd_txt}，反映供需相对节奏。"
    )
    return {
        "headline": f"{lat['month']} 市场快照（占位示例，未接入 LLM）",
        "summary": summary,
        "watch": [
            "环比/同比价格变化是否由季节性驱动，需结合往年同月比较。",
            f"新增挂牌/成交比 {sd_txt} 偏离 1 的程度，提示供给或需求侧压力。",
            "中位 DOM 的月度波动是否对应利率或季节变化。",
        ],
        "source": "stub",
    }


def generate_narrative(metrics, provider=None, model=None):
    """Return {headline, summary, watch[], source}.

    Provider via the `provider` arg, else LLM_PROVIDER env (anthropic|ollama|
    stub), else auto-detect. Any failure degrades gracefully to the stub.
    """
    provider = llm.resolve_provider(provider)
    if provider == "stub":
        return _stub_narrative(metrics)

    model = llm.resolve_model(provider, model)
    try:
        raw = llm.complete(
            render_prompt(metrics), system=SYSTEM_PROMPT, force_json=True,
            provider=provider, model=model, temperature=0.3,
        )
        data = _parse_json_narrative(raw)
        data.setdefault("headline", "")
        data.setdefault("summary", "")
        data.setdefault("watch", [])
        data["source"] = f"{provider}:{model}"
        return data
    except Exception as e:  # noqa: BLE001 — any failure should degrade, not crash
        stub = _stub_narrative(metrics)
        stub["headline"] += f"  [LLM({provider}) 调用失败，已回退：{e}]"
        return stub


# --------------------------------------------------------------------------- #
# 4. HTML rendering (for embedding in eda_report.py)
# --------------------------------------------------------------------------- #
def narrative_to_html(narrative):
    paras = "".join(
        f"<p>{p}</p>" for p in str(narrative.get("summary", "")).split("\n\n") if p.strip()
    )
    watch = "".join(f"<li>{w}</li>" for w in narrative.get("watch", []))
    badge = narrative.get("source", "stub")
    badge_label = "⚙️ 占位示例（未接入 LLM）" if badge.startswith("stub") else f"🤖 LLM 生成（{badge}）"
    return f"""
<div class="ai-summary">
  <h2 style="border:none;padding:0;margin:0 0 4px 0;">AI 市场综述
    <span class="ai-badge">{badge_label}</span></h2>
  <p class="ai-headline">{narrative.get('headline', '')}</p>
  {paras}
  <p style="margin:10px 0 4px;"><b>值得深挖</b></p>
  <ul>{watch}</ul>
</div>
"""


# --------------------------------------------------------------------------- #
# Demo / self-test
# --------------------------------------------------------------------------- #
def main():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tab = os.path.join(base, "data", "processed", "tableau")
    monthly = pd.read_csv(os.path.join(tab, "monthly_market.csv"))
    newlist = pd.read_csv(os.path.join(tab, "monthly_new_listings.csv"))

    metrics = build_market_metrics(monthly, newlist)
    print("=== metrics ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    narrative = generate_narrative(metrics)
    print(f"\n=== narrative (source={narrative['source']}) ===")
    print("HEADLINE:", narrative["headline"])
    print("SUMMARY:\n", narrative["summary"])
    print("WATCH:")
    for w in narrative["watch"]:
        print("  -", w)


if __name__ == "__main__":
    main()
