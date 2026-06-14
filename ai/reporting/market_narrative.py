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
import urllib.request
import urllib.error

import pandas as pd

# Reuse the tiny .env loader from the FRED connector (single source of truth).
try:
    from connectors.fred_connector import load_dotenv
except Exception:  # pragma: no cover - fallback if import path differs
    def load_dotenv(path=None):
        return

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-4-6"  # good quality / low cost for a report; override via ANTHROPIC_MODEL


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
    m = monthly.sort_values("yr_mo").reset_index(drop=True)
    nl = newlist.sort_values("yr_mo").set_index("yr_mo")

    last = m.iloc[-1]
    prev = m.iloc[-2] if len(m) >= 2 else last
    yoy = m.iloc[-13] if len(m) >= 13 else None  # same month, prior year

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
            "median_close_price": _pct_change(last["median_close_price"], prev["median_close_price"]),
            "closed_sales": _pct_change(last["closed_sales"], prev["closed_sales"]),
            "median_dom": _pct_change(last["median_dom"], prev["median_dom"]),
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


def render_prompt(metrics):
    schema = (
        '{"headline": "<一句话标题>", '
        '"summary": "<2-3 段市场综述，用 \\n\\n 分段>", '
        '"watch": ["<值得深挖/异常关注点>", "..."]}'
    )
    return (
        "下面是 CRMLS 加州住宅市场的月度 KPI（JSON）。请基于这些数字撰写一份市场综述。\n\n"
        f"数据：\n{json.dumps(metrics, ensure_ascii=False, indent=2)}\n\n"
        "要求：\n"
        "1) headline：一句话点出当前市场状态。\n"
        "2) summary：2-3 段，覆盖价格走势(环比/同比)、市场速度(DOM)、成交量与供需、"
        "以及相对峰值的位置。只用上面给出的数字。\n"
        "3) watch：2-4 条「值得深挖 / 异常关注」，指向数据里值得进一步调查的点。\n\n"
        f"严格只输出如下结构的 JSON（不要解释、不要 markdown 代码块）：\n{schema}"
    )


# --------------------------------------------------------------------------- #
# 3. LLM call (Anthropic Messages API via urllib) with stub fallback
# --------------------------------------------------------------------------- #
def _get_api_key():
    load_dotenv()
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or key.startswith("your_"):
        return None
    return key


def _call_anthropic(prompt, api_key, model, timeout=60):
    body = json.dumps({
        "model": model,
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        ANTHROPIC_URL, data=body, method="POST",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    # Messages API: {"content": [{"type":"text","text": "..."}], ...}
    parts = [c.get("text", "") for c in payload.get("content", []) if c.get("type") == "text"]
    return "".join(parts).strip()


def _parse_json_narrative(text):
    """Tolerant parse: strip markdown fences, locate the JSON object."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        t = t[4:] if t.lower().startswith("json") else t
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1:
        t = t[start:end + 1]
    return json.loads(t)


def _stub_narrative(metrics):
    """Deterministic placeholder so the report builds with no API key."""
    lat = metrics["latest"]
    mom = metrics["mom_change_pct"]
    yoy = metrics.get("yoy_change_pct") or {}
    sd = metrics["supply_demand"]["new_listings_to_sales_ratio"]
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
        f"当月新增挂牌/成交比约 {sd}，反映供需相对节奏。"
    )
    return {
        "headline": f"{lat['month']} 市场快照（占位示例，未接入 LLM）",
        "summary": summary,
        "watch": [
            "环比/同比价格变化是否由季节性驱动，需结合往年同月比较。",
            f"新增挂牌/成交比 {sd} 偏离 1 的程度，提示供给或需求侧压力。",
            "中位 DOM 的月度波动是否对应利率或季节变化。",
        ],
        "source": "stub",
    }


def generate_narrative(metrics, api_key=None, model=None):
    """Return {headline, summary, watch[], source}. Uses Claude if a key is
    available, otherwise a deterministic stub (so it always returns something)."""
    api_key = api_key or _get_api_key()
    if not api_key:
        return _stub_narrative(metrics)

    model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    try:
        raw = _call_anthropic(render_prompt(metrics), api_key, model)
        data = _parse_json_narrative(raw)
        data.setdefault("headline", "")
        data.setdefault("summary", "")
        data.setdefault("watch", [])
        data["source"] = f"llm:{model}"
        return data
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError, KeyError) as e:
        stub = _stub_narrative(metrics)
        stub["headline"] += f"  [LLM 调用失败，已回退：{e}]"
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
    badge_label = "🤖 LLM 生成" if badge.startswith("llm") else "⚙️ 占位示例（未接入 LLM）"
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
