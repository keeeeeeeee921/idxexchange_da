"""
Unit tests for the core logic of the AI modules. These run WITHOUT the
(confidential, gitignored) data and WITHOUT heavy/system-dependent libraries
(xgboost/shap/streamlit/statsmodels) — so they're fast and CI-friendly.
"""
import pandas as pd

from ai.assistant.text_to_sql import is_safe_select
from ai.reporting.market_narrative import (
    build_market_metrics, _parse_json_narrative, _fmt_pct,
)
from ai.models.avm import train_avm


# --------------------------------------------------------------------------- #
# M2 — SQL safety guard
# --------------------------------------------------------------------------- #
def test_is_safe_select_allows_read_only():
    assert is_safe_select("SELECT * FROM sold")
    assert is_safe_select("WITH x AS (SELECT 1 AS a) SELECT a FROM x")
    assert is_safe_select("select count(*) from sold where yr_mo = '2026-04'")


def test_is_safe_select_blocks_mutations():
    for bad in [
        "DROP TABLE sold",
        "DELETE FROM sold",
        "INSERT INTO sold VALUES (1)",
        "UPDATE sold SET ClosePrice = 0",
        "CREATE TABLE t AS SELECT 1",
        "ATTACH 'x.db'",
        "COPY sold TO 'out.csv'",
        "SELECT 1; DROP TABLE sold",     # stacked statements
    ]:
        assert not is_safe_select(bad), bad


# --------------------------------------------------------------------------- #
# M1 — metric math (calendar-month MoM/YoY, gap robustness)
# --------------------------------------------------------------------------- #
def _frame(months, price, sales):
    n = len(months)
    monthly = pd.DataFrame({
        "yr_mo": months,
        "closed_sales": sales,
        "median_close_price": price,
        "median_dom": [20] * n,
        "median_price_per_sqft": [500] * n,
        "avg_close_to_orig_ratio": [1.0] * n,
    })
    newlist = pd.DataFrame({"yr_mo": months, "new_listings": [s * 2 for s in sales]})
    return monthly, newlist


def test_mom_yoy_by_calendar_month():
    months = [f"2024-{m:02d}" for m in range(1, 13)] + ["2025-01", "2025-02"]
    price = [100000] * 14
    price[1] = 100000   # 2024-02 (same month last year)
    price[12] = 200000  # 2025-01 (prev month)
    price[13] = 220000  # 2025-02 (latest)
    sales = [500] * 14
    sales[13] = 1100
    m, nl = _frame(months, price, sales)
    met = build_market_metrics(m, nl)

    assert met["latest"]["month"] == "2025-02"
    assert met["mom_change_pct"]["median_close_price"] == 10.0    # 200k -> 220k
    assert met["yoy_change_pct"]["median_close_price"] == 120.0   # 100k -> 220k
    assert met["supply_demand"]["new_listings_to_sales_ratio"] == 2.0


def test_yoy_none_when_prior_year_missing():
    # only 6 months -> no same-month-last-year row -> YoY must be None, not wrong
    months = [f"2025-{m:02d}" for m in range(1, 7)]
    m, nl = _frame(months, [100000] * 6, [500] * 6)
    met = build_market_metrics(m, nl)
    assert met["yoy_change_pct"] is None


# --------------------------------------------------------------------------- #
# M3 — leakage discipline
# --------------------------------------------------------------------------- #
def test_avm_features_exclude_leakage():
    feats = set(train_avm.NUMERIC + train_avm.BOOLEAN + train_avm.CATEGORICAL)
    leakage = {
        "ListPrice", "OriginalListPrice", "price_per_sqft", "price_ratio",
        "close_to_original_list_ratio", "DaysOnMarket",
        "listing_to_contract_days", "contract_to_close_days",
    }
    assert feats.isdisjoint(leakage), feats & leakage


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_fmt_pct():
    assert _fmt_pct(1.4) == "+1.4%"
    assert _fmt_pct(-1.5) == "-1.5%"
    assert _fmt_pct(0) == "+0%"
    assert _fmt_pct(None) == "数据不足"


def test_parse_json_narrative_tolerant():
    assert _parse_json_narrative('```json\n{"a": 1}\n```')["a"] == 1
    assert _parse_json_narrative('noise {"b": 2} trailing')["b"] == 2
