"""
Tests for the hardening fixes from the multi-agent self-check: SQL-guard
file-access blocking + over-rejection, robust JSON extraction, flag coercion,
forecast CI non-degeneracy, and the FRED connector's pure logic. All run
without the (gitignored) data, network, or an LLM.
"""
import numpy as np
import pandas as pd
import pytest

from ai.assistant.text_to_sql import is_safe_select
from ai.shared.llm import extract_json
from ai.dataqa.data_quality import _flag_int
from ai.forecast.forecast_market import fit_forecast
from connectors.fred_connector import to_monthly, get_api_key


# --- SQL guard: file access must be blocked (self-check HIGH security) ---
def test_guard_blocks_duckdb_file_functions():
    for bad in [
        "SELECT * FROM read_text('/etc/hosts')",
        "SELECT * FROM read_csv_auto('/tmp/x.csv')",
        "SELECT * FROM glob('/tmp/*')",
        "WITH x AS (SELECT * FROM read_parquet('a.pq')) SELECT * FROM x",
    ]:
        assert not is_safe_select(bad), bad


# --- SQL guard must NOT over-reject valid read-only queries (self-check MED) ---
def test_guard_allows_strings_comments_and_scalars():
    for ok in [
        "SELECT * FROM sold WHERE City = ';'",          # semicolon inside a string
        "SELECT * FROM sold -- todo: drop later",        # keyword inside a comment
        "SELECT replace(City, 'x', 'y') FROM sold",      # legit scalar function
        "SELECT * FROM sold WHERE City = 'a;b'",
    ]:
        assert is_safe_select(ok), ok


# --- robust JSON extraction (brace-in-string) ---
def test_extract_json_ignores_braces_in_strings():
    assert extract_json('noise {"sql": "a } b"} trailing}')["sql"] == "a } b"
    assert extract_json("```json\n{\"a\": 1}\n```")["a"] == 1


# --- data-quality flag coercion ---
def test_flag_int_coercion():
    s = pd.Series([True, False, "True", "true", "False", 1, 0, "1", "0", 1.0, 0.0, np.nan])
    assert _flag_int(s).tolist() == [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0]


# --- forecast CI must stay non-degenerate (highest-value: silent failure) ---
def test_forecast_ci_non_degenerate():
    idx = pd.date_range("2024-01-01", periods=28, freq="MS")
    vals = 100 + 10 * np.sin(np.arange(28) * 2 * np.pi / 12) + np.arange(28) * 0.5
    _, ci = fit_forecast(pd.Series(vals, index=idx), 3)
    widths = (ci.iloc[:, 1] - ci.iloc[:, 0]).to_numpy()
    assert (widths > 1e-6).all(), f"degenerate CI: {widths}"


# --- FRED connector pure logic ---
def test_fred_to_monthly_resamples_weekly_to_monthly():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-04", "2024-01-11", "2024-02-01", "2024-02-08"]),
        "value": [6.0, 7.0, 5.0, 5.0],
    })
    out = to_monthly(df, "rate")
    assert list(out["rate"]) == [6.5, 5.0]


def test_fred_get_api_key_rejects_placeholder(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "your_fred_api_key_here")
    with pytest.raises(RuntimeError):
        get_api_key()
