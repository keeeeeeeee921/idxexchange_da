"""
FRED live-data connector
========================
A small, reusable client for the St. Louis Fed **FRED REST API**
(https://api.stlouisfed.org/fred). Unlike the lightweight CSV-download endpoint
used inline in week2_3_*.py, this hits the proper REST API:

  * key-based authentication (api_key query param, read from the environment)
  * JSON response parsing
  * graceful error handling (FRED error payloads + HTTP/transport errors)
  * weekly -> monthly resampling so the series can join the monthly MLS data

Stdlib only (urllib + json + pandas) — no extra pip installs required.

Usage:
    # 1. put your key in .env  ->  FRED_API_KEY=xxxxxxxx
    # 2. run the demo:
    python3 connectors/fred_connector.py

    # or import it:
    from connectors.fred_connector import fetch_series_monthly
    monthly = fetch_series_monthly("MORTGAGE30US", observation_start="2024-01-01")
"""

import os
import json
import urllib.parse
import urllib.request
import urllib.error

import pandas as pd

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"


# --------------------------------------------------------------------------- #
# Config / secrets
# --------------------------------------------------------------------------- #
def load_dotenv(path=None):
    """Minimal .env loader (no python-dotenv dependency).

    Reads KEY=VALUE lines from `path` into os.environ without overwriting
    variables that are already set in the real environment.
    """
    if path is None:
        # default: a .env sitting next to the project root (one level up)
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def get_api_key():
    """Return the FRED API key from the environment, or raise a clear error."""
    load_dotenv()
    key = os.environ.get("FRED_API_KEY")
    if not key or key == "your_fred_api_key_here":
        raise RuntimeError(
            "FRED_API_KEY not set. Copy .env.example to .env and add your key "
            "(get one free at https://fredaccount.stlouisfed.org/apikeys)."
        )
    return key


# --------------------------------------------------------------------------- #
# Core fetch
# --------------------------------------------------------------------------- #
def fetch_series(series_id, api_key=None, observation_start=None,
                 observation_end=None, timeout=30):
    """Fetch one FRED series via the REST API and return a tidy DataFrame.

    Returns columns ['date', 'value'] with `value` coerced to float
    (FRED encodes missing observations as '.', which become NaN).
    """
    if api_key is None:
        api_key = get_api_key()

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    if observation_start:
        params["observation_start"] = observation_start
    if observation_end:
        params["observation_end"] = observation_end

    url = FRED_OBSERVATIONS_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "idx-exchange-fred-connector/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # FRED returns a JSON error body (e.g. bad key -> 400 with error_message)
        detail = e.read().decode("utf-8", "ignore")
        try:
            detail = json.loads(detail).get("error_message", detail)
        except (ValueError, AttributeError):
            pass
        raise RuntimeError(f"FRED API HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach FRED API: {e.reason}") from None

    observations = payload.get("observations")
    if observations is None:
        raise RuntimeError(f"Unexpected FRED response (no 'observations'): {payload}")

    df = pd.DataFrame(observations)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # '.' -> NaN
    return df.dropna(subset=["value"]).reset_index(drop=True)


def to_monthly(df, value_name):
    """Resample a [date, value] series to a monthly average keyed by year_month."""
    out = df.copy()
    out["year_month"] = out["date"].dt.to_period("M")
    monthly = (
        out.groupby("year_month")["value"].mean().reset_index()
        .rename(columns={"value": value_name})
    )
    return monthly


def fetch_series_monthly(series_id, value_name=None, **kwargs):
    """Convenience: fetch a series and return it resampled to monthly averages."""
    value_name = value_name or series_id.lower()
    raw = fetch_series(series_id, **kwargs)
    return to_monthly(raw, value_name)


# --------------------------------------------------------------------------- #
# Mortgage rate (MORTGAGE30US): REST-first with no-key CSV fallback
# --------------------------------------------------------------------------- #
def _fetch_mortgage30us_csv_monthly(value_name="rate_30yr_fixed"):
    """No-key fallback: the public FRED graph-CSV endpoint (no API key needed)."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
    df = pd.read_csv(url, parse_dates=["observation_date"])
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return to_monthly(df, value_name)


def fetch_mortgage30us_monthly(value_name="rate_30yr_fixed", observation_start=None):
    """Monthly average 30-yr fixed mortgage rate, REST-first with CSV fallback.

    Tries the FRED REST API (key from .env). If the key is missing or the REST
    call fails for any reason (no key, transport error, FRED outage), it falls
    back to the public no-key CSV endpoint so the pipeline still runs.
    Returns columns ['year_month', value_name].
    """
    try:
        monthly = fetch_series_monthly(
            "MORTGAGE30US", value_name=value_name,
            observation_start=observation_start,
        )
        print("  [fred] source: REST API (JSON, key auth)")
        return monthly
    except Exception as e:  # noqa: BLE001 — any failure should degrade, not crash
        print(f"  [fred] REST unavailable ({e}); falling back to no-key CSV endpoint.")
        return _fetch_mortgage30us_csv_monthly(value_name=value_name)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def main():
    print("Fetching MORTGAGE30US from the FRED REST API (JSON) ...")
    monthly = fetch_series_monthly(
        "MORTGAGE30US",
        value_name="rate_30yr_fixed",
        observation_start="2024-01-01",
    )
    print(f"Got {len(monthly)} monthly observations.")
    print(f"Rate range: {monthly['rate_30yr_fixed'].min():.2f}% – "
          f"{monthly['rate_30yr_fixed'].max():.2f}%\n")
    print("Most recent 6 months:")
    print(monthly.tail(6).to_string(index=False))


if __name__ == "__main__":
    main()
