"""
M4 — Market forecasting + alerting
==================================
Forecasts the monthly market series (median close price, closed sales) a few
months ahead with SARIMAX, backtests accuracy on a holdout, and raises a simple
deviation alert when the latest month falls outside the model's expectation.

NOTE on data length: the series is only ~28 monthly points (~2.3 years). That is
short for seasonal forecasting — there are barely two annual cycles — so the
forecasts are indicative, not precise. SARIMAX (lighter, with built-in
confidence intervals) is a better fit here than a heavyweight model like Prophet.

Outputs (to outputs/, gitignored):
    outputs/forecast.png      history + forecast + confidence band
    outputs/forecast.json     forecasts, backtest MAPE, alerts

Run:  .venv/bin/python ai/forecast/forecast_market.py
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAB = os.path.join(BASE_DIR, "data", "processed", "tableau")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

HORIZON = 3        # months to forecast
HOLDOUT = 3        # months held out for backtest
CI_ALPHA = 0.20    # 80% confidence interval

TARGETS = [
    {"col": "median_close_price", "label": "中位成交价", "label_en": "Median Close Price", "money": True},
    {"col": "closed_sales", "label": "成交量", "label_en": "Closed Sales", "money": False},
]


def load_series(col):
    df = pd.read_csv(os.path.join(TAB, "monthly_market.csv"))
    idx = pd.to_datetime(df["yr_mo"] + "-01")
    return pd.Series(df[col].to_numpy(dtype=float), index=idx).asfreq("MS")


def _fit(series, order, seasonal_order):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(series, order=order, seasonal_order=seasonal_order,
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


def fit_forecast(series, steps):
    """SARIMAX forecast. Uses a seasonal-AR model (1,1,0)(1,0,0,12): on this
    short 28-point series, seasonal *differencing* (…,1,…,12) collapses the
    forecast variance to a near-zero CI, whereas seasonal AR keeps realistic
    intervals. Falls back to a non-seasonal model if the fit fails."""
    try:
        res = _fit(series, (1, 1, 0), (1, 0, 0, 12))
    except Exception:
        res = _fit(series, (1, 1, 1), (0, 0, 0, 0))
    fc = res.get_forecast(steps=steps)
    return fc.predicted_mean, fc.conf_int(alpha=CI_ALPHA)


def backtest_mape(series):
    train, test = series.iloc[:-HOLDOUT], series.iloc[-HOLDOUT:]
    mean, _ = fit_forecast(train, HOLDOUT)
    return float(np.mean(np.abs((test.to_numpy() - mean.to_numpy()) / test.to_numpy())) * 100)


def latest_alert(series):
    """Fit on all-but-last month, predict that month, flag if actual is outside the CI."""
    mean, ci = fit_forecast(series.iloc[:-1], 1)
    actual = float(series.iloc[-1])
    expected = float(mean.iloc[0])
    lo, hi = float(ci.iloc[0, 0]), float(ci.iloc[0, 1])
    return {
        "month": series.index[-1].strftime("%Y-%m"),
        "actual": actual, "expected": expected, "lo": lo, "hi": hi,
        "deviation_pct": round((actual - expected) / expected * 100, 1),
        "alert": not (lo <= actual <= hi),
    }


def main():
    print("=" * 70)
    print("MARKET FORECAST + ALERTING (SARIMAX)")
    print("=" * 70)

    results = {}
    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(11, 8))

    for ax, tgt in zip(axes, TARGETS):
        col, label, money = tgt["col"], tgt["label"], tgt["money"]
        s = load_series(col)

        mape = backtest_mape(s)
        mean, ci = fit_forecast(s, HORIZON)
        alert = latest_alert(s)

        fc = [{"month": m.strftime("%Y-%m"), "mean": float(v),
               "lo": float(ci.iloc[i, 0]), "hi": float(ci.iloc[i, 1])}
              for i, (m, v) in enumerate(mean.items())]
        results[col] = {"backtest_mape_pct": round(mape, 1), "forecast": fc, "alert": alert}

        # --- print ---
        unit = (lambda x: f"${x:,.0f}") if money else (lambda x: f"{x:,.0f}")
        print(f"\n[{label}]  backtest MAPE = {mape:.1f}%")
        for f in fc:
            print(f"   {f['month']}  预测 {unit(f['mean'])}   "
                  f"(80% CI {unit(f['lo'])} ~ {unit(f['hi'])})")
        flag = "⚠️ 异常" if alert["alert"] else "正常"
        print(f"   最新月 {alert['month']}: 实际 {unit(alert['actual'])} vs 预期 "
              f"{unit(alert['expected'])} ({alert['deviation_pct']:+.1f}%) → {flag}")

        # --- plot ---
        ax.plot(s.index, s.values, marker="o", ms=3, lw=1.5, label="history")
        ax.plot(mean.index, mean.values, marker="s", ms=4, lw=2, color="#c44e52", label="forecast")
        ax.fill_between(mean.index, ci.iloc[:, 0], ci.iloc[:, 1], color="#c44e52", alpha=0.18, label="80% CI")
        ax.set_title(f"{tgt['label_en']}  (backtest MAPE {mape:.1f}%)")
        ax.legend(fontsize=8)
        if money:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "forecast.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()

    with open(os.path.join(OUT_DIR, "forecast.json"), "w") as f:
        json.dump({"horizon_months": HORIZON, "results": results}, f, indent=2, ensure_ascii=False)

    print(f"\nPlot -> {plot_path}")
    print(f"JSON -> {os.path.join(OUT_DIR, 'forecast.json')}")
    print("\nForecast — Complete!  (注：序列仅 28 个月，预测为指示性结果)")


if __name__ == "__main__":
    main()
