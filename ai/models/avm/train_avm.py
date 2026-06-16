"""
M3 — AVM (Automated Valuation Model) + SHAP
===========================================
Trains an explainable home-price model on the cleaned CRMLS sold data and
explains it with SHAP, as a pricing decision-support tool. Predicts ClosePrice
from property + location + time features.

LEAKAGE DISCIPLINE (important): we deliberately EXCLUDE any field that equals or
derives from the answer, or that is only known after the sale:
    ListPrice, OriginalListPrice          -> ≈ the answer
    price_per_sqft, price_ratio, *_ratio  -> derived from ClosePrice
    DaysOnMarket, *_days                  -> only known after the sale closes
An AVM estimates value from the home itself, so none of those may be features.

Outputs (to outputs/, which is gitignored):
    outputs/avm_metrics.json        metrics + top SHAP drivers
    outputs/avm_shap_summary.png    SHAP beeswarm of value drivers
    outputs/avm_model.ubj           trained model

Run:  .venv/bin/python ai/models/avm/train_avm.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# xgboost + shap are imported lazily inside main() so this module can be
# imported (e.g. by unit tests / CI) without the system OpenMP runtime xgboost
# needs on macOS.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = os.path.join(BASE_DIR, "data", "processed", "week7_sold_clean.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "ClosePrice"
NUMERIC = ["LivingArea", "BedroomsTotal", "BathroomsTotalInteger", "YearBuilt",
           "LotSizeSquareFeet", "GarageSpaces", "Latitude", "Longitude",
           "close_year", "close_month", "rate_30yr_fixed"]
BOOLEAN = ["PoolPrivateYN", "FireplaceYN", "ViewYN", "NewConstructionYN"]
CATEGORICAL = ["PropertySubType", "CountyOrParish"]


def load_features():
    cols = [TARGET] + NUMERIC + BOOLEAN + CATEGORICAL
    df = pd.read_csv(DATA, usecols=cols, low_memory=False)
    df = df[df[TARGET] > 0].copy()                      # drop non-positive prices

    X_num = df[NUMERIC]
    X_bool = df[BOOLEAN].apply(lambda s: s.map({True: 1, False: 0}))  # NaN stays NaN
    X_cat = pd.get_dummies(df[CATEGORICAL].astype("object"), prefix=CATEGORICAL)
    X = pd.concat([X_num, X_bool, X_cat], axis=1)
    X.columns = [str(c) for c in X.columns]

    y = np.log1p(df[TARGET].to_numpy())                 # log target (price is right-skewed)
    tkey = (df["close_year"].astype(int) * 12 + df["close_month"].astype(int)).to_numpy()
    return X, y, df[TARGET].to_numpy(), tkey


def main():
    import xgboost as xgb
    import shap
    print("=" * 70)
    print("AVM — Automated Valuation Model (XGBoost + SHAP)")
    print("=" * 70)
    print(f"\nLoading {DATA} ...")
    X, y_log, price, tkey = load_features()
    print(f"  {len(X):,} rows x {X.shape[1]} features (after one-hot)")

    # Time-based split on a MONTH BOUNDARY: train on earlier sales, test on the
    # most recent ~20%. Splitting on whole months (not a row index) ensures no
    # calendar month straddles the cut — otherwise the boundary month leaks into
    # both train and test and inflates the "forward-in-time" evaluation.
    months = np.sort(np.unique(tkey))
    test_months, acc = set(), 0
    for mk in months[::-1]:                 # walk back from the latest month
        test_months.add(mk)
        acc += int((tkey == mk).sum())
        if acc >= 0.2 * len(tkey):
            break
    is_test = np.array([mk in test_months for mk in tkey])
    X_tr, X_te = X[~is_test], X[is_test]
    y_tr, y_te = y_log[~is_test], y_log[is_test]
    price_te = price[is_test]
    tp = sorted(set(zip(X_te["close_year"].astype(int), X_te["close_month"].astype(int))))
    test_period = f"{tp[0][0]}-{tp[0][1]:02d} -> {tp[-1][0]}-{tp[-1][1]:02d}"
    print(f"  time-based split (whole months): test = most recent {len(test_months)} months ({test_period})")

    print("\nTraining XGBoost ...")
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        tree_method="hist", n_jobs=-1, random_state=42,
    )
    model.fit(X_tr, y_tr)

    # --- Evaluate in dollar space ---
    pred = np.expm1(model.predict(X_te))
    mae = mean_absolute_error(price_te, pred)
    rmse = np.sqrt(mean_squared_error(price_te, pred))
    r2 = r2_score(price_te, pred)
    ape = np.abs(pred - price_te) / price_te
    mdape = float(np.median(ape) * 100)             # median abs % error (AVM standard)
    ppe10 = float((ape <= 0.10).mean() * 100)       # % of predictions within 10%

    print("\n--- Test-set performance ---")
    print(f"  MAE   : ${mae:,.0f}")
    print(f"  RMSE  : ${rmse:,.0f}")
    print(f"  R^2   : {r2:.3f}")
    print(f"  MdAPE : {mdape:.1f}%   (median absolute % error)")
    print(f"  PPE10 : {ppe10:.1f}%   (within ±10% of actual)")

    # --- SHAP explainability ---
    print("\nComputing SHAP values (sample of 2,000) ...")
    sample = X_te.sample(min(2000, len(X_te)), random_state=0)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(sample)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    drivers = sorted(zip(X.columns, mean_abs), key=lambda t: -t[1])[:15]

    print("\n--- Top value drivers (mean |SHAP|, log-price space) ---")
    for name, val in drivers:
        print(f"  {name:32s} {val:.4f}")

    shap.summary_plot(shap_vals, sample, show=False, max_display=15)
    plot_path = os.path.join(OUT_DIR, "avm_shap_summary.png")
    plt.title("AVM — SHAP value drivers (test sample)")
    plt.savefig(plot_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"\nSHAP summary plot -> {plot_path}")

    # --- Persist artifacts ---
    model_path = os.path.join(OUT_DIR, "avm_model.ubj")
    model.save_model(model_path)
    metrics = {
        "n_rows": int(len(X)), "n_features": int(X.shape[1]),
        "split": "time-based (most recent 20% by sale month)", "test_period": test_period,
        "mae": mae, "rmse": rmse, "r2": r2, "mdape_pct": mdape, "ppe10_pct": ppe10,
        "top_drivers": [{"feature": n, "mean_abs_shap": float(v)} for n, v in drivers],
        "excluded_leakage": ["ListPrice", "OriginalListPrice", "price_per_sqft",
                             "price_ratio", "close_to_original_list_ratio",
                             "DaysOnMarket", "listing_to_contract_days",
                             "contract_to_close_days"],
    }
    with open(os.path.join(OUT_DIR, "avm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Model -> {model_path}")
    print(f"Metrics -> {os.path.join(OUT_DIR, 'avm_metrics.json')}")
    print("\nAVM — Complete!")


if __name__ == "__main__":
    main()
