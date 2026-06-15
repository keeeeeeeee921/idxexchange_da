"""
Pipeline orchestrator
=====================
Runs every stage of the IDX Exchange data pipeline in dependency order so the
Sold and Listed chains never drift out of sync. Re-run this whenever a new
month of raw CRMLS data is added — running stages by hand is how the Listed
chain fell a month behind the Sold chain (see eda_report.py finding P0).

Dependency chain:
  Sold:   week1_sold -> week2_3_sold -> week4_5_sold -> week6_sold -> week7_sold
  Listed: week1_listed -> week2_3_listed -> week4_5_listed -> week6_listed -> week7_listed
  Then:   week8_tableau_prep  (needs week7_*_clean from BOTH chains)
          eda_report          (needs week7_*_flagged + tableau/monthly_*; embeds the M1 LLM summary)
          county_reports      (M1+: per-county LLM market narratives; needs tableau_sold/listed)
          train_avm           (M3: XGBoost valuation model + SHAP)
          forecast_market     (M4: SARIMAX forecast + alerts)
          data_quality        (M5: rule + IsolationForest data-quality report)

The two reporting stages call an LLM (local Ollama by default) and degrade
gracefully to a deterministic stub if none is available, so the pipeline never
breaks on a missing model. Run with the project venv so every stage gets the
right deps:  .venv/bin/python run_pipeline.py

Usage:
  .venv/bin/python run_pipeline.py              # run the whole pipeline
  .venv/bin/python run_pipeline.py --from week6 # resume from the first stage matching "week6"
  .venv/bin/python run_pipeline.py --list       # show stages and exit
"""

import subprocess
import sys
import os
import time
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ordered so each stage's inputs are produced by an earlier stage.
STAGES = [
    "week1_sold.py",
    "week2_3_sold.py",
    "week4_5_sold.py",
    "week6_sold.py",
    "week7_sold.py",
    "week1_listed.py",
    "week2_3_listed.py",
    "week4_5_listed.py",
    "week6_listed.py",
    "week7_listed.py",
    "week8_tableau_prep.py",
    "eda_report.py",                      # statewide EDA report + M1 LLM summary
    "ai/reporting/county_reports.py",     # M1+ per-county LLM narratives
    "ai/models/avm/train_avm.py",         # M3 AVM home-valuation model + SHAP
    "ai/forecast/forecast_market.py",     # M4 forecasting + alerting
    "ai/dataqa/data_quality.py",          # M5 data-quality report
]


def run_stage(script):
    path = os.path.join(BASE_DIR, script)
    if not os.path.exists(path):
        raise FileNotFoundError(f"stage script not found: {path}")
    print("\n" + "=" * 70)
    print(f"RUN  {script}")
    print("=" * 70, flush=True)
    t0 = time.time()
    # check=True -> raises CalledProcessError on non-zero exit, so the
    # orchestrator stops at the first broken stage instead of feeding stale
    # data downstream.
    subprocess.run([sys.executable, path], cwd=BASE_DIR, check=True)
    print(f"\nOK   {script}  ({time.time() - t0:.1f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Run the data pipeline end to end.")
    ap.add_argument("--from", dest="start", metavar="MATCH",
                    help="resume from the first stage whose filename contains MATCH")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    args = ap.parse_args()

    if args.list:
        for i, s in enumerate(STAGES, 1):
            print(f"{i:2d}. {s}")
        return

    stages = STAGES
    if args.start:
        match = [i for i, s in enumerate(STAGES) if args.start in s]
        if not match:
            sys.exit(f"--from {args.start!r} matched no stage. Use --list to see names.")
        stages = STAGES[match[0]:]

    print(f"Pipeline: {len(stages)} stage(s) to run")
    t0 = time.time()
    for s in stages:
        run_stage(s)

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE — {len(stages)} stage(s) in {time.time() - t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
