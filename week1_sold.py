"""
Week 1 – Sold Transactions: Monthly Dataset Aggregation
=========================================================
Loads and concatenates all monthly CRMLS Sold CSV files (Jan 2024 – Mar 2026)
into a single combined dataset, filters to PropertyType == 'Residential',
and saves the result for downstream analysis.

Input:  CRMLSSold202401.csv – CRMLSSold202603.csv (27 files)
Output: data/processed/week1_sold.csv
"""

import pandas as pd
import glob
import os

# =============================================================================
# Paths
# =============================================================================
RAW_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(RAW_DIR, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Step 1: Load and concatenate all monthly Sold files
# =============================================================================
print("=" * 70)
print("WEEK 1 – SOLD: MONTHLY DATASET AGGREGATION")
print("=" * 70)

files = sorted(glob.glob(os.path.join(RAW_DIR, "CRMLSSold*.csv")))
print(f"\nFound {len(files)} monthly Sold files:\n")

dfs = []
for f in files:
    df = pd.read_csv(f, low_memory=False)
    print(f"  {os.path.basename(f)}: {len(df):,} rows")
    dfs.append(df)

sold = pd.concat(dfs, ignore_index=True)
print(f"\n  Total rows after concatenation: {len(sold):,}")
print(f"  Total columns: {len(sold.columns)}")

# =============================================================================
# Step 2: PropertyType frequency table (before filter)
# =============================================================================
print(f"\n--- PropertyType distribution (BEFORE filter) ---")
pt_before = sold['PropertyType'].value_counts()
for pt, count in pt_before.items():
    pct = count / len(sold) * 100
    print(f"  {pt}: {count:,} ({pct:.1f}%)")

# =============================================================================
# Step 3: Filter to Residential only
# =============================================================================
rows_before = len(sold)
sold = sold[sold['PropertyType'] == 'Residential'].copy()
rows_after = len(sold)

print(f"\n--- Residential Filter ---")
print(f"  Rows before filter: {rows_before:,}")
print(f"  Rows after filter:  {rows_after:,}")
print(f"  Rows removed:       {rows_before - rows_after:,}")

# =============================================================================
# Step 4: PropertyType frequency table (after filter) — confirm only Residential
# =============================================================================
print(f"\n--- PropertyType distribution (AFTER filter) ---")
pt_after = sold['PropertyType'].value_counts()
for pt, count in pt_after.items():
    print(f"  {pt}: {count:,}")

# =============================================================================
# Step 5: Save combined dataset
# =============================================================================
output_path = os.path.join(OUTPUT_DIR, "week1_sold.csv")
sold.to_csv(output_path, index=False)

print(f"\n--- Output ---")
print(f"  Saved to: {output_path}")
print(f"  Final shape: {sold.shape[0]:,} rows x {sold.shape[1]} columns")
print(f"\nWeek 1 Sold – Complete!")
