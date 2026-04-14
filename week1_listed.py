"""
Week 1 – Listings: Monthly Dataset Aggregation
================================================
Loads and concatenates all monthly CRMLS Listing CSV files (Jan 2024 – Mar 2026)
into a single combined dataset, filters to PropertyType == 'Residential',
and saves the result for downstream analysis.

Input:  CRMLSListing202401.csv – CRMLSListing202603.csv (27 files)
Output: data/processed/week1_listed.csv
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
# Step 1: Load and concatenate all monthly Listing files
# =============================================================================
print("=" * 70)
print("WEEK 1 – LISTINGS: MONTHLY DATASET AGGREGATION")
print("=" * 70)

files = sorted(glob.glob(os.path.join(RAW_DIR, "CRMLSListing*.csv")))
print(f"\nFound {len(files)} monthly Listing files:\n")

dfs = []
for f in files:
    df = pd.read_csv(f, low_memory=False)
    print(f"  {os.path.basename(f)}: {len(df):,} rows, {len(df.columns)} cols")
    dfs.append(df)

listings = pd.concat(dfs, ignore_index=True)
print(f"\n  Total rows after concatenation: {len(listings):,}")
print(f"  Total columns: {len(listings.columns)}")

# =============================================================================
# Step 2: Handle duplicate column names (e.g., PropertyType.1, ListPrice.1)
# These are CSV export artifacts — drop the .1 suffix columns
# =============================================================================
dup_cols = [c for c in listings.columns if c.endswith('.1')]
if dup_cols:
    print(f"\n--- Dropping {len(dup_cols)} duplicate columns ---")
    for c in dup_cols:
        print(f"  {c}")
    listings = listings.drop(columns=dup_cols)
    print(f"  Columns after cleanup: {len(listings.columns)}")

# =============================================================================
# Step 3: PropertyType frequency table (before filter)
# =============================================================================
print(f"\n--- PropertyType distribution (BEFORE filter) ---")
pt_before = listings['PropertyType'].value_counts()
for pt, count in pt_before.items():
    pct = count / len(listings) * 100
    print(f"  {pt}: {count:,} ({pct:.1f}%)")

# =============================================================================
# Step 4: Filter to Residential only
# =============================================================================
rows_before = len(listings)
listings = listings[listings['PropertyType'] == 'Residential'].copy()
rows_after = len(listings)

print(f"\n--- Residential Filter ---")
print(f"  Rows before filter: {rows_before:,}")
print(f"  Rows after filter:  {rows_after:,}")
print(f"  Rows removed:       {rows_before - rows_after:,}")

# =============================================================================
# Step 5: PropertyType frequency table (after filter) — confirm only Residential
# =============================================================================
print(f"\n--- PropertyType distribution (AFTER filter) ---")
pt_after = listings['PropertyType'].value_counts()
for pt, count in pt_after.items():
    print(f"  {pt}: {count:,}")

# =============================================================================
# Step 6: Save combined dataset
# =============================================================================
output_path = os.path.join(OUTPUT_DIR, "week1_listed.csv")
listings.to_csv(output_path, index=False)

print(f"\n--- Output ---")
print(f"  Saved to: {output_path}")
print(f"  Final shape: {listings.shape[0]:,} rows x {listings.shape[1]} columns")
print(f"\nWeek 1 Listings – Complete!")
