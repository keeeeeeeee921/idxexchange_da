"""
Week 7 – Sold Transactions: Outlier Detection and Data Quality
==============================================================
Applies IQR-based outlier detection to key numeric fields (ClosePrice,
LivingArea, DaysOnMarket). Flags extreme values without deleting records,
then saves both a fully flagged dataset and a clean filtered dataset.
Includes a before/after comparison of dataset size and key statistics.

Input:  data/processed/week6_sold.csv
Output: data/processed/week7_sold_flagged.csv  (all rows + outlier flags)
        data/processed/week7_sold_clean.csv    (outliers removed)
"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# Paths
# =============================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "week6_sold.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fields to apply IQR outlier detection
IQR_FIELDS = ['ClosePrice', 'LivingArea', 'DaysOnMarket']

# =============================================================================
# Load Week 6 output
# =============================================================================
print("=" * 70)
print("WEEK 7 – SOLD: OUTLIER DETECTION AND DATA QUALITY")
print("=" * 70)

sold = pd.read_csv(INPUT_PATH, low_memory=False)
rows_before = len(sold)
print(f"\nLoaded: {INPUT_PATH}")
print(f"Shape: {rows_before:,} rows x {sold.shape[1]} columns")

# =============================================================================
# Part 1: Business Rule Flags (recap from Week 4-5)
# =============================================================================
print(f"\n{'='*70}")
print("PART 1: BUSINESS RULE FLAGS (recap)")
print(f"{'='*70}")

# These were already added in week4_5 — summarise counts here for context
business_flags = [
    'invalid_price_flag', 'invalid_area_flag', 'invalid_dom_flag',
    'listing_after_close_flag', 'purchase_after_close_flag',
]
print(f"\n  Business rule violations (Week 4-5 flags):")
for flag in business_flags:
    if flag in sold.columns:
        count = int(sold[flag].sum())
        print(f"    {flag}: {count:,} records")

# =============================================================================
# Part 2: IQR Outlier Detection
# =============================================================================
print(f"\n{'='*70}")
print("PART 2: IQR OUTLIER DETECTION")
print(f"{'='*70}")

print("""
  Method: Interquartile Range (IQR)
    lower bound = Q1 - 1.5 × IQR
    upper bound = Q3 + 1.5 × IQR
  Records outside these bounds are flagged, not deleted.
  Why: A $50M estate or a $10K distressed sale can skew market averages —
  but the record itself may be valid and should be preserved in the raw data.
""")

iqr_bounds = {}

for field in IQR_FIELDS:
    if field not in sold.columns:
        print(f"  {field}: NOT FOUND — skipping")
        continue

    series = sold[field].dropna()
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    iqr_bounds[field] = {'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
                         'lower': lower, 'upper': upper}

    flag_col = f"outlier_{field.lower().replace('price', 'price')}_flag"
    # Normalise flag column name
    flag_map = {
        'ClosePrice':    'outlier_price_flag',
        'LivingArea':    'outlier_area_flag',
        'DaysOnMarket':  'outlier_dom_flag',
    }
    flag_col = flag_map[field]

    sold[flag_col] = (
        sold[field].notna() &
        ((sold[field] < lower) | (sold[field] > upper))
    ).astype(int)

    count = int(sold[flag_col].sum())
    pct   = count / len(sold) * 100

    print(f"  {field}:")
    print(f"    Q1={Q1:,.0f}  Q3={Q3:,.0f}  IQR={IQR:,.0f}")
    print(f"    Lower bound : {lower:,.0f}")
    print(f"    Upper bound : {upper:,.0f}")
    print(f"    Outliers    : {count:,} records ({pct:.2f}%)")

    # Percentile context
    p01 = series.quantile(0.01)
    p99 = series.quantile(0.99)
    print(f"    p1={p01:,.0f}  p99={p99:,.0f}  "
          f"min={series.min():,.0f}  max={series.max():,.0f}")
    print()

# Combined outlier flag — any of the three fields is an outlier
flag_cols = ['outlier_price_flag', 'outlier_area_flag', 'outlier_dom_flag']
existing_flags = [f for f in flag_cols if f in sold.columns]
sold['outlier_any_flag'] = sold[existing_flags].max(axis=1)

total_outliers = int(sold['outlier_any_flag'].sum())
print(f"  outlier_any_flag (any field flagged): "
      f"{total_outliers:,} records ({total_outliers/len(sold)*100:.2f}%)")

# =============================================================================
# Part 3: Combined Quality Flag
# =============================================================================
print(f"\n{'='*70}")
print("PART 3: COMBINED DATA QUALITY FLAG")
print(f"{'='*70}")

# A record is considered low-quality if it has ANY business rule violation
# OR any IQR outlier flag
quality_flags = existing_flags + [
    f for f in ['invalid_price_flag', 'invalid_area_flag', 'invalid_dom_flag',
                'listing_after_close_flag', 'purchase_after_close_flag',
                'coord_missing_flag']
    if f in sold.columns
]

sold['quality_issue_flag'] = (
    sold[quality_flags].max(axis=1)
)

total_quality = int(sold['quality_issue_flag'].sum())
print(f"\n  quality_issue_flag (any IQR outlier or business rule violation):")
print(f"    {total_quality:,} records flagged ({total_quality/len(sold)*100:.2f}%)")
print(f"    {len(sold) - total_quality:,} records clean ({(len(sold)-total_quality)/len(sold)*100:.2f}%)")

# =============================================================================
# Part 4: Before / After Comparison
# =============================================================================
print(f"\n{'='*70}")
print("PART 4: BEFORE / AFTER COMPARISON")
print(f"{'='*70}")

clean_mask = sold['outlier_any_flag'] == 0
sold_clean = sold[clean_mask].copy()
rows_after = len(sold_clean)

print(f"\n  {'Metric':<30} {'Before (all)':<20} {'After (IQR clean)':<20} {'Change':>10}")
print(f"  {'-'*30} {'-'*20} {'-'*20} {'-'*10}")

print(f"  {'Row count':<30} {rows_before:<20,} {rows_after:<20,} "
      f"{rows_after-rows_before:>+10,}")

for field in IQR_FIELDS:
    if field not in sold.columns:
        continue
    before_median = sold[field].median()
    after_median  = sold_clean[field].median()
    before_mean   = sold[field].mean()
    after_mean    = sold_clean[field].mean()

    print(f"\n  {field} — Median:")
    print(f"  {'':4} Before : {before_median:>15,.1f}")
    print(f"  {'':4} After  : {after_median:>15,.1f}  "
          f"({(after_median-before_median)/before_median*100:+.2f}%)")
    print(f"  {field} — Mean:")
    print(f"  {'':4} Before : {before_mean:>15,.1f}")
    print(f"  {'':4} After  : {after_mean:>15,.1f}  "
          f"({(after_mean-before_mean)/before_mean*100:+.2f}%)")

# =============================================================================
# Part 5: Outlier Profile (what are the outliers?)
# =============================================================================
print(f"\n{'='*70}")
print("PART 5: OUTLIER PROFILE")
print(f"{'='*70}")

outlier_rows = sold[sold['outlier_any_flag'] == 1]

print(f"\n  Price outliers — top 5 highest ClosePrice:")
if 'outlier_price_flag' in sold.columns:
    price_out = sold[sold['outlier_price_flag'] == 1].nlargest(5, 'ClosePrice')
    cols = [c for c in ['CloseDate', 'CountyOrParish', 'City',
                        'PropertySubType', 'ClosePrice', 'LivingArea'] if c in sold.columns]
    print(price_out[cols].to_string(index=False))

print(f"\n  LivingArea outliers — top 5 largest:")
if 'outlier_area_flag' in sold.columns:
    area_out = sold[sold['outlier_area_flag'] == 1].nlargest(5, 'LivingArea')
    cols = [c for c in ['CloseDate', 'CountyOrParish', 'City',
                        'PropertySubType', 'LivingArea', 'ClosePrice'] if c in sold.columns]
    print(area_out[cols].to_string(index=False))

print(f"\n  DaysOnMarket outliers — top 5 longest:")
if 'outlier_dom_flag' in sold.columns:
    dom_out = sold[sold['outlier_dom_flag'] == 1].nlargest(5, 'DaysOnMarket')
    cols = [c for c in ['CloseDate', 'CountyOrParish', 'City',
                        'PropertySubType', 'DaysOnMarket', 'ClosePrice'] if c in sold.columns]
    print(dom_out[cols].to_string(index=False))

# =============================================================================
# Save Outputs
# =============================================================================
print(f"\n{'='*70}")
print("SAVING OUTPUT")
print(f"{'='*70}")

# Full flagged dataset (all rows preserved)
flagged_path = os.path.join(OUTPUT_DIR, "week7_sold_flagged.csv")
sold.to_csv(flagged_path, index=False)
print(f"\n  Flagged dataset : {flagged_path}")
print(f"  Shape           : {sold.shape[0]:,} rows x {sold.shape[1]} columns")

# Clean filtered dataset (outliers removed — for Tableau and downstream analysis)
clean_path = os.path.join(OUTPUT_DIR, "week7_sold_clean.csv")
sold_clean.to_csv(clean_path, index=False)
print(f"\n  Clean dataset   : {clean_path}")
print(f"  Shape           : {sold_clean.shape[0]:,} rows x {sold_clean.shape[1]} columns")
print(f"  Rows removed    : {rows_before - rows_after:,} "
      f"({(rows_before-rows_after)/rows_before*100:.2f}%)")

print(f"\nWeek 7 Sold – Complete!")
