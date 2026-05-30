"""
Week 7 – Listings: Outlier Detection and Data Quality
======================================================
Applies IQR-based outlier detection to key numeric fields (ListPrice,
LivingArea, DaysOnMarket). Flags extreme values without deleting records,
then saves both a fully flagged dataset and a clean filtered dataset.
Includes a before/after comparison of dataset size and key statistics.

Input:  data/processed/week6_listed.csv
Output: data/processed/week7_listed_flagged.csv  (all rows + outlier flags)
        data/processed/week7_listed_clean.csv    (outliers removed)
"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# Paths
# =============================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "week6_listed.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IQR_FIELDS = ['ListPrice', 'LivingArea', 'DaysOnMarket']

# =============================================================================
# Load Week 6 output
# =============================================================================
print("=" * 70)
print("WEEK 7 – LISTINGS: OUTLIER DETECTION AND DATA QUALITY")
print("=" * 70)

listings = pd.read_csv(INPUT_PATH, low_memory=False)
rows_before = len(listings)
print(f"\nLoaded: {INPUT_PATH}")
print(f"Shape: {rows_before:,} rows x {listings.shape[1]} columns")

# =============================================================================
# Part 1: Business Rule Flags (recap from Week 4-5)
# =============================================================================
print(f"\n{'='*70}")
print("PART 1: BUSINESS RULE FLAGS (recap)")
print(f"{'='*70}")

business_flags = [
    'invalid_price_flag', 'invalid_area_flag', 'invalid_dom_flag',
    'listing_after_close_flag', 'purchase_after_close_flag',
]
print(f"\n  Business rule violations (Week 4-5 flags):")
for flag in business_flags:
    if flag in listings.columns:
        count = int(listings[flag].sum())
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
""")

flag_map = {
    'ListPrice':    'outlier_price_flag',
    'LivingArea':   'outlier_area_flag',
    'DaysOnMarket': 'outlier_dom_flag',
}

for field in IQR_FIELDS:
    if field not in listings.columns:
        print(f"  {field}: NOT FOUND — skipping")
        continue

    series = listings[field].dropna()
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    flag_col = flag_map[field]
    listings[flag_col] = (
        listings[field].notna() &
        ((listings[field] < lower) | (listings[field] > upper))
    ).astype(int)

    count = int(listings[flag_col].sum())
    pct   = count / len(listings) * 100

    print(f"  {field}:")
    print(f"    Q1={Q1:,.0f}  Q3={Q3:,.0f}  IQR={IQR:,.0f}")
    print(f"    Lower bound : {lower:,.0f}")
    print(f"    Upper bound : {upper:,.0f}")
    print(f"    Outliers    : {count:,} records ({pct:.2f}%)")
    p01 = series.quantile(0.01)
    p99 = series.quantile(0.99)
    print(f"    p1={p01:,.0f}  p99={p99:,.0f}  "
          f"min={series.min():,.0f}  max={series.max():,.0f}")
    print()

# Combined outlier flag
flag_cols = [f for f in flag_map.values() if f in listings.columns]
listings['outlier_any_flag'] = listings[flag_cols].max(axis=1)

total_outliers = int(listings['outlier_any_flag'].sum())
print(f"  outlier_any_flag (any field flagged): "
      f"{total_outliers:,} records ({total_outliers/len(listings)*100:.2f}%)")

# =============================================================================
# Part 3: Combined Quality Flag
# =============================================================================
print(f"\n{'='*70}")
print("PART 3: COMBINED DATA QUALITY FLAG")
print(f"{'='*70}")

quality_flags = flag_cols + [
    f for f in ['invalid_price_flag', 'invalid_area_flag', 'invalid_dom_flag',
                'listing_after_close_flag', 'purchase_after_close_flag',
                'coord_missing_flag']
    if f in listings.columns
]

listings['quality_issue_flag'] = listings[quality_flags].max(axis=1)
total_quality = int(listings['quality_issue_flag'].sum())
print(f"\n  quality_issue_flag (any IQR outlier or business rule violation):")
print(f"    {total_quality:,} records flagged ({total_quality/len(listings)*100:.2f}%)")
print(f"    {len(listings)-total_quality:,} records clean "
      f"({(len(listings)-total_quality)/len(listings)*100:.2f}%)")

# =============================================================================
# Part 4: Before / After Comparison
# =============================================================================
print(f"\n{'='*70}")
print("PART 4: BEFORE / AFTER COMPARISON")
print(f"{'='*70}")

clean_mask    = listings['outlier_any_flag'] == 0
listings_clean = listings[clean_mask].copy()
rows_after    = len(listings_clean)

print(f"\n  {'Metric':<30} {'Before (all)':<20} {'After (IQR clean)':<20} {'Change':>10}")
print(f"  {'-'*30} {'-'*20} {'-'*20} {'-'*10}")
print(f"  {'Row count':<30} {rows_before:<20,} {rows_after:<20,} "
      f"{rows_after-rows_before:>+10,}")

for field in IQR_FIELDS:
    if field not in listings.columns:
        continue
    before_median = listings[field].median()
    after_median  = listings_clean[field].median()
    before_mean   = listings[field].mean()
    after_mean    = listings_clean[field].mean()

    print(f"\n  {field} — Median:")
    print(f"  {'':4} Before : {before_median:>15,.1f}")
    print(f"  {'':4} After  : {after_median:>15,.1f}  "
          f"({(after_median-before_median)/before_median*100:+.2f}%)")
    print(f"  {field} — Mean:")
    print(f"  {'':4} Before : {before_mean:>15,.1f}")
    print(f"  {'':4} After  : {after_mean:>15,.1f}  "
          f"({(after_mean-before_mean)/before_mean*100:+.2f}%)")

# =============================================================================
# Part 5: Outlier Profile
# =============================================================================
print(f"\n{'='*70}")
print("PART 5: OUTLIER PROFILE")
print(f"{'='*70}")

print(f"\n  Price outliers — top 5 highest ListPrice:")
if 'outlier_price_flag' in listings.columns:
    top = listings[listings['outlier_price_flag'] == 1].nlargest(5, 'ListPrice')
    cols = [c for c in ['ListingContractDate', 'CountyOrParish', 'City',
                        'PropertySubType', 'ListPrice', 'LivingArea', 'MlsStatus']
            if c in listings.columns]
    print(top[cols].to_string(index=False))

print(f"\n  LivingArea outliers — top 5 largest:")
if 'outlier_area_flag' in listings.columns:
    top = listings[listings['outlier_area_flag'] == 1].nlargest(5, 'LivingArea')
    cols = [c for c in ['ListingContractDate', 'CountyOrParish', 'City',
                        'PropertySubType', 'LivingArea', 'ListPrice']
            if c in listings.columns]
    print(top[cols].to_string(index=False))

print(f"\n  DaysOnMarket outliers — top 5 longest:")
if 'outlier_dom_flag' in listings.columns:
    top = listings[listings['outlier_dom_flag'] == 1].nlargest(5, 'DaysOnMarket')
    cols = [c for c in ['ListingContractDate', 'CountyOrParish', 'City',
                        'PropertySubType', 'DaysOnMarket', 'ListPrice']
            if c in listings.columns]
    print(top[cols].to_string(index=False))

# =============================================================================
# Save Outputs
# =============================================================================
print(f"\n{'='*70}")
print("SAVING OUTPUT")
print(f"{'='*70}")

flagged_path = os.path.join(OUTPUT_DIR, "week7_listed_flagged.csv")
listings.to_csv(flagged_path, index=False)
print(f"\n  Flagged dataset : {flagged_path}")
print(f"  Shape           : {listings.shape[0]:,} rows x {listings.shape[1]} columns")

clean_path = os.path.join(OUTPUT_DIR, "week7_listed_clean.csv")
listings_clean.to_csv(clean_path, index=False)
print(f"\n  Clean dataset   : {clean_path}")
print(f"  Shape           : {listings_clean.shape[0]:,} rows x {listings_clean.shape[1]} columns")
print(f"  Rows removed    : {rows_before - rows_after:,} "
      f"({(rows_before-rows_after)/rows_before*100:.2f}%)")

print(f"\nWeek 7 Listings – Complete!")
