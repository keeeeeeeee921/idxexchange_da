"""
Week 8-10 – Tableau Data Preparation
====================================
Prepares lean, Tableau-ready extracts from the Week 7 clean datasets and
builds the pre-aggregated monthly tables that power the Market Analysis
dashboard (market_analysis.twbx).

Why a separate prep step:
  - The Week 7 clean files carry 80-96 columns. Tableau loads faster and is
    easier to build against when we keep only the fields the dashboards use.
  - Pre-aggregated monthly tables make KPI tiles and trend lines trivial to
    build and give us a validation reference for the row-level views.

Row-level extracts keep one row per transaction/listing so Tableau can do its
own filtering (city / county / zip / PropertySubType) and aggregation.

Input:  data/processed/week7_sold_clean.csv
        data/processed/week7_listed_clean.csv
Output: data/processed/tableau/tableau_sold.csv          (lean row-level sold)
        data/processed/tableau/tableau_listed.csv        (lean row-level listings)
        data/processed/tableau/monthly_market.csv        (monthly sold metrics)
        data/processed/tableau/monthly_new_listings.csv  (monthly new-listing counts)
"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# Paths
# =============================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
SOLD_PATH   = os.path.join(PROC_DIR, "week7_sold_clean.csv")
LISTED_PATH = os.path.join(PROC_DIR, "week7_listed_clean.csv")
OUT_DIR     = os.path.join(PROC_DIR, "tableau")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("WEEK 8-10 – TABLEAU DATA PREPARATION")
print("=" * 70)


def keep_existing(df, cols):
    """Return only the columns that actually exist in df, preserving order."""
    return [c for c in cols if c in df.columns]


# =============================================================================
# Part 1: Lean row-level SOLD extract
# =============================================================================
print(f"\n{'='*70}")
print("PART 1: SOLD ROW-LEVEL EXTRACT")
print(f"{'='*70}")

sold = pd.read_csv(SOLD_PATH, low_memory=False)
print(f"\nLoaded: {SOLD_PATH}")
print(f"Shape:  {sold.shape[0]:,} rows x {sold.shape[1]} columns")

# Re-parse dates (lost on CSV round-trip)
for col in ['CloseDate', 'PurchaseContractDate', 'ListingContractDate']:
    if col in sold.columns:
        sold[col] = pd.to_datetime(sold[col], errors='coerce')

# Data-quality guard: close_to_original_list_ratio
# ~0.25% of rows have data-entry errors in OriginalListPrice (e.g. a $450k home
# with an OriginalListPrice of $450), producing ratios up to 1,000,000x. Because
# the monthly KPI uses mean(), those few rows blow the average up to 60-170. The
# ratio's median is 1.00 and its 1st-99th pct band is 0.79-1.28, so anything
# outside [0.5, 2.0] is a data error, not a real sale. Null those values so both
# the monthly mean and Tableau's own AVG() over the row-level extract are clean.
RATIO_LO, RATIO_HI = 0.5, 2.0
if 'close_to_original_list_ratio' in sold.columns:
    out_of_band = ~sold['close_to_original_list_ratio'].between(RATIO_LO, RATIO_HI)
    n_bad = int(out_of_band.sum())
    sold.loc[out_of_band, 'close_to_original_list_ratio'] = np.nan
    print(f"\n  Data-quality guard: nulled {n_bad:,} close_to_original_list_ratio "
          f"values outside [{RATIO_LO}, {RATIO_HI}] (OriginalListPrice typos)")

sold_cols = [
    # Geography / filters
    'CountyOrParish', 'City', 'PostalCode', 'MLSAreaMajor',
    'PropertyType', 'PropertySubType',
    # Agents / brokerages (competitive analysis)
    'ListOfficeName', 'BuyerOfficeName', 'ListAgentFullName', 'BuyerAgentFullName',
    # Dates / time-series keys
    'CloseDate', 'yr_mo', 'close_year', 'close_month',
    # Core measures
    'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
    'BedroomsTotal', 'BathroomsTotalInteger', 'DaysOnMarket',
    # Engineered (Week 6)
    'price_ratio', 'close_to_original_list_ratio', 'price_per_sqft',
    'listing_to_contract_days', 'contract_to_close_days',
    # Geo for maps
    'Latitude', 'Longitude',
]
sold_lean = sold[keep_existing(sold, sold_cols)].copy()

sold_out = os.path.join(OUT_DIR, "tableau_sold.csv")
sold_lean.to_csv(sold_out, index=False)
print(f"\n  Saved: {sold_out}")
print(f"  Shape: {sold_lean.shape[0]:,} rows x {sold_lean.shape[1]} columns")
print(f"  Columns: {list(sold_lean.columns)}")

# =============================================================================
# Part 2: Lean row-level LISTED extract
# =============================================================================
print(f"\n{'='*70}")
print("PART 2: LISTINGS ROW-LEVEL EXTRACT")
print(f"{'='*70}")

listed = pd.read_csv(LISTED_PATH, low_memory=False)
print(f"\nLoaded: {LISTED_PATH}")
print(f"Shape:  {listed.shape[0]:,} rows x {listed.shape[1]} columns")

for col in ['ListingContractDate', 'PurchaseContractDate', 'CloseDate']:
    if col in listed.columns:
        listed[col] = pd.to_datetime(listed[col], errors='coerce')

listed_cols = [
    'CountyOrParish', 'City', 'PostalCode', 'MLSAreaMajor',
    'PropertyType', 'PropertySubType', 'MlsStatus',
    'ListOfficeName', 'ListAgentFullName',
    'ListingContractDate', 'yr_mo', 'list_year', 'list_month',
    'ListPrice', 'OriginalListPrice', 'ClosePrice', 'LivingArea',
    'BedroomsTotal', 'BathroomsTotalInteger', 'DaysOnMarket',
    'list_price_per_sqft', 'price_reduction_ratio', 'close_to_list_ratio',
    'listing_to_contract_days', 'contract_to_close_days',
    'Latitude', 'Longitude',
]
listed_lean = listed[keep_existing(listed, listed_cols)].copy()

listed_out = os.path.join(OUT_DIR, "tableau_listed.csv")
listed_lean.to_csv(listed_out, index=False)
print(f"\n  Saved: {listed_out}")
print(f"  Shape: {listed_lean.shape[0]:,} rows x {listed_lean.shape[1]} columns")
print(f"  Columns: {list(listed_lean.columns)}")

# =============================================================================
# Part 3: Monthly market metrics (SOLD) — powers the KPI tiles & trend lines
# =============================================================================
print(f"\n{'='*70}")
print("PART 3: MONTHLY MARKET METRICS (SOLD)")
print(f"{'='*70}")

if 'yr_mo' in sold.columns:
    monthly = (
        sold.groupby('yr_mo')
        .agg(
            closed_sales            = ('ClosePrice',                  'count'),
            median_close_price      = ('ClosePrice',                  'median'),
            avg_close_price         = ('ClosePrice',                  'mean'),
            median_price_per_sqft   = ('price_per_sqft',              'median'),
            avg_dom                 = ('DaysOnMarket',                'mean'),
            median_dom              = ('DaysOnMarket',                'median'),
            avg_close_to_orig_ratio = ('close_to_original_list_ratio','mean'),
            avg_price_ratio         = ('price_ratio',                 'mean'),
        )
        .reset_index()
        .sort_values('yr_mo')
    )
    monthly_out = os.path.join(OUT_DIR, "monthly_market.csv")
    monthly.to_csv(monthly_out, index=False)
    print(f"\n  Saved: {monthly_out}")
    print(f"  Months: {monthly.shape[0]}  ({monthly['yr_mo'].min()} -> {monthly['yr_mo'].max()})")
    print(f"\n{monthly.to_string(index=False)}")
else:
    print("  'yr_mo' not found in sold — skipping monthly aggregation")

# =============================================================================
# Part 4: Monthly new listings (LISTED) — new inventory over time
# =============================================================================
print(f"\n{'='*70}")
print("PART 4: MONTHLY NEW LISTINGS")
print(f"{'='*70}")

if 'yr_mo' in listed.columns:
    new_listings = (
        listed.groupby('yr_mo')
        .agg(
            new_listings        = ('ListPrice',           'count'),
            median_list_price   = ('ListPrice',           'median'),
            median_list_per_sqft= ('list_price_per_sqft', 'median'),
        )
        .reset_index()
        .sort_values('yr_mo')
    )
    nl_out = os.path.join(OUT_DIR, "monthly_new_listings.csv")
    new_listings.to_csv(nl_out, index=False)
    print(f"\n  Saved: {nl_out}")
    print(f"  Months: {new_listings.shape[0]}  ({new_listings['yr_mo'].min()} -> {new_listings['yr_mo'].max()})")
    print(f"\n{new_listings.to_string(index=False)}")
else:
    print("  'yr_mo' not found in listed — skipping monthly aggregation")

# =============================================================================
# Summary
# =============================================================================
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"""
  Tableau-ready files written to: {OUT_DIR}
    tableau_sold.csv          row-level sold      (primary fact table)
    tableau_listed.csv        row-level listings  (inventory / new listings)
    monthly_market.csv        monthly sold KPIs    (trend + KPI tiles)
    monthly_new_listings.csv  monthly new listings (inventory trend)

  Next: open Tableau Public, connect tableau_sold.csv as the primary source,
  add tableau_listed.csv for new-listing/inventory views, and build the
  Market Analysis dashboard (median close price, avg DOM, avg close-to-
  original-list ratio, new listings, closed sales) filterable by
  city / county / zip / PropertySubType.
""")
print("Week 8-10 Tableau Prep – Complete!")
