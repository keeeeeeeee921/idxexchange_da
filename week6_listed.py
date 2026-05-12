"""
Week 6 – Listings: Feature Engineering and Market Metrics
==========================================================
Engineers the key market indicators for the listings dataset.
Creates price metrics, time-series variables, timeline calculations,
and segment-level summary tables grouped by key dimensions.

Input:  data/processed/week4_5_listed.csv
Output: data/processed/week6_listed.csv
"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# Paths
# =============================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "week4_5_listed.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Load Week 4-5 output
# =============================================================================
print("=" * 70)
print("WEEK 6 – LISTINGS: FEATURE ENGINEERING AND MARKET METRICS")
print("=" * 70)

listings = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"\nLoaded: {INPUT_PATH}")
print(f"Shape: {listings.shape[0]:,} rows x {listings.shape[1]} columns")

# Re-parse date fields (lost after CSV round-trip)
for col in ['ListingContractDate', 'PurchaseContractDate', 'CloseDate']:
    if col in listings.columns:
        listings[col] = pd.to_datetime(listings[col], errors='coerce')

# =============================================================================
# Part 1: Price Metrics
# =============================================================================
print(f"\n{'='*70}")
print("PART 1: PRICE METRICS")
print(f"{'='*70}")

# Price Per Square Foot — based on ListPrice (ClosePrice is mostly null for active listings)
listings['list_price_per_sqft'] = listings['ListPrice'] / listings['LivingArea']

# Price Reduction — OriginalListPrice vs current ListPrice
# Values < 1 mean the price was reduced from the original ask
listings['price_reduction_ratio'] = listings['ListPrice'] / listings['OriginalListPrice']

# Close-to-List Ratio — only populated for Closed records
listings['close_to_list_ratio'] = listings['ClosePrice'] / listings['ListPrice']

# Replace infinite values from division by zero
for col in ['list_price_per_sqft', 'price_reduction_ratio', 'close_to_list_ratio']:
    listings[col] = listings[col].replace([np.inf, -np.inf], np.nan)

print(f"\n  list_price_per_sqft (ListPrice / LivingArea):")
print(f"    Mean   : ${listings['list_price_per_sqft'].mean():,.0f}")
print(f"    Median : ${listings['list_price_per_sqft'].median():,.0f}")

print(f"\n  price_reduction_ratio (ListPrice / OriginalListPrice):")
print(f"    Mean   : {listings['price_reduction_ratio'].mean():.4f}")
print(f"    Median : {listings['price_reduction_ratio'].median():.4f}")
reduced = (listings['price_reduction_ratio'] < 1).sum()
print(f"    Price reduced (<1.0): {reduced:,} records ({reduced/len(listings)*100:.1f}%)")

print(f"\n  close_to_list_ratio (ClosePrice / ListPrice — Closed records only):")
non_null = listings['close_to_list_ratio'].notna().sum()
print(f"    Non-null : {non_null:,} records")
print(f"    Mean     : {listings['close_to_list_ratio'].mean():.4f}")
print(f"    Median   : {listings['close_to_list_ratio'].median():.4f}")

# =============================================================================
# Part 2: Time-Series Variables
# =============================================================================
print(f"\n{'='*70}")
print("PART 2: TIME-SERIES VARIABLES")
print(f"{'='*70}")

# Year, Month, and YrMo derived from ListingContractDate
listings['list_year']  = listings['ListingContractDate'].dt.year
listings['list_month'] = listings['ListingContractDate'].dt.month
listings['yr_mo']      = listings['ListingContractDate'].dt.to_period('M').astype(str)

print(f"\n  list_year  — unique values: {sorted(listings['list_year'].dropna().unique().astype(int).tolist())}")
print(f"  list_month — range: {int(listings['list_month'].min())} – {int(listings['list_month'].max())}")
print(f"  yr_mo      — range: {listings['yr_mo'].min()} – {listings['yr_mo'].max()}")
print(f"               unique months: {listings['yr_mo'].nunique()}")

# =============================================================================
# Part 3: Timeline Metrics
# =============================================================================
print(f"\n{'='*70}")
print("PART 3: TIMELINE METRICS")
print(f"{'='*70}")

# Listing-to-Contract Days — how quickly a listing goes under contract
listings['listing_to_contract_days'] = (
    (listings['PurchaseContractDate'] - listings['ListingContractDate'])
    .dt.days
)

# Contract-to-Close Days — escrow and closing duration (Closed records only)
listings['contract_to_close_days'] = (
    (listings['CloseDate'] - listings['PurchaseContractDate'])
    .dt.days
)

# Set negatives to NaN (data quality issues flagged in Week 4-5)
listings.loc[listings['listing_to_contract_days'] < 0, 'listing_to_contract_days'] = np.nan
listings.loc[listings['contract_to_close_days']   < 0, 'contract_to_close_days']   = np.nan

print(f"\n  listing_to_contract_days (ListingContractDate → PurchaseContractDate):")
print(f"    Non-null : {listings['listing_to_contract_days'].notna().sum():,} records")
print(f"    Mean     : {listings['listing_to_contract_days'].mean():.1f} days")
print(f"    Median   : {listings['listing_to_contract_days'].median():.0f} days")
print(f"    p25–p75  : {listings['listing_to_contract_days'].quantile(0.25):.0f} – "
      f"{listings['listing_to_contract_days'].quantile(0.75):.0f} days")

print(f"\n  contract_to_close_days (PurchaseContractDate → CloseDate — Closed records):")
print(f"    Non-null : {listings['contract_to_close_days'].notna().sum():,} records")
print(f"    Mean     : {listings['contract_to_close_days'].mean():.1f} days")
print(f"    Median   : {listings['contract_to_close_days'].median():.0f} days")
print(f"    p25–p75  : {listings['contract_to_close_days'].quantile(0.25):.0f} – "
      f"{listings['contract_to_close_days'].quantile(0.75):.0f} days")

# =============================================================================
# Part 4: Sample Output Table
# =============================================================================
print(f"\n{'='*70}")
print("PART 4: SAMPLE OUTPUT TABLE (first 10 rows, engineered columns)")
print(f"{'='*70}")

sample_cols = [
    'ListingContractDate', 'yr_mo', 'list_year', 'list_month', 'MlsStatus',
    'ListPrice', 'OriginalListPrice', 'LivingArea',
    'list_price_per_sqft', 'price_reduction_ratio',
    'DaysOnMarket', 'listing_to_contract_days',
]
available_sample = [c for c in sample_cols if c in listings.columns]
print(f"\n{listings[available_sample].head(10).to_string(index=False)}")

# =============================================================================
# Part 5: Segment Analysis
# =============================================================================
print(f"\n{'='*70}")
print("PART 5: SEGMENT ANALYSIS")
print(f"{'='*70}")

def segment_summary(df, group_col):
    if group_col not in df.columns:
        print(f"  '{group_col}' not found — skipping")
        return None
    return (
        df.groupby(group_col)
        .agg(
            listing_count              = ('ListPrice',                'count'),
            median_list_price          = ('ListPrice',                'median'),
            median_list_price_per_sqft = ('list_price_per_sqft',     'median'),
            median_price_reduction     = ('price_reduction_ratio',   'median'),
            median_dom                 = ('DaysOnMarket',             'median'),
            median_listing_to_contract = ('listing_to_contract_days','median'),
        )
        .sort_values('listing_count', ascending=False)
        .reset_index()
    )

# --- By MlsStatus (unique to listings) ---
print(f"\n--- By MlsStatus ---")
seg_status = segment_summary(listings, 'MlsStatus')
if seg_status is not None:
    print(seg_status.to_string(index=False))

# --- By PropertySubType ---
print(f"\n--- By PropertySubType ---")
seg_subtype = segment_summary(listings, 'PropertySubType')
if seg_subtype is not None:
    print(seg_subtype.head(8).to_string(index=False))

# --- By CountyOrParish (Top 10 by volume) ---
print(f"\n--- By CountyOrParish (Top 10 by volume) ---")
seg_county = segment_summary(listings, 'CountyOrParish')
if seg_county is not None:
    print(seg_county.head(10).to_string(index=False))

# --- By ListOfficeName (Top 10 by volume) ---
print(f"\n--- By ListOfficeName (Top 10 by volume) ---")
seg_office = segment_summary(listings, 'ListOfficeName')
if seg_office is not None:
    print(seg_office.head(10).to_string(index=False))

# =============================================================================
# Part 6: Engineered Column Summary
# =============================================================================
print(f"\n{'='*70}")
print("PART 6: ENGINEERED COLUMN SUMMARY")
print(f"{'='*70}")

new_cols = {
    'list_price_per_sqft':      'ListPrice / LivingArea',
    'price_reduction_ratio':    'ListPrice / OriginalListPrice',
    'close_to_list_ratio':      'ClosePrice / ListPrice (Closed only)',
    'list_year':                'Year from ListingContractDate',
    'list_month':               'Month from ListingContractDate',
    'yr_mo':                    'Year-Month string from ListingContractDate',
    'listing_to_contract_days': 'PurchaseContractDate - ListingContractDate',
    'contract_to_close_days':   'CloseDate - PurchaseContractDate',
}

print(f"\n  {'Column':<35} {'Formula':<45} {'Non-null':>10}")
print(f"  {'-'*35} {'-'*45} {'-'*10}")
for col, formula in new_cols.items():
    if col in listings.columns:
        non_null = listings[col].notna().sum()
        print(f"  {col:<35} {formula:<45} {non_null:>10,}")

# =============================================================================
# Save Output
# =============================================================================
print(f"\n{'='*70}")
print("SAVING OUTPUT")
print(f"{'='*70}")

output_path = os.path.join(OUTPUT_DIR, "week6_listed.csv")
listings.to_csv(output_path, index=False)
print(f"\n  Saved to: {output_path}")
print(f"  Final shape: {listings.shape[0]:,} rows x {listings.shape[1]} columns")
print(f"  New columns added: {len(new_cols)}")
print(f"\nWeek 6 Listings – Complete!")
