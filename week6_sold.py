"""
Week 6 – Sold Transactions: Feature Engineering and Market Metrics
==================================================================
Engineers the key market indicators that power the Tableau dashboards.
Creates price ratios, per-square-foot metrics, timeline calculations,
and segment-level summary tables.

Input:  data/processed/week4_5_sold.csv
Output: data/processed/week6_sold.csv
"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# Paths
# =============================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "week4_5_sold.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Load Week 4-5 output
# =============================================================================
print("=" * 70)
print("WEEK 6 – SOLD: FEATURE ENGINEERING AND MARKET METRICS")
print("=" * 70)

sold = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"\nLoaded: {INPUT_PATH}")
print(f"Shape: {sold.shape[0]:,} rows x {sold.shape[1]} columns")

# Re-parse date fields (lost after CSV round-trip)
for col in ['CloseDate', 'PurchaseContractDate', 'ListingContractDate']:
    if col in sold.columns:
        sold[col] = pd.to_datetime(sold[col], errors='coerce')

# =============================================================================
# Part 1: Price Metrics
# =============================================================================
print(f"\n{'='*70}")
print("PART 1: PRICE METRICS")
print(f"{'='*70}")

# Price Ratio — ClosePrice vs current ListPrice (measures negotiation strength)
# Values >1 mean sold above list; <1 mean sold below list
sold['price_ratio'] = sold['ClosePrice'] / sold['ListPrice']

# Close-to-Original-List Ratio — ClosePrice vs OriginalListPrice
# Captures the full price reduction history from first asking price
sold['close_to_original_list_ratio'] = sold['ClosePrice'] / sold['OriginalListPrice']

# Price Per Square Foot — normalises price across different home sizes
sold['price_per_sqft'] = sold['ClosePrice'] / sold['LivingArea']

# Replace infinite values created by division by zero
for col in ['price_ratio', 'close_to_original_list_ratio', 'price_per_sqft']:
    sold[col] = sold[col].replace([np.inf, -np.inf], np.nan)

print(f"\n  price_ratio (ClosePrice / ListPrice):")
print(f"    Mean   : {sold['price_ratio'].mean():.4f}")
print(f"    Median : {sold['price_ratio'].median():.4f}")
print(f"    > 1.0  : {(sold['price_ratio'] > 1).sum():,} records (sold above list)")
print(f"    < 1.0  : {(sold['price_ratio'] < 1).sum():,} records (sold below list)")

print(f"\n  close_to_original_list_ratio (ClosePrice / OriginalListPrice):")
print(f"    Mean   : {sold['close_to_original_list_ratio'].mean():.4f}")
print(f"    Median : {sold['close_to_original_list_ratio'].median():.4f}")

print(f"\n  price_per_sqft (ClosePrice / LivingArea):")
print(f"    Mean   : ${sold['price_per_sqft'].mean():,.0f}")
print(f"    Median : ${sold['price_per_sqft'].median():,.0f}")

# =============================================================================
# Part 2: Time-Series Variables
# =============================================================================
print(f"\n{'='*70}")
print("PART 2: TIME-SERIES VARIABLES")
print(f"{'='*70}")

# Year, Month, and YrMo derived from CloseDate — enables time-series grouping
sold['close_year']  = sold['CloseDate'].dt.year
sold['close_month'] = sold['CloseDate'].dt.month
sold['yr_mo']       = sold['CloseDate'].dt.to_period('M').astype(str)

print(f"\n  close_year  — unique values: {sorted(sold['close_year'].dropna().unique().astype(int).tolist())}")
print(f"  close_month — range: {int(sold['close_month'].min())} – {int(sold['close_month'].max())}")
print(f"  yr_mo       — range: {sold['yr_mo'].min()} – {sold['yr_mo'].max()}")
print(f"                unique months: {sold['yr_mo'].nunique()}")

# =============================================================================
# Part 3: Timeline Metrics
# =============================================================================
print(f"\n{'='*70}")
print("PART 3: TIMELINE METRICS")
print(f"{'='*70}")

# Listing-to-Contract Days — measures how quickly a listing goes under contract
# (time from listing date to accepted offer)
sold['listing_to_contract_days'] = (
    (sold['PurchaseContractDate'] - sold['ListingContractDate'])
    .dt.days
)

# Contract-to-Close Days — measures the escrow and closing period duration
# (time from accepted offer to final sale)
sold['contract_to_close_days'] = (
    (sold['CloseDate'] - sold['PurchaseContractDate'])
    .dt.days
)

# Set negatives to NaN (data quality issues, will be handled in Week 7)
sold.loc[sold['listing_to_contract_days'] < 0, 'listing_to_contract_days'] = np.nan
sold.loc[sold['contract_to_close_days']   < 0, 'contract_to_close_days']   = np.nan

print(f"\n  listing_to_contract_days (ListingContractDate → PurchaseContractDate):")
print(f"    Non-null : {sold['listing_to_contract_days'].notna().sum():,} records")
print(f"    Mean     : {sold['listing_to_contract_days'].mean():.1f} days")
print(f"    Median   : {sold['listing_to_contract_days'].median():.0f} days")
print(f"    p25–p75  : {sold['listing_to_contract_days'].quantile(0.25):.0f} – "
      f"{sold['listing_to_contract_days'].quantile(0.75):.0f} days")

print(f"\n  contract_to_close_days (PurchaseContractDate → CloseDate):")
print(f"    Non-null : {sold['contract_to_close_days'].notna().sum():,} records")
print(f"    Mean     : {sold['contract_to_close_days'].mean():.1f} days")
print(f"    Median   : {sold['contract_to_close_days'].median():.0f} days")
print(f"    p25–p75  : {sold['contract_to_close_days'].quantile(0.25):.0f} – "
      f"{sold['contract_to_close_days'].quantile(0.75):.0f} days")

# =============================================================================
# Part 4: Sample Output Table
# =============================================================================
print(f"\n{'='*70}")
print("PART 4: SAMPLE OUTPUT TABLE (first 10 rows, engineered columns)")
print(f"{'='*70}")

sample_cols = [
    'CloseDate', 'yr_mo', 'close_year', 'close_month',
    'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
    'price_ratio', 'close_to_original_list_ratio', 'price_per_sqft',
    'DaysOnMarket', 'listing_to_contract_days', 'contract_to_close_days',
]
available_sample = [c for c in sample_cols if c in sold.columns]
print(f"\n{sold[available_sample].head(10).to_string(index=False)}")

# =============================================================================
# Part 5: Segment Analysis
# =============================================================================
print(f"\n{'='*70}")
print("PART 5: SEGMENT ANALYSIS")
print(f"{'='*70}")

def segment_summary(df, group_col, label):
    """Generate a segment summary table for a given grouping column."""
    if group_col not in df.columns:
        print(f"  {label}: column '{group_col}' not found — skipping")
        return None
    summary = (
        df.groupby(group_col)
        .agg(
            transaction_count        = ('ClosePrice',                 'count'),
            median_close_price       = ('ClosePrice',                 'median'),
            median_price_per_sqft    = ('price_per_sqft',             'median'),
            median_price_ratio       = ('price_ratio',                'median'),
            median_dom               = ('DaysOnMarket',               'median'),
            median_listing_to_contract = ('listing_to_contract_days', 'median'),
            median_contract_to_close   = ('contract_to_close_days',   'median'),
        )
        .sort_values('transaction_count', ascending=False)
        .reset_index()
    )
    return summary

# --- By PropertySubType ---
print(f"\n--- By PropertySubType ---")
seg_subtype = segment_summary(sold, 'PropertySubType', 'PropertySubType')
if seg_subtype is not None:
    print(seg_subtype.head(8).to_string(index=False))

# --- By CountyOrParish ---
print(f"\n--- By CountyOrParish (Top 10 by volume) ---")
seg_county = segment_summary(sold, 'CountyOrParish', 'CountyOrParish')
if seg_county is not None:
    print(seg_county.head(10).to_string(index=False))

# --- By ListOfficeName (Top 10 by volume) ---
print(f"\n--- By ListOfficeName (Top 10 by volume) ---")
seg_list_office = segment_summary(sold, 'ListOfficeName', 'ListOfficeName')
if seg_list_office is not None:
    print(seg_list_office.head(10).to_string(index=False))

# --- By BuyerOfficeName (Top 10 by volume) ---
print(f"\n--- By BuyerOfficeName (Top 10 by volume) ---")
seg_buyer_office = segment_summary(sold, 'BuyerOfficeName', 'BuyerOfficeName')
if seg_buyer_office is not None:
    print(seg_buyer_office.head(10).to_string(index=False))

# =============================================================================
# Part 6: Engineered Column Summary
# =============================================================================
print(f"\n{'='*70}")
print("PART 6: ENGINEERED COLUMN SUMMARY")
print(f"{'='*70}")

new_cols = {
    'price_ratio':                  'ClosePrice / ListPrice',
    'close_to_original_list_ratio': 'ClosePrice / OriginalListPrice',
    'price_per_sqft':               'ClosePrice / LivingArea',
    'close_year':                   'Year from CloseDate',
    'close_month':                  'Month from CloseDate',
    'yr_mo':                        'Year-Month string from CloseDate',
    'listing_to_contract_days':     'PurchaseContractDate - ListingContractDate',
    'contract_to_close_days':       'CloseDate - PurchaseContractDate',
}

print(f"\n  {'Column':<35} {'Formula':<45} {'Non-null':>10}")
print(f"  {'-'*35} {'-'*45} {'-'*10}")
for col, formula in new_cols.items():
    if col in sold.columns:
        non_null = sold[col].notna().sum()
        print(f"  {col:<35} {formula:<45} {non_null:>10,}")

# =============================================================================
# Save Output
# =============================================================================
print(f"\n{'='*70}")
print("SAVING OUTPUT")
print(f"{'='*70}")

output_path = os.path.join(OUTPUT_DIR, "week6_sold.csv")
sold.to_csv(output_path, index=False)
print(f"\n  Saved to: {output_path}")
print(f"  Final shape: {sold.shape[0]:,} rows x {sold.shape[1]} columns")
print(f"  New columns added: {len(new_cols)}")
print(f"\nWeek 6 Sold – Complete!")
