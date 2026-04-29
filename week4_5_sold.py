"""
Weeks 4–5 – Sold Transactions: Data Cleaning and Preparation
=============================================================
Prepares the sold dataset for reliable analytics by converting date fields,
removing high-missing columns, validating numeric types, flagging invalid
values, checking date consistency, and auditing geographic coordinates.

Input:  data/processed/week2_3_sold.csv
Output: data/processed/week4_5_sold.csv
"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# Paths
# =============================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH  = os.path.join(BASE_DIR, "data", "processed", "week2_3_sold.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# California coordinate bounds (plausible range)
CA_LAT_MIN, CA_LAT_MAX = 32.5, 42.0
CA_LON_MIN, CA_LON_MAX = -124.5, -114.1

# =============================================================================
# Load Week 2-3 output
# =============================================================================
print("=" * 70)
print("WEEKS 4–5 – SOLD: DATA CLEANING AND PREPARATION")
print("=" * 70)

sold = pd.read_csv(INPUT_PATH, low_memory=False)
rows_original = len(sold)
cols_original = len(sold.columns)

print(f"\nLoaded: {INPUT_PATH}")
print(f"Shape (before cleaning): {rows_original:,} rows x {cols_original} columns")

# =============================================================================
# Part 1: Date Field Conversion
# =============================================================================
print(f"\n{'='*70}")
print("PART 1: DATE FIELD CONVERSION")
print(f"{'='*70}")

date_fields = [
    'CloseDate',
    'PurchaseContractDate',
    'ListingContractDate',
    'ContractStatusChangeDate',
]

print(f"\n  Converting {len(date_fields)} date fields to datetime:")
for col in date_fields:
    if col in sold.columns:
        before_nulls = sold[col].isnull().sum()
        sold[col] = pd.to_datetime(sold[col], errors='coerce')
        after_nulls = sold[col].isnull().sum()
        coerced = after_nulls - before_nulls
        print(f"  {col}: converted — {after_nulls:,} nulls "
              f"({'+'  if coerced >= 0 else ''}{coerced} from coercion)")
    else:
        print(f"  {col}: NOT FOUND in dataset")

# =============================================================================
# Part 2: Remove High-Missing Columns (>90% null)
# =============================================================================
print(f"\n{'='*70}")
print("PART 2: REMOVE HIGH-MISSING COLUMNS (>90% null)")
print(f"{'='*70}")

null_pct = sold.isnull().mean() * 100
high_missing_cols = null_pct[null_pct > 90].index.tolist()

print(f"\n  Columns dropped (>90% missing) — {len(high_missing_cols)} columns:")
for col in high_missing_cols:
    print(f"    {col}: {null_pct[col]:.1f}% missing "
          f"({int(sold[col].isnull().sum()):,} nulls)")

# Why: Columns with >90% missing values carry no analytical value and
# would introduce noise into downstream models and visualisations.
sold = sold.drop(columns=high_missing_cols)
print(f"\n  Columns after removal: {len(sold.columns)} "
      f"(removed {cols_original - len(sold.columns)})")

# =============================================================================
# Part 3: Numeric Type Validation
# =============================================================================
print(f"\n{'='*70}")
print("PART 3: NUMERIC TYPE VALIDATION")
print(f"{'='*70}")

numeric_fields = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice',
    'LivingArea', 'LotSizeAcres',
    'BedroomsTotal', 'BathroomsTotalInteger',
    'DaysOnMarket', 'YearBuilt',
    'Latitude', 'Longitude',
]

print(f"\n  Validating {len(numeric_fields)} numeric fields:")
for col in numeric_fields:
    if col in sold.columns:
        before_dtype = sold[col].dtype
        sold[col] = pd.to_numeric(sold[col], errors='coerce')
        after_dtype = sold[col].dtype
        status = "OK" if before_dtype == after_dtype else f"converted {before_dtype} → {after_dtype}"
        print(f"  {col}: {after_dtype}  [{status}]")
    else:
        print(f"  {col}: NOT FOUND in dataset")

# =============================================================================
# Part 4: Invalid Value Flags
# =============================================================================
print(f"\n{'='*70}")
print("PART 4: INVALID VALUE FLAGS")
print(f"{'='*70}")

# ClosePrice <= 0  (a sale price must be positive)
sold['invalid_price_flag'] = (
    sold['ClosePrice'].notna() & (sold['ClosePrice'] <= 0)
).astype(int)

# LivingArea <= 0  (a home must have positive square footage)
sold['invalid_area_flag'] = (
    sold['LivingArea'].notna() & (sold['LivingArea'] <= 0)
).astype(int)

# DaysOnMarket < 0  (negative time-on-market is logically impossible)
sold['invalid_dom_flag'] = (
    sold['DaysOnMarket'].notna() & (sold['DaysOnMarket'] < 0)
).astype(int)

# Negative bedrooms or bathrooms
sold['invalid_beds_flag'] = (
    sold['BedroomsTotal'].notna() & (sold['BedroomsTotal'] < 0)
).astype(int)
sold['invalid_baths_flag'] = (
    sold['BathroomsTotalInteger'].notna() & (sold['BathroomsTotalInteger'] < 0)
).astype(int)

flags = {
    'invalid_price_flag':  'ClosePrice <= 0',
    'invalid_area_flag':   'LivingArea <= 0',
    'invalid_dom_flag':    'DaysOnMarket < 0',
    'invalid_beds_flag':   'BedroomsTotal < 0',
    'invalid_baths_flag':  'BathroomsTotalInteger < 0',
}

print(f"\n  Invalid value flag summary:")
for flag, description in flags.items():
    count = sold[flag].sum()
    pct   = count / len(sold) * 100
    print(f"  {flag} ({description}): {count:,} records ({pct:.2f}%)")

# =============================================================================
# Part 5: Date Consistency Flags
# =============================================================================
print(f"\n{'='*70}")
print("PART 5: DATE CONSISTENCY FLAGS")
print(f"{'='*70}")

print("""
  Logic:
    ListingContractDate → PurchaseContractDate → CloseDate
    Any violation of this order is flagged.
""")

# listing_after_close_flag: listing date is AFTER close date
sold['listing_after_close_flag'] = (
    sold['ListingContractDate'].notna() &
    sold['CloseDate'].notna() &
    (sold['ListingContractDate'] > sold['CloseDate'])
).astype(int)

# purchase_after_close_flag: purchase contract date is AFTER close date
sold['purchase_after_close_flag'] = (
    sold['PurchaseContractDate'].notna() &
    sold['CloseDate'].notna() &
    (sold['PurchaseContractDate'] > sold['CloseDate'])
).astype(int)

# negative_timeline_flag: any date order violation
sold['negative_timeline_flag'] = (
    (sold['listing_after_close_flag'] == 1) |
    (sold['purchase_after_close_flag'] == 1)
).astype(int)

date_flags = {
    'listing_after_close_flag':  'ListingContractDate > CloseDate',
    'purchase_after_close_flag': 'PurchaseContractDate > CloseDate',
    'negative_timeline_flag':    'Any date order violation',
}

print(f"  Date consistency flag summary:")
for flag, description in date_flags.items():
    count = sold[flag].sum()
    pct   = count / len(sold) * 100
    print(f"  {flag} ({description}): {count:,} records ({pct:.2f}%)")

# =============================================================================
# Part 6: Geographic Data Checks
# =============================================================================
print(f"\n{'='*70}")
print("PART 6: GEOGRAPHIC DATA CHECKS")
print(f"{'='*70}")

print(f"""
  California coordinate bounds:
    Latitude  : {CA_LAT_MIN}° – {CA_LAT_MAX}°N
    Longitude : {CA_LON_MIN}° – {CA_LON_MAX}°W (must be negative)
""")

# Missing coordinates
sold['coord_missing_flag'] = (
    sold['Latitude'].isnull() | sold['Longitude'].isnull()
).astype(int)

# Zero coordinates (sentinel null values — data export artifact)
sold['coord_zero_flag'] = (
    sold['Latitude'].notna() & sold['Longitude'].notna() &
    ((sold['Latitude'] == 0) | (sold['Longitude'] == 0))
).astype(int)

# Positive longitude (California longitudes must be negative)
sold['coord_positive_lon_flag'] = (
    sold['Longitude'].notna() & (sold['Longitude'] > 0)
).astype(int)

# Out-of-state / implausible coordinates
sold['coord_out_of_state_flag'] = (
    sold['Latitude'].notna()  & sold['Longitude'].notna() &
    (sold['coord_zero_flag'] == 0) &
    (
        (sold['Latitude']  < CA_LAT_MIN) | (sold['Latitude']  > CA_LAT_MAX) |
        (sold['Longitude'] < CA_LON_MIN) | (sold['Longitude'] > CA_LON_MAX)
    )
).astype(int)

geo_flags = {
    'coord_missing_flag':       'Latitude or Longitude is null',
    'coord_zero_flag':          'Latitude = 0 or Longitude = 0',
    'coord_positive_lon_flag':  'Longitude > 0 (should be negative for CA)',
    'coord_out_of_state_flag':  'Coordinates outside California bounds',
}

print(f"  Geographic flag summary:")
for flag, description in geo_flags.items():
    count = sold[flag].sum()
    pct   = count / len(sold) * 100
    print(f"  {flag}: {count:,} records ({pct:.2f}%)")
    print(f"    → {description}")

any_geo_issue = (
    (sold['coord_missing_flag']      == 1) |
    (sold['coord_zero_flag']         == 1) |
    (sold['coord_positive_lon_flag'] == 1) |
    (sold['coord_out_of_state_flag'] == 1)
).sum()
print(f"\n  Total records with any geographic issue: {any_geo_issue:,} "
      f"({any_geo_issue/len(sold)*100:.2f}%)")

# =============================================================================
# Part 7: Before / After Summary Report
# =============================================================================
print(f"\n{'='*70}")
print("PART 7: BEFORE / AFTER SUMMARY REPORT")
print(f"{'='*70}")

rows_after = len(sold)
cols_after = len(sold.columns)

print(f"""
  Row count    : {rows_original:,} → {rows_after:,}  (no rows removed — flags only)
  Column count : {cols_original} → {cols_after}  ({cols_original - cols_after} high-missing columns dropped)

  Data type confirmations (key numeric fields):
""")
for col in ['ClosePrice', 'ListPrice', 'LivingArea', 'DaysOnMarket',
            'BedroomsTotal', 'BathroomsTotalInteger', 'YearBuilt',
            'Latitude', 'Longitude']:
    if col in sold.columns:
        print(f"    {col}: {sold[col].dtype}")

print(f"""
  Date consistency flags:
    listing_after_close_flag  : {sold['listing_after_close_flag'].sum():,}
    purchase_after_close_flag : {sold['purchase_after_close_flag'].sum():,}
    negative_timeline_flag    : {sold['negative_timeline_flag'].sum():,}

  Geographic data quality:
    Records with missing coords     : {sold['coord_missing_flag'].sum():,}
    Records with zero coords        : {sold['coord_zero_flag'].sum():,}
    Records with positive longitude : {sold['coord_positive_lon_flag'].sum():,}
    Records out of CA bounds        : {sold['coord_out_of_state_flag'].sum():,}
    Total with any geo issue        : {any_geo_issue:,}

  Invalid value flags:
    ClosePrice <= 0    : {sold['invalid_price_flag'].sum():,}
    LivingArea <= 0    : {sold['invalid_area_flag'].sum():,}
    DaysOnMarket < 0   : {sold['invalid_dom_flag'].sum():,}
    BedroomsTotal < 0  : {sold['invalid_beds_flag'].sum():,}
    BathroomsTotal < 0 : {sold['invalid_baths_flag'].sum():,}
""")

# =============================================================================
# Save Output
# =============================================================================
print(f"{'='*70}")
print("SAVING OUTPUT")
print(f"{'='*70}")

output_path = os.path.join(OUTPUT_DIR, "week4_5_sold.csv")
sold.to_csv(output_path, index=False)
print(f"\n  Saved to: {output_path}")
print(f"  Final shape: {sold.shape[0]:,} rows x {sold.shape[1]} columns")
print(f"\nWeeks 4–5 Sold – Complete!")
