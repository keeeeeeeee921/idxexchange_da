"""
Weeks 4–5 – Listings: Data Cleaning and Preparation
=====================================================
Prepares the listings dataset for reliable analytics by converting date fields,
removing high-missing columns, validating numeric types, flagging invalid
values, checking date consistency, and auditing geographic coordinates.

Input:  data/processed/week2_3_listed.csv
Output: data/processed/week4_5_listed.csv
"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# Paths
# =============================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH  = os.path.join(BASE_DIR, "data", "processed", "week2_3_listed.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# California coordinate bounds (plausible range)
CA_LAT_MIN, CA_LAT_MAX = 32.5, 42.0
CA_LON_MIN, CA_LON_MAX = -124.5, -114.1

# =============================================================================
# Load Week 2-3 output
# =============================================================================
print("=" * 70)
print("WEEKS 4–5 – LISTINGS: DATA CLEANING AND PREPARATION")
print("=" * 70)

listings = pd.read_csv(INPUT_PATH, low_memory=False)
rows_original = len(listings)
cols_original = len(listings.columns)

print(f"\nLoaded: {INPUT_PATH}")
print(f"Shape (before cleaning): {rows_original:,} rows x {cols_original} columns")

# =============================================================================
# Part 1: Date Field Conversion
# =============================================================================
print(f"\n{'='*70}")
print("PART 1: DATE FIELD CONVERSION")
print(f"{'='*70}")

date_fields = [
    'ListingContractDate',
    'PurchaseContractDate',
    'CloseDate',
    'ContractStatusChangeDate',
]

print(f"\n  Converting {len(date_fields)} date fields to datetime:")
for col in date_fields:
    if col in listings.columns:
        before_nulls = listings[col].isnull().sum()
        listings[col] = pd.to_datetime(listings[col], errors='coerce')
        after_nulls = listings[col].isnull().sum()
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

null_pct = listings.isnull().mean() * 100
high_missing_cols = null_pct[null_pct > 90].index.tolist()

print(f"\n  Columns dropped (>90% missing) — {len(high_missing_cols)} columns:")
for col in high_missing_cols:
    print(f"    {col}: {null_pct[col]:.1f}% missing "
          f"({int(listings[col].isnull().sum()):,} nulls)")

# Why: Columns with >90% missing values carry no analytical value and
# would introduce noise into downstream models and visualisations.
listings = listings.drop(columns=high_missing_cols)
print(f"\n  Columns after removal: {len(listings.columns)} "
      f"(removed {cols_original - len(listings.columns)})")

# =============================================================================
# Part 3: Numeric Type Validation
# =============================================================================
print(f"\n{'='*70}")
print("PART 3: NUMERIC TYPE VALIDATION")
print(f"{'='*70}")

numeric_fields = [
    'ListPrice', 'OriginalListPrice',
    'LivingArea', 'LotSizeAcres',
    'BedroomsTotal', 'BathroomsTotalInteger',
    'DaysOnMarket', 'YearBuilt',
    'Latitude', 'Longitude',
]

print(f"\n  Validating {len(numeric_fields)} numeric fields:")
for col in numeric_fields:
    if col in listings.columns:
        before_dtype = listings[col].dtype
        listings[col] = pd.to_numeric(listings[col], errors='coerce')
        after_dtype = listings[col].dtype
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

# ListPrice <= 0  (a listing price must be positive)
listings['invalid_price_flag'] = (
    listings['ListPrice'].notna() & (listings['ListPrice'] <= 0)
).astype(int)

# LivingArea <= 0  (a home must have positive square footage)
listings['invalid_area_flag'] = (
    listings['LivingArea'].notna() & (listings['LivingArea'] <= 0)
).astype(int)

# DaysOnMarket < 0  (negative time-on-market is logically impossible)
listings['invalid_dom_flag'] = (
    listings['DaysOnMarket'].notna() & (listings['DaysOnMarket'] < 0)
).astype(int)

# Negative bedrooms or bathrooms
listings['invalid_beds_flag'] = (
    listings['BedroomsTotal'].notna() & (listings['BedroomsTotal'] < 0)
).astype(int)
listings['invalid_baths_flag'] = (
    listings['BathroomsTotalInteger'].notna() & (listings['BathroomsTotalInteger'] < 0)
).astype(int)

flags = {
    'invalid_price_flag':  'ListPrice <= 0',
    'invalid_area_flag':   'LivingArea <= 0',
    'invalid_dom_flag':    'DaysOnMarket < 0',
    'invalid_beds_flag':   'BedroomsTotal < 0',
    'invalid_baths_flag':  'BathroomsTotalInteger < 0',
}

print(f"\n  Invalid value flag summary:")
for flag, description in flags.items():
    count = listings[flag].sum()
    pct   = count / len(listings) * 100
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
    Note: CloseDate is partially populated for listings (present on Closed/Pending records).
""")

# listing_after_close_flag: listing date is AFTER close date
listings['listing_after_close_flag'] = (
    listings['ListingContractDate'].notna() &
    listings['CloseDate'].notna() &
    (listings['ListingContractDate'] > listings['CloseDate'])
).astype(int)

# purchase_after_close_flag: purchase contract date is AFTER close date
listings['purchase_after_close_flag'] = (
    listings['PurchaseContractDate'].notna() &
    listings['CloseDate'].notna() &
    (listings['PurchaseContractDate'] > listings['CloseDate'])
).astype(int)

# negative_timeline_flag: any date order violation
listings['negative_timeline_flag'] = (
    (listings['listing_after_close_flag'] == 1) |
    (listings['purchase_after_close_flag'] == 1)
).astype(int)

date_flags = {
    'listing_after_close_flag':  'ListingContractDate > CloseDate',
    'purchase_after_close_flag': 'PurchaseContractDate > CloseDate',
    'negative_timeline_flag':    'Any date order violation',
}

print(f"  Date consistency flag summary:")
for flag, description in date_flags.items():
    count = listings[flag].sum()
    pct   = count / len(listings) * 100
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
listings['coord_missing_flag'] = (
    listings['Latitude'].isnull() | listings['Longitude'].isnull()
).astype(int)

# Zero coordinates (sentinel null values — data export artifact)
listings['coord_zero_flag'] = (
    listings['Latitude'].notna() & listings['Longitude'].notna() &
    ((listings['Latitude'] == 0) | (listings['Longitude'] == 0))
).astype(int)

# Positive longitude (California longitudes must be negative)
listings['coord_positive_lon_flag'] = (
    listings['Longitude'].notna() & (listings['Longitude'] > 0)
).astype(int)

# Out-of-state / implausible coordinates
listings['coord_out_of_state_flag'] = (
    listings['Latitude'].notna()  & listings['Longitude'].notna() &
    (listings['coord_zero_flag'] == 0) &
    (
        (listings['Latitude']  < CA_LAT_MIN) | (listings['Latitude']  > CA_LAT_MAX) |
        (listings['Longitude'] < CA_LON_MIN) | (listings['Longitude'] > CA_LON_MAX)
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
    count = listings[flag].sum()
    pct   = count / len(listings) * 100
    print(f"  {flag}: {count:,} records ({pct:.2f}%)")
    print(f"    → {description}")

any_geo_issue = (
    (listings['coord_missing_flag']      == 1) |
    (listings['coord_zero_flag']         == 1) |
    (listings['coord_positive_lon_flag'] == 1) |
    (listings['coord_out_of_state_flag'] == 1)
).sum()
print(f"\n  Total records with any geographic issue: {any_geo_issue:,} "
      f"({any_geo_issue/len(listings)*100:.2f}%)")

# =============================================================================
# Part 7: Before / After Summary Report
# =============================================================================
print(f"\n{'='*70}")
print("PART 7: BEFORE / AFTER SUMMARY REPORT")
print(f"{'='*70}")

rows_after = len(listings)
cols_after = len(listings.columns)

print(f"""
  Row count    : {rows_original:,} → {rows_after:,}  (no rows removed — flags only)
  Column count : {cols_original} → {cols_after}  ({cols_original - cols_after} high-missing columns dropped)

  Data type confirmations (key numeric fields):
""")
for col in ['ListPrice', 'OriginalListPrice', 'LivingArea', 'DaysOnMarket',
            'BedroomsTotal', 'BathroomsTotalInteger', 'YearBuilt',
            'Latitude', 'Longitude']:
    if col in listings.columns:
        print(f"    {col}: {listings[col].dtype}")

print(f"""
  Date consistency flags:
    listing_after_close_flag  : {listings['listing_after_close_flag'].sum():,}
    purchase_after_close_flag : {listings['purchase_after_close_flag'].sum():,}
    negative_timeline_flag    : {listings['negative_timeline_flag'].sum():,}

  Geographic data quality:
    Records with missing coords     : {listings['coord_missing_flag'].sum():,}
    Records with zero coords        : {listings['coord_zero_flag'].sum():,}
    Records with positive longitude : {listings['coord_positive_lon_flag'].sum():,}
    Records out of CA bounds        : {listings['coord_out_of_state_flag'].sum():,}
    Total with any geo issue        : {any_geo_issue:,}

  Invalid value flags:
    ListPrice <= 0     : {listings['invalid_price_flag'].sum():,}
    LivingArea <= 0    : {listings['invalid_area_flag'].sum():,}
    DaysOnMarket < 0   : {listings['invalid_dom_flag'].sum():,}
    BedroomsTotal < 0  : {listings['invalid_beds_flag'].sum():,}
    BathroomsTotal < 0 : {listings['invalid_baths_flag'].sum():,}
""")

# =============================================================================
# Save Output
# =============================================================================
print(f"{'='*70}")
print("SAVING OUTPUT")
print(f"{'='*70}")

output_path = os.path.join(OUTPUT_DIR, "week4_5_listed.csv")
listings.to_csv(output_path, index=False)
print(f"\n  Saved to: {output_path}")
print(f"  Final shape: {listings.shape[0]:,} rows x {listings.shape[1]} columns")
print(f"\nWeeks 4–5 Listings – Complete!")
