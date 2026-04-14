"""
Listings Analysis Pipeline
============================
Complete ETL pipeline for CRMLS Listing data.
Covers: Week 1 (aggregation) → Weeks 2-3 (EDA + mortgage rates) →
        Weeks 4-5 (cleaning) → Week 6 (feature engineering) → Week 7 (outliers)

Input:  CRMLSListing202401.csv through CRMLSListing202603.csv (27 monthly files)
Output: data/processed/listed_final_clean.csv (Tableau-ready dataset)

Note: Do NOT upload this output CSV or the raw CSVs to GitHub.
"""

import pandas as pd
import numpy as np
import glob
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Configuration
# =============================================================================
RAW_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(RAW_DIR, "data", "processed")
PLOT_DIR = os.path.join(RAW_DIR, "outputs", "listed_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Key fields for listings analysis
DATE_COLS = ['ListingContractDate', 'ContractStatusChangeDate']
NUMERIC_COLS = ['ListPrice', 'OriginalListPrice', 'LivingArea',
                'LotSizeAcres', 'BedroomsTotal', 'BathroomsTotalInteger',
                'DaysOnMarket', 'YearBuilt']
IQR_COLS = ['ListPrice', 'LivingArea', 'DaysOnMarket']


# =============================================================================
# WEEK 1: Monthly Dataset Aggregation
# =============================================================================
def aggregate_monthly_files():
    """Load and concatenate all monthly Listing CSV files into one dataset."""
    print("=" * 70)
    print("WEEK 1: MONTHLY DATASET AGGREGATION")
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

    # Handle duplicate column names (e.g., PropertyType.1, ListPrice.1)
    # These are artifacts from the CSV export — drop the .1 suffix duplicates
    dup_cols = [c for c in listings.columns if c.endswith('.1')]
    if dup_cols:
        print(f"\n  Dropping {len(dup_cols)} duplicate columns: {dup_cols}")
        listings = listings.drop(columns=dup_cols)
        print(f"  Columns after dropping duplicates: {len(listings.columns)}")

    # PropertyType frequency before filter
    print(f"\n  PropertyType distribution (before filter):")
    pt_counts = listings['PropertyType'].value_counts()
    for pt, count in pt_counts.items():
        pct = count / len(listings) * 100
        print(f"    {pt}: {count:,} ({pct:.1f}%)")

    # Filter to Residential only
    rows_before = len(listings)
    listings = listings[listings['PropertyType'] == 'Residential'].copy()
    rows_after = len(listings)

    print(f"\n  Rows before Residential filter: {rows_before:,}")
    print(f"  Rows after Residential filter:  {rows_after:,}")
    print(f"  Rows removed: {rows_before - rows_after:,}")

    # PropertyType frequency after filter
    print(f"\n  PropertyType distribution (after filter):")
    for pt, count in listings['PropertyType'].value_counts().items():
        print(f"    {pt}: {count:,}")

    return listings


# =============================================================================
# WEEKS 2-3: Exploratory Data Analysis
# =============================================================================
def run_eda(listings):
    """Perform EDA: missing values, distributions, and key questions."""
    print("\n" + "=" * 70)
    print("WEEKS 2-3: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # --- Dataset overview ---
    print(f"\n  Shape: {listings.shape[0]:,} rows x {listings.shape[1]} columns")
    print(f"\n  Data types:")
    for dtype, count in listings.dtypes.value_counts().items():
        print(f"    {dtype}: {count}")

    # --- Missing value analysis ---
    print(f"\n  --- Missing Value Report ---")
    null_report = listings.isnull().sum()
    null_pct = (null_report / len(listings) * 100).round(2)
    null_df = pd.DataFrame({'missing_count': null_report, 'missing_pct': null_pct})
    null_df = null_df.sort_values('missing_pct', ascending=False)

    print(f"\n  Columns with >90% missing:")
    high_missing = null_df[null_df['missing_pct'] > 90]
    if len(high_missing) > 0:
        for col, row in high_missing.iterrows():
            print(f"    {col}: {row['missing_pct']:.1f}%")
    else:
        print("    None")

    print(f"\n  Columns with >50% missing:")
    mid_missing = null_df[(null_df['missing_pct'] > 50) & (null_df['missing_pct'] <= 90)]
    for col, row in mid_missing.iterrows():
        print(f"    {col}: {row['missing_pct']:.1f}%")

    # --- Numeric distribution summary ---
    print(f"\n  --- Numeric Distribution Summary ---")
    available_numeric = [c for c in NUMERIC_COLS if c in listings.columns]
    desc = listings[available_numeric].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(desc.to_string())

    # --- Key analyst questions ---
    print(f"\n  --- Key Market Questions ---")

    median_price = listings['ListPrice'].median()
    mean_price = listings['ListPrice'].mean()
    print(f"  Median list price: ${median_price:,.0f}")
    print(f"  Mean list price:   ${mean_price:,.0f}")

    median_dom = listings['DaysOnMarket'].median()
    print(f"  Median days on market: {median_dom:.0f}")

    print(f"\n  Top 10 counties by median list price:")
    county_median = listings.groupby('CountyOrParish')['ListPrice'].median().sort_values(ascending=False)
    for county, price in county_median.head(10).items():
        count = listings[listings['CountyOrParish'] == county].shape[0]
        print(f"    {county}: ${price:,.0f} (n={count:,})")

    # --- MLS Status distribution ---
    if 'MlsStatus' in listings.columns:
        print(f"\n  MLS Status distribution:")
        for status, count in listings['MlsStatus'].value_counts().items():
            pct = count / len(listings) * 100
            print(f"    {status}: {count:,} ({pct:.1f}%)")

    # --- Generate plots ---
    print(f"\n  Generating distribution plots...")
    for col in ['ListPrice', 'LivingArea', 'DaysOnMarket']:
        if col not in listings.columns:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        data = listings[col].dropna()

        axes[0].hist(data, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_title(f'{col} - Histogram')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frequency')

        axes[1].boxplot(data, vert=True)
        axes[1].set_title(f'{col} - Boxplot')
        axes[1].set_ylabel(col)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f'dist_{col.lower()}.png'), dpi=150)
        plt.close()
        print(f"    Saved: dist_{col.lower()}.png")

    return listings


# =============================================================================
# WEEKS 2-3 (continued): Mortgage Rate Enrichment
# =============================================================================
def enrich_mortgage_rates(listings):
    """Fetch FRED MORTGAGE30US series and merge onto listings dataset."""
    print(f"\n  --- Mortgage Rate Enrichment ---")

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
    try:
        mortgage = pd.read_csv(url, parse_dates=['observation_date'])
        mortgage.columns = ['date', 'rate_30yr_fixed']
        mortgage['rate_30yr_fixed'] = pd.to_numeric(mortgage['rate_30yr_fixed'], errors='coerce')
        mortgage = mortgage.dropna(subset=['rate_30yr_fixed'])

        # Resample weekly to monthly average
        mortgage['year_month'] = mortgage['date'].dt.to_period('M')
        mortgage_monthly = (
            mortgage.groupby('year_month')['rate_30yr_fixed']
            .mean()
            .reset_index()
        )

        # Create year_month key on listings (keyed off ListingContractDate)
        listings['year_month'] = pd.to_datetime(listings['ListingContractDate']).dt.to_period('M')

        # Merge
        rows_before = len(listings)
        listings = listings.merge(mortgage_monthly, on='year_month', how='left')
        assert len(listings) == rows_before, "Merge changed row count!"

        null_rates = listings['rate_30yr_fixed'].isnull().sum()
        print(f"  Mortgage rates merged successfully.")
        print(f"  Unmatched rows (null rate): {null_rates}")
        print(f"  Rate range: {listings['rate_30yr_fixed'].min():.2f}% - {listings['rate_30yr_fixed'].max():.2f}%")

    except Exception as e:
        print(f"  WARNING: Could not fetch mortgage rates from FRED: {e}")
        print(f"  Continuing without mortgage rate data.")
        listings['rate_30yr_fixed'] = np.nan

    return listings


# =============================================================================
# WEEKS 4-5: Data Cleaning and Preparation
# =============================================================================
def clean_data(listings):
    """Clean the dataset: type conversions, invalid values, geo checks."""
    print("\n" + "=" * 70)
    print("WEEKS 4-5: DATA CLEANING AND PREPARATION")
    print("=" * 70)

    rows_start = len(listings)
    print(f"\n  Starting rows: {rows_start:,}")

    # --- Convert date fields ---
    print(f"\n  --- Date Conversions ---")
    for col in DATE_COLS:
        if col in listings.columns:
            listings[col] = pd.to_datetime(listings[col], errors='coerce')
            null_count = listings[col].isnull().sum()
            print(f"  {col}: converted to datetime (nulls: {null_count:,})")

    # --- Ensure numeric fields are properly typed ---
    print(f"\n  --- Numeric Type Checks ---")
    for col in NUMERIC_COLS:
        if col in listings.columns:
            listings[col] = pd.to_numeric(listings[col], errors='coerce')

    # --- Flag invalid numeric values ---
    print(f"\n  --- Invalid Value Flags ---")
    listings['flag_price_invalid'] = listings['ListPrice'] <= 0
    listings['flag_area_invalid'] = listings['LivingArea'] <= 0
    listings['flag_dom_invalid'] = listings['DaysOnMarket'] < 0
    listings['flag_beds_invalid'] = listings['BedroomsTotal'] < 0
    listings['flag_baths_invalid'] = listings['BathroomsTotalInteger'] < 0

    for flag_col in [c for c in listings.columns if c.startswith('flag_') and c.endswith('_invalid')]:
        count = listings[flag_col].sum()
        print(f"  {flag_col}: {count:,} records")

    # --- Geographic data checks ---
    print(f"\n  --- Geographic Data Quality ---")
    listings['flag_missing_coords'] = listings['Latitude'].isnull() | listings['Longitude'].isnull()
    listings['flag_zero_coords'] = (listings['Latitude'] == 0) | (listings['Longitude'] == 0)
    listings['flag_positive_lon'] = listings['Longitude'] > 0
    listings['flag_out_of_state'] = (
        (listings['Latitude'].notna()) & (listings['Longitude'].notna()) &
        ~listings['flag_missing_coords'] & ~listings['flag_zero_coords'] &
        ((listings['Latitude'] < 32) | (listings['Latitude'] > 42) |
         (listings['Longitude'] < -125) | (listings['Longitude'] > -114))
    )

    geo_flags = ['flag_missing_coords', 'flag_zero_coords', 'flag_positive_lon', 'flag_out_of_state']
    for flag in geo_flags:
        print(f"  {flag}: {listings[flag].sum():,}")

    # --- Remove records with critical invalid values ---
    print(f"\n  --- Removing Invalid Records ---")
    valid_mask = (
        (listings['ListPrice'] > 0) &
        (listings['ListPrice'].notna())
    )
    rows_removed = (~valid_mask).sum()
    listings = listings[valid_mask].copy()
    print(f"  Removed {rows_removed:,} records with invalid ListPrice")
    print(f"  Rows after cleaning: {len(listings):,}")

    # --- Data type summary ---
    print(f"\n  --- Final Data Type Summary ---")
    for dtype, count in listings.dtypes.value_counts().items():
        print(f"    {dtype}: {count}")

    return listings


# =============================================================================
# WEEK 6: Feature Engineering and Market Metrics
# =============================================================================
def engineer_features(listings):
    """Create key market indicators for Tableau dashboards."""
    print("\n" + "=" * 70)
    print("WEEK 6: FEATURE ENGINEERING AND MARKET METRICS")
    print("=" * 70)

    # --- Price metrics ---
    print(f"\n  --- Price Metrics ---")
    listings['price_per_sqft'] = listings['ListPrice'] / listings['LivingArea']
    listings['price_per_sqft'] = listings['price_per_sqft'].replace([np.inf, -np.inf], np.nan)

    if 'OriginalListPrice' in listings.columns:
        listings['list_to_original_ratio'] = listings['ListPrice'] / listings['OriginalListPrice']
        listings['list_to_original_ratio'] = listings['list_to_original_ratio'].replace([np.inf, -np.inf], np.nan)
        valid = listings['list_to_original_ratio'].dropna()
        print(f"  list_to_original_ratio: median={valid.median():.4f}, mean={valid.mean():.4f}")

    valid_ppsf = listings['price_per_sqft'].dropna()
    print(f"  price_per_sqft: median={valid_ppsf.median():.2f}, mean={valid_ppsf.mean():.2f}")

    # --- Time-series variables ---
    print(f"\n  --- Time-Series Variables ---")
    listings['list_year'] = listings['ListingContractDate'].dt.year
    listings['list_month'] = listings['ListingContractDate'].dt.month
    listings['list_yrmo'] = listings['ListingContractDate'].dt.to_period('M').astype(str)

    print(f"  Year range: {listings['list_year'].min()} - {listings['list_year'].max()}")
    print(f"  Months covered: {listings['list_yrmo'].nunique()}")

    # --- Segment analysis ---
    print(f"\n  --- Segment Analysis: PropertySubType ---")
    if 'PropertySubType' in listings.columns:
        seg = listings.groupby('PropertySubType').agg(
            count=('ListPrice', 'size'),
            median_price=('ListPrice', 'median'),
            median_dom=('DaysOnMarket', 'median'),
            median_ppsf=('price_per_sqft', 'median')
        ).sort_values('count', ascending=False)
        print(seg.head(10).to_string())

    print(f"\n  --- Segment Analysis: CountyOrParish (Top 10) ---")
    seg_county = listings.groupby('CountyOrParish').agg(
        count=('ListPrice', 'size'),
        median_price=('ListPrice', 'median'),
        median_dom=('DaysOnMarket', 'median'),
        median_ppsf=('price_per_sqft', 'median')
    ).sort_values('count', ascending=False)
    print(seg_county.head(10).to_string())

    # --- MLS Status breakdown by month ---
    if 'MlsStatus' in listings.columns:
        print(f"\n  --- Monthly New Listings Count (sample) ---")
        monthly_count = listings.groupby('list_yrmo').size()
        print(monthly_count.tail(6).to_string())

    # --- Sample output ---
    print(f"\n  --- Sample Output (engineered columns) ---")
    eng_cols = ['ListingContractDate', 'list_yrmo', 'ListPrice', 'OriginalListPrice',
                'price_per_sqft', 'list_to_original_ratio', 'DaysOnMarket',
                'rate_30yr_fixed']
    available = [c for c in eng_cols if c in listings.columns]
    print(listings[available].head(5).to_string())

    return listings


# =============================================================================
# WEEK 7: Outlier Detection and Data Quality
# =============================================================================
def detect_outliers(listings):
    """Apply IQR-based outlier flagging and create a filtered dataset."""
    print("\n" + "=" * 70)
    print("WEEK 7: OUTLIER DETECTION AND DATA QUALITY")
    print("=" * 70)

    print(f"\n  Rows before outlier detection: {len(listings):,}")

    for col in IQR_COLS:
        if col not in listings.columns:
            continue

        data = listings[col].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        flag_col = f'outlier_{col.lower()}'
        listings[flag_col] = (listings[col] < lower) | (listings[col] > upper)
        outlier_count = listings[flag_col].sum()

        print(f"\n  {col}:")
        print(f"    Q1={Q1:,.2f}, Q3={Q3:,.2f}, IQR={IQR:,.2f}")
        print(f"    Lower bound: {lower:,.2f}")
        print(f"    Upper bound: {upper:,.2f}")
        print(f"    Outliers flagged: {outlier_count:,} ({outlier_count/len(listings)*100:.1f}%)")

    # Create combined outlier flag
    outlier_flags = [c for c in listings.columns if c.startswith('outlier_')]
    listings['outlier_any'] = listings[outlier_flags].any(axis=1)
    print(f"\n  Records with ANY outlier flag: {listings['outlier_any'].sum():,}")

    # --- Before/after comparison ---
    print(f"\n  --- Before vs After Outlier Filtering ---")
    clean = listings[~listings['outlier_any']]
    print(f"  {'Metric':<25} {'Full Dataset':>15} {'After Filtering':>15}")
    print(f"  {'-'*55}")
    print(f"  {'Row count':<25} {len(listings):>15,} {len(clean):>15,}")
    for col in IQR_COLS:
        if col in listings.columns:
            full_med = listings[col].median()
            clean_med = clean[col].median()
            print(f"  {'Median ' + col:<25} {full_med:>15,.2f} {clean_med:>15,.2f}")

    return listings, clean


# =============================================================================
# SAVE OUTPUTS
# =============================================================================
def save_outputs(listings_full, listings_clean):
    """Save both the full flagged dataset and the clean filtered dataset."""
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    # Drop temporary helper columns
    drop_cols = ['year_month']
    for col in drop_cols:
        if col in listings_full.columns:
            listings_full = listings_full.drop(columns=[col])
        if col in listings_clean.columns:
            listings_clean = listings_clean.drop(columns=[col])

    # Save full flagged dataset
    full_path = os.path.join(OUTPUT_DIR, "listed_full_flagged.csv")
    listings_full.to_csv(full_path, index=False)
    print(f"  Full flagged dataset: {full_path}")
    print(f"    Rows: {len(listings_full):,}, Columns: {len(listings_full.columns)}")

    # Save clean filtered dataset (Tableau-ready)
    clean_path = os.path.join(OUTPUT_DIR, "listed_final_clean.csv")
    listings_clean.to_csv(clean_path, index=False)
    print(f"  Clean filtered dataset: {clean_path}")
    print(f"    Rows: {len(listings_clean):,}, Columns: {len(listings_clean.columns)}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "#" * 70)
    print("#  LISTINGS ANALYSIS PIPELINE")
    print("#  Data range: January 2024 - March 2026")
    print("#" * 70)

    # Week 1: Aggregate
    listings = aggregate_monthly_files()

    # Weeks 2-3: EDA + Mortgage Rates
    listings = run_eda(listings)
    listings = enrich_mortgage_rates(listings)

    # Weeks 4-5: Clean
    listings = clean_data(listings)

    # Week 6: Feature Engineering
    listings = engineer_features(listings)

    # Week 7: Outlier Detection
    listings_full, listings_clean = detect_outliers(listings)

    # Save
    save_outputs(listings_full, listings_clean)

    print("\n" + "#" * 70)
    print("#  PIPELINE COMPLETE")
    print("#" * 70)


if __name__ == "__main__":
    main()
