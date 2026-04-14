"""
Sold Transactions Analysis Pipeline
=====================================
Complete ETL pipeline for CRMLS Sold transaction data.
Covers: Week 1 (aggregation) → Weeks 2-3 (EDA + mortgage rates) →
        Weeks 4-5 (cleaning) → Week 6 (feature engineering) → Week 7 (outliers)

Input:  CRMLSSold202401.csv through CRMLSSold202603.csv (27 monthly files)
Output: data/processed/sold_final_clean.csv (Tableau-ready dataset)

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
PLOT_DIR = os.path.join(RAW_DIR, "outputs", "sold_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Key fields used throughout the analysis
DATE_COLS = ['CloseDate', 'PurchaseContractDate', 'ListingContractDate', 'ContractStatusChangeDate']
PRICE_COLS = ['ClosePrice', 'ListPrice', 'OriginalListPrice']
NUMERIC_COLS = ['ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
                'LotSizeAcres', 'BedroomsTotal', 'BathroomsTotalInteger',
                'DaysOnMarket', 'YearBuilt']
IQR_COLS = ['ClosePrice', 'LivingArea', 'DaysOnMarket']


# =============================================================================
# WEEK 1: Monthly Dataset Aggregation
# =============================================================================
def aggregate_monthly_files():
    """Load and concatenate all monthly Sold CSV files into one dataset."""
    print("=" * 70)
    print("WEEK 1: MONTHLY DATASET AGGREGATION")
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

    # PropertyType frequency before filter
    print(f"\n  PropertyType distribution (before filter):")
    pt_counts = sold['PropertyType'].value_counts()
    for pt, count in pt_counts.items():
        pct = count / len(sold) * 100
        print(f"    {pt}: {count:,} ({pct:.1f}%)")

    # Filter to Residential only
    rows_before = len(sold)
    sold = sold[sold['PropertyType'] == 'Residential'].copy()
    rows_after = len(sold)

    print(f"\n  Rows before Residential filter: {rows_before:,}")
    print(f"  Rows after Residential filter:  {rows_after:,}")
    print(f"  Rows removed: {rows_before - rows_after:,}")

    # PropertyType frequency after filter (confirm)
    print(f"\n  PropertyType distribution (after filter):")
    for pt, count in sold['PropertyType'].value_counts().items():
        print(f"    {pt}: {count:,}")

    return sold


# =============================================================================
# WEEKS 2-3: Exploratory Data Analysis
# =============================================================================
def run_eda(sold):
    """Perform EDA: missing values, distributions, and key questions."""
    print("\n" + "=" * 70)
    print("WEEKS 2-3: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # --- Dataset overview ---
    print(f"\n  Shape: {sold.shape[0]:,} rows x {sold.shape[1]} columns")
    print(f"\n  Data types:")
    for dtype, count in sold.dtypes.value_counts().items():
        print(f"    {dtype}: {count}")

    # --- Missing value analysis ---
    print(f"\n  --- Missing Value Report ---")
    null_report = sold.isnull().sum()
    null_pct = (null_report / len(sold) * 100).round(2)
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
    available_numeric = [c for c in NUMERIC_COLS if c in sold.columns]
    desc = sold[available_numeric].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(desc.to_string())

    # --- Key analyst questions ---
    print(f"\n  --- Key Market Questions ---")

    median_price = sold['ClosePrice'].median()
    mean_price = sold['ClosePrice'].mean()
    print(f"  Median close price: ${median_price:,.0f}")
    print(f"  Mean close price:   ${mean_price:,.0f}")

    median_dom = sold['DaysOnMarket'].median()
    print(f"  Median days on market: {median_dom:.0f}")

    if 'ListPrice' in sold.columns:
        above_list = (sold['ClosePrice'] > sold['ListPrice']).sum()
        below_list = (sold['ClosePrice'] < sold['ListPrice']).sum()
        at_list = (sold['ClosePrice'] == sold['ListPrice']).sum()
        total_valid = above_list + below_list + at_list
        print(f"  Sold above list price: {above_list:,} ({above_list/total_valid*100:.1f}%)")
        print(f"  Sold below list price: {below_list:,} ({below_list/total_valid*100:.1f}%)")
        print(f"  Sold at list price:    {at_list:,} ({at_list/total_valid*100:.1f}%)")

    print(f"\n  Top 10 counties by median close price:")
    county_median = sold.groupby('CountyOrParish')['ClosePrice'].median().sort_values(ascending=False)
    for county, price in county_median.head(10).items():
        count = sold[sold['CountyOrParish'] == county].shape[0]
        print(f"    {county}: ${price:,.0f} (n={count:,})")

    # --- Generate plots ---
    print(f"\n  Generating distribution plots...")
    for col in ['ClosePrice', 'LivingArea', 'DaysOnMarket']:
        if col not in sold.columns:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        data = sold[col].dropna()

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

    return sold


# =============================================================================
# WEEKS 2-3 (continued): Mortgage Rate Enrichment
# =============================================================================
def enrich_mortgage_rates(sold):
    """Fetch FRED MORTGAGE30US series and merge onto sold dataset."""
    print(f"\n  --- Mortgage Rate Enrichment ---")

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
    try:
        mortgage = pd.read_csv(url, parse_dates=['observation_date'])
        mortgage.columns = ['date', 'rate_30yr_fixed']
        # Remove any non-numeric rates (FRED sometimes has '.')
        mortgage['rate_30yr_fixed'] = pd.to_numeric(mortgage['rate_30yr_fixed'], errors='coerce')
        mortgage = mortgage.dropna(subset=['rate_30yr_fixed'])

        # Resample weekly to monthly average
        mortgage['year_month'] = mortgage['date'].dt.to_period('M')
        mortgage_monthly = (
            mortgage.groupby('year_month')['rate_30yr_fixed']
            .mean()
            .reset_index()
        )

        # Create year_month key on sold dataset
        sold['year_month'] = pd.to_datetime(sold['CloseDate']).dt.to_period('M')

        # Merge
        rows_before = len(sold)
        sold = sold.merge(mortgage_monthly, on='year_month', how='left')
        assert len(sold) == rows_before, "Merge changed row count!"

        null_rates = sold['rate_30yr_fixed'].isnull().sum()
        print(f"  Mortgage rates merged successfully.")
        print(f"  Unmatched rows (null rate): {null_rates}")
        print(f"  Rate range: {sold['rate_30yr_fixed'].min():.2f}% - {sold['rate_30yr_fixed'].max():.2f}%")
        print(f"  Sample:")
        print(sold[['CloseDate', 'year_month', 'ClosePrice', 'rate_30yr_fixed']].head().to_string())

    except Exception as e:
        print(f"  WARNING: Could not fetch mortgage rates from FRED: {e}")
        print(f"  Continuing without mortgage rate data.")
        sold['rate_30yr_fixed'] = np.nan

    return sold


# =============================================================================
# WEEKS 4-5: Data Cleaning and Preparation
# =============================================================================
def clean_data(sold):
    """Clean the dataset: type conversions, invalid values, date checks, geo checks."""
    print("\n" + "=" * 70)
    print("WEEKS 4-5: DATA CLEANING AND PREPARATION")
    print("=" * 70)

    rows_start = len(sold)
    print(f"\n  Starting rows: {rows_start:,}")

    # --- Convert date fields ---
    print(f"\n  --- Date Conversions ---")
    for col in DATE_COLS:
        if col in sold.columns:
            sold[col] = pd.to_datetime(sold[col], errors='coerce')
            null_count = sold[col].isnull().sum()
            print(f"  {col}: converted to datetime (nulls: {null_count:,})")

    # --- Ensure numeric fields are properly typed ---
    print(f"\n  --- Numeric Type Checks ---")
    for col in NUMERIC_COLS:
        if col in sold.columns:
            sold[col] = pd.to_numeric(sold[col], errors='coerce')

    # --- Flag invalid numeric values ---
    print(f"\n  --- Invalid Value Flags ---")
    sold['flag_price_invalid'] = sold['ClosePrice'] <= 0
    sold['flag_area_invalid'] = sold['LivingArea'] <= 0
    sold['flag_dom_invalid'] = sold['DaysOnMarket'] < 0
    sold['flag_beds_invalid'] = sold['BedroomsTotal'] < 0
    sold['flag_baths_invalid'] = sold['BathroomsTotalInteger'] < 0

    for flag_col in [c for c in sold.columns if c.startswith('flag_') and c.endswith('_invalid')]:
        count = sold[flag_col].sum()
        print(f"  {flag_col}: {count:,} records")

    # --- Date consistency checks ---
    print(f"\n  --- Date Consistency Checks ---")
    if 'ListingContractDate' in sold.columns and 'CloseDate' in sold.columns:
        sold['listing_after_close_flag'] = sold['ListingContractDate'] > sold['CloseDate']
        print(f"  listing_after_close_flag: {sold['listing_after_close_flag'].sum():,}")

    if 'PurchaseContractDate' in sold.columns and 'CloseDate' in sold.columns:
        sold['purchase_after_close_flag'] = sold['PurchaseContractDate'] > sold['CloseDate']
        print(f"  purchase_after_close_flag: {sold['purchase_after_close_flag'].sum():,}")

    if 'ListingContractDate' in sold.columns and 'PurchaseContractDate' in sold.columns:
        sold['negative_timeline_flag'] = sold['ListingContractDate'] > sold['PurchaseContractDate']
        print(f"  negative_timeline_flag: {sold['negative_timeline_flag'].sum():,}")

    # --- Geographic data checks ---
    print(f"\n  --- Geographic Data Quality ---")
    sold['flag_missing_coords'] = sold['Latitude'].isnull() | sold['Longitude'].isnull()
    sold['flag_zero_coords'] = (sold['Latitude'] == 0) | (sold['Longitude'] == 0)
    sold['flag_positive_lon'] = sold['Longitude'] > 0  # CA should be negative
    # CA rough bounding box: lat 32-42, lon -124 to -114
    sold['flag_out_of_state'] = (
        (sold['Latitude'].notna()) & (sold['Longitude'].notna()) &
        ~sold['flag_missing_coords'] & ~sold['flag_zero_coords'] &
        ((sold['Latitude'] < 32) | (sold['Latitude'] > 42) |
         (sold['Longitude'] < -125) | (sold['Longitude'] > -114))
    )

    geo_flags = ['flag_missing_coords', 'flag_zero_coords', 'flag_positive_lon', 'flag_out_of_state']
    for flag in geo_flags:
        print(f"  {flag}: {sold[flag].sum():,}")

    # --- Remove records with critical invalid values ---
    print(f"\n  --- Removing Invalid Records ---")
    valid_mask = (
        (sold['ClosePrice'] > 0) &
        (sold['ClosePrice'].notna())
    )
    rows_removed = (~valid_mask).sum()
    sold = sold[valid_mask].copy()
    print(f"  Removed {rows_removed:,} records with invalid ClosePrice")
    print(f"  Rows after cleaning: {len(sold):,}")

    # --- Data type summary ---
    print(f"\n  --- Final Data Type Summary ---")
    for dtype, count in sold.dtypes.value_counts().items():
        print(f"    {dtype}: {count}")

    return sold


# =============================================================================
# WEEK 6: Feature Engineering and Market Metrics
# =============================================================================
def engineer_features(sold):
    """Create key market indicators for Tableau dashboards."""
    print("\n" + "=" * 70)
    print("WEEK 6: FEATURE ENGINEERING AND MARKET METRICS")
    print("=" * 70)

    # --- Price metrics ---
    print(f"\n  --- Price Metrics ---")
    sold['price_ratio'] = sold['ClosePrice'] / sold['ListPrice']
    sold['close_to_original_list_ratio'] = sold['ClosePrice'] / sold['OriginalListPrice']
    sold['price_per_sqft'] = sold['ClosePrice'] / sold['LivingArea']

    for col in ['price_ratio', 'close_to_original_list_ratio', 'price_per_sqft']:
        valid = sold[col].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  {col}: median={valid.median():.4f}, mean={valid.mean():.4f}")

    # Replace inf values with NaN
    sold['price_ratio'] = sold['price_ratio'].replace([np.inf, -np.inf], np.nan)
    sold['close_to_original_list_ratio'] = sold['close_to_original_list_ratio'].replace([np.inf, -np.inf], np.nan)
    sold['price_per_sqft'] = sold['price_per_sqft'].replace([np.inf, -np.inf], np.nan)

    # --- Time-series variables ---
    print(f"\n  --- Time-Series Variables ---")
    sold['close_year'] = sold['CloseDate'].dt.year
    sold['close_month'] = sold['CloseDate'].dt.month
    sold['close_yrmo'] = sold['CloseDate'].dt.to_period('M').astype(str)

    print(f"  Year range: {sold['close_year'].min()} - {sold['close_year'].max()}")
    print(f"  Months covered: {sold['close_yrmo'].nunique()}")

    # --- Timeline duration metrics ---
    print(f"\n  --- Timeline Duration Metrics ---")
    if 'ListingContractDate' in sold.columns and 'PurchaseContractDate' in sold.columns:
        sold['listing_to_contract_days'] = (
            sold['PurchaseContractDate'] - sold['ListingContractDate']
        ).dt.days

    if 'PurchaseContractDate' in sold.columns and 'CloseDate' in sold.columns:
        sold['contract_to_close_days'] = (
            sold['CloseDate'] - sold['PurchaseContractDate']
        ).dt.days

    for col in ['listing_to_contract_days', 'contract_to_close_days']:
        if col in sold.columns:
            valid = sold[col].dropna()
            print(f"  {col}: median={valid.median():.0f}, mean={valid.mean():.1f}")

    # --- Segment analysis ---
    print(f"\n  --- Segment Analysis: PropertySubType ---")
    if 'PropertySubType' in sold.columns:
        seg = sold.groupby('PropertySubType').agg(
            count=('ClosePrice', 'size'),
            median_price=('ClosePrice', 'median'),
            median_dom=('DaysOnMarket', 'median'),
            median_ppsf=('price_per_sqft', 'median')
        ).sort_values('count', ascending=False)
        print(seg.head(10).to_string())

    print(f"\n  --- Segment Analysis: CountyOrParish (Top 10) ---")
    seg_county = sold.groupby('CountyOrParish').agg(
        count=('ClosePrice', 'size'),
        median_price=('ClosePrice', 'median'),
        median_dom=('DaysOnMarket', 'median'),
        median_ppsf=('price_per_sqft', 'median')
    ).sort_values('count', ascending=False)
    print(seg_county.head(10).to_string())

    # --- Sample output ---
    print(f"\n  --- Sample Output (engineered columns) ---")
    eng_cols = ['CloseDate', 'close_yrmo', 'ClosePrice', 'ListPrice', 'OriginalListPrice',
                'price_ratio', 'close_to_original_list_ratio', 'price_per_sqft',
                'DaysOnMarket', 'listing_to_contract_days', 'contract_to_close_days',
                'rate_30yr_fixed']
    available = [c for c in eng_cols if c in sold.columns]
    print(sold[available].head(5).to_string())

    return sold


# =============================================================================
# WEEK 7: Outlier Detection and Data Quality
# =============================================================================
def detect_outliers(sold):
    """Apply IQR-based outlier flagging and create a filtered dataset."""
    print("\n" + "=" * 70)
    print("WEEK 7: OUTLIER DETECTION AND DATA QUALITY")
    print("=" * 70)

    print(f"\n  Rows before outlier detection: {len(sold):,}")

    for col in IQR_COLS:
        if col not in sold.columns:
            continue

        data = sold[col].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        flag_col = f'outlier_{col.lower()}'
        sold[flag_col] = (sold[col] < lower) | (sold[col] > upper)
        outlier_count = sold[flag_col].sum()

        print(f"\n  {col}:")
        print(f"    Q1={Q1:,.2f}, Q3={Q3:,.2f}, IQR={IQR:,.2f}")
        print(f"    Lower bound: {lower:,.2f}")
        print(f"    Upper bound: {upper:,.2f}")
        print(f"    Outliers flagged: {outlier_count:,} ({outlier_count/len(sold)*100:.1f}%)")

    # Create combined outlier flag
    outlier_flags = [c for c in sold.columns if c.startswith('outlier_')]
    sold['outlier_any'] = sold[outlier_flags].any(axis=1)
    print(f"\n  Records with ANY outlier flag: {sold['outlier_any'].sum():,}")

    # --- Before/after comparison ---
    print(f"\n  --- Before vs After Outlier Filtering ---")
    clean = sold[~sold['outlier_any']]
    print(f"  {'Metric':<25} {'Full Dataset':>15} {'After Filtering':>15}")
    print(f"  {'-'*55}")
    print(f"  {'Row count':<25} {len(sold):>15,} {len(clean):>15,}")
    for col in IQR_COLS:
        if col in sold.columns:
            full_med = sold[col].median()
            clean_med = clean[col].median()
            print(f"  {'Median ' + col:<25} {full_med:>15,.2f} {clean_med:>15,.2f}")

    return sold, clean


# =============================================================================
# SAVE OUTPUTS
# =============================================================================
def save_outputs(sold_full, sold_clean):
    """Save both the full flagged dataset and the clean filtered dataset."""
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    # Drop temporary helper columns not needed for Tableau
    drop_cols = ['year_month']
    for col in drop_cols:
        if col in sold_full.columns:
            sold_full = sold_full.drop(columns=[col])
        if col in sold_clean.columns:
            sold_clean = sold_clean.drop(columns=[col])

    # Save full flagged dataset
    full_path = os.path.join(OUTPUT_DIR, "sold_full_flagged.csv")
    sold_full.to_csv(full_path, index=False)
    print(f"  Full flagged dataset: {full_path}")
    print(f"    Rows: {len(sold_full):,}, Columns: {len(sold_full.columns)}")

    # Save clean filtered dataset (Tableau-ready)
    clean_path = os.path.join(OUTPUT_DIR, "sold_final_clean.csv")
    sold_clean.to_csv(clean_path, index=False)
    print(f"  Clean filtered dataset: {clean_path}")
    print(f"    Rows: {len(sold_clean):,}, Columns: {len(sold_clean.columns)}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "#" * 70)
    print("#  SOLD TRANSACTIONS ANALYSIS PIPELINE")
    print("#  Data range: January 2024 - March 2026")
    print("#" * 70)

    # Week 1: Aggregate
    sold = aggregate_monthly_files()

    # Weeks 2-3: EDA + Mortgage Rates
    sold = run_eda(sold)
    sold = enrich_mortgage_rates(sold)

    # Weeks 4-5: Clean
    sold = clean_data(sold)

    # Week 6: Feature Engineering
    sold = engineer_features(sold)

    # Week 7: Outlier Detection
    sold_full, sold_clean = detect_outliers(sold)

    # Save
    save_outputs(sold_full, sold_clean)

    print("\n" + "#" * 70)
    print("#  PIPELINE COMPLETE")
    print("#" * 70)


if __name__ == "__main__":
    main()
