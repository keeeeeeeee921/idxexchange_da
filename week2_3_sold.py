"""
Weeks 2–3 – Sold Transactions: EDA, Validation & Mortgage Rate Enrichment
===========================================================================
Performs exploratory data analysis on the combined Sold dataset, generates
missing value reports, numeric distribution summaries, and enriches the
dataset with FRED 30-year fixed mortgage rate data.

Input:  data/processed/week1_sold.csv
Output: data/processed/week2_3_sold.csv
Plots:  outputs/sold_plots/
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "week1_sold.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
PLOT_DIR = os.path.join(BASE_DIR, "outputs", "sold_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Key numeric fields for distribution analysis
NUMERIC_FIELDS = ['ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
                  'LotSizeAcres', 'BedroomsTotal', 'BathroomsTotalInteger',
                  'DaysOnMarket', 'YearBuilt']

# =============================================================================
# Load Week 1 output
# =============================================================================
print("=" * 70)
print("WEEKS 2–3 – SOLD: EDA, VALIDATION & MORTGAGE RATE ENRICHMENT")
print("=" * 70)

sold = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"\nLoaded: {INPUT_PATH}")
print(f"Shape: {sold.shape[0]:,} rows x {sold.shape[1]} columns")

# =============================================================================
# Part 1: Dataset Understanding
# =============================================================================
print(f"\n{'='*70}")
print("PART 1: DATASET UNDERSTANDING")
print(f"{'='*70}")

print(f"\n--- Data Types ---")
for dtype, count in sold.dtypes.value_counts().items():
    print(f"  {dtype}: {count}")

print(f"\n--- First 5 Rows (key fields) ---")
key_fields = ['CloseDate', 'ClosePrice', 'ListPrice', 'LivingArea',
              'DaysOnMarket', 'PropertySubType', 'CountyOrParish', 'City']
available = [c for c in key_fields if c in sold.columns]
print(sold[available].head().to_string())

# =============================================================================
# Part 2: Missing Value Analysis
# =============================================================================
print(f"\n{'='*70}")
print("PART 2: MISSING VALUE ANALYSIS")
print(f"{'='*70}")

null_counts = sold.isnull().sum()
null_pct = (null_counts / len(sold) * 100).round(2)
null_df = pd.DataFrame({
    'missing_count': null_counts,
    'missing_pct': null_pct
}).sort_values('missing_pct', ascending=False)

# Columns with >90% missing — candidates for dropping
print(f"\n--- Columns with >90% missing (candidates for dropping) ---")
high_missing = null_df[null_df['missing_pct'] > 90]
for col, row in high_missing.iterrows():
    print(f"  {col}: {row['missing_pct']:.1f}% ({int(row['missing_count']):,} nulls)")

# Columns with 50-90% missing
print(f"\n--- Columns with 50–90% missing ---")
mid_missing = null_df[(null_df['missing_pct'] > 50) & (null_df['missing_pct'] <= 90)]
for col, row in mid_missing.iterrows():
    print(f"  {col}: {row['missing_pct']:.1f}% ({int(row['missing_count']):,} nulls)")

# Core fields — check completeness
print(f"\n--- Core Field Completeness ---")
core_fields = ['ClosePrice', 'ListPrice', 'CloseDate', 'LivingArea',
               'DaysOnMarket', 'CountyOrParish', 'PropertySubType',
               'Latitude', 'Longitude']
for col in core_fields:
    if col in sold.columns:
        pct = null_df.loc[col, 'missing_pct']
        print(f"  {col}: {pct:.1f}% missing")

# =============================================================================
# Part 3: Numeric Distribution Review
# =============================================================================
print(f"\n{'='*70}")
print("PART 3: NUMERIC DISTRIBUTION REVIEW")
print(f"{'='*70}")

available_numeric = [c for c in NUMERIC_FIELDS if c in sold.columns]
desc = sold[available_numeric].describe(
    percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
)
print(f"\n{desc.to_string()}")

# Generate histograms and boxplots
print(f"\n--- Generating Distribution Plots ---")
for col in ['ClosePrice', 'LivingArea', 'DaysOnMarket']:
    if col not in sold.columns:
        continue
    data = sold[col].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_title(f'Sold – {col} Histogram')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Frequency')

    axes[1].boxplot(data, vert=True)
    axes[1].set_title(f'Sold – {col} Boxplot')
    axes[1].set_ylabel(col)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f'dist_{col.lower()}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

# =============================================================================
# Part 4: Key Analyst Questions
# =============================================================================
print(f"\n{'='*70}")
print("PART 4: KEY ANALYST QUESTIONS")
print(f"{'='*70}")

# Q1: Median and average close prices
median_price = sold['ClosePrice'].median()
mean_price = sold['ClosePrice'].mean()
print(f"\n  Q1: Median close price: ${median_price:,.0f}")
print(f"      Mean close price:   ${mean_price:,.0f}")

# Q2: Days on Market distribution
median_dom = sold['DaysOnMarket'].median()
mean_dom = sold['DaysOnMarket'].mean()
print(f"\n  Q2: Median days on market: {median_dom:.0f}")
print(f"      Mean days on market:   {mean_dom:.1f}")

# Q3: Percentage sold above vs below list price
above = (sold['ClosePrice'] > sold['ListPrice']).sum()
below = (sold['ClosePrice'] < sold['ListPrice']).sum()
at = (sold['ClosePrice'] == sold['ListPrice']).sum()
total = above + below + at
print(f"\n  Q3: Sold above list price: {above:,} ({above/total*100:.1f}%)")
print(f"      Sold below list price: {below:,} ({below/total*100:.1f}%)")
print(f"      Sold at list price:    {at:,} ({at/total*100:.1f}%)")

# Q4: Date consistency issues
sold['CloseDate_dt'] = pd.to_datetime(sold['CloseDate'], errors='coerce')
sold['ListingContractDate_dt'] = pd.to_datetime(sold['ListingContractDate'], errors='coerce')
date_issue = (sold['ListingContractDate_dt'] > sold['CloseDate_dt']).sum()
print(f"\n  Q4: Records with listing date after close date: {date_issue:,}")

# Q5: Top counties by median price
print(f"\n  Q5: Top 10 counties by median close price:")
county_stats = sold.groupby('CountyOrParish').agg(
    count=('ClosePrice', 'size'),
    median_price=('ClosePrice', 'median')
).sort_values('median_price', ascending=False)
for county, row in county_stats.head(10).iterrows():
    print(f"      {county}: ${row['median_price']:,.0f} (n={int(row['count']):,})")

# Clean up temp columns
sold = sold.drop(columns=['CloseDate_dt', 'ListingContractDate_dt'])

# =============================================================================
# Part 5: Mortgage Rate Enrichment (FRED MORTGAGE30US)
# =============================================================================
print(f"\n{'='*70}")
print("PART 5: MORTGAGE RATE ENRICHMENT")
print(f"{'='*70}")

url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
try:
    mortgage = pd.read_csv(url, parse_dates=['observation_date'])
    mortgage.columns = ['date', 'rate_30yr_fixed']
    mortgage['rate_30yr_fixed'] = pd.to_numeric(mortgage['rate_30yr_fixed'], errors='coerce')
    mortgage = mortgage.dropna(subset=['rate_30yr_fixed'])

    # Resample weekly → monthly average
    mortgage['year_month'] = mortgage['date'].dt.to_period('M')
    mortgage_monthly = (
        mortgage.groupby('year_month')['rate_30yr_fixed']
        .mean()
        .reset_index()
    )
    print(f"  FRED data fetched: {len(mortgage_monthly)} monthly rates")

    # Create year_month key on sold (keyed off CloseDate)
    sold['year_month'] = pd.to_datetime(sold['CloseDate']).dt.to_period('M')

    # Merge
    rows_before = len(sold)
    sold = sold.merge(mortgage_monthly, on='year_month', how='left')
    assert len(sold) == rows_before, "Merge changed row count!"

    null_rates = sold['rate_30yr_fixed'].isnull().sum()
    print(f"  Merged successfully. Unmatched rows: {null_rates}")
    print(f"  Rate range: {sold['rate_30yr_fixed'].min():.2f}% – {sold['rate_30yr_fixed'].max():.2f}%")

    # Preview
    print(f"\n  Sample:")
    preview_cols = ['CloseDate', 'year_month', 'ClosePrice', 'rate_30yr_fixed']
    print(sold[preview_cols].head(5).to_string(index=False))

except Exception as e:
    print(f"  WARNING: Could not fetch FRED data: {e}")
    print(f"  Continuing without mortgage rate enrichment.")
    sold['rate_30yr_fixed'] = np.nan
    sold['year_month'] = pd.to_datetime(sold['CloseDate']).dt.to_period('M')

# =============================================================================
# Save output
# =============================================================================
print(f"\n{'='*70}")
print("SAVING OUTPUT")
print(f"{'='*70}")

output_path = os.path.join(OUTPUT_DIR, "week2_3_sold.csv")
sold.to_csv(output_path, index=False)
print(f"  Saved to: {output_path}")
print(f"  Final shape: {sold.shape[0]:,} rows x {sold.shape[1]} columns")
print(f"\nWeeks 2–3 Sold – Complete!")
