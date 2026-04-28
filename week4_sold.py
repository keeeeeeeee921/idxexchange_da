"""
Week 4 – Sold Transactions: Time Series & Trend Analysis
=========================================================
Aggregates sold transactions by month to reveal market trends:
price trajectory, sales volume, days on market, and the relationship
between mortgage rates and market activity.

Input:  data/processed/week2_3_sold.csv
Output: data/processed/week4_sold.csv
        data/processed/week4_sold_monthly.csv
Plots:  outputs/sold_plots/
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "week2_3_sold.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
PLOT_DIR   = os.path.join(BASE_DIR, "outputs", "sold_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,   exist_ok=True)

# =============================================================================
# Load Week 2-3 output
# =============================================================================
print("=" * 70)
print("WEEK 4 – SOLD: TIME SERIES & TREND ANALYSIS")
print("=" * 70)

sold = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"\nLoaded: {INPUT_PATH}")
print(f"Shape: {sold.shape[0]:,} rows x {sold.shape[1]} columns")

# =============================================================================
# Part 1: Feature Engineering (transaction-level)
# =============================================================================
print(f"\n{'='*70}")
print("PART 1: TRANSACTION-LEVEL FEATURE ENGINEERING")
print(f"{'='*70}")

# List-to-sale ratio (ClosePrice / ListPrice)
sold['list_to_sale_ratio'] = sold['ClosePrice'] / sold['ListPrice']

# Price per square foot
sold['price_per_sqft'] = sold['ClosePrice'] / sold['LivingArea']

# Flag: sold above list price
sold['sold_above_list'] = (sold['ClosePrice'] > sold['ListPrice']).astype(int)

# Parse CloseDate
sold['CloseDate_dt'] = pd.to_datetime(sold['CloseDate'], errors='coerce')
sold['close_year']  = sold['CloseDate_dt'].dt.year
sold['close_month'] = sold['CloseDate_dt'].dt.month

print(f"  list_to_sale_ratio — mean: {sold['list_to_sale_ratio'].mean():.4f}  "
      f"median: {sold['list_to_sale_ratio'].median():.4f}")
print(f"  price_per_sqft     — mean: ${sold['price_per_sqft'].mean():,.0f}  "
      f"median: ${sold['price_per_sqft'].median():,.0f}")
print(f"  sold_above_list    — {sold['sold_above_list'].sum():,} records "
      f"({sold['sold_above_list'].mean()*100:.1f}%)")

# =============================================================================
# Part 2: Monthly Aggregation
# =============================================================================
print(f"\n{'='*70}")
print("PART 2: MONTHLY AGGREGATION")
print(f"{'='*70}")

# Ensure year_month is string for groupby (Period can cause issues after CSV round-trip)
sold['year_month_str'] = sold['CloseDate_dt'].dt.to_period('M').astype(str)

monthly = (
    sold.groupby('year_month_str')
    .agg(
        sales_volume      = ('ClosePrice',          'count'),
        median_close_price= ('ClosePrice',          'median'),
        mean_close_price  = ('ClosePrice',          'mean'),
        median_list_price = ('ListPrice',           'median'),
        median_dom        = ('DaysOnMarket',        'median'),
        mean_dom          = ('DaysOnMarket',        'mean'),
        median_lts_ratio  = ('list_to_sale_ratio',  'median'),
        pct_above_list    = ('sold_above_list',     'mean'),
        median_price_sqft = ('price_per_sqft',      'median'),
        avg_rate_30yr     = ('rate_30yr_fixed',     'mean'),
    )
    .reset_index()
    .rename(columns={'year_month_str': 'year_month'})
    .sort_values('year_month')
)

monthly['pct_above_list'] = (monthly['pct_above_list'] * 100).round(2)

# Month-over-Month changes
monthly['mom_price_change_pct'] = monthly['median_close_price'].pct_change(1) * 100
monthly['mom_volume_change_pct'] = monthly['sales_volume'].pct_change(1) * 100

# Year-over-Year changes
monthly['yoy_price_change_pct'] = monthly['median_close_price'].pct_change(12) * 100
monthly['yoy_volume_change_pct'] = monthly['sales_volume'].pct_change(12) * 100

print(f"\n  Monthly summary ({len(monthly)} months):")
cols = ['year_month','sales_volume','median_close_price',
        'median_dom','avg_rate_30yr','mom_price_change_pct','yoy_price_change_pct']
print(monthly[cols].to_string(index=False))

# =============================================================================
# Part 3: Key Trend Insights
# =============================================================================
print(f"\n{'='*70}")
print("PART 3: KEY TREND INSIGHTS")
print(f"{'='*70}")

# Peak and trough months
peak_price_row  = monthly.loc[monthly['median_close_price'].idxmax()]
trough_price_row= monthly.loc[monthly['median_close_price'].idxmin()]
peak_vol_row    = monthly.loc[monthly['sales_volume'].idxmax()]
trough_vol_row  = monthly.loc[monthly['sales_volume'].idxmin()]

print(f"\n  Median Close Price:")
print(f"    Highest : {peak_price_row['year_month']}  "
      f"${peak_price_row['median_close_price']:,.0f}")
print(f"    Lowest  : {trough_price_row['year_month']}  "
      f"${trough_price_row['median_close_price']:,.0f}")

print(f"\n  Sales Volume:")
print(f"    Highest : {peak_vol_row['year_month']}  "
      f"{int(peak_vol_row['sales_volume']):,} sales")
print(f"    Lowest  : {trough_vol_row['year_month']}  "
      f"{int(trough_vol_row['sales_volume']):,} sales")

# Rate correlation with price and volume
corr_rate_price  = monthly['avg_rate_30yr'].corr(monthly['median_close_price'])
corr_rate_volume = monthly['avg_rate_30yr'].corr(monthly['sales_volume'])
print(f"\n  Mortgage Rate Correlations:")
print(f"    Rate vs Median Price  : r = {corr_rate_price:.3f}")
print(f"    Rate vs Sales Volume  : r = {corr_rate_volume:.3f}")

# Most recent 3 months YoY
print(f"\n  Most Recent Months (YoY price change):")
for _, row in monthly.tail(6).iterrows():
    yoy = row['yoy_price_change_pct']
    yoy_str = f"{yoy:+.1f}%" if not pd.isna(yoy) else "N/A"
    print(f"    {row['year_month']}  price=${row['median_close_price']:,.0f}  "
          f"vol={int(row['sales_volume']):,}  YoY={yoy_str}")

# =============================================================================
# Part 4: Trend Plots
# =============================================================================
print(f"\n{'='*70}")
print("PART 4: GENERATING TREND PLOTS")
print(f"{'='*70}")

x_labels = monthly['year_month'].tolist()
x_idx    = range(len(x_labels))

# Tick positions: show every 3rd month label
tick_every = 3
tick_pos   = [i for i in x_idx if i % tick_every == 0]
tick_lbls  = [x_labels[i] for i in tick_pos]

# --- Plot 1: Median Close Price Over Time ---
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(list(x_idx), monthly['median_close_price'], color='steelblue',
        linewidth=2, marker='o', markersize=4, label='Median Close Price')
ax.fill_between(list(x_idx), monthly['median_close_price'],
                alpha=0.15, color='steelblue')
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lbls, rotation=45, ha='right')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.2f}M'))
ax.set_title('Sold – Median Close Price Over Time', fontsize=14)
ax.set_ylabel('Median Close Price')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
path = os.path.join(PLOT_DIR, 'trend_median_close_price.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"  Saved: {path}")

# --- Plot 2: Monthly Sales Volume ---
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(list(x_idx), monthly['sales_volume'], color='steelblue', alpha=0.7)
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lbls, rotation=45, ha='right')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.set_title('Sold – Monthly Sales Volume', fontsize=14)
ax.set_ylabel('Number of Sales')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
path = os.path.join(PLOT_DIR, 'trend_sales_volume.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"  Saved: {path}")

# --- Plot 3: Price vs Mortgage Rate (dual axis) ---
fig, ax1 = plt.subplots(figsize=(14, 5))
ax2 = ax1.twinx()

ax1.plot(list(x_idx), monthly['median_close_price'], color='steelblue',
         linewidth=2, marker='o', markersize=4, label='Median Close Price')
ax2.plot(list(x_idx), monthly['avg_rate_30yr'], color='tomato',
         linewidth=2, linestyle='--', marker='s', markersize=4,
         label='30yr Mortgage Rate')

ax1.set_xticks(tick_pos)
ax1.set_xticklabels(tick_lbls, rotation=45, ha='right')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.2f}M'))
ax2.set_ylabel('30yr Fixed Rate (%)', color='tomato')
ax2.tick_params(axis='y', labelcolor='tomato')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

ax1.set_title('Sold – Median Price vs Mortgage Rate', fontsize=14)
ax1.set_ylabel('Median Close Price')
ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
path = os.path.join(PLOT_DIR, 'trend_price_vs_rate.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"  Saved: {path}")

# --- Plot 4: Median Days on Market ---
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(list(x_idx), monthly['median_dom'], color='darkorange',
        linewidth=2, marker='o', markersize=4, label='Median DOM')
ax.fill_between(list(x_idx), monthly['median_dom'], alpha=0.15, color='darkorange')
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lbls, rotation=45, ha='right')
ax.set_title('Sold – Median Days on Market Over Time', fontsize=14)
ax.set_ylabel('Median Days on Market')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
path = os.path.join(PLOT_DIR, 'trend_median_dom.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"  Saved: {path}")

# --- Plot 5: % Sold Above List Price ---
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(list(x_idx), monthly['pct_above_list'], color='seagreen',
        linewidth=2, marker='o', markersize=4)
ax.fill_between(list(x_idx), monthly['pct_above_list'], alpha=0.15, color='seagreen')
ax.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% line')
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lbls, rotation=45, ha='right')
ax.set_title('Sold – % Transactions Above List Price (Seller vs Buyer Market)', fontsize=14)
ax.set_ylabel('% Above List Price')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
path = os.path.join(PLOT_DIR, 'trend_pct_above_list.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"  Saved: {path}")

# =============================================================================
# Part 5: County-Level Trend (Top 5 Counties)
# =============================================================================
print(f"\n{'='*70}")
print("PART 5: TOP 5 COUNTIES — MEDIAN PRICE TREND")
print(f"{'='*70}")

top_counties = (
    sold.groupby('CountyOrParish')['ClosePrice']
    .count()
    .sort_values(ascending=False)
    .head(5)
    .index.tolist()
)
print(f"\n  Top 5 counties by volume: {top_counties}")

county_monthly = (
    sold[sold['CountyOrParish'].isin(top_counties)]
    .groupby(['year_month_str', 'CountyOrParish'])['ClosePrice']
    .median()
    .reset_index()
    .rename(columns={'year_month_str': 'year_month', 'ClosePrice': 'median_close_price'})
    .sort_values('year_month')
)

fig, ax = plt.subplots(figsize=(14, 6))
colors = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'purple']
for county, color in zip(top_counties, colors):
    sub = county_monthly[county_monthly['CountyOrParish'] == county]
    sub_x = [x_labels.index(ym) for ym in sub['year_month'] if ym in x_labels]
    sub_y = sub[sub['year_month'].isin(x_labels)]['median_close_price'].tolist()
    ax.plot(sub_x, sub_y, linewidth=2, marker='o', markersize=3,
            label=county, color=color)

ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lbls, rotation=45, ha='right')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.2f}M'))
ax.set_title('Sold – Median Close Price by County (Top 5)', fontsize=14)
ax.set_ylabel('Median Close Price')
ax.legend(loc='upper left')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
path = os.path.join(PLOT_DIR, 'trend_county_price.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"  Saved: {path}")

# =============================================================================
# Save outputs
# =============================================================================
print(f"\n{'='*70}")
print("SAVING OUTPUT")
print(f"{'='*70}")

# Drop helper columns before saving transaction-level file
sold = sold.drop(columns=['CloseDate_dt'])

sold_path = os.path.join(OUTPUT_DIR, "week4_sold.csv")
sold.to_csv(sold_path, index=False)
print(f"  Transaction-level saved: {sold_path}")
print(f"  Shape: {sold.shape[0]:,} rows x {sold.shape[1]} columns")

monthly_path = os.path.join(OUTPUT_DIR, "week4_sold_monthly.csv")
monthly.to_csv(monthly_path, index=False)
print(f"  Monthly summary saved:   {monthly_path}")
print(f"  Shape: {monthly.shape[0]} rows x {monthly.shape[1]} columns")

print(f"\nWeek 4 Sold – Complete!")
