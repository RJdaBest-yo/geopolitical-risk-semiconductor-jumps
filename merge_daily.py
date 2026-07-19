"""
Merge daily GPR data with daily stock/market data.
Document all cleaning steps.
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DATA = 'e:/Eddik/Documents/AI/jump-diffusion-dual-domain/gpr_deep_dive/data'

# ============================================================
# 1. LOAD DAILY GPR
# ============================================================
print("=" * 70)
print("STEP 1: Load Daily GPR")
print("=" * 70)

gpr = pd.read_excel(f'{DATA}/data_gpr_daily_recent.xls')
gpr = gpr[['date', 'GPRD', 'GPRD_ACT', 'GPRD_THREAT', 'GPRD_MA30', 'GPRD_MA7', 'event']].copy()
gpr['date'] = pd.to_datetime(gpr['date'])
gpr = gpr.dropna(subset=['GPRD'])
gpr = gpr.drop_duplicates(subset=['date'])
print(f"  GPR Daily: {len(gpr)} days, {gpr['date'].min()} to {gpr['date'].max()}")
print(f"  GPRD range: {gpr['GPRD'].min():.0f} to {gpr['GPRD'].max():.0f}")

# ============================================================
# 2. LOAD DAILY STOCK DATA
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Load Daily Stock Data (REAL)")
print("=" * 70)

market = pd.read_csv(f'{DATA}/all_market_daily.csv')
market['Date'] = pd.to_datetime(market['Date']).dt.tz_localize(None)
market['date'] = market['Date'].dt.normalize()

# De-duplicate: some dates have multiple rows (from outer merge of tickers)
# For each date, take the last non-null value per column
market = market.groupby('date').last().reset_index()
print(f"  Market: {len(market)} unique dates")

# Quality check
for col in ['SOX_Close', 'SPY_Close', 'VIX_Close', 'OIL_Close', 'USD_Close']:
    if col in market.columns:
        n = market[col].notna().sum()
        first_valid_idx = market[col].first_valid_index()
        first_valid = market.loc[first_valid_idx, 'date'].strftime('%Y-%m-%d') if first_valid_idx is not None else 'N/A'
        print(f"    {col}: {n} non-null, starts {first_valid}")

# ============================================================
# 3. MERGE
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Merge GPR + Market (Daily)")
print("=" * 70)

gpr['date'] = gpr['date'].dt.normalize()
merged = pd.merge(gpr, market, on='date', how='inner')
merged = merged.sort_values('date').reset_index(drop=True)
print(f"  Merged: {len(merged)} days")

# ============================================================
# 4. DATA CLEANING
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Data Cleaning")
print("=" * 70)

# 4a. Log returns
for prefix in ['SOX', 'SPY', 'XLI', 'XLK', 'VIX', 'OIL', 'USD', 'TLT']:
    close_col = f'{prefix}_Close'
    if close_col in merged.columns:
        valid = merged[close_col].dropna()
        merged[f'{prefix}_log_return'] = np.log(valid / valid.shift(1))

# 4b. Remove outliers
print("  Outlier removal (|return| > 20%)...")
for col in [c for c in merged.columns if c.endswith('_log_return')]:
    outliers = merged[col].abs() > 0.20
    if outliers.sum() > 0:
        print(f"    {col}: {outliers.sum()} outliers")
        merged.loc[outliers, col] = np.nan

# 4c. GPR features
merged['GPRD_sma30'] = merged['GPRD'].rolling(30, min_periods=10).mean()
merged['GPRD_momentum'] = merged['GPRD'] - merged['GPRD_sma30']
merged['GPRD_rising'] = (merged['GPRD_momentum'] > 0).astype(int)
merged['GPRD_change_5d'] = merged['GPRD'].diff(5)
merged['GPRD_change_20d'] = merged['GPRD'].diff(20)

# 4d. VIX features
if 'VIX_Close' in merged.columns:
    merged['VIX_sma20'] = merged['VIX_Close'].rolling(20, min_periods=10).mean()
    merged['VIX_momentum'] = merged['VIX_Close'] - merged['VIX_sma20']

# 4e. Oil features
if 'OIL_Close' in merged.columns:
    merged['OIL_sma20'] = merged['OIL_Close'].rolling(20, min_periods=10).mean()
    merged['OIL_momentum'] = merged['OIL_Close'] - merged['OIL_sma20']

# ============================================================
# 5. ANALYSIS DATASET (filter to SOX trading days first)
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Analysis Dataset Summary")
print("=" * 70)

analysis = merged[merged['date'] >= '1994-06-01'].copy()
analysis = analysis.dropna(subset=['SOX_log_return'])
analysis = analysis.reset_index(drop=True)

# Compute forward returns ON THE CLEAN DATASET (no NaN gaps)
analysis['fwd_return_5d'] = analysis['SOX_log_return'].rolling(5).sum().shift(-5)
analysis['fwd_return_20d'] = analysis['SOX_log_return'].rolling(20).sum().shift(-20)
analysis['is_jump_5d'] = (analysis['fwd_return_5d'] < -0.05).astype(int)
analysis['is_jump_20d'] = (analysis['fwd_return_20d'] < -0.10).astype(int)

print(f"  Days: {len(analysis)}")
print(f"  Range: {analysis['date'].min().strftime('%Y-%m-%d')} to {analysis['date'].max().strftime('%Y-%m-%d')}")
print(f"  GPRD non-null: {analysis['GPRD'].notna().sum()}")
print(f"  SOX return: {analysis['SOX_log_return'].notna().sum()}")
print(f"  VIX: {analysis['VIX_Close'].notna().sum() if 'VIX_Close' in analysis.columns else 0}")
print(f"  OIL: {analysis['OIL_Close'].notna().sum() if 'OIL_Close' in analysis.columns else 0}")
print(f"  5d forward return: {analysis['fwd_return_5d'].notna().sum()}")
print(f"  20d forward return: {analysis['fwd_return_20d'].notna().sum()}")
print(f"  5d jumps (<-5%): {analysis['is_jump_5d'].sum()} ({analysis['is_jump_5d'].mean()*100:.2f}%)")
print(f"  20d jumps (<-10%): {analysis['is_jump_20d'].sum()} ({analysis['is_jump_20d'].mean()*100:.2f}%)")

# Save
analysis.to_csv(f'{DATA}/analysis_daily_clean.csv', index=False)
print(f"\n  Saved: analysis_daily_clean.csv")

# ============================================================
# 6. CORRELATIONS
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: GPR-Return Correlations (REAL Daily Data)")
print("=" * 70)

factors = [
    ('GPRD', 'GPR Level'),
    ('GPRD_momentum', 'GPR Momentum (30d)'),
    ('GPRD_change_5d', 'GPR Change 5d'),
    ('GPRD_change_20d', 'GPR Change 20d'),
    ('GPRD_ACT', 'GPR Acts'),
    ('GPRD_THREAT', 'GPR Threats'),
]
if 'VIX_Close' in analysis.columns:
    factors.extend([('VIX_Close', 'VIX Level'), ('VIX_momentum', 'VIX Momentum')])
if 'OIL_Close' in analysis.columns:
    factors.extend([('OIL_Close', 'Oil Level'), ('OIL_momentum', 'Oil Momentum')])

print(f"\n  {'Factor':<25} {'vs 5d Ret':>10} {'p':>10} {'vs 20d Ret':>12} {'p':>10}")
print(f"  {'-'*70}")

for col, label in factors:
    if col not in analysis.columns:
        continue
    sub5 = analysis.dropna(subset=[col, 'fwd_return_5d'])
    sub20 = analysis.dropna(subset=[col, 'fwd_return_20d'])

    r5, p5 = stats.pearsonr(sub5[col], sub5['fwd_return_5d']) if len(sub5) > 100 else (0, 1)
    r20, p20 = stats.pearsonr(sub20[col], sub20['fwd_return_20d']) if len(sub20) > 100 else (0, 1)

    sig5 = '***' if p5 < 0.01 else '**' if p5 < 0.05 else '*' if p5 < 0.10 else ''
    sig20 = '***' if p20 < 0.01 else '**' if p20 < 0.05 else '*' if p20 < 0.10 else ''

    print(f"  {label:<25} {r5:>9.4f}{sig5} {p5:>9.4f} {r20:>10.4f}{sig20} {p20:>9.4f}")

# ============================================================
# 7. DAILY REGIME ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Daily Regime Jump Probability")
print("=" * 70)

reg_data = analysis.dropna(subset=['GPRD_momentum', 'is_jump_20d'])
gpr_median = reg_data['GPRD'].median()

def get_regime(row):
    h = row['GPRD'] > gpr_median
    r = row['GPRD_momentum'] > 0
    if h and r: return 'high_rising'
    if h and not r: return 'high_falling'
    if not h and r: return 'low_rising'
    return 'low_stable'

reg_data = reg_data.copy()
reg_data['regime'] = reg_data.apply(get_regime, axis=1)

print(f"\n  GPR median: {gpr_median:.0f}")
print(f"  Target: 20-day forward return < -10% (jump)")
print(f"\n  {'Regime':<15} {'Days':>8} {'Jumps':>8} {'Jump Prob':>10} {'Mean 20d Ret':>13}")
print(f"  {'-'*58}")

for regime in ['low_stable', 'low_rising', 'high_rising', 'high_falling']:
    sub = reg_data[reg_data['regime'] == regime]
    if len(sub) > 0:
        jp = sub['is_jump_20d'].mean()
        mr = sub['fwd_return_20d'].mean()
        print(f"  {regime:<15} {len(sub):>8} {sub['is_jump_20d'].sum():>8} {jp:>9.2%} {mr:>12.4%}")
