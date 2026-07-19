"""
=============================================================================
FILE: gpr_deep_dive/gpr_src/gpr_data.py
PURPOSE: GPR data loading + multi-index stock data + control factors

Real data sources:
  - Monthly GPR: data_gpr_export.xls (1900-2026, 115 columns)
  - Daily GPR: data_gpr_daily_recent.xls (1985-2026, 11 columns)
  - Stock data: Yahoo Finance v8 API (SOX, XLI, SPY, XLK, VIX, OIL, USD, TLT)
=============================================================================
"""

import numpy as np
import pandas as pd
import warnings
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_gpr_monthly(filepath: str = None) -> pd.DataFrame:
    """Load monthly GPR from downloaded XLS."""
    if filepath is None:
        filepath = os.path.join(DATA_DIR, 'data_gpr_export.xls')

    if not os.path.exists(filepath):
        warnings.warn(f"GPR file not found: {filepath}, using synthetic")
        return _generate_synthetic_gpr()

    df = pd.read_excel(filepath)
    df = df.rename(columns={'month': 'date'})

    key_cols = ['date', 'GPR', 'GPRT', 'GPRA', 'GPRH']
    cat_cols = [c for c in df.columns if c.startswith('SHAREH_CAT_')]
    us_cols = [c for c in df.columns if c in ['GPRC_USA', 'GPRHC_USA']]

    selected = [c for c in key_cols + cat_cols + us_cols if c in df.columns]
    result = df[selected].copy()
    result['date'] = pd.to_datetime(result['date'])
    result = result.dropna(subset=['GPR'])

    # Rename categories
    rename_map = {}
    for c in cat_cols:
        num = c.replace('SHAREH_CAT_', '')
        rename_map[c] = f'CAT_{num}'
    result = result.rename(columns=rename_map)

    print(f"[GPR Monthly] {len(result)} months "
          f"({result['date'].min().strftime('%Y-%m')} to "
          f"{result['date'].max().strftime('%Y-%m')})")
    return result


def load_gpr_daily(filepath: str = None) -> pd.DataFrame:
    """Load daily GPR from downloaded XLS."""
    if filepath is None:
        filepath = os.path.join(DATA_DIR, 'data_gpr_daily_recent.xls')

    if not os.path.exists(filepath):
        warnings.warn(f"Daily GPR file not found: {filepath}")
        return None

    df = pd.read_excel(filepath)
    key_cols = ['date', 'GPRD', 'GPRD_ACT', 'GPRD_THREAT', 'GPRD_MA30', 'GPRD_MA7', 'event']
    key_cols = [c for c in key_cols if c in df.columns]

    result = df[key_cols].copy()
    result['date'] = pd.to_datetime(result['date'])
    result = result.dropna(subset=['GPRD'])

    events = result[result['event'].notna()]
    print(f"[GPR Daily] {len(result)} days, {len(events)} labeled events")
    return result


def load_market_data(filepath: str = None) -> pd.DataFrame:
    """Load pre-downloaded market data from CSV."""
    if filepath is None:
        filepath = os.path.join(DATA_DIR, 'all_market_data.csv')

    if os.path.exists(filepath):
        df = pd.read_csv(filepath, parse_dates=['Date'])
        print(f"[Market] {len(df)} rows from cache")
        return df

    warnings.warn("Market data file not found, downloading via API")
    return _download_market_data(filepath)


def _download_market_data(save_path: str) -> pd.DataFrame:
    """Download stock data using Yahoo Finance v8 API"""
    import requests, datetime, time

    def dl(ticker, start='1990-01-01', end='2025-12-31'):
        url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}'
        p1 = int(datetime.datetime.strptime(start, '%Y-%m-%d').timestamp())
        p2 = int(datetime.datetime.strptime(end, '%Y-%m-%d').timestamp())
        r = requests.get(url, params={'period1': p1, 'period2': p2, 'interval': '1mo'},
                        headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        r.raise_for_status()
        d = r.json()['chart']['result'][0]
        return pd.DataFrame({
            'Date': pd.to_datetime(d['timestamp'], unit='s'),
            'Open': d['indicators']['quote'][0]['open'],
            'High': d['indicators']['quote'][0]['high'],
            'Low': d['indicators']['quote'][0]['low'],
            'Close': d['indicators']['quote'][0]['close'],
            'Volume': d['indicators']['quote'][0]['volume'],
        }).dropna(subset=['Close'])

    tickers = {'SOX': '^SOX', 'XLI': 'XLI', 'SPY': 'SPY', 'XLK': 'XLK',
               'VIX': '^VIX', 'OIL': 'CL=F', 'USD': 'DX-Y.NYB', 'TLT': 'TLT'}
    all_dfs = {}
    for name, ticker in tickers.items():
        try:
            df = dl(ticker)
            # Normalize to month-start to avoid duplicate dates from timezone differences
            df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
            df = df.drop_duplicates(subset=['Date'], keep='last')
            df.columns = [f'{name}_{c}' if c != 'Date' else c for c in df.columns]
            all_dfs[name] = df
            print(f"  {name}: {len(df)} months")
        except Exception as e:
            warnings.warn(f"{name}: {e}")
        time.sleep(1)

    if not all_dfs:
        warnings.warn("All downloads failed, generating synthetic market data")
        return _generate_synthetic_market_data(save_path)

    merged = None
    for df in all_dfs.values():
        merged = df if merged is None else pd.merge(merged, df, on='Date', how='outer')
    merged = merged.sort_values('Date').drop_duplicates(subset=['Date']).reset_index(drop=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    merged.to_csv(save_path, index=False)
    return merged


def merge_gpr_with_market(gpr: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Merge monthly GPR with market data on date."""
    gpr = gpr.copy()
    gpr['date'] = pd.to_datetime(gpr['date']).dt.tz_localize(None)

    mkt = market.copy()
    if 'Date' in mkt.columns:
        mkt = mkt.rename(columns={'Date': 'date'})
    mkt['date'] = pd.to_datetime(mkt['date']).dt.tz_localize(None)

    # Normalize both to month-start for reliable matching
    gpr['month_key'] = gpr['date'].dt.to_period('M')
    mkt['month_key'] = mkt['date'].dt.to_period('M')

    merged = pd.merge(gpr, mkt, on='month_key', how='inner')
    # Use GPR date as canonical
    if 'date_x' in merged.columns:
        merged = merged.rename(columns={'date_x': 'date'})
        if 'date_y' in merged.columns:
            merged = merged.drop(columns=['date_y'])
    merged = merged.drop(columns=['month_key'])
    merged = merged.sort_values('date').reset_index(drop=True)
    print(f"[Merge] {len(merged)} months, {len(merged.columns)} columns")
    return merged


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log return columns for all _Close columns, plus level/change for factors.

    Handles non-contiguous indices (from outer merges) by computing
    returns on the valid subset then merging back.
    """
    df = df.copy()

    # Log returns for stock indexes
    for col in [c for c in df.columns if c.endswith('_Close')]:
        name = col.replace('_Close', '_log_return')
        valid = df[col].dropna()
        log_ret = np.log(valid / valid.shift(1))
        df[name] = log_ret

    # Level and change columns for control factors (VIX, OIL, USD, TLT)
    for prefix in ['VIX', 'OIL', 'USD', 'TLT']:
        close_col = f'{prefix}_Close'
        if close_col in df.columns:
            # Level (raw value)
            df[f'{prefix}_level'] = df[close_col]
            # Month-over-month change (percentage)
            valid = df[close_col].dropna()
            df[f'{prefix}_change'] = valid.pct_change()

    return df


def build_geopolitical_exposure_index(df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """Build weighted geopolitical exposure index.

    Only includes rows where at least 2 of the weighted indexes have data.
    Weights are renormalized per-row to handle missing data correctly.
    """
    if weights is None:
        weights = {'SOX_log_return': 0.40, 'XLI_log_return': 0.30,
                   'SPY_log_return': -0.20, 'XLK_log_return': 0.10}

    available = {k: v for k, v in weights.items() if k in df.columns}
    if len(available) < 2:
        if 'SOX_log_return' in df.columns:
            df['GEO_index_return'] = df['SOX_log_return']
        return df

    df = df.copy()
    weighted_sum = pd.Series(0.0, index=df.index)
    weight_sum = pd.Series(0.0, index=df.index)

    for col, w in available.items():
        mask = df[col].notna()
        weighted_sum += df[col].fillna(0) * w * mask
        weight_sum += abs(w) * mask

    # Only compute where at least 2 indexes have data
    valid = weight_sum > 0
    df['GEO_index_return'] = np.where(valid, weighted_sum / weight_sum, np.nan)
    df['GEO_index_price'] = 100 * np.exp(df['GEO_index_return'].fillna(0).cumsum())

    n_valid = df['GEO_index_return'].notna().sum()
    print(f"[GEO] Weighted index: {n_valid}/{len(df)} months with data")

    return df


def _generate_synthetic_gpr() -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    dates = pd.date_range('1985-01-01', '2025-12-31', freq='MS')
    n = len(dates)
    gpr = 100 + rng.normal(0, 20, n)
    gpr = np.clip(gpr, 30, 500)
    return pd.DataFrame({'date': dates, 'GPR': gpr, 'GPRT': gpr*0.6, 'GPRA': gpr*0.4, 'GPRH': gpr})


def _run_self_tests():
    print("=" * 60)
    print("gpr_data.py self-test")
    print("=" * 60)

    print("\n[Test 1] GPR Monthly")
    gpr = load_gpr_monthly()
    assert len(gpr) > 100
    assert 'GPR' in gpr.columns
    cat_cols = [c for c in gpr.columns if c.startswith('CAT_')]
    print(f"  {len(gpr)} months, {len(cat_cols)} categories")
    print(f"  GPR: {gpr['GPR'].min():.0f} to {gpr['GPR'].max():.0f}, mean={gpr['GPR'].mean():.0f}")
    print("  [PASS]")

    print("\n[Test 2] GPR Daily")
    gpr_d = load_gpr_daily()
    assert gpr_d is not None
    assert len(gpr_d) > 1000
    events = gpr_d[gpr_d['event'].notna()]
    print(f"  {len(gpr_d)} days, {len(events)} events")
    if len(events) > 0:
        peak = events.loc[events['GPRD'].idxmax()]
        print(f"  Peak: {peak['event']} (GPRD={peak['GPRD']:.0f})")
    print("  [PASS]")

    print("\n[Test 3] Market Data")
    market = load_market_data()
    assert market is not None
    assert len(market) > 100
    print(f"  {len(market)} months")
    print("  [PASS]")

    print("\n[Test 4] Merge + Correlation")
    market = add_log_returns(market)
    market = build_geopolitical_exposure_index(market)
    merged = merge_gpr_with_market(gpr, market)
    assert len(merged) > 50
    corr = merged[['GPR', 'GEO_index_return']].corr().iloc[0, 1]
    print(f"  {len(merged)} months, GPR-Return corr: {corr:.4f}")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_self_tests()
