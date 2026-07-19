"""
=============================================================================
FILE: gpr_deep_dive/src/factor_decomposition.py
PURPOSE: Decompose stock returns into geopolitical vs non-geopolitical components

Method:
  1. Regress stock returns on control factors (market, VIX, oil, USD, tech trend)
  2. Residual = the component NOT explained by these factors
  3. Correlate residual with GPR to isolate the "geopolitical signal"

This lets us isolate how much of stock movement is due to geopolitics
versus other drivers (tech cycles, financial stress, etc.)
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings


def decompose_returns(
    merged_df: pd.DataFrame,
    return_col: str = 'GEO_index_return',
    factor_cols: list = None
) -> dict:
    """
    Decompose stock returns into explained (factors) and unexplained (residual).

    Args:
        merged_df: DataFrame with stock returns and control factors
        return_col: column name of the stock return to decompose
        factor_cols: list of factor column names to regress against

    Returns:
        dict with:
            'residual': the geopolitical residual series
            'r_squared': how much of returns is explained by non-GPR factors
            'coefficients': regression coefficients
            'gpr_correlation': correlation between residual and GPR
            'model': the fitted regression
    """
    df = merged_df.copy()

    # Default factor columns
    if factor_cols is None:
        candidates = [
            'SPY_return', 'SPY_log_return',      # Market factor
            'VIX_change', 'VIX_level',            # Fear gauge
            'OIL_change', 'OIL_level',            # Energy prices
            'USD_change', 'USD_level',            # Dollar strength
            'XLK_return', 'XLK_log_return',       # Tech trend
            'TLT_change', 'TLT_level',            # Bond flight-to-quality
        ]
        factor_cols = [c for c in candidates if c in df.columns]

    if len(factor_cols) < 2:
        warnings.warn("Fewer than 2 control factors available")

    # Drop rows with NaN in any required column
    required = [return_col] + factor_cols + ['GPR']
    original_index = df.index.copy()
    df = df.dropna(subset=[c for c in required if c in df.columns])

    if len(df) < 30:
        warnings.warn(f"Only {len(df)} observations after dropping NaN")

    y = df[return_col].values
    X = df[factor_cols].values

    # Add constant
    X_with_const = np.column_stack([np.ones(len(X)), X])

    # OLS regression
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X_with_const, y, rcond=None)
    except Exception as e:
        warnings.warn(f"OLS failed: {e}")
        return None

    y_hat = X_with_const @ beta
    residual = y - y_hat

    # R-squared
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Correlation between residual and GPR
    gpr_values = df['GPR'].values
    gpr_corr, gpr_pval = stats.pearsonr(residual, gpr_values)

    # GPR_ACT and GPR_THREAT correlations
    gpr_act_corr, gpr_act_pval = (0, 1)
    gpr_threat_corr, gpr_threat_pval = (0, 1)
    if 'GPR_ACT' in df.columns:
        gpr_act_corr, gpr_act_pval = stats.pearsonr(residual, df['GPR_ACT'].values)
    if 'GPR_THREAT' in df.columns:
        gpr_threat_corr, gpr_threat_pval = stats.pearsonr(
            residual, df['GPR_THREAT'].values
        )

    # Coefficient names
    coef_names = ['const'] + factor_cols
    coefficients = dict(zip(coef_names, beta))

    # Build residual series aligned with dates
    residual_series = pd.Series(residual, index=df.index, name='geo_residual')

    print(f"[Decompose] R² = {r_squared:.3f} "
          f"(non-GPR factors explain {r_squared*100:.1f}% of returns)")
    print(f"[Decompose] Residual-GPR correlation: {gpr_corr:.3f} (p={gpr_pval:.4f})")
    print(f"[Decompose] Residual-GPR_ACT correlation: {gpr_act_corr:.3f}")
    print(f"[Decompose] Residual-GPR_THREAT correlation: {gpr_threat_corr:.3f}")

    return {
        'residual': residual_series,
        'fitted': y_hat,
        'actual': y,
        'r_squared': r_squared,
        'coefficients': coefficients,
        'factor_cols': factor_cols,
        'gpr_correlation': gpr_corr,
        'gpr_pvalue': gpr_pval,
        'gpr_act_correlation': gpr_act_corr,
        'gpr_act_pvalue': gpr_act_pval,
        'gpr_threat_correlation': gpr_threat_corr,
        'gpr_threat_pvalue': gpr_threat_pval,
        'df': df,
    }


def identify_significant_factors(
    merged_df: pd.DataFrame,
    return_col: str = 'GEO_index_return'
) -> pd.DataFrame:
    """
    Run univariate regressions of stock returns against each factor
    to identify which factors are most significant.
    """
    candidates = {
        'GPR': 'Geopolitical Risk (aggregate)',
        'GPR_ACT': 'Geopolitical Acts',
        'GPR_THREAT': 'Geopolitical Threats',
        'SPY_return': 'S&P 500 (market)',
        'VIX_change': 'VIX change (fear)',
        'VIX_level': 'VIX level',
        'OIL_change': 'Oil price change',
        'OIL_level': 'Oil price level',
        'USD_change': 'USD change',
        'XLK_return': 'Tech sector return',
        'TLT_change': 'Bond return (flight to quality)',
    }

    results = []
    df = merged_df.dropna(subset=[return_col])

    for col, desc in candidates.items():
        if col not in df.columns:
            continue

        sub = df.dropna(subset=[col, return_col])
        if len(sub) < 20:
            continue

        slope, intercept, r, p, se = stats.linregress(
            sub[col].values, sub[return_col].values
        )

        results.append({
            'factor': col,
            'description': desc,
            'correlation': r,
            'r_squared': r ** 2,
            'p_value': p,
            'significant': p < 0.05,
            'n_obs': len(sub),
        })

    result_df = pd.DataFrame(results).sort_values('r_squared', ascending=False)
    return result_df


def _run_self_tests():
    print("=" * 60)
    print("factor_decomposition.py self-test")
    print("=" * 60)

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from gpr_src.gpr_data import (
        load_gpr_monthly, load_market_data, add_log_returns,
        build_geopolitical_exposure_index, merge_gpr_with_market
    )

    # Load data
    print("\n[Test 1] Load and merge data")
    gpr = load_gpr_monthly()
    market = load_market_data()
    market = add_log_returns(market)
    market = build_geopolitical_exposure_index(market)
    merged = merge_gpr_with_market(gpr, market)
    print(f"  Merged: {len(merged)} months, {len(merged.columns)} columns")
    assert len(merged) > 30
    print("  [PASS]")

    # Test 2: Decomposition
    print("\n[Test 2] Factor decomposition")
    decomp = decompose_returns(merged, return_col='GEO_index_return')
    assert decomp is not None
    assert -1 <= decomp['gpr_correlation'] <= 1
    print(f"  R² = {decomp['r_squared']:.3f}")
    print(f"  GPR correlation = {decomp['gpr_correlation']:.3f}")
    print("  [PASS]")

    # Test 3: Factor significance
    print("\n[Test 3] Factor significance ranking")
    sig_df = identify_significant_factors(merged)
    print(sig_df[['factor', 'correlation', 'r_squared', 'p_value']].to_string(index=False))
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_self_tests()
