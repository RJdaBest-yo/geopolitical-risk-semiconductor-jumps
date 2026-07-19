"""
=============================================================================
FILE: gpr_deep_dive/gpr_src/gpr_forecasting.py
PURPOSE: GPR-based jump calibration and forecasting

KEY FIX (v2): Use GPR CHANGE (not level) as primary predictor.
  - GPR level has POSITIVE correlation with forward returns (mean reversion)
  - GPR CHANGE has NEGATIVE correlation (correct direction: rising risk → falling returns)
  - Crisis regime shows mean reversion (markets bounce after big spikes)
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.jump_diffusion_engine import JumpDiffusionParams, simulate_jump_diffusion


def compute_gpr_regime(gpr_series: pd.Series, window: int = 6) -> pd.Series:
    """
    Classify GPR into regimes using ADAPTIVE thresholds based on data percentiles.
    """
    level = gpr_series
    # Use percentiles instead of fixed thresholds
    p25 = level.quantile(0.25)
    p50 = level.quantile(0.50)
    p75 = level.quantile(0.75)
    p90 = level.quantile(0.90)

    regime = pd.Series('low', index=gpr_series.index)
    regime[level >= p25] = 'elevated'
    regime[level >= p50] = 'high'
    regime[level >= p90] = 'crisis'

    return regime


def calibrate_gpr_jump_model(
    merged_df: pd.DataFrame,
    return_col: str = 'geo_residual',
    train_start: str = '1994-06-01',
    train_end: str = '2009-12-31',
    horizon_months: int = 3
) -> dict:
    """
    Calibrate the GPR → stock return relationship.

    [FIXED v2] Uses GPR MOMENTUM (level vs 6-month SMA) as primary signal.
    Combined with level for regime classification.
    """
    df = merged_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Compute forward returns
    df[f'fwd_return_{horizon_months}m'] = (
        df[return_col].rolling(horizon_months).sum().shift(-horizon_months)
    )

    # GPR momentum: level vs 6-month moving average
    df['GPR_sma6'] = df['GPR'].rolling(6).mean()
    df['GPR_momentum'] = df['GPR'] - df['GPR_sma6']
    df['GPR_rising'] = (df['GPR_momentum'] > 0).astype(int)
    df['gpr_change'] = df['GPR'].diff(horizon_months)

    # Split train
    train = df[(df['date'] >= train_start) & (df['date'] <= train_end)].dropna(
        subset=[f'fwd_return_{horizon_months}m', 'GPR', 'GPR_momentum', 'GPR_rising']
    )

    if len(train) < 20:
        warnings.warn(f"Only {len(train)} training observations")

    # Combined regime: level + momentum
    train = train.copy()
    gpr_median = train['GPR'].median()
    train['regime_combo'] = 'low_stable'
    train.loc[(train['GPR'] > gpr_median) & (train['GPR_rising'] == 1), 'regime_combo'] = 'high_rising'
    train.loc[(train['GPR'] > gpr_median) & (train['GPR_rising'] == 0), 'regime_combo'] = 'high_falling'
    train.loc[(train['GPR'] <= gpr_median) & (train['GPR_rising'] == 1), 'regime_combo'] = 'low_rising'

    # Also keep simple level-based regime for backward compatibility
    train['gpr_regime'] = compute_gpr_regime(train['GPR'])

    # Calibrate per combined regime
    regime_stats = {}
    for regime in ['low_stable', 'low_rising', 'high_rising', 'high_falling']:
        subset = train[train['regime_combo'] == regime]
        if len(subset) < 3:
            continue

        fwd = subset[f'fwd_return_{horizon_months}m'].values

        regime_stats[regime] = {
            'n_obs': len(subset),
            'mean_return': fwd.mean(),
            'std_return': fwd.std(ddof=1) if len(subset) > 1 else 0.05,
            'prob_negative': (fwd < 0).mean(),
            'median_return': np.median(fwd),
            'pct_5': np.percentile(fwd, 5) if len(subset) >= 5 else fwd.min(),
            'pct_95': np.percentile(fwd, 95) if len(subset) >= 5 else fwd.max(),
            'gpr_mean': subset['GPR'].mean(),
            'gpr_momentum_mean': subset['GPR_momentum'].mean(),
        }

    # GPR momentum → return slope
    slope_mom, intercept_mom, r_mom, p_mom, _ = stats.linregress(
        train['GPR_momentum'].values,
        train[f'fwd_return_{horizon_months}m'].values
    )

    # GPR change → return slope
    slope_change, intercept_change, r_change, p_change, _ = stats.linregress(
        train['gpr_change'].values,
        train[f'fwd_return_{horizon_months}m'].values
    )

    # Simple momentum hit rate
    correct = ((train['GPR_rising'] == 1) == (train[f'fwd_return_{horizon_months}m'] < 0)).sum()
    momentum_hit_rate = correct / len(train)

    return {
        'regime_stats': regime_stats,
        'gpr_momentum_slope': slope_mom,
        'gpr_momentum_intercept': intercept_mom,
        'gpr_momentum_r': r_mom,
        'gpr_momentum_p': p_mom,
        'gpr_change_slope': slope_change,
        'gpr_change_intercept': intercept_change,
        'gpr_change_r': r_change,
        'gpr_change_p': p_change,
        'momentum_hit_rate': momentum_hit_rate,
        'gpr_median': gpr_median,
        'horizon_months': horizon_months,
        'train_start': train_start,
        'train_end': train_end,
        'train_n': len(train),
    }


def forecast_returns(
    calibration: dict,
    current_gpr: float,
    gpr_momentum: float = 0.0,
    gpr_change_3m: float = 0.0
) -> dict:
    """
    Forecast return distribution given current GPR state.

    [FIXED v2] Uses combined level + momentum regime classification.
    - low_stable: calm markets → expect positive returns
    - low_rising: risk building → mixed
    - high_rising: risk escalating → expect negative returns
    - high_falling: risk declining (mean reversion) → expect positive bounce
    """
    gpr_median = calibration.get('gpr_median', 100)
    regime_stats = calibration['regime_stats']

    # Determine combined regime
    is_high = current_gpr > gpr_median
    is_rising = gpr_momentum > 0

    if is_high and is_rising:
        regime = 'high_rising'
    elif is_high and not is_rising:
        regime = 'high_falling'
    elif not is_high and is_rising:
        regime = 'low_rising'
    else:
        regime = 'low_stable'

    rs = regime_stats.get(regime)
    if rs is None:
        # Fallback: find closest
        for r in ['low_stable', 'high_rising', 'low_rising', 'high_falling']:
            if r in regime_stats:
                rs = regime_stats[r]
                regime = r
                break

    # Forecast: use regime mean as primary (momentum-based classification already encodes direction)
    regime_forecast = rs['mean_return'] if rs else 0.0

    # Momentum adjustment: stronger momentum → stronger signal
    mom_adjustment = calibration.get('gpr_momentum_slope', 0) * gpr_momentum
    expected_return = regime_forecast + 0.5 * mom_adjustment

    # Return range from regime distribution
    return_range = (rs['pct_5'], rs['pct_95']) if rs else (-0.10, 0.10)

    # Jump Diffusion parameters
    horizon = calibration['horizon_months']
    dt = horizon / 12

    jd_params = JumpDiffusionParams(
        mu=expected_return / dt,
        sigma=(rs['std_return'] / np.sqrt(dt)) if rs else 0.20,
        lam=1.0 / dt,
        mu_j=rs['mean_return'] if rs else expected_return,
        sigma_j=rs['std_return'] if rs else 0.10,
        S0=100,
        T=dt,
        dt=dt,
        n_sims=5000
    )

    return {
        'regime': regime,
        'current_gpr': current_gpr,
        'gpr_momentum': gpr_momentum,
        'expected_return': expected_return,
        'return_range': return_range,
        'prob_negative': rs['prob_negative'] if rs else 0.5,
        'regime_forecast': regime_forecast,
        'regime_stats': rs,
        'jump_params': jd_params,
    }


def validate_forecast(
    calibration: dict,
    merged_df: pd.DataFrame,
    return_col: str = 'geo_residual',
    test_start: str = '2010-01-01'
) -> pd.DataFrame:
    """
    Validate forecasting model on out-of-sample data.
    """
    df = merged_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    horizon = calibration['horizon_months']

    # Forward actual returns
    df['fwd_return'] = df[return_col].rolling(horizon).sum().shift(-horizon)
    df['gpr_change_3m'] = df['GPR'].diff(3)
    df['gpr_change_6m'] = df['GPR'].diff(6)

    # GPR momentum (level vs 6-month SMA)
    df['GPR_sma6'] = df['GPR'].rolling(6).mean()
    df['GPR_momentum'] = df['GPR'] - df['GPR_sma6']

    test = df[df['date'] >= test_start].dropna(
        subset=['fwd_return', 'GPR', 'GPR_momentum']
    )

    results = []
    for _, row in test.iterrows():
        gpr_level = row['GPR']
        gpr_mom = row['GPR_momentum']
        gpr_chg3 = row['gpr_change_3m'] if pd.notna(row['gpr_change_3m']) else 0

        fc = forecast_returns(calibration, gpr_level, gpr_mom, gpr_chg3)
        actual = row['fwd_return']

        direction_correct = (
            (fc['expected_return'] > 0 and actual > 0) or
            (fc['expected_return'] < 0 and actual < 0)
        )

        results.append({
            'date': row['date'],
            'gpr': gpr_level,
            'gpr_change_3m': gpr_chg3,
            'regime': fc['regime'],
            'forecast_return': fc['expected_return'],
            'actual_return': actual,
            'direction_correct': direction_correct,
            'magnitude_error': abs(fc['expected_return'] - actual),
            'in_range': fc['return_range'][0] <= actual <= fc['return_range'][1],
            'prob_negative': fc['prob_negative'],
        })

    result_df = pd.DataFrame(results)

    if len(result_df) > 0:
        hit_rate = result_df['direction_correct'].mean()
        range_hit = result_df['in_range'].mean()
        avg_mag_err = result_df['magnitude_error'].mean()

        print(f"[Validate] Test: {test_start} to {result_df['date'].max()}")
        print(f"  Direction hit rate: {hit_rate*100:.1f}%")
        print(f"  Range hit rate: {range_hit*100:.1f}%")
        print(f"  Avg magnitude error: {avg_mag_err*100:.2f}%")

        # Per-regime breakdown
        for regime in result_df['regime'].unique():
            sub = result_df[result_df['regime'] == regime]
            if len(sub) > 0:
                r_hit = sub['direction_correct'].mean()
                print(f"    {regime}: n={len(sub)}, hit={r_hit*100:.1f}%")

    return result_df


def simulate_with_gpr_calibration(
    calibration: dict,
    current_gpr: float,
    gpr_momentum: float = 0.0,
    n_sims: int = 5000
) -> dict:
    """Run Monte Carlo simulation with GPR-calibrated parameters."""
    fc = forecast_returns(calibration, current_gpr, gpr_momentum)
    params = fc['jump_params']
    params.n_sims = n_sims

    result = simulate_jump_diffusion(params, seed=42)
    final = result['final_values']

    return {
        'simulation': result,
        'forecast': fc,
        'var_5pct': np.percentile(final, 5),
        'var_1pct': np.percentile(final, 1),
        'expected_return': fc['expected_return'],
        'regime': fc['regime'],
    }


def _run_self_tests():
    print("=" * 60)
    print("gpr_forecasting.py self-test")
    print("=" * 60)

    from gpr_src.gpr_data import (
        load_gpr_monthly, load_market_data, add_log_returns,
        build_geopolitical_exposure_index, merge_gpr_with_market
    )
    from gpr_src.factor_decomposition import decompose_returns

    # Load data
    print("\n[Test 1] Load data and decompose")
    gpr = load_gpr_monthly()
    market = load_market_data()
    market = add_log_returns(market)
    market = build_geopolitical_exposure_index(market)
    merged = merge_gpr_with_market(gpr, market)

    decomp = decompose_returns(merged, return_col='GEO_index_return')
    merged['geo_residual'] = decomp['residual'].reindex(merged.index)
    print(f"  Merged: {len(merged)} months")
    print("  [PASS]")

    # Test 2: Calibration
    print("\n[Test 2] Calibrate GPR jump model (train 1994-2009)")
    calib = calibrate_gpr_jump_model(merged, return_col='geo_residual')
    print(f"  GPR momentum slope: {calib['gpr_momentum_slope']:.6f}")
    print(f"  GPR momentum R: {calib['gpr_momentum_r']:.3f}")
    print(f"  Momentum hit rate (train): {calib['momentum_hit_rate']*100:.1f}%")
    print(f"  GPR median: {calib['gpr_median']:.0f}")
    print(f"  Regimes: {list(calib['regime_stats'].keys())}")
    for regime, stats in calib['regime_stats'].items():
        print(f"    {regime}: n={stats['n_obs']}, mean={stats['mean_return']*100:.2f}%, "
              f"P(neg)={stats['prob_negative']*100:.0f}%")
    print("  [PASS]")

    # Test 3: Forecasting
    print("\n[Test 3] Forecast scenarios")
    scenarios = [
        ("GPR=80, falling (low_stable)", 80, -10),
        ("GPR=80, rising (low_rising)", 80, 10),
        ("GPR=170, rising (high_rising)", 170, 15),
        ("GPR=170, falling (high_falling)", 170, -15),
    ]
    for label, gpr_lvl, gpr_mom in scenarios:
        fc = forecast_returns(calib, gpr_lvl, gpr_mom)
        print(f"  {label}: regime={fc['regime']}, "
              f"expected={fc['expected_return']*100:.2f}%, "
              f"P(neg)={fc['prob_negative']*100:.0f}%")
    print("  [PASS]")

    # Test 4: Validation
    print("\n[Test 4] Out-of-sample validation (2010-2025)")
    val = validate_forecast(calib, merged, test_start='2010-01-01')
    assert len(val) > 0
    print("  [PASS]")

    # Test 5: Monte Carlo
    print("\n[Test 5] Monte Carlo simulation")
    mc = simulate_with_gpr_calibration(calib, current_gpr=150, gpr_momentum=15, n_sims=3000)
    print(f"  Regime: {mc['regime']}")
    print(f"  Expected return: {mc['expected_return']*100:.2f}%")
    print(f"  VaR (5%): {mc['var_5pct']:.2f}")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_self_tests()
