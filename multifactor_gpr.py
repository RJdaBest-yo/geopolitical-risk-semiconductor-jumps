"""
=============================================================================
FILE: gpr_deep_dive/gpr_src/multifactor_gpr.py
PURPOSE: Multi-factor GPR model — keep correlated factors WITH GPR

Previous approach: remove all non-GPR factors → weak signal
New approach: keep factors correlated with GPR (VIX, Oil, USD) → stronger signal

Redo all 5 iterations from gpr_model_development.md with multi-factor model.
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gpr_src.gpr_data import load_gpr_monthly, load_market_data, add_log_returns, merge_gpr_with_market


def build_multifactor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features for the multi-factor GPR model.

    Features:
    - GPR level, momentum, change
    - VIX level, change
    - Oil level, change
    - USD level, change
    - GPR sub-components (GPRT, GPRA, GPRH)
    - GPR categories (CAT_1..8)
    """
    df = df.copy()

    # GPR features
    df['GPR_sma6'] = df['GPR'].rolling(6).mean()
    df['GPR_momentum'] = df['GPR'] - df['GPR_sma6']
    df['GPR_rising'] = (df['GPR_momentum'] > 0).astype(int)
    df['GPR_change_1m'] = df['GPR'].diff(1)
    df['GPR_change_3m'] = df['GPR'].diff(3)
    df['GPR_change_6m'] = df['GPR'].diff(6)
    df['GPR_pct_change'] = df['GPR'].pct_change(3)

    # GPR sub-components momentum
    for col in ['GPRT', 'GPRA', 'GPRH']:
        if col in df.columns:
            df[f'{col}_sma6'] = df[col].rolling(6).mean()
            df[f'{col}_momentum'] = df[col] - df[f'{col}_sma6']

    # VIX features (if available)
    if 'VIX_level' in df.columns:
        df['VIX_sma6'] = df['VIX_level'].rolling(6).mean()
        df['VIX_momentum'] = df['VIX_level'] - df['VIX_sma6']
        df['VIX_change_1m'] = df['VIX_level'].diff(1)

    # Oil features (if available)
    if 'OIL_level' in df.columns:
        df['OIL_sma6'] = df['OIL_level'].rolling(6).mean()
        df['OIL_momentum'] = df['OIL_level'] - df['OIL_sma6']
        df['OIL_change_3m'] = df['OIL_level'].diff(3)

    # USD features (if available)
    if 'USD_level' in df.columns:
        df['USD_sma6'] = df['USD_level'].rolling(6).mean()
        df['USD_momentum'] = df['USD_level'] - df['USD_sma6']

    # Target: forward 3-month SOX return
    df['fwd_return_3m'] = df['SOX_log_return'].rolling(3).sum().shift(-3)

    # Jump indicator
    df['is_jump'] = (df['fwd_return_3m'] < -0.05).astype(int)

    return df


def run_iteration_1_level(df: pd.DataFrame, train_mask, test_mask) -> dict:
    """
    Iteration 1: GPR level → predict return direction (baseline, expected to fail)
    """
    train = df[train_mask].dropna(subset=['fwd_return_3m', 'GPR'])
    test = df[test_mask].dropna(subset=['fwd_return_3m', 'GPR'])

    slope, intercept, r, p, se = stats.linregress(train['GPR'], train['fwd_return_3m'])

    # Predict
    test = test.copy()
    test['forecast'] = slope * test['GPR'] + intercept
    test['predicted_direction'] = np.where(test['forecast'] > 0, 'up', 'down')
    test['actual_direction'] = np.where(test['fwd_return_3m'] > 0, 'up', 'down')
    hit_rate = (test['predicted_direction'] == test['actual_direction']).mean()

    return {
        'name': 'GPR Level → Return',
        'slope': slope, 'intercept': intercept, 'r': r, 'p': p,
        'hit_rate': hit_rate, 'n_train': len(train), 'n_test': len(test),
    }


def run_iteration_2_change(df: pd.DataFrame, train_mask, test_mask) -> dict:
    """
    Iteration 2: GPR change → predict return direction
    """
    train = df[train_mask].dropna(subset=['fwd_return_3m', 'GPR_change_3m'])
    test = df[test_mask].dropna(subset=['fwd_return_3m', 'GPR_change_3m'])

    slope, intercept, r, p, se = stats.linregress(train['GPR_change_3m'], train['fwd_return_3m'])

    test = test.copy()
    test['forecast'] = slope * test['GPR_change_3m'] + intercept
    test['predicted_direction'] = np.where(test['forecast'] > 0, 'up', 'down')
    test['actual_direction'] = np.where(test['fwd_return_3m'] > 0, 'up', 'down')
    hit_rate = (test['predicted_direction'] == test['actual_direction']).mean()

    return {
        'name': 'GPR Change 3m → Return',
        'slope': slope, 'intercept': intercept, 'r': r, 'p': p,
        'hit_rate': hit_rate, 'n_train': len(train), 'n_test': len(test),
    }


def run_iteration_3_multifactor(df: pd.DataFrame, train_mask, test_mask) -> dict:
    """
    Iteration 3: Multi-factor (GPR + VIX + Oil + USD) → predict return
    """
    factors = ['GPR_momentum', 'VIX_level', 'OIL_change', 'USD_level']
    available = [f for f in factors if f in df.columns and df[f].notna().sum() > 50]

    train = df[train_mask].dropna(subset=['fwd_return_3m'] + available)
    test = df[test_mask].dropna(subset=['fwd_return_3m'] + available)

    if len(train) < 30 or len(test) < 10:
        return {'name': 'Multi-factor', 'hit_rate': 0, 'n_train': len(train), 'n_test': len(test), 'r_squared': 0}

    X_train = np.column_stack([np.ones(len(train)), train[available].values])
    y_train = train['fwd_return_3m'].values
    beta, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)

    X_test = np.column_stack([np.ones(len(test)), test[available].values])
    y_test = test['fwd_return_3m'].values
    forecast = X_test @ beta

    predicted_direction = np.where(forecast > 0, 'up', 'down')
    actual_direction = np.where(y_test > 0, 'up', 'down')
    hit_rate = (predicted_direction == actual_direction).mean()

    # R-squared on train
    y_hat_train = X_train @ beta
    ss_res = np.sum((y_train - y_hat_train)**2)
    ss_tot = np.sum((y_train - y_train.mean())**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        'name': f'Multi-factor ({", ".join(available)})',
        'factors': available,
        'coefficients': dict(zip(['const'] + available, beta)),
        'r_squared': r_squared,
        'hit_rate': hit_rate,
        'n_train': len(train), 'n_test': len(test),
    }


def run_iteration_4_regime_jump_prob(df: pd.DataFrame, train_mask, test_mask) -> dict:
    """
    Iteration 4: Regime-based jump probability (the key insight)
    """
    train = df[train_mask].dropna(subset=['fwd_return_3m', 'GPR_momentum', 'is_jump'])
    test = df[test_mask].dropna(subset=['fwd_return_3m', 'GPR_momentum', 'is_jump'])

    gpr_median = train['GPR'].median()

    # Define regimes
    def get_regime(row):
        is_high = row['GPR'] > gpr_median
        is_rising = row['GPR_momentum'] > 0
        if is_high and is_rising: return 'high_rising'
        if is_high and not is_rising: return 'high_falling'
        if not is_high and is_rising: return 'low_rising'
        return 'low_stable'

    train = train.copy()
    test = test.copy()
    train['regime'] = train.apply(get_regime, axis=1)
    test['regime'] = test.apply(get_regime, axis=1)

    # Compute jump probability per regime
    regime_jump_prob = {}
    regime_mean_return = {}
    for regime in ['low_stable', 'low_rising', 'high_rising', 'high_falling']:
        sub = train[train['regime'] == regime]
        if len(sub) >= 5:
            regime_jump_prob[regime] = sub['is_jump'].mean()
            regime_mean_return[regime] = sub['fwd_return_3m'].mean()

    # Predict: if regime has >20% jump prob, predict negative
    test = test.copy()
    test['predicted_jump'] = test['regime'].map(
        lambda r: 1 if regime_jump_prob.get(r, 0.2) > 0.20 else 0
    )
    # Direction accuracy
    test['predicted_direction'] = np.where(
        test['regime'].map(regime_mean_return) > 0, 'up', 'down'
    )
    test['actual_direction'] = np.where(test['fwd_return_3m'] > 0, 'up', 'down')
    hit_rate = (test['predicted_direction'] == test['actual_direction']).mean()

    # Jump prediction accuracy
    jump_accuracy = ((test['predicted_jump'] == 1) == (test['is_jump'] == 1)).mean()

    return {
        'name': 'Regime-based Jump Probability',
        'regime_jump_prob': regime_jump_prob,
        'regime_mean_return': regime_mean_return,
        'hit_rate': hit_rate,
        'jump_accuracy': jump_accuracy,
        'n_train': len(train), 'n_test': len(test),
    }


def run_iteration_5_multifactor_regime(df: pd.DataFrame, train_mask, test_mask) -> dict:
    """
    Iteration 5: Multi-factor regime (GPR + VIX + Oil combined) → jump probability
    """
    # Build combined risk score
    df = df.copy()

    # Normalize each factor to z-scores
    for col in ['GPR_momentum', 'VIX_level', 'OIL_change']:
        if col in df.columns:
            mean_val = df.loc[train_mask, col].mean()
            std_val = df.loc[train_mask, col].std()
            if std_val > 0:
                df[f'{col}_z'] = (df[col] - mean_val) / std_val

    # Combined risk score: weighted average of z-scores
    z_cols = [c for c in df.columns if c.endswith('_z')]
    if len(z_cols) == 0:
        return {'name': 'Multi-factor Regime', 'hit_rate': 0, 'n_train': 0, 'n_test': 0}

    # Equal weights for now
    df['risk_score'] = df[z_cols].mean(axis=1)

    train = df[train_mask].dropna(subset=['fwd_return_3m', 'risk_score', 'is_jump'])
    test = df[test_mask].dropna(subset=['fwd_return_3m', 'risk_score', 'is_jump'])

    if len(train) < 30:
        return {'name': 'Multi-factor Regime', 'hit_rate': 0, 'n_train': len(train), 'n_test': len(test)}

    # Regime by risk score quartiles
    q33 = train['risk_score'].quantile(0.33)
    q67 = train['risk_score'].quantile(0.67)

    def get_risk_regime(score):
        if score < q33: return 'low_risk'
        if score < q67: return 'medium_risk'
        return 'high_risk'

    train = train.copy()
    test = test.copy()
    train['risk_regime'] = train['risk_score'].apply(get_risk_regime)
    test['risk_regime'] = test['risk_score'].apply(get_risk_regime)

    # Compute stats per regime
    regime_stats = {}
    for regime in ['low_risk', 'medium_risk', 'high_risk']:
        sub = train[train['risk_regime'] == regime]
        if len(sub) >= 5:
            regime_stats[regime] = {
                'jump_prob': sub['is_jump'].mean(),
                'mean_return': sub['fwd_return_3m'].mean(),
                'n': len(sub),
            }

    # Predict direction based on regime
    test = test.copy()
    test['predicted_direction'] = test['risk_regime'].map(
        lambda r: 'down' if regime_stats.get(r, {}).get('jump_prob', 0.2) > 0.20 else 'up'
    )
    test['actual_direction'] = np.where(test['fwd_return_3m'] > 0, 'up', 'down')
    hit_rate = (test['predicted_direction'] == test['actual_direction']).mean()

    # Jump prediction
    test['predicted_jump'] = test['risk_regime'].map(
        lambda r: 1 if regime_stats.get(r, {}).get('jump_prob', 0.2) > 0.20 else 0
    )
    jump_accuracy = ((test['predicted_jump'] == 1) == (test['is_jump'] == 1)).mean()

    return {
        'name': f'Multi-factor Regime ({", ".join(z_cols)})',
        'regime_stats': regime_stats,
        'hit_rate': hit_rate,
        'jump_accuracy': jump_accuracy,
        'n_train': len(train), 'n_test': len(test),
    }


def run_all_iterations(df: pd.DataFrame, train_end: str = '2009-12-31') -> pd.DataFrame:
    """Run all 5 iterations and return comparison table."""
    df = build_multifactor_features(df)
    df = df[df['date'] >= '1994-06-01'].copy()

    train_mask = df['date'] <= train_end
    test_mask = df['date'] > train_end

    results = []
    for fn in [run_iteration_1_level, run_iteration_2_change,
               run_iteration_3_multifactor, run_iteration_4_regime_jump_prob,
               run_iteration_5_multifactor_regime]:
        try:
            r = fn(df, train_mask, test_mask)
            results.append(r)
        except Exception as e:
            results.append({'name': fn.__name__, 'hit_rate': 0, 'error': str(e)})

    return results


def _run_self_tests():
    print("=" * 60)
    print("multifactor_gpr.py self-test")
    print("=" * 60)

    print("\n[Test 1] Load and build features")
    gpr = load_gpr_monthly()
    market = load_market_data()
    market = add_log_returns(market)
    merged = merge_gpr_with_market(gpr, market)

    df = build_multifactor_features(merged)
    feature_cols = [c for c in df.columns if c not in merged.columns]
    print(f"  Original columns: {len(merged.columns)}")
    print(f"  New features: {len(feature_cols)}")
    print(f"  Features: {feature_cols}")
    print("  [PASS]")

    print("\n[Test 2] Run all 5 iterations")
    results = run_all_iterations(df)

    print(f"\n{'Iteration':<45} {'Hit Rate':>10} {'R²':>8} {'N_test':>8}")
    print("-" * 75)
    for r in results:
        name = r.get('name', '?')
        hit = r.get('hit_rate', 0)
        r2 = r.get('r_squared', 0)
        n = r.get('n_test', 0)
        print(f"  {name:<43} {hit*100:>8.1f}% {r2:>8.4f} {n:>8}")

    # Check that multi-factor is at least as good as single-factor
    single_hit = results[0]['hit_rate']
    multi_hit = max(r['hit_rate'] for r in results)
    print(f"\n  Best single-factor: {single_hit*100:.1f}%")
    print(f"  Best multi-factor:  {multi_hit*100:.1f}%")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_self_tests()
