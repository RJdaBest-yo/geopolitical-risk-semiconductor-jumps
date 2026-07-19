"""
=============================================================================
FILE: multifactor_study/src/multifactor_jump.py
PURPOSE: Multi-factor Jump Diffusion model combining:
  - Geopolitics (GPR)
  - Market Fear (VIX)
  - Economic Transmission (Oil)
  - Policy/Fight-to-Safety (USD)

Compare how spikes in different factors impact:
  1. Jump frequency (lambda)
  2. Jump amplitude (mu_j)
  3. VaR under bidirectional jumps
  4. Which factor matters most for semiconductor tail risk
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
import sys, os

# Add main project to path
_main_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _main_root not in sys.path:
    sys.path.insert(0, _main_root)

from src.jump_diffusion_engine import JumpDiffusionParams, simulate_jump_diffusion, simulate_paired_paths
from src.var_calculator import compute_var_caseB, decompose_risk

DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'gpr_deep_dive', 'data')


def load_multifactor_data() -> pd.DataFrame:
    """Load and prepare multi-factor dataset."""
    df = pd.read_csv(f'{DATA}/analysis_daily_clean.csv', parse_dates=['date'])
    df = df.dropna(subset=['SOX_log_return']).reset_index(drop=True)

    # GPR features
    df['GPRD_sma30'] = df['GPRD'].rolling(30, min_periods=10).mean()
    df['GPRD_momentum'] = df['GPRD'] - df['GPRD_sma30']

    # VIX features
    if 'VIX_Close' in df.columns:
        df['VIX_sma20'] = df['VIX_Close'].rolling(20, min_periods=10).mean()
        df['VIX_momentum'] = df['VIX_Close'] - df['VIX_sma20']

    # Oil features
    if 'OIL_Close' in df.columns:
        df['OIL_sma20'] = df['OIL_Close'].rolling(20, min_periods=10).mean()
        df['OIL_momentum'] = df['OIL_Close'] - df['OIL_sma20']

    # USD features
    if 'USD_Close' in df.columns:
        df['USD_sma20'] = df['USD_Close'].rolling(20, min_periods=10).mean()
        df['USD_momentum'] = df['USD_Close'] - df['USD_sma20']

    # Forward returns
    df['fwd_return_5d'] = df['SOX_log_return'].rolling(5).sum().shift(-5)
    df['fwd_return_20d'] = df['SOX_log_return'].rolling(20).sum().shift(-20)

    # Jump indicators
    df['is_neg_jump_5d'] = (df['fwd_return_5d'] < -0.05).astype(int)
    df['is_pos_jump_5d'] = (df['fwd_return_5d'] > 0.05).astype(int)
    df['is_neg_jump_20d'] = (df['fwd_return_20d'] < -0.10).astype(int)
    df['is_pos_jump_20d'] = (df['fwd_return_20d'] > 0.10).astype(int)

    return df


def compute_factor_regimes(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """
    Classify each day into multi-factor regimes.

    For each factor, classify as 'high' or 'low' relative to its median.
    Then combine into a composite regime.
    """
    df = df.copy()

    # Compute medians from training data
    medians = {}
    for col in ['GPRD', 'VIX_Close', 'OIL_Close', 'USD_Close']:
        if col in df.columns:
            medians[col] = df.loc[train_mask, col].median()

    # Classify each factor
    if 'GPRD' in medians:
        df['gpr_high'] = (df['GPRD'] > medians['GPRD']).astype(int)
    if 'VIX_Close' in medians:
        df['vix_high'] = (df['VIX_Close'] > medians['VIX_Close']).astype(int)
    if 'OIL_Close' in medians:
        df['oil_high'] = (df['OIL_Close'] > medians['OIL_Close']).astype(int)
    if 'USD_Close' in medians:
        df['usd_high'] = (df['USD_Close'] > medians['USD_Close']).astype(int)

    # Composite risk score (normalized)
    risk_components = []
    for col in ['GPRD', 'VIX_Close', 'OIL_Close', 'USD_Close']:
        if col in df.columns:
            z = (df[col] - df.loc[train_mask, col].mean()) / df.loc[train_mask, col].std()
            risk_components.append(z)

    if risk_components:
        df['composite_risk'] = np.mean(risk_components, axis=0)
    else:
        df['composite_risk'] = 0

    return df


def analyze_factor_impact(
    df: pd.DataFrame,
    factor_col: str,
    factor_label: str,
    train_end: str = '2019-12-31'
) -> dict:
    """
    Analyze how a single factor impacts jump probability and returns.

    Returns:
        dict with regime-specific jump probabilities, mean returns, and VaR
    """
    train = df[df['date'] <= train_end].copy()
    test = df[df['date'] > train_end].copy()

    # Classify by factor level
    median_val = train[factor_col].median()
    train['factor_regime'] = np.where(train[factor_col] > median_val, 'high', 'low')
    test['factor_regime'] = np.where(test[factor_col] > median_val, 'high', 'low')

    results = {}
    for regime in ['low', 'high']:
        sub_train = train[train['factor_regime'] == regime]
        sub_test = test[test['factor_regime'] == regime]

        if len(sub_train) > 20 and len(sub_test) > 5:
            results[regime] = {
                'train_n': len(sub_train),
                'test_n': len(sub_test),
                'train_neg_jump_20d': sub_train['is_neg_jump_20d'].mean(),
                'test_neg_jump_20d': sub_test['is_neg_jump_20d'].mean(),
                'train_pos_jump_20d': sub_train['is_pos_jump_20d'].mean(),
                'test_pos_jump_20d': sub_test['is_pos_jump_20d'].mean(),
                'train_mean_20d': sub_train['fwd_return_20d'].mean(),
                'test_mean_20d': sub_test['fwd_return_20d'].mean(),
                'test_var_20d': np.percentile(sub_test['fwd_return_20d'].dropna(), 5),
                'median_factor': sub_train[factor_col].median(),
            }

    return results


def analyze_multifactor_interactions(
    df: pd.DataFrame,
    train_end: str = '2019-12-31'
) -> pd.DataFrame:
    """
    Analyze how combinations of factors impact jump probability.

    For each pair of factors, compute jump probability when both are high,
    both are low, or mixed.
    """
    train = df[df['date'] <= train_end].copy()

    # Get available factor regimes
    factor_pairs = []
    if 'gpr_high' in train.columns and 'vix_high' in train.columns:
        factor_pairs.append(('GPR', 'VIX', 'gpr_high', 'vix_high'))
    if 'gpr_high' in train.columns and 'oil_high' in train.columns:
        factor_pairs.append(('GPR', 'OIL', 'gpr_high', 'oil_high'))
    if 'gpr_high' in train.columns and 'usd_high' in train.columns:
        factor_pairs.append(('GPR', 'USD', 'gpr_high', 'usd_high'))
    if 'vix_high' in train.columns and 'oil_high' in train.columns:
        factor_pairs.append(('VIX', 'OIL', 'vix_high', 'oil_high'))

    results = []
    for f1_label, f2_label, f1_col, f2_col in factor_pairs:
        for combo in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            mask = (train[f1_col] == combo[0]) & (train[f2_col] == combo[1])
            sub = train[mask]
            if len(sub) > 10:
                results.append({
                    'Factor_1': f1_label,
                    'Factor_2': f2_label,
                    'F1_State': 'low' if combo[0] == 0 else 'high',
                    'F2_State': 'low' if combo[1] == 0 else 'high',
                    'Days': len(sub),
                    'Neg_Jump_20d': sub['is_neg_jump_20d'].mean() * 100,
                    'Pos_Jump_20d': sub['is_pos_jump_20d'].mean() * 100,
                    'Mean_20d_Return': sub['fwd_return_20d'].mean() * 100,
                })

    return pd.DataFrame(results)


def simulate_multifactor_jumps(
    df: pd.DataFrame,
    factor_col: str,
    train_end: str = '2019-12-31',
    n_sims: int = 3000
) -> dict:
    """
    Simulate Jump Diffusion with factor-conditional lambda.

    Compare: static lambda vs factor-conditional lambda.
    """
    train = df[df['date'] <= train_end]
    test = df[df['date'] > train_end]

    # Base parameters from training
    train_returns = train['SOX_log_return'].dropna()
    mu = train_returns.mean() * 252
    sigma = train_returns.std() * np.sqrt(252)
    S0 = float(train['SOX_Close'].dropna().iloc[-1])

    # Factor-conditional lambda
    median_val = train[factor_col].median()
    overall_jump_prob = train['is_neg_jump_20d'].mean()
    overall_lambda = overall_jump_prob * 12

    low_sub = train[train[factor_col] <= median_val]
    high_sub = train[train[factor_col] > median_val]

    low_lambda = (low_sub['is_neg_jump_20d'].mean() * 12) if len(low_sub) > 10 else overall_lambda
    high_lambda = (high_sub['is_neg_jump_20d'].mean() * 12) if len(high_sub) > 10 else overall_lambda

    # Jump parameters from training
    neg_jumps = train.loc[train['is_neg_jump_20d'] == 1, 'fwd_return_20d'].dropna()
    pos_jumps = train.loc[train['is_pos_jump_20d'] == 1, 'fwd_return_20d'].dropna()

    T, dt = 2.0, 1/252

    # Simulate for low and high regimes
    results = {}
    for regime, lam in [('low', low_lambda), ('high', high_lambda)]:
        p = JumpDiffusionParams(
            mu=mu, sigma=sigma, lam=lam,
            mu_j=neg_jumps.mean() if len(neg_jumps) > 0 else -0.05,
            sigma_j=neg_jumps.std() if len(neg_jumps) > 1 else 0.03,
            S0=S0, T=T, dt=dt, n_sims=n_sims
        )
        r = simulate_jump_diffusion(p, seed=42)
        final = r['final_values']
        var_5 = np.percentile((final - S0) / S0, 5) * 100
        cvar_5 = ((final - S0) / S0)[(final - S0) / S0 <= np.percentile((final - S0) / S0, 5)].mean() * 100

        results[regime] = {
            'lambda': lam,
            'var_5': var_5,
            'cvar_5': cvar_5,
            'median': np.median(final),
        }

    return {
        'factor': factor_col,
        'overall_lambda': overall_lambda,
        'low_lambda': low_lambda,
        'high_lambda': high_lambda,
        'results': results,
    }


def _run_self_tests():
    print("=" * 70)
    print("Multi-Factor Jump Diffusion Self-Test")
    print("=" * 70)

    # Load data
    print("\n[Test 1] Load multi-factor data")
    df = load_multifactor_data()
    train_mask = df['date'] <= '2019-12-31'
    df = compute_factor_regimes(df, train_mask)
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"  Available factors: {[c for c in ['GPRD', 'VIX_Close', 'OIL_Close', 'USD_Close'] if c in df.columns]}")
    print("  [PASS]")

    # Test 2: Single factor analysis
    print("\n[Test 2] Single factor impact analysis")
    factors = [
        ('GPRD', 'Geopolitics (GPR)'),
        ('VIX_Close', 'Market Fear (VIX)'),
        ('OIL_Close', 'Oil Price'),
        ('USD_Close', 'USD Strength'),
    ]

    for col, label in factors:
        if col in df.columns:
            result = analyze_factor_impact(df, col, label)
            print(f"\n  {label}:")
            for regime, stats in result.items():
                print(f"    {regime}: n={stats['train_n']}, "
                      f"neg_jump={stats['train_neg_jump_20d']*100:.1f}%, "
                      f"mean_ret={stats['train_mean_20d']*100:.2f}%")
    print("  [PASS]")

    # Test 3: Multi-factor interactions
    print("\n[Test 3] Multi-factor interactions")
    interactions = analyze_multifactor_interactions(df)
    print(f"  Combinations tested: {len(interactions)}")
    if len(interactions) > 0:
        print(interactions.to_string(index=False))
    print("  [PASS]")

    # Test 4: Factor-conditional simulation
    print("\n[Test 4] Factor-conditional JD simulation")
    for col, label in factors:
        if col in df.columns:
            sim = simulate_multifactor_jumps(df, col)
            print(f"  {label}:")
            print(f"    Low {col}: lambda={sim['low_lambda']:.2f}, "
                  f"VaR={sim['results']['low']['var_5']:.1f}%")
            print(f"    High {col}: lambda={sim['high_lambda']:.2f}, "
                  f"VaR={sim['results']['high']['var_5']:.1f}%")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All tests PASSED")
    print("=" * 70)


if __name__ == "__main__":
    _run_self_tests()
