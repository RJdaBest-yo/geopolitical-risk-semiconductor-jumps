"""
Walk-Forward Expanding Window Analysis
=======================================

For each cycle:
  - Training: 1994 to year Y
  - Testing: year Y+1 to 2026

Cycles:
  Cycle 1:  Train 1994-2000, Test 2001-2026
  Cycle 2:  Train 1994-2001, Test 2002-2026
  ...
  Cycle 19: Train 1994-2019, Test 2020-2026

For each cycle, compute:
  1. GPR-return correlation (training period)
  2. Regime jump probabilities (training period)
  3. Historical VaR by regime (test period)
  4. Direction hit rate (test period)
  5. Which cycle gives the best fit?
"""

import numpy as np
import pandas as pd
from scipy import stats
import sys, os

# Add paths
_gpr_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _gpr_dir)
sys.path.insert(0, os.path.join(_gpr_dir, '..'))

from src.save_results import TeeOutput, save_csv

DATA = os.path.join(_gpr_dir, 'data')
OUTPUT = os.path.join(_gpr_dir, 'pilot')

def main():
    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    print("Loading data...")
    df = pd.read_csv(f'{DATA}/analysis_daily_clean.csv', parse_dates=['date'])
    df = df.dropna(subset=['SOX_log_return', 'GPRD']).reset_index(drop=True)
    df['GPRD_sma30'] = df['GPRD'].rolling(30, min_periods=10).mean()
    df['GPRD_momentum'] = df['GPRD'] - df['GPRD_sma30']
    df['fwd_return_20d'] = df['SOX_log_return'].rolling(20).sum().shift(-20)
    df['is_jump_20d'] = (df['fwd_return_20d'] < -0.10).astype(int)

    analysis = df[df['date'] >= '1994-06-01'].dropna(subset=['fwd_return_20d']).copy()
    print(f"  Analysis dataset: {len(analysis)} days, {analysis['date'].min()} to {analysis['date'].max()}")

    # ============================================================
    # 2. WALK-FORWARD CYCLES
    # ============================================================
    print("\n" + "=" * 70)
    print("WALK-FORWARD EXPANDING WINDOW ANALYSIS")
    print("=" * 70)

    results = []

    for train_end_year in range(2000, 2020):
        test_start_year = train_end_year + 1

        train = analysis[analysis['date'] <= f'{train_end_year}-12-31'].copy()
        test = analysis[analysis['date'] >= f'{test_start_year}-01-01'].copy()

        if len(train) < 100 or len(test) < 50:
            continue

        # --- Training: Compute GPR regime parameters ---
        gpr_median = train['GPRD'].median()

        def get_regime(row, median):
            h = row['GPRD'] > median
            r = row['GPRD_momentum'] > 0
            if h and r: return 'high_rising'
            if h and not r: return 'high_falling'
            if not h and r: return 'low_rising'
            return 'low_stable'

        train['regime'] = train.apply(lambda r: get_regime(r, gpr_median), axis=1)
        test['regime'] = test.apply(lambda r: get_regime(r, gpr_median), axis=1)

        # Regime jump probabilities (training)
        regime_stats = {}
        for regime in ['low_stable', 'low_rising', 'high_rising', 'high_falling']:
            sub = train[train['regime'] == regime]
            if len(sub) >= 5:
                regime_stats[regime] = {
                    'n': len(sub),
                    'jump_prob': sub['is_jump_20d'].mean(),
                    'mean_return': sub['fwd_return_20d'].mean(),
                }

        # Overall lambda
        overall_jump_prob = train['is_jump_20d'].mean()
        overall_lambda = overall_jump_prob * 12

        # Dynamic lambdas
        dynamic_lambdas = {}
        for regime, rs in regime_stats.items():
            scale = rs['jump_prob'] / overall_jump_prob if overall_jump_prob > 0 else 1.0
            dynamic_lambdas[regime] = overall_lambda * scale

        # --- Test: Compute historical VaR by regime ---
        test_var = {}
        for regime in ['low_stable', 'low_rising', 'high_rising', 'high_falling']:
            sub = test[test['regime'] == regime]
            fwd = sub['fwd_return_20d'].dropna()
            if len(fwd) >= 5:
                test_var[regime] = {
                    'n': len(fwd),
                    'var_5': np.percentile(fwd, 5) * 100,
                    'mean': fwd.mean() * 100,
                }

        # --- Direction hit rate (GPR level → 20d return direction) ---
        train_clean = train.dropna(subset=['fwd_return_20d', 'GPRD'])
        test_clean = test.dropna(subset=['fwd_return_20d', 'GPRD'])

        if len(train_clean) > 30 and len(test_clean) > 10:
            slope, intercept, r, p, _ = stats.linregress(
                train_clean['GPRD'], train_clean['fwd_return_20d']
            )
            test_clean = test_clean.copy()
            test_clean['forecast'] = slope * test_clean['GPRD'] + intercept
            hit_rate = (np.sign(test_clean['forecast']) == np.sign(test_clean['fwd_return_20d'])).mean()
        else:
            hit_rate = np.nan

        # --- GPR-return correlation (training period) ---
        corr, corr_p = stats.pearsonr(train_clean['GPRD'], train_clean['fwd_return_20d']) if len(train_clean) > 20 else (np.nan, np.nan)

        # --- Record results ---
        result = {
            'cycle': train_end_year - 1999,
            'train_start': '1994',
            'train_end': str(train_end_year),
            'test_start': str(test_start_year),
            'test_end': '2026',
            'train_days': len(train),
            'test_days': len(test),
            'gpr_median': gpr_median,
            'gpr_corr': corr,
            'gpr_corr_p': corr_p,
            'overall_lambda': overall_lambda,
            'hit_rate': hit_rate,
        }

        # Add regime-specific data
        for regime in ['low_stable', 'low_rising', 'high_rising', 'high_falling']:
            if regime in regime_stats:
                result[f'{regime}_jump_prob'] = regime_stats[regime]['jump_prob']
                result[f'{regime}_lambda'] = dynamic_lambdas[regime]
            if regime in test_var:
                result[f'{regime}_test_var'] = test_var[regime]['var_5']
                result[f'{regime}_test_mean'] = test_var[regime]['mean']
                result[f'{regime}_test_n'] = test_var[regime]['n']

        results.append(result)

        # Print summary
        print(f"\n  Cycle {result['cycle']:>2}: Train {train_end_year} | Test {test_start_year}-2026")
        print(f"    Train: {len(train)} days, GPR median: {gpr_median:.0f}")
        print(f"    GPR-return corr: {corr:.3f} (p={corr_p:.4f})")
        print(f"    Overall lambda: {overall_lambda:.2f}/yr")
        print(f"    Hit rate: {hit_rate*100:.1f}%" if not np.isnan(hit_rate) else "    Hit rate: N/A")
        print(f"    Regime lambdas: low_stable={dynamic_lambdas.get('low_stable', 0):.2f}, "
              f"low_rising={dynamic_lambdas.get('low_rising', 0):.2f}, "
              f"high_rising={dynamic_lambdas.get('high_rising', 0):.2f}, "
              f"high_falling={dynamic_lambdas.get('high_falling', 0):.2f}")

    # ============================================================
    # 3. SAVE RESULTS
    # ============================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    results_path = os.path.join(OUTPUT, 'walk_forward_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n  Saved: {results_path}")

    # Also save to output/results
    save_csv(results_df, '01_walk_forward', 'walk_forward_results.csv')

    # ============================================================
    # 4. ANALYSIS: Which cycle is best?
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS: Which Training Period Produces Best Results?")
    print("=" * 70)

    print(f"\n  {'Cycle':>6} {'Train End':>10} {'GPR Corr':>10} {'Hit Rate':>10} {'Overall λ':>10}")
    print(f"  {'-'*50}")
    for _, row in results_df.iterrows():
        hr = f"{row['hit_rate']*100:.1f}%" if not np.isnan(row['hit_rate']) else "N/A"
        print(f"  {int(row['cycle']):>6} {row['train_end']:>10} {row['gpr_corr']:>10.3f} {hr:>10} {row['overall_lambda']:>10.2f}")

    # Best cycle by hit rate
    valid = results_df.dropna(subset=['hit_rate'])
    if len(valid) > 0:
        best_hit = valid.loc[valid['hit_rate'].idxmax()]
        print(f"\n  Best by hit rate: Cycle {int(best_hit['cycle'])} "
              f"(train {best_hit['train_end']}, hit rate {best_hit['hit_rate']*100:.1f}%)")

    # Best cycle by GPR correlation (absolute value)
    best_corr = results_df.loc[results_df['gpr_corr'].abs().idxmax()]
    print(f"  Best by GPR correlation: Cycle {int(best_corr['cycle'])} "
          f"(train {best_corr['train_end']}, corr {best_corr['gpr_corr']:.3f})")

    # ============================================================
    # 5. REGIME STABILITY ANALYSIS
    # ============================================================
    print("\n" + "=" * 70)
    print("REGIME STABILITY: How consistent are regime patterns across cycles?")
    print("=" * 70)

    regime_cols = ['low_stable_jump_prob', 'low_rising_jump_prob',
                   'high_rising_jump_prob', 'high_falling_jump_prob']

    if all(c in results_df.columns for c in regime_cols):
        print(f"\n  {'Cycle':>6} {'low_stable':>12} {'low_rising':>12} {'high_rising':>12} {'high_falling':>12}")
        print(f"  {'-'*60}")
        for _, row in results_df.iterrows():
            vals = []
            for c in regime_cols:
                v = row.get(c, np.nan)
                vals.append(f"{v*100:>10.1f}%" if not np.isnan(v) else "N/A")
            print(f"  {int(row['cycle']):>6} {'  '.join(vals)}")

        # Is low_rising always the highest?
        low_rising_wins = 0
        total = 0
        for _, row in results_df.iterrows():
            vals = {c: row.get(c, np.nan) for c in regime_cols}
            if not any(np.isnan(v) for v in vals.values()):
                total += 1
                if vals['low_rising_jump_prob'] == max(vals.values()):
                    low_rising_wins += 1

        if total > 0:
            print(f"\n  low_rising is highest jump prob in {low_rising_wins}/{total} cycles")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

if __name__ == "__main__":
    with TeeOutput("01_walk_forward", "walk_forward.txt"):
        main()
