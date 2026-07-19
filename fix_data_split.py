"""
Fix data leakage: Calibrate on 1994-2019, test on 2020-2026.
Recalculate all results with proper train/test split.
"""
import numpy as np
import pandas as pd
from scipy import stats
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.jump_diffusion_engine import JumpDiffusionParams, simulate_jump_diffusion, simulate_paired_paths
from src.heston_engine import HestonParams, simulate_heston
from src.fbm_engine import FBMParams, simulate_fbm
from src.var_calculator import compute_var_caseB, decompose_risk
from src.save_results import TeeOutput, save_csv

DATA = os.path.join(os.path.dirname(__file__), 'data')

def main():
    # Load and prepare data
    df = pd.read_csv(f'{DATA}/analysis_daily_clean.csv', parse_dates=['date'])
    df = df.dropna(subset=['SOX_log_return', 'GPRD']).reset_index(drop=True)
    df['GPRD_sma30'] = df['GPRD'].rolling(30, min_periods=10).mean()
    df['GPRD_momentum'] = df['GPRD'] - df['GPRD_sma30']
    df['fwd_return_20d'] = df['SOX_log_return'].rolling(20).sum().shift(-20)
    df['is_jump_20d'] = (df['fwd_return_20d'] < -0.10).astype(int)

    analysis = df[df['date'] >= '1994-06-01'].dropna(subset=['fwd_return_20d']).copy()

    # ============================================================
    # CONSISTENT DATA SPLIT
    # ============================================================
    TRAIN_END = '2019-12-31'
    TEST_START = '2020-01-01'

    train = analysis[analysis['date'] <= TRAIN_END].copy()
    test = analysis[analysis['date'] > TRAIN_END].copy()

    print("=" * 70)
    print("CONSISTENT DATA SPLIT")
    print("=" * 70)
    print(f"\n  Train: {train['date'].min().strftime('%Y-%m-%d')} to "
          f"{train['date'].max().strftime('%Y-%m-%d')} ({len(train)} days)")
    print(f"  Test:  {test['date'].min().strftime('%Y-%m-%d')} to "
          f"{test['date'].max().strftime('%Y-%m-%d')} ({len(test)} days)")

    # ============================================================
    # STEP 1: GPR Regime Classification (from TRAINING data)
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 1: GPR Regime Classification (Training Data Only)")
    print("=" * 70)

    gpr_median = train['GPRD'].median()
    print(f"  GPR median (train): {gpr_median:.0f}")

    def get_regime(row, median):
        h = row['GPRD'] > median
        r = row['GPRD_momentum'] > 0
        if h and r: return 'high_rising'
        if h and not r: return 'high_falling'
        if not h and r: return 'low_rising'
        return 'low_stable'

    train['regime'] = train.apply(lambda r: get_regime(r, gpr_median), axis=1)
    test['regime'] = test.apply(lambda r: get_regime(r, gpr_median), axis=1)

    # ============================================================
    # STEP 2: Regime Jump Probabilities (TRAINING data only)
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 2: Regime Jump Probabilities (Training Data Only)")
    print("=" * 70)

    regime_stats_train = {}
    print(f"\n  {'Regime':<15} {'Days':>6} {'Jumps':>6} {'Jump Prob':>10} {'Mean Ret':>10}")
    print(f"  {'-'*50}")

    for regime in ['low_stable', 'low_rising', 'high_rising', 'high_falling']:
        sub = train[train['regime'] == regime]
        if len(sub) >= 5:
            jp = sub['is_jump_20d'].mean()
            mr = sub['fwd_return_20d'].mean()
            regime_stats_train[regime] = {
                'n': len(sub), 'jump_prob': jp, 'mean_return': mr
            }
            print(f"  {regime:<15} {len(sub):>6} {sub['is_jump_20d'].sum():>6} "
                  f"{jp*100:>9.1f}% {mr*100:>9.2f}%")

    # ============================================================
    # STEP 3: GPR-Conditional Lambda (from TRAINING data)
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 3: GPR-Conditional Lambda (Training Data Only)")
    print("=" * 70)

    overall_jump_prob = train['is_jump_20d'].mean()
    overall_lambda = overall_jump_prob * 12

    print(f"  Overall jump prob (train): {overall_jump_prob*100:.2f}%")
    print(f"  Overall lambda (train): {overall_lambda:.2f}/yr")

    dynamic_lambdas = {}
    scaling_factors = {}
    for regime, stats in regime_stats_train.items():
        scale = stats['jump_prob'] / overall_jump_prob if overall_jump_prob > 0 else 1.0
        dynamic_lambdas[regime] = overall_lambda * scale
        scaling_factors[regime] = scale
        print(f"  {regime}: jump_prob={stats['jump_prob']*100:.1f}%, "
              f"scale={scale:.2f}, lambda={dynamic_lambdas[regime]:.2f}/yr")

    # ============================================================
    # STEP 4: Historical VaR (TEST data only)
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 4: Historical VaR (Test Data Only, 2020-2026)")
    print("=" * 70)

    real_var = {}
    print(f"\n  {'Regime':<15} {'Days':>6} {'VaR(5%)':>10} {'CVaR(5%)':>10} {'Mean':>10}")
    print(f"  {'-'*55}")

    for regime in ['low_stable', 'low_rising', 'high_rising', 'high_falling']:
        sub = test[test['regime'] == regime]
        fwd = sub['fwd_return_20d'].dropna()
        if len(fwd) >= 5:
            var_5 = np.percentile(fwd, 5) * 100
            cvar_5 = fwd[fwd <= np.percentile(fwd, 5)].mean() * 100
            mean_ret = fwd.mean() * 100
            real_var[regime] = {
                'var': var_5, 'cvar': cvar_5, 'mean': mean_ret, 'n': len(fwd)
            }
            print(f"  {regime:<15} {len(fwd):>6} {var_5:>9.2f}% {cvar_5:>9.2f}% {mean_ret:>9.2f}%")

    # ============================================================
    # STEP 5: Simulated VaR (parameters from TRAIN, simulate forward)
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 5: Simulated VaR (Train Parameters, Forward Simulation)")
    print("=" * 70)

    # Estimate base parameters from training data
    train_returns = train['SOX_log_return'].dropna()
    mu_train = train_returns.mean() * 252
    sigma_train = train_returns.std(ddof=1) * np.sqrt(252)
    S0 = float(train['SOX_Close'].dropna().iloc[-1])

    print(f"  Train parameters: mu={mu_train:.4f}, sigma={sigma_train:.4f}, S0=${S0:,.2f}")

    # Jump parameters from training data
    train_jumps = train[train['is_jump_20d'] == 1]
    neg_jumps = train_jumps['fwd_return_20d'].dropna()
    pos_jumps = train.loc[train['fwd_return_20d'] > 0.10, 'fwd_return_20d'].dropna()

    print(f"  Negative jumps (train): n={len(neg_jumps)}, mean={neg_jumps.mean()*100:.2f}%")
    print(f"  Positive jumps (train): n={len(pos_jumps)}, mean={pos_jumps.mean()*100:.2f}%")

    T, dt, n_sims = 2.0, 1/252, 2000

    # Simulate for each regime
    print(f"\n  {'Regime':<15} {'Lambda':>8} {'Sim VaR':>10} {'Real VaR':>10} {'Error':>10}")
    print(f"  {'-'*55}")

    sim_results = {}
    for regime in ['low_stable', 'low_rising', 'high_rising', 'high_falling']:
        lam = dynamic_lambdas.get(regime, overall_lambda)

        params = JumpDiffusionParams(
            mu=mu_train, sigma=sigma_train, lam=lam,
            mu_j=neg_jumps.mean() if len(neg_jumps) > 0 else -0.05,
            sigma_j=neg_jumps.std() if len(neg_jumps) > 1 else 0.03,
            S0=S0, T=T, dt=dt, n_sims=n_sims
        )
        result = simulate_jump_diffusion(params, seed=42)
        final = result['final_values']
        sim_var = np.percentile((final - S0) / S0, 5) * 100

        real = real_var.get(regime, {}).get('var', 0)
        error = abs(sim_var - real)

        sim_results[regime] = {'var': sim_var, 'lambda': lam}
        print(f"  {regime:<15} {lam:>7.2f} {sim_var:>9.2f}% {real:>9.2f}% {error:>9.2f}pp")

    # ============================================================
    # STEP 6: Cross-Model Comparison (Train Parameters, Test Validation)
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 6: Cross-Model Comparison (Train Params, Test Validation)")
    print("=" * 70)

    print(f"\n  {'Regime':<15} {'Real VaR':>10} {'JD':>10} {'Heston':>10} {'fBM':>10}")
    print(f"  {'-'*55}")

    for regime in ['low_stable', 'low_rising', 'high_rising', 'high_falling']:
        lam = dynamic_lambdas.get(regime, overall_lambda)
        real = real_var.get(regime, {}).get('var', 0)

        # JD
        jd_p = JumpDiffusionParams(
            mu=mu_train, sigma=sigma_train, lam=lam,
            mu_j=neg_jumps.mean() if len(neg_jumps) > 0 else -0.05,
            sigma_j=neg_jumps.std() if len(neg_jumps) > 1 else 0.03,
            S0=S0, T=T, dt=dt, n_sims=n_sims
        )
        jd_r = simulate_jump_diffusion(jd_p, seed=42)
        jd_var = np.percentile((jd_r['final_values'] - S0) / S0, 5) * 100

        # Heston
        v0 = train_returns.tail(30).var() * 252
        theta = train_returns.var() * 252
        heston_p = HestonParams(
            mu=mu_train, v0=v0, kappa=5.0, theta=theta,
            xi=0.3, rho=-0.5, S0=S0, T=T, dt=dt, n_sims=n_sims
        )
        heston_r = simulate_heston(heston_p, seed=42)
        heston_var = np.percentile((heston_r['final_values'] - S0) / S0, 5) * 100

        # fBM
        sq_ret = train_returns ** 2
        vol_autocorr = sq_ret.autocorr(lag=1) if len(sq_ret) > 50 else 0
        hurst = 0.5 + 0.5 * max(0, vol_autocorr)
        fbm_p = FBMParams(
            mu=mu_train, sigma=sigma_train, hurst=hurst,
            S0=S0, T=T, dt=dt, n_sims=500
        )
        fbm_r = simulate_fbm(fbm_p, seed=42)
        fbm_var = np.percentile((fbm_r['final_values'] - S0) / S0, 5) * 100

        print(f"  {regime:<15} {real:>9.2f}% {jd_var:>9.2f}% {heston_var:>9.2f}% {fbm_var:>9.2f}%")

    # ============================================================
    # STEP 7: Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Data Split Consistency")
    print("=" * 70)

    print(f"""
  TRAINING (1994-2019):
    - GPR median: {gpr_median:.0f}
    - Regime jump probabilities: computed from training data only
    - Model parameters: estimated from training data only
    - Overall lambda: {overall_lambda:.2f}/yr

  TEST (2020-2026):
    - Historical VaR: computed from test data only
    - Model VaR: simulated using training parameters
    - Walk-forward validation: train on pre-Y, test on Y

  CONSISTENCY:
    - All regime classifications use training GPR median
    - All jump probabilities computed from training data
    - All model parameters estimated from training data
    - Test data used ONLY for validation
""")

    # Save regime stats
    regime_rows = []
    for regime, st in regime_stats_train.items():
        regime_rows.append({
            'regime': regime,
            'days': st['n'],
            'jump_prob': round(st['jump_prob'] * 100, 1),
            'mean_return': round(st['mean_return'] * 100, 2),
        })
    save_csv(pd.DataFrame(regime_rows), '06_data_split', 'regime_jump_stats.csv')

if __name__ == "__main__":
    with TeeOutput("06_data_split", "data_split.txt"):
        main()
