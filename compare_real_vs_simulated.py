"""
Compare model architectures using REAL 2020-2026 data
vs the old Monte Carlo approach.

Goal: Can we use real data to compare JD vs Heston vs fBM?
"""
import numpy as np
import pandas as pd
from scipy import stats
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.jump_diffusion_engine import JumpDiffusionParams, simulate_jump_diffusion
from src.heston_engine import HestonParams, simulate_heston
from src.fbm_engine import FBMParams, simulate_fbm
from src.save_results import TeeOutput, save_csv

DATA = os.path.join(os.path.dirname(__file__), 'data')

def main():
    # Load real data
    df = pd.read_csv(f'{DATA}/analysis_daily_clean.csv', parse_dates=['date'])
    df = df.dropna(subset=['SOX_log_return', 'GPRD']).reset_index(drop=True)
    df['GPRD_sma30'] = df['GPRD'].rolling(30, min_periods=10).mean()
    df['GPRD_momentum'] = df['GPRD'] - df['GPRD_sma30']
    df['fwd_return_20d'] = df['SOX_log_return'].rolling(20).sum().shift(-20)

    analysis = df[df['date'] >= '1994-06-01'].dropna(subset=['fwd_return_20d']).copy()

    # GPR regime
    gpr_median = analysis['GPRD'].median()

    def get_regime(row):
        h = row['GPRD'] > gpr_median
        r = row['GPRD_momentum'] > 0
        if h and r: return 'high_rising'
        if h and not r: return 'high_falling'
        if not h and r: return 'low_rising'
        return 'low_stable'

    analysis['regime'] = analysis.apply(get_regime, axis=1)

    print("=" * 70)
    print("REAL DATA MODEL COMPARISON (2020-2026)")
    print("=" * 70)

    # ============================================================
    # 1. Compute REAL historical VaR by regime (2020-2026)
    # ============================================================
    print("\n--- Real Historical VaR by Regime (2020-2026) ---")

    test_data = analysis[analysis['date'] >= '2020-01-01'].copy()

    real_var = {}
    for regime in ['low_stable', 'low_rising', 'high_rising', 'high_falling']:
        sub = test_data[test_data['regime'] == regime]
        fwd = sub['fwd_return_20d'].dropna()
        if len(fwd) > 5:
            var_5 = np.percentile(fwd, 5) * 100
            cvar_5 = fwd[fwd <= np.percentile(fwd, 5)].mean() * 100
            real_var[regime] = {
                'var': var_5, 'cvar': cvar_5,
                'mean': fwd.mean() * 100, 'n': len(fwd)
            }
            print(f"  {regime}: VaR={var_5:.2f}%, CVaR={cvar_5:.2f}%, n={len(fwd)}")

    # ============================================================
    # 2. Walk-forward model comparison (real data)
    # ============================================================
    print("\n" + "=" * 70)
    print("WALK-FORWARD: JD vs Heston vs fBM (2020-2026)")
    print("=" * 70)

    # For each year, train on all prior data, test on that year
    results = []

    for test_year in range(2020, 2027):
        train = analysis[analysis['date'] < f'{test_year}-01-01']
        test = analysis[
            (analysis['date'] >= f'{test_year}-01-01') &
            (analysis['date'] < f'{test_year+1}-01-01')
        ].dropna(subset=['fwd_return_20d', 'GPRD'])

        if len(train) < 500 or len(test) < 10:
            continue

        # Estimate parameters from training data
        train_returns = train['SOX_log_return'].dropna()
        mu = train_returns.mean() * 252
        sigma = train_returns.std(ddof=1) * np.sqrt(252)
        S0 = float(train['SOX_Close'].dropna().iloc[-1])

        # JD parameters
        train_jumps = train[train['fwd_return_20d'] < -0.10]
        jump_freq = len(train_jumps) / (len(train) / 252)
        jump_sizes = train_jumps['fwd_return_20d'].dropna()

        # Heston parameters (simplified)
        v0 = train_returns.tail(30).var() * 252
        theta = train_returns.var() * 252

        # fBM Hurst exponent (simplified: use rolling autocorrelation)
        sq_returns = train_returns ** 2
        if len(sq_returns) > 50:
            vol_autocorr = sq_returns.autocorr(lag=1)
            hurst_est = 0.5 + 0.5 * max(0, vol_autocorr)
        else:
            hurst_est = 0.55

        # GPR regime for test period
        test_regimes = test['regime'].value_counts()
        dominant_regime = test_regimes.index[0] if len(test_regimes) > 0 else 'low_stable'

        # Real test returns
        real_fwd = test['fwd_return_20d'].dropna()
        if len(real_fwd) < 5:
            continue

        real_var_5 = np.percentile(real_fwd, 5) * 100
        real_mean = real_fwd.mean() * 100

        # JD-predicted VaR (using calibrated lambda)
        jd_lambda = jump_freq * (1 + 0.5 * (test['GPRD'].mean() - gpr_median) / gpr_median)
        jd_params = JumpDiffusionParams(
            mu=mu, sigma=sigma, lam=max(jd_lambda, 0.1),
            mu_j=jump_sizes.mean() if len(jump_sizes) > 0 else -0.05,
            sigma_j=jump_sizes.std() if len(jump_sizes) > 1 else 0.03,
            S0=S0, T=1.0, dt=1/252, n_sims=2000
        )
        jd_result = simulate_jump_diffusion(jd_params, seed=42)
        jd_var = np.percentile(
            (jd_result['final_values'] - S0) / S0, 5
        ) * 100

        # Heston-predicted VaR
        heston_params = HestonParams(
            mu=mu, v0=v0, kappa=5.0, theta=theta,
            xi=0.3, rho=-0.5,
            S0=S0, T=1.0, dt=1/252, n_sims=2000
        )
        heston_result = simulate_heston(heston_params, seed=42)
        heston_var = np.percentile(
            (heston_result['final_values'] - S0) / S0, 5
        ) * 100

        # fBM-predicted VaR
        fbm_params = FBMParams(
            mu=mu, sigma=sigma, hurst=hurst_est,
            S0=S0, T=1.0, dt=1/252, n_sims=500
        )
        fbm_result = simulate_fbm(fbm_params, seed=42)
        fbm_var = np.percentile(
            (fbm_result['final_values'] - S0) / S0, 5
        ) * 100

        results.append({
            'year': test_year,
            'n_days': len(test),
            'dominant_regime': dominant_regime,
            'gpr_mean': test['GPRD'].mean(),
            'real_var': real_var_5,
            'real_mean': real_mean,
            'jd_var': jd_var,
            'heston_var': heston_var,
            'fbm_var': fbm_var,
            'jd_error': abs(jd_var - real_var_5),
            'heston_error': abs(heston_var - real_var_5),
            'fbm_error': abs(fbm_var - real_var_5),
        })

    res_df = pd.DataFrame(results)

    # ============================================================
    # 3. Results comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("RESULTS: Model VaR vs Real VaR")
    print("=" * 70)

    print(f"\n  {'Year':>6} {'Regime':<14} {'Real VaR':>10} {'JD VaR':>10} {'Heston':>10} {'fBM VaR':>10}")
    print(f"  {'-'*65}")
    for _, row in res_df.iterrows():
        print(f"  {int(row['year']):>6} {row['dominant_regime']:<14} "
              f"{row['real_var']:>9.2f}% {row['jd_var']:>9.2f}% "
              f"{row['heston_var']:>9.2f}% {row['fbm_var']:>9.2f}%")

    print(f"\n  {'':>6} {'AVERAGE':<14} "
          f"{res_df['real_var'].mean():>9.2f}% {res_df['jd_var'].mean():>9.2f}% "
          f"{res_df['heston_var'].mean():>9.2f}% {res_df['fbm_var'].mean():>9.2f}%")

    # ============================================================
    # 4. Error analysis
    # ============================================================
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS: Which model is closest to reality?")
    print("=" * 70)

    print(f"\n  {'Metric':<30} {'JD':>10} {'Heston':>10} {'fBM':>10}")
    print(f"  {'-'*65}")
    print(f"  {'Mean Absolute Error (pp)':<30} "
          f"{res_df['jd_error'].mean():>9.1f} "
          f"{res_df['heston_error'].mean():>9.1f} "
          f"{res_df['fbm_error'].mean():>9.1f}")
    print(f"  {'Median Absolute Error (pp)':<30} "
          f"{res_df['jd_error'].median():>9.1f} "
          f"{res_df['heston_error'].median():>9.1f} "
          f"{res_df['fbm_error'].median():>9.1f}")
    print(f"  {'Std of Error (pp)':<30} "
          f"{res_df['jd_error'].std():>9.1f} "
          f"{res_df['heston_error'].std():>9.1f} "
          f"{res_df['fbm_error'].std():>9.1f}")
    print(f"  {'Direction accuracy (%)':<30} "
          f"{(np.sign(res_df['jd_var']) == np.sign(res_df['real_var'])).mean()*100:>9.1f} "
          f"{(np.sign(res_df['heston_var']) == np.sign(res_df['real_var'])).mean()*100:>9.1f} "
          f"{(np.sign(res_df['fbm_var']) == np.sign(res_df['real_var'])).mean()*100:>9.1f}")

    # Best model per year
    print(f"\n  {'Year':>6} {'Best Model':>12} {'Error (pp)':>12}")
    print(f"  {'-'*35}")
    for _, row in res_df.iterrows():
        errors = {
            'JD': row['jd_error'],
            'Heston': row['heston_error'],
            'fBM': row['fbm_error']
        }
        best = min(errors, key=errors.get)
        print(f"  {int(row['year']):>6} {best:>12} {errors[best]:>11.1f}")

    # ============================================================
    # 5. Old Monte Carlo approach comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("OLD APPROACH vs NEW APPROACH")
    print("=" * 70)

    print("""
  OLD APPROACH (Monte Carlo):
    - Estimate parameters from ALL historical data
    - Simulate 5,000 paths forward
    - Report simulated VaR
    Problem: Simulated VaR (-62.9% to -69.5%) is 3-4x worse than
    real VaR (-13.9% to -21.8%). The pattern is correct but the
    magnitude is wrong.

  NEW APPROACH (Real Data Walk-Forward):
    - Train on data up to year Y
    - Test on real year Y+1 data
    - Report real VaR vs model-predicted VaR
    Advantage: Shows which model actually forecasts best on real data.
""")

    # Winner
    avg_errors = {
        'JD': res_df['jd_error'].mean(),
        'Heston': res_df['heston_error'].mean(),
        'fBM': res_df['fbm_error'].mean()
    }
    winner = min(avg_errors, key=avg_errors.get)
    print(f"  WINNER: {winner} (avg error: {avg_errors[winner]:.1f}pp)")

    # Save results
    save_csv(res_df, '05_real_vs_simulated', 'error_analysis.csv')

if __name__ == "__main__":
    with TeeOutput("05_real_vs_simulated", "real_vs_simulated.txt"):
        main()
