"""
Test bidirectional jump model with real data.
Compare negative-only vs double-exponential (both directions) jumps.
"""
import numpy as np
import pandas as pd
from scipy import stats
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.jump_diffusion_engine import JumpDiffusionParams, simulate_jump_diffusion
from src.var_calculator import compute_var_caseB
from src.save_results import TeeOutput, save_csv

DATA = os.path.join(os.path.dirname(__file__), 'data')

def main():
    # Load real daily data
    df = pd.read_csv(f'{DATA}/analysis_daily_clean.csv', parse_dates=['date'])
    df = df.dropna(subset=['SOX_log_return', 'GPRD']).reset_index(drop=True)
    df['GPRD_sma30'] = df['GPRD'].rolling(30, min_periods=10).mean()
    df['GPRD_momentum'] = df['GPRD'] - df['GPRD_sma30']
    df['fwd_return_5d'] = df['SOX_log_return'].rolling(5).sum().shift(-5)

    analysis = df[df['date'] >= '1994-06-01'].dropna(subset=['fwd_return_5d']).copy()

    # ============================================================
    # 1. Identify both positive and negative jumps
    # ============================================================
    print("=" * 70)
    print("STEP 1: Identify Both Positive and Negative Jumps")
    print("=" * 70)

    analysis['is_neg_jump'] = (analysis['fwd_return_5d'] < -0.05).astype(int)
    analysis['is_pos_jump'] = (analysis['fwd_return_5d'] > 0.05).astype(int)
    analysis['is_any_jump'] = ((analysis['is_neg_jump'] == 1) | (analysis['is_pos_jump'] == 1)).astype(int)

    print(f"  Total trading days: {len(analysis)}")
    print(f"  Negative jumps (<-5% in 5d): {analysis['is_neg_jump'].sum()} ({analysis['is_neg_jump'].mean()*100:.2f}%)")
    print(f"  Positive jumps (>+5% in 5d): {analysis['is_pos_jump'].sum()} ({analysis['is_pos_jump'].mean()*100:.2f}%)")

    neg_jumps = analysis.loc[analysis['is_neg_jump'] == 1, 'fwd_return_5d']
    pos_jumps = analysis.loc[analysis['is_pos_jump'] == 1, 'fwd_return_5d']

    print(f"\n  Negative jumps: mean={neg_jumps.mean()*100:.2f}%, std={neg_jumps.std()*100:.2f}%")
    print(f"  Positive jumps: mean={pos_jumps.mean()*100:.2f}%, std={pos_jumps.std()*100:.2f}%")

    # ============================================================
    # 2. Fit distributions
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 2: Fit Jump Distributions")
    print("=" * 70)

    total_jumps = len(neg_jumps) + len(pos_jumps)
    p_pos = len(pos_jumps) / total_jumps
    p_neg = len(neg_jumps) / total_jumps

    neg_mu = neg_jumps.mean()
    neg_sigma = neg_jumps.std(ddof=1)
    pos_mu = pos_jumps.mean()
    pos_sigma = pos_jumps.std(ddof=1)

    n_years = len(analysis) / 252
    lambda_neg = len(neg_jumps) / n_years
    lambda_pos = len(pos_jumps) / n_years
    lambda_total = total_jumps / n_years

    print(f"  P(positive): {p_pos:.3f}, P(negative): {p_neg:.3f}")
    print(f"  Negative: N({neg_mu:.4f}, {neg_sigma:.4f}^2), freq={lambda_neg:.2f}/yr")
    print(f"  Positive: N({pos_mu:.4f}, {pos_sigma:.4f}^2), freq={lambda_pos:.2f}/yr")
    print(f"  Total jump freq: {lambda_total:.2f}/yr")

    # ============================================================
    # 3. Monte Carlo: 3 models
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 3: Monte Carlo Simulation - 3 Models")
    print("=" * 70)

    normal_returns = analysis.loc[analysis['is_any_jump'] == 0, 'SOX_log_return']
    mu = normal_returns.mean() * 252
    sigma = normal_returns.std(ddof=1) * np.sqrt(252)
    S0 = float(analysis['SOX_Close'].dropna().iloc[-1])
    T, dt, n_sims = 2.0, 1/252, 5000
    n_steps = int(T / dt)

    print(f"  Base: mu={mu:.4f}, sigma={sigma:.4f}, S0=${S0:.2f}")

    # Model 1: Negative only
    print("\n  [Model 1] Negative jumps only")
    p1 = JumpDiffusionParams(mu=mu, sigma=sigma, lam=lambda_neg,
                              mu_j=neg_mu, sigma_j=neg_sigma,
                              S0=S0, T=T, dt=dt, n_sims=n_sims)
    r1 = simulate_jump_diffusion(p1, seed=42)
    v1 = compute_var_caseB(r1)
    print(f"    VaR={v1.var_pct*100:.2f}%, CVaR={v1.cvar_pct*100:.2f}%, Median=${v1.median_final:,.0f}")

    # Model 2: Bidirectional
    print("\n  [Model 2] Bidirectional jumps")
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((n_sims, n_steps))
    jump_counts = rng.poisson(lambda_total * dt, (n_sims, n_steps))
    max_j = int(jump_counts.max())
    jump_sums = np.zeros((n_sims, n_steps))

    for k in range(max_j):
        mask = (jump_counts > k)
        direction = rng.binomial(1, p_pos, (n_sims, n_steps))
        pos_sizes = rng.normal(pos_mu, pos_sigma, (n_sims, n_steps))
        neg_sizes = rng.normal(abs(neg_mu), neg_sigma, (n_sims, n_steps))
        sizes = np.where(direction == 1, pos_sizes, -neg_sizes)
        jump_sums[mask] += sizes[mask]

    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_ret = drift + diffusion + jump_sums
    cum_log = np.zeros((n_sims, n_steps + 1))
    cum_log[:, 1:] = np.cumsum(log_ret, axis=1)
    paths_bi = S0 * np.exp(cum_log)

    final_bi = paths_bi[:, -1]
    pct_bi = (final_bi - S0) / S0
    var_bi = np.percentile(pct_bi, 5) * 100
    cvar_bi = pct_bi[pct_bi <= np.percentile(pct_bi, 5)].mean() * 100
    print(f"    VaR={var_bi:.2f}%, CVaR={cvar_bi:.2f}%, Median=${np.median(final_bi):,.0f}")

    # Model 3: Positive only
    print("\n  [Model 3] Positive jumps only")
    p3 = JumpDiffusionParams(mu=mu, sigma=sigma, lam=lambda_pos,
                              mu_j=pos_mu, sigma_j=pos_sigma,
                              S0=S0, T=T, dt=dt, n_sims=n_sims)
    r3 = simulate_jump_diffusion(p3, seed=42)
    v3 = compute_var_caseB(r3)
    print(f"    VaR={v3.var_pct*100:.2f}%, CVaR={v3.cvar_pct*100:.2f}%, Median=${v3.median_final:,.0f}")

    # ============================================================
    # 4. Comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 4: Comparison Summary")
    print("=" * 70)
    print(f"\n  {'Model':<30} {'VaR(95%)':>10} {'CVaR(95%)':>10} {'Median':>12}")
    print(f"  {'-'*65}")
    print(f"  {'Neg jumps only':<30} {v1.var_pct*100:>9.2f}% {v1.cvar_pct*100:>9.2f}% ${v1.median_final:>11,.0f}")
    print(f"  {'Bidirectional':<30} {var_bi:>9.2f}% {cvar_bi:>9.2f}% ${np.median(final_bi):>11,.0f}")
    print(f"  {'Pos jumps only':<30} {v3.var_pct*100:>9.2f}% {v3.cvar_pct*100:>9.2f}% ${v3.median_final:>11,.0f}")

    # ============================================================
    # 5. Regime-specific jump direction
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 5: Jump Direction by GPR Regime")
    print("=" * 70)

    gpr_median = analysis['GPRD'].median()

    def get_regime(row):
        h = row['GPRD'] > gpr_median
        r = row['GPRD_momentum'] > 0
        if h and r: return 'high_rising'
        if h and not r: return 'high_falling'
        if not h and r: return 'low_rising'
        return 'low_stable'

    analysis['regime'] = analysis.apply(get_regime, axis=1)

    print(f"\n  GPR median: {gpr_median:.0f}")
    print(f"\n  {'Regime':<15} {'Days':>6} {'Neg%':>8} {'Pos%':>8} {'Net Bias':>10}")
    print(f"  {'-'*50}")

    for regime in ['low_stable', 'low_rising', 'high_rising', 'high_falling']:
        sub = analysis[analysis['regime'] == regime]
        neg_pct = sub['is_neg_jump'].mean() * 100
        pos_pct = sub['is_pos_jump'].mean() * 100
        net = pos_pct - neg_pct
        bias = 'NEG' if net < 0 else 'POS'
        print(f"  {regime:<15} {len(sub):>6} {neg_pct:>7.1f}% {pos_pct:>7.1f}% {net:>+8.1f}% {bias}")

    # Save regime table as CSV
    regime_data = []
    for regime in ['low_stable', 'low_rising', 'high_rising', 'high_falling']:
        sub = analysis[analysis['regime'] == regime]
        neg_pct = sub['is_neg_jump'].mean() * 100
        pos_pct = sub['is_pos_jump'].mean() * 100
        regime_data.append({
            'regime': regime, 'days': len(sub),
            'neg_jump_pct': round(neg_pct, 1),
            'pos_jump_pct': round(pos_pct, 1),
            'net_bias': round(pos_pct - neg_pct, 1)
        })
    save_csv(pd.DataFrame(regime_data), '03_bidirectional_jumps', 'jump_direction_regime.csv')

if __name__ == "__main__":
    with TeeOutput("03_bidirectional_jumps", "bidirectional_jumps.txt"):
        main()
