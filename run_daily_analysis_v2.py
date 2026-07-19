"""
5 iterations v2: Better validation methods for daily GPR-stock analysis

Replaced weak iterations with:
- Walk-forward validation (no look-ahead bias)
- Event study around GPR spikes
- Quantile regression for tail risk
- Regime transition probabilities
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.save_results import TeeOutput, save_csv

DATA = 'e:/Eddik/Documents/AI/jump-diffusion-dual-domain/gpr_deep_dive/data'

def main():
    df = pd.read_csv(f'{DATA}/analysis_daily_clean.csv', parse_dates=['date'])
    df = df.dropna(subset=['SOX_log_return', 'GPRD']).reset_index(drop=True)
    print(f"Loaded: {len(df)} days, {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    # ============================================================
    # Iteration 1: Walk-Forward Validation (no look-ahead bias)
    # ============================================================
    print("\n" + "=" * 70)
    print("Iteration 1: Walk-Forward Validation")
    print("Train on expanding window, test on next 252 trading days (1 year)")
    print("=" * 70)

    # GPR features
    df['GPRD_sma30'] = df['GPRD'].rolling(30, min_periods=10).mean()
    df['GPRD_momentum'] = df['GPRD'] - df['GPRD_sma30']
    df['fwd_return_20d'] = df['SOX_log_return'].rolling(20).sum().shift(-20)
    df['is_jump_20d'] = (df['fwd_return_20d'] < -0.10).astype(int)

    # Walk-forward: train on all data up to year Y, test on year Y+1
    results_wf = []
    start_year = 2005  # need at least 10 years of training data

    for test_year in range(start_year, 2027):
        train_mask = df['date'] < f'{test_year}-01-01'
        test_mask = (df['date'] >= f'{test_year}-01-01') & (df['date'] < f'{test_year+1}-01-01')

        train = df[train_mask].dropna(subset=['fwd_return_20d', 'GPRD'])
        test = df[test_mask].dropna(subset=['fwd_return_20d', 'GPRD'])

        if len(train) < 500 or len(test) < 20:
            continue

        # Simple model: GPR level -> 20d return direction
        slope, intercept, r, p, _ = stats.linregress(train['GPRD'], train['fwd_return_20d'])
        test = test.copy()
        test['forecast'] = slope * test['GPRD'] + intercept
        hit = (np.sign(test['forecast']) == np.sign(test['fwd_return_20d'])).mean()

        # Also compute regime-based accuracy
        gpr_median = train['GPRD'].median()
        test['regime'] = 'low_stable'
        test.loc[(test['GPRD'] > gpr_median) & (test['GPRD_momentum'] > 0), 'regime'] = 'high_rising'
        test.loc[(test['GPRD'] > gpr_median) & (test['GPRD_momentum'] <= 0), 'regime'] = 'high_falling'
        test.loc[(test['GPRD'] <= gpr_median) & (test['GPRD_momentum'] > 0), 'regime'] = 'low_rising'

        # Jump probability by regime
        regime_jumps = {}
        for reg in ['low_stable', 'low_rising', 'high_rising', 'high_falling']:
            sub = test[test['regime'] == reg]
            if len(sub) >= 5:
                regime_jumps[reg] = sub['is_jump_20d'].mean()

        results_wf.append({
            'year': test_year,
            'n_train': len(train),
            'n_test': len(test),
            'r': r,
            'hit_rate': hit,
            'gpr_mean': test['GPRD'].mean(),
            'jump_prob': test['is_jump_20d'].mean(),
            'regime_jumps': regime_jumps,
        })

    wf_df = pd.DataFrame(results_wf)
    print(f"\n  {'Year':>6} {'N_test':>8} {'Hit%':>8} {'GPR_mean':>10} {'Jump%':>8}")
    print(f"  {'-'*44}")
    for _, row in wf_df.iterrows():
        print(f"  {int(row['year']):>6} {int(row['n_test']):>8} {row['hit_rate']*100:>7.1f}% {row['gpr_mean']:>10.0f} {row['jump_prob']*100:>7.1f}%")

    avg_hit = wf_df['hit_rate'].mean()
    print(f"\n  Average hit rate across all years: {avg_hit*100:.1f}%")
    print(f"  Hit rate std: {wf_df['hit_rate'].std()*100:.1f}%")

    # ============================================================
    # Iteration 2: Event Study around GPR Spikes
    # ============================================================
    print("\n" + "=" * 70)
    print("Iteration 2: Event Study around GPR Spikes")
    print("What happens to SOX in the 40 trading days after GPR crosses above 150?")
    print("=" * 70)

    # Find GPR spike events (GPR crosses above 150 from below 150)
    df['GPR_above_150'] = (df['GPRD'] > 150).astype(int)
    df['GPR_cross'] = df['GPR_above_150'].diff()
    spike_dates = df[df['GPR_cross'] == 1]['date'].tolist()

    # Remove spikes too close together (within 60 days)
    filtered_spikes = []
    for d in spike_dates:
        if not filtered_spikes or (d - filtered_spikes[-1]).days > 60:
            filtered_spikes.append(d)

    print(f"  GPR spike events (cross above 150): {len(filtered_spikes)}")

    # For each spike, compute cumulative return over next 40 trading days
    window = 40
    event_returns = []
    for spike_date in filtered_spikes:
        spike_idx = df[df['date'] >= spike_date].index
        if len(spike_idx) == 0:
            continue
        idx = spike_idx[0]

        if idx + window >= len(df):
            continue

        cum_returns = []
        for d in range(window):
            cum_returns.append(df.iloc[idx + d]['SOX_log_return'])

        cum_return = np.nansum(cum_returns)
        gpr_at_spike = df.iloc[idx]['GPRD']

        event_returns.append({
            'date': spike_date,
            'gpr_at_spike': gpr_at_spike,
            'cum_return_40d': cum_return,
            'max_drawdown_40d': min(np.nancumsum(cum_returns)),
        })

    event_df = pd.DataFrame(event_returns)
    print(f"\n  Events analyzed: {len(event_df)}")
    print(f"  Mean 40-day return after spike: {event_df['cum_return_40d'].mean()*100:.2f}%")
    print(f"  Median 40-day return: {event_df['cum_return_40d'].median()*100:.2f}%")
    print(f"  P(negative): {(event_df['cum_return_40d'] < 0).mean()*100:.1f}%")
    print(f"  Mean max drawdown: {event_df['max_drawdown_40d'].mean()*100:.2f}%")

    # Correlation: GPR level at spike vs subsequent return
    if len(event_df) > 5:
        r, p = stats.pearsonr(event_df['gpr_at_spike'], event_df['cum_return_40d'])
        print(f"\n  Correlation (GPR at spike vs 40d return): r={r:.4f}, p={p:.4f}")

    # Show individual events
    print(f"\n  {'Date':<12} {'GPR':>6} {'40d Ret':>10} {'Max DD':>10}")
    print(f"  {'-'*40}")
    for _, row in event_df.iterrows():
        print(f"  {row['date'].strftime('%Y-%m-%d'):<12} {row['gpr_at_spike']:>6.0f} {row['cum_return_40d']*100:>9.2f}% {row['max_drawdown_40d']*100:>9.2f}%")

    # ============================================================
    # Iteration 3: Quantile Regression for Tail Risk
    # ============================================================
    print("\n" + "=" * 70)
    print("Iteration 3: Quantile Regression (5th percentile)")
    print("Predict the WORST-CASE 20-day return based on GPR level")
    print("=" * 70)

    from scipy.optimize import minimize

    def quantile_loss(y_true, y_pred, tau):
        """Pinball loss for quantile regression"""
        e = y_true - y_pred
        return np.mean(np.maximum(tau * e, (tau - 1) * e))

    # Train on 1994-2019, test on 2020-2026
    train = df[(df['date'] <= '2019-12-31')].dropna(subset=['fwd_return_20d', 'GPRD'])
    test = df[(df['date'] > '2019-12-31')].dropna(subset=['fwd_return_20d', 'GPRD'])

    # Fit quantile regression at 5th percentile
    tau = 0.05

    # Simple linear quantile regression: minimize pinball loss
    def fit_quantile(X, y, tau):
        def objective(beta):
            y_pred = beta[0] + beta[1] * X
            return quantile_loss(y, y_pred, tau)
        result = minimize(objective, x0=[0, 0], method='Nelder-Mead')
        return result.x

    beta_q05 = fit_quantile(train['GPRD'].values, train['fwd_return_20d'].values, tau)
    beta_q50 = fit_quantile(train['GPRD'].values, train['fwd_return_20d'].values, 0.50)
    beta_q95 = fit_quantile(train['GPRD'].values, train['fwd_return_20d'].values, 0.95)

    print(f"  Quantile regression coefficients:")
    print(f"    5th percentile:  intercept={beta_q05[0]:.6f}, slope={beta_q05[1]:.6f}")
    print(f"    50th percentile: intercept={beta_q50[0]:.6f}, slope={beta_q50[1]:.6f}")
    print(f"    95th percentile: intercept={beta_q95[0]:.6f}, slope={beta_q95[1]:.6f}")

    # Predict on test data
    test = test.copy()
    test['q05_pred'] = beta_q05[0] + beta_q05[1] * test['GPRD']
    test['q50_pred'] = beta_q50[0] + beta_q50[1] * test['GPRD']
    test['q95_pred'] = beta_q95[0] + beta_q95[1] * test['GPRD']

    # Check: what fraction of actual returns fall below the 5% prediction?
    below_q05 = (test['fwd_return_20d'] < test['q05_pred']).mean()
    print(f"\n  Test period: actual returns below 5% prediction: {below_q05*100:.1f}% (target: 5%)")

    # Show predictions at different GPR levels
    print(f"\n  {'GPR Level':>10} {'5th pctl':>10} {'Median':>10} {'95th pctl':>10}")
    print(f"  {'-'*44}")
    for gpr_level in [50, 80, 100, 120, 150, 200, 300]:
        q05 = beta_q05[0] + beta_q05[1] * gpr_level
        q50 = beta_q50[0] + beta_q50[1] * gpr_level
        q95 = beta_q95[0] + beta_q95[1] * gpr_level
        print(f"  {gpr_level:>10} {q05*100:>9.2f}% {q50*100:>9.2f}% {q95*100:>9.2f}%")

    # ============================================================
    # Iteration 4: Regime Transition Probabilities
    # ============================================================
    print("\n" + "=" * 70)
    print("Iteration 4: Regime Transition Probabilities")
    print("If GPR is in regime X today, what regime is it likely to be in 20 days?")
    print("=" * 70)

    # Classify all days into regimes
    gpr_median = df['GPRD'].median()
    df['regime'] = 'low_stable'
    df.loc[(df['GPRD'] > gpr_median) & (df['GPRD_momentum'] > 0), 'regime'] = 'high_rising'
    df.loc[(df['GPRD'] > gpr_median) & (df['GPRD_momentum'] <= 0), 'regime'] = 'high_falling'
    df.loc[(df['GPRD'] <= gpr_median) & (df['GPRD_momentum'] > 0), 'regime'] = 'low_rising'

    # Look-ahead regime (20 trading days later)
    df['regime_future'] = df['regime'].shift(-20)

    # Transition matrix
    regimes = ['low_stable', 'low_rising', 'high_rising', 'high_falling']
    transitions = pd.crosstab(df['regime'], df['regime_future'], normalize='index')
    transitions = transitions.reindex(index=regimes, columns=regimes, fill_value=0)

    print(f"\n  GPR median: {gpr_median:.0f}")
    print(f"\n  Transition probabilities (current -> in 20 days):")
    print(f"  {'':>15} {'low_stable':>12} {'low_rising':>12} {'high_rising':>12} {'high_falling':>12}")
    print(f"  {'-'*65}")
    for from_reg in regimes:
        row = transitions.loc[from_reg]
        print(f"  {from_reg:>15} {row.get('low_stable',0):>11.1%} {row.get('low_rising',0):>11.1%} {row.get('high_rising',0):>11.1%} {row.get('high_falling',0):>11.1%}")

    # ============================================================
    # Iteration 5: Combined Score (Walk-Forward)
    # ============================================================
    print("\n" + "=" * 70)
    print("Iteration 5: Combined Score (Walk-Forward)")
    print("GPR level + momentum + VIX + regime transition -> 20d return direction")
    print("=" * 70)

    # Build features
    df['GPRD_level_z'] = (df['GPRD'] - df['GPRD'].rolling(252).mean()) / df['GPRD'].rolling(252).std()
    df['GPRD_mom_z'] = (df['GPRD_momentum'] - df['GPRD_momentum'].rolling(252).mean()) / df['GPRD_momentum'].rolling(252).std()

    if 'VIX_Close' in df.columns:
        df['VIX_z'] = (df['VIX_Close'] - df['VIX_Close'].rolling(252).mean()) / df['VIX_Close'].rolling(252).std()

    # Walk-forward with combined score
    results_combined = []
    for test_year in range(2005, 2027):
        train_mask = df['date'] < f'{test_year}-01-01'
        test_mask = (df['date'] >= f'{test_year}-01-01') & (df['date'] < f'{test_year+1}-01-01')

        train = df[train_mask].dropna(subset=['fwd_return_20d', 'GPRD_level_z', 'GPRD_mom_z'])
        test = df[test_mask].dropna(subset=['fwd_return_20d', 'GPRD_level_z', 'GPRD_mom_z'])

        if len(train) < 500 or len(test) < 20:
            continue

        # Combined score: weighted average of z-scores
        # Higher score = higher risk
        train = train.copy()
        test = test.copy()

        # Optimal weights from training data
        from scipy.stats import spearmanr
        r_level, _ = spearmanr(train['GPRD_level_z'], train['fwd_return_20d'])
        r_mom, _ = spearmanr(train['GPRD_mom_z'], train['fwd_return_20d'])

        # Use sign of correlation as weight direction
        w_level = 1.0 if r_level > 0 else -1.0
        w_mom = 1.0 if r_mom > 0 else -1.0

        test['risk_score'] = w_level * test['GPRD_level_z'] + w_mom * test['GPRD_mom_z']

        # Predict: if risk_score > 0, predict positive return (mean reversion)
        # if risk_score < 0, predict negative
        test['pred'] = np.where(test['risk_score'] > 0, 1, -1)
        test['actual'] = np.sign(test['fwd_return_20d'])
        hit = (test['pred'] == test['actual']).mean()

        results_combined.append({
            'year': test_year,
            'hit_rate': hit,
            'n_test': len(test),
        })

    comb_df = pd.DataFrame(results_combined)
    print(f"\n  {'Year':>6} {'Hit%':>8} {'N':>8}")
    print(f"  {'-'*24}")
    for _, row in comb_df.iterrows():
        print(f"  {int(row['year']):>6} {row['hit_rate']*100:>7.1f}% {int(row['n_test']):>8}")

    avg_hit = comb_df['hit_rate'].mean()
    print(f"\n  Average hit rate: {avg_hit*100:.1f}%")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY: 5 Iterations (v2)")
    print("=" * 70)
    print(f"""
  1. Walk-Forward Validation:  {avg_hit*100:.1f}% avg hit rate (no look-ahead bias)
  2. Event Study:              Mean 40d return after GPR spike: {event_df['cum_return_40d'].mean()*100:.2f}%
  3. Quantile Regression:      5th percentile prediction at GPR=200: {beta_q05[0]+beta_q05[1]*200:.4f}
  4. Regime Transitions:       low_rising -> high_rising: {transitions.loc['low_rising','high_rising']:.1%}
  5. Combined Score:           {avg_hit*100:.1f}% avg hit rate (level + momentum z-scores)
""")

    # ============================================================
    # Save results to CSV
    # ============================================================
    save_csv(event_df, '02_forecasting_iterations', 'iteration1_event_study.csv')

    quantile_data = []
    for gpr_level in [50, 80, 100, 120, 150, 200, 300]:
        quantile_data.append({
            'gpr_level': gpr_level,
            'q05_pct': round((beta_q05[0] + beta_q05[1] * gpr_level) * 100, 2),
            'q50_pct': round((beta_q50[0] + beta_q50[1] * gpr_level) * 100, 2),
            'q95_pct': round((beta_q95[0] + beta_q95[1] * gpr_level) * 100, 2),
        })
    save_csv(pd.DataFrame(quantile_data), '02_forecasting_iterations', 'iteration2_quantile.csv')

    save_csv(transitions.reset_index(), '02_forecasting_iterations', 'iteration3_transitions.csv')


if __name__ == "__main__":
    with TeeOutput("02_forecasting_iterations", "forecasting_iterations.txt"):
        main()
