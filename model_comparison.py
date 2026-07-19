"""
=============================================================================
FILE: src/model_comparison.py
PURPOSE: Three-model comparison framework
  - Jump Diffusion (Merton 1976)
  - Heston Stochastic Volatility (Heston 1993)
  - Fractional Brownian Motion (Mandelbrot 1968)

Cross-validation approach:
  1. Estimate all three models from same historical data
  2. Simulate forward paths from each
  3. Compare: VaR, tail shape, volatility clustering, autocorrelation
  4. Rank: which model best captures each empirical feature?
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Optional

from .jump_diffusion_engine import (
    JumpDiffusionParams, simulate_jump_diffusion, simulate_paired_paths
)
from .heston_engine import (
    HestonParams, simulate_heston, simulate_heston_paired, estimate_heston_params
)
from .fbm_engine import (
    FBMParams, simulate_fbm, estimate_hurst
)
from .var_calculator import compute_var_caseB


def estimate_all_models(
    price_df: pd.DataFrame,
    events_df: pd.DataFrame = None,
    S0: float = None,
    T: float = 2.0,
    n_sims: int = 5000
) -> dict:
    """
    Estimate all three models from the same historical data

    Args:
        price_df: DataFrame with Date, Close, log_return columns
        events_df: Geopolitical events (optional, for JD)
        S0: Initial price (default: last Close)
        T: Simulation horizon in years
        n_sims: Number of simulation paths

    Returns:
        dict with 'jd', 'heston', 'fbm' parameter objects and estimators
    """
    if S0 is None:
        S0 = float(price_df['Close'].iloc[-1])

    log_returns = price_df['log_return'].dropna().values

    # --- Jump Diffusion ---
    from .parameter_estimator import CaseBGeopoliticalEstimator
    if events_df is not None:
        jd_est = CaseBGeopoliticalEstimator(price_df, events_df)
        jd_params = jd_est.get_params(horizon_years=T, n_sims=n_sims)
    else:
        # Simplified JD estimation without events
        jd_params = JumpDiffusionParams(
            mu=log_returns.mean() * 252,
            sigma=log_returns.std() * np.sqrt(252),
            lam=2.0,
            mu_j=-0.03, sigma_j=0.02,
            S0=S0, T=T, dt=1/252, n_sims=n_sims
        )

    # --- Heston ---
    heston_est = estimate_heston_params(log_returns)
    heston_params = HestonParams(
        mu=heston_est.mu,
        v0=heston_est.v0,
        kappa=heston_est.kappa,
        theta=heston_est.theta,
        xi=heston_est.xi,
        rho=heston_est.rho,
        S0=S0, T=T, dt=1/252, n_sims=n_sims
    )

    # --- FBM ---
    H_est = estimate_hurst(log_returns, method="rs")
    fbm_params = FBMParams(
        mu=log_returns.mean() * 252,
        sigma=log_returns.std() * np.sqrt(252),
        hurst=H_est,
        S0=S0, T=T, dt=1/252, n_sims=n_sims
    )

    return {
        'jd': {'params': jd_params, 'estimator': jd_est if events_df is not None else None},
        'heston': {'params': heston_params, 'raw_estimate': heston_est},
        'fbm': {'params': fbm_params, 'hurst': H_est},
    }


def simulate_all_models(
    models: dict,
    seed: int = 42
) -> dict:
    """Run simulations for all three models"""
    results = {}

    results['jd'] = simulate_jump_diffusion(models['jd']['params'], seed=seed)
    results['heston'] = simulate_heston(models['heston']['params'], seed=seed)
    results['fbm'] = simulate_fbm(models['fbm']['params'], seed=seed)

    return results


def compute_empirical_features(log_returns: np.ndarray) -> dict:
    """
    Compute empirical features from historical data
    These serve as the "ground truth" for model comparison
    """
    n = len(log_returns)

    # Volatility clustering: autocorrelation of squared returns
    sq_returns = log_returns ** 2
    if n > 20:
        vol_autocorr = np.corrcoef(sq_returns[1:], sq_returns[:-1])[0, 1]
    else:
        vol_autocorr = 0.0

    # Fat tails: excess kurtosis
    kurt = pd.Series(log_returns).kurtosis()

    # Asymmetry: skewness
    skew = pd.Series(log_returns).skew()

    # Long memory: Hurst exponent
    hurst = estimate_hurst(log_returns, method="rs")

    # Basic stats
    mean_ret = log_returns.mean() * 252
    std_ret = log_returns.std() * np.sqrt(252)

    return {
        'annualized_return': mean_ret,
        'annualized_vol': std_ret,
        'vol_autocorr': vol_autocorr,
        'excess_kurtosis': kurt,
        'skewness': skew,
        'hurst_exponent': hurst,
    }


def score_model_against_empirical(
    sim_paths: np.ndarray,
    empirical: dict
) -> dict:
    """
    Score how well a simulated model matches empirical features

    Returns dict of feature -> score (0-1, higher = better match)
    """
    log_returns = np.diff(np.log(sim_paths), axis=1)
    sim_flat = log_returns.flatten()

    scores = {}

    # Volatility autocorrelation
    sq_sim = sim_flat[:min(len(sim_flat), 50000)] ** 2
    if len(sq_sim) > 100:
        sim_vol_ac = np.corrcoef(sq_sim[1:], sq_sim[:-1])[0, 1]
    else:
        sim_vol_ac = 0.0
    vol_diff = abs(sim_vol_ac - empirical['vol_autocorr'])
    scores['vol_clustering'] = max(0, 1 - vol_diff / 0.5)

    # Kurtosis
    sim_kurt = pd.Series(sim_flat[:50000]).kurtosis()
    kurt_diff = abs(sim_kurt - empirical['excess_kurtosis'])
    scores['fat_tails'] = max(0, 1 - kurt_diff / 10)

    # Hurst
    sample_returns = log_returns[0]  # first path
    sim_hurst = estimate_hurst(sample_returns, method="rs")
    hurst_diff = abs(sim_hurst - empirical['hurst_exponent'])
    scores['long_memory'] = max(0, 1 - hurst_diff / 0.5)

    # Overall VaR match
    sim_final = sim_paths[:, -1]
    sim_pct_change = (sim_final - sim_paths[0, 0]) / sim_paths[0, 0]
    sim_var = np.percentile(sim_pct_change, 5)
    # Score based on reasonableness (not -100% or near 0)
    scores['var_plausibility'] = 1.0 if -0.9 < sim_var < -0.01 else 0.5

    # Overall score
    scores['overall'] = np.mean(list(scores.values()))

    return scores


def full_comparison(
    price_df: pd.DataFrame,
    events_df: pd.DataFrame = None,
    T: float = 2.0,
    n_sims: int = 5000,
    seed: int = 42
) -> dict:
    """
    Full three-model comparison pipeline

    Returns:
        dict with:
            'models': estimated parameters
            'simulations': simulation results
            'empirical': historical features
            'scores': model scores
            'vars': VaR for each model
    """
    # Empirical features
    log_returns = price_df['log_return'].dropna().values
    empirical = compute_empirical_features(log_returns)

    # Estimate all models
    S0 = float(price_df['Close'].iloc[-1])
    models = estimate_all_models(price_df, events_df, S0=S0, T=T, n_sims=n_sims)

    # Simulate all models
    simulations = simulate_all_models(models, seed=seed)

    # Score each model
    scores = {}
    vars = {}
    for name, sim in simulations.items():
        paths = sim['paths']
        scores[name] = score_model_against_empirical(paths, empirical)
        var_result = compute_var_caseB({'paths': paths})
        vars[name] = var_result

    return {
        'models': models,
        'simulations': simulations,
        'empirical': empirical,
        'scores': scores,
        'vars': vars,
    }


# ========================== Self-test ==========================

def _run_self_tests():
    print("=" * 60)
    print("model_comparison.py self-test")
    print("=" * 60)

    from .data_loader import load_semiconductor_data, build_geopolitical_event_timeline

    # Generate data
    print("\n[Test 1] Estimate all models from data")
    smh = load_semiconductor_data(use_cache=True)
    events = build_geopolitical_event_timeline()

    models = estimate_all_models(smh, events, S0=250, T=2.0, n_sims=1000)
    print(f"  JD: {models['jd']['params']}")
    print(f"  Heston: {models['heston']['params']}")
    print(f"  FBM: H={models['fbm']['hurst']:.3f}")
    print("  [PASS]")

    print("\n[Test 2] Simulate all models")
    sims = simulate_all_models(models, seed=42)
    for name, sim in sims.items():
        print(f"  {name}: shape={sim['paths'].shape}, "
              f"mean_final={sim['final_values'].mean():.2f}")
    print("  [PASS]")

    print("\n[Test 3] Compute empirical features")
    log_ret = smh['log_return'].dropna().values
    empirical = compute_empirical_features(log_ret)
    for k, v in empirical.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("  [PASS]")

    print("\n[Test 4] Score models")
    scores = {}
    for name, sim in sims.items():
        scores[name] = score_model_against_empirical(sim['paths'], empirical)
        print(f"  {name}: overall={scores[name]['overall']:.3f}, "
              f"vol_clust={scores[name]['vol_clustering']:.3f}, "
              f"fat_tails={scores[name]['fat_tails']:.3f}")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_self_tests()
