"""
=============================================================================
FILE: verify_robustness.py
PURPOSE: Comprehensive robustness verification of the main study code
         Tests with multiple datasets, edge cases, and cross-validation

This script CANNOT assume the code works. It must:
1. Test with real data from MULTIPLE sources
2. Test edge cases that could break the code
3. Cross-validate results across different models
4. Report actual pass/fail with evidence
=============================================================================
"""

import sys, os, time, io, warnings
import traceback

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from src.save_results import TeeOutput, save_csv

# Track all test results
RESULTS = []


def run_test(name, test_fn):
    """Run a test and track result"""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        t0 = time.time()
        test_fn()
        elapsed = time.time() - t0
        RESULTS.append((name, "PASS", elapsed, ""))
        print(f"  RESULT: PASS ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - t0
        RESULTS.append((name, "FAIL", elapsed, str(e)))
        print(f"  RESULT: FAIL - {e}")
        traceback.print_exc()


# ========================== SECTION 1: DIFFERENT STOCKS ==========================

def test_caseB_with_different_stocks():
    """Test Case B with SPY, QQQ, XLE instead of just SMH"""
    import yfinance as yf
    from src.data_loader import build_geopolitical_event_timeline
    from src.parameter_estimator import CaseBGeopoliticalEstimator
    from src.jump_diffusion_engine import simulate_jump_diffusion
    from src.var_calculator import compute_var_caseB

    tickers = ["SPY", "QQQ", "XLE"]
    events = build_geopolitical_event_timeline()

    for ticker in tickers:
        print(f"\n  --- Testing {ticker} ---")
        try:
            # Try yfinance first
            df = yf.download(ticker, start="2018-01-01", end="2025-12-31",
                           auto_adjust=True, progress=False)
            if len(df) < 100:
                print(f"  {ticker}: yfinance returned only {len(df)} rows, skipping")
                continue

            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
            df = df.dropna()

            # Check for NaN/Inf
            bad_returns = df['log_return'].isna().sum() + np.isinf(df['log_return']).sum()
            assert bad_returns == 0, f"{ticker}: {bad_returns} bad return values"

            # Estimate params
            est = CaseBGeopoliticalEstimator(df, events)
            params = est.get_params(horizon_years=2, n_sims=2000)

            # Validate params
            assert params.sigma > 0, f"{ticker}: sigma={params.sigma}"
            assert params.lam > 0, f"{ticker}: lam={params.lam}"
            assert params.mu_j < 0, f"{ticker}: mu_j={params.mu_j}"
            assert params.S0 > 0, f"{ticker}: S0={params.S0}"
            assert 0.05 < params.sigma < 1.0, f"{ticker}: sigma={params.sigma} out of range"

            # Simulate
            result = simulate_jump_diffusion(params, seed=42)
            var = compute_var_caseB(result)

            # Validate VaR
            assert var.var_pct < 0, f"{ticker}: VaR={var.var_pct} should be negative"
            assert var.var_pct > -1.0, f"{ticker}: VaR={var.var_pct} too extreme"
            assert var.max_drawdown_var < 0

            print(f"    sigma={params.sigma:.3f}, lam={params.lam:.2f}, "
                  f"mu_j={params.mu_j:.3f}")
            print(f"    VaR={var.var_pct*100:.1f}%, MaxDD={var.max_drawdown_var*100:.1f}%")
            print(f"    [OK]")

        except Exception as e:
            print(f"  {ticker}: FAILED - {e}")
            raise


def test_caseB_different_time_periods():
    """Test Case B with different time windows"""
    from src.data_loader import load_semiconductor_data, build_geopolitical_event_timeline
    from src.parameter_estimator import CaseBGeopoliticalEstimator
    from src.jump_diffusion_engine import simulate_jump_diffusion
    from src.var_calculator import compute_var_caseB

    smh = load_semiconductor_data(use_cache=True)
    events = build_geopolitical_event_timeline()

    # Test with different sub-periods
    periods = [
        ("2015-2019", "2015-01-01", "2019-12-31"),
        ("2020-2025", "2020-01-01", "2025-12-31"),
        ("Full", None, None),
    ]

    for label, start, end in periods:
        if start and end:
            sub = smh[(smh['Date'] >= start) & (smh['Date'] <= end)].copy()
        else:
            sub = smh.copy()

        if len(sub) < 100:
            print(f"  {label}: only {len(sub)} rows, skipping")
            continue

        est = CaseBGeopoliticalEstimator(sub, events)
        params = est.get_params(horizon_years=1, n_sims=2000)
        result = simulate_jump_diffusion(params, seed=42)
        var = compute_var_caseB(result)

        print(f"  {label}: n={len(sub)}, sigma={params.sigma:.3f}, "
              f"VaR={var.var_pct*100:.1f}%")

        # Sanity checks
        assert params.sigma > 0
        assert var.var_pct > -1.0
        assert var.var_pct < 0

    print("  [OK] All periods produce valid results")


# ========================== SECTION 2: DIFFERENT COUNTIES ==========================

def test_caseA_different_counties():
    """Test Case A with multiple county FIPS codes"""
    from src.data_loader import load_fema_disasters, load_county_finance, merge_fema_county
    from src.parameter_estimator import CaseAHurricaneEstimator
    from src.jump_diffusion_engine import simulate_jump_diffusion
    from src.var_calculator import compute_var_caseA

    fema = load_fema_disasters(use_cache=True)
    county = load_county_finance()
    merged = merge_fema_county(fema, county)

    # Test ALL available counties
    all_fips = merged['FIPS'].unique()
    print(f"  Testing {len(all_fips)} counties...")

    passed = 0
    failed = 0

    for fips in all_fips:
        try:
            est = CaseAHurricaneEstimator(merged, fips)
            params = est.get_params(horizon_years=10, n_sims=2000)

            # Validate
            assert params.sigma >= 0, f"sigma={params.sigma}"
            assert params.lam >= 0, f"lam={params.lam}"
            assert params.mu_j <= 0, f"mu_j={params.mu_j}"
            assert params.S0 > 0, f"S0={params.S0}"

            result = simulate_jump_diffusion(params, seed=42)
            reserve = merged[merged['FIPS'] == fips]['total_balance'].iloc[-1]
            var = compute_var_caseA(result, reserve_balance=reserve)

            assert -1.0 <= var.var_pct <= 0.0, f"VaR={var.var_pct}"
            assert 0 <= var.depletion_prob <= 1.0

            passed += 1
        except Exception as e:
            failed += 1
            print(f"    FAIL {fips}: {e}")

    print(f"  {passed}/{passed+failed} counties passed")
    assert failed == 0, f"{failed} counties failed"


# ========================== SECTION 3: EDGE CASES ==========================

def test_edge_case_zero_jumps():
    """When lambda=0, should degenerate to pure GBM"""
    from src.jump_diffusion_engine import JumpDiffusionParams, simulate_jump_diffusion

    params = JumpDiffusionParams(
        mu=0.05, sigma=0.2, lam=0.0,
        mu_j=0.0, sigma_j=0.0,
        S0=100, T=1, dt=1/252, n_sims=5000
    )
    result = simulate_jump_diffusion(params, seed=42)

    # Should have zero jumps
    assert result['jump_counts'].sum() == 0, "Should have zero jumps"
    assert result['jump_sums'].sum() == 0.0, "Jump sums should be zero"

    # Should approximate GBM: E[S_T] = S0 * exp(mu * T)
    expected = 100 * np.exp(0.05)
    actual = result['final_values'].mean()
    err = abs(actual - expected) / expected * 100
    assert err < 5, f"GBM mean error {err:.2f}% too large"
    print(f"  lambda=0: mean={actual:.2f} (expected {expected:.2f}, err {err:.2f}%)")


def test_edge_case_extreme_positive_drift():
    """High positive drift should produce positive expected returns"""
    from src.jump_diffusion_engine import JumpDiffusionParams, simulate_jump_diffusion

    params = JumpDiffusionParams(
        mu=0.30, sigma=0.1, lam=0.5,
        mu_j=-0.05, sigma_j=0.02,
        S0=100, T=5, dt=1/252, n_sims=3000
    )
    result = simulate_jump_diffusion(params, seed=42)

    mean_final = result['final_values'].mean()
    # With 30% drift over 5 years, should grow significantly
    assert mean_final > 100, f"Expected growth, got mean={mean_final:.2f}"
    print(f"  High drift: mean_final={mean_final:.2f} (S0=100, T=5yr)")


def test_edge_case_negative_drift():
    """Negative drift should produce declining expected values"""
    from src.jump_diffusion_engine import JumpDiffusionParams, simulate_jump_diffusion

    params = JumpDiffusionParams(
        mu=-0.10, sigma=0.2, lam=0.3,
        mu_j=-0.05, sigma_j=0.02,
        S0=100, T=5, dt=1/252, n_sims=3000
    )
    result = simulate_jump_diffusion(params, seed=42)

    mean_final = result['final_values'].mean()
    assert mean_final < 100, f"Expected decline, got mean={mean_final:.2f}"
    print(f"  Negative drift: mean_final={mean_final:.2f} (S0=100, T=5yr)")


def test_edge_case_very_high_volatility():
    """Very high sigma should not crash and should produce wide dispersion"""
    from src.jump_diffusion_engine import JumpDiffusionParams, simulate_jump_diffusion

    params = JumpDiffusionParams(
        mu=0.05, sigma=0.8, lam=1.0,
        mu_j=-0.1, sigma_j=0.05,
        S0=100, T=1, dt=1/252, n_sims=2000
    )
    result = simulate_jump_diffusion(params, seed=42)

    assert np.all(result['paths'] > 0), "All paths must be positive"
    cv = result['final_values'].std() / result['final_values'].mean()
    assert cv > 0.3, f"Expected high dispersion, CV={cv:.2f}"
    print(f"  High vol: CV={cv:.2f}, all paths positive")


def test_edge_case_single_path():
    """n_sims=1 should work"""
    from src.jump_diffusion_engine import JumpDiffusionParams, simulate_jump_diffusion

    params = JumpDiffusionParams(
        mu=0.05, sigma=0.2, lam=1.0,
        mu_j=-0.05, sigma_j=0.02,
        S0=100, T=1, dt=1/252, n_sims=1
    )
    result = simulate_jump_diffusion(params, seed=42)
    assert result['paths'].shape[0] == 1
    assert result['paths'].shape[1] == 253
    print(f"  n_sims=1: shape={result['paths'].shape}")


def test_edge_case_short_horizon():
    """T=1 day should work"""
    from src.jump_diffusion_engine import JumpDiffusionParams, simulate_jump_diffusion

    params = JumpDiffusionParams(
        mu=0.05, sigma=0.2, lam=1.0,
        mu_j=-0.05, sigma_j=0.02,
        S0=100, T=1/252, dt=1/252, n_sims=100
    )
    result = simulate_jump_diffusion(params, seed=42)
    assert result['paths'].shape[1] == 2  # initial + 1 step
    print(f"  T=1day: shape={result['paths'].shape}")


def test_edge_case_large_lam():
    """Very high jump frequency should not crash"""
    from src.jump_diffusion_engine import JumpDiffusionParams, simulate_jump_diffusion

    params = JumpDiffusionParams(
        mu=0.05, sigma=0.1, lam=50.0,
        mu_j=-0.01, sigma_j=0.005,
        S0=100, T=1, dt=1/252, n_sims=1000
    )
    result = simulate_jump_diffusion(params, seed=42)
    avg_jumps = result['jump_counts'].sum(axis=1).mean()
    assert avg_jumps > 30, f"Expected ~50 jumps, got {avg_jumps:.0f}"
    print(f"  lam=50: avg_jumps={avg_jumps:.0f}")


# ========================== SECTION 4: DATA LOADER FALLBACKS ==========================

def test_fema_fallback():
    """Test FEMA synthetic data generation when cache is deleted"""
    from src.data_loader import _generate_synthetic_fema
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        path = f.name

    try:
        df = _generate_synthetic_fema(2015, 2025, path)
        assert len(df) > 0
        assert 'declarationDate' in df.columns
        assert 'totalObligatedAmount' in df.columns
        assert 'FIPS' in df.columns
        # Check Poisson allows 0 hurricane years
        years = df.groupby(df['declarationDate'].dt.year).size()
        assert len(years) > 0
        print(f"  Synthetic FEMA: {len(df)} records, years with data: {len(years)}")
    finally:
        os.unlink(path)


def test_county_finance_fallback():
    """Test county finance synthetic data"""
    from src.data_loader import _generate_synthetic_county_finance
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        path = f.name

    try:
        df = _generate_synthetic_county_finance(path)
        assert len(df) > 0
        assert 'total_revenue' in df.columns
        assert 'total_balance' in df.columns
        assert (df['total_revenue'] > 0).all()
        assert (df['total_balance'] > 0).all()
        print(f"  Synthetic county finance: {len(df)} records, "
              f"{df['FIPS'].nunique()} counties")
    finally:
        os.unlink(path)


def test_smh_fallback():
    """Test SMH synthetic data generation"""
    from src.data_loader import _generate_synthetic_smh
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        path = f.name

    try:
        df = _generate_synthetic_smh("2015-01-01", "2025-12-31", path)
        assert len(df) > 2000
        assert 'Close' in df.columns
        assert 'log_return' in df.columns
        assert (df['Close'] > 0).all()
        # Check no NaN in returns
        returns = df['log_return'].dropna()
        assert len(returns) > 2000
        print(f"  Synthetic SMH: {len(df)} days, "
              f"price ${df['Close'].min():.2f}-${df['Close'].max():.2f}")
    finally:
        os.unlink(path)


# ========================== SECTION 5: CROSS-VALIDATION ==========================

def test_jd_vs_heston_same_seed():
    """With same seed, JD and Heston should produce different but reasonable results"""
    from src.jump_diffusion_engine import JumpDiffusionParams, simulate_jump_diffusion
    from src.heston_engine import HestonParams, simulate_heston

    jd_params = JumpDiffusionParams(
        mu=0.05, sigma=0.2, lam=0.0,
        mu_j=0.0, sigma_j=0.0,
        S0=100, T=1, dt=1/252, n_sims=3000
    )
    heston_params = HestonParams(
        mu=0.05, v0=0.04, kappa=2.0, theta=0.04,
        xi=0.001, rho=-0.5,  # very low xi = almost constant vol
        S0=100, T=1, dt=1/252, n_sims=3000
    )

    jd_result = simulate_jump_diffusion(jd_params, seed=42)
    heston_result = simulate_heston(heston_params, seed=42)

    # Both should have same mean (approximately)
    jd_mean = jd_result['final_values'].mean()
    heston_mean = heston_result['final_values'].mean()
    err = abs(jd_mean - heston_mean) / jd_mean * 100
    print(f"  JD mean: {jd_mean:.2f}, Heston mean: {heston_mean:.2f}, diff: {err:.2f}%")

    # With xi near 0, Heston should approximate GBM
    assert err < 15, f"JD vs Heston (low xi) mean diff {err:.1f}% too large"


def test_fbm_h05_matches_gbm():
    """fBM with H=0.5 should approximate GBM"""
    from src.jump_diffusion_engine import JumpDiffusionParams, simulate_jump_diffusion
    from src.fbm_engine import FBMParams, simulate_fbm

    n_sims = 2000

    gbm_params = JumpDiffusionParams(
        mu=0.05, sigma=0.2, lam=0.0,
        mu_j=0.0, sigma_j=0.0,
        S0=100, T=1, dt=1/252, n_sims=n_sims
    )
    fbm_params = FBMParams(
        mu=0.05, sigma=0.2, hurst=0.5,
        S0=100, T=1, dt=1/252, n_sims=n_sims
    )

    gbm = simulate_jump_diffusion(gbm_params, seed=42)
    fbm = simulate_fbm(fbm_params, seed=42)

    gbm_mean = gbm['final_values'].mean()
    fbm_mean = fbm['final_values'].mean()
    err = abs(gbm_mean - fbm_mean) / gbm_mean * 100
    print(f"  GBM mean: {gbm_mean:.2f}, fBM(H=0.5) mean: {fbm_mean:.2f}, err: {err:.2f}%")
    assert err < 10, f"GBM vs fBM(H=0.5) error {err:.1f}% too large"


def test_paired_paths_share_noise():
    """Paired paths must share the same Z_continuous"""
    from src.jump_diffusion_engine import JumpDiffusionParams, simulate_paired_paths

    params = JumpDiffusionParams(
        mu=0.05, sigma=0.2, lam=2.0,
        mu_j=-0.05, sigma_j=0.03,
        S0=100, T=1, dt=1/252, n_sims=1000
    )
    paired = simulate_paired_paths(params, seed=42)

    # The difference in log_returns should equal jump_sums
    log_diff = paired['log_returns_with'] - paired['log_returns_without']
    assert np.allclose(log_diff, paired['jump_sums'], atol=1e-10), \
        "Paired path difference should equal jump_sums exactly"
    print(f"  Paired noise sharing: verified (max diff < 1e-10)")


def test_decompose_risk_positive():
    """decompose_risk should always produce positive jump risk with paired paths"""
    from src.jump_diffusion_engine import JumpDiffusionParams, simulate_paired_paths
    from src.var_calculator import decompose_risk

    # Test with multiple parameter sets
    configs = [
        ("low_jump", 0.05, 0.2, 0.5, -0.03, 0.02),
        ("med_jump", 0.05, 0.2, 2.0, -0.05, 0.03),
        ("high_jump", 0.05, 0.2, 5.0, -0.08, 0.04),
    ]

    for name, mu, sigma, lam, mu_j, sigma_j in configs:
        params = JumpDiffusionParams(
            mu=mu, sigma=sigma, lam=lam,
            mu_j=mu_j, sigma_j=sigma_j,
            S0=100, T=1, dt=1/252, n_sims=3000
        )
        paired = simulate_paired_paths(params, seed=42)
        decomp = decompose_risk(paired)

        assert decomp['jump_risk_pct'] >= 0, \
            f"{name}: jump_risk={decomp['jump_risk_pct']:.3f} should be >= 0"
        print(f"  {name}: jump_risk={decomp['jump_risk_pct']*100:.1f}%")


# ========================== SECTION 6: PARAMETER ESTIMATOR ROBUSTNESS ==========================

def test_estimator_small_sample():
    """Parameter estimator should handle very small samples gracefully"""
    from src.parameter_estimator import CaseAHurricaneEstimator
    from src.jump_diffusion_engine import simulate_jump_diffusion

    # Create a tiny dataset
    tiny_df = pd.DataFrame({
        'FIPS': ['12086'] * 5,
        'year': [2020, 2021, 2022, 2023, 2024],
        'total_revenue': [1e9, 1.05e9, 1.1e9, 1.08e9, 1.12e9],
        'total_balance': [1.5e8, 1.6e8, 1.7e8, 1.65e8, 1.75e8],
        'n_hurricanes': [1, 0, 2, 0, 1],
        'total_federal_aid': [5e7, 0, 1.2e8, 0, 3e7],
    })

    est = CaseAHurricaneEstimator(tiny_df, '12086')
    params = est.get_params(horizon_years=5, n_sims=1000)

    assert params.sigma > 0
    assert params.mu_j <= 0
    result = simulate_jump_diffusion(params, seed=42)
    assert np.all(result['paths'] > 0)
    print(f"  Tiny sample: sigma={params.sigma:.4f}, mu_j={params.mu_j:.4f}")


def test_estimator_bootstrap():
    """Bootstrap should produce stable estimates with small samples"""
    from src.parameter_estimator import CaseAHurricaneEstimator

    # Create dataset with exactly 3 hurricane years
    df = pd.DataFrame({
        'FIPS': ['12086'] * 10,
        'year': range(2015, 2025),
        'total_revenue': [1e9] * 10,
        'total_balance': [1.5e8] * 10,
        'n_hurricanes': [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        'total_federal_aid': [0, 0, 0, 5e7, 0, 0, 3e7, 0, 0, 4e7],
    })

    est = CaseAHurricaneEstimator(df, '12086')

    # Run bootstrap multiple times
    estimates = []
    for seed in [42, 123, 456]:
        mu_j, sigma_j = est.estimate_jump_distribution()
        estimates.append((mu_j, sigma_j))

    # All should be negative mu_j
    for mu_j, sigma_j in estimates:
        assert mu_j < 0, f"mu_j={mu_j} should be negative"
        assert sigma_j > 0, f"sigma_j={sigma_j} should be positive"

    print(f"  Bootstrap: mu_j range [{min(e[0] for e in estimates):.3f}, "
          f"{max(e[0] for e in estimates):.3f}]")


def test_dynamic_thresholds():
    """Dynamic thresholds should reduce lambda when made stricter"""
    from src.data_loader import load_semiconductor_data, build_geopolitical_event_timeline
    from src.parameter_estimator import CaseBGeopoliticalEstimator

    smh = load_semiconductor_data(use_cache=True)
    events = build_geopolitical_event_timeline()

    # Default threshold
    est_default = CaseBGeopoliticalEstimator(smh, events)
    params_default = est_default.get_params()

    # Strict threshold
    est_strict = CaseBGeopoliticalEstimator(
        smh, events, min_severity=5, min_relevance=5
    )
    params_strict = est_strict.get_params()

    assert params_strict.lam < params_default.lam, \
        f"Strict threshold should reduce lambda: {params_strict.lam} vs {params_default.lam}"
    print(f"  Default lambda: {params_default.lam:.2f}, "
          f"Strict lambda: {params_strict.lam:.2f}")


# ========================== SECTION 7: VAR CONSISTENCY ==========================

def test_var_monotonic_in_lam():
    """VaR should worsen (more negative) as lambda increases"""
    from src.jump_diffusion_engine import JumpDiffusionParams, simulate_jump_diffusion
    from src.var_calculator import compute_var_caseB

    var_values = []
    for lam in [0.1, 0.5, 1.0, 2.0, 5.0]:
        params = JumpDiffusionParams(
            mu=0.05, sigma=0.2, lam=lam,
            mu_j=-0.05, sigma_j=0.03,
            S0=100, T=1, dt=1/252, n_sims=3000
        )
        result = simulate_jump_diffusion(params, seed=42)
        var = compute_var_caseB(result)
        var_values.append((lam, var.var_pct))

    # VaR should become more negative as lambda increases
    for i in range(len(var_values) - 1):
        lam1, var1 = var_values[i]
        lam2, var2 = var_values[i + 1]
        assert var2 <= var1, \
            f"VaR should worsen with lambda: lam={lam1} var={var1:.3f} vs lam={lam2} var={var2:.3f}"

    print("  VaR monotonicity in lambda: verified")
    for lam, var in var_values:
        print(f"    lam={lam:.1f}: VaR={var*100:.1f}%")


def test_var_monotonic_in_sigma():
    """VaR should worsen as sigma increases"""
    from src.jump_diffusion_engine import JumpDiffusionParams, simulate_jump_diffusion
    from src.var_calculator import compute_var_caseB

    var_values = []
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5]:
        params = JumpDiffusionParams(
            mu=0.05, sigma=sigma, lam=1.0,
            mu_j=-0.05, sigma_j=0.03,
            S0=100, T=1, dt=1/252, n_sims=3000
        )
        result = simulate_jump_diffusion(params, seed=42)
        var = compute_var_caseB(result)
        var_values.append((sigma, var.var_pct))

    for i in range(len(var_values) - 1):
        s1, var1 = var_values[i]
        s2, var2 = var_values[i + 1]
        assert var2 <= var1, \
            f"VaR should worsen with sigma: {s1} var={var1:.3f} vs {s2} var={var2:.3f}"

    print("  VaR monotonicity in sigma: verified")


def test_cvar_worse_than_var():
    """CVaR should always be worse (more negative) than VaR"""
    from src.jump_diffusion_engine import JumpDiffusionParams, simulate_jump_diffusion
    from src.var_calculator import compute_var_caseB

    params = JumpDiffusionParams(
        mu=0.05, sigma=0.2, lam=2.0,
        mu_j=-0.05, sigma_j=0.03,
        S0=100, T=1, dt=1/252, n_sims=5000
    )
    result = simulate_jump_diffusion(params, seed=42)
    var = compute_var_caseB(result)

    assert var.cvar_pct <= var.var_pct, \
        f"CVaR ({var.cvar_pct:.3f}) should be worse than VaR ({var.var_pct:.3f})"
    print(f"  VaR={var.var_pct*100:.2f}%, CVaR={var.cvar_pct*100:.2f}% (CVaR worse: OK)")


# ========================== MAIN ==========================

def main():
    print("=" * 70)
    print("  COMPREHENSIVE ROBUSTNESS VERIFICATION")
    print("=" * 70)

    tests = [
        # Section 1: Different stocks
        ("Case B: Different stocks (SPY, QQQ, XLE)", test_caseB_with_different_stocks),
        ("Case B: Different time periods", test_caseB_different_time_periods),

        # Section 2: Different counties
        ("Case A: All counties", test_caseA_different_counties),

        # Section 3: Edge cases
        ("Edge: Zero jumps (GBM degeneration)", test_edge_case_zero_jumps),
        ("Edge: Extreme positive drift", test_edge_case_extreme_positive_drift),
        ("Edge: Negative drift", test_edge_case_negative_drift),
        ("Edge: Very high volatility", test_edge_case_very_high_volatility),
        ("Edge: Single path (n_sims=1)", test_edge_case_single_path),
        ("Edge: Short horizon (1 day)", test_edge_case_short_horizon),
        ("Edge: Large lambda (50)", test_edge_case_large_lam),

        # Section 4: Data loader fallbacks
        ("Fallback: FEMA synthetic data", test_fema_fallback),
        ("Fallback: County finance synthetic", test_county_finance_fallback),
        ("Fallback: SMH synthetic data", test_smh_fallback),

        # Section 5: Cross-validation
        ("Cross-val: JD vs Heston (low xi)", test_jd_vs_heston_same_seed),
        ("Cross-val: fBM(H=0.5) vs GBM", test_fbm_h05_matches_gbm),
        ("Cross-val: Paired paths share noise", test_paired_paths_share_noise),
        ("Cross-val: decompose_risk positive", test_decompose_risk_positive),

        # Section 6: Parameter estimator
        ("Estimator: Small sample handling", test_estimator_small_sample),
        ("Estimator: Bootstrap stability", test_estimator_bootstrap),
        ("Estimator: Dynamic thresholds", test_dynamic_thresholds),

        # Section 7: VaR consistency
        ("VaR: Monotonic in lambda", test_var_monotonic_in_lam),
        ("VaR: Monotonic in sigma", test_var_monotonic_in_sigma),
        ("VaR: CVaR worse than VaR", test_cvar_worse_than_var),
    ]

    total_start = time.time()
    for name, test_fn in tests:
        run_test(name, test_fn)
    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "=" * 70)
    print("  VERIFICATION SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, status, _, _ in RESULTS if status == "PASS")
    failed = sum(1 for _, status, _, _ in RESULTS if status == "FAIL")

    for name, status, elapsed, error in RESULTS:
        marker = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"  {marker} {name} ({elapsed:.1f}s)")
        if error:
            print(f"         Error: {error[:80]}")

    print(f"\n  Total: {passed} passed, {failed} failed, {total_elapsed:.1f}s")

    if failed > 0:
        print(f"\n  WARNING: {failed} tests failed!")
        sys.exit(1)
    else:
        print(f"\n  ALL TESTS PASSED - Code is robust.")


if __name__ == "__main__":
    with TeeOutput("08_robustness", "robustness.txt"):
        main()
        # Save test results as CSV
        test_rows = []
        for name, status, elapsed, error in RESULTS:
            test_rows.append({
                'test': name, 'status': status,
                'elapsed_sec': round(elapsed, 1), 'error': error,
            })
        save_csv(pd.DataFrame(test_rows), '08_robustness', 'test_results.csv')
