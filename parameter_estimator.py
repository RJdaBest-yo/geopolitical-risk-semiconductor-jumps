"""
=============================================================================
FILE: src/parameter_estimator.py
PURPOSE: Parameter estimation from real data
RELATED:
  - Risk #3 fix: Real bootstrap for Case A small samples; improved Case B bootstrap
  - Risk #2 fix: Dynamic thresholds exposed as instance attributes
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings

from .jump_diffusion_engine import JumpDiffusionParams


class CaseAHurricaneEstimator:
    """
    Case A: Hurricane -> county fiscal jump diffusion parameter estimation
    """

    def __init__(self, merged_df: pd.DataFrame, target_fips: str):
        self.df = merged_df[merged_df["FIPS"] == target_fips].copy()
        self.df = self.df.sort_values("year").reset_index(drop=True)
        self.fips = target_fips

        if len(self.df) < 5:
            warnings.warn(
                f"County {target_fips} has only {len(self.df)} years of data"
            )

    def estimate_drift_and_volatility(self) -> tuple:
        """Estimate mu and sigma from non-hurricane years"""
        df = self.df.copy()
        df["revenue_growth"] = df["total_revenue"].pct_change()

        normal = df[df["n_hurricanes"] == 0].dropna(subset=["revenue_growth"])

        if len(normal) < 3:
            warnings.warn("Too few non-hurricane years, using all years")
            normal = df.dropna(subset=["revenue_growth"])

        mu = normal["revenue_growth"].mean()
        sigma = normal["revenue_growth"].std(ddof=1)
        sigma = max(sigma, 0.01)

        return mu, sigma

    def estimate_jump_frequency(self) -> float:
        """Estimate lambda: average hurricanes per year"""
        lam = self.df["n_hurricanes"].mean()
        return max(lam, 0.01)

    def estimate_jump_distribution(
        self, bootstrap_n: int = 200
    ) -> tuple:
        """
        Estimate jump amplitude distribution (mu_j, sigma_j)

        [FIXED Risk #3] Real bootstrap for small samples:
        - >= 5 samples: direct MLE fit
        - 2-4 samples: bootstrap resampling with jitter
        - < 2 samples: empirical fallback constants
        """
        hurricane_years = self.df[self.df["n_hurricanes"] > 0].copy()

        if len(hurricane_years) < 2:
            # Fallback: 5% revenue loss = log(0.95) = -0.051
            return np.log(0.95), 0.05

        loss_ratios = (
            hurricane_years["total_federal_aid"]
            / hurricane_years["total_revenue"]
        )
        loss_ratios = loss_ratios[(loss_ratios > 0) & (loss_ratios < 1)]

        if len(loss_ratios) < 2:
            return np.log(0.95), 0.05

        # Convert loss ratio to log-return space correctly:
        # loss_ratio=0.05 means 5% loss -> log(1 - 0.05) = log(0.95) = -0.051
        # NOT log(0.05) = -3.0 (that would mean a 95% loss per jump)
        log_returns = np.log(1.0 - loss_ratios.values)

        if len(log_returns) >= 5:
            mu_j = log_returns.mean()  # already negative
            sigma_j = max(log_returns.std(ddof=1), 0.01)
        else:
            # Bootstrap with jitter for small samples
            rng = np.random.default_rng(42)
            boot_means = []
            for _ in range(bootstrap_n):
                sample = rng.choice(log_returns, size=len(log_returns), replace=True)
                jitter = rng.normal(0, 0.005, size=len(sample))
                boot_means.append((sample + jitter).mean())

            boot_means = np.array(boot_means)
            mu_j = boot_means.mean()  # already negative
            sigma_j = max(boot_means.std(ddof=1), 0.01)

        # Ensure negative direction
        mu_j = min(mu_j, 0.0)
        sigma_j = max(sigma_j, 0.01)

        return mu_j, sigma_j

    def get_params(
        self,
        initial_revenue: float = None,
        horizon_years: float = 10.0,
        n_sims: int = 10_000
    ) -> JumpDiffusionParams:
        mu, sigma = self.estimate_drift_and_volatility()
        lam = self.estimate_jump_frequency()
        mu_j, sigma_j = self.estimate_jump_distribution()

        if initial_revenue is None:
            initial_revenue = self.df["total_revenue"].iloc[-1]

        return JumpDiffusionParams(
            mu=mu, sigma=sigma, lam=lam,
            mu_j=mu_j, sigma_j=sigma_j,
            S0=initial_revenue, T=horizon_years,
            dt=1.0, n_sims=n_sims
        )

    def summary(self) -> str:
        mu, sigma = self.estimate_drift_and_volatility()
        lam = self.estimate_jump_frequency()
        mu_j, sigma_j = self.estimate_jump_distribution()

        # mu_j is in log-return space: exp(mu_j) = fraction remaining
        # e.g., mu_j=-0.156 -> exp(-0.156)=0.856 -> 14.4% loss per jump
        loss_pct = (1 - np.exp(mu_j)) * 100

        lines = [
            f"{'='*50}",
            f"Case A Estimator - County {self.fips}",
            f"{'='*50}",
            f"Data: {self.df['year'].min()}-{self.df['year'].max()} ({len(self.df)} yr)",
            f"Hurricane years: {(self.df['n_hurricanes']>0).sum()}",
            f"",
            f"Drift mu      = {mu:.4f} ({mu*100:.2f}%/yr)",
            f"Volatility    = {sigma:.4f} ({sigma*100:.2f}%/yr)",
            f"Jump freq     = {lam:.2f}/yr",
            f"Jump mu_j     = {mu_j:.4f} (log-return space)",
            f"Jump sigma_j  = {sigma_j:.4f}",
            f"Jump median   = {loss_pct:.2f}% revenue loss per event",
            f"Initial rev   = ${self.df['total_revenue'].iloc[-1]:,.0f}",
            f"Reserve       = ${self.df['total_balance'].iloc[-1]:,.0f}",
            f"{'='*50}",
        ]
        return "\n".join(lines)


class CaseBGeopoliticalEstimator:
    """
    Case B: Geopolitical shock -> semiconductor ETF estimation

    [FIXED Risk #2] Dynamic thresholds as instance attributes
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        events_df: pd.DataFrame,
        window_days: int = 5,
        min_severity: int = 3,
        min_relevance: int = 3
    ):
        self.prices = price_df.sort_values("Date").reset_index(drop=True)
        self.events = events_df
        self.window_days = window_days
        # [FIXED] Dynamic thresholds as instance attributes
        self.min_severity = min_severity
        self.min_relevance = min_relevance

    def _get_significant_events(self) -> pd.DataFrame:
        """Filter events by current threshold settings"""
        return self.events[
            (self.events["severity"] >= self.min_severity) &
            (self.events["semiconductor_relevance"] >= self.min_relevance)
        ].copy()

    def _get_normal_mask(self) -> pd.Series:
        """Build mask for 'normal' trading days (excluding event windows)"""
        mask = pd.Series(True, index=self.prices.index)
        sig_events = self._get_significant_events()

        for _, event in sig_events.iterrows():
            ed = event["event_date"]
            future = self.prices[self.prices["Date"] >= ed]
            if len(future) == 0:
                continue
            idx = future.index[0]
            low = max(0, idx - self.window_days)
            high = min(len(self.prices), idx + self.window_days + 1)
            mask.iloc[low:high] = False

        return mask

    def estimate_drift_and_volatility(self) -> tuple:
        """Estimate normal-period mu and sigma (annualized)"""
        mask = self._get_normal_mask()
        normal_returns = self.prices.loc[mask, "log_return"].dropna()

        if len(normal_returns) < 100:
            warnings.warn("Too few normal trading days, using all data")
            normal_returns = self.prices["log_return"].dropna()

        mu = normal_returns.mean() * 252
        sigma = normal_returns.std(ddof=1) * np.sqrt(252)

        return mu, sigma

    def estimate_jump_frequency(self) -> float:
        """Estimate lambda using current threshold settings"""
        sig = self._get_significant_events()
        date_range = self.events["event_date"].max() - self.events["event_date"].min()
        years = max(date_range.days / 365.25, 1.0)
        lam = len(sig) / years
        return max(lam, 0.5)

    def estimate_jump_distribution(
        self, bootstrap_n: int = 200
    ) -> tuple:
        """
        Estimate jump amplitude distribution

        [FIXED Risk #3] Improved bootstrap with jitter
        [FIXED Risk #2] Uses self.min_severity / self.min_relevance
        """
        sig_events = self._get_significant_events()
        jump_sizes = []

        for _, event in sig_events.iterrows():
            ed = event["event_date"]
            window = self.prices[
                (self.prices["Date"] >= ed - pd.Timedelta(days=1)) &
                (self.prices["Date"] <= ed + pd.Timedelta(days=self.window_days * 2))
            ]
            if len(window) == 0:
                continue
            # Use actual log-return from the window
            min_ret = window["log_return"].min()
            if not np.isnan(min_ret):
                jump_sizes.append(min_ret)

        jump_sizes = np.array(jump_sizes)
        # Filter: keep negative jumps larger than 0.5% (in log-return space)
        negative_jumps = jump_sizes[jump_sizes < -0.005]

        if len(negative_jumps) >= 5:
            # negative_jumps are already log-returns (e.g., -0.05 for 5% drop)
            mu_j = negative_jumps.mean()  # already negative
            sigma_j = max(negative_jumps.std(ddof=1), 0.01)
        elif len(negative_jumps) >= 2:
            # Bootstrap with jitter on log-returns directly
            rng = np.random.default_rng(42)
            boot_means = []
            for _ in range(bootstrap_n):
                sample = rng.choice(negative_jumps, size=len(negative_jumps), replace=True)
                jitter = rng.normal(0, 0.002, size=len(sample))
                boot_means.append((sample + jitter).mean())

            boot_means = np.array(boot_means)
            mu_j = boot_means.mean()  # already negative
            sigma_j = max(boot_means.std(ddof=1), 0.01)
        else:
            # Empirical fallback: typical semiconductor single-day drops
            # 2% to 5% in log-return space
            mu_j = -0.035
            sigma_j = 0.015

        return mu_j, sigma_j

    def get_params(
        self,
        horizon_years: float = 2.0,
        n_sims: int = 10_000
    ) -> JumpDiffusionParams:
        mu, sigma = self.estimate_drift_and_volatility()
        lam = self.estimate_jump_frequency()
        mu_j, sigma_j = self.estimate_jump_distribution()

        current_price = float(self.prices["Close"].iloc[-1])

        return JumpDiffusionParams(
            mu=mu, sigma=sigma, lam=lam,
            mu_j=mu_j, sigma_j=sigma_j,
            S0=current_price, T=horizon_years,
            dt=1.0/252, n_sims=n_sims
        )

    def summary(self) -> str:
        mu, sigma = self.estimate_drift_and_volatility()
        lam = self.estimate_jump_frequency()
        mu_j, sigma_j = self.estimate_jump_distribution()
        sig = self._get_significant_events()

        lines = [
            f"{'='*50}",
            f"Case B Estimator - Semiconductor ETF",
            f"(severity>={self.min_severity}, relevance>={self.min_relevance})",
            f"{'='*50}",
            f"Prices: {self.prices['Date'].min().strftime('%Y-%m-%d')} to "
            f"{self.prices['Date'].max().strftime('%Y-%m-%d')} ({len(self.prices)} days)",
            f"Events: {len(self.events)} total, {len(sig)} significant",
            f"",
            f"Drift mu      = {mu:.4f} ({mu*100:.2f}%/yr)",
            f"Volatility    = {sigma:.4f} ({sigma*100:.2f}%/yr)",
            f"Jump freq     = {lam:.2f}/yr",
            f"Jump mu_j     = {mu_j:.4f} (log-return)",
            f"Jump sigma_j  = {sigma_j:.4f}",
            f"Jump median   = {abs(mu_j)*100:.2f}% daily drop per event",
            f"Current price = ${self.prices['Close'].iloc[-1]:.2f}",
            f"{'='*50}",
        ]
        return "\n".join(lines)


# ========================== Self-test ==========================

def _run_self_tests():
    print("=" * 60)
    print("parameter_estimator.py self-test")
    print("=" * 60)

    from .data_loader import (
        load_fema_disasters, load_county_finance, merge_fema_county,
        load_semiconductor_data, build_geopolitical_event_timeline
    )

    # Test 1: Case A
    print("\n[Test 1] Case A: Hurricane parameter estimation")
    fema = load_fema_disasters(use_cache=True)
    county = load_county_finance()
    merged = merge_fema_county(fema, county)

    target_fips = merged["FIPS"].value_counts().index[0]
    est_a = CaseAHurricaneEstimator(merged, target_fips)
    params_a = est_a.get_params()
    print(est_a.summary())
    assert params_a.sigma > 0
    assert params_a.lam > 0
    assert params_a.mu_j < 0
    assert params_a.S0 > 0
    print("  [PASS]")

    # Test 2: Case B with dynamic thresholds
    print("\n[Test 2] Case B: Geopolitical with dynamic thresholds")
    smh = load_semiconductor_data(use_cache=True)
    events = build_geopolitical_event_timeline()

    # Default threshold
    est_b_default = CaseBGeopoliticalEstimator(smh, events)
    params_b_default = est_b_default.get_params()
    print(f"  Default (>=3): lambda={params_b_default.lam:.2f}")

    # Stricter threshold
    est_b_strict = CaseBGeopoliticalEstimator(
        smh, events, min_severity=4, min_relevance=5
    )
    params_b_strict = est_b_strict.get_params()
    print(f"  Strict (>=4,>=5): lambda={params_b_strict.lam:.2f}")

    assert params_b_strict.lam < params_b_default.lam, \
        "Stricter threshold should reduce lambda"
    print(est_b_default.summary())
    print("  [PASS] Dynamic thresholds work")

    # Test 3: Parameter ranges
    print("\n[Test 3] Parameter range check")
    assert 0.005 < params_a.sigma < 0.30, f"Case A sigma={params_a.sigma}"
    assert 0.10 < params_b_default.sigma < 0.80, f"Case B sigma={params_b_default.sigma}"
    assert params_a.mu_j < 0
    assert params_b_default.mu_j < 0
    print("  [PASS] All parameters in realistic range")

    print("\n" + "=" * 60)
    print("All tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_self_tests()
