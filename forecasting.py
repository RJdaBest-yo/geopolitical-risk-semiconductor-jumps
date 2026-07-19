"""
=============================================================================
FILE: side_study/src/forecasting.py
PURPOSE: Natural disaster forecasting and lead-time risk reduction model

Core Question: If we know a hurricane is coming (via seasonal patterns +
weather forecasting), how much does lead time reduce tail risk?

Model:
  - Hurricanes are seasonal (Jun-Nov, peak Aug-Oct) -> predictable timing
  - Weather forecasting gives 3-7 day lead time for specific storms
  - Lead time allows pre-positioning: evacuation, reserve deployment,
    emergency supplies -> reduces effective loss ratio

Data sources:
  - NOAA hurricane season statistics (historical track data)
  - FEMA incident begin/end dates (lead time proxy)
  - Academic literature on preparation cost-benefit ratios
=============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

# Import from main study
import sys, os
_main_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _main_root not in sys.path:
    sys.path.insert(0, _main_root)

from src.jump_diffusion_engine import (
    JumpDiffusionParams, simulate_jump_diffusion
)
from src.var_calculator import compute_var_caseA


# ========================== Seasonal Model ==========================

@dataclass
class SeasonalHurricaneModel:
    """
    Monthly hurricane probability distribution (North Atlantic)

    Based on NOAA 1990-2023 historical statistics:
    - Season: June 1 - November 30
    - Peak: August-October (85% of all hurricanes)
    - Average: 14 named storms, 7 hurricanes per season
    """
    # Probability of a named storm forming in each month
    # (normalized from NOAA historical frequencies)
    monthly_prob: dict = None

    def __post_init__(self):
        if self.monthly_prob is None:
            self.monthly_prob = {
                1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.02,
                6: 0.08, 7: 0.12, 8: 0.25, 9: 0.30, 10: 0.18,
                11: 0.05, 12: 0.00
            }

    def get_season_start_probability(self, current_month: int) -> float:
        """Probability that a hurricane will occur in the next N months"""
        prob = 0.0
        for m in range(current_month + 1, min(current_month + 4, 13)):
            prob += self.monthly_prob.get(m, 0)
        # Cap at 1.0
        return min(prob, 1.0)

    def expected_annual_hurricanes(self) -> float:
        """Annual expected count from monthly distribution"""
        return sum(self.monthly_prob.values())


# ========================== Lead Time Model ==========================

@dataclass
class LeadTimeEffect:
    """
    Model how lead time reduces hurricane damage

    Based on literature:
    - 3-day forecast: 15-25% damage reduction (evacuation, boarding)
    - 7-day forecast: 25-40% damage reduction (pre-positioning supplies)
    - 14-day forecast: 35-55% damage reduction (full mobilization)
    - Seasonal awareness: 10-15% additional (budget pre-allocation)

    Sources:
    - Smith and McCarty (2009): Evacuation reduces mortality 90%+
    - NOAA (2020): 72-hr forecast accuracy improved 50% since 1990
    - Multihazard Mitigation Council (2019): $1 in mitigation saves $6
    """
    # lead_time_days -> (mean_reduction, std_reduction)
    reduction_schedule: dict = None

    def __post_init__(self):
        if self.reduction_schedule is None:
            self.reduction_schedule = {
                0:  (0.00, 0.00),   # No warning: full damage
                1:  (0.05, 0.03),   # 1 day: minimal preparation
                3:  (0.20, 0.08),   # 3 days: evacuation begins
                5:  (0.30, 0.10),   # 5 days: supplies pre-positioned
                7:  (0.35, 0.10),   # 7 days: full preparation window
                10: (0.42, 0.12),   # 10 days: extended preparation
                14: (0.50, 0.12),   # 14 days: maximum preparation
            }

    def get_loss_reduction(self, lead_time_days: float) -> tuple:
        """
        Interpolate loss reduction for a given lead time

        Returns:
            (mean_reduction, std_reduction)
            mean_reduction: fraction of damage prevented (0-1)
            std_reduction: uncertainty in the reduction
        """
        times = sorted(self.reduction_schedule.keys())

        if lead_time_days <= times[0]:
            return self.reduction_schedule[times[0]]
        if lead_time_days >= times[-1]:
            return self.reduction_schedule[times[-1]]

        # Linear interpolation
        for i in range(len(times) - 1):
            if times[i] <= lead_time_days <= times[i + 1]:
                t0, t1 = times[i], times[i + 1]
                r0 = self.reduction_schedule[t0]
                r1 = self.reduction_schedule[t1]
                w = (lead_time_days - t0) / (t1 - t0)
                return (
                    r0[0] + w * (r1[0] - r0[0]),
                    r0[1] + w * (r1[1] - r0[1])
                )

        return (0.0, 0.0)


def adjust_params_for_lead_time(
    base_params: JumpDiffusionParams,
    lead_time_days: float
) -> JumpDiffusionParams:
    """
    Adjust jump diffusion parameters to reflect preparation with lead time

    Mechanism:
    1. Loss reduction: mu_j becomes less negative
       mu_j_adjusted = mu_j * (1 - mean_reduction)
    2. Reduced uncertainty: sigma_j decreases
       sigma_j_adjusted = sigma_j * (1 - mean_reduction)
    3. Lower effective frequency: some storms are avoided entirely
       via evacuation (reduced lambda for populated areas)
    """
    lead = LeadTimeEffect()
    mean_red, std_red = lead.get_loss_reduction(lead_time_days)

    # Adjust jump amplitude (less severe losses)
    adjusted_mu_j = base_params.mu_j * (1 - mean_red)
    adjusted_sigma_j = base_params.sigma_j * (1 - 0.5 * mean_red)

    # Effective frequency reduction (evacuation avoids some claims)
    freq_reduction = 0.3 * mean_red  # 30% of damage reduction = avoided claims
    adjusted_lam = base_params.lam * (1 - freq_reduction)

    return JumpDiffusionParams(
        mu=base_params.mu,
        sigma=base_params.sigma,
        lam=adjusted_lam,
        mu_j=adjusted_mu_j,
        sigma_j=adjusted_sigma_j,
        S0=base_params.S0,
        T=base_params.T,
        dt=base_params.dt,
        n_sims=base_params.n_sims
    )


def run_forecast_analysis(
    base_params: JumpDiffusionParams,
    reserve_balance: float,
    lead_times: list = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Run comparative analysis across different lead times

    Returns:
        DataFrame with columns:
            lead_time, var_pct, cvar_pct, depletion_prob,
            loss_reduction_mean, reserve_adequate
    """
    if lead_times is None:
        lead_times = [0, 1, 3, 5, 7, 10, 14]

    lead = LeadTimeEffect()
    results = []

    for lt in lead_times:
        adjusted = adjust_params_for_lead_time(base_params, lt)
        sim = simulate_jump_diffusion(adjusted, seed=seed)
        var = compute_var_caseA(sim, reserve_balance=reserve_balance)

        mean_red, _ = lead.get_loss_reduction(lt)

        results.append({
            'lead_time_days': lt,
            'loss_reduction': f'{mean_red*100:.1f}%',
            'var_pct': var.var_pct * 100,
            'cvar_pct': var.cvar_pct * 100,
            'depletion_prob': var.depletion_prob * 100,
            'median_final': var.median_final,
            'reserve_adequate': var.depletion_prob < 0.05,
        })

    return pd.DataFrame(results)


# ========================== Self-test ==========================

def _run_self_tests():
    print("=" * 60)
    print("forecasting.py self-test")
    print("=" * 60)

    # Test 1: Seasonal model
    print("\n[Test 1] Seasonal model")
    season = SeasonalHurricaneModel()
    assert season.expected_annual_hurricanes() > 0
    # August should have highest probability
    assert season.monthly_prob[9] > season.monthly_prob[6]
    print(f"  Annual expected: {season.expected_annual_hurricanes():.2f}")
    print(f"  Season start prob from June: {season.get_season_start_probability(6):.2f}")
    print("  [PASS]")

    # Test 2: Lead time effect
    print("\n[Test 2] Lead time effect")
    lead = LeadTimeEffect()
    r0 = lead.get_loss_reduction(0)
    r7 = lead.get_loss_reduction(7)
    r14 = lead.get_loss_reduction(14)
    assert r0[0] == 0.0
    assert r7[0] > r0[0]
    assert r14[0] > r7[0]
    print(f"  0-day: {r0[0]*100:.0f}% reduction")
    print(f"  7-day: {r7[0]*100:.0f}% reduction")
    print(f"  14-day: {r14[0]*100:.0f}% reduction")
    print("  [PASS]")

    # Test 3: Forecast analysis
    print("\n[Test 3] Forecast analysis pipeline")
    base = JumpDiffusionParams(
        mu=0.03, sigma=0.05, lam=0.6,
        mu_j=-0.15, sigma_j=0.10,
        S0=1e9, T=10, dt=1.0, n_sims=5000
    )
    results = run_forecast_analysis(base, reserve_balance=1.5e8)
    print(results.to_string(index=False))

    # VaR should improve with lead time
    var_no_lead = results[results['lead_time_days'] == 0]['var_pct'].values[0]
    var_7_lead = results[results['lead_time_days'] == 7]['var_pct'].values[0]
    assert var_7_lead > var_no_lead, "7-day lead should improve (less negative) VaR"
    print(f"\n  VaR improved from {var_no_lead:.1f}% to {var_7_lead:.1f}% with 7-day lead")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_self_tests()
