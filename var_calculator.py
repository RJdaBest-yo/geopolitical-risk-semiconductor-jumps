"""
=============================================================================
FILE: src/var_calculator.py
PURPOSE: VaR calculation module -- risk value and tail risk metrics
RELATED:
  - Risk #5 fix: decompose_risk now uses simulate_paired_paths() (shared noise)
  - Risk #5 fix: jump_risk measured via CVaR comparison, not std difference

Contains:
  - compute_var_caseA: County finance VaR + reserve depletion probability
  - compute_var_caseB: Semiconductor ETF VaR + max drawdown
  - decompose_risk:   Jump risk vs continuous risk decomposition (FIXED)
  - compare_domains:  Cross-domain comparison table
=============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class VaRResult:
    """VaR calculation result container"""
    domain: str
    confidence_level: float
    var_pct: float
    var_absolute: float
    cvar_pct: float
    cvar_absolute: float
    median_final: float
    mean_final: float
    depletion_prob: float = 0.0
    max_drawdown_var: float = 0.0
    jump_risk_pct: float = 0.0
    interpretation: str = ""

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "confidence": self.confidence_level,
            "var_pct": f"{self.var_pct*100:.2f}%",
            "var_absolute": f"${self.var_absolute:,.0f}",
            "cvar_pct": f"{self.cvar_pct*100:.2f}%",
            "cvar_absolute": f"${self.cvar_absolute:,.0f}",
            "median_final": f"${self.median_final:,.0f}",
            "depletion_prob": f"{self.depletion_prob*100:.2f}%",
            "max_drawdown_var": f"{self.max_drawdown_var*100:.2f}%",
            "jump_risk_pct": f"{self.jump_risk_pct*100:.1f}%",
        }


def compute_var_caseA(
    sim_result: dict,
    reserve_balance: float,
    confidence: float = 0.95
) -> VaRResult:
    """
    Case A VaR: County fiscal risk from hurricanes
    """
    paths = sim_result["paths"]
    S0 = paths[0, 0]
    n_years = paths.shape[1] - 1

    final_values = paths[:, -1]
    pct_change = (final_values - S0) / S0

    alpha = 1 - confidence
    var_pct = np.percentile(pct_change, alpha * 100)
    var_absolute = S0 * var_pct

    tail_mask = pct_change <= var_pct
    cvar_pct = pct_change[tail_mask].mean() if tail_mask.sum() > 0 else var_pct
    cvar_absolute = S0 * cvar_pct

    depletion_threshold = reserve_balance / S0
    depletion_prob = (pct_change < -depletion_threshold).mean()

    jump_risk_pct = sim_result.get("jump_risk_pct", 0.0)

    interpretation = (
        f"At {confidence*100:.0f}% confidence, "
        f"county revenue may decline {abs(var_pct)*100:.1f}% "
        f"(${abs(var_absolute):,.0f}) over {n_years} years. "
        f"Reserve depletion probability: {depletion_prob*100:.1f}%"
    )

    return VaRResult(
        domain="caseA_natural",
        confidence_level=confidence,
        var_pct=var_pct,
        var_absolute=var_absolute,
        cvar_pct=cvar_pct,
        cvar_absolute=cvar_absolute,
        median_final=np.median(final_values),
        mean_final=np.mean(final_values),
        depletion_prob=depletion_prob,
        jump_risk_pct=jump_risk_pct,
        interpretation=interpretation
    )


def compute_var_caseB(
    sim_result: dict,
    confidence: float = 0.95
) -> VaRResult:
    """
    Case B VaR: Semiconductor ETF risk from geopolitical shocks
    """
    paths = sim_result["paths"]
    S0 = paths[0, 0]

    final_values = paths[:, -1]
    pct_change = (final_values - S0) / S0
    alpha = 1 - confidence
    var_pct = np.percentile(pct_change, alpha * 100)
    var_absolute = S0 * var_pct

    tail_mask = pct_change <= var_pct
    cvar_pct = pct_change[tail_mask].mean() if tail_mask.sum() > 0 else var_pct
    cvar_absolute = S0 * cvar_pct

    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths - running_max) / running_max
    max_drawdowns = drawdowns.min(axis=1)
    max_dd_var = np.percentile(max_drawdowns, alpha * 100)

    jump_risk_pct = sim_result.get("jump_risk_pct", 0.0)

    n_days = paths.shape[1] - 1
    interpretation = (
        f"At {confidence*100:.0f}% confidence, "
        f"semiconductor ETF max drawdown can reach {abs(max_dd_var)*100:.1f}%, "
        f"final price VaR: {abs(var_pct)*100:.1f}%"
    )

    return VaRResult(
        domain="caseB_geopolitical",
        confidence_level=confidence,
        var_pct=var_pct,
        var_absolute=var_absolute,
        cvar_pct=cvar_pct,
        cvar_absolute=cvar_absolute,
        median_final=np.median(final_values),
        mean_final=np.mean(final_values),
        max_drawdown_var=max_dd_var,
        jump_risk_pct=jump_risk_pct,
        interpretation=interpretation
    )


def decompose_risk(paired_result: dict) -> dict:
    """
    Risk decomposition using PAIRED paths (shared Z_continuous)

    [FIXED] Previously compared two independent simulations with different
    random noise. Now uses simulate_paired_paths() output where both paths
    share identical continuous diffusion, isolating only the jump contribution.

    Measuring method: CVaR-based (not std-based).
    Std is symmetric and doesn't capture tail asymmetry from jumps.
    CVaR(5%) directly measures the left tail where jumps dominate.

    Args:
        paired_result: output from simulate_paired_paths()

    Returns:
        dict with jump risk metrics
    """
    final_with = paired_result["final_with_jump"]
    final_without = paired_result["final_without_jump"]
    S0 = final_without[0]  # initial value

    # Percentage changes
    pct_with = (final_with - S0) / S0
    pct_without = (final_without - S0) / S0

    # VaR at 5%
    var_with = np.percentile(pct_with, 5)
    var_without = np.percentile(pct_without, 5)

    # CVaR at 5%
    tail_with = pct_with[pct_with <= var_with]
    tail_without = pct_without[pct_without <= var_without]
    cvar_with = tail_with.mean() if len(tail_with) > 0 else var_with
    cvar_without = tail_without.mean() if len(tail_without) > 0 else var_without

    # Jump risk contribution via CVaR difference
    # cvar_with < cvar_without (jumps make tail worse)
    # jump_risk_pct = additional tail risk from jumps / total tail risk
    cvar_diff = abs(cvar_with) - abs(cvar_without)
    cvar_diff = max(cvar_diff, 0.0)
    jump_risk_pct = cvar_diff / abs(cvar_with) if abs(cvar_with) > 0 else 0.0

    # Standard deviation comparison (for reference)
    std_with = final_with.std()
    std_without = final_without.std()
    std_jump_contrib = max(0, std_with - std_without)
    std_jump_pct = std_jump_contrib / std_with if std_with > 0 else 0

    # Median outcome difference
    median_with = np.median(final_with)
    median_without = np.median(final_without)

    interpretation = (
        f"Jump risk contributes {jump_risk_pct*100:.1f}% of total tail risk "
        f"(CVaR-based). "
        f"CVaR(5%): with jumps={cvar_with*100:.2f}%, "
        f"without jumps={cvar_without*100:.2f}%. "
        f"VaR(5%): with jumps={var_with*100:.2f}%, "
        f"without jumps={var_without*100:.2f}%"
    )

    return {
        "var_with_jump": np.percentile(final_with, 5),
        "var_without_jump": np.percentile(final_without, 5),
        "cvar_with_jump": cvar_with * S0,
        "cvar_without_jump": cvar_without * S0,
        "cvar_with_pct": cvar_with,
        "cvar_without_pct": cvar_without,
        "jump_risk_pct_cvar": jump_risk_pct,
        "jump_risk_pct_std": std_jump_pct,
        "jump_risk_pct": jump_risk_pct,  # primary metric
        "std_with_jump": std_with,
        "std_without_jump": std_without,
        "median_with_jump": median_with,
        "median_without_jump": median_without,
        "interpretation": interpretation,
    }


def compare_domains(var_a: VaRResult, var_b: VaRResult) -> pd.DataFrame:
    """Cross-domain VaR comparison table"""
    data = {
        "Metric": [
            "Domain", "Confidence", "VaR (%)", "CVaR (%)",
            "Median Final", "Depletion Prob", "Max Drawdown VaR",
            "Jump Risk (%)",
        ],
        "Case A (Natural Disaster)": [
            var_a.domain,
            f"{var_a.confidence_level*100:.0f}%",
            f"{var_a.var_pct*100:.2f}%",
            f"{var_a.cvar_pct*100:.2f}%",
            f"${var_a.median_final:,.0f}",
            f"{var_a.depletion_prob*100:.1f}%",
            "N/A",
            f"{var_a.jump_risk_pct*100:.1f}%",
        ],
        "Case B (Geopolitical)": [
            var_b.domain,
            f"{var_b.confidence_level*100:.0f}%",
            f"{var_b.var_pct*100:.2f}%",
            f"{var_b.cvar_pct*100:.2f}%",
            f"${var_b.median_final:,.0f}",
            "N/A",
            f"{var_b.max_drawdown_var*100:.2f}%",
            f"{var_b.jump_risk_pct*100:.1f}%",
        ],
    }
    return pd.DataFrame(data)


# ========================== Self-test ==========================

def _run_self_tests():
    print("=" * 60)
    print("var_calculator.py self-test")
    print("=" * 60)

    from .jump_diffusion_engine import (
        JumpDiffusionParams, simulate_jump_diffusion, simulate_paired_paths
    )

    # Test 1: Case A VaR with realistic parameters
    print("\n[Test 1] Case A VaR (realistic params)")
    params_a = JumpDiffusionParams(
        mu=0.03, sigma=0.05, lam=0.5,
        mu_j=-0.10, sigma_j=0.08,  # ~10% loss per jump
        S0=1e9, T=10, dt=1.0, n_sims=5000
    )
    result_a = simulate_jump_diffusion(params_a, seed=42)
    var_a = compute_var_caseA(result_a, reserve_balance=2e8, confidence=0.95)
    print(f"  VaR: {var_a.var_pct*100:.2f}% (${var_a.var_absolute:,.0f})")
    print(f"  CVaR: {var_a.cvar_pct*100:.2f}%")
    print(f"  Depletion: {var_a.depletion_prob*100:.1f}%")
    assert -0.80 < var_a.var_pct < 0, f"VaR {var_a.var_pct:.2f} out of realistic range"
    print("  [PASS]")

    # Test 2: Case B VaR with realistic parameters
    print("\n[Test 2] Case B VaR (realistic params)")
    params_b = JumpDiffusionParams(
        mu=0.10, sigma=0.25, lam=3.0,
        mu_j=-0.05, sigma_j=0.03,  # ~5% jump per event
        S0=250, T=2, dt=1/252, n_sims=5000
    )
    result_b = simulate_jump_diffusion(params_b, seed=42)
    var_b = compute_var_caseB(result_b, confidence=0.95)
    print(f"  VaR: {var_b.var_pct*100:.2f}% (${var_b.var_absolute:,.0f})")
    print(f"  Max DD VaR: {var_b.max_drawdown_var*100:.2f}%")
    assert -0.90 < var_b.var_pct < 0, f"VaR {var_b.var_pct:.2f} out of range"
    assert var_b.max_drawdown_var < 0
    print("  [PASS]")

    # Test 3: FIXED decompose_risk with paired paths
    print("\n[Test 3] decompose_risk with paired paths (FIXED)")
    paired = simulate_paired_paths(params_a, seed=42)
    decomp = decompose_risk(paired)
    print(f"  {decomp['interpretation']}")
    print(f"  CVaR-based jump risk: {decomp['jump_risk_pct_cvar']*100:.1f}%")
    print(f"  Std-based jump risk:  {decomp['jump_risk_pct_std']*100:.1f}%")
    # With paired paths, the difference should be > 0
    assert decomp['jump_risk_pct'] > 0, "Jump risk should be > 0 with paired paths"
    print("  [PASS] Jump risk > 0 (bug fixed)")

    # Test 4: Paired decomposition for Case B
    print("\n[Test 4] Case B paired decomposition")
    paired_b = simulate_paired_paths(params_b, seed=42)
    decomp_b = decompose_risk(paired_b)
    print(f"  {decomp_b['interpretation']}")
    assert decomp_b['jump_risk_pct'] > 0, "Jump risk should be > 0"
    print("  [PASS]")

    # Test 5: Compare domains table
    print("\n[Test 5] Cross-domain comparison table")
    result_a["jump_risk_pct"] = decomp["jump_risk_pct"]
    result_b["jump_risk_pct"] = decomp_b["jump_risk_pct"]
    var_a = compute_var_caseA(result_a, reserve_balance=2e8)
    var_b = compute_var_caseB(result_b)
    compare_df = compare_domains(var_a, var_b)
    print(compare_df.to_string(index=False))
    assert len(compare_df) == 8
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_self_tests()
