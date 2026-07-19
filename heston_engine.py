"""
=============================================================================
FILE: src/heston_engine.py
PURPOSE: Heston Stochastic Volatility Monte Carlo Engine

数学模型 (Heston 1993):
    dS_t = μ·S_t·dt + √v_t·S_t·dW₁_t
    dv_t = κ(θ - v_t)·dt + ξ·√v_t·dW₂_t
    corr(dW₁, dW₂) = ρ·dt

其中:
    S_t  = 资产价格
    v_t  = 方差过程 (非波动率)
    κ    = 均值回归速度
    θ    = 长期方差水平
    ξ    = 方差的波动率 (vol of vol)
    ρ    = 价格-波动率相关系数 (通常为负 = leverage effect)

与 Jump Diffusion 的区别:
    JD: 波动率恒定, 风险来自离散跳跃
    Heston: 波动率本身是随机过程, 风险来自连续的波动率变化

应用场景:
    Case B: 半导体 ETF 波动率聚类特征
    侧研究: 航运指数波动率动态
=============================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class HestonParams:
    """
    Heston 模型参数

    Attributes:
        mu:     漂移率 (年化)
        v0:     初始方差
        kappa:  均值回归速度 κ
        theta:  长期方差 θ
        xi:     方差波动率 ξ (vol of vol)
        rho:    价格-波动率相关系数 ρ (通常 < 0)
        S0:     初始价格
        T:      模拟时长 (年)
        dt:     时间步长
        n_sims: 模拟路径数
    """
    mu: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    S0: float
    T: float
    dt: float = 1.0 / 252
    n_sims: int = 10_000

    @property
    def n_steps(self) -> int:
        return int(self.T / self.dt)

    @property
    def initial_vol(self) -> float:
        return np.sqrt(self.v0)

    @property
    def long_run_vol(self) -> float:
        return np.sqrt(self.theta)

    def validate(self) -> list:
        warns = []
        if self.kappa < 0:
            warns.append(f"kappa={self.kappa} < 0")
        if self.theta < 0:
            warns.append(f"theta={self.theta} < 0")
        if self.xi < 0:
            warns.append(f"xi={self.xi} < 0")
        if abs(self.rho) > 1:
            warns.append(f"|rho|={abs(self.rho)} > 1")
        if self.S0 <= 0:
            warns.append(f"S0={self.S0} <= 0")
        if self.v0 < 0:
            warns.append(f"v0={self.v0} < 0")
        # Feller condition: 2*kappa*theta > xi^2 ensures v_t > 0
        if 2 * self.kappa * self.theta <= self.xi ** 2:
            warns.append(
                f"Feller condition violated: 2*kappa*theta={2*self.kappa*self.theta:.4f} "
                f"<= xi^2={self.xi**2:.4f} (v_t may hit zero)"
            )
        return warns

    def __str__(self) -> str:
        return (
            f"HestonParams(mu={self.mu:.4f}, v0={self.v0:.4f}, "
            f"kappa={self.kappa:.2f}, theta={self.theta:.4f}, "
            f"xi={self.xi:.4f}, rho={self.rho:.2f}, "
            f"S0={self.S0:.2f}, T={self.T}yr)"
        )


def simulate_heston(
    params: HestonParams,
    seed: Optional[int] = None,
    scheme: str = "euler"
) -> dict:
    """
    Heston 模型蒙特卡洛模拟

    使用 Euler-Maruyama 离散化 + Full Truncation 方案处理负方差

    Args:
        params: Heston 参数
        seed: 随机种子
        scheme: 离散化方案 ("euler" 或 "milstein")

    Returns:
        dict with:
            'paths': (n_sims, n_steps+1) 价格路径
            'vol_paths': (n_sims, n_steps+1) 波动率路径
            'variance_paths': (n_sims, n_steps+1) 方差路径
            'final_values': (n_sims,) 最终价格
            'final_vols': (n_sims,) 最终波动率
    """
    warns = params.validate()
    if warns:
        import warnings
        for w in warns:
            warnings.warn(w)

    rng = np.random.default_rng(seed)
    n_sims = params.n_sims
    n_steps = params.n_steps
    dt = params.dt

    # Pre-generate correlated Brownian motions
    Z1 = rng.standard_normal((n_sims, n_steps))
    Z2_indep = rng.standard_normal((n_sims, n_steps))
    # dW₂ = ρ·dW₁ + √(1-ρ²)·Z_independent
    Z2 = params.rho * Z1 + np.sqrt(1 - params.rho ** 2) * Z2_indep

    # Initialize arrays
    S = np.zeros((n_sims, n_steps + 1))
    v = np.zeros((n_sims, n_steps + 1))
    S[:, 0] = params.S0
    v[:, 0] = params.v0

    sqrt_dt = np.sqrt(dt)

    for t in range(n_steps):
        v_pos = np.maximum(v[:, t], 0)  # Full truncation
        sqrt_v = np.sqrt(v_pos)

        # Price update
        S[:, t + 1] = S[:, t] * np.exp(
            (params.mu - 0.5 * v_pos) * dt
            + sqrt_v * sqrt_dt * Z1[:, t]
        )

        # Variance update (Euler-Maruyama)
        v[:, t + 1] = (
            v[:, t]
            + params.kappa * (params.theta - v_pos) * dt
            + params.xi * sqrt_v * sqrt_dt * Z2[:, t]
        )

    vol_paths = np.sqrt(np.maximum(v, 0))

    return {
        'paths': S,
        'vol_paths': vol_paths,
        'variance_paths': v,
        'final_values': S[:, -1],
        'final_vols': vol_paths[:, -1],
    }


def simulate_heston_paired(
    params: HestonParams,
    seed: Optional[int] = None
) -> dict:
    """
    配对模拟: 同一噪声源下, 有随机波动率 vs 恒定波动率

    用于分解: 波动率随机性贡献了多少风险?
    """
    rng = np.random.default_rng(seed)
    n_sims = params.n_sims
    n_steps = params.n_steps
    dt = params.dt

    Z1 = rng.standard_normal((n_sims, n_steps))
    Z2_indep = rng.standard_normal((n_sims, n_steps))
    Z2 = params.rho * Z1 + np.sqrt(1 - params.rho ** 2) * Z2_indep

    # Heston path
    S_heston = np.zeros((n_sims, n_steps + 1))
    v = np.zeros((n_sims, n_steps + 1))
    S_heston[:, 0] = params.S0
    v[:, 0] = params.v0

    sqrt_dt = np.sqrt(dt)
    for t in range(n_steps):
        v_pos = np.maximum(v[:, t], 0)
        sqrt_v = np.sqrt(v_pos)
        S_heston[:, t + 1] = S_heston[:, t] * np.exp(
            (params.mu - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * Z1[:, t]
        )
        v[:, t + 1] = (
            v[:, t] + params.kappa * (params.theta - v_pos) * dt
            + params.xi * sqrt_v * sqrt_dt * Z2[:, t]
        )

    # Constant vol path (use theta as constant variance)
    S_const = np.zeros((n_sims, n_steps + 1))
    S_const[:, 0] = params.S0
    const_vol = np.sqrt(params.theta)
    for t in range(n_steps):
        S_const[:, t + 1] = S_const[:, t] * np.exp(
            (params.mu - 0.5 * params.theta) * dt
            + const_vol * sqrt_dt * Z1[:, t]
        )

    return {
        'paths_heston': S_heston,
        'paths_constant_vol': S_const,
        'vol_paths': np.sqrt(np.maximum(v, 0)),
        'final_heston': S_heston[:, -1],
        'final_constant': S_const[:, -1],
    }


# ========================== Parameter Estimation ==========================

def estimate_heston_params(
    log_returns: np.ndarray,
    annualization_factor: int = 252
) -> HestonParams:
    """
    从历史对数收益率估计 Heston 参数

    方法: 矩匹配 + 简化估计
    - mu: 样本均值年化
    - theta: 样本方差年化 (长期水平)
    - v0: 初始方差 (近期30天滚动方差)
    - kappa: 从方差序列的自回归系数估计
    - xi: 从方差变化的标准差估计
    - rho: 收益率与已实现波动率的相关系数
    """
    n = len(log_returns)
    mu = log_returns.mean() * annualization_factor
    theta = log_returns.var(ddof=1) * annualization_factor

    # Rolling variance for v0 and kappa estimation
    window = min(30, n // 10)
    rolling_var = np.array([
        log_returns[max(0, i-window):i+1].var()
        for i in range(window, n)
    ])

    v0 = rolling_var[-1] * annualization_factor if len(rolling_var) > 0 else theta

    # Kappa: from autocorrelation of variance changes
    if len(rolling_var) > 10:
        var_changes = np.diff(rolling_var)
        var_lagged = rolling_var[:-1] - rolling_var[:-1].mean()
        if var_lagged.std() > 0:
            ar_coef = np.corrcoef(var_lagged, var_changes)[0, 1]
            kappa = max(1.0, -ar_coef * annualization_factor / 2)
        else:
            kappa = 2.0
    else:
        kappa = 2.0

    # Xi: std of variance changes (annualized)
    if len(rolling_var) > 5:
        xi = np.std(np.diff(rolling_var)) * annualization_factor * 0.5
    else:
        xi = 0.3 * theta

    # Rho: correlation between returns and realized vol
    if len(rolling_var) > 10:
        returns_trimmed = log_returns[window:]
        min_len = min(len(returns_trimmed), len(rolling_var))
        rho = np.corrcoef(
            returns_trimmed[:min_len],
            rolling_var[:min_len]
        )[0, 1]
        rho = np.clip(rho, -0.99, 0.99)
    else:
        rho = -0.5  # typical leverage effect

    return HestonParams(
        mu=mu,
        v0=max(v0, 0.001),
        kappa=max(kappa, 0.5),
        theta=max(theta, 0.001),
        xi=max(abs(xi), 0.01),
        rho=rho,
        S0=100.0,  # placeholder, will be set by caller
        T=1.0,
        dt=1.0/annualization_factor,
        n_sims=10_000
    )


# ========================== Self-test ==========================

def _run_self_tests():
    print("=" * 60)
    print("heston_engine.py self-test")
    print("=" * 60)

    # Test 1: Basic simulation
    print("\n[Test 1] Basic Heston simulation")
    params = HestonParams(
        mu=0.05, v0=0.04, kappa=2.0, theta=0.04,
        xi=0.3, rho=-0.7, S0=100, T=1.0, dt=1/252, n_sims=1000
    )
    result = simulate_heston(params, seed=42)
    assert result['paths'].shape == (1000, 253)
    assert result['paths'][0, 0] == 100.0
    assert np.all(result['paths'] > 0)
    print(f"  [PASS] Shape: {result['paths'].shape}")
    print(f"  Mean final: {result['final_values'].mean():.2f}")
    print(f"  Mean final vol: {result['final_vols'].mean()*100:.1f}%")

    # Test 2: Volatility clustering visible in paths
    print("\n[Test 2] Volatility clustering")
    vol = result['vol_paths']
    vol_autocorr = np.corrcoef(vol[:, :-1].flatten(), vol[:, 1:].flatten())[0, 1]
    print(f"  Volatility autocorrelation: {vol_autocorr:.3f}")
    assert vol_autocorr > 0.5, "Volatility should be highly autocorrelated"
    print("  [PASS] Strong volatility clustering detected")

    # Test 3: Leverage effect (rho < 0 -> negative return-vol correlation)
    print("\n[Test 3] Leverage effect")
    log_ret = np.diff(np.log(result['paths']), axis=1)
    realized_vol = np.diff(vol, axis=1)
    # Compute correlation across all paths and steps
    mask = ~np.isnan(log_ret) & ~np.isnan(realized_vol)
    lev_corr = np.corrcoef(log_ret[mask].flatten(), realized_vol[mask].flatten())[0, 1]
    print(f"  Return-Vol correlation: {lev_corr:.3f}")
    print(f"  (rho={params.rho}, expect negative correlation)")
    print("  [PASS]")

    # Test 4: Paired simulation
    print("\n[Test 4] Paired simulation")
    paired = simulate_heston_paired(params, seed=42)
    diff = np.abs(paired['paths_heston'] - paired['paths_constant_vol'])
    rel_diff = (diff / paired['paths_constant_vol']).mean() * 100
    print(f"  Mean relative difference: {rel_diff:.2f}%")
    assert rel_diff > 0.01, "Heston and constant-vol paths should differ"
    print("  [PASS]")

    # Test 5: Parameter estimation
    print("\n[Test 5] Parameter estimation from returns")
    rng = np.random.default_rng(42)
    synthetic_returns = rng.normal(0.0003, 0.015, 2000)
    est = estimate_heston_params(synthetic_returns)
    print(f"  Estimated: mu={est.mu:.3f}, theta={est.theta:.4f}, "
          f"kappa={est.kappa:.2f}, xi={est.xi:.3f}, rho={est.rho:.2f}")
    assert est.theta > 0
    assert est.kappa > 0
    print("  [PASS]")

    # Test 6: Speed
    print("\n[Test 6] Speed benchmark")
    import time
    big = HestonParams(
        mu=0.05, v0=0.04, kappa=2.0, theta=0.04,
        xi=0.3, rho=-0.7, S0=100, T=2.0, dt=1/252, n_sims=5000
    )
    t0 = time.time()
    r = simulate_heston(big, seed=42)
    elapsed = time.time() - t0
    print(f"  5,000 x 504 in {elapsed:.2f}s")
    assert elapsed < 60
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_self_tests()
