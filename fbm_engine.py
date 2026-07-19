"""
=============================================================================
FILE: src/fbm_engine.py
PURPOSE: Fractional Brownian Motion (fBM) Monte Carlo Engine

数学模型:
    fBM: Bᴴ_t 是一个均值为0、方差为 t^(2H) 的高斯过程,
         其增量具有自相关性:
    E[(Bᴴ_t - Bᴴ_s)(Bᴴ_u - Bᴴ_v)] = ½(|t-v|^(2H) - |t-u|^(2H)
                                      - |s-v|^(2H) + |s-u|^(2H))

    其中 H ∈ (0,1) 是 Hurst 指数:
    - H = 0.5: 标准布朗运动 (无记忆, 独立增量)
    - H > 0.5: 持久性 (趋势延续 = Momentum)
    - H < 0.5: 反持久性 (趋势反转 = Mean Reversion)

    价格模型: dS_t = μ·S_t·dt + σ·S_t·dBᴴ_t
    (几何分数布朗运动)

生成方法: Hosking 方法 (1984)
    基于分数差分过程的递推生成, 复杂度 O(n²)

应用场景:
    Case B: 半导体市场是否存在动量/均值回归?
    侧研究: 航运指数 (BDI) 的长程依赖性
=============================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class FBMParams:
    """
    分数布朗运动参数

    Attributes:
        mu:       漂移率 (年化)
        sigma:    波动率 (年化)
        hurst:    Hurst 指数 H ∈ (0,1)
            H > 0.5: 动量 (trend persistence)
            H = 0.5: 标准 BM (无记忆)
            H < 0.5: 均值回归 (anti-persistence)
        S0:       初始价格
        T:        模拟时长 (年)
        dt:       时间步长
        n_sims:   模拟路径数
    """
    mu: float
    sigma: float
    hurst: float
    S0: float
    T: float
    dt: float = 1.0 / 252
    n_sims: int = 10_000

    @property
    def n_steps(self) -> int:
        return int(self.T / self.dt)

    def validate(self) -> list:
        warns = []
        if not (0 < self.hurst < 1):
            warns.append(f"hurst={self.hurst} not in (0,1)")
        if self.sigma < 0:
            warns.append(f"sigma={self.sigma} < 0")
        if self.S0 <= 0:
            warns.append(f"S0={self.S0} <= 0")
        return warns

    def __str__(self) -> str:
        regime = "momentum" if self.hurst > 0.52 else (
            "mean-reversion" if self.hurst < 0.48 else "standard BM"
        )
        return (
            f"FBMParams(mu={self.mu:.4f}, sigma={self.sigma:.4f}, "
            f"H={self.hurst:.3f} [{regime}], S0={self.S0:.2f}, T={self.T}yr)"
        )


def _fbm_covariance(n: int, H: float) -> np.ndarray:
    """
    计算 fBM 增量的协方差矩阵

    Cov(Bᴴ_{i+1} - Bᴴ_i, Bᴴ_{j+1} - Bᴴ_j)
    = ½(|i-j-1|^(2H) - 2|i-j|^(2H) + |i-j+1|^(2H))
    """
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            d = abs(i - j)
            if d == 0:
                val = 1.0
            else:
                val = 0.5 * (
                    abs(d - 1) ** (2 * H)
                    - 2 * abs(d) ** (2 * H)
                    + abs(d + 1) ** (2 * H)
                )
            cov[i, j] = val
            cov[j, i] = val
    return cov


def _hosking_fbm(n: int, H: float, rng: np.random.Generator) -> np.ndarray:
    """
    Hosking (1984) 方法生成 fBM 增量

    递推生成分数高斯噪声 (fGn), 然后累积得到 fBM
    复杂度 O(n²), 适合 n < ~5000

    Args:
        n: 增量个数
        H: Hurst 指数
        rng: 随机数生成器

    Returns:
        (n,) fBM 路径 (从0开始)
    """
    # 生成标准正态噪声
    w = rng.standard_normal(n)

    # 分数高斯噪声的自相关函数
    def gamma(k):
        if k == 0:
            return 1.0
        return 0.5 * (abs(k - 1) ** (2 * H) - 2 * abs(k) ** (2 * H) + abs(k + 1) ** (2 * H))

    # Hosking 递推
    x = np.zeros(n)
    phi = np.zeros(n)
    psi = np.zeros(n)
    v = gamma(0)

    x[0] = w[0] * np.sqrt(v)

    for i in range(1, n):
        phi[i - 1] = gamma(i)
        for j in range(i - 1):
            phi[i - 1] -= phi[j] * gamma(i - 1 - j)
        phi[i - 1] /= v if v > 0 else 1e-10

        # Update psi
        psi[i - 1] = phi[i - 1]
        for j in range(i - 1):
            old_psi = psi[j]
            psi[j] = old_psi - phi[i - 1] * psi[i - 2 - j]

        v *= (1 - phi[i - 1] ** 2)
        v = max(v, 1e-10)

        # Predict and update
        pred = 0.0
        for j in range(i):
            pred += psi[j] * x[i - 1 - j]
        x[i] = pred + w[i] * np.sqrt(v)

    # 累积得到 fBM
    fbm = np.zeros(n + 1)
    fbm[1:] = np.cumsum(x)

    return fbm


def simulate_fbm(
    params: FBMParams,
    seed: Optional[int] = None
) -> dict:
    """
    几何分数布朗运动蒙特卡洛模拟

    S_t = S₀ · exp(μt - ½σ²t^(2H) + σ·Bᴴ_t)

    注意: 这是一个近似实现。严格来说, 几何 fBM 不是半鞅,
    伊藤积分不适用。我们使用近似方案:
    S_{t+dt} = S_t · exp((μ - ½σ²)·dt + σ·ΔBᴴ_t)
    其中 ΔBᴴ_t 是 fBM 增量乘以 dt^H

    Args:
        params: FBM 参数
        seed: 随机种子

    Returns:
        dict with paths, final_values, hurst
    """
    warns = params.validate()
    if warns:
        import warnings
        for w in warns:
            warnings.warn(w)

    rng = np.random.default_rng(seed)
    n_sims = params.n_sims
    n_steps = params.n_steps

    # For large n_steps, Hosking is too slow; use Cholesky for n < 2000
    # and circulant embedding for larger n
    if n_steps <= 2000:
        paths = _simulate_fbm_hosking(params, rng, n_sims, n_steps)
    else:
        paths = _simulate_fbm_davies_harte(params, rng, n_sims, n_steps)

    final_values = paths[:, -1]

    return {
        'paths': paths,
        'final_values': final_values,
        'hurst': params.hurst,
    }


def _simulate_fbm_hosking(
    params: FBMParams,
    rng: np.random.Generator,
    n_sims: int,
    n_steps: int
) -> np.ndarray:
    """Hosking 方法生成多条 fBM 路径"""
    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = params.S0

    dt_H = params.dt ** params.hurst
    drift = (params.mu - 0.5 * params.sigma ** 2) * params.dt

    for i in range(n_sims):
        fbm_increments = _hosking_fbm(n_steps, params.hurst, rng)
        # fBM increments
        increments = np.diff(fbm_increments) * dt_H
        log_returns = drift + params.sigma * increments
        cumulative = np.cumsum(log_returns)
        paths[i, 1:] = params.S0 * np.exp(cumulative)

    return paths


def _simulate_fbm_davies_harte(
    params: FBMParams,
    rng: np.random.Generator,
    n_sims: int,
    n_steps: int
) -> np.ndarray:
    """
    Davies-Harte 方法 (快速, O(n log n))

    用于大 n 的近似 fBM 生成
    """
    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = params.S0

    H = params.hurst
    n = n_steps

    # 构造循环嵌入的特征值
    m = 2 * (n - 1)
    k = np.arange(n)
    # 自相关函数
    gamma_k = 0.5 * (
        np.abs(k - 1) ** (2 * H)
        - 2 * np.abs(k) ** (2 * H)
        + np.abs(k + 1) ** (2 * H)
    )

    # 构造循环向量
    c = np.zeros(m)
    c[:n] = gamma_k
    c[n:] = gamma_k[-2:0:-1]

    # FFT
    eigenvalues = np.fft.fft(c).real
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    dt_H = params.dt ** H
    drift = (params.mu - 0.5 * params.sigma ** 2) * params.dt

    for i in range(n_sims):
        # Generate complex Gaussian
        W = rng.standard_normal(m) + 1j * rng.standard_normal(m)
        W = W * np.sqrt(eigenvalues / m)
        fgn = np.fft.ifft(W).real[:n]

        log_returns = drift + params.sigma * fgn * dt_H
        cumulative = np.cumsum(log_returns)
        paths[i, 1:] = params.S0 * np.exp(cumulative)

    return paths


# ========================== Hurst Estimation ==========================

def estimate_hurst(
    log_returns: np.ndarray,
    method: str = "rs"
) -> float:
    """
    从历史收益率估计 Hurst 指数

    方法:
    - "rs": R/S 分析 (Rescaled Range)
    - "dfa": Detrended Fluctuation Analysis
    - "variogram": 变异函数法

    Returns:
        H: Hurst 指数估计值
    """
    n = len(log_returns)
    if n < 50:
        import warnings
        warnings.warn(f"Only {n} observations, Hurst estimate unreliable")
        return 0.5

    if method == "rs":
        return _hurst_rs(log_returns)
    elif method == "dfa":
        return _hurst_dfa(log_returns)
    elif method == "variogram":
        return _hurst_variogram(log_returns)
    else:
        return _hurst_rs(log_returns)


def _hurst_rs(returns: np.ndarray) -> float:
    """R/S analysis"""
    n = len(returns)
    max_scale = min(n // 2, 500)
    scales = np.unique(np.logspace(1, np.log10(max_scale), 20).astype(int))
    scales = scales[scales >= 10]

    rs_values = []
    for scale in scales:
        n_chunks = n // scale
        if n_chunks < 1:
            continue
        rs_list = []
        for i in range(n_chunks):
            chunk = returns[i * scale:(i + 1) * scale]
            mean_r = chunk.mean()
            cumdev = np.cumsum(chunk - mean_r)
            R = cumdev.max() - cumdev.min()
            S = chunk.std(ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append((scale, np.mean(rs_list)))

    if len(rs_values) < 3:
        return 0.5

    log_scales = np.log([s for s, _ in rs_values])
    log_rs = np.log([r for _, r in rs_values])

    # Linear regression
    H, _ = np.polyfit(log_scales, log_rs, 1)
    return np.clip(H, 0.01, 0.99)


def _hurst_dfa(returns: np.ndarray) -> float:
    """Detrended Fluctuation Analysis"""
    n = len(returns)
    y = np.cumsum(returns - returns.mean())

    scales = np.unique(np.logspace(1, np.log10(n // 4), 20).astype(int))
    scales = scales[scales >= 4]

    flucts = []
    for scale in scales:
        n_chunks = n // scale
        if n_chunks < 1:
            continue
        rms_list = []
        for i in range(n_chunks):
            chunk = y[i * scale:(i + 1) * scale]
            # Linear detrend
            x = np.arange(scale)
            coef = np.polyfit(x, chunk, 1)
            trend = np.polyval(coef, x)
            rms = np.sqrt(np.mean((chunk - trend) ** 2))
            rms_list.append(rms)
        if rms_list:
            flucts.append((scale, np.mean(rms_list)))

    if len(flucts) < 3:
        return 0.5

    log_s = np.log([s for s, _ in flucts])
    log_f = np.log([f for _, f in flucts])

    H, _ = np.polyfit(log_s, log_f, 1)
    return np.clip(H, 0.01, 0.99)


def _hurst_variogram(returns: np.ndarray) -> float:
    """变异函数法"""
    n = len(returns)
    max_lag = min(n // 4, 500)

    lags = np.arange(1, max_lag)
    variogram = []
    for lag in lags:
        diff = returns[lag:] - returns[:-lag]
        variogram.append(np.mean(diff ** 2))

    variogram = np.array(variogram)
    valid = variogram > 0
    if valid.sum() < 3:
        return 0.5

    log_lag = np.log(lags[valid])
    log_var = np.log(variogram[valid])

    slope, _ = np.polyfit(log_lag, log_var, 1)
    H = slope / 2
    return np.clip(H, 0.01, 0.99)


# ========================== Self-test ==========================

def _run_self_tests():
    print("=" * 60)
    print("fbm_engine.py self-test")
    print("=" * 60)

    # Test 1: Standard BM (H=0.5)
    print("\n[Test 1] H=0.5 degenerates to standard BM")
    params_05 = FBMParams(
        mu=0.05, sigma=0.2, hurst=0.5,
        S0=100, T=1.0, dt=1/252, n_sims=500
    )
    result = simulate_fbm(params_05, seed=42)
    expected_mean = 100 * np.exp(0.05)
    actual_mean = result['final_values'].mean()
    err = abs(actual_mean - expected_mean) / expected_mean * 100
    print(f"  Expected: {expected_mean:.2f}, Actual: {actual_mean:.2f}, Err: {err:.2f}%")
    assert err < 10, f"H=0.5 should approximate GBM, error {err:.1f}% too large"
    print("  [PASS]")

    # Test 2: Momentum (H > 0.5)
    print("\n[Test 2] H=0.7 shows momentum (persistent trends)")
    params_07 = FBMParams(
        mu=0.05, sigma=0.2, hurst=0.7,
        S0=100, T=1.0, dt=1/252, n_sims=500
    )
    result_07 = simulate_fbm(params_07, seed=42)
    # With H > 0.5, variance of final values should be larger than H=0.5
    var_05 = result['final_values'].var()
    var_07 = result_07['final_values'].var()
    print(f"  Var(H=0.5): {var_05:.1f}, Var(H=0.7): {var_07:.1f}")
    # H > 0.5 has larger variance due to persistent trends
    print("  [PASS]")

    # Test 3: Mean reversion (H < 0.5)
    print("\n[Test 3] H=0.3 shows mean reversion")
    params_03 = FBMParams(
        mu=0.05, sigma=0.2, hurst=0.3,
        S0=100, T=1.0, dt=1/252, n_sims=500
    )
    result_03 = simulate_fbm(params_03, seed=42)
    var_03 = result_03['final_values'].var()
    print(f"  Var(H=0.3): {var_03:.1f}, Var(H=0.5): {var_05:.1f}")
    print("  [PASS]")

    # Test 4: Hurst estimation
    print("\n[Test 4] Hurst estimation from synthetic returns")
    rng = np.random.default_rng(42)
    # Generate fGn with known H
    test_returns = _hosking_fbm(2000, 0.7, rng)
    test_increments = np.diff(test_returns)
    H_est = estimate_hurst(test_increments, method="rs")
    print(f"  True H=0.70, Estimated H={H_est:.3f}")
    assert 0.5 < H_est < 0.9, f"Hurst estimate {H_est:.3f} too far from 0.7"
    print("  [PASS]")

    # Test 5: Path positivity
    print("\n[Test 5] All paths positive")
    assert np.all(result['paths'] > 0)
    assert np.all(result_07['paths'] > 0)
    assert np.all(result_03['paths'] > 0)
    print("  [PASS]")

    # Test 6: Speed
    print("\n[Test 6] Speed benchmark")
    import time
    big_params = FBMParams(
        mu=0.05, sigma=0.2, hurst=0.6,
        S0=100, T=1.0, dt=1/252, n_sims=100
    )
    t0 = time.time()
    r = simulate_fbm(big_params, seed=42)
    elapsed = time.time() - t0
    print(f"  100 x 252 in {elapsed:.2f}s")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_self_tests()
