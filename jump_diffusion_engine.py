"""
=============================================================================
FILE: src/jump_diffusion_engine.py
PURPOSE: 跳跃扩散模型蒙特卡洛模拟引擎 (共享核心模块)
RELATED:
  - 风险清单 #4: 蒙特卡洛模拟太慢 → 全向量化计算
  - 风险清单 #5: 模拟结果相似 → simulate_paired_paths() 支持共享噪声对比
  - 风险 #3 修复: decompose_risk 核心 bug → 同一 Z_continuous 用于两种路径

数学模型:
    dS_t = μ·S_t·dt + σ·S_t·dW_t + J·S_t·dN_t

离散化方案 (对数空间):
    S_{t+dt} = S_t · exp[(μ - σ²/2)·dt + σ·√dt·Z₁ + ΣJᵢ]

其中:
    Z₁   ~ N(0,1)           标准正态 (连续波动)
    N_t  ~ Poisson(λ·dt)    泊松计数 (跳跃发生次数)
    J    ~ N(μ_j, σ_j²)     跳跃幅度
=============================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class JumpDiffusionParams:
    """
    跳跃扩散模型参数容器

    Attributes:
        mu:       漂移率 (年化)
        sigma:    波动率 (年化)
        lam:      泊松过程强度 λ (年均跳跃次数)
        mu_j:     跳跃幅度的对数均值 (负值 = 损失)
        sigma_j:  跳跃幅度的对数标准差
        S0:       初始值
        T:        模拟总时长 (年)
        dt:       时间步长 (默认 1/252 = 日频)
        n_sims:   模拟路径数 (默认 10,000)
    """
    mu: float
    sigma: float
    lam: float
    mu_j: float
    sigma_j: float
    S0: float
    T: float
    dt: float = 1.0 / 252
    n_sims: int = 10_000

    @property
    def n_steps(self) -> int:
        return int(self.T / self.dt)

    def validate(self) -> list:
        """验证参数合理性, 返回警告列表"""
        warns = []
        if self.sigma < 0:
            warns.append(f"sigma={self.sigma} < 0")
        if self.lam < 0:
            warns.append(f"lam={self.lam} < 0")
        if self.sigma_j < 0:
            warns.append(f"sigma_j={self.sigma_j} < 0")
        if self.S0 <= 0:
            warns.append(f"S0={self.S0} <= 0")
        if self.T <= 0:
            warns.append(f"T={self.T} <= 0")
        if self.dt <= 0:
            warns.append(f"dt={self.dt} <= 0")
        if self.n_sims < 1:
            warns.append(f"n_sims={self.n_sims} < 1")
        if self.n_steps > 1_000_000:
            warns.append(f"n_steps={self.n_steps} too large")
        return warns

    def __str__(self) -> str:
        freq = "daily" if abs(self.dt - 1/252) < 0.001 else "annual"
        return (
            f"JumpDiffusionParams(mu={self.mu:.4f}, sigma={self.sigma:.4f}, "
            f"lam={self.lam:.2f}, mu_j={self.mu_j:.4f}, "
            f"S0={self.S0:,.2f}, T={self.T}yr, freq={freq})"
        )


def _build_paths(
    S0: float,
    log_returns: np.ndarray
) -> np.ndarray:
    """从对数收益率构建价格路径"""
    cumulative_log = np.zeros((log_returns.shape[0], log_returns.shape[1] + 1))
    cumulative_log[:, 1:] = np.cumsum(log_returns, axis=1)
    return S0 * np.exp(cumulative_log)


def _generate_jump_sums(
    n_sims: int,
    n_steps: int,
    lam: float,
    dt: float,
    mu_j: float,
    sigma_j: float,
    rng: np.random.Generator
) -> tuple:
    """
    生成跳跃计数和跳跃幅度总和

    Returns:
        (jump_counts, jump_sums)
    """
    jump_counts = rng.poisson(lam * dt, (n_sims, n_steps))
    max_jumps = int(jump_counts.max())
    jump_sums = np.zeros((n_sims, n_steps))

    if max_jumps > 0:
        all_jump_sizes = rng.normal(
            mu_j, sigma_j, (n_sims, n_steps, max_jumps)
        )
        for k in range(1, max_jumps + 1):
            mask = (jump_counts >= k)
            jump_sums[mask] += all_jump_sizes[:, :, k - 1][mask]

    return jump_counts, jump_sums


def simulate_jump_diffusion(
    params: JumpDiffusionParams,
    seed: Optional[int] = None,
    include_jump_details: bool = False,
    external_Z: Optional[np.ndarray] = None,
    external_jump_counts: Optional[np.ndarray] = None,
    external_jump_sums: Optional[np.ndarray] = None
) -> dict:
    """
    执行跳跃扩散蒙特卡洛模拟 (完全向量化)

    Args:
        params: 模型参数
        seed: 随机种子 (可复现)
        include_jump_details: 是否返回每步的跳跃细节
        external_Z: 预生成的连续随机数矩阵 (n_sims, n_steps)
                    用于 paired 模拟确保同一噪声源
        external_jump_counts: 预生成的跳跃计数
        external_jump_sums: 预生成的跳跃幅度总和

    Returns:
        dict with paths, log_returns, final_values, jump_counts, jump_sums
    """
    warnings_list = params.validate()
    if warnings_list:
        raise ValueError(f"参数验证失败:\n" + "\n".join(warnings_list))

    rng = np.random.default_rng(seed)
    n_sims = params.n_sims
    n_steps = params.n_steps

    # ---- 连续波动 ----
    if external_Z is not None:
        Z_continuous = external_Z
    else:
        Z_continuous = rng.standard_normal((n_sims, n_steps))

    # ---- 跳跃组件 ----
    if external_jump_counts is not None and external_jump_sums is not None:
        jump_counts = external_jump_counts
        jump_sums = external_jump_sums
    elif params.lam > 0:
        jump_counts, jump_sums = _generate_jump_sums(
            n_sims, n_steps, params.lam, params.dt,
            params.mu_j, params.sigma_j, rng
        )
    else:
        jump_counts = np.zeros((n_sims, n_steps), dtype=int)
        jump_sums = np.zeros((n_sims, n_steps))

    # ---- 组装对数收益率 ----
    drift = (params.mu - 0.5 * params.sigma ** 2) * params.dt
    diffusion = params.sigma * np.sqrt(params.dt) * Z_continuous
    log_returns = drift + diffusion + jump_sums

    paths = _build_paths(params.S0, log_returns)

    result = {
        'paths': paths,
        'log_returns': log_returns,
        'final_values': paths[:, -1],
        'jump_counts': jump_counts,
        'jump_sums': jump_sums,
    }

    if include_jump_details:
        result['has_jump'] = (jump_counts > 0)
        result['n_total_jumps'] = jump_counts.sum(axis=1)

    return result


def simulate_paired_paths(
    params: JumpDiffusionParams,
    seed: Optional[int] = None
) -> dict:
    """
    生成配对路径: 同一 Z_continuous 下的有跳跃 vs 无跳跃路径

    这是 decompose_risk 的正确实现基础。
    两次模拟共享完全相同的连续扩散噪声, 唯一差异是跳跃组件。

    Args:
        params: 跳跃扩散参数
        seed: 随机种子

    Returns:
        dict:
            'paths_with_jump':    有跳跃的路径 (n_sims, n_steps+1)
            'paths_without_jump': 无跳跃的路径 (n_sims, n_steps+1)
            'log_returns_with':   有跳跃的对数收益率
            'log_returns_without':无跳跃的对数收益率
            'jump_counts':        跳跃计数
            'jump_sums':          跳跃幅度总和
            'Z_continuous':       共享的连续随机数
    """
    rng = np.random.default_rng(seed)
    n_sims = params.n_sims
    n_steps = params.n_steps

    # 共享噪声源
    Z_continuous = rng.standard_normal((n_sims, n_steps))

    # 跳跃组件 (只生成一次)
    if params.lam > 0:
        jump_counts, jump_sums = _generate_jump_sums(
            n_sims, n_steps, params.lam, params.dt,
            params.mu_j, params.sigma_j, rng
        )
    else:
        jump_counts = np.zeros((n_sims, n_steps), dtype=int)
        jump_sums = np.zeros((n_sims, n_steps))

    # 有跳跃路径
    result_with = simulate_jump_diffusion(
        params, seed=None,
        external_Z=Z_continuous,
        external_jump_counts=jump_counts,
        external_jump_sums=jump_sums
    )

    # 无跳跃路径 (共享 Z, 跳跃为 0)
    zero_jumps = np.zeros((n_sims, n_steps))
    zero_counts = np.zeros((n_sims, n_steps), dtype=int)
    result_without = simulate_jump_diffusion(
        params, seed=None,
        external_Z=Z_continuous,
        external_jump_counts=zero_counts,
        external_jump_sums=zero_jumps
    )

    return {
        'paths_with_jump': result_with['paths'],
        'paths_without_jump': result_without['paths'],
        'log_returns_with': result_with['log_returns'],
        'log_returns_without': result_without['log_returns'],
        'final_with_jump': result_with['final_values'],
        'final_without_jump': result_without['final_values'],
        'jump_counts': jump_counts,
        'jump_sums': jump_sums,
        'Z_continuous': Z_continuous,
    }


def simulate_gbm_only(
    params: JumpDiffusionParams,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    仅模拟 GBM (无跳跃), 用于独立参考
    """
    gbm_params = JumpDiffusionParams(
        mu=params.mu, sigma=params.sigma,
        lam=0.0, mu_j=0.0, sigma_j=0.0,
        S0=params.S0, T=params.T,
        dt=params.dt, n_sims=params.n_sims
    )
    result = simulate_jump_diffusion(gbm_params, seed=seed)
    return result['paths']


# ==================== 单元测试 ====================
def _run_self_tests():
    """模块内自测"""
    print("=" * 60)
    print("jump_diffusion_engine.py self-test")
    print("=" * 60)

    # Test 1: 基本模拟
    print("\n[Test 1] Basic simulation")
    params = JumpDiffusionParams(
        mu=0.05, sigma=0.2, lam=1.0,
        mu_j=-0.03, sigma_j=0.02,
        S0=100.0, T=1.0, dt=1/252, n_sims=100
    )
    result = simulate_jump_diffusion(params, seed=42)
    paths = result['paths']
    assert paths.shape == (100, 253), f"Expected (100, 253), got {paths.shape}"
    assert paths[0, 0] == 100.0
    assert np.all(paths > 0)
    print(f"  [PASS] Shape: {paths.shape}, Mean: {result['final_values'].mean():.2f}")

    # Test 2: GBM degeneration
    print("\n[Test 2] lambda=0 degenerates to GBM")
    gbm_params = JumpDiffusionParams(
        mu=0.05, sigma=0.2, lam=0.0,
        mu_j=0.0, sigma_j=0.0,
        S0=100.0, T=1.0, dt=1/252, n_sims=1000
    )
    gbm_result = simulate_jump_diffusion(gbm_params, seed=42)
    expected_mean = 100.0 * np.exp(0.05)
    actual_mean = gbm_result['final_values'].mean()
    error_pct = abs(actual_mean - expected_mean) / expected_mean * 100
    assert error_pct < 5, f"GBM mean error {error_pct:.2f}% > 5%"
    print(f"  [PASS] Theoretical: {expected_mean:.2f}, Actual: {actual_mean:.2f}, Err: {error_pct:.2f}%")

    # Test 3: Jump statistics
    print("\n[Test 3] Jump statistics")
    jump_params = JumpDiffusionParams(
        mu=0.05, sigma=0.1, lam=5.0,
        mu_j=-0.05, sigma_j=0.02,
        S0=100.0, T=1.0, dt=1/252, n_sims=1000
    )
    jump_result = simulate_jump_diffusion(jump_params, seed=42, include_jump_details=True)
    avg_jumps = jump_result['n_total_jumps'].mean()
    assert 3.0 < avg_jumps < 7.0, f"Jump freq {avg_jumps:.1f} out of range"
    print(f"  [PASS] Expected ~5.0, Actual: {avg_jumps:.1f}")

    # Test 4: Parameter validation
    print("\n[Test 4] Parameter validation")
    bad_params = JumpDiffusionParams(
        mu=0.05, sigma=-0.1, lam=1.0,
        mu_j=0.0, sigma_j=0.02, S0=100.0, T=1.0
    )
    try:
        simulate_jump_diffusion(bad_params)
        assert False, "Should raise ValueError"
    except ValueError:
        print("  [PASS] Correctly raises ValueError")

    # Test 5: Reproducibility
    print("\n[Test 5] Seed reproducibility")
    r1 = simulate_jump_diffusion(params, seed=123)
    r2 = simulate_jump_diffusion(params, seed=123)
    assert np.allclose(r1['paths'], r2['paths'])
    print("  [PASS] seed=123 produces identical results")

    # Test 6: Paired paths share noise
    print("\n[Test 6] Paired paths share Z_continuous")
    paired = simulate_paired_paths(params, seed=42)
    # With small jumps, the paths should be very similar
    diff = np.abs(paired['paths_with_jump'] - paired['paths_without_jump'])
    max_diff_pct = (diff / paired['paths_without_jump']).max() * 100
    # The difference should come entirely from jumps (small in this case)
    print(f"  Max path difference: {max_diff_pct:.3f}%")
    # Both paths share the same noise - verify by checking that the
    # GBM component is identical
    # log_return_with - log_return_without = jump_sums (approximately)
    log_diff = paired['log_returns_with'] - paired['log_returns_without']
    assert np.allclose(log_diff, paired['jump_sums'], atol=1e-10), \
        "Difference should equal jump_sums"
    print("  [PASS] log_return_diff == jump_sums (shared Z verified)")

    # Test 7: Speed
    print("\n[Test 7] Speed benchmark")
    import time
    big_params = JumpDiffusionParams(
        mu=0.05, sigma=0.2, lam=2.0,
        mu_j=-0.03, sigma_j=0.02,
        S0=100.0, T=10.0, dt=1/252, n_sims=10_000
    )
    t0 = time.time()
    big_result = simulate_jump_diffusion(big_params, seed=42)
    elapsed = time.time() - t0
    print(f"  10,000 x 2,520 in {elapsed:.2f}s")
    assert elapsed < 30
    print("  [PASS] Speed acceptable")

    print("\n" + "=" * 60)
    print("All tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_self_tests()
