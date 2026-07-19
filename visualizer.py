"""
=============================================================================
FILE: src/visualizer.py
PURPOSE: 可视化模块 — 论文图表生成
RELATED: 所有图表均可在 output/figures/ 下查看

图表清单:
  1. plot_simulation_fan:     模拟路径扇形图 (Fan Chart)
  2. plot_final_distribution: 最终值分布直方图 + 核密度
  3. plot_var_comparison:     双域 VaR 对比柱状图
  4. plot_risk_decomposition: 风险来源分解 (跳跃 vs 连续)
  5. plot_sensitivity_heatmap: 敏感性分析热力图
  6. plot_tail_comparison:    双域尾部形状对比
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非交互后端, 适用于无 GUI 环境
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ---- 全局样式 ----
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.framealpha": 0.9,
})

# 调色板
COLORS = {
    "caseA": "#2196F3",         # 蓝 = 天灾
    "caseB": "#F44336",         # 红 = 人祸
    "caseA_light": "#90CAF9",
    "caseB_light": "#EF9A9A",
    "neutral": "#607D8B",
    "highlight": "#FF9800",
    "background": "#FAFAFA",
}

FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output", "figures"
)


def _ensure_fig_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ========================== 1. Fan Chart ==========================

def plot_simulation_fan(
    paths: np.ndarray,
    title: str,
    ylabel: str = "Value",
    color: str = COLORS["caseA"],
    show_percentiles: list = None,
    n_sample_paths: int = 20,
    save_name: str = None,
    ax: plt.Axes = None
) -> plt.Figure:
    """
    绘制模拟路径扇形图

    展示: 90% CI, 50% CI, 中位数, 随机采样路径
    """
    if show_percentiles is None:
        show_percentiles = [5, 10, 25, 50, 75, 90, 95]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    n_steps = paths.shape[1]
    x = np.arange(n_steps)

    pcts = np.percentile(paths, show_percentiles, axis=0)

    # 90% CI
    ax.fill_between(x, pcts[0], pcts[-1],
                    alpha=0.10, color=color, label="90% CI")
    # 50% CI
    ax.fill_between(x, pcts[1], pcts[-2],
                    alpha=0.25, color=color, label="50% CI")
    # 中位数
    ax.plot(x, pcts[3], color=color, linewidth=2.5, label="Median")

    # 随机路径
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(paths.shape[0], min(n_sample_paths, paths.shape[0]),
                            replace=False)
    for i in sample_idx:
        ax.plot(x, paths[i], alpha=0.06, color=color, linewidth=0.5)

    # 标签
    if n_steps <= 50:
        ax.set_xlabel("Year")
    else:
        ax.set_xlabel("Trading Day")

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper left")

    if "revenue" in ylabel.lower() or "price" in ylabel.lower() or "value" in ylabel.lower():
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, p: f"${v:,.0f}" if v < 1e6 else f"${v/1e6:,.1f}M"
        ))

    if standalone and save_name:
        _ensure_fig_dir()
        fig.savefig(os.path.join(FIGURES_DIR, save_name))
        print(f"  📊 已保存: {save_name}")

    return fig


# ========================== 2. Distribution ==========================

def plot_final_distribution(
    final_values: np.ndarray,
    title: str,
    xlabel: str = "Final Value",
    color: str = COLORS["caseA"],
    reference_line: float = None,
    var_line: float = None,
    save_name: str = None
) -> plt.Figure:
    """绘制最终值分布直方图 + 核密度"""
    fig, ax = plt.subplots(figsize=(10, 5))

    # 直方图
    ax.hist(final_values, bins=80, density=True, alpha=0.5,
            color=color, edgecolor="white", linewidth=0.3)

    # 核密度
    sns.kdeplot(final_values, ax=ax, color=color, linewidth=2)

    # 参考线
    if reference_line is not None:
        ax.axvline(reference_line, color="black", linestyle="--",
                   linewidth=1.5, label=f"Initial = ${reference_line:,.0f}")

    if var_line is not None:
        ax.axvline(var_line, color=COLORS["highlight"], linestyle="-",
                   linewidth=2, label=f"5% VaR = ${var_line:,.0f}")

    # 5% 分位数填充
    p5 = np.percentile(final_values, 5)
    ax.axvspan(ax.get_xlim()[0], p5, alpha=0.1, color=COLORS["highlight"])

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title, fontweight="bold")
    ax.legend()

    if save_name:
        _ensure_fig_dir()
        fig.savefig(os.path.join(FIGURES_DIR, save_name))
        print(f"  📊 已保存: {save_name}")

    return fig


# ========================== 3. VaR Comparison ==========================

def plot_var_comparison(
    var_a: "VaRResult",
    var_b: "VaRResult",
    save_name: str = None
) -> plt.Figure:
    """双域 VaR / CVaR 对比柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: VaR & CVaR
    metrics = ["VaR", "CVaR"]
    vals_a = [abs(var_a.var_pct) * 100, abs(var_a.cvar_pct) * 100]
    vals_b = [abs(var_b.var_pct) * 100, abs(var_b.cvar_pct) * 100]

    x = np.arange(len(metrics))
    w = 0.35

    axes[0].bar(x - w/2, vals_a, w, color=COLORS["caseA"],
                label="Natural Disaster", alpha=0.85)
    axes[0].bar(x + w/2, vals_b, w, color=COLORS["caseB"],
                label="Geopolitical Shock", alpha=0.85)

    axes[0].set_ylabel("Risk (%)")
    axes[0].set_title("VaR & CVaR Comparison", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()

    # 数值标签
    for i, (va, vb) in enumerate(zip(vals_a, vals_b)):
        axes[0].text(i - w/2, va + 0.3, f"{va:.1f}%", ha="center", fontsize=9)
        axes[0].text(i + w/2, vb + 0.3, f"{vb:.1f}%", ha="center", fontsize=9)

    # 右图: 特殊指标
    special_metrics = []
    special_vals_a = []
    special_vals_b = []

    if var_a.depletion_prob > 0:
        special_metrics.append("Depletion\nProb (%)")
        special_vals_a.append(var_a.depletion_prob * 100)
        special_vals_b.append(0)

    special_metrics.append("Max DD\nVaR (%)")
    special_vals_a.append(0)  # Case A 不适用
    special_vals_b.append(abs(var_b.max_drawdown_var) * 100)

    special_metrics.append("Jump Risk\n(%)")
    special_vals_a.append(var_a.jump_risk_pct * 100)
    special_vals_b.append(var_b.jump_risk_pct * 100)

    x2 = np.arange(len(special_metrics))
    axes[1].bar(x2 - w/2, special_vals_a, w, color=COLORS["caseA"],
                label="Case A", alpha=0.85)
    axes[1].bar(x2 + w/2, special_vals_b, w, color=COLORS["caseB"],
                label="Case B", alpha=0.85)

    axes[1].set_ylabel("Value (%)")
    axes[1].set_title("Special Risk Metrics", fontweight="bold")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(special_metrics)
    axes[1].legend()

    plt.tight_layout()

    if save_name:
        _ensure_fig_dir()
        fig.savefig(os.path.join(FIGURES_DIR, save_name))
        print(f"  📊 已保存: {save_name}")

    return fig


# ========================== 4. Risk Decomposition ==========================

def plot_risk_decomposition(
    decomp_a: dict,
    decomp_b: dict,
    save_name: str = None
) -> plt.Figure:
    """风险来源分解: 跳跃 vs 连续"""
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = ["Natural Disaster\n(Case A)", "Geopolitical Shock\n(Case B)"]
    jump_pcts = [decomp_a["jump_risk_pct"] * 100, decomp_b["jump_risk_pct"] * 100]
    cont_pcts = [100 - j for j in jump_pcts]

    x = np.arange(len(labels))
    w = 0.5

    bars_jump = ax.bar(x, jump_pcts, w, label="Jump Risk",
                       color=COLORS["highlight"], alpha=0.85)
    bars_cont = ax.bar(x, cont_pcts, w, bottom=jump_pcts,
                       label="Continuous Risk",
                       color=COLORS["neutral"], alpha=0.7)

    ax.set_ylabel("Risk Contribution (%)")
    ax.set_title("Risk Decomposition: Jump vs Continuous", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 110)

    for i, j in enumerate(jump_pcts):
        ax.text(i, j / 2, f"{j:.1f}%", ha="center", va="center",
                fontweight="bold", color="white", fontsize=12)
        ax.text(i, j + (100 - j) / 2, f"{100-j:.1f}%", ha="center",
                va="center", fontsize=11)

    if save_name:
        _ensure_fig_dir()
        fig.savefig(os.path.join(FIGURES_DIR, save_name))
        print(f"  📊 已保存: {save_name}")

    return fig


# ========================== 5. Sensitivity Heatmap ==========================

def plot_sensitivity_heatmap(
    sensitivity_df: pd.DataFrame,
    title: str,
    value_col: str = "var_pct",
    save_name: str = None
) -> plt.Figure:
    """
    敏感性分析热力图

    假设 sensitivity_df 有 columns: [param_name, param_value, var_pct]
    """
    # Pivot 为矩阵
    pivot = sensitivity_df.pivot_table(
        index="param_name",
        columns="param_value",
        values=value_col
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r",
                ax=ax, linewidths=0.5, cbar_kws={"label": value_col})

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Parameter")

    if save_name:
        _ensure_fig_dir()
        fig.savefig(os.path.join(FIGURES_DIR, save_name))
        print(f"  📊 已保存: {save_name}")

    return fig


# ========================== 6. Tail Comparison ==========================

def plot_tail_comparison(
    final_a: np.ndarray,
    final_b: np.ndarray,
    label_a: str = "Natural Disaster",
    label_b: str = "Geopolitical Shock",
    save_name: str = None
) -> plt.Figure:
    """双域尾部形状对比 (核密度图)"""
    fig, ax = plt.subplots(figsize=(10, 5))

    # 标准化为收益率
    ret_a = (final_a - final_a.mean()) / final_a.std()
    ret_b = (final_b - final_b.mean()) / final_b.std()

    sns.kdeplot(ret_a, ax=ax, color=COLORS["caseA"], linewidth=2,
                label=label_a, fill=True, alpha=0.2)
    sns.kdeplot(ret_b, ax=ax, color=COLORS["caseB"], linewidth=2,
                label=label_b, fill=True, alpha=0.2)

    # 标准正态参考
    x_norm = np.linspace(-4, 4, 200)
    from scipy.stats import norm
    ax.plot(x_norm, norm.pdf(x_norm), "k--", linewidth=1,
            alpha=0.5, label="Normal Distribution")

    ax.set_xlabel("Standardized Returns")
    ax.set_ylabel("Density")
    ax.set_title("Tail Shape Comparison (Standardized)", fontweight="bold")
    ax.legend()
    ax.set_xlim(-5, 5)

    if save_name:
        _ensure_fig_dir()
        fig.savefig(os.path.join(FIGURES_DIR, save_name))
        print(f"  📊 已保存: {save_name}")

    return fig


# ========================== 自测 ==========================

def _run_self_tests():
    print("=" * 60)
    print("visualizer.py 自测开始")
    print("=" * 60)

    from .jump_diffusion_engine import (
        JumpDiffusionParams, simulate_jump_diffusion, simulate_paired_paths
    )
    from .var_calculator import decompose_risk

    _ensure_fig_dir()

    # Generate test data
    print("\nGenerating test data...")
    params = JumpDiffusionParams(
        mu=0.05, sigma=0.2, lam=2.0,
        mu_j=-0.05, sigma_j=0.03,
        S0=100, T=2, dt=1/252, n_sims=3000
    )
    result = simulate_jump_diffusion(params, seed=42)
    paired = simulate_paired_paths(params, seed=42)

    # Test 1: Fan Chart
    print("\n[Test 1] Fan Chart")
    fig = plot_simulation_fan(
        result["paths"][:500],  # 取子集加速
        title="Test Fan Chart",
        ylabel="Value",
        color=COLORS["caseA"],
        save_name="test_fan_chart.png"
    )
    plt.close(fig)
    assert os.path.exists(os.path.join(FIGURES_DIR, "test_fan_chart.png"))
    print("  ✅ Fan Chart 生成成功")

    # Test 2: Distribution
    print("\n[Test 2] Distribution")
    fig = plot_final_distribution(
        result["final_values"],
        title="Test Distribution",
        color=COLORS["caseB"],
        var_line=np.percentile(result["final_values"], 5),
        save_name="test_distribution.png"
    )
    plt.close(fig)
    print("  ✅ Distribution 生成成功")

    # Test 3: Risk Decomposition (using paired paths)
    print("\n[Test 3] Risk Decomposition")
    decomp_a = decompose_risk(paired)

    # Generate a Case-B-like decomposition (higher vol, lower jump share)
    params_b = JumpDiffusionParams(
        mu=0.10, sigma=0.25, lam=3.0,
        mu_j=-0.03, sigma_j=0.02,
        S0=100, T=2, dt=1/252, n_sims=3000
    )
    paired_b = simulate_paired_paths(params_b, seed=42)
    decomp_b = decompose_risk(paired_b)

    fig = plot_risk_decomposition(decomp_a, decomp_b, save_name="test_decomp.png")
    plt.close(fig)
    print(f"  Case A jump risk: {decomp_a['jump_risk_pct']*100:.1f}%")
    print(f"  Case B jump risk: {decomp_b['jump_risk_pct']*100:.1f}%")
    print("  ✅ Risk Decomposition 生成成功")

    # Test 4: Tail Comparison
    print("\n[Test 4] Tail Comparison")
    result_b = simulate_jump_diffusion(
        JumpDiffusionParams(
            mu=0.10, sigma=0.25, lam=3.0,
            mu_j=-0.08, sigma_j=0.04,
            S0=100, T=2, dt=1/252, n_sims=3000
        ), seed=42
    )
    fig = plot_tail_comparison(
        result["final_values"], result_b["final_values"],
        save_name="test_tail.png"
    )
    plt.close(fig)
    print("  ✅ Tail Comparison 生成成功")

    print(f"\n  📁 所有图表保存在: {FIGURES_DIR}")

    print("\n" + "=" * 60)
    print("所有测试通过 ✅")
    print("=" * 60)


if __name__ == "__main__":
    _run_self_tests()
