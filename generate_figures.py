"""
Generate all visualizations for the GPR paper.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')

import os
os.makedirs('output/figures', exist_ok=True)

df = pd.read_csv('gpr_deep_dive/data/analysis_daily_clean.csv', parse_dates=['date'])
wf = pd.read_csv('output/results/01_walk_forward/walk_forward_results.csv')

print('Generating visualizations...')

# ============================================================
# Figure 1: Walk-Forward Hit Rate by Cycle
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
cycles = wf['cycle'].values
hit_rates = wf['hit_rate'].values * 100

colors = ['#2ecc71' if h >= 60 else '#f39c12' if h >= 55 else '#e74c3c' for h in hit_rates]
bars = ax.bar(cycles, hit_rates, color=colors, edgecolor='white', linewidth=0.5)

mean_hr = hit_rates.mean()
ax.axhline(y=mean_hr, color='#3498db', linestyle='--', linewidth=2, label=f'Mean: {mean_hr:.1f}%')

best_idx = np.argmax(hit_rates)
ax.annotate(f'Best: {hit_rates[best_idx]:.1f}%\n(Cycle {int(cycles[best_idx])})',
            xy=(cycles[best_idx], hit_rates[best_idx]),
            xytext=(cycles[best_idx]+2, hit_rates[best_idx]+2),
            arrowprops=dict(arrowstyle='->', color='#2c3e50'),
            fontsize=10, fontweight='bold')

ax.set_xlabel('Cycle (Training End Year)', fontsize=12)
ax.set_ylabel('Hit Rate (%)', fontsize=12)
ax.set_title('Walk-Forward Directional Hit Rate by Cycle', fontsize=14, fontweight='bold')
ax.set_xticks(cycles)
ax.set_xticklabels([f'{int(c)}\n({1999+int(c)})' for c in cycles], fontsize=8)
ax.legend(fontsize=11)
ax.set_ylim(50, 68)
plt.tight_layout()
plt.savefig('output/figures/fig1_hit_rate_by_cycle.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [1/6] Hit rate by cycle')

# ============================================================
# Figure 2: GPR vs SOX Time Series
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

ax1.plot(df['date'], df['GPRD'], color='#e74c3c', alpha=0.7, linewidth=0.8)
ax1.fill_between(df['date'], df['GPRD'], alpha=0.3, color='#e74c3c')
ax1.axhline(y=df['GPRD'].median(), color='#2c3e50', linestyle='--', alpha=0.5,
            label=f'Median: {df["GPRD"].median():.0f}')
ax1.set_ylabel('GPR Index', fontsize=12)
ax1.set_title('Geopolitical Risk Index vs Semiconductor Stocks (1994-2026)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)

ax2.plot(df['date'], df['SOX_Close'], color='#2980b9', alpha=0.8, linewidth=1)
ax2.set_ylabel('SOX Price ($)', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_yscale('log')

events = [('2001-09-11', '9/11'), ('2008-09-15', 'Lehman'), ('2022-02-24', 'Russia-Ukraine')]
for date_str, label in events:
    date = pd.Timestamp(date_str)
    if date >= df['date'].min() and date <= df['date'].max():
        ax1.axvline(x=date, color='#8e44ad', linestyle=':', alpha=0.7)
        ax1.annotate(label, xy=(date, ax1.get_ylim()[1]*0.9), fontsize=9,
                    rotation=45, ha='right', color='#8e44ad')

plt.tight_layout()
plt.savefig('output/figures/fig2_gpr_vs_sox.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [2/6] GPR vs SOX time series')

# ============================================================
# Figure 3: Regime Jump Probability Heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

regimes = ['low_stable', 'low_rising', 'high_rising', 'high_falling']
metrics = ['Neg Jump%', 'Pos Jump%', 'Total Jump%']
data = np.array([
    [11.5, 13.5, 25.0],
    [19.3, 20.1, 39.3],
    [10.5, 11.8, 22.3],
    [8.6, 10.6, 19.2],
])

im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(metrics, fontsize=11)
ax.set_yticks(range(len(regimes)))
ax.set_yticklabels([r.replace('_', ' ').title() for r in regimes], fontsize=11)

for i in range(len(regimes)):
    for j in range(len(metrics)):
        ax.text(j, i, f'{data[i, j]:.1f}%',
                ha='center', va='center', fontsize=12, fontweight='bold',
                color='white' if data[i, j] > 20 else 'black')

plt.colorbar(im, ax=ax, label='Jump Probability (%)')
ax.set_title('Jump Probability by GPR Regime (Cycle 19 Calibration)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/figures/fig3_regime_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [3/6] Regime heatmap')

# ============================================================
# Figure 4: VaR Comparison Across Models
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

regime_labels = [r.replace('_', ' ').title() for r in regimes]
models = ['GPR-enabled JD', 'GPR+Heston', 'GPR+fBM']
var_data = np.array([
    [-68.0, -53.8, -52.9],
    [-74.1, -53.9, -57.4],
    [-67.5, -55.2, -55.7],
    [-65.4, -54.9, -50.0],
])

x = np.arange(len(regimes))
width = 0.25
colors = ['#e74c3c', '#3498db', '#2ecc71']

for i, (model, color) in enumerate(zip(models, colors)):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, var_data[:, i], width, label=model, color=color, alpha=0.8)
    for bar, val in zip(bars, var_data[:, i]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() - 2,
                f'{val:.1f}%', ha='center', va='top', fontsize=9, fontweight='bold')

ax.set_xlabel('GPR Regime', fontsize=12)
ax.set_ylabel('VaR (95%)', fontsize=12)
ax.set_title('VaR Comparison Across Models and Regimes', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(regime_labels, fontsize=11)
ax.legend(fontsize=11, loc='lower right')
ax.set_ylim(-80, -45)
plt.tight_layout()
plt.savefig('output/figures/fig4_var_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [4/6] VaR comparison')

# ============================================================
# Figure 5: VaR Gap by Horizon
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

horizons = ['20d', '60d', '120d', '252d']
hist_var = [-13.33, -17.82, -24.04, -30.16]
sim_var = [-19.66, -32.85, -45.29, -59.82]
gap = [1.47, 1.84, 1.88, 1.98]

x = np.arange(len(horizons))
width = 0.35

bars1 = ax.bar(x - width/2, hist_var, width, label='Historical VaR', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, sim_var, width, label='Simulated VaR', color='#e74c3c', alpha=0.8)

for i, (h, s, g) in enumerate(zip(hist_var, sim_var, gap)):
    ax.annotate(f'{g:.1f}x', xy=(i, min(h, s) - 3), fontsize=11, ha='center',
               fontweight='bold', color='#2c3e50')

ax.set_xlabel('Horizon', fontsize=12)
ax.set_ylabel('VaR (5%)', fontsize=12)
ax.set_title('VaR Gap: Historical vs Simulated by Horizon', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(horizons, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(-70, 0)
plt.tight_layout()
plt.savefig('output/figures/fig5_var_gap_horizon.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [5/6] VaR gap by horizon')

# ============================================================
# Figure 6: Parameter Robustness Check
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

cycle19_var = [-75.5, -84.1, -74.1, -71.0]
conservative_var = [-79.8, -86.7, -78.0, -75.5]

x = np.arange(len(regimes))
width = 0.35

bars1 = ax.bar(x - width/2, cycle19_var, width, label='Cycle 19 Parameters', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, conservative_var, width, label='Conservative Parameters', color='#e74c3c', alpha=0.8)

for bar, val in zip(bars1, cycle19_var):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() - 1,
            f'{val:.1f}%', ha='center', va='top', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, conservative_var):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() - 1,
            f'{val:.1f}%', ha='center', va='top', fontsize=9, fontweight='bold')

ax.set_xlabel('GPR Regime', fontsize=12)
ax.set_ylabel('VaR (95%)', fontsize=12)
ax.set_title('Parameter Robustness: Cycle 19 vs Conservative', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(regime_labels, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(-90, -65)
plt.tight_layout()
plt.savefig('output/figures/fig6_parameter_robustness.png', dpi=150, bbox_inches='tight')
plt.close()
print('  [6/6] Parameter robustness')

print('\nAll visualizations saved to output/figures/')
