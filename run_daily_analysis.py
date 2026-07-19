"""
5 iterations with REAL daily GPR + stock data
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DATA = 'e:/Eddik/Documents/AI/jump-diffusion-dual-domain/gpr_deep_dive/data'

# Load clean daily data
df = pd.read_csv(f'{DATA}/analysis_daily_clean.csv', parse_dates=['date'])
df = df.dropna(subset=['SOX_log_return', 'GPRD'])
print(f"Loaded: {len(df)} days, {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

# Train/test split
train_end = '2019-12-31'
train_mask = df['date'] <= train_end
test_mask = df['date'] > train_end
print(f"Train: {train_mask.sum()} days, Test: {test_mask.sum()} days")

# ============================================================
# Iteration 1: GPR Level -> 20d Return
# ============================================================
print("\n" + "=" * 60)
print("Iteration 1: GPR Level -> 20d Forward Return")
print("=" * 60)

train = df[train_mask].dropna(subset=['fwd_return_20d', 'GPRD'])
test = df[test_mask].dropna(subset=['fwd_return_20d', 'GPRD'])

slope, intercept, r, p, se = stats.linregress(train['GPRD'], train['fwd_return_20d'])
test = test.copy()
test['forecast'] = slope * test['GPRD'] + intercept
hit = (np.sign(test['forecast']) == np.sign(test['fwd_return_20d'])).mean()
print(f"  Slope: {slope:.6f}, R: {r:.4f}, p: {p:.4f}")
print(f"  Hit rate (20d direction): {hit*100:.1f}%")
print(f"  N train: {len(train)}, N test: {len(test)}")

# ============================================================
# Iteration 2: GPR Change 5d -> 5d Return
# ============================================================
print("\n" + "=" * 60)
print("Iteration 2: GPR Change 5d -> 5d Forward Return")
print("=" * 60)

train = df[train_mask].dropna(subset=['fwd_return_5d', 'GPRD_change_5d'])
test = df[test_mask].dropna(subset=['fwd_return_5d', 'GPRD_change_5d'])

slope, intercept, r, p, se = stats.linregress(train['GPRD_change_5d'], train['fwd_return_5d'])
test = test.copy()
test['forecast'] = slope * test['GPRD_change_5d'] + intercept
hit = (np.sign(test['forecast']) == np.sign(test['fwd_return_5d'])).mean()
print(f"  Slope: {slope:.6f}, R: {r:.4f}, p: {p:.4f}")
print(f"  Hit rate (5d direction): {hit*100:.1f}%")

# ============================================================
# Iteration 3: Multi-factor (GPR + VIX + Oil)
# ============================================================
print("\n" + "=" * 60)
print("Iteration 3: Multi-factor OLS (GPR + VIX + Oil)")
print("=" * 60)

# Use 20d return as target
mf_cols = ['GPRD', 'VIX_Close', 'OIL_momentum']
mf_available = [c for c in mf_cols if c in df.columns and df[c].notna().sum() > 1000]

train = df[train_mask].dropna(subset=['fwd_return_20d'] + mf_available)
test = df[test_mask].dropna(subset=['fwd_return_20d'] + mf_available)

if len(train) > 100 and len(test) > 100:
    X_tr = np.column_stack([np.ones(len(train)), train[mf_available].values])
    y_tr = train['fwd_return_20d'].values
    beta, _, _, _ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
    y_hat = X_tr @ beta
    r2 = 1 - np.sum((y_tr - y_hat)**2) / np.sum((y_tr - y_tr.mean())**2)

    X_te = np.column_stack([np.ones(len(test)), test[mf_available].values])
    fc = X_te @ beta
    hit = (np.sign(fc) == np.sign(test['fwd_return_20d'].values)).mean()

    print(f"  Factors: {mf_available}")
    print(f"  R-squared: {r2:.4f}")
    print(f"  Hit rate (20d): {hit*100:.1f}%")
    print(f"  Coefficients:")
    for name, b in zip(['const'] + mf_available, beta):
        print(f"    {name}: {b:.6f}")
else:
    print(f"  Not enough data")

# ============================================================
# Iteration 4: Regime-based (GPR level + momentum)
# ============================================================
print("\n" + "=" * 60)
print("Iteration 4: Regime-based Jump Probability (20d)")
print("=" * 60)

train = df[train_mask].dropna(subset=['fwd_return_20d', 'GPRD_momentum', 'is_jump_20d'])
test = df[test_mask].dropna(subset=['fwd_return_20d', 'GPRD_momentum', 'is_jump_20d'])

gpr_median = train['GPRD'].median()

def get_regime(row):
    h = row['GPRD'] > gpr_median
    r = row['GPRD_momentum'] > 0
    if h and r: return 'high_rising'
    if h and not r: return 'high_falling'
    if not h and r: return 'low_rising'
    return 'low_stable'

train = train.copy()
test = test.copy()
train['regime'] = train.apply(get_regime, axis=1)
test['regime'] = test.apply(get_regime, axis=1)

rs = {}
print(f"\n  GPR median: {gpr_median:.0f}")
print(f"  {'Regime':<15} {'Train N':>8} {'Jump%':>8} {'Mean Ret':>10} {'Test N':>8} {'Jump%':>8}")
print(f"  {'-'*60}")

for reg in ['low_stable', 'low_rising', 'high_rising', 'high_falling']:
    tr = train[train['regime'] == reg]
    te = test[test['regime'] == reg]
    if len(tr) >= 50:
        rs[reg] = {
            'n_train': len(tr),
            'jump_prob': tr['is_jump_20d'].mean(),
            'mean_return': tr['fwd_return_20d'].mean(),
            'n_test': len(te),
            'test_jump_prob': te['is_jump_20d'].mean() if len(te) > 0 else 0,
        }
        s = rs[reg]
        print(f"  {reg:<15} {s['n_train']:>8} {s['jump_prob']*100:>7.1f}% {s['mean_return']*100:>9.2f}% {s['n_test']:>8} {s['test_jump_prob']*100:>7.1f}%")

# Direction accuracy
test['pred'] = test['regime'].map(lambda r: 'down' if rs.get(r, {}).get('mean_return', 0) < 0 else 'up')
test['act'] = np.where(test['fwd_return_20d'] < 0, 'down', 'up')
hit = (test['pred'] == test['act']).mean()
print(f"\n  Direction hit rate (test): {hit*100:.1f}%")

# ============================================================
# Iteration 5: Multi-factor Regime
# ============================================================
print("\n" + "=" * 60)
print("Iteration 5: Multi-factor Regime (GPR + VIX)")
print("=" * 60)

z_factors = ['GPRD_momentum', 'VIX_momentum']
z_avail = [f for f in z_factors if f in df.columns and df[f].notna().sum() > 1000]

if len(z_avail) >= 2:
    for col in z_avail:
        m = df.loc[train_mask, col].mean()
        s = df.loc[train_mask, col].std()
        df[f'{col}_z'] = (df[col] - m) / s if s > 0 else 0

    z_cols = [f'{c}_z' for c in z_avail]
    df['risk_score'] = df[z_cols].mean(axis=1)

    train = df[train_mask].dropna(subset=['fwd_return_20d', 'risk_score', 'is_jump_20d'])
    test = df[test_mask].dropna(subset=['fwd_return_20d', 'risk_score', 'is_jump_20d'])

    q33 = train['risk_score'].quantile(0.33)
    q67 = train['risk_score'].quantile(0.67)

    def rreg(s):
        if s < q33: return 'low_risk'
        if s < q67: return 'med_risk'
        return 'high_risk'

    train = train.copy()
    test = test.copy()
    train['rr'] = train['risk_score'].apply(rreg)
    test['rr'] = test['risk_score'].apply(rreg)

    print(f"\n  Factors: {z_avail}")
    print(f"  {'Regime':<15} {'Train N':>8} {'Jump%':>8} {'Mean Ret':>10} {'Test N':>8} {'Jump%':>8}")
    print(f"  {'-'*60}")

    for reg in ['low_risk', 'med_risk', 'high_risk']:
        tr = train[train['rr'] == reg]
        te = test[test['rr'] == reg]
        if len(tr) >= 50:
            print(f"  {reg:<15} {len(tr):>8} {tr['is_jump_20d'].mean()*100:>7.1f}% {tr['fwd_return_20d'].mean()*100:>9.2f}% {len(te):>8} {te['is_jump_20d'].mean()*100:>7.1f}%")

    test['pred'] = test['rr'].map(
        lambda r: 'down' if train[train['rr'] == r]['fwd_return_20d'].mean() < 0 else 'up'
    )
    test['act'] = np.where(test['fwd_return_20d'] < 0, 'down', 'up')
    hit = (test['pred'] == test['act']).mean()
    print(f"\n  Direction hit rate (test): {hit*100:.1f}%")
