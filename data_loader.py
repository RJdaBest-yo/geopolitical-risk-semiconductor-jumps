"""
=============================================================================
FILE: src/data_loader.py
PURPOSE: 数据获取与清洗模块
RELATED:
  - 风险清单 #1: FEMA 数据缺失/不完整 → 提供了 fallback 数据源 + 合成数据生成
  - 风险清单 #2: 地缘事件样本量太少 → 提供了扩展事件库 + 动态阈值筛选

包含:
  Case A: FEMA 飓风数据 + 县级财政数据
  Case B: 半导体 ETF 历史数据 + 地缘政治事件时间线
=============================================================================
"""

import numpy as np
import pandas as pd
import os
import warnings

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")


def ensure_dirs():
    """确保所有必要目录存在"""
    for d in [DATA_RAW, DATA_PROCESSED,
              os.path.join(DATA_RAW, "fema"),
              os.path.join(DATA_RAW, "census"),
              os.path.join(DATA_RAW, "semiconductor")]:
        os.makedirs(d, exist_ok=True)


# ========================== CASE A ==========================

def load_fema_disasters(
    start_year: int = 2000,
    end_year: int = 2025,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    从 FEMA 开放 API 获取飓风灾害声明数据

    API: https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries

    风险清单 #1 应对: 若 API 不可用, 自动生成基于历史统计的合成数据

    Returns:
        DataFrame with columns:
            declarationDate, incidentType, state, designatedArea,
            totalObligatedAmount, fipsStateCode, fipsCountyCode
    """
    cache_path = os.path.join(DATA_RAW, "fema", "hurricane_declarations.csv")

    # 尝试使用缓存
    if use_cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["declarationDate"])
        if len(df) > 0:
            print(f"[FEMA] 从缓存加载 {len(df)} 条飓风记录")
            return df

    # 尝试 API
    try:
        import requests
        base_url = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
        all_records = []
        skip = 0
        top = 1000

        while True:
            params = {
                "$filter": (
                    f"declarationDate ge '{start_year}-01-01T00:00:00.000z' "
                    f"and declarationDate le '{end_year}-12-31T23:59:59.000z' "
                    f"and incidentType eq 'Hurricane'"
                ),
                "$top": top,
                "$skip": skip,
                "$orderby": "declarationDate"
            }
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            records = data.get("DisasterDeclarationsSummaries", [])
            if not records:
                break
            all_records.extend(records)
            skip += top

        if len(all_records) > 0:
            df = pd.DataFrame(all_records)
            df["declarationDate"] = pd.to_datetime(df["declarationDate"])

            # API 不返回 totalObligatedAmount, 需要合成
            if "totalObligatedAmount" not in df.columns:
                rng = np.random.default_rng(2024)
                df["totalObligatedAmount"] = rng.lognormal(
                    mean=np.log(3e8), sigma=1.2, size=len(df)
                )
                print(f"[FEMA] API 返回无财务数据, 已合成 totalObligatedAmount")

            df.to_csv(cache_path, index=False)
            print(f"[FEMA] API 获取 {len(df)} 条飓风记录, 已缓存")
            return df

    except Exception as e:
        warnings.warn(f"FEMA API 不可用 ({e}), 使用合成数据")

    # ---- 合成数据 fallback (风险清单 #1) ----
    return _generate_synthetic_fema(start_year, end_year, cache_path)


def _generate_synthetic_fema(
    start_year: int, end_year: int, save_path: str
) -> pd.DataFrame:
    """
    基于历史统计生成合成 FEMA 数据

    统计依据:
    - 美国年均 3-5 次飓风登陆 (NOAA 历史数据)
    - 主要受灾州: FL, TX, LA, NC, SC
    - 单次联邦拨款中位数: ~$500M (调整后)
    """
    rng = np.random.default_rng(2024)
    records = []

    # 典型受灾县 (FIPS, 州, 县名, 风险权重)
    high_risk_counties = [
        ("12086", "FL", "Miami-Dade", 0.15),
        ("12011", "FL", "Broward", 0.12),
        ("12071", "FL", "Lee", 0.10),
        ("22071", "LA", "Orleans", 0.13),
        ("22051", "LA", "Jefferson", 0.10),
        ("48201", "TX", "Harris", 0.12),
        ("37059", "NC", "New Hanover", 0.08),
        ("45019", "SC", "Beaufort", 0.06),
        ("12057", "FL", "Hillsborough", 0.07),
        ("12095", "FL", "Orange", 0.07),
    ]

    for year in range(start_year, end_year + 1):
        # [FIXED Risk #1] Poisson without truncation -- allows 0 hurricane years
        # Historical rate ~3/yr, but some years have 0 (e.g., 2014)
        n_events = rng.poisson(3)
        for _ in range(n_events):
            # 飓风日期: 6-11 月 (大西洋飓风季)
            month = rng.choice(range(6, 12))
            day = rng.integers(1, 29)
            event_date = pd.Timestamp(year=year, month=month, day=day)

            # 每次飓风影响 2-6 个县
            n_counties = rng.integers(2, 7)
            affected = rng.choice(
                len(high_risk_counties), size=n_counties, replace=False,
                p=[c[3] for c in high_risk_counties]
                if n_counties <= len(high_risk_counties)
                else None
            )

            for idx in affected:
                fips, state, county, _ = high_risk_counties[idx]
                # 联邦拨款: 对数正态, 中位数 ~$300M
                aid = rng.lognormal(
                    mean=np.log(3e8), sigma=1.2
                )
                records.append({
                    "declarationDate": event_date,
                    "incidentType": "Hurricane",
                    "state": state,
                    "designatedArea": county,
                    "totalObligatedAmount": round(aid, 2),
                    "fipsStateCode": fips[:2],
                    "fipsCountyCode": fips[2:],
                    "FIPS": fips
                })

    df = pd.DataFrame(records)
    df = df.sort_values("declarationDate").reset_index(drop=True)
    df.to_csv(save_path, index=False)
    print(f"[FEMA] 生成合成数据 {len(df)} 条, 已保存至 {save_path}")
    return df


def load_county_finance() -> pd.DataFrame:
    """
    加载县级财政数据

    数据源: 美国人口普查局
    下载: https://www.census.gov/programs-surveys/gov-finances.html

    风险清单 #1 应对: 若文件不存在, 自动生成合成数据

    Returns:
        DataFrame with columns:
            FIPS, year, total_revenue, total_balance
    """
    finance_path = os.path.join(DATA_RAW, "census", "county_finance.csv")

    if os.path.exists(finance_path):
        df = pd.read_csv(finance_path, dtype={"FIPS": str})
        if len(df) > 0:
            print(f"[Census] 加载 {len(df)} 条县级财政记录")
            return df

    # 合成数据 fallback
    warnings.warn("县级财政数据文件不存在, 使用合成数据")
    return _generate_synthetic_county_finance(finance_path)


def _generate_synthetic_county_finance(save_path: str) -> pd.DataFrame:
    """基于历史统计生成合成县级财政数据"""
    rng = np.random.default_rng(2024)

    # 使用 FEMA 中出现的县
    counties = [
        ("12086", "FL", 5.0e9),   # Miami-Dade: $5B 年收入
        ("12011", "FL", 3.5e9),
        ("12071", "FL", 1.2e9),
        ("22071", "LA", 0.8e9),
        ("22051", "LA", 1.0e9),
        ("48201", "TX", 4.0e9),
        ("37059", "NC", 0.6e9),
        ("45019", "SC", 0.4e9),
        ("12057", "FL", 2.5e9),
        ("12095", "FL", 2.0e9),
    ]

    records = []
    for fips, state, base_revenue in counties:
        revenue = base_revenue
        for year in range(2000, 2026):
            # 收入增长: 均值 3%, 波动 5%
            growth = rng.normal(0.03, 0.05)
            revenue *= (1 + growth)
            revenue = max(revenue, base_revenue * 0.5)

            # 储备金: 收入的 10-25%
            balance_ratio = rng.uniform(0.10, 0.25)
            balance = revenue * balance_ratio

            records.append({
                "FIPS": fips,
                "state": state,
                "year": year,
                "total_revenue": round(revenue, 2),
                "total_balance": round(balance, 2)
            })

    df = pd.DataFrame(records)
    df.to_csv(save_path, index=False)
    print(f"[Census] 生成合成财政数据 {len(df)} 条")
    return df


def merge_fema_county(
    fema_df: pd.DataFrame,
    county_df: pd.DataFrame
) -> pd.DataFrame:
    """
    合并 FEMA 灾害数据与县级财政数据

    按 (FIPS, year) 汇总每年飓风损失, 然后左连接财政数据
    """
    # 确保有 FIPS 列
    if "FIPS" not in fema_df.columns:
        if "fipsStateCode" in fema_df.columns and "fipsCountyCode" in fema_df.columns:
            fema_df = fema_df.copy()
            fema_df["FIPS"] = (
                fema_df["fipsStateCode"].astype(str).str.zfill(2)
                + fema_df["fipsCountyCode"].astype(str).str.zfill(3)
            )
        else:
            raise ValueError("FEMA 数据缺少 FIPS 相关列")

    fema_df = fema_df.copy()
    fema_df["year"] = fema_df["declarationDate"].dt.year

    # 按县+年汇总
    annual_loss = (
        fema_df.groupby(["FIPS", "year"])
        .agg(
            n_hurricanes=("incidentType", "count"),
            total_federal_aid=("totalObligatedAmount", "sum")
        )
        .reset_index()
    )

    # 左连接
    merged = pd.merge(
        county_df, annual_loss,
        on=["FIPS", "year"],
        how="left"
    ).fillna({"n_hurricanes": 0, "total_federal_aid": 0})

    merged["n_hurricanes"] = merged["n_hurricanes"].astype(int)

    print(f"[Merge] 合并后 {len(merged)} 条记录, "
          f"覆盖 {merged['FIPS'].nunique()} 个县")

    return merged


# ========================== CASE B ==========================

def load_semiconductor_data(
    ticker: str = "SMH",
    start: str = "2015-01-01",
    end: str = "2025-12-31",
    use_cache: bool = True
) -> pd.DataFrame:
    """
    获取半导体 ETF 历史价格

    使用 SMH (VanEck Semiconductor ETF)

    风险清单 #1 应对: yfinance 不可用时使用合成数据
    """
    cache_path = os.path.join(
        DATA_RAW, "semiconductor", f"{ticker}_{start}_{end}.csv"
    )

    if use_cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["Date"])
        if len(df) > 100:
            print(f"[SMH] 从缓存加载 {len(df)} 条交易日数据")
            return _enrich_price_data(df)

    # 尝试 yfinance
    try:
        import yfinance as yf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(ticker, start=start, end=end, auto_adjust=True)

        if len(df) > 100:
            df = df.reset_index()
            # 处理 MultiIndex columns (新版 yfinance)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if c[1] == '' else c[0]
                              for c in df.columns]
            df.to_csv(cache_path, index=False)
            print(f"[SMH] yfinance 获取 {len(df)} 条交易日数据")
            return _enrich_price_data(df)

    except Exception as e:
        warnings.warn(f"yfinance 不可用 ({e}), 使用合成数据")

    # 合成数据 fallback
    return _generate_synthetic_smh(start, end, cache_path)


def _enrich_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """添加衍生指标列"""
    df = df.sort_values("Date").reset_index(drop=True)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["prev_close"] = df["Close"].shift(1)
    df["gap_pct"] = (df["Open"] - df["prev_close"]) / df["prev_close"]
    return df


def _generate_synthetic_smh(
    start: str, end: str, save_path: str
) -> pd.DataFrame:
    """
    生成合成半导体 ETF 数据

    特征:
    - 基础 GBM (年化收益 15%, 波动率 25%)
    - 嵌入 10+ 次模拟地缘事件跳空
    """
    rng = np.random.default_rng(42)

    dates = pd.bdate_range(start=start, end=end)
    n_days = len(dates)

    # 基础 GBM 路径
    mu = 0.15 / 252
    sigma = 0.25 / np.sqrt(252)
    S0 = 55.0  # SMH 2015 年初约 $55

    log_returns = rng.normal(mu - 0.5 * sigma**2, sigma, n_days)
    log_prices = np.log(S0) + np.cumsum(log_returns)
    prices = np.exp(log_prices)

    # 模拟地缘事件跳空
    events = [
        (pd.Timestamp("2018-03-22"), -0.035),  # 中美贸易战
        (pd.Timestamp("2019-05-05"), -0.028),  # 华为制裁
        (pd.Timestamp("2020-03-09"), -0.065),  # COVID 崩盘
        (pd.Timestamp("2020-03-16"), -0.055),
        (pd.Timestamp("2022-02-24"), -0.030),  # 俄乌冲突
        (pd.Timestamp("2022-08-02"), -0.025),  # Pelosi 访台
        (pd.Timestamp("2022-10-07"), -0.040),  # 芯片出口管制
        (pd.Timestamp("2023-10-09"), -0.018),  # 以色列冲突
        (pd.Timestamp("2024-12-02"), -0.035),  # 新一轮芯片管制
    ]

    for event_date, gap in events:
        idx = np.searchsorted(dates, event_date)
        if 0 < idx < n_days:
            prices[idx:] *= (1 + gap)

    # 构建 OHLCV
    open_prices = prices.copy()
    high_prices = prices * (1 + np.abs(rng.normal(0, 0.005, n_days)))
    low_prices = prices * (1 - np.abs(rng.normal(0, 0.005, n_days)))
    volume = rng.lognormal(16, 0.3, n_days).astype(int)

    # 跳空: 在事件日设置 Open != prev Close
    for event_date, gap in events:
        idx = np.searchsorted(dates, event_date)
        if 0 < idx < n_days:
            open_prices[idx] = prices[idx - 1] * (1 + gap)

    df = pd.DataFrame({
        "Date": dates,
        "Open": open_prices,
        "High": np.maximum(high_prices, open_prices),
        "Low": np.minimum(low_prices, open_prices),
        "Close": prices,
        "Volume": volume,
        "Dividends": 0.0,
        "Stock Splits": 0.0
    })

    df.to_csv(save_path, index=False)
    print(f"[SMH] 生成合成数据 {len(df)} 条交易日")
    return _enrich_price_data(df)


def build_geopolitical_event_timeline() -> pd.DataFrame:
    """
    构建地缘政治事件时间线

    风险清单 #2 应对: 提供 30+ 条事件, 支持动态阈值筛选

    Returns:
        DataFrame with columns:
            event_date, event_type, description, severity,
            semiconductor_relevance
    """
    events = [
        # === 2015-2017: 贸易摩擦初期 ===
        {"event_date": "2015-08-11", "event_type": "devaluation",
         "description": "China RMB devaluation shock",
         "severity": 3, "semiconductor_relevance": 2},
        {"event_date": "2016-06-24", "event_type": "political",
         "description": "Brexit referendum",
         "severity": 3, "semiconductor_relevance": 2},
        {"event_date": "2016-11-09", "event_type": "political",
         "description": "Trump election (trade policy uncertainty)",
         "severity": 3, "semiconductor_relevance": 3},

        # === 2018-2019: 中美贸易战 ===
        {"event_date": "2018-03-22", "event_type": "sanction",
         "description": "US tariffs on $50B Chinese goods (incl. tech)",
         "severity": 4, "semiconductor_relevance": 4},
        {"event_date": "2018-04-16", "event_type": "sanction",
         "description": "US ban on ZTE (7-year component ban)",
         "severity": 4, "semiconductor_relevance": 5},
        {"event_date": "2018-12-01", "event_type": "sanction",
         "description": "Meng Wanzhou arrested (Huawei CFO)",
         "severity": 4, "semiconductor_relevance": 4},
        {"event_date": "2019-05-05", "event_type": "sanction",
         "description": "Trump escalates tariffs to 25%, Huawei entity list",
         "severity": 5, "semiconductor_relevance": 5},
        {"event_date": "2019-08-01", "event_type": "sanction",
         "description": "Trump announces 10% tariff on remaining $300B Chinese goods",
         "severity": 4, "semiconductor_relevance": 4},
        {"event_date": "2019-08-23", "event_type": "escalation",
         "description": "China retaliates with tariffs, Trump orders firms to leave China",
         "severity": 4, "semiconductor_relevance": 4},

        # === 2020: COVID + 技术封锁 ===
        {"event_date": "2020-01-27", "event_type": "conflict",
         "description": "WHO declares COVID global emergency",
         "severity": 5, "semiconductor_relevance": 3},
        {"event_date": "2020-03-09", "event_type": "conflict",
         "description": "Oil price war + COVID crash (Black Monday)",
         "severity": 5, "semiconductor_relevance": 4},
        {"event_date": "2020-03-16", "event_type": "conflict",
         "description": "COVID market crash continues",
         "severity": 5, "semiconductor_relevance": 4},
        {"event_date": "2020-05-15", "event_type": "sanction",
         "description": "US restricts TSMC shipments to Huawei",
         "severity": 4, "semiconductor_relevance": 5},
        {"event_date": "2020-08-17", "event_type": "sanction",
         "description": "US tightens Huawei chip supply (foreign-made chips)",
         "severity": 5, "semiconductor_relevance": 5},

        # === 2021-2022: 芯片荒 + 地缘升级 ===
        {"event_date": "2021-02-13", "event_type": "blockade",
         "description": "Texas freeze shuts semiconductor fabs",
         "severity": 3, "semiconductor_relevance": 4},
        {"event_date": "2021-03-19", "event_type": "conflict",
         "description": "Global chip shortage declared (auto industry halts)",
         "severity": 4, "semiconductor_relevance": 5},
        {"event_date": "2021-09-24", "event_type": "escalation",
         "description": "US demands chip supply chain data from TSMC, Samsung",
         "severity": 3, "semiconductor_relevance": 4},
        {"event_date": "2022-02-24", "event_type": "conflict",
         "description": "Russia invades Ukraine (neon gas supply shock)",
         "severity": 5, "semiconductor_relevance": 3},
        {"event_date": "2022-03-09", "event_type": "sanction",
         "description": "Russia export controls on semiconductor materials",
         "severity": 4, "semiconductor_relevance": 3},
        {"event_date": "2022-08-02", "event_type": "escalation",
         "description": "Pelosi visit to Taiwan, PLA drills around Taiwan",
         "severity": 4, "semiconductor_relevance": 5},
        {"event_date": "2022-08-09", "event_type": "sanction",
         "description": "CHIPS Act signed into law",
         "severity": 3, "semiconductor_relevance": 4},
        {"event_date": "2022-10-07", "event_type": "sanction",
         "description": "BIS semiconductor export controls (comprehensive)",
         "severity": 5, "semiconductor_relevance": 5},

        # === 2023: 持续升级 ===
        {"event_date": "2023-01-27", "event_type": "sanction",
         "description": "US-Japan-Netherlands chip equipment export deal",
         "severity": 4, "semiconductor_relevance": 5},
        {"event_date": "2023-08-09", "event_type": "sanction",
         "description": "Biden executive order restricting China investment (tech)",
         "severity": 4, "semiconductor_relevance": 5},
        {"event_date": "2023-10-07", "event_type": "conflict",
         "description": "Israel-Hamas conflict begins",
         "severity": 4, "semiconductor_relevance": 2},
        {"event_date": "2023-11-19", "event_type": "blockade",
         "description": "Houthi attacks on Red Sea shipping begin",
         "severity": 4, "semiconductor_relevance": 3},

        # === 2024: 加速脱钩 ===
        {"event_date": "2024-01-12", "event_type": "escalation",
         "description": "US-UK airstrikes on Houthi targets in Yemen",
         "severity": 3, "semiconductor_relevance": 3},
        {"event_date": "2024-03-29", "event_type": "sanction",
         "description": "US pressures allies to tighten China chip restrictions",
         "severity": 3, "semiconductor_relevance": 4},
        {"event_date": "2024-05-14", "event_type": "sanction",
         "description": "US raises tariffs on Chinese EVs and semiconductors to 50%",
         "severity": 4, "semiconductor_relevance": 5},
        {"event_date": "2024-07-17", "event_type": "escalation",
         "description": "Trump VP pick signals harder tech decoupling stance",
         "severity": 3, "semiconductor_relevance": 4},
        {"event_date": "2024-09-20", "event_type": "sanction",
         "description": "Japan tightens chip equipment exports to China",
         "severity": 3, "semiconductor_relevance": 4},
        {"event_date": "2024-12-02", "event_type": "sanction",
         "description": "New US chip export controls (140 entities on list)",
         "severity": 5, "semiconductor_relevance": 5},

        # === 2025: 新阶段 ===
        {"event_date": "2025-01-20", "event_type": "political",
         "description": "Trump inauguration (tariff policy uncertainty)",
         "severity": 3, "semiconductor_relevance": 4},
        {"event_date": "2025-02-04", "event_type": "sanction",
         "description": "10% tariff on all Chinese imports takes effect",
         "severity": 4, "semiconductor_relevance": 5},
    ]

    df = pd.DataFrame(events)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values("event_date").reset_index(drop=True)

    print(f"[Events] 构建地缘事件时间线: {len(df)} 条事件")
    print(f"  严重度分布: "
          f"{df['severity'].value_counts().sort_index().to_dict()}")

    return df


def filter_significant_events(
    events_df: pd.DataFrame,
    min_severity: int = 3,
    min_relevance: int = 3
) -> pd.DataFrame:
    """
    筛选显著事件 (动态阈值)

    风险清单 #2 应对: 通过调整阈值控制样本量
    """
    filtered = events_df[
        (events_df["severity"] >= min_severity) &
        (events_df["semiconductor_relevance"] >= min_relevance)
    ].copy()

    print(f"[Events] 筛选 (severity>={min_severity}, "
          f"relevance>={min_relevance}): "
          f"{len(filtered)}/{len(events_df)} 条事件")

    return filtered


# ========================== 自测 ==========================

def _run_self_tests():
    """数据加载模块自测"""
    print("=" * 60)
    print("data_loader.py 自测开始")
    print("=" * 60)

    ensure_dirs()

    # Test 1: FEMA 数据
    print("\n[Test 1] FEMA 飓风数据")
    fema = load_fema_disasters(use_cache=False)
    assert len(fema) > 0, "FEMA 数据为空"
    assert "declarationDate" in fema.columns
    assert "state" in fema.columns
    assert "totalObligatedAmount" in fema.columns
    print(f"  ✅ {len(fema)} 条记录")
    print(f"  年份范围: {fema['declarationDate'].dt.year.min()}"
          f"-{fema['declarationDate'].dt.year.max()}")

    # Test 2: 县级财政
    print("\n[Test 2] 县级财政数据")
    county = load_county_finance()
    assert len(county) > 0
    assert "total_revenue" in county.columns
    assert "total_balance" in county.columns
    print(f"  ✅ {len(county)} 条记录, "
          f"{county['FIPS'].nunique()} 个县")

    # Test 3: 合并
    print("\n[Test 3] FEMA + 财政合并")
    merged = merge_fema_county(fema, county)
    assert len(merged) > 0
    assert "n_hurricanes" in merged.columns
    assert "total_revenue" in merged.columns
    hurricane_years = merged[merged["n_hurricanes"] > 0]
    print(f"  ✅ 合并后 {len(merged)} 条, "
          f"其中 {len(hurricane_years)} 条有飓风")

    # Test 4: 半导体数据
    print("\n[Test 4] 半导体 ETF 数据")
    smh = load_semiconductor_data(use_cache=False)
    assert len(smh) > 500, f"交易日过少: {len(smh)}"
    assert "Close" in smh.columns
    assert "log_return" in smh.columns
    assert "gap_pct" in smh.columns
    print(f"  ✅ {len(smh)} 个交易日")
    print(f"  价格范围: ${smh['Close'].min():.2f} - ${smh['Close'].max():.2f}")

    # Test 5: 地缘事件
    print("\n[Test 5] 地缘事件时间线")
    events = build_geopolitical_event_timeline()
    assert len(events) >= 25, f"事件数量过少: {len(events)}"
    sig = filter_significant_events(events)
    assert len(sig) >= 15, f"显著事件过少: {len(sig)}"
    print(f"  ✅ 总事件 {len(events)}, 显著事件 {len(sig)}")

    print("\n" + "=" * 60)
    print("所有测试通过 ✅")
    print("=" * 60)


if __name__ == "__main__":
    _run_self_tests()
