#!/usr/bin/env python3
"""
券商个股领涨/滞涨节奏分析（Step 1 最小验证）。

定义 B：每轮周期启动后前 N 个交易日（默认 20）的累计涨幅排序。
按涨幅等分三档：tier 1 = 先涨组（涨幅最高 1/3），tier 2 = 中段，tier 3 = 后涨组。

输出：
  - 跨轮 Spearman 秩相关矩阵（基于原始涨幅排名）
  - 先涨组（tier 1）跨轮重合度（Jaccard）
  - 个体稳定性指标（每只股票 tier 的均值/标准差）
  - 先涨组全轮交集（跨所有轮的稳定先涨股）

Usage:
    python lead_lag.py                 # 默认 20 日窗口
    python lead_lag.py --window 10     # 10 日窗口
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.signal import argrelextrema

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data_fetcher import INDEX_CONSTITUENTS, _ts_code

OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')
CYCLES_PATH = os.path.join(OUTPUT_DIR, 'cycles.csv')

MIN_WINDOW_COVERAGE = 0.75  # 窗口内至少要有 75% 的交易日数据，否则该股该轮跳过
LOW_MATCH_WINDOW = 60  # 个股低点与板块周期起点匹配窗口（天），超出则跳过该股该轮


def detect_stock_troughs(df, smooth_window=20, order=10):
    """
    Detect all local troughs in stock price using same algorithm as detect_cycles.py.
    Returns list of (pd.Timestamp, close_price) sorted by date.
    """
    close = df['close'].values
    dates = df['trade_date'].values
    smoothed = pd.Series(close).rolling(smooth_window, min_periods=1).mean().values
    troughs = argrelextrema(smoothed, np.less, order=order)[0]
    return [(pd.Timestamp(dates[i]), float(close[i])) for i in troughs]


def find_nearest_trough(troughs, target_date, max_days=LOW_MATCH_WINDOW):
    """
    Find the stock trough closest to target_date within max_days.
    Returns (trough_date, trough_price) or None.
    """
    if not troughs:
        return None
    nearest = min(troughs, key=lambda x: abs((x[0] - target_date).days))
    if abs((nearest[0] - target_date).days) > max_days:
        return None
    return nearest


def load_cycles():
    """Load 2010+ bull cycles."""
    df = pd.read_csv(CYCLES_PATH)
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df = df[df['start_date'] >= pd.Timestamp('2010-01-01')].reset_index(drop=True)
    return df


def load_stock_daily(code):
    """Load stock daily CSV, return sorted DataFrame or None."""
    path = os.path.join(DATA_DIR, f'stock_daily_{code}.csv')
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df.sort_values('trade_date').reset_index(drop=True)
    except Exception:
        return None


def compute_window_return(stock_df, cycle_start, window_days=20):
    """
    Compute cumulative return over the first `window_days` trading days
    starting from (and including) cycle_start.

    Returns float or None if insufficient data or stock not yet listed.

    Guards against the "future data" trap: if the stock's first available
    bar is far after cycle_start (e.g. stock listed in 2020 but cycle starts
    in 2010), the filter `trade_date >= cycle_start` would wrongly grab 2020
    data. We require the first bar in the window to be within 10 calendar
    days of cycle_start.
    """
    seg = stock_df[stock_df['trade_date'] >= cycle_start].head(window_days + 1)
    if seg.empty:
        return None
    # Guard: first bar must be close to cycle_start (within 10 calendar days),
    # otherwise the stock wasn't actually trading at cycle start.
    first_date = seg['trade_date'].iloc[0]
    if (first_date - cycle_start).days > 10:
        return None
    if len(seg) < window_days * MIN_WINDOW_COVERAGE:
        return None
    start_price = float(seg['close'].iloc[0])
    end_price = float(seg['close'].iloc[-1])
    if start_price <= 0:
        return None
    return end_price / start_price - 1.0


def compute_window_return_from_low(stock_df, trough_date, trough_price,
                                   cycle_end, window_days=20):
    """
    Compute cumulative return over the first `window_days` trading days
    starting from the stock's own trough date.

    Uses trough_price (the actual low) as start price, and the close on
    the window-th trading day after trough as end price.

    Returns float or None if insufficient data.
    """
    seg = stock_df[stock_df['trade_date'] >= trough_date].head(window_days + 1)
    if len(seg) < window_days * MIN_WINDOW_COVERAGE:
        return None
    end_price = float(seg['close'].iloc[-1])
    if trough_price <= 0:
        return None
    return end_price / trough_price - 1.0


def assign_tiers(group):
    """Assign tier (1=先涨, 2=中段, 3=后涨) by return ranking within a cycle."""
    group = group.copy()
    n = len(group)
    if n < 3:
        group['tier'] = np.nan
        return group
    ranks = group['window_return'].rank(method='first', ascending=False).astype(int)
    tier_size = n / 3.0
    group['tier'] = ranks.apply(
        lambda r: 1 if r <= tier_size else (2 if r <= tier_size * 2 else 3)
    )
    return group


def compute_lead_lag(window_days=20, use_stock_low=True):
    """
    Main computation: for each cycle × stock, compute front-window return and tier.

    Args:
        window_days: number of trading days in the front window
        use_stock_low: if True, use each stock's own detected trough (via
            argrelextrema, same algo as detect_cycles.py) as the start point.
            If False, use the index cycle start date uniformly.

    Returns:
        detail_df: long-format DataFrame [ts_code, name, cycle_idx, cycle_start,
                    window_return, tier, stock_start_date]
    """
    cycles = load_cycles()
    records = []

    for ci, (_, cycle) in enumerate(cycles.iterrows()):
        idx_start = cycle['start_date']
        idx_end = cycle['end_date']

        for code, name in INDEX_CONSTITUENTS.items():
            sdf = load_stock_daily(code)
            if sdf is None:
                continue

            if use_stock_low:
                # Detect stock's own troughs, find nearest to index cycle start
                troughs = detect_stock_troughs(sdf)
                nearest = find_nearest_trough(troughs, idx_start)
                if nearest is None:
                    continue
                trough_date, trough_price = nearest
                ret = compute_window_return_from_low(
                    sdf, trough_date, trough_price, idx_end, window_days
                )
                stock_start = trough_date.strftime('%Y-%m-%d')
            else:
                ret = compute_window_return(sdf, idx_start, window_days)
                stock_start = idx_start.strftime('%Y-%m-%d')

            if ret is None:
                continue

            records.append({
                'ts_code': _ts_code(code),
                'code': code,
                'name': name,
                'cycle_idx': ci,
                'cycle_start': idx_start.strftime('%Y-%m-%d'),
                'stock_start': stock_start,
                'cycle_change_pct': cycle['change_pct'],
                'window_return': ret,
            })

    detail = pd.DataFrame(records)
    if detail.empty:
        return detail

    # Assign tiers within each cycle
    tiered_parts = []
    for ci, group in detail.groupby('cycle_idx', sort=False):
        tiered = assign_tiers(group.copy())
        tiered_parts.append(tiered)
    detail = pd.concat(tiered_parts, ignore_index=True)
    return detail


def analyze_consistency(detail_df):
    """
    Run all consistency analyses on the detail DataFrame.

    Returns dict with:
        - spearman_matrix: pairwise Spearman rank correlation between cycles
        - jaccard_matrix: pairwise Jaccard of tier-1 (先涨) sets
        - stock_stability: per-stock mean tier, std, participation count
        - lead_intersection: stocks in tier-1 across ALL cycles they participated
        - n_cycles: number of cycles with data
        - n_stocks: number of stocks
    """
    if detail_df.empty:
        return {}

    cycles = sorted(detail_df['cycle_idx'].unique())
    stocks = detail_df['ts_code'].unique()

    # ---- 1. Spearman rank correlation matrix (on raw returns) ----
    pivot_ret = detail_df.pivot_table(
        index='ts_code', columns='cycle_idx', values='window_return'
    )
    spearman = pivot_ret.corr(method='spearman')

    # ---- 2. Jaccard of tier-1 sets between cycle pairs ----
    tier1_sets = {}
    for ci in cycles:
        sub = detail_df[(detail_df['cycle_idx'] == ci) & (detail_df['tier'] == 1)]
        tier1_sets[ci] = set(sub['ts_code'])

    jaccard = pd.DataFrame(np.nan, index=cycles, columns=cycles, dtype=float)
    for c1, c2 in combinations(cycles, 2):
        s1, s2 = tier1_sets[c1], tier1_sets[c2]
        union = s1 | s2
        if len(union) == 0:
            j = np.nan
        else:
            j = len(s1 & s2) / len(union)
        jaccard.loc[c1, c2] = j
        jaccard.loc[c2, c1] = j
    for c in cycles:
        jaccard.loc[c, c] = 1.0

    # ---- 3. Per-stock stability ----
    stock_stats = []
    for ts_code in stocks:
        sub = detail_df[detail_df['ts_code'] == ts_code].sort_values('cycle_idx')
        n = len(sub)
        name = sub['name'].iloc[0]
        stock_stats.append({
            'ts_code': ts_code,
            'code': ts_code.split('.')[0],
            'name': name,
            'participation': n,
            'mean_tier': sub['tier'].mean() if n > 0 else np.nan,
            'tier_std': sub['tier'].std(ddof=0) if n > 0 else np.nan,
            'lead_count': int((sub['tier'] == 1).sum()),   # 先涨次数
            'lag_count': int((sub['tier'] == 3).sum()),    # 后涨次数
            'mean_return': sub['window_return'].mean() if n > 0 else np.nan,
        })
    stock_stability = pd.DataFrame(stock_stats).sort_values('mean_tier')

    # ---- 4. Lead-group intersection across all cycles ----
    # A stock is "stable lead" if it's tier-1 in >= 50% of cycles it participated
    stock_stability['lead_rate'] = stock_stability['lead_count'] / stock_stability['participation']
    stable_leads = stock_stability[
        (stock_stability['participation'] >= 3) & (stock_stability['lead_rate'] >= 0.5)
    ]

    # ---- 5. Summary stats ----
    # Average off-diagonal Spearman
    off_diag_spearman = []
    for c1, c2 in combinations(cycles, 2):
        v = spearman.loc[c1, c2]
        if not np.isnan(v):
            off_diag_spearman.append(v)

    # Average off-diagonal Jaccard
    off_diag_jaccard = []
    for c1, c2 in combinations(cycles, 2):
        v = jaccard.loc[c1, c2]
        if not np.isnan(v):
            off_diag_jaccard.append(v)

    # Random baseline for Jaccard: if tier-1 is 1/3 of N stocks,
    # P(both in tier-1) = (1/3)^2 = 1/9, Jaccard expected ~ 1/9 / (2/3) ≈ 1/6? 
    # More precisely: |A∩B|/|A∪B|, E[|A∩B|] = n/9, E[|A∪B|] = 2n/3 - n/9 = 5n/9
    # E[Jaccard] ≈ (1/9)/(5/9) = 1/5 = 0.2
    # But this assumes same N per cycle. Use actual mean tier-1 size.
    avg_tier1_size = np.mean([len(s) for s in tier1_sets.values()])
    n_stocks_avg = len(stocks)
    # Approximate random baseline
    p = avg_tier1_size / n_stocks_avg  # probability a stock is tier-1
    random_jaccard = p**2 / (2*p - p**2) if (2*p - p**2) > 0 else 0

    return {
        'spearman_matrix': spearman,
        'jaccard_matrix': jaccard,
        'stock_stability': stock_stability,
        'stable_leads': stable_leads,
        'n_cycles': len(cycles),
        'n_stocks': len(stocks),
        'avg_off_diag_spearman': np.mean(off_diag_spearman) if off_diag_spearman else np.nan,
        'avg_off_diag_jaccard': np.mean(off_diag_jaccard) if off_diag_jaccard else np.nan,
        'random_jaccard_baseline': random_jaccard,
        'tier1_sets': tier1_sets,
        'cycles_meta': detail_df[['cycle_idx', 'cycle_start', 'cycle_change_pct']].drop_duplicates().set_index('cycle_idx'),
    }


def print_report(detail_df, analysis):
    """Print text report to stdout."""
    if not analysis:
        print("无数据")
        return

    print("=" * 70)
    print("券商个股领涨/滞涨节奏分析")
    print("=" * 70)

    # Participation per cycle
    print("\n【各轮参与股票数】")
    for ci in sorted(detail_df['cycle_idx'].unique()):
        sub = detail_df[detail_df['cycle_idx'] == ci]
        start = sub['cycle_start'].iloc[0]
        chg = sub['cycle_change_pct'].iloc[0]
        n1 = int((sub['tier'] == 1).sum())
        print(f"  轮{ci+1}  {start}  板块+{chg:.1f}%  参与{len(sub)}只  先涨组{n1}只")

    # Spearman
    print(f"\n【跨轮 Spearman 秩相关】")
    print(f"  平均非对角线相关系数: {analysis['avg_off_diag_spearman']:+.3f}")
    print(f"  （>0 表示不同轮次启动节奏有正相关性，越接近 1 越稳定）")
    sp = analysis['spearman_matrix']
    print("\n  Spearman 矩阵:")
    print(sp.round(3).to_string())

    # Jaccard
    print(f"\n【先涨组（tier 1）跨轮重合度 Jaccard】")
    print(f"  平均 Jaccard: {analysis['avg_off_diag_jaccard']:.3f}")
    print(f"  随机基线:     {analysis['random_jaccard_baseline']:.3f}")
    print(f"  （实际 > 随机基线 = 有稳定先涨股的信号）")
    jac = analysis['jaccard_matrix']
    print("\n  Jaccard 矩阵:")
    print(jac.round(3).to_string())

    # Stock stability
    print(f"\n【个股节奏稳定性】")
    ss = analysis['stock_stability']
    print(f"  （mean_tier 越接近 1 = 越常先涨；越接近 3 = 越常后涨；std 越小 = 越稳定）")
    print(f"\n  稳定先涨股（参与≥3轮 且 先涨率≥50%）:")
    sl = analysis['stable_leads'][['name', 'code', 'participation', 'lead_count', 'lead_rate', 'mean_tier']].copy()
    if sl.empty:
        print("    无")
    else:
        for _, r in sl.iterrows():
            print(f"    {r['name']}({r['code']})  参与{r['participation']}轮 先涨{r['lead_count']}次 "
                  f"先涨率{r['lead_rate']:.0%} 平均档位{r['mean_tier']:.2f}")

    print(f"\n  稳定后涨股（参与≥3轮 且 后涨率≥50%）:")
    stable_lags = ss[(ss['participation'] >= 3) & (ss['lag_count'] / ss['participation'] >= 0.5)]
    if stable_lags.empty:
        print("    无")
    else:
        for _, r in stable_lags.iterrows():
            lag_rate = r['lag_count'] / r['participation']
            print(f"    {r['name']}({r['code']})  参与{r['participation']}轮 后涨{r['lag_count']}次 "
                  f"后涨率{lag_rate:.0%} 平均档位{r['mean_tier']:.2f}")

    print(f"\n  全部个股稳定性明细（按平均档位排序）:")
    show = ss[['name', 'code', 'participation', 'lead_count', 'lag_count', 'mean_tier', 'tier_std']].copy()
    print(show.to_string(index=False))

    print("\n" + "=" * 70)
    print("解读要点：")
    print("1. 平均 Spearman 若 > 0.3 且统计显著 → 启动节奏有跨周期稳定性")
    print("2. 平均 Jaccard 显著 > 随机基线 → 存在稳定的先涨股群体")
    print("3. stable_leads 列表中的股票 = 历史上多次率先启动的候选")
    print("4. 样本量小（9 轮，早期轮次股票不全），结论仅供参考，非统计显著结论")
    print("=" * 70)


def save_results(detail_df, analysis, window_days=20):
    """Save CSV outputs."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Detail long-format
    detail_path = os.path.join(OUTPUT_DIR, f'lead_lag_detail_w{window_days}.csv')
    detail_df.to_csv(detail_path, index=False, encoding='utf-8-sig')

    # Tier pivot (wide: stocks × cycles)
    if not detail_df.empty:
        tier_pivot = detail_df.pivot_table(
            index=['ts_code', 'name'], columns='cycle_idx', values='tier'
        )
        tier_path = os.path.join(OUTPUT_DIR, f'lead_lag_tiers_w{window_days}.csv')
        tier_pivot.to_csv(tier_path, encoding='utf-8-sig')

    # Stock stability
    if analysis and 'stock_stability' in analysis:
        stab_path = os.path.join(OUTPUT_DIR, f'lead_lag_stability_w{window_days}.csv')
        analysis['stock_stability'].to_csv(stab_path, index=False, encoding='utf-8-sig')

    print(f"\n已保存:")
    print(f"  {detail_path}")
    if not detail_df.empty:
        print(f"  {os.path.join(OUTPUT_DIR, f'lead_lag_tiers_w{window_days}.csv')}")
    if analysis and 'stock_stability' in analysis:
        print(f"  {os.path.join(OUTPUT_DIR, f'lead_lag_stability_w{window_days}.csv')}")


def detect_recent_low(index_df, lookback_days=120):
    """Detect the most recent local low in index daily data as cycle start candidate.
    Searches within the last `lookback_days` bars for the lowest low.
    Returns pd.Timestamp.
    """
    recent = index_df.tail(lookback_days)
    low_idx = recent['low'].idxmin()
    return index_df.loc[low_idx, 'trade_date']


def compute_current_returns(start_date=None, stability_df=None, odds_df=None,
                            use_stock_low=False, low_lookback_days=120):
    """
    Compute cumulative return of all constituent stocks.

    Two modes:
    - Fixed start (use_stock_low=False): all stocks measured from same start_date
    - Per-stock low (use_stock_low=True): each stock measured from its own lowest
      low within the last `low_lookback_days`. This captures stocks that bottomed
      earlier than the index (e.g. CITIC Securities bottomed 2 months before index).

    Args:
        start_date: pd.Timestamp/str, used when use_stock_low=False, or as the
                    left bound when searching for per-stock lows.
        stability_df: optional stock_stability DataFrame from analyze_consistency.
        odds_df: optional stock_odds DataFrame with median_z etc.

    Returns:
        DataFrame sorted by return_from_start desc, with columns:
            ts_code, code, name, start_date, start_price, current_price,
            current_date, return_from_start, low_in_window, return_from_low
    """
    records = []

    for code, name in INDEX_CONSTITUENTS.items():
        sdf = load_stock_daily(code)
        if sdf is None:
            continue
        sdf = sdf.sort_values('trade_date').reset_index(drop=True)
        current_price = float(sdf['close'].iloc[-1])
        current_date = sdf['trade_date'].iloc[-1]

        if use_stock_low:
            # Search window: last N days, bounded by start_date if provided
            if start_date is not None:
                search_seg = sdf[sdf['trade_date'] >= pd.Timestamp(start_date)]
            else:
                search_seg = sdf.tail(low_lookback_days)
            if search_seg.empty:
                continue
            low_idx = search_seg['low'].idxmin()
            stock_start_date = sdf.loc[low_idx, 'trade_date']
            start_price = float(sdf.loc[low_idx, 'low'])
            # Window for "from low" = same as start since start IS the low
            low_in_window = start_price
        else:
            if start_date is None:
                continue
            stock_start_date = pd.Timestamp(start_date)
            seg = sdf[sdf['trade_date'] >= stock_start_date]
            if len(seg) < 3:
                continue
            start_price = float(seg['close'].iloc[0])
            low_in_window = float(seg['low'].min())

        records.append({
            'ts_code': _ts_code(code),
            'code': code,
            'name': name,
            'start_date': stock_start_date.strftime('%Y-%m-%d'),
            'start_price': round(start_price, 2),
            'current_price': round(current_price, 2),
            'current_date': current_date.strftime('%Y-%m-%d'),
            'return_from_start': (current_price / start_price - 1) * 100,
            'low_in_window': round(low_in_window, 2),
            'return_from_low': (current_price / low_in_window - 1) * 100,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Annotate with historical stability if provided
    if stability_df is not None and not stability_df.empty:
        stab = stability_df[['code', 'mean_tier', 'lead_count', 'participation']].copy()
        stab = stab.rename(columns={'mean_tier': 'hist_mean_tier'})
        df = df.merge(stab, on='code', how='left')
        df['hist_lead_rate'] = (df['lead_count'] / df['participation']).fillna(0)

    # Annotate with odds (Z-score) if provided
    if odds_df is not None and not odds_df.empty:
        odds = odds_df[['ts_code', 'median_z', 'positive_z_rate', 'beat_index_rate', 'confidence', 'cycle_count']].copy()
        odds = odds.rename(columns={
            'median_z': 'odds_z', 'positive_z_rate': 'odds_pos_rate',
            'beat_index_rate': 'odds_beat_rate', 'confidence': 'odds_confidence',
            'cycle_count': 'odds_cycles',
        })
        df = df.merge(odds, on='ts_code', how='left')

    df = df.sort_values('return_from_start', ascending=False).reset_index(drop=True)
    return df


def run(window_days=20, use_stock_low=True, verbose=True):
    """Full pipeline: compute + analyze + save + print."""
    detail = compute_lead_lag(window_days=window_days, use_stock_low=use_stock_low)
    if detail.empty:
        print("ERROR: 未计算出任何数据")
        return detail, {}

    analysis = analyze_consistency(detail)
    save_results(detail, analysis, window_days)
    if verbose:
        print_report(detail, analysis)
    return detail, analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='券商个股领涨/滞涨节奏分析')
    parser.add_argument('--window', type=int, default=20,
                        help='前 N 个交易日窗口（默认 20）')
    args = parser.parse_args()
    run(window_days=args.window)
