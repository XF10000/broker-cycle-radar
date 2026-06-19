#!/usr/bin/env python3
"""
Securities sector stock odds screener.
Computes per-cycle Z-scores for index constituent stocks,
aggregates into historical odds rankings.

Usage:
    python screener.py --init           # First run: fetch all + compute
    python screener.py --refresh        # Daily: incremental data + recompute Z
    python screener.py --calc-only      # Recompute Z only (data already fresh)
    python screener.py --refresh-index  # Refetch constituents + recompute thresholds
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data_fetcher import fetch_index_constituents, fetch_stock_daily, fetch_index_daily

OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
CYCLES_PATH = os.path.join(OUTPUT_DIR, 'cycles.csv')
ODDS_PATH = os.path.join(OUTPUT_DIR, 'stock_odds.csv')


def load_cycles():
    """Load bull cycles from output/cycles.csv, 2010+ only."""
    df = pd.read_csv(CYCLES_PATH)
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df = df[df['start_date'] >= pd.Timestamp('2010-01-01')]
    return df.reset_index(drop=True)


def ensure_stock_data(ts_code, force=False):
    """Fetch stock daily data, return sorted DataFrame or None."""
    try:
        df = fetch_stock_daily(ts_code, force=force)
        if df is not None and not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            return df.sort_values('trade_date').reset_index(drop=True)
    except Exception:
        pass
    return None


def compute_cycle_return(seg, cycle_start, cycle_end):
    """
    Compute max return within a cycle window for a given DataFrame segment.
    Returns float (e.g. 0.37 = +37%) or None if data insufficient.

    注意：这里用区间内 close 的【最大值】而非结束价，衡量的是该轮周期内
    个股的"弹性/最高可达收益"而非"持有至结束的收益"。早期冲高回落的个股
    会被记为高收益。这是有意的"弹性评估"设计选择，用于横向比较领涨强度。
    """
    # seg should be a DataFrame filtered to [cycle_start, cycle_end]
    if seg.empty or len(seg) < 3:
        return None
    start_price = float(seg['close'].iloc[0])
    if start_price <= 0:
        return None
    max_price = float(seg['close'].max())
    return max_price / start_price - 1.0


def compute_all_z_scores(constituents, cycles, force=False):
    """
    For each constituent and each cycle, compute raw return and Z-score.

    Returns DataFrame with columns: ts_code, cycle_idx, return, index_return, z_score
    """
    records = []
    index_df = fetch_index_daily()
    index_df['trade_date'] = pd.to_datetime(index_df['trade_date'])

    total = len(constituents)
    for i, stock in enumerate(constituents):
        ts_code = stock['ts_code']
        name = stock.get('name', '')
        print(f"  [{i+1}/{total}] 处理 {ts_code} {name}...", end=' ')
        df = ensure_stock_data(ts_code, force=force)
        if df is None:
            print("无数据，跳过")
            continue

        n_cycles = 0
        for ci, (_, cycle) in enumerate(cycles.iterrows()):
            mask = (df['trade_date'] >= cycle['start_date']) & (df['trade_date'] <= cycle['end_date'])
            seg = df[mask]
            ret = compute_cycle_return(seg, cycle['start_date'], cycle['end_date'])
            if ret is not None:
                idx_mask = (index_df['trade_date'] >= cycle['start_date']) & (index_df['trade_date'] <= cycle['end_date'])
                idx_seg = index_df[idx_mask]
                idx_ret = compute_cycle_return(idx_seg, cycle['start_date'], cycle['end_date'])
                records.append({
                    'ts_code': ts_code,
                    'cycle_idx': ci,
                    'return': ret,
                    'index_return': idx_ret if idx_ret is not None else 0.0,
                })
                n_cycles += 1
        print(f"{n_cycles} 轮覆盖")

    detail = pd.DataFrame(records)
    if detail.empty:
        return detail

    # Sort by cycle_idx so that group iteration order matches row order.
    # This ensures z_scores are assigned to the correct rows.
    detail = detail.sort_values('cycle_idx').reset_index(drop=True)

    z_col = []
    for cycle_idx, group in detail.groupby('cycle_idx', sort=False):
        returns = group['return'].values
        n = len(returns)
        if n < 2:
            z_vals = [0.0] * n
        else:
            mu = returns.mean()
            sigma = returns.std(ddof=1)
            if sigma < 1e-8:
                z_vals = [0.0] * n
            else:
                z_vals = ((returns - mu) / sigma).tolist()
        z_col.extend(z_vals)

    detail['z_score'] = z_col
    return detail


def aggregate_odds(detail_df, constituents, existing_meta=None):
    """
    Aggregate per-cycle Z-scores into per-stock odds metrics.

    Args:
        detail_df: output of compute_all_z_scores
        constituents: list of dicts with ts_code, name
        existing_meta: dict with optional 'M', 'N' keys to reuse

    Returns:
        (odds_df, meta) where meta contains M, N, update_time
    """
    if detail_df.empty:
        return pd.DataFrame(), {}

    agg = detail_df.groupby('ts_code').agg(
        median_z=('z_score', 'median'),
        max_z=('z_score', 'max'),
        positive_z_rate=('z_score', lambda x: (x > 0).sum() / len(x)),
        median_return=('return', 'median'),
        max_return=('return', 'max'),
        cycle_count=('cycle_idx', 'nunique'),
    ).reset_index()

    # Compute beat_index_rate: for each stock, fraction of cycles where return > index return
    beat_rates = {}
    for code, group in detail_df.groupby('ts_code'):
        beats = (group['return'] > group['index_return']).sum()
        beat_rates[code] = beats / len(group)
    agg['beat_index_rate'] = agg['ts_code'].map(beat_rates)

    # Add names from constituents
    name_map = {c['ts_code']: c.get('name', '') for c in constituents}
    agg['name'] = agg['ts_code'].map(name_map)

    # Confidence thresholds
    if existing_meta and 'M' in existing_meta:
        M = int(existing_meta['M'])
    else:
        cycle_counts = agg['cycle_count'].values
        M = int(np.median(cycle_counts)) if len(cycle_counts) > 0 else 3

    def confidence(n):
        if n >= M:
            return '可信'
        elif n >= 2:
            return '参考'
        else:
            return '有限'

    agg['confidence'] = agg['cycle_count'].apply(confidence)

    # Common window N: most recent consecutive cycles with >= 50% participation
    if existing_meta and 'N' in existing_meta:
        N = int(existing_meta['N'])
    else:
        total_stocks = len(agg)
        cycle_participation = detail_df.groupby('cycle_idx')['ts_code'].nunique()
        N = 0
        for ci in sorted(cycle_participation.index, reverse=True):
            if total_stocks > 0 and cycle_participation[ci] / total_stocks >= 0.5:
                N += 1
            else:
                break
        N = max(N, 1)

    meta = {
        'M': M,
        'N': N,
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }

    agg = agg.sort_values('median_z', ascending=False).reset_index(drop=True)
    return agg, meta


def write_odds_csv(agg_df, meta):
    """Write stock_odds.csv with metadata header lines."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(ODDS_PATH, 'w') as f:
        f.write(f"# M={meta['M']}, N={meta['N']}, updated={meta['update_time']}\n")
        f.write(f"# M: 置信度阈值（参与轮数中位数）\n")
        f.write(f"# N: 共同窗口轮数（>=50%个股参与的最近连续周期数）\n")
    agg_df.to_csv(ODDS_PATH, mode='a', index=False)
    return ODDS_PATH


def read_odds_meta():
    """Read metadata from existing stock_odds.csv. Returns dict with M, N or empty."""
    meta = {}
    if not os.path.exists(ODDS_PATH):
        return meta
    with open(ODDS_PATH, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                break
            parts = line.strip('# ').split(',')
            for p in parts:
                if '=' in p:
                    k, v = p.split('=', 1)
                    meta[k.strip()] = v.strip()
    result = {}
    if 'M' in meta:
        result['M'] = meta['M']
    if 'N' in meta:
        result['N'] = meta['N']
    return result


def run_init(force=True):
    """Full initialization: fetch constituents + all stock data + compute."""
    print("拉取指数成分股...")
    constituents = fetch_index_constituents(force=force)
    if not constituents:
        print("ERROR: 无法获取成分股列表")
        return
    print(f"  获取到 {len(constituents)} 只成分股")

    cycles = load_cycles()
    print(f"  加载 {len(cycles)} 轮牛市周期（2010年后）")

    print("拉取个股日线数据并计算Z评分...")
    detail = compute_all_z_scores(constituents, cycles, force=force)
    if detail.empty:
        print("ERROR: 没有计算出任何周期数据")
        return
    print(f"  计算完成: {len(detail)} 条周期记录，{detail['ts_code'].nunique()} 只个股参与")

    print("汇总赔率指标...")
    agg, meta = aggregate_odds(detail, constituents)
    print(f"  汇总: {len(agg)} 只个股")

    path = write_odds_csv(agg, meta)
    print(f"\n输出: {path}")
    print(f"  置信度阈值 M={meta['M']}（参与轮数中位数）")
    print(f"  共同窗口 N={meta['N']}（>=50%参与连续周期数）")
    print(f"  可信: {len(agg[agg['confidence'] == '可信'])} 只")
    print(f"  参考: {len(agg[agg['confidence'] == '参考'])} 只")
    print(f"  有限: {len(agg[agg['confidence'] == '有限'])} 只")


def run_refresh():
    """Incremental data refresh + recompute Z (preserve M, N)."""
    existing_meta = read_odds_meta()

    constituents = fetch_index_constituents(force=False)
    if not constituents:
        print("ERROR: 无法获取成分股列表")
        return

    cycles = load_cycles()
    print(f"加载 {len(cycles)} 轮牛市周期（2010年后）")
    print("刷新个股日线数据并重算Z评分...")
    detail = compute_all_z_scores(constituents, cycles, force=True)
    if detail.empty:
        print("ERROR: 没有计算出任何周期数据")
        return
    agg, meta = aggregate_odds(detail, constituents, existing_meta=existing_meta)
    path = write_odds_csv(agg, meta)
    print(f"刷新完成: {path}（阈值不变: M={meta['M']}, N={meta['N']}）")


def run_calc_only():
    """Recompute Z scores only, without fetching new data."""
    existing_meta = read_odds_meta()

    constituents = fetch_index_constituents(force=False)
    if not constituents:
        print("ERROR: 无法获取成分股列表")
        return

    cycles = load_cycles()
    print(f"加载 {len(cycles)} 轮牛市周期（2010年后）")
    print("仅重算Z评分（不拉取新数据）...")
    detail = compute_all_z_scores(constituents, cycles, force=False)
    if detail.empty:
        print("ERROR: 没有计算出任何周期数据")
        return
    agg, meta = aggregate_odds(detail, constituents, existing_meta=existing_meta)
    path = write_odds_csv(agg, meta)
    print(f"重算完成: {path}")


def run_refresh_index():
    """Refetch constituents + recompute thresholds."""
    print("刷新成分股列表...")
    constituents = fetch_index_constituents(force=True)
    if not constituents:
        print("ERROR: 无法获取成分股列表")
        return

    cycles = load_cycles()
    print(f"加载 {len(cycles)} 轮牛市周期（2010年后）")
    detail = compute_all_z_scores(constituents, cycles, force=True)
    if detail.empty:
        print("ERROR: 没有计算出任何周期数据")
        return
    agg, meta = aggregate_odds(detail, constituents)  # no existing_meta — fresh thresholds
    path = write_odds_csv(agg, meta)
    print(f"成分股 + 阈值已更新: {path}")
    print(f"  新阈值: M={meta['M']}, N={meta['N']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='券商个股赔率筛选')
    parser.add_argument('--init', action='store_true', help='首次初始化')
    parser.add_argument('--refresh', action='store_true', help='日常刷新（保留阈值）')
    parser.add_argument('--calc-only', action='store_true', help='仅重算Z评分')
    parser.add_argument('--refresh-index', action='store_true', help='刷新成分股列表并重算阈值')
    args = parser.parse_args()

    if args.init:
        run_init(force=True)
    elif args.refresh:
        run_refresh()
    elif args.calc_only:
        run_calc_only()
    elif args.refresh_index:
        run_refresh_index()
    else:
        parser.print_help()
