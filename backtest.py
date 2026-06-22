"""
Backtest engine: evaluates technical indicator signals against known market cycles.
"""
import os
import pandas as pd
import numpy as np

from config import (
    MA_PERIOD, SIGNAL_WINDOW_INDEX, SIGNAL_WINDOW_STOCK,
    LATE_CUTOFF_DAYS, RESONANCE_WINDOW_DAYS, CYCLE_FILTER_DATE,
    SMOOTH_WINDOW, MIN_AMPLITUDE_PCT, MIN_DURATION_DAYS, ARGRELEXTREMA_ORDER,
    SCORE_IDEAL_DAYS, SCORE_DAYS_NORMALIZE, WEIGHT_DAYS_SCORE,
    WEIGHT_HIT_RATE, WEIGHT_PRECISION,
    SELL_SEARCH_WINDOW, SELL_HIT_CAPTURE,
    WEIGHT_SELL_CAPTURE, WEIGHT_SELL_HIT_RATE, WEIGHT_SELL_PRECISION,
    SCORE_SELL_MAX,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Chinese param name mapping
_PARAM_CN = {
    'fast': '快线', 'slow': '慢线', 'signal': '信号线',
    'lookback': '回溯', 'ma_period': '均线', 'below_days': '线下天数',
    'period': '周期', 'adx_low': 'ADX阈值',
    'consecutive': '连续天数', 'contract_window': '缩量窗口',
    'expand_ratio': '放量倍数', 'max_ratio': '最大倍数',
    'std': '标准差', 'ndays_newlow': '新低天数',
    'vol_ratio': '量比', 'flat_days': '横盘天数', 'flat_pct': '振幅',
    'window': '窗口', 'threshold': '阈值', 'n': 'N', 'm1': 'M1', 'm2': 'M2',
    'above_days': '线上天数', 'adx_high': 'ADX高位',
    'stagnant_days': '滞涨天数', 'chg_pct': '涨跌幅',
    'newhigh_days': '新高天数', 'shrink_days': '缩量天数',
    'ndays_newhigh': '新高窗口',
}

def _param_str(params):
    """Convert param dict to readable Chinese string."""
    parts = []
    for k, v in params.items():
        cn = _PARAM_CN.get(k, k)
        parts.append(f'{cn}={v}')
    return ', '.join(parts)


def judge_signal(signal_series, cycle_start, cycle_end, signal_window=SIGNAL_WINDOW_INDEX):
    """
    Judge whether a signal successfully predicted a market cycle.

    Args:
        signal_series: pd.Series of bool, indexed by date
        cycle_start: pd.Timestamp, the start date of the bull cycle
        cycle_end: pd.Timestamp, the end date of the bull cycle
        signal_window: int, days before cycle_start to look for signals

    Returns:
        dict with keys: hit (bool), signal_date (pd.Timestamp or None),
        days_before (int or None), late (bool)
    """
    # Valid signal window: [cycle_start - signal_window, cycle_start]
    window_start = cycle_start - pd.Timedelta(days=signal_window)
    window_mask = (signal_series.index >= window_start) & (signal_series.index <= cycle_start)
    window_signals = signal_series[window_mask]

    if window_signals.any():
        # Take the latest signal in the window (closest to cycle_start)
        signal_dates = window_signals[window_signals].index
        signal_date = signal_dates[-1]
        days_before = (cycle_start - signal_date).days
        return {
            'hit': True,
            'signal_date': signal_date,
            'days_before': days_before,
            'late': False,
        }

    # Check if signal fires too late (after cycle_start + 10 days)
    late_cutoff = cycle_start + pd.Timedelta(days=LATE_CUTOFF_DAYS)
    late_mask = (signal_series.index > cycle_start) & (signal_series.index <= late_cutoff)
    if signal_series[late_mask].any():
        return {
            'hit': False,
            'signal_date': signal_series[late_mask].index[0],
            'days_before': None,
            'late': True,
        }

    return {
        'hit': False,
        'signal_date': None,
        'days_before': None,
        'late': False,
    }


def count_false_signals(signal_series, cycles,  signal_window=SIGNAL_WINDOW_INDEX):
    """
    Count signals that fall outside any cycle's valid window.
    A signal is 'false' if it's not in any [start-signal_window, start+10] window.
    """
    # Build union of all valid+late windows
    valid_mask = pd.Series(False, index=signal_series.index)
    for cycle in cycles:
        start = pd.Timestamp(cycle['start_date'])
        end = pd.Timestamp(cycle['end_date'])
        window_start = start - pd.Timedelta(days=signal_window)
        window_end = start + pd.Timedelta(days=LATE_CUTOFF_DAYS)
        valid_mask[(signal_series.index >= window_start) & (signal_series.index <= window_end)] = True

    false_signals = signal_series & ~valid_mask
    return false_signals.sum()


def run_backtest(cycle_df, signals_list, data_dfs,  signal_window=SIGNAL_WINDOW_INDEX):
    """
    Run backtest for all signal rules against all cycles.

    Args:
        cycle_df: pd.DataFrame with columns [start_date, end_date, ...]
        signals_list: list of dicts from indicators.get_all_signal_rules()
        data_dfs: dict with keys 'index_daily', 'index_weekly', 'index_moneyflow'
        signal_window: int, days to look back before cycle start

    Returns:
        pd.DataFrame with backtest results
    """
    # All cycles for reference (including pre-2010), but only 2010+ for hit counting
    cycles_all = cycle_df.to_dict('records')
    cycles = [c for c in cycles_all if pd.Timestamp(c['start_date']) >= pd.Timestamp(CYCLE_FILTER_DATE)]
    total_cycles = len(cycles)

    # Build cycle amplitude weights (weighted hit rate)
    cycle_weights = np.array([c['change_pct'] / 100 for c in cycles])
    total_weight = cycle_weights.sum()

    results = []

    for freq_key, df in [('日线', data_dfs.get('index_daily')),
                          ('周线', data_dfs.get('index_weekly'))]:
        if df is None or df.empty:
            continue

        df = df.set_index('trade_date').sort_index()

        # Build cycle-based reference highs for decline calculation
        cycle_peaks = sorted([(pd.Timestamp(c['end_date']), c['end_price']) for c in cycles_all])
        ref_highs = []
        for d in df.index:
            rh = None
            for peak_date, peak_price in cycle_peaks:
                if peak_date <= d:
                    rh = peak_price
                else:
                    break
            ref_highs.append(rh if rh is not None else df.loc[d, 'close'])
        ref_highs_arr = ref_highs

        for rule in signals_list:
            work_df = df.copy()

            needs_mf = rule.get('requires_moneyflow', False)
            if needs_mf:
                mf_df = data_dfs.get('index_moneyflow')
                if mf_df is None or mf_df.empty:
                    continue  # Skip moneyflow rules if no data
                mf_df = mf_df.set_index('trade_date').sort_index()
                # Merge moneyflow columns into work_df
                for col in mf_df.columns:
                    if col not in work_df.columns:
                        work_df[col] = mf_df[col]

            try:
                signal_series = rule['func'](work_df.reset_index(), **rule['params'])
                signal_series.index = work_df.index
                # Apply trend context + cooldown filter
                from indicators import filter_signals
                signal_series = filter_signals(
                    work_df.reset_index(), signal_series,
                    ref_highs=ref_highs_arr,
                )
                signal_series.index = work_df.index
            except Exception as e:
                print(f"  [SKIP] {rule['name']} ({freq_key}): {e}")
                continue

            # Judge each cycle
            total_signals = int(signal_series.sum())
            hits = 0
            lates = 0
            days_list = []
            hit_signal_count = 0
            cycle_signal_count = 0  # cycles with at least 1 signal in range
            weighted_hits = 0

            for cyc in cycles:
                ws = pd.Timestamp(cyc['start_date']) - pd.Timedelta(days=signal_window)
                we = pd.Timestamp(cyc['end_date']) + pd.Timedelta(days=LATE_CUTOFF_DAYS)
                wm = (signal_series.index >= ws) & (signal_series.index <= we)
                if signal_series[wm].any():
                    cycle_signal_count += 1

            for i, cycle in enumerate(cycles):
                start = pd.Timestamp(cycle['start_date'])
                end = pd.Timestamp(cycle['end_date'])
                window_start = start - pd.Timedelta(days=signal_window)
                window_end = start + pd.Timedelta(days=LATE_CUTOFF_DAYS)

                # Count at most 1 effective signal per cycle for precision
                window_mask = (signal_series.index >= window_start) & (signal_series.index <= window_end)
                if signal_series[window_mask].any():
                    hit_signal_count += 1

                judgment = judge_signal(
                    signal_series, start, end, signal_window,
                )
                if judgment['hit']:
                    hits += 1
                    days_list.append(judgment['days_before'])
                    weighted_hits += cycle_weights[i]
                elif judgment['late']:
                    lates += 1

            hit_rate = hits / total_cycles if total_cycles > 0 else 0
            weighted_hr = weighted_hits / total_weight if total_weight > 0 else 0
            avg_days = np.mean(days_list) if days_list else None
            precision = hit_signal_count / cycle_signal_count if cycle_signal_count > 0 else 0

            # Composite score: balance hit_rate, signal precision, and timing
            days_score = 0
            if avg_days is not None:
                days_score = abs(avg_days - SCORE_IDEAL_DAYS) / SCORE_DAYS_NORMALIZE * WEIGHT_DAYS_SCORE
            score = weighted_hr * WEIGHT_HIT_RATE + precision * WEIGHT_PRECISION - days_score

            results.append({
                '指标名': rule['name'],
                '类别': rule['category'],
                '周期': freq_key,
                '参数': _param_str(rule['params']),
                '命中率': round(weighted_hr, 3),
                '命中轮数': f'{hits}/{total_cycles}',
                '平均提前天': round(avg_days, 1) if avg_days is not None else None,
                '信号有效率': round(precision, 3),
                '总信号数': total_signals,
                '综合得分': round(score, 3),
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('综合得分', ascending=False).reset_index(drop=True)
    return results_df


def run_and_save(data_dfs,  signal_window=SIGNAL_WINDOW_INDEX):
    """Convenience: run backtest and save to CSV."""
    from indicators import get_all_signal_rules

    cycle_path = os.path.join(OUTPUT_DIR, 'cycles.csv')
    if not os.path.exists(cycle_path):
        raise FileNotFoundError(f"{cycle_path} not found. Run detect_cycles.py first.")

    cycle_df = pd.read_csv(cycle_path)
    rules = get_all_signal_rules()
    results = run_backtest(cycle_df, rules, data_dfs, signal_window)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, 'backtest_results.csv')
    results.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"Saved {len(results)} results to {path}")
    return results


def run_resonance_backtest(cycle_df, standalone_results, data_dfs,  signal_window=SIGNAL_WINDOW_STOCK):
    """
    Multi-timeframe resonance backtest: weekly indicator fires first,
    then daily indicator confirms within a 30-day observation window.
    """
    from indicators import INDICATOR_REGISTRY, filter_signals

    cycles_all = cycle_df.to_dict('records')
    cycles = [c for c in cycles_all if pd.Timestamp(c['start_date']) >= pd.Timestamp(CYCLE_FILTER_DATE)]
    total_cycles = len(cycles)
    cycle_weights = np.array([c['change_pct'] / 100 for c in cycles])
    total_weight = cycle_weights.sum()

    weekly_top = standalone_results[standalone_results['周期'] == '周线'].head(3)
    daily_top = standalone_results[standalone_results['周期'] == '日线'].head(3)

    weekly_names = weekly_top['指标名'].unique()
    daily_names = daily_top['指标名'].unique()

    weekly_df = data_dfs['index_weekly'].set_index('trade_date').sort_index() if data_dfs.get('index_weekly') is not None else None
    daily_df = data_dfs['index_daily'].set_index('trade_date').sort_index() if data_dfs.get('index_daily') is not None else None

    if weekly_df is None or daily_df is None:
        return pd.DataFrame()

    # Build ref_highs for weekly and daily
    cycle_peaks = sorted([(pd.Timestamp(c['end_date']), c['end_price']) for c in cycles_all])
    def _build_ref_highs(idx):
        rh = []
        for d in idx:
            r = None
            for pd_, pp in cycle_peaks:
                if pd_ <= d: r = pp
                else: break
            rh.append(r if r is not None else 0)
        return rh
    w_ref = _build_ref_highs(weekly_df.index)
    d_ref = _build_ref_highs(daily_df.index)

    results = []

    for w_name in weekly_names:
        w_cfg = INDICATOR_REGISTRY.get(w_name)
        if w_cfg is None:
            continue
        w_params = w_cfg['params'][0]

        try:
            w_raw = w_cfg['func'](weekly_df.reset_index(), **w_params)
            w_raw.index = weekly_df.index
            w_sig = filter_signals(weekly_df.reset_index(), w_raw, ref_highs=w_ref)
            w_sig.index = weekly_df.index
        except Exception:
            continue

        w_signal_dates = w_sig[w_sig].index

        for d_name in daily_names:
            d_cfg = INDICATOR_REGISTRY.get(d_name)
            if d_cfg is None:
                continue
            d_params = d_cfg['params'][0]

            try:
                d_raw = d_cfg['func'](daily_df.reset_index(), **d_params)
                d_raw.index = daily_df.index
                d_sig = filter_signals(daily_df.reset_index(), d_raw, ref_highs=d_ref)
                d_sig.index = daily_df.index
            except Exception:
                continue

            resonance_sig = pd.Series(False, index=daily_df.index)
            total_signals = 0

            for ws_date in w_signal_dates:
                window_end = ws_date + pd.Timedelta(days=RESONANCE_WINDOW_DAYS)
                window_mask = (daily_df.index >= ws_date) & (daily_df.index <= window_end)
                window_daily_sigs = d_sig[window_mask]
                if window_daily_sigs.any():
                    first_sig_date = window_daily_sigs[window_daily_sigs].index[0]
                    resonance_sig[first_sig_date] = True
                    total_signals += 1

            hits = 0
            lates = 0
            days_list = []
            hit_signal_count = 0
            weighted_hits = 0

            for i, cycle in enumerate(cycles):
                start = pd.Timestamp(cycle['start_date'])
                end = pd.Timestamp(cycle['end_date'])
                window_start = start - pd.Timedelta(days=signal_window)
                window_end = start + pd.Timedelta(days=LATE_CUTOFF_DAYS)

                window_mask = (resonance_sig.index >= window_start) & (resonance_sig.index <= window_end)
                if resonance_sig[window_mask].any():
                    hit_signal_count += 1

                judgment = judge_signal(resonance_sig, start, end, signal_window)
                if judgment['hit']:
                    hits += 1
                    days_list.append(judgment['days_before'])
                    weighted_hits += cycle_weights[i]
                elif judgment['late']:
                    lates += 1

            hit_rate = hits / total_cycles if total_cycles > 0 else 0
            weighted_hr = weighted_hits / total_weight if total_weight > 0 else 0
            avg_days = np.mean(days_list) if days_list else None

            precision = hit_signal_count / total_signals if total_signals > 0 else 0
            days_score = 0
            if avg_days is not None:
                days_score = abs(avg_days - SCORE_IDEAL_DAYS) / SCORE_DAYS_NORMALIZE * WEIGHT_DAYS_SCORE
            score = weighted_hr * WEIGHT_HIT_RATE + precision * WEIGHT_PRECISION - days_score

            rule_name = f'周线:{w_name} + 日线:{d_name}'
            results.append({
                '指标名': rule_name,
                '类别': '共振',
                '周期': '周+日',
                '参数': f'周线: {_param_str(w_params)} | 日线: {_param_str(d_params)}',
                '命中率': round(weighted_hr, 3),
                '命中轮数': f'{hits}/{total_cycles}',
                '平均提前天': round(avg_days, 1) if avg_days is not None else None,
                '信号有效率': round(precision, 3),
                '总信号数': total_signals,
                '综合得分': round(score, 3),
            })

    return pd.DataFrame(results)


def run_and_save_all(data_dfs,  signal_window=SIGNAL_WINDOW_STOCK):
    """Run standalone + resonance backtest, save merged results."""
    standalone = run_and_save(data_dfs, signal_window)
    resonance = run_resonance_backtest(
        pd.read_csv(os.path.join(OUTPUT_DIR, 'cycles.csv')),
        standalone, data_dfs, signal_window,
    )
    if not resonance.empty:
        combined = pd.concat([standalone, resonance], ignore_index=True)
        combined.to_csv(os.path.join(OUTPUT_DIR, 'backtest_results.csv'), index=False, encoding='utf-8-sig')
        print(f"Total results: {len(standalone)} standalone + {len(resonance)} resonance = {len(combined)}")
        return combined
    return standalone


_ref_highs_cache = {}  # (stock_code, len(stock_df)) -> ref_highs list


def _build_stock_ref_highs(stock_df, stock_code):
    """
    Detect stock's own bull cycles and build date-indexed reference highs.
    Uses same algorithm as detect_cycles.py. Cycles cached to data/stock_cycles_{code}.csv.
    Returns list of ref_high values aligned to stock_df.index, or None on failure.

    进程内缓存：同一 session 内多次调用（回测/信号检测/Tab4渲染）只算一次。
    """
    from scipy.signal import argrelextrema

    cache_key = (stock_code, len(stock_df))
    if cache_key in _ref_highs_cache:
        return _ref_highs_cache[cache_key]

    cache_path = os.path.join(DATA_DIR, f'stock_cycles_{stock_code}.csv')
    cycle_df = None

    # Try cache
    if os.path.exists(cache_path):
        cycle_df = pd.read_csv(cache_path)
        if 'start_date' in cycle_df.columns:
            cycle_df['start_date'] = pd.to_datetime(cycle_df['start_date'])
            cycle_df['end_date'] = pd.to_datetime(cycle_df['end_date'])
        else:
            cycle_df = None  # force re-detection

    # Detect if not cached
    if cycle_df is None or cycle_df.empty:
        close = stock_df['close'].values.astype(float)
        dates = stock_df.index
        smoothed = pd.Series(close).rolling(SMOOTH_WINDOW, min_periods=1).mean().values
        troughs = argrelextrema(smoothed, np.less, order=ARGRELEXTREMA_ORDER)[0]
        peaks = argrelextrema(smoothed, np.greater, order=ARGRELEXTREMA_ORDER)[0]

        cycles = []
        for t_idx in troughs:
            later = peaks[peaks > t_idx]
            if len(later) == 0: continue
            p_idx = later[0]
            dur = int(p_idx - t_idx)
            if dur < MIN_DURATION_DAYS: continue
            chg = (close[p_idx] - close[t_idx]) / close[t_idx] * 100
            if chg < MIN_AMPLITUDE_PCT: continue
            cycles.append({
                'end_date': dates[p_idx],
                'end_price': round(float(close[p_idx]), 2),
                'change_pct': round(chg, 2),
            })

        if not cycles:
            _ref_highs_cache[cache_key] = None
            return None

        cycle_df = pd.DataFrame(cycles)
        cycle_df.to_csv(cache_path, index=False)

    # Hybrid: before first cycle end → rolling 250-day max; after → own cycle end price
    # After last cycle end: max(last_cycle_peak, rolling_max) — 防止漏检周期
    # 导致 ref_high 卡在旧高点（如东吴证券2024-11的8.22，实际2025-08已达10.69）
    first_cycle_end = cycle_df['end_date'].min()
    close_vals = stock_df['close'].values.astype(float)
    rolling_max = pd.Series(close_vals).rolling(MA_PERIOD, min_periods=1).max().values
    cycle_peaks_sorted = sorted([(c['end_date'], c['end_price']) for _, c in cycle_df.iterrows()])
    last_cycle_end = cycle_peaks_sorted[-1][0]

    ref_highs = []
    for i, d in enumerate(stock_df.index):
        if d < first_cycle_end:
            ref_highs.append(rolling_max[i])
        else:
            rh = None
            for peak_date, peak_price in cycle_peaks_sorted:
                if peak_date <= d: rh = peak_price
                else: break
            if rh is not None:
                # 最后一个周期终点之后：取 max(周期高点, rolling_max)
                # 防止漏检周期的高点丢失（如某轮涨幅被平滑缩水后低于阈值被过滤）
                if d > last_cycle_end:
                    rh = max(rh, rolling_max[i])
                ref_highs.append(rh)
            else:
                ref_highs.append(rolling_max[i])

    _ref_highs_cache[cache_key] = ref_highs
    return ref_highs


def run_stock_backtest(stock_code, stock_df, cycle_df,  signal_window=SIGNAL_WINDOW_STOCK):
    """
    Run backtest for a single stock against index-defined market cycles.
    Uses top 3 indicators from index backtest.

    Returns: pd.DataFrame with indicator results for this stock.
    """
    from indicators import INDICATOR_REGISTRY, filter_signals, get_all_signal_rules

    cycles_all = cycle_df.to_dict('records')
    cycles = [c for c in cycles_all if pd.Timestamp(c['start_date']) >= pd.Timestamp(CYCLE_FILTER_DATE)]

    df = stock_df.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date').sort_index()

    # Filter to cycles where stock has sufficient history (≥ 60 trading days before start)
    MIN_STOCK_HISTORY = 60
    stock_start = df.index[0]
    valid_cycles = []
    for c in cycles:
        cycle_start = pd.Timestamp(c['start_date'])
        history_days = len(df[df.index < cycle_start])
        if history_days >= MIN_STOCK_HISTORY:
            valid_cycles.append(c)
    if not valid_cycles:
        return pd.DataFrame()
    cycles = valid_cycles
    total_cycles = len(cycles)
    cycle_weights = np.array([c['change_pct'] / 100 for c in cycles])
    total_weight = cycle_weights.sum()

    # Build stock-specific ref_highs from stock's own detected bull cycles
    stock_ref_highs = _build_stock_ref_highs(df, stock_code)
    if stock_ref_highs is None:
        # Fallback: 该股无法检测到任何周期（涨幅不达标或数据太短）。
        # 用当天收盘价作 ref_high → decline≈0% → filter_signals 的"跌幅>阈值"条件
        # 恒不成立 → 信号基本不被周期过滤（仅受 MA/context 冷却约束）。
        # 这会导致此类股票发出更多信号，回测命中率通常偏低。属有意兜底，
        # 保证无周期股仍能进入回测而非被静默丢弃。
        stock_ref_highs = [float(df.loc[d, 'close']) if d in df.index else 0.0 for d in df.index]

    rules = get_all_signal_rules()
    results = []

    for rule in rules:
        # Skip moneyflow and resonance indicators for stock backtest
        if rule.get('requires_moneyflow') or rule['category'] == '共振':
            continue
        try:
            sig_raw = rule['func'](df.reset_index(), **rule['params'])
            sig_raw.index = df.index
            sig = filter_signals(df.reset_index(), sig_raw, ref_highs=stock_ref_highs)
            sig.index = df.index
        except Exception:
            continue

        total_signals = int(sig.sum())
        hits = 0
        days_list = []
        hit_count = 0
        cycle_signal_count = 0  # cycles with at least 1 signal in range
        weighted_hits = 0

        for cyc in cycles:
            ws = pd.Timestamp(cyc['start_date']) - pd.Timedelta(days=signal_window)
            we = pd.Timestamp(cyc['end_date']) + pd.Timedelta(days=LATE_CUTOFF_DAYS)
            wm = (sig.index >= ws) & (sig.index <= we)
            if sig[wm].any():
                cycle_signal_count += 1

        for i, cycle in enumerate(cycles):
            start = pd.Timestamp(cycle['start_date'])
            end = pd.Timestamp(cycle['end_date'])
            ws = start - pd.Timedelta(days=signal_window)
            we = start + pd.Timedelta(days=LATE_CUTOFF_DAYS)
            wm = (sig.index >= ws) & (sig.index <= we)
            if sig[wm].any():
                hit_count += 1

            j = judge_signal(sig, start, end, signal_window)
            if j['hit']:
                hits += 1
                days_list.append(j['days_before'])
                weighted_hits += cycle_weights[i]

        hit_rate = hits / total_cycles if total_cycles > 0 else 0
        weighted_hr = weighted_hits / total_weight if total_weight > 0 else 0
        avg_days = np.mean(days_list) if days_list else None
        precision = hit_count / cycle_signal_count if cycle_signal_count > 0 else 0
        days_score = 0
        if avg_days is not None:
            days_score = abs(avg_days - 5) / 30 * 0.3
        score = weighted_hr * 0.5 + precision * 0.2 - days_score

        results.append({
            '股票': stock_code,
            '指标名': rule['name'],
            '类别': rule['category'],
            '周期': '日线',
            '参数': _param_str(rule['params']),
            '命中率': round(weighted_hr, 3),
            '命中轮数': f'{hits}/{total_cycles}',
            '平均提前天': round(avg_days, 1) if avg_days is not None else None,
            '信号有效率': round(precision, 3),
            '总信号数': total_signals,
            '综合得分': round(score, 3),
        })

    return pd.DataFrame(results)


def run_all_stocks_backtest(data_dfs,  signal_window=SIGNAL_WINDOW_STOCK, force_regenerate=False):
    """Run backtest for all stocks, save to CSV.
    设置 force_regenerate=True 可在点击"强制刷新"时跳过磁盘缓存。
    """
    from data_fetcher import INDEX_CONSTITUENTS
    import pandas as pd

    cycle_path = os.path.join(OUTPUT_DIR, 'cycles.csv')
    cycle_df = pd.read_csv(cycle_path)
    stocks = {code: df for code, df in data_dfs.get('stocks_daily', {}).items() if df is not None and not df.empty}

    # 磁盘缓存：按数据最新日期判断是否需要重算
    result_path = os.path.join(OUTPUT_DIR, 'stock_results.csv')
    cache_meta_path = os.path.join(OUTPUT_DIR, '.stock_results_cache_meta')
    if stocks:
        # 取所有个股最新日期作为缓存 key
        latest_dates = []
        for code, df in stocks.items():
            df_tmp = df.copy()
            df_tmp['trade_date'] = pd.to_datetime(df_tmp['trade_date'])
            latest_dates.append(str(df_tmp['trade_date'].max().date()))
        data_sig = ','.join(sorted(latest_dates))

        if os.path.exists(result_path) and os.path.exists(cache_meta_path):
            try:
                with open(cache_meta_path, 'r') as f:
                    cached_sig = f.read().strip()
                # Skip cache if force_regenerate or code version changed
                cache_key = f'{data_sig}|{signal_window}'
                if not force_regenerate and cached_sig == cache_key:
                    combined = pd.read_csv(result_path)
                    if not combined.empty:
                        print(f"Stock results (cached): {len(combined)} rows")
                        return combined
            except Exception:
                pass

    all_results = []
    for code, df in stocks.items():
        name = INDEX_CONSTITUENTS.get(code, code)
        r = run_stock_backtest(code, df, cycle_df, signal_window)
        r['股票名'] = name
        all_results.append(r)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(result_path, index=False, encoding='utf-8-sig')
        try:
            with open(cache_meta_path, 'w') as f:
                f.write(f'{data_sig}|{signal_window}')
        except Exception:
            pass
        print(f"Stock results: {len(combined)} rows for {len(all_results)} stocks")
        return combined
    return pd.DataFrame()


# ============================================================
# Sell-side backtest engine
# ============================================================

_ref_lows_cache = {}


def _build_stock_ref_lows(stock_df, stock_code):
    """
    Detect stock's own bull cycles and build date-indexed reference lows.
    Mirror of _build_stock_ref_highs, using cycle troughs instead of peaks.
    """
    from scipy.signal import argrelextrema

    cache_key = (stock_code, len(stock_df), 'low')
    if cache_key in _ref_lows_cache:
        return _ref_lows_cache[cache_key]

    cache_path = os.path.join(DATA_DIR, f'stock_cycles_{stock_code}.csv')
    cycle_df = None

    if os.path.exists(cache_path):
        cycle_df = pd.read_csv(cache_path)
        if 'start_date' in cycle_df.columns:
            cycle_df['start_date'] = pd.to_datetime(cycle_df['start_date'])
            cycle_df['end_date'] = pd.to_datetime(cycle_df['end_date'])
        else:
            cycle_df = None  # force re-detection

    if cycle_df is None or cycle_df.empty:
        close = stock_df['close'].values.astype(float)
        dates = stock_df.index
        smoothed = pd.Series(close).rolling(SMOOTH_WINDOW, min_periods=1).mean().values
        troughs = argrelextrema(smoothed, np.less, order=ARGRELEXTREMA_ORDER)[0]
        peaks = argrelextrema(smoothed, np.greater, order=ARGRELEXTREMA_ORDER)[0]

        cycles = []
        for t_idx in troughs:
            later = peaks[peaks > t_idx]
            if len(later) == 0: continue
            p_idx = later[0]
            dur = int(p_idx - t_idx)
            if dur < MIN_DURATION_DAYS: continue
            chg = (close[p_idx] - close[t_idx]) / close[t_idx] * 100
            if chg < MIN_AMPLITUDE_PCT: continue
            cycles.append({
                'start_date': dates[t_idx],
                'start_price': round(float(close[t_idx]), 2),
                'end_date': dates[p_idx],
                'end_price': round(float(close[p_idx]), 2),
                'change_pct': round(chg, 2),
            })

        if not cycles:
            _ref_lows_cache[cache_key] = None
            return None

        cycle_df = pd.DataFrame(cycles)
        cycle_df.to_csv(cache_path, index=False)

    first_cycle_start = cycle_df['start_date'].min()
    close_vals = stock_df['close'].values.astype(float)
    rolling_min = pd.Series(close_vals).rolling(MA_PERIOD, min_periods=1).min().values
    cycle_troughs_sorted = sorted([(c['start_date'], c['start_price']) for _, c in cycle_df.iterrows()])
    last_cycle_start = cycle_troughs_sorted[-1][0]

    ref_lows = []
    for i, d in enumerate(stock_df.index):
        if d < first_cycle_start:
            ref_lows.append(rolling_min[i])
        else:
            rl = None
            for trough_date, trough_price in cycle_troughs_sorted:
                if trough_date <= d: rl = trough_price
                else: break
            if rl is not None:
                if d > last_cycle_start:
                    rl = min(rl, rolling_min[i])
                ref_lows.append(rl)
            else:
                ref_lows.append(rolling_min[i])

    _ref_lows_cache[cache_key] = ref_lows
    return ref_lows


def _compute_actual_peaks(df, cycles):
    """Replace smoothed peak price/date with actual highest close in each cycle."""
    for cycle in cycles:
        start = pd.Timestamp(cycle['start_date'])
        end = pd.Timestamp(cycle['end_date'])
        mask = (df.index >= start) & (df.index <= end)
        cycle_data = df[mask]
        if not cycle_data.empty:
            cycle['actual_peak_price'] = cycle_data['close'].max()
            cycle['actual_peak_date'] = cycle_data['close'].idxmax()
        else:
            cycle['actual_peak_price'] = cycle.get('end_price', 0)
            cycle['actual_peak_date'] = end
    return cycles


def judge_sell_signal(signal_series, actual_peak_date, actual_peak_price,
                      close_series):
    """
    Judge whether a sell signal fired near the cycle peak.

    Searches [peak - SELL_SEARCH_WINDOW, peak + SELL_SEARCH_WINDOW]
    for the closest signal. A signal is a "hit" if its capture rate
    (sell_price / peak_price) meets the SELL_HIT_CAPTURE threshold.

    Args:
        signal_series: pd.Series of bool, indexed by date
        actual_peak_date: pd.Timestamp, the actual peak date
        actual_peak_price: float, the actual peak close price
        close_series: pd.Series of close prices indexed by date

    Returns:
        dict with keys: hit, signal_date, days_from_peak, capture_rate
    """
    peak = pd.Timestamp(actual_peak_date)

    search_start = peak - pd.Timedelta(days=SELL_SEARCH_WINDOW)
    search_end = peak + pd.Timedelta(days=SELL_SEARCH_WINDOW)
    search_mask = pd.Series((signal_series.index >= search_start) & (signal_series.index <= search_end),
                            index=signal_series.index)
    search_window = signal_series[search_mask]

    if not search_window.any():
        return {'hit': False, 'signal_date': None, 'days_from_peak': None,
                'capture_rate': None}

    signal_dates = search_window[search_window].index
    signal_date = min(signal_dates, key=lambda d: abs((d - peak).days))
    days_from_peak = (signal_date - peak).days
    signal_close = close_series.loc[signal_date]
    capture_rate = signal_close / actual_peak_price

    return {
        'hit': capture_rate >= SELL_HIT_CAPTURE,
        'signal_date': signal_date,
        'days_from_peak': days_from_peak,
        'capture_rate': capture_rate,
    }


def count_false_sell_signals(signal_series, cycles):
    """Count sell signals outside any cycle peak's search window."""
    valid_mask = pd.Series(False, index=signal_series.index)
    for cycle in cycles:
        peak = pd.Timestamp(cycle['actual_peak_date'])
        ws = peak - pd.Timedelta(days=SELL_SEARCH_WINDOW)
        we = peak + pd.Timedelta(days=SELL_SEARCH_WINDOW)
        valid_mask[(signal_series.index >= ws) & (signal_series.index <= we)] = True
    return (signal_series & ~valid_mask).sum()


def run_sell_backtest(cycle_df, sell_signals_list, data_dfs):
    """
    Run backtest for all sell signal rules against all cycles.
    Mirror of run_backtest with peak as anchor instead of trough.
    """
    from indicators import filter_sell_signals

    cycles_all = cycle_df.to_dict('records')
    cycles = [c for c in cycles_all if pd.Timestamp(c['start_date']) >= pd.Timestamp(CYCLE_FILTER_DATE)]
    total_cycles = len(cycles)
    cycle_weights = np.array([c['change_pct'] / 100 for c in cycles])
    total_weight = cycle_weights.sum()

    results = []

    for freq_key, df in [('日线', data_dfs.get('index_daily')),
                          ('周线', data_dfs.get('index_weekly'))]:
        if df is None or df.empty:
            continue

        df = df.set_index('trade_date').sort_index()
        cycles = _compute_actual_peaks(df, cycles)

        # Build ref_lows from cycle troughs (mirror of ref_highs)
        cycle_troughs = sorted([(pd.Timestamp(c['start_date']), c['start_price']) for c in cycles_all])
        ref_lows = []
        for d in df.index:
            rl = None
            for td, tp in cycle_troughs:
                if td <= d: rl = tp
                else: break
            ref_lows.append(rl if rl is not None else df.loc[d, 'close'])

        close_series = df['close']

        for rule in sell_signals_list:
            work_df = df.copy()

            if rule.get('requires_moneyflow', False):
                mf_df = data_dfs.get('index_moneyflow')
                if mf_df is None or mf_df.empty:
                    continue
                mf_df = mf_df.set_index('trade_date').sort_index()
                for col in mf_df.columns:
                    if col not in work_df.columns:
                        work_df[col] = mf_df[col]

            try:
                signal_series = rule['func'](work_df.reset_index(), **rule['params'])
                signal_series.index = work_df.index
                signal_series = filter_sell_signals(
                    work_df.reset_index(), signal_series, ref_lows=ref_lows,
                )
                signal_series.index = work_df.index
            except Exception as e:
                print(f"  [SKIP-SELL] {rule['name']} ({freq_key}): {e}")
                continue

            total_signals = int(signal_series.sum())
            hits = 0
            capture_list = []
            days_list = []
            hit_signal_count = 0
            cycle_signal_count = 0  # cycles with at least 1 signal in range
            weighted_hits = 0

            for cyc in cycles:
                ws = pd.Timestamp(cyc['start_date']) - pd.Timedelta(days=SELL_SEARCH_WINDOW)
                we = pd.Timestamp(cyc['end_date']) + pd.Timedelta(days=SELL_SEARCH_WINDOW)
                wm = (signal_series.index >= ws) & (signal_series.index <= we)
                if signal_series[wm].any():
                    cycle_signal_count += 1

            for i, cycle in enumerate(cycles):
                peak_date = pd.Timestamp(cycle['actual_peak_date'])
                peak_price = cycle['actual_peak_price']

                ws = peak_date - pd.Timedelta(days=SELL_SEARCH_WINDOW)
                we = peak_date + pd.Timedelta(days=SELL_SEARCH_WINDOW)
                wm = (signal_series.index >= ws) & (signal_series.index <= we)
                if signal_series[wm].any():
                    hit_signal_count += 1

                judgment = judge_sell_signal(
                    signal_series, peak_date, peak_price, close_series,
                )
                if judgment['capture_rate'] is not None:
                    capture_list.append(judgment['capture_rate'])
                    days_list.append(judgment['days_from_peak'])
                if judgment['hit']:
                    hits += 1
                    weighted_hits += cycle_weights[i]

            hit_rate = hits / total_cycles if total_cycles > 0 else 0
            weighted_hr = weighted_hits / total_weight if total_weight > 0 else 0
            avg_capture = np.mean(capture_list) if capture_list else 0
            avg_days = np.mean(days_list) if days_list else None
            precision = hit_signal_count / cycle_signal_count if cycle_signal_count > 0 else 0

            score = (avg_capture * WEIGHT_SELL_CAPTURE
                     + weighted_hr * WEIGHT_SELL_HIT_RATE
                     + precision * WEIGHT_SELL_PRECISION)

            results.append({
                '指标名': rule['name'],
                '类别': rule['category'],
                '周期': freq_key,
                '参数': _param_str(rule['params']),
                '命中率': round(weighted_hr, 3),
                '命中轮数': f'{hits}/{total_cycles}',
                '平均捕获率': round(avg_capture, 3),
                '平均距峰天': round(avg_days, 1) if avg_days is not None else None,
                '信号有效率': round(precision, 3),
                '总信号数': total_signals,
                '综合得分': round(score, 3),
            })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('综合得分', ascending=False).reset_index(drop=True)
    return results_df


def run_and_save_sell(data_dfs):
    """Run sell backtest and save to CSV."""
    from indicators import get_all_sell_rules

    cycle_path = os.path.join(OUTPUT_DIR, 'cycles.csv')
    if not os.path.exists(cycle_path):
        raise FileNotFoundError(f"{cycle_path} not found. Run detect_cycles.py first.")

    cycle_df = pd.read_csv(cycle_path)
    rules = get_all_sell_rules()
    results = run_sell_backtest(cycle_df, rules, data_dfs)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, 'sell_backtest_results.csv')
    results.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"Saved {len(results)} sell results to {path}")
    return results


def run_combined_backtest(cycle_df, buy_results, sell_results, index_daily,
                          signal_window=SIGNAL_WINDOW_INDEX):
    """
    Combined buy-sell backtest: for each cycle, simulate a full trade
    using top-ranked buy and sell indicators.
    """
    from indicators import INDICATOR_REGISTRY, SELL_INDICATOR_REGISTRY, filter_signals, filter_sell_signals

    if buy_results.empty or sell_results.empty:
        return pd.DataFrame()

    cycles_all = cycle_df.to_dict('records')
    cycles = [c for c in cycles_all if pd.Timestamp(c['start_date']) >= pd.Timestamp(CYCLE_FILTER_DATE)]

    df = index_daily.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date').sort_index()
    cycles = _compute_actual_peaks(df, cycles)
    close_series = df['close']

    buy_top = buy_results.iloc[0]
    sell_top = sell_results.iloc[0]
    buy_name = buy_top['指标名']
    sell_name = sell_top['指标名']

    buy_cfg = INDICATOR_REGISTRY.get(buy_name)
    sell_cfg = SELL_INDICATOR_REGISTRY.get(sell_name)
    if buy_cfg is None or sell_cfg is None:
        return pd.DataFrame()

    # Build ref_highs for buy filter
    cycle_peaks = sorted([(pd.Timestamp(c['end_date']), c['end_price']) for c in cycles_all])
    ref_highs = []
    for d in df.index:
        rh = None
        for pd_, pp in cycle_peaks:
            if pd_ <= d: rh = pp
            else: break
        ref_highs.append(rh if rh is not None else df.loc[d, 'close'])

    # Build ref_lows for sell filter
    cycle_troughs = sorted([(pd.Timestamp(c['start_date']), c['start_price']) for c in cycles_all])
    ref_lows = []
    for d in df.index:
        rl = None
        for td, tp in cycle_troughs:
            if td <= d: rl = tp
            else: break
        ref_lows.append(rl if rl is not None else df.loc[d, 'close'])

    # Compute buy signals
    buy_sig = buy_cfg['func'](df.reset_index(), **buy_cfg['params'][0])
    buy_sig.index = df.index
    buy_sig = filter_signals(df.reset_index(), buy_sig, ref_highs=ref_highs)
    buy_sig.index = df.index

    # Compute sell signals
    sell_sig = sell_cfg['func'](df.reset_index(), **sell_cfg['params'][0])
    sell_sig.index = df.index
    sell_sig = filter_sell_signals(df.reset_index(), sell_sig, ref_lows=ref_lows)
    sell_sig.index = df.index

    results = []
    for i, cycle in enumerate(cycles):
        start = pd.Timestamp(cycle['start_date'])
        peak_date = pd.Timestamp(cycle['actual_peak_date'])
        peak_price = cycle['actual_peak_price']
        end = pd.Timestamp(cycle['end_date'])

        # Find buy signal in [start - signal_window, start + LATE_CUTOFF_DAYS]
        buy_ws = start - pd.Timedelta(days=signal_window)
        buy_we = start + pd.Timedelta(days=LATE_CUTOFF_DAYS)
        buy_wm = (buy_sig.index >= buy_ws) & (buy_sig.index <= buy_we)
        buy_signals = buy_sig[buy_wm]
        buy_signal_date = buy_signals[buy_signals].index[-1] if buy_signals.any() else None

        if buy_signal_date is None:
            continue

        buy_price = close_series.loc[buy_signal_date]

        # Find sell signal after buy, closest to peak
        sell_wm = (sell_sig.index > buy_signal_date) & (sell_sig.index <= end)
        sell_signals = sell_sig[sell_wm]
        if sell_signals.any():
            sell_dates = sell_signals[sell_signals].index
            sell_signal_date = min(sell_dates, key=lambda d: abs((d - peak_date).days))
            sell_price = close_series.loc[sell_signal_date]
            holding_days = (sell_signal_date - buy_signal_date).days
            exited = True
        else:
            sell_signal_date = None
            sell_price = close_series.loc[end]
            holding_days = (end - buy_signal_date).days
            exited = False

        actual_return = (sell_price - buy_price) / buy_price * 100
        max_return = (peak_price - buy_price) / buy_price * 100
        combined_capture = actual_return / max_return * 100 if max_return > 0 else 0

        if exited:
            if combined_capture > 80: rating = '★★★'
            elif combined_capture > 50: rating = '★★'
            elif combined_capture > 0: rating = '★'
            else: rating = '✗'
        else:
            rating = '未退出'

        results.append({
            '周期': i + 1,
            '起始日': start.strftime('%Y-%m-%d'),
            '峰值日': peak_date.strftime('%Y-%m-%d'),
            '买入指标': buy_name,
            '买入日': buy_signal_date.strftime('%Y-%m-%d'),
            '买入价': round(buy_price, 2),
            '卖出指标': sell_name if exited else '—',
            '卖出日': sell_signal_date.strftime('%Y-%m-%d') if exited else '持有至周期结束',
            '卖出价': round(sell_price, 2),
            '持有天数': holding_days,
            '实际收益率%': round(actual_return, 1),
            '最大可获收益率%': round(max_return, 1),
            '捕获率%': round(combined_capture, 1),
            '评级': rating,
        })

    return pd.DataFrame(results)


# ============================================================
# Stock-level sell backtest
# ============================================================

def run_stock_sell_backtest(stock_code, stock_df, cycle_df):
    """Run sell backtest for a single stock against index-defined market cycles.
    Mirror of run_stock_backtest but for sell indicators."""
    from indicators import SELL_INDICATOR_REGISTRY, filter_sell_signals, get_all_sell_rules

    cycles_all = cycle_df.to_dict('records')
    cycles = [c for c in cycles_all if pd.Timestamp(c['start_date']) >= pd.Timestamp(CYCLE_FILTER_DATE)]

    df = stock_df.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date').sort_index()

    # Filter to cycles where stock has sufficient history
    MIN_STOCK_HISTORY = 60
    valid_cycles = []
    for c in cycles:
        cycle_start = pd.Timestamp(c['start_date'])
        history_days = len(df[df.index < cycle_start])
        if history_days >= MIN_STOCK_HISTORY:
            valid_cycles.append(c)
    if not valid_cycles:
        return pd.DataFrame()
    cycles = valid_cycles
    total_cycles = len(cycles)
    cycle_weights = np.array([c['change_pct'] / 100 for c in cycles])
    total_weight = cycle_weights.sum()

    cycles = _compute_actual_peaks(df, cycles)

    # Build ref_lows for sell filter
    stock_ref_lows = _build_stock_ref_lows(df, stock_code)
    if stock_ref_lows is None:
        stock_ref_lows = [float(df.loc[d, 'close']) if d in df.index else 0.0 for d in df.index]

    close_series = df['close']

    rules = get_all_sell_rules()
    results = []

    for rule in rules:
        if rule.get('requires_moneyflow') or rule['category'] == '共振':
            continue
        try:
            sig_raw = rule['func'](df.reset_index(), **rule['params'])
            sig_raw.index = df.index
            sig = filter_sell_signals(df.reset_index(), sig_raw, ref_lows=stock_ref_lows)
            sig.index = df.index
        except Exception:
            continue

        total_signals = int(sig.sum())
        hits = 0
        capture_list = []
        days_list = []
        hit_signal_count = 0
        cycle_signal_count = 0
        weighted_hits = 0

        for cyc in cycles:
            ws = pd.Timestamp(cyc['start_date']) - pd.Timedelta(days=SELL_SEARCH_WINDOW)
            we = pd.Timestamp(cyc['end_date']) + pd.Timedelta(days=SELL_SEARCH_WINDOW)
            wm = (sig.index >= ws) & (sig.index <= we)
            if sig[wm].any():
                cycle_signal_count += 1

        for i, cycle in enumerate(cycles):
            peak_date = pd.Timestamp(cycle['actual_peak_date'])
            peak_price = cycle['actual_peak_price']

            ws = peak_date - pd.Timedelta(days=SELL_SEARCH_WINDOW)
            we = peak_date + pd.Timedelta(days=SELL_SEARCH_WINDOW)
            wm = (sig.index >= ws) & (sig.index <= we)
            if sig[wm].any():
                hit_signal_count += 1

            judgment = judge_sell_signal(sig, peak_date, peak_price, close_series)
            if judgment['capture_rate'] is not None:
                capture_list.append(judgment['capture_rate'])
                days_list.append(judgment['days_from_peak'])
            if judgment['hit']:
                hits += 1
                weighted_hits += cycle_weights[i]

        hit_rate = hits / total_cycles if total_cycles > 0 else 0
        weighted_hr = weighted_hits / total_weight if total_weight > 0 else 0
        avg_capture = np.mean(capture_list) if capture_list else 0
        avg_days = np.mean(days_list) if days_list else None
        precision = hit_signal_count / cycle_signal_count if cycle_signal_count > 0 else 0

        score = (avg_capture * WEIGHT_SELL_CAPTURE
                 + weighted_hr * WEIGHT_SELL_HIT_RATE
                 + precision * WEIGHT_SELL_PRECISION)

        results.append({
            '股票': stock_code,
            '指标名': rule['name'],
            '类别': rule['category'],
            '周期': '日线',
            '参数': _param_str(rule['params']),
            '命中率': round(weighted_hr, 3),
            '命中轮数': f'{hits}/{total_cycles}',
            '平均捕获率': round(avg_capture, 3),
            '平均距峰天': round(avg_days, 1) if avg_days is not None else None,
            '信号有效率': round(precision, 3),
            '总信号数': total_signals,
            '综合得分': round(score, 3),
        })

    return pd.DataFrame(results)


def run_all_stocks_sell_backtest(data_dfs, force_regenerate=False):
    """Run sell backtest for all stocks, save to CSV."""
    from data_fetcher import INDEX_CONSTITUENTS

    cycle_path = os.path.join(OUTPUT_DIR, 'cycles.csv')
    cycle_df = pd.read_csv(cycle_path)
    stocks = {code: df for code, df in data_dfs.get('stocks_daily', {}).items() if df is not None and not df.empty}

    result_path = os.path.join(OUTPUT_DIR, 'stock_sell_results.csv')
    cache_meta_path = os.path.join(OUTPUT_DIR, '.stock_sell_results_cache_meta')
    if stocks:
        latest_dates = []
        for code, df in stocks.items():
            df_tmp = df.copy()
            df_tmp['trade_date'] = pd.to_datetime(df_tmp['trade_date'])
            latest_dates.append(str(df_tmp['trade_date'].max().date()))
        data_sig = ','.join(sorted(latest_dates))

        if not force_regenerate and os.path.exists(result_path) and os.path.exists(cache_meta_path):
            try:
                with open(cache_meta_path, 'r') as f:
                    cached_sig = f.read().strip()
                if cached_sig == data_sig:
                    combined = pd.read_csv(result_path)
                    if not combined.empty:
                        print(f"Stock sell results (cached): {len(combined)} rows")
                        return combined
            except Exception:
                pass

    all_results = []
    for code, df in stocks.items():
        name = INDEX_CONSTITUENTS.get(code, code)
        r = run_stock_sell_backtest(code, df, cycle_df)
        if not r.empty:
            r['股票名'] = name
            all_results.append(r)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(result_path, index=False, encoding='utf-8-sig')
        try:
            with open(cache_meta_path, 'w') as f:
                f.write(data_sig)
        except Exception:
            pass
        print(f"Stock sell results: {len(combined)} rows for {len(all_results)} stocks")
        return combined
    return pd.DataFrame()
