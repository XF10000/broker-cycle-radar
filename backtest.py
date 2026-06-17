"""
Backtest engine: evaluates technical indicator signals against known market cycles.
"""
import os
import pandas as pd
import numpy as np

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
}

def _param_str(params):
    """Convert param dict to readable Chinese string."""
    parts = []
    for k, v in params.items():
        cn = _PARAM_CN.get(k, k)
        parts.append(f'{cn}={v}')
    return ', '.join(parts)


def judge_signal(signal_series, cycle_start, cycle_end, signal_window=30):
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
    late_cutoff = cycle_start + pd.Timedelta(days=10)
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


def count_false_signals(signal_series, cycles, signal_window=30):
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
        window_end = start + pd.Timedelta(days=10)
        valid_mask[(signal_series.index >= window_start) & (signal_series.index <= window_end)] = True

    false_signals = signal_series & ~valid_mask
    return false_signals.sum()


def run_backtest(cycle_df, signals_list, data_dfs, signal_window=30):
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
    cycles = [c for c in cycles_all if pd.Timestamp(c['start_date']) >= pd.Timestamp('2010-01-01')]
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
            weighted_hits = 0

            for i, cycle in enumerate(cycles):
                start = pd.Timestamp(cycle['start_date'])
                end = pd.Timestamp(cycle['end_date'])
                window_start = start - pd.Timedelta(days=signal_window)
                window_end = start + pd.Timedelta(days=10)

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
            precision = hit_signal_count / total_signals if total_signals > 0 else 0

            # Composite score: balance hit_rate, signal precision, and timing
            days_score = 0
            if avg_days is not None:
                days_score = abs(avg_days - 5) / 30 * 0.3
            score = weighted_hr * 0.5 + precision * 0.2 - days_score

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


def run_and_save(data_dfs, signal_window=30):
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


def run_resonance_backtest(cycle_df, standalone_results, data_dfs, signal_window=45):
    """
    Multi-timeframe resonance backtest: weekly indicator fires first,
    then daily indicator confirms within a 30-day observation window.
    """
    from indicators import INDICATOR_REGISTRY, filter_signals

    cycles_all = cycle_df.to_dict('records')
    cycles = [c for c in cycles_all if pd.Timestamp(c['start_date']) >= pd.Timestamp('2010-01-01')]
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
                window_end = ws_date + pd.Timedelta(days=30)
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
                window_end = start + pd.Timedelta(days=10)

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
                days_score = abs(avg_days - 5) / 30 * 0.3
            score = weighted_hr * 0.5 + precision * 0.2 - days_score

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


def run_and_save_all(data_dfs, signal_window=45):
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


def _build_stock_ref_highs(stock_df, stock_code):
    """
    Detect stock's own bull cycles and build date-indexed reference highs.
    Uses same algorithm as detect_cycles.py. Cycles cached to data/stock_cycles_{code}.csv.
    Returns list of ref_high values aligned to stock_df.index, or None on failure.
    """
    from scipy.signal import argrelextrema

    cache_path = os.path.join(DATA_DIR, f'stock_cycles_{stock_code}.csv')
    cycle_df = None

    # Try cache
    if os.path.exists(cache_path):
        cycle_df = pd.read_csv(cache_path)
        cycle_df['end_date'] = pd.to_datetime(cycle_df['end_date'])

    # Detect if not cached
    if cycle_df is None or cycle_df.empty:
        close = stock_df['close'].values.astype(float)
        dates = stock_df.index
        smoothed = pd.Series(close).rolling(20, min_periods=1).mean().values
        troughs = argrelextrema(smoothed, np.less, order=10)[0]
        peaks = argrelextrema(smoothed, np.greater, order=10)[0]

        cycles = []
        for t_idx in troughs:
            later = peaks[peaks > t_idx]
            if len(later) == 0: continue
            p_idx = later[0]
            dur = int(p_idx - t_idx)
            if dur < 20: continue
            chg = (close[p_idx] - close[t_idx]) / close[t_idx] * 100
            if chg < 25: continue
            cycles.append({
                'end_date': dates[p_idx],
                'end_price': round(float(close[p_idx]), 2),
                'change_pct': round(chg, 2),
            })

    if not cycles:
        return None

    cycle_df = pd.DataFrame(cycles)
    cycle_df.to_csv(cache_path, index=False)

    # Hybrid: before first cycle end → rolling 250-day max; after → own cycle end price
    first_cycle_end = cycle_df['end_date'].min()
    close_vals = stock_df['close'].values.astype(float)
    rolling_max = pd.Series(close_vals).rolling(250, min_periods=1).max().values
    cycle_peaks_sorted = sorted([(c['end_date'], c['end_price']) for _, c in cycle_df.iterrows()])

    ref_highs = []
    for i, d in enumerate(stock_df.index):
        if d < first_cycle_end:
            ref_highs.append(rolling_max[i])
        else:
            rh = None
            for peak_date, peak_price in cycle_peaks_sorted:
                if peak_date <= d: rh = peak_price
                else: break
            ref_highs.append(rh if rh is not None else rolling_max[i])

    return ref_highs


def run_stock_backtest(stock_code, stock_df, cycle_df, signal_window=45):
    """
    Run backtest for a single stock against index-defined market cycles.
    Uses top 3 indicators from index backtest.

    Returns: pd.DataFrame with indicator results for this stock.
    """
    from indicators import INDICATOR_REGISTRY, filter_signals, get_all_signal_rules

    cycles_all = cycle_df.to_dict('records')
    cycles = [c for c in cycles_all if pd.Timestamp(c['start_date']) >= pd.Timestamp('2010-01-01')]
    total_cycles = len(cycles)
    cycle_weights = np.array([c['change_pct'] / 100 for c in cycles])
    total_weight = cycle_weights.sum()

    df = stock_df.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date').sort_index()

    # Build stock-specific ref_highs from stock's own detected bull cycles
    stock_ref_highs = _build_stock_ref_highs(df, stock_code)
    if stock_ref_highs is None:
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
        weighted_hits = 0

        for cycle in cycles:
            start = pd.Timestamp(cycle['start_date'])
            end = pd.Timestamp(cycle['end_date'])
            ws = start - pd.Timedelta(days=signal_window)
            we = start + pd.Timedelta(days=10)
            wm = (sig.index >= ws) & (sig.index <= we)
            if sig[wm].any():
                hit_count += 1

            j = judge_signal(sig, start, end, signal_window)
            if j['hit']:
                hits += 1
                days_list.append(j['days_before'])

            hit_rate = hits / total_cycles if total_cycles > 0 else 0
            weighted_hr = weighted_hits / total_weight if total_weight > 0 else 0
            avg_days = np.mean(days_list) if days_list else None
        precision = hit_count / total_signals if total_signals > 0 else 0
        days_score = 0
        if avg_days is not None:
            days_score = abs(avg_days - 5) / 30 * 0.3
        score = weighted_hr * 0.5 + precision * 0.2 - days_score

        results.append({
            '股票': stock_code,
            '指标名': rule['name'],
            '类别': rule['category'],
            '周期': '日线',
            '命中率': round(weighted_hr, 3),
            '命中轮数': f'{hits}/{total_cycles}',
            '平均提前天': round(avg_days, 1) if avg_days is not None else None,
            '信号有效率': round(precision, 3),
            '总信号数': total_signals,
            '综合得分': round(score, 3),
        })

    return pd.DataFrame(results)


def run_all_stocks_backtest(data_dfs, signal_window=45):
    """Run backtest for all stocks, save to CSV."""
    from data_fetcher import INDEX_CONSTITUENTS
    import pandas as pd

    cycle_path = os.path.join(OUTPUT_DIR, 'cycles.csv')
    cycle_df = pd.read_csv(cycle_path)
    all_results = []

    for code, df in data_dfs.get('stocks_daily', {}).items():
        if df is None or df.empty:
            continue
        name = INDEX_CONSTITUENTS.get(code, code)
        r = run_stock_backtest(code, df, cycle_df, signal_window)
        r['股票名'] = name
        all_results.append(r)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        path = os.path.join(OUTPUT_DIR, 'stock_results.csv')
        combined.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"Stock results: {len(combined)} rows for {len(all_results)} stocks")
        return combined
    return pd.DataFrame()
