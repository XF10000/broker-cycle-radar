"""
Streamlit app for securities sector technical indicator backtesting.
Three tabs: Market Cycles, Indicator Rankings, Stock Verification.

Usage: streamlit run app.py
"""
import os
import ast
import sys
import pandas as pd
import numpy as np
import talib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import (
    fetch_index_daily, fetch_all_stocks_daily, daily_to_weekly,
    STOCKS, INDEX_CODE, INDEX_CONSTITUENTS,
)
from indicators import get_all_signal_rules, INDICATOR_REGISTRY, filter_signals, get_indicator_lines
from backtest import run_backtest, run_and_save, run_and_save_all, judge_signal, count_false_signals, run_all_stocks_backtest
from lead_lag import compute_lead_lag, analyze_consistency, compute_current_returns, detect_recent_low

from config import (
    MA_PERIOD, DECLINE_PCT, DECLINE_FACTOR, SIGNAL_WINDOW_INDEX,
    SIGNAL_WINDOW_STOCK, LATE_CUTOFF_DAYS, RESONANCE_WINDOW_DAYS,
    SCORE_MAX, HEATMAP_HIT_FAST, HEATMAP_LATE_FAST,
    CYCLE_FILTER_DATE, CACHE_TTL_SECONDS, CACHE_FRESH_HOURS,
    SMOOTH_WINDOW, MIN_AMPLITUDE_PCT, MIN_DURATION_DAYS, ARGRELEXTREMA_ORDER,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')


st.set_page_config(
    page_title="券商板块信号回测",
    page_icon="📊",
    layout="wide",
)


def init_session():
    defaults = {
        'data_loaded': False,
        'index_daily': None,
        'index_weekly': None,
        'stocks_daily': {},
        'stocks_weekly': {},
        'backtest_results': None,
        'stock_results': None,
        'cycles_df': None,
        'display_cycles': None,
        'signal_window': SIGNAL_WINDOW_INDEX,
        'last_data_date': '',
        'data_error': '',
        'odds_df': None,
        'odds_meta': {},
        'odds_signal_cache': None,
        'odds_favorites': set(),
        'odds_data_date': '',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_all_data_cached(force_refresh=False):
    """Load all data with caching. Returns dict. Set force_refresh=True to bypass local CSV cache."""
    from data_fetcher import fetch_index_daily as _fetch_idx, fetch_stock_daily as _fetch_stk
    result = {}
    errors = []

    try:
        result['index_daily'] = _fetch_idx(force=force_refresh)
        result['index_weekly'] = daily_to_weekly(result['index_daily'])
    except Exception as e:
        errors.append(f'指数数据: {e}')

    result['stocks_daily'] = {}
    result['stocks_weekly'] = {}
    for code in INDEX_CONSTITUENTS:
        ts_code = f'{code}.SH' if code.startswith('6') else f'{code}.SZ'
        try:
            sd = _fetch_stk(ts_code, force=force_refresh)
            if not sd.empty:
                result['stocks_daily'][code] = sd
                result['stocks_weekly'][code] = daily_to_weekly(sd)
        except Exception as e:
            errors.append(f'个股{code}: {e}')

    result['errors'] = errors
    return result


def load_data(force=False):
    if force:
        st.cache_data.clear()
    with st.spinner('加载数据中...'):
        data = load_all_data_cached(force_refresh=force)

    index_daily = data.get('index_daily')
    # 指数数据是后续一切分析的基础：失败则不标记 data_loaded，避免在不完整数据上跑回测
    if index_daily is None or index_daily.empty:
        st.session_state.data_loaded = False
        st.session_state.index_daily = None
        st.session_state.index_weekly = None
        st.session_state.stocks_daily = {}
        st.session_state.stocks_weekly = {}
        st.session_state.last_data_date = ''
        st.session_state.data_error = '指数数据加载失败：' + '; '.join(data.get('errors', [])) or '未知错误'
        return

    st.session_state.index_daily = index_daily
    st.session_state.index_weekly = data.get('index_weekly')
    st.session_state.stocks_daily = data.get('stocks_daily', {})
    st.session_state.stocks_weekly = data.get('stocks_weekly', {})
    st.session_state.last_data_date = str(index_daily['trade_date'].max().date())

    # 个股部分失败：分类汇总，醒目提示
    errors = data.get('errors', [])
    stock_errors = [e for e in errors if e.startswith('个股')]
    other_errors = [e for e in errors if not e.startswith('个股')]
    total = len(INDEX_CONSTITUENTS)
    failed = len(stock_errors)
    if failed == 0 and not other_errors:
        st.session_state.data_error = ''
    else:
        parts = []
        if other_errors:
            parts.append('; '.join(other_errors))
        if failed > 0:
            ratio = failed / total if total else 0
            tag = '⚠️ 部分失败' if ratio < 0.3 else '❌ 大量失败'
            parts.append(f'{tag}: {failed}/{total} 只个股加载失败，回测将在剩余个股上运行')
        st.session_state.data_error = ' | '.join(parts)

    st.session_state.data_loaded = True


def load_cycles():
    path = os.path.join(OUTPUT_DIR, 'cycles.csv')
    if os.path.exists(path):
        all_df = pd.read_csv(path)
        st.session_state.cycles_df = all_df  # keep all for reference calc
        # Display only 2010+ cycles
        st.session_state.display_cycles = all_df[
            pd.to_datetime(all_df['start_date']) >= pd.Timestamp(CYCLE_FILTER_DATE)
        ]
        return True
    return False


def load_odds():
    """Load stock odds CSV into session state."""
    path = os.path.join(OUTPUT_DIR, 'stock_odds.csv')
    if not os.path.exists(path):
        return False
    meta = {}
    lines = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if 'M=' in line:
                    parts = line.strip('# ').split(',')
                    for p in parts:
                        if '=' in p:
                            k, v = p.split('=', 1)
                            meta[k.strip()] = v.strip()
            else:
                lines.append(line)
    import io
    df = pd.read_csv(io.StringIO(''.join(lines)))
    st.session_state.odds_df = df
    st.session_state.odds_meta = meta
    st.session_state.odds_data_date = meta.get('updated', '')
    return True


def compute_odds_signals():
    """
    Compute buy signals for all stocks in odds_df using top 3 indicators.
    Returns DataFrame with ts_code + signal columns.
    Cached to output/signal_cache.csv.
    """
    from indicators import INDICATOR_REGISTRY, filter_signals

    cache_path = os.path.join(OUTPUT_DIR, 'signal_cache.csv')
    last_date = st.session_state.get('last_data_date', '')

    if os.path.exists(cache_path) and last_date:
        with open(cache_path, 'r') as f:
            first = f.readline()
        if last_date in first:
            try:
                cached = pd.read_csv(cache_path, comment='#')
                if not cached.empty:
                    st.session_state.odds_signal_cache = cached
                    return cached
            except Exception:
                pass

    df = st.session_state.odds_df
    if df is None or df.empty:
        return pd.DataFrame()

    top_inds = [
        ('OBV底背离', INDICATOR_REGISTRY['OBV底背离']['params'][0]),
        ('CCI脱离超卖', INDICATOR_REGISTRY['CCI脱离超卖']['params'][0]),
        ('MACD柱线缩短', INDICATOR_REGISTRY['MACD柱线缩短']['params'][0]),
    ]

    from backtest import _build_stock_ref_highs

    records = []
    progress = st.progress(0, '检测各股信号...')
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        ts_code = row['ts_code']
        code = ts_code.split('.')[0]
        stock_path = os.path.join(DATA_DIR, f'stock_daily_{code}.csv')
        if not os.path.exists(stock_path):
            records.append({'ts_code': ts_code, 'signal': 'no_data', 'signal_count': 0})
        else:
            sdf = pd.read_csv(stock_path)
            sdf['trade_date'] = pd.to_datetime(sdf['trade_date'])
            sdf = sdf.sort_values('trade_date').reset_index(drop=True)
            # 与回测口径一致：用个股自身周期高点构建 ref_highs
            try:
                stock_ref_highs = _build_stock_ref_highs(sdf.set_index('trade_date'), code)
            except Exception:
                stock_ref_highs = None
            sig_count = 0
            for ind_name, params in top_inds:
                cfg = INDICATOR_REGISTRY.get(ind_name)
                if cfg is None:
                    continue
                try:
                    raw = cfg['func'](sdf, **params)
                    filtered = filter_signals(sdf, raw, ref_highs=stock_ref_highs)
                    if filtered.tail(30).any():
                        sig_count += 1
                except Exception:
                    pass
            signal_label = 'buy' if sig_count >= 1 else 'none'
            records.append({'ts_code': ts_code, 'signal': signal_label, 'signal_count': sig_count})
        progress.progress((i + 1) / total)

    progress.empty()
    result = pd.DataFrame(records)
    with open(cache_path, 'w') as f:
        f.write(f"# cached_date={last_date}\n")
    result.to_csv(cache_path, mode='a', index=False)
    st.session_state.odds_signal_cache = result
    return result


def toggle_favorite(ts_code):
    """Toggle a stock in/out of favorites set."""
    favs = st.session_state.odds_favorites
    if ts_code in favs:
        favs.discard(ts_code)
    else:
        favs.add(ts_code)


def run_backtest_now():
    if st.session_state.index_daily is None:
        st.error("请先加载数据")
        return False
    data_dfs = {
        'index_daily': st.session_state.index_daily,
        'index_weekly': st.session_state.index_weekly,
        'index_moneyflow': None,
    }
    try:
        results = run_and_save_all(data_dfs, st.session_state.signal_window)
        st.session_state.backtest_results = results
        # 个股回测：优先用磁盘缓存（同交易日不重算）
        stock_data = {'stocks_daily': st.session_state.stocks_daily}
        st.session_state.stock_results = run_all_stocks_backtest(stock_data, st.session_state.signal_window)
        return True
    except Exception as e:
        st.error(f"回测失败: {e}")
        return False


def render_stock_backtest():
    st.header("个股回测对比")

    # Try loading cached results first
    if st.session_state.stock_results is None:
        path = os.path.join(OUTPUT_DIR, 'stock_results.csv')
        if os.path.exists(path):
            st.session_state.stock_results = pd.read_csv(path)
        else:
            st.info("请先在侧边栏点击「重新回测」生成个股结果")
            return

    df = st.session_state.stock_results
    if df is None or df.empty:
        st.info("暂无个股回测数据")
        return

    # Comparison table: best indicator per stock
    st.subheader("各股票最佳指标得分")
    stock_names = df['股票名'].unique()
    summary = []
    for sn in stock_names:
        sdf = df[df['股票名'] == sn].sort_values('综合得分', ascending=False)
        if sdf.empty:
            continue
        best = sdf.iloc[0]
        summary.append({
            '股票': f"{sn}({best['股票']})",
            '最佳指标': best['指标名'],
            '命中': best['命中轮数'],
            '信号数': int(best['总信号数']),
            '有效率': f"{best['信号有效率']:.1%}",
            '得分': best['综合得分'],
        })
    if summary:
        sm = pd.DataFrame(summary)
        # Sort by Z score if available
        if st.session_state.odds_df is not None and not st.session_state.odds_df.empty:
            z_map = {}
            for _, r in st.session_state.odds_df.iterrows():
                raw = r['ts_code'].split('.')[0]
                z_map[raw] = r.get('median_z', -999)
            sm['_z'] = sm['股票'].apply(
                lambda x: z_map.get(x.split('(')[-1].rstrip(')'), -999))
            sm = sm.sort_values('_z', ascending=False).drop(columns=['_z'])
        else:
            sm = sm.sort_values('得分', ascending=False)
        st.dataframe(sm, use_container_width=True, hide_index=True)

    # Heatmap: stocks × top indicators
    st.subheader("个股 × 指标 热力图")
    top_inds = df['指标名'].unique()[:8]  # top 8 indicators
    stock_list = df['股票名'].unique()

    z = []; y = []
    for sn in stock_list:
        sdf = df[df['股票名'] == sn]
        row = []
        for ind in top_inds:
            match = sdf[sdf['指标名'] == ind]
            if not match.empty:
                row.append(match.iloc[0]['综合得分'])
            else:
                row.append(-1)
        z.append(row)
        y.append(sn)

    fig = go.Figure(data=go.Heatmap(
        z=z, x=list(top_inds), y=y,
        colorscale='RdYlGn', zmid=0,
        hovertemplate='%{y} × %{x}<br>得分=%{z:.3f}<extra></extra>',
    ))
    fig.update_layout(
        height=max(200, len(stock_list) * 40 + 100),
        margin=dict(l=10, r=10, t=10, b=80),
        yaxis=dict(autorange='reversed'),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Stock detail: select stock + indicator, show per-cycle K-line charts
    st.divider()
    st.subheader("个股行情信号详情")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        sel_stock = st.selectbox('选择股票', list(stock_list), key='stock_detail')
    with col_s2:
        sel_ind = st.selectbox('选择指标', list(top_inds), key='stock_detail_ind')

    if sel_stock and sel_ind and st.session_state.display_cycles is not None:
        cycles = st.session_state.display_cycles.to_dict('records')
        stock_code = df[df['股票名'] == sel_stock].iloc[0]['股票']
        stock_data = st.session_state.stocks_daily.get(stock_code)
        if stock_data is not None:
            sd = stock_data.copy()
            sd['trade_date'] = pd.to_datetime(sd['trade_date'])
            sd = sd.sort_values('trade_date')
            n_cycles = len(cycles)
            n_cols = 2
            n_rows = (n_cycles + 1) // n_cols
            for row_idx in range(n_rows):
                cols = st.columns(n_cols)
                for col_idx in range(n_cols):
                    ci = row_idx * n_cols + col_idx
                    if ci >= n_cycles: break
                    cycle = cycles[ci]
                    with cols[col_idx]:
                        _render_stock_cycle_detail(cycle, ci, sel_ind, sel_stock, sd, stock_code)


def _render_stock_cycle_detail(cycle, cycle_idx, indicator_name, stock_name, stock_data, stock_code=''):
    """Render K-line chart for a stock with indicator signals."""
    start = pd.Timestamp(cycle['start_date'])
    end = pd.Timestamp(cycle['end_date'])
    df = stock_data.copy()
    compute_start = start - pd.Timedelta(days=365 * 3)
    display_start = start - pd.Timedelta(days=60)
    view_end = end + pd.Timedelta(days=30)

    comp_mask = (df['trade_date'] >= compute_start) & (df['trade_date'] <= view_end)
    compute_seg = df[comp_mask].reset_index(drop=True)
    if compute_seg.empty:
        st.caption(f"{stock_name} 行情{cycle_idx+1}: 无数据")
        return

    cfg = INDICATOR_REGISTRY.get(indicator_name)
    if cfg is None: return
    params = cfg['params'][0]

    # Build stock-specific ref_highs — detect stock's own cycles (cached)
    from backtest import _build_stock_ref_highs
    full_df = stock_data.copy()
    full_df['trade_date'] = pd.to_datetime(full_df['trade_date'])
    full_df = full_df.set_index('trade_date').sort_index()
    stock_ref_highs_full = _build_stock_ref_highs(full_df, stock_code)

    # Map ref_highs to compute_seg dates (or use None to trigger rolling 250 default)
    stock_ref_highs = None
    if stock_ref_highs_full is not None:
        stock_ref_highs = []
        for d in compute_seg['trade_date']:
            if d in full_df.index:
                idx = full_df.index.get_loc(d)
                stock_ref_highs.append(stock_ref_highs_full[idx])
            else:
                stock_ref_highs.append(float(compute_seg[compute_seg['trade_date'] == d]['close'].iloc[0]))

    try:
        sig_kwargs = {'ref_highs': stock_ref_highs} if stock_ref_highs is not None else {}
        signals = cfg['func'](compute_seg, **params)
        signals = filter_signals(compute_seg, signals, **sig_kwargs)
    except Exception:
        return

    disp_mask = (compute_seg['trade_date'] >= display_start) & (compute_seg['trade_date'] <= view_end)
    segment = compute_seg[disp_mask].reset_index(drop=True)
    sig_seg = signals[disp_mask.values].reset_index(drop=True)
    if segment.empty: return

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.55, 0.45])
    fig.add_trace(go.Candlestick(
        x=segment['trade_date'], open=segment['open'], high=segment['high'],
        low=segment['low'], close=segment['close'], name=stock_name,
        increasing_line_color='red', decreasing_line_color='green',
        showlegend=False,
    ), row=1, col=1)

    # MA250
    full_close = compute_seg['close'].values.astype(float)
    ma250 = talib.SMA(full_close, MA_PERIOD)
    ma250_disp = ma250[disp_mask.values]
    fig.add_trace(go.Scatter(
        x=segment['trade_date'], y=ma250_disp,
        name='年线(250)', line=dict(color='orange', width=1),
        connectgaps=False,
    ), row=1, col=1)

    # Stock-specific decline line
    if stock_ref_highs is not None:
        decline_line = np.array(stock_ref_highs) * DECLINE_FACTOR
    else:
        rolling_max = pd.Series(compute_seg['close'].values.astype(float)).rolling(MA_PERIOD, min_periods=1).max().values
        decline_line = rolling_max * DECLINE_FACTOR
    decline_disp = decline_line[disp_mask.values]
    fig.add_trace(go.Scatter(
        x=segment['trade_date'], y=decline_disp,
        name='跌20%线', line=dict(color='gray', width=1, dash='dash'),
        connectgaps=False,
    ), row=1, col=1)

    if sig_seg.any():
        sd = segment['trade_date'][sig_seg.values]
        sc = segment['close'][sig_seg.values]
        fig.add_trace(go.Scatter(x=sd, y=sc, mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='blue'), name='信号'), row=1, col=1)

    all_lines = get_indicator_lines(compute_seg, indicator_name, params)
    disp_idx = disp_mask.values
    for line in all_lines:
        vals = line['values']
        if isinstance(vals, np.ndarray): vals = vals[disp_idx]
        if line.get('type') == 'bar':
            cb = ['red' if segment['close'].iloc[i] >= segment['open'].iloc[i] else 'green' for i in range(len(segment))]
            fig.add_trace(go.Bar(x=segment['trade_date'], y=vals, name=line['name'],
                marker_color=cb, opacity=0.5), row=2, col=1)
        else:
            fig.add_trace(go.Scatter(x=segment['trade_date'], y=vals, name=line['name'],
                line=dict(color=line.get('color','blue'), dash=line.get('dash')), opacity=0.7), row=2, col=1)

    fig.add_vline(x=start, line_dash='dash', line_color='purple', row=1, col=1)
    fig.update_xaxes(rangebreaks=_build_rangebreaks(segment['trade_date']), tickformat='%Y-%m-%d',
                     rangeslider=dict(visible=True, thickness=0.04), row=2, col=1)
    fig.update_xaxes(rangebreaks=_build_rangebreaks(segment['trade_date']), tickformat='%Y-%m-%d',
                     rangeslider_visible=False, row=1, col=1)
    fig.update_layout(title=f"{stock_name} 行情{cycle_idx+1}  {cycle['start_date'][:10]}→{cycle['end_date'][:10]}",
                       height=420,
                       showlegend=True,
                       legend=dict(orientation='h', y=-0.08, x=0.5, xanchor='center', font=dict(size=9)),
                       margin=dict(l=10, r=10, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)


def _build_ref_for_df(df, cycles_df=None):
    """Build reference high array for a DataFrame using cycle peaks.
    After the last cycle end, uses the max price since that end date."""
    if cycles_df is None:
        cycles_df = st.session_state.cycles_df
    if cycles_df is None or cycles_df.empty:
        return None
    cycle_peaks = sorted([(pd.Timestamp(r['end_date']), r['end_price']) for _, r in cycles_df.iterrows()])
    close = df['close'].values.astype(float)
    # 统一取日期序列：兼容 trade_date 为列或为 DatetimeIndex 两种情况
    dates = df['trade_date'] if 'trade_date' in df.columns else df.index
    last_peak_date = cycle_peaks[-1][0]
    rh = []
    for i, d in enumerate(dates):
        ref = None
        for pd_, pp in cycle_peaks:
            if pd_ <= d: ref = pp
            else: break
        # After last cycle: use max close since last cycle end
        if ref is None:
            ref = close[i]
        elif d > last_peak_date:
            since = close[(dates > last_peak_date) & (dates <= d)]
            since_max = since.max() if len(since) > 0 else ref
            ref = max(ref, since_max)
        rh.append(ref)
    return rh


def _detect_stock_peaks(stock_df):
    """Find cycle end prices for a stock using same algorithm as detect_cycles.py."""
    from scipy.signal import argrelextrema
    close = stock_df['close'].values.astype(float)
    smoothed = pd.Series(close).rolling(SMOOTH_WINDOW, min_periods=1).mean().values
    peaks = argrelextrema(smoothed, np.greater, order=ARGRELEXTREMA_ORDER)[0]
    troughs = argrelextrema(smoothed, np.less, order=ARGRELEXTREMA_ORDER)[0]
    cycle_highs = []
    for t_idx in troughs:
        later = peaks[peaks > t_idx]
        if len(later) == 0: continue
        p_idx = later[0]
        if int(p_idx - t_idx) < MIN_DURATION_DAYS: continue
        chg = (close[p_idx] - close[t_idx]) / close[t_idx] * 100
        if chg < MIN_AMPLITUDE_PCT: continue
        cycle_highs.append((stock_df['trade_date'].iloc[p_idx], close[p_idx]))
    return sorted(cycle_highs)


def render_sidebar():
    st.sidebar.title("券商板块信号回测")

    st.sidebar.subheader("数据状态")
    if st.session_state.data_loaded and st.session_state.index_daily is not None:
        total = len(INDEX_CONSTITUENTS)
        loaded = len(st.session_state.stocks_daily)
        st.sidebar.success(f"已加载 | 最新: {st.session_state.last_data_date} | 个股 {loaded}/{total}")
    elif st.session_state.data_error and not st.session_state.data_loaded:
        st.sidebar.error("数据加载失败")
    else:
        st.sidebar.warning("数据未加载")

    if st.session_state.data_error:
        st.sidebar.warning(f"⚠️ {st.session_state.data_error}")

    col_load, col_force = st.sidebar.columns([2, 2])
    with col_load:
        load_clicked = st.button("📥 加载数据", use_container_width=True)
    with col_force:
        force_refresh = st.checkbox("强制刷新", value=False,
                                    help="勾选后绕过本地缓存，从API重新拉取。不勾选时优先用本地CSV")

    if load_clicked:
        load_data(force=force_refresh)
        load_cycles()
        run_backtest_now()
        st.rerun()

    st.sidebar.divider()

    st.sidebar.subheader("行情窗口")
    if load_cycles():
        n = len(st.session_state.display_cycles)
        st.sidebar.success(f"已加载 {n} 轮行情")
    else:
        st.sidebar.warning("未找到 cycles.csv\n请先运行 detect_cycles.py")

    st.sidebar.divider()

    st.sidebar.subheader("回测参数")
    st.session_state.signal_window = st.sidebar.slider(
        "信号提前窗口(天)", 10, 60, st.session_state.signal_window, 5,
    )

    if st.sidebar.button("🔄 重新回测", use_container_width=True):
        if st.session_state.index_daily is not None and load_cycles():
            if run_backtest_now():
                st.sidebar.success("回测完成")
                st.rerun()
        else:
            st.sidebar.error("请先加载数据")

    st.sidebar.divider()
    st.sidebar.caption("⚠️ 历史回测结果不代表未来表现")


def render_cycle_overview():
    st.header("行情周期总览")

    if st.session_state.index_daily is None:
        st.warning("请先在侧边栏加载数据")
        return

    df = st.session_state.index_daily.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date')

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df['trade_date'],
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='399975.SZ',
        increasing_line_color='red',
        decreasing_line_color='green',
    ))

    if st.session_state.display_cycles is not None and not st.session_state.display_cycles.empty:
        colors = ['rgba(255,0,0,0.08)', 'rgba(0,100,255,0.08)']
        for j, (_, row) in enumerate(st.session_state.display_cycles.iterrows()):
            color = colors[j % 2]
            fig.add_vrect(
                x0=pd.Timestamp(row['start_date']),
                x1=pd.Timestamp(row['end_date']),
                fillcolor=color,
                opacity=0.4,
                layer="below",
                line_width=0,
                annotation_text=f"行情{j+1}",
                annotation_position="top left",
            )

    fig.update_layout(
        title='中证证券公司指数 (399975.SZ) — 历史行情周期',
        xaxis_title='日期',
        yaxis_title='价格',
        height=600,
        xaxis_rangeslider=dict(visible=True, thickness=0.04),
    )
    fig.update_xaxes(
        rangebreaks=_build_rangebreaks(df['trade_date']),
        tickformat='%Y-%m-%d',
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.session_state.display_cycles is not None and not st.session_state.display_cycles.empty:
        st.subheader("行情周期明细")
        display_df = st.session_state.display_cycles.copy()
        display_df['start_date'] = pd.to_datetime(display_df['start_date']).dt.date
        display_df['end_date'] = pd.to_datetime(display_df['end_date']).dt.date
        display_df.index = range(1, len(display_df) + 1)
        display_df.index.name = '#'
        st.dataframe(
            display_df[['start_date', 'end_date', 'duration_days', 'start_price', 'end_price', 'change_pct']],
            use_container_width=True,
        )


def _render_cycle_detail(cycle, cycle_idx, indicator_name, freq='日线'):
    """Render a small chart for one cycle showing indicator signals."""
    start = pd.Timestamp(cycle['start_date'])
    end = pd.Timestamp(cycle['end_date'])

    # Large window for indicator computation (enough warm-up)
    if freq == '周线' and st.session_state.index_weekly is not None:
        df = st.session_state.index_weekly.copy()
        compute_start = start - pd.Timedelta(days=365 * 3)  # 3 years warm-up for weekly MACD
        display_start = start - pd.Timedelta(weeks=26)
    else:
        df = st.session_state.index_daily.copy()
        compute_start = start - pd.Timedelta(days=365 * 3)  # 3 years for 250MA warm-up
        display_start = start - pd.Timedelta(days=60)

    view_end = end + pd.Timedelta(days=30)

    df['trade_date'] = pd.to_datetime(df['trade_date'])

    # Compute on full warm-up window
    comp_mask = (df['trade_date'] >= compute_start) & (df['trade_date'] <= view_end)
    compute_seg = df[comp_mask].reset_index(drop=True)

    if compute_seg.empty:
        st.caption(f"行情{cycle_idx+1}: 无数据")
        return

    cfg = INDICATOR_REGISTRY.get(indicator_name)
    if cfg is None:
        return

    params = cfg['params'][0]
    try:
        signals = cfg['func'](compute_seg, **params)
        signals = filter_signals(compute_seg, signals)
    except Exception:
        return

    # Display only narrow window
    disp_mask = (compute_seg['trade_date'] >= display_start) & (compute_seg['trade_date'] <= view_end)
    segment = compute_seg[disp_mask].reset_index(drop=True)
    sig_segment = signals[disp_mask.values].reset_index(drop=True)

    if segment.empty:
        st.caption(f"行情{cycle_idx+1}: 无数据")
        return

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.45],
    )

    fig.add_trace(go.Candlestick(
        x=segment['trade_date'],
        open=segment['open'], high=segment['high'],
        low=segment['low'], close=segment['close'],
        name='价格',
        increasing_line_color='red',
        decreasing_line_color='green',
        showlegend=False,
    ), row=1, col=1)

    # Add MA250 and decline threshold lines for context
    if freq == '日线':
        full_close = compute_seg['close'].values.astype(float)
        ma250 = talib.SMA(full_close, MA_PERIOD)
        ma250_disp = ma250[disp_mask.values]
        fig.add_trace(go.Scatter(
            x=segment['trade_date'], y=ma250_disp,
            name='年线(250)', line=dict(color='orange', width=1),
            connectgaps=False,
        ), row=1, col=1)

        # Build ref_highs from cycle end prices (same as backtest engine)
        cycles_all = pd.read_csv(os.path.join(OUTPUT_DIR, 'cycles.csv'))
        cycle_peaks = sorted([(pd.Timestamp(c['end_date']), c['end_price']) for _, c in cycles_all.iterrows()])
        rh = []
        close_vals = compute_seg['close'].values.astype(float)
        for i, d in enumerate(compute_seg['trade_date']):
            ref = None
            for pd_, pp in cycle_peaks:
                if pd_ <= d: ref = pp
                else: break
            rh.append(ref if ref is not None else close_vals[i])
        decline_line = np.array(rh) * DECLINE_FACTOR
        decline_disp = decline_line[disp_mask.values]
        fig.add_trace(go.Scatter(
            x=segment['trade_date'], y=decline_disp,
            name='跌20%线', line=dict(color='gray', width=1, dash='dash'),
            connectgaps=False,
        ), row=1, col=1)

    if sig_segment is not None and sig_segment.any():
        sig_dates = segment['trade_date'][sig_segment.values]
        sig_closes = segment['close'][sig_segment.values]
        fig.add_trace(go.Scatter(
            x=sig_dates, y=sig_closes,
            mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='blue'),
            name='信号',
        ), row=1, col=1)

    # Compute indicator lines on full warm-up window, then slice to display
    all_lines = get_indicator_lines(compute_seg, indicator_name, params)
    disp_idx = disp_mask.values  # already relative to compute_seg

    for line in all_lines:
        values = line['values']
        if isinstance(values, np.ndarray):
            values = values[disp_idx]
        if line.get('type') == 'bar':
            if line.get('colors'):
                colors_bar = [line['colors'][i] for i in range(len(line['colors'])) if disp_idx[i]]
            else:
                colors_bar = ['red' if segment['close'].iloc[i] >= segment['open'].iloc[i] else 'green'
                              for i in range(len(segment))]
            fig.add_trace(go.Bar(
                x=segment['trade_date'], y=values,
                name=line['name'], marker_color=colors_bar,
                opacity=0.7,
            ), row=2, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=segment['trade_date'], y=values,
                name=line['name'],
                line=dict(color=line.get('color', 'blue'),
                         dash=line.get('dash')),
                opacity=0.7,
            ), row=2, col=1)

    fig.add_vline(x=start, line_dash='dash', line_color='purple', row=1, col=1)

    fig.update_layout(
        title=f"行情{cycle_idx+1}: {start.date()} → {end.date()} (+{cycle['change_pct']}%)",
        height=420,
        showlegend=True,
        legend=dict(orientation='h', y=-0.08, x=0.5, xanchor='center', font=dict(size=9)),
        margin=dict(l=10, r=10, t=40, b=40),
    )
    fig.update_xaxes(
        rangebreaks=_build_rangebreaks(segment['trade_date']) if freq != '周线' else None,
        tickformat='%Y-%m-%d',
        rangeslider_visible=False,
        row=1, col=1,
    )
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.04, borderwidth=0),
        row=2, col=1,
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_heatmap(results_df, cycles_df):
    """Render indicator x cycle heatmap using Plotly."""
    import ast
    from indicators import INDICATOR_REGISTRY, filter_signals
    from backtest import judge_signal

    if st.session_state.index_daily is None:
        st.info("请先在侧边栏加载数据")
        return

    cycles = cycles_df.to_dict('records')
    heat_df = results_df.copy()
    heat_df = heat_df.sort_values('综合得分', ascending=False)

    z_data = []
    z_text = []
    y_labels = []

    for _, row in heat_df.iterrows():
        is_resonance = (row['类别'] == '共振')
        y_labels.append(f"{row['指标名']}({row['周期']})")
        z_row = []
        t_row = []

        if is_resonance:
            # For resonance rows, re-compute the combined signal
            rule_name = row['指标名']  # e.g. "周线:布林带收窄 + 日线:CCI反转"
            parts = rule_name.split(' + ')
            if len(parts) != 2:
                for _ in cycles:
                    z_row.append(0); t_row.append('parse error')
                z_data.append(z_row); z_text.append(t_row)
                continue
            w_name = parts[0].replace('周线:', '')
            d_name = parts[1].replace('日线:', '')

            w_cfg = INDICATOR_REGISTRY.get(w_name)
            d_cfg = INDICATOR_REGISTRY.get(d_name)
            if w_cfg is None or d_cfg is None:
                for _ in cycles:
                    z_row.append(0); t_row.append('no cfg')
                z_data.append(z_row); z_text.append(t_row)
                continue

            try:
                w_df = st.session_state.index_weekly.copy()
                d_df = st.session_state.index_daily.copy()
                w_df['trade_date'] = pd.to_datetime(w_df['trade_date'])
                d_df['trade_date'] = pd.to_datetime(d_df['trade_date'])
                w_df = w_df.set_index('trade_date').sort_index()
                d_df = d_df.set_index('trade_date').sort_index()

                # Build ref_highs consistent with backtest
                cycles_all = pd.read_csv(os.path.join(OUTPUT_DIR, 'cycles.csv'))
                cycle_peaks = sorted([(pd.Timestamp(c['end_date']), c['end_price']) for _, c in cycles_all.iterrows()])
                def _build_ref_highs(idx):
                    ref = []
                    for d in idx:
                        rh = None
                        for peak_date, peak_price in cycle_peaks:
                            if peak_date <= d: rh = peak_price
                            else: break
                        ref.append(rh if rh is not None else 0)
                    return ref
                w_ref = _build_ref_highs(w_df.index)
                d_ref = _build_ref_highs(d_df.index)

                w_raw = w_cfg['func'](w_df.reset_index(), **w_cfg['params'][0])
                w_raw.index = w_df.index
                w_sig = filter_signals(w_df.reset_index(), w_raw, ref_highs=w_ref)
                w_sig.index = w_df.index

                d_raw = d_cfg['func'](d_df.reset_index(), **d_cfg['params'][0])
                d_raw.index = d_df.index
                d_sig = filter_signals(d_df.reset_index(), d_raw, ref_highs=d_ref)
                d_sig.index = d_df.index

                # Build resonance signal
                w_signal_dates = w_sig[w_sig].index
                resonance_sig = pd.Series(False, index=d_df.index)
                for ws_date in w_signal_dates:
                    win_end = ws_date + pd.Timedelta(days=RESONANCE_WINDOW_DAYS)
                    win_mask = (d_df.index >= ws_date) & (d_df.index <= win_end)
                    if d_sig[win_mask].any():
                        resonance_sig[d_sig[win_mask].index[0]] = True

                for cycle in cycles:
                    start = pd.Timestamp(cycle['start_date'])
                    end = pd.Timestamp(cycle['end_date'])
                    j = judge_signal(resonance_sig, start, end, signal_window=st.session_state.signal_window)
                    if j['hit']:
                        z_row.append(3 if j['days_before'] <= HEATMAP_HIT_FAST else 2)
                        t_row.append(f"共振命中 提前{j['days_before']}天")
                    elif j['late']:
                        sd = j['signal_date']
                        days_after = (sd - start).days
                        z_row.append(1 if days_after <= HEATMAP_LATE_FAST else 0.5)
                        t_row.append(f"晚了{days_after}天")
                    else:
                        z_row.append(0); t_row.append('错过')

                z_data.append(z_row)
                z_text.append(t_row)
            except Exception as e:
                for _ in cycles:
                    z_row.append(0); t_row.append(f'err:{str(e)[:20]}')
                z_data.append(z_row); z_text.append(t_row)
                continue

        else:
            freq = str(row['周期'])
            if '周' in freq and st.session_state.index_weekly is not None:
                df = st.session_state.index_weekly.copy()
            else:
                df = st.session_state.index_daily.copy()
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.set_index('trade_date').sort_index()

            cfg = INDICATOR_REGISTRY.get(row['指标名'])
            if cfg is None:
                for _ in cycles:
                    z_row.append(0); t_row.append('no cfg')
                z_data.append(z_row); z_text.append(t_row)
                continue

            try:
                params_str = row['参数']
                if isinstance(params_str, str):
                    params = ast.literal_eval(params_str)
                else:
                    params = cfg['params'][0]
            except Exception:
                params = cfg['params'][0]

            # Build ref_highs consistent with backtest engine
            cycles_all = pd.read_csv(os.path.join(OUTPUT_DIR, 'cycles.csv'))
            cycle_peaks = sorted([(pd.Timestamp(c['end_date']), c['end_price']) for _, c in cycles_all.iterrows()])
            ref_highs = []
            for d in df.index:
                rh = None
                for peak_date, peak_price in cycle_peaks:
                    if peak_date <= d:
                        rh = peak_price
                    else:
                        break
                ref_highs.append(rh if rh is not None else df.loc[d, 'close'])

            try:
                sig_raw = cfg['func'](df.reset_index(), **params)
                sig_raw.index = df.index
                sig = filter_signals(df.reset_index(), sig_raw, ref_highs=ref_highs)
                sig.index = df.index
            except Exception:
                for _ in cycles:
                    z_row.append(0); t_row.append('error')
                z_data.append(z_row); z_text.append(t_row)
                continue

            for cycle in cycles:
                start = pd.Timestamp(cycle['start_date'])
                end = pd.Timestamp(cycle['end_date'])
                j = judge_signal(sig, start, end, signal_window=st.session_state.signal_window)
                if j['hit']:
                    days = j['days_before']
                    z_row.append(3 if days <= HEATMAP_HIT_FAST else 2)
                    t_row.append(f"命中 提前{days}天")
                elif j['late']:
                    sd = j['signal_date']
                    days_after = (sd - start).days
                    z_row.append(1 if days_after <= HEATMAP_LATE_FAST else 0.5)
                    t_row.append(f"晚了{days_after}天")
                else:
                    z_row.append(0); t_row.append('错过')

            z_data.append(z_row)
            z_text.append(t_row)

    if not z_data:
        st.info("无数据可显示")
        return

    x_labels = [f"{c['start_date'][:10]}\n+{c['change_pct']}%" for c in cycles]

    colorscale = [
        [0.0, '#9e9e9e'],    # 0: 错过 (gray)
        [0.2, '#e6a817'],    # 0.5: 晚>5天 (dark yellow)
        [0.4, '#fff176'],    # 1: 晚≤5天 (light yellow)
        [0.7, '#a5d6a7'],    # 2: 命中 >15天前 (light green)
        [1.0, '#2e7d32'],    # 3: 命中 ≤15天前 (dark green)
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_data, text=z_text,
        x=x_labels, y=y_labels,
        colorscale=colorscale,
        zmin=0, zmax=3,
        showscale=False,
        hovertemplate='%{y}<br>%{x}<br>%{text}<extra></extra>',
    ))

    fig.update_layout(
        height=max(300, len(y_labels) * 22 + 100),
        margin=dict(l=10, r=10, t=20, b=60),
        xaxis=dict(type='category', side='bottom', tickangle=0),
        yaxis=dict(autorange='reversed'),
    )

    st.plotly_chart(fig, use_container_width=True)


def _build_rangebreaks(trade_dates):
    """Build rangebreaks list for all non-trading days (weekends + holidays)."""
    from datetime import timedelta
    trade_set = set(pd.Timestamp(d).date() for d in trade_dates)
    all_days = pd.date_range(min(trade_dates), max(trade_dates))
    breaks = []
    gap_start = None
    for d in all_days:
        if d.date() not in trade_set:
            if gap_start is None:
                gap_start = d
        else:
            if gap_start is not None:
                breaks.append(dict(bounds=[gap_start.strftime('%Y-%m-%d'), d.strftime('%Y-%m-%d')]))
                gap_start = None
    if gap_start is not None:
        breaks.append(dict(bounds=[gap_start.strftime('%Y-%m-%d'), (all_days[-1] + timedelta(days=1)).strftime('%Y-%m-%d')]))
    return breaks


def render_indicator_rankings():
    st.header("指标回测排行榜")

    if st.session_state.backtest_results is None:
        path = os.path.join(OUTPUT_DIR, 'backtest_results.csv')
        if os.path.exists(path):
            st.session_state.backtest_results = pd.read_csv(path)
        else:
            st.info("请先在侧边栏点击「重新回测」")
            return

    df = st.session_state.backtest_results

    col1, col2 = st.columns(2)
    with col1:
        categories = ['全部'] + sorted(df['类别'].dropna().unique().tolist())
        selected_cat = st.selectbox('指标类别', categories)
    with col2:
        freqs = ['全部'] + sorted(df['周期'].dropna().unique().tolist())
        selected_freq = st.selectbox('周期', freqs)

    filtered = df.copy()
    if selected_cat != '全部':
        filtered = filtered[filtered['类别'] == selected_cat]
    if selected_freq != '全部':
        filtered = filtered[filtered['周期'] == selected_freq]

    st.dataframe(
        filtered,
        use_container_width=True,
        column_config={
            '综合得分': st.column_config.ProgressColumn(
                '综合得分', format='%.3f', min_value=0, max_value=SCORE_MAX,
            ),
        },
        hide_index=True,
    )

    st.divider()
    st.subheader("指标详情 (选择指标查看各行情信号)")

    indicator_names = filtered['指标名'].unique()
    col_i1, col_i2 = st.columns([2, 1])
    with col_i1:
        selected_indicator = st.selectbox('选择指标', indicator_names)
    with col_i2:
        detail_freq = st.selectbox('周期', ['日线', '周线'], key='detail_freq')

    if selected_indicator:
        if st.session_state.display_cycles is not None and st.session_state.index_daily is not None:
            cycles = st.session_state.display_cycles.to_dict('records')
            n_cycles = len(cycles)
            n_cols = min(2, n_cycles)
            n_rows = (n_cycles + n_cols - 1) // n_cols

            for row_idx in range(n_rows):
                cols = st.columns(n_cols)
                for col_idx in range(n_cols):
                    cycle_idx = row_idx * n_cols + col_idx
                    if cycle_idx >= n_cycles:
                        break
                    cycle = cycles[cycle_idx]
                    with cols[col_idx]:
                        _render_cycle_detail(cycle, cycle_idx, selected_indicator, detail_freq)

    st.divider()
    st.subheader("指标 × 行情 热力图")

    if st.session_state.cycles_df is not None and st.session_state.backtest_results is not None:
        _render_heatmap(st.session_state.backtest_results, st.session_state.display_cycles)


def render_live_tracking():
    st.header("行情跟踪")

    if not st.session_state.stocks_daily or st.session_state.index_daily is None:
        st.warning("请先在侧边栏加载数据")
        return

    # Top indicators from backtest
    top_inds = ['CCI脱离超卖', 'OBV底背离', 'MACD柱线缩短']

    # ---- Usage guide ----
    with st.expander("📖 买入信号使用指南", expanded=True):
        st.markdown("""
        **主买入信号：CCI 脱离超卖**（蓝三角出现在 CCI 栏）
        - CCI 从 -100 以下回升 → 下跌动量衰减
        - 必须出现在 **灰色跌20%线之下** → 市场处于超跌状态
        - 一个蓝三角 = 可以买入
        
        **确认信号：OBV 底背离**（蓝三角出现在 OBV 栏）
        - 价格下跌但 OBV 在涨 → 资金在悄悄进场
        - CCI 响了 + OBV 也响了 → 加仓
        
        **辅助参考：MACD 柱线缩短**（蓝三角出现在 MACD 栏）
        - 绿柱连续收窄 → 空头力竭
        - 有最好，没有也不影响决策
        """)

    # ---- Index chart ----
    st.subheader("中证证券公司指数 (399975.SZ)")
    _render_live_chart(st.session_state.index_daily, '399975.SZ', top_inds, is_index=True)

    # ---- Current signal summary ----
    st.subheader("当前信号汇总")
    _render_signal_summary(top_inds)

    # ---- Individual stock charts ----
    st.subheader("个股详情")
    stock_codes = list(st.session_state.stocks_daily.keys())
    stock_names = {code: INDEX_CONSTITUENTS.get(code, code) for code in stock_codes}
    z_map = {}
    if st.session_state.odds_df is not None and not st.session_state.odds_df.empty:
        for _, r in st.session_state.odds_df.iterrows():
            raw = r['ts_code'].split('.')[0]
            z_map[raw] = r.get('median_z', 0)
        stock_codes.sort(key=lambda c: z_map.get(c, -999), reverse=True)
    sel_live = st.selectbox('选择个股', stock_codes,
        format_func=lambda c: f"{c} {stock_names[c]}" + (f"  [Z={z_map.get(c, 0):+.2f}]" if z_map else ""),
        key='live_stock')
    if sel_live and sel_live in st.session_state.stocks_daily:
        _render_live_chart(st.session_state.stocks_daily[sel_live], sel_live, top_inds, is_index=False)


def _render_live_chart(stock_df, label, indicators, is_index=False):
    """Render recent K-line chart with FILTERED indicator signals."""
    df = stock_df.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date')

    # Compute signals on full data (for filter context), display last 1 year
    display_start = df['trade_date'].max() - pd.Timedelta(days=365)
    compute_seg = df.copy()
    disp_mask = df['trade_date'] >= display_start
    seg = df[disp_mask].reset_index(drop=True)

    if seg.empty: return

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                         vertical_spacing=0.02, row_heights=[0.45, 0.20, 0.20, 0.15])

    fig.add_trace(go.Candlestick(
        x=seg['trade_date'], open=seg['open'], high=seg['high'],
        low=seg['low'], close=seg['close'], name=label,
        increasing_line_color='red', decreasing_line_color='green',
        showlegend=False,
    ), row=1, col=1)

    # Add 250MA and decline threshold to K-line
    full_close = compute_seg['close'].values.astype(float)
    ma250 = talib.SMA(full_close, MA_PERIOD)
    # For index: use cycle peaks; for stocks: detect own cycle peaks
    if is_index:
        ref_highs = _build_ref_for_df(compute_seg)
    else:
        # 与回测口径一致：用 _build_stock_ref_highs（含最后周期后 rolling_max 兜底）
        from backtest import _build_stock_ref_highs
        stock_df_idx = compute_seg.set_index('trade_date').sort_index()
        ref_highs = _build_stock_ref_highs(stock_df_idx, label)
    decline_disp = (np.array(ref_highs) * DECLINE_FACTOR)[disp_mask.values] if ref_highs else None

    ma250_disp = ma250[disp_mask.values]

    fig.add_trace(go.Scatter(x=seg['trade_date'], y=ma250_disp, name='年线(250)',
        line=dict(color='orange', width=1), connectgaps=False), row=1, col=1)
    if decline_disp is not None:
        fig.add_trace(go.Scatter(x=seg['trade_date'], y=decline_disp, name='跌20%线',
            line=dict(color='gray', width=1, dash='dash'), connectgaps=False), row=1, col=1)

    # Mark reference high as horizontal line
    if ref_highs is not None and len(ref_highs) > 0:
        last_rh = ref_highs[-1]  # latest reference high (includes post-cycle peaks)
        fig.add_hline(y=last_rh, line_dash='dot', line_color='purple',
                      annotation_text=f'参考高点 {last_rh:.0f}', row=1, col=1)

    # CCI in row 2
    if 'CCI脱离超卖' in indicators:
        cfg = INDICATOR_REGISTRY.get('CCI脱离超卖')
        if cfg:
            # Compute on full data, filter, then slice to display
            sig_raw = cfg['func'](compute_seg, **cfg['params'][0])
            sig_filt = filter_signals(compute_seg, sig_raw, ref_highs=ref_highs)
            sig_disp = sig_filt[disp_mask.values].reset_index(drop=True)

            high_v = seg['high'].values.astype(float)
            low_v = seg['low'].values.astype(float)
            close_v = seg['close'].values.astype(float)
            cci_vals = talib.CCI(high_v, low_v, close_v, 20)

            fig.add_trace(go.Scatter(x=seg['trade_date'], y=cci_vals, name='CCI',
                line=dict(color='blue')), row=2, col=1)
            fig.add_hline(y=-100, line_dash='dash', line_color='gray', row=2, col=1)
            fig.add_hline(y=100, line_dash='dash', line_color='gray', row=2, col=1)
            if sig_disp.any():
                sd = seg['trade_date'][sig_disp.values]
                sv = cci_vals[sig_disp.values]
                fig.add_trace(go.Scatter(x=sd, y=sv, mode='markers',
                    marker=dict(symbol='triangle-up', size=10, color='blue'), name='CCI信号',
                    hovertemplate='日期: %{x|%Y-%m-%d}<br>CCI: %{y:.1f}<extra></extra>'), row=2, col=1)

    # MACD柱线缩短 in row 3 (visual confirmation, #3 by score)
    if 'MACD柱线缩短' in indicators:
        cfg = INDICATOR_REGISTRY.get('MACD柱线缩短')
        if cfg:
            sig_raw = cfg['func'](compute_seg, **cfg['params'][0])
            sig_filt = filter_signals(compute_seg, sig_raw, ref_highs=ref_highs)
            sig_disp = sig_filt[disp_mask.values].reset_index(drop=True)

            close_v = seg['close'].values.astype(float)
            dif, dea, hist = talib.MACD(close_v)
            # Histogram bars with colors
            colors = []
            for i in range(len(hist)):
                if np.isnan(hist[i]): colors.append('gray')
                elif hist[i] >= 0: colors.append('#ef5350' if (i>0 and hist[i]>hist[i-1]) else '#ef9a9a')
                else: colors.append('#2e7d32' if (i>0 and hist[i]<hist[i-1]) else '#81c784')
            fig.add_trace(go.Bar(x=seg['trade_date'], y=hist, name='MACD柱',
                marker_color=colors, opacity=0.7), row=3, col=1)
            fig.add_trace(go.Scatter(x=seg['trade_date'], y=dif, name='DIF',
                line=dict(color='blue')), row=3, col=1)
            fig.add_trace(go.Scatter(x=seg['trade_date'], y=dea, name='DEA',
                line=dict(color='orange')), row=3, col=1)
            if sig_disp.any():
                sd = seg['trade_date'][sig_disp.values]
                sh = hist[sig_disp.values]
                fig.add_trace(go.Scatter(x=sd, y=sh, mode='markers',
                    marker=dict(symbol='triangle-up', size=10, color='blue'), name='MACD信号',
                    hovertemplate='日期: %{x|%Y-%m-%d}<br>HIST: %{y:.4f}<extra></extra>'), row=3, col=1)

    # OBV in row 4
    obv_v = talib.OBV(seg['close'].values.astype(float), seg['vol'].values.astype(float))
    fig.add_trace(go.Scatter(x=seg['trade_date'], y=obv_v, name='OBV',
        line=dict(color='purple', width=1)), row=4, col=1)

    # OBV signal markers
    obv_cfg = INDICATOR_REGISTRY.get('OBV底背离')
    if obv_cfg:
        obv_raw = obv_cfg['func'](compute_seg, **obv_cfg['params'][0])
        obv_filt = filter_signals(compute_seg, obv_raw, ref_highs=ref_highs)
        obv_disp = obv_filt[disp_mask.values].reset_index(drop=True)
        if obv_disp.any():
            sd_dates = seg['trade_date'][obv_disp.values]
            sd_vals = obv_v[obv_disp.values]
            fig.add_trace(go.Scatter(x=sd_dates, y=sd_vals, mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='blue'), name='OBV信号',
                hovertemplate='日期: %{x|%Y-%m-%d}<br>OBV: %{y:.0f}<extra></extra>'), row=4, col=1)

    fig.update_xaxes(rangebreaks=_build_rangebreaks(seg['trade_date']), tickformat='%Y-%m',
                     rangeslider=dict(visible=True, thickness=0.03), row=4, col=1)
    fig.update_xaxes(rangebreaks=_build_rangebreaks(seg['trade_date']), tickformat='%Y-%m',
                     rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(rangebreaks=_build_rangebreaks(seg['trade_date']), tickformat='%Y-%m',
                     rangeslider_visible=False, row=2, col=1)
    fig.update_layout(
        title=f"{'板块指数' if is_index else label} — 近1年走势",
        height=900,
        showlegend=True,
        legend=dict(orientation='h', y=-0.02, x=0.5, xanchor='center', font=dict(size=10)),
        margin=dict(l=10, r=10, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_signal_summary(top_inds):
    """Show current signal status for all stocks, sorted by signal strength then Z score."""
    z_map = {}
    if st.session_state.odds_df is not None and not st.session_state.odds_df.empty:
        for _, r in st.session_state.odds_df.iterrows():
            raw = r['ts_code'].split('.')[0]
            z = r.get('median_z', -999)
            if pd.isna(z):
                z = -999
            z_map[raw] = float(z)

    rows = []
    from backtest import _build_stock_ref_highs
    for code, df in st.session_state.stocks_daily.items():
        name = INDEX_CONSTITUENTS.get(code, code)
        z_val = z_map.get(code, -999)
        df = df.copy()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date')
        recent_cutoff = df['trade_date'].max() - pd.Timedelta(days=90)

        # 与回测口径一致：用个股自身周期高点构建 ref_highs，避免信号汇总与回测过滤不一致
        df_indexed = df.set_index('trade_date')
        try:
            stock_ref_highs = _build_stock_ref_highs(df_indexed, code)
        except Exception:
            stock_ref_highs = None

        sig_count = 0
        sig_names = []
        sig_dates = []
        for ind_name in top_inds:
            cfg = INDICATOR_REGISTRY.get(ind_name)
            if cfg is None:
                continue
            try:
                sig = cfg['func'](df, **cfg['params'][0])
                sig = filter_signals(df, sig, ref_highs=stock_ref_highs)
            except Exception:
                continue
            recent = (df['trade_date'] >= recent_cutoff).values
            recent_sig = sig.values & recent
            if recent_sig.any():
                sig_count += 1
                sig_names.append(ind_name)
                last_idx = np.where(recent_sig)[0][-1]
                sig_dates.append(df['trade_date'].iloc[last_idx].date())

        if sig_count >= 3:
            stars = '★★★'
        elif sig_count == 2:
            stars = '★★☆'
        elif sig_count == 1:
            stars = '★☆☆'
        else:
            stars = '——'

        rows.append({
            '股票': name,
            'Z评分': f"{z_val:+.2f}" if z_val > -999 else '—',
            '信号': stars,
            '触发指标': ' + '.join(sig_names) if sig_names else '—',
            '最近信号日': max(sig_dates).strftime('%Y-%m-%d') if sig_dates else '—',
            '_z': z_val,
            '_cnt': sig_count,
        })

    if rows:
        rows.sort(key=lambda r: (r['_cnt'], r['_z']), reverse=True)
        df_rows = pd.DataFrame(rows).drop(columns=['_z', '_cnt'])
        st.dataframe(df_rows, use_container_width=True, hide_index=True)
    else:
        st.info("暂无信号数据")


def render_odds_tab():
    """Tab 5: 优选赔率 — Stock odds screening and ranking."""
    st.header("优选赔率")

    if st.session_state.odds_df is None or st.session_state.odds_df.empty:
        st.warning("未找到赔率数据。请先在终端运行: python screener.py --init")
        return

    df = st.session_state.odds_df.copy()
    meta = st.session_state.odds_meta

    # ---- Top status bar ----
    col_s1, col_s2, col_s3, col_s4 = st.columns([2, 2, 2, 2])
    with col_s1:
        st.caption(f"更新至: {st.session_state.odds_data_date}")
    with col_s2:
        st.caption(f"覆盖: {len(df)} 只成分股")
    with col_s3:
        data_m = int(meta.get('M', 7))
        if st.button("刷新数据", key='odds_refresh'):
            st.cache_data.clear()
            load_odds()
            st.rerun()

    # ---- Confidence threshold control ----
    max_cycles = int(df['cycle_count'].max())
    user_m = st.slider(
        "置信度阈值（≥M轮=可信）",
        min_value=1, max_value=max_cycles,
        value=int(data_m), key='odds_m_slider',
        help="低于此值的个股标记为'参考'或'有限'。默认值=M=49只成分股参与轮数的中位数。"
    )
    st.caption(f"当前: 可信={len(df[df['cycle_count'] >= user_m])}只, 参考={len(df[(df['cycle_count'] >= 2) & (df['cycle_count'] < user_m)])}只, 有限={len(df[df['cycle_count'] < 2])}只")

    # ---- Signal filter ----
    show_signals_only = st.checkbox("仅显示当前有信号", value=False, key='odds_signal_filter',
                                     help="勾选后只展示当前有买入信号的股票")

    # ---- Compute signals ----
    signal_df = compute_odds_signals()
    signal_map = {}
    signal_count_map = {}
    if not signal_df.empty:
        signal_map = dict(zip(signal_df['ts_code'], signal_df['signal']))
        signal_count_map = dict(zip(signal_df['ts_code'], signal_df['signal_count']))

    # ---- Main ranking table ----
    st.subheader("券商个股赔率排名")

    with st.expander("ℹ️ Z 评分说明", expanded=False):
        st.markdown(
            "**Z 评分** 衡量个股在一轮板块牛市里相对同行的表现强度：\n\n"
            "> Z = (个股周期涨幅 − 同周期所有个股涨幅均值) / 标准差\n\n"
            "每轮周期内部独立标准化，消除「大牛市+50%」和「小行情+15%」幅度差异导致的不可比问题。"
            "Z = +1.5 表示该股在该轮比同行平均高 1.5 个标准差，是显著的大胜。\n\n"
            "| 列 | 含义 |\n"
            "|---|---|\n"
            "| **中位数Z** | 跨周期 Z 的中位数，**主排序依据**。正值=常态跑赢同行，负值=常态落后 |\n"
            "| **Z最高值** | 历史单轮最佳 Z，反映爆发力上限 |\n"
            "| **Z正值率** | Z>0 的周期占比，反映稳定性（100%=每轮都跑赢同行中位数）|\n"
            "| **中位涨幅** | 各周期原始涨幅的中位数，直觉参考值 |\n"
            "| **最大涨幅** | 历史单轮最大原始涨幅 |\n"
            "| **跑赢概率** | 涨幅超过板块指数的周期占比 |\n"
            "| **轮数** | 参与的周期数（越多越可信）|\n"
            "| **置信度** | ≥M 轮=可信，2~M-1 轮=参考，<2 轮=有限 |\n"
            "| **信号** | ★数 = CCI/OBV/MACD 当前触发买入信号的数量（近90天内）|\n"
        )

    st.caption("按中位数Z降序排列。★数 = 当前 CCI/OBV/MACD 中有几个触发买入信号（近90天内），★越多信号质量越高。 —— = 当前无信号")

    display = df[['ts_code', 'name', 'median_z', 'max_z', 'positive_z_rate',
                   'median_return', 'max_return', 'beat_index_rate',
                   'confidence', 'cycle_count']].copy()

    display['#'] = range(1, len(display) + 1)
    display['中位数Z'] = display['median_z'].apply(lambda x: f'{x:+.2f}')
    display['Z最高值'] = display['max_z'].apply(lambda x: f'{x:+.2f}')
    display['Z正值率'] = display['positive_z_rate'].apply(lambda x: f'{x:.0%}')
    display['中位涨幅'] = display['median_return'].apply(lambda x: f'{x:+.1%}')
    display['最大涨幅'] = display['max_return'].apply(lambda x: f'{x:+.1%}')
    display['跑赢概率'] = display['beat_index_rate'].apply(lambda x: f'{x:.0%}')
    display['轮数'] = display['cycle_count']
    display['置信度'] = display['cycle_count'].apply(
        lambda n: '✓ 可信' if n >= user_m else ('⚠ 参考' if n >= 2 else '▷ 有限'))

    def fmt_signal(code):
        cnt = int(signal_count_map.get(code, -1))
        if cnt == 3:
            return '★★★'
        elif cnt == 2:
            return '★★☆'
        elif cnt == 1:
            return '★☆☆'
        elif cnt == 0:
            return '——'
        return '—'
    display['信号'] = display['ts_code'].apply(fmt_signal)

    display = display.sort_values('median_z', ascending=False).reset_index(drop=True)

    if show_signals_only:
        display = display[display['信号'] != '——']

    show_cols = ['#', 'name', '中位数Z', 'Z最高值', 'Z正值率', '中位涨幅',
                 '最大涨幅', '跑赢概率', '置信度', '轮数', '信号']

    if show_signals_only:
        st.caption(f"筛选结果: {len(display)} 只有信号的股票")

    st.dataframe(
        display[show_cols].rename(columns={'name': '名称'}),
        use_container_width=True,
        hide_index=True,
        height=min(35 * len(display) + 38, 600),
    )

    # ---- Auxiliary charts ----
    st.divider()
    col_ch1, col_ch2 = st.columns(2)

    with col_ch1:
        st.subheader("中位数Z评分")
        st.caption("Z>0 = 历史上多数行情涨得比平均多（弹性好），Z<0 = 涨得比平均少。取所有周期中位数")
        _render_z_bar(display)

    with col_ch2:
        st.subheader(f"中位涨幅%（共同窗口最近{meta.get('N', '—')}轮）")
        st.caption(f"最近{meta.get('N', '—')}轮均有数据的行情中，每只股票各轮涨幅的中位数。正值=赚钱，负值=亏钱")
        _render_return_bar(display)

    # ---- Expandable detail ----
    st.divider()
    st.subheader("个股周期明细")
    sel_name = st.selectbox(
        '选择个股查看逐轮明细',
        display['name'].tolist(),
        key='odds_detail'
    )
    if sel_name:
        sel_row = display[display['name'] == sel_name].iloc[0]
        _render_odds_detail(sel_row['ts_code'], sel_name)


def _render_z_bar(display_df):
    """Bar chart of median Z scores."""
    df = display_df.head(20).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['name'].tolist(),
        y=df['median_z'].tolist(),
        marker_color=['#22c55e' if v > 0 else '#f87171' for v in df['median_z']],
    ))
    fig.update_layout(
        height=350, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=80),
    )
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def _render_return_bar(display_df):
    """Bar chart of median returns."""
    df = display_df.head(20).copy()
    fig = go.Figure()
    rets = [v * 100 for v in df['median_return'].tolist()]
    fig.add_trace(go.Bar(
        x=df['name'].tolist(),
        y=rets,
        marker_color=['#22c55e' if v > 0 else '#f87171' for v in rets],
    ))
    fig.update_layout(
        height=350, showlegend=False,
        yaxis_title='涨幅 %',
        margin=dict(l=10, r=10, t=10, b=80),
    )
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def _render_odds_detail(ts_code, name):
    """Show per-cycle detail table for a selected stock."""
    cycles_path = os.path.join(OUTPUT_DIR, 'cycles.csv')
    if not os.path.exists(cycles_path):
        st.info("未找到周期数据文件")
        return
    cycles = pd.read_csv(cycles_path)
    cycles['start_date'] = pd.to_datetime(cycles['start_date'])
    cycles['end_date'] = pd.to_datetime(cycles['end_date'])
    cycles = cycles[cycles['start_date'] >= pd.Timestamp('2010-01-01')]

    code = ts_code.split('.')[0]
    stock_path = os.path.join(DATA_DIR, f'stock_daily_{code}.csv')
    if not os.path.exists(stock_path):
        st.info("该个股无本地数据")
        return

    sdf = pd.read_csv(stock_path)
    sdf['trade_date'] = pd.to_datetime(sdf['trade_date'])
    sdf = sdf.sort_values('trade_date').reset_index(drop=True)

    idx_path = os.path.join(DATA_DIR, 'index_daily.csv')
    idf = pd.read_csv(idx_path)
    idf['trade_date'] = pd.to_datetime(idf['trade_date'])

    records = []
    for i, (_, cycle) in enumerate(cycles.iterrows()):
        smask = (sdf['trade_date'] >= cycle['start_date']) & (sdf['trade_date'] <= cycle['end_date'])
        imask = (idf['trade_date'] >= cycle['start_date']) & (idf['trade_date'] <= cycle['end_date'])
        sseg = sdf[smask]
        iseg = idf[imask]
        if sseg.empty or iseg.empty or len(sseg) < 3:
            continue
        s_ret = float(sseg['close'].max()) / float(sseg['close'].iloc[0]) - 1
        i_ret = float(iseg['close'].max()) / float(iseg['close'].iloc[0]) - 1
        records.append({
            '周期': f"#{i+1} ({cycle['start_date'].strftime('%Y-%m')})",
            '起始': cycle['start_date'].strftime('%Y-%m-%d'),
            '结束': cycle['end_date'].strftime('%Y-%m-%d'),
            '板块涨幅': f'{i_ret:+.1%}',
            '个股涨幅': f'{s_ret:+.1%}',
            '跑赢': '✓' if s_ret > i_ret else '',
        })

    if records:
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
    else:
        st.info("该个股未参与任何2010年后周期")


def render_lead_lag_tab():
    """Tab 6: 领涨/滞涨节奏分析 — 历史规律 + 当前轮次实况."""
    st.header("领涨/滞涨节奏分析")
    st.caption('每轮行情启动后前 N 个交易日的涨幅排序，检验"谁先涨"是否跨周期稳定')

    # ============ 当前轮次实况 ============
    st.subheader("📍 当前轮次实况")
    st.caption("从本轮起点至今，哪些个股已经领涨、哪些还没动。与下方历史规律对照看。")

    # Auto-detect recent low as default
    default_start = None
    if st.session_state.index_daily is not None:
        idx_df = st.session_state.index_daily.copy()
        idx_df['trade_date'] = pd.to_datetime(idx_df['trade_date'])
        idx_df = idx_df.sort_values('trade_date')
        default_start = detect_recent_low(idx_df)

    col_d1, col_d2, col_d3 = st.columns([2, 1, 1])
    with col_d1:
        start_date = st.date_input(
            "本轮起点（板块基准）",
            value=default_start.date() if default_start is not None else pd.Timestamp('2026-06-09').date(),
            key='lead_lag_start',
            help="默认自动检测板块最近低点。个股涨幅计算方式见下方选项"
        )
    with col_d2:
        show_hist_tag = st.checkbox("标注历史档位", value=True,
                                    help="在表格中显示该股历史平均档位（1=常先涨，3=常后涨）")
    with col_d3:
        show_odds = st.checkbox("标注赔率Z分", value=True,
                                help="显示历史弹性 Z 评分（来了能涨多少）")

    use_stock_low = st.radio(
        "个股涨幅计算方式",
        ["以各自最近低点为起点", "以板块起点统一计算"],
        index=0,
        horizontal=True,
        help="个股低点≠板块低点。如中信建投4月7日见底，板块6月9日才见底。用统一起点会严重低估先见底个股的涨幅"
    ) == "以各自最近低点为起点"

    st.metric("板块最新日期", st.session_state.get('last_data_date', '—'))

    if st.session_state.index_daily is not None and start_date:
        # Index return for reference
        idx_df = st.session_state.index_daily.copy()
        idx_df['trade_date'] = pd.to_datetime(idx_df['trade_date'])
        idx_df = idx_df.sort_values('trade_date')
        idx_seg = idx_df[idx_df['trade_date'] >= pd.Timestamp(start_date)]
        if not idx_seg.empty:
            idx_ret = (float(idx_seg['close'].iloc[-1]) / float(idx_seg['close'].iloc[0]) - 1) * 100
            st.info(f"板块指数 399975.SZ：{start_date} 起 {idx_ret:+.1f}%")

    # Compute current returns (with history + odds annotation)
    stab_for_annot = None
    if show_hist_tag:
        with st.spinner('计算历史档位...'):
            _detail = compute_lead_lag(window_days=20, use_stock_low=True)
            if not _detail.empty:
                _analysis = analyze_consistency(_detail)
                stab_for_annot = _analysis.get('stock_stability')

    odds_for_annot = None
    if show_odds and st.session_state.odds_df is not None:
        odds_for_annot = st.session_state.odds_df

    with st.spinner('计算当前个股涨幅...'):
        current_df = compute_current_returns(
            start_date=start_date, stability_df=stab_for_annot, odds_df=odds_for_annot,
            use_stock_low=use_stock_low,
        )

    if current_df.empty:
        st.warning("无数据，请先加载数据")
    else:
        # ---- Bar chart: all stocks sorted by return ----
        cd = current_df.copy()
        fig = go.Figure()
        colors = ['#22c55e' if v > 0 else '#f87171' for v in cd['return_from_start']]
        fig.add_trace(go.Bar(
            x=cd['name'], y=cd['return_from_start'],
            marker_color=colors, text=cd['return_from_start'].round(1),
            texttemplate='%{text:.1f}%', textposition='outside',
            textfont=dict(size=9),
        ))
        # Add index return as horizontal line
        if not idx_seg.empty:
            fig.add_hline(y=idx_ret, line_dash='dash', line_color='blue',
                          annotation_text=f'板块 {idx_ret:+.1f}%')
        fig.update_layout(
            height=max(350, len(cd) * 14 + 80),
            margin=dict(l=10, r=10, t=10, b=120),
            yaxis_title='涨幅 %',
            showlegend=False,
        )
        fig.update_xaxes(tickangle=-60, tickfont=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)

        # ---- Detail table ----
        display = cd.copy()
        display['#'] = range(1, len(display) + 1)
        display['涨幅'] = display['return_from_start'].apply(lambda x: f'{x:+.1f}%')
        display['从低点'] = display['return_from_low'].apply(lambda x: f'{x:+.1f}%')
        display['状态'] = display['return_from_start'].apply(
            lambda x: '🔴 已领涨' if x > idx_ret * 1.5 else
                      ('🟡 跟涨' if x > idx_ret * 0.5 else
                       ('⚪ 滞涨' if x >= 0 else '🟢 仍跌'))
        )
        show_cols = ['#', 'name', 'code', 'start_date', '涨幅', '从低点', '状态', 'current_price']
        col_rename = {'name': '名称', 'code': '代码', 'start_date': '起点',
                      'current_price': '现价', 'current_date': '数据日期'}
        if show_hist_tag and 'hist_mean_tier' in display.columns:
            display['历史档位'] = display['hist_mean_tier'].apply(
                lambda x: f'{x:.1f}' if pd.notna(x) else '—')
            display['历史先涨率'] = display['hist_lead_rate'].apply(
                lambda x: f'{x:.0%}' if pd.notna(x) and x > 0 else '—')
            show_cols += ['历史档位', '历史先涨率']
        if show_odds and 'odds_z' in display.columns:
            display['赔率Z'] = display['odds_z'].apply(
                lambda x: f'{x:+.2f}' if pd.notna(x) else '—')
            display['跑赢率'] = display['odds_beat_rate'].apply(
                lambda x: f'{x:.0%}' if pd.notna(x) else '—')
            show_cols += ['赔率Z', '跑赢率']

        # ---- Opportunity filter ----
        if show_hist_tag and show_odds and 'odds_z' in display.columns:
            st.divider()
            with st.expander("🎯 机会筛选（滞涨 + 历史先涨 + 高赔率）", expanded=True):
                st.caption('找出"该涨还没涨、历史上总是先涨、且弹性好"的候选')
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    max_ret = st.slider("当前涨幅上限%", -5, 20, 5, 1,
                                        key='opp_max_ret',
                                        help="只看涨幅低于此值的（还没怎么涨的）")
                with col_f2:
                    max_tier = st.slider("历史档位≤", 1.0, 3.0, 2.0, 0.1,
                                         key='opp_max_tier',
                                         help="只看历史平均档位低于此值的（常先涨的）")
                with col_f3:
                    min_z = st.slider("赔率Z≥", -2.0, 2.0, 0.0, 0.1,
                                      key='opp_min_z',
                                      help="只看 Z 评分高于此值的（弹性好的）")

                opp = display[
                    (display['return_from_start'] <= max_ret) &
                    (display['return_from_start'] >= -10) &
                    (display['hist_mean_tier'].fillna(99) <= max_tier) &
                    (display['odds_z'].fillna(-99) >= min_z)
                ].copy()

                if opp.empty:
                    st.info("无符合条件的个股（试试放宽阈值）")
                else:
                    opp['综合分'] = (
                        (max_tier - opp['hist_mean_tier']) / max_tier * 40 +
                        opp['odds_z'] / max(opp['odds_z'].max(), 0.01) * 40 +
                        (1 - opp['return_from_start'] / max(opp['return_from_start'].max(), 1)) * 20
                    ).round(1)
                    opp = opp.sort_values('综合分', ascending=False)
                    st.success(f"找到 {len(opp)} 只候选（综合分 = 历史先涨倾向 40% + 弹性Z 40% + 尚未涨幅度 20%）")
                    opp_show = opp[['name', 'code', '涨幅', '从低点', '历史档位', '赔率Z', '跑赢率', '综合分']].copy()
                    opp_show = opp_show.rename(columns={'name': '名称', 'code': '代码'})
                    st.dataframe(opp_show, use_container_width=True, hide_index=True)
                    st.caption("⚠️ 仅为历史规律筛选，不构成投资建议。需结合 Tab 4 买入信号确认")

        st.dataframe(display[show_cols].rename(columns=col_rename),
                     use_container_width=True, hide_index=True,
                     height=min(35 * len(display) + 38, 600))

        # ---- Insight: history vs current cross-check ----
        if show_hist_tag and 'hist_mean_tier' in display.columns:
            st.divider()
            st.subheader("🔍 历史 vs 当前 交叉验证")
            valid = display.dropna(subset=['hist_mean_tier'])
            if not valid.empty:
                top_cur = valid.head(10)
                top_hist = valid.nsmallest(10, 'hist_mean_tier')
                overlap = set(top_cur['code']) & set(top_hist['code'])
                col_x1, col_x2 = st.columns(2)
                with col_x1:
                    st.caption("**当前涨幅前 10** vs 历史档位")
                    for _, r in top_cur.iterrows():
                        tag = ' ✅' if r['code'] in overlap else ''
                        st.write(f"{r['name']}  +{r['return_from_start']:.1f}%  (历史档位 {r.get('hist_mean_tier','—'):.1f}){tag}")
                with col_x2:
                    st.caption("**历史稳定先涨前 10** vs 当前涨幅")
                    for _, r in top_hist.iterrows():
                        tag = ' ✅' if r['code'] in overlap else ''
                        st.write(f"{r['name']}  (历史档位 {r['hist_mean_tier']:.1f})  +{r['return_from_start']:.1f}%{tag}")
                if overlap:
                    st.success(f"历史先涨股中本次也领涨的: {len(overlap)} 只 — {', '.join([valid[valid['code']==c]['name'].iloc[0] for c in overlap])}")
                else:
                    st.warning("历史稳定先涨股本次均未进入当前涨幅前 10 — 本轮节奏可能与历史不同")

    st.divider()

    # ============ 历史规律分析 ============
    st.subheader("📊 历史领涨/滞涨规律分析")

    # ---- Controls ----
    col_w1, col_w2, col_w3 = st.columns([1, 1, 2])
    with col_w1:
        window = st.slider("启动窗口（交易日）", 5, 40, 20, 1,
                           help='从周期起点开始计算的交易日数。窗口越短越反映"谁最先动"，越长越反映"前期谁涨得多"')
    with col_w2:
        hist_use_stock_low = st.radio(
            "低点定义",
            ["个股自身K线", "板块统一起点"],
            index=0,
            horizontal=True,
            key='hist_low_mode',
            help='个股自身K线：用argrelextrema找每只股自己的低点（和detect_cycles同算法）。板块统一：所有股票从板块起点起算'
        ) == "个股自身K线"
    with col_w3:
        with st.expander("ℹ️ 方法说明", expanded=False):
            st.markdown("""
            **定义**：每轮行情从起点开始的前 N 个交易日累计涨幅，按涨幅等分三档：
            - **tier 1 先涨组**（涨幅最高 1/3）
            - **tier 2 中段**
            - **tier 3 后涨组**（涨幅最低 1/3）

            **检验指标**：
            - **Spearman 秩相关**：两轮之间个股涨幅排名的相关性。>0 = 正相关，=0 无关联
            - **Jaccard 重合度**：两轮先涨组（tier 1）的交集/并集。高于随机基线 = 有稳定先涨股
            - **个体稳定性**：每只股票跨轮的 tier 均值（越接近 1 越常先涨）和标准差（越小越稳定）

            ⚠️ 9 轮周期、早期轮次个股不全，结论为粗略估计，非统计显著结论
            """)

    # ---- Compute ----
    with st.spinner('计算各轮个股启动涨幅...'):
        detail = compute_lead_lag(window_days=window, use_stock_low=hist_use_stock_low)
    if detail.empty:
        st.warning("无数据")
        return
    analysis = analyze_consistency(detail)

    # ---- Summary metrics ----
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("跨轮平均 Spearman", f"{analysis['avg_off_diag_spearman']:+.3f}",
                  help=">0.3 提示有跨周期稳定性")
    with col_m2:
        jac = analysis['avg_off_diag_jaccard']
        base = analysis['random_jaccard_baseline']
        st.metric("先涨组 Jaccard", f"{jac:.3f}",
                  delta=f"vs 随机 {base:.3f}",
                  delta_color="normal" if jac > base else "inverse")
    with col_m3:
        st.metric("稳定先涨股数", len(analysis['stable_leads']),
                  help="参与≥3轮且先涨率≥50%")
    with col_m4:
        stable_lags = analysis['stock_stability'][
            (analysis['stock_stability']['participation'] >= 3) &
            (analysis['stock_stability']['lag_count'] / analysis['stock_stability']['participation'] >= 0.5)
        ]
        st.metric("稳定后涨股数", len(stable_lags))

    # ---- Participation per cycle ----
    st.subheader("各轮参与度")
    cycle_info = detail.groupby('cycle_idx').agg(
        起始=('cycle_start', 'first'),
        板块涨幅=('cycle_change_pct', 'first'),
        参与股票数=('ts_code', 'count'),
    )
    cycle_info['先涨组'] = detail[detail['tier'] == 1].groupby('cycle_idx')['ts_code'].count().reindex(cycle_info.index, fill_value=0)
    cycle_info['起始'] = pd.to_datetime(cycle_info['起始']).dt.strftime('%Y-%m-%d')
    cycle_info['板块涨幅'] = cycle_info['板块涨幅'].apply(lambda x: f'+{x:.1f}%')
    cycle_info.index = [f'轮{i+1}' for i in cycle_info.index]
    st.dataframe(cycle_info, use_container_width=True, hide_index=False)

    # ---- Tier consistency heatmap (stocks × cycles) ----
    st.divider()
    st.subheader("个股 × 周期 档位矩阵")
    st.caption("行=个股（按平均档位排序），列=轮次。🟢=先涨(tier1) 🟡=中段(tier2) 🔴=后涨(tier3)。空白=该轮未上市")

    pivot = detail.pivot_table(index=['name', 'code'], columns='cycle_idx', values='tier')
    stock_order = analysis['stock_stability'].sort_values('mean_tier')[['name', 'code']]
    ordered_index = [(r['name'], r['code']) for _, r in stock_order.iterrows() if (r['name'], r['code']) in pivot.index]
    pivot = pivot.reindex(ordered_index)

    cycle_labels = [f"轮{i+1}\n{detail[detail['cycle_idx']==i]['cycle_start'].iloc[0][:7]}" for i in pivot.columns]

    z = pivot.values
    # Build text labels: 先/中/后 for tiers, empty for NaN
    tier_labels = {1.0: '先', 2.0: '中', 3.0: '后'}
    text = np.full(z.shape, '', dtype=object)
    for val, label in tier_labels.items():
        text[z == val] = label

    z_safe = np.nan_to_num(z, nan=0)
    colorscale = [[0, '#e5e5e5'], [0.34, '#f87171'], [0.5, '#fbbf24'], [0.66, '#22c55e'], [1, '#22c55e']]
    fig = go.Figure(data=go.Heatmap(
        z=z_safe, text=text, texttemplate='%{text}',
        x=cycle_labels,
        y=[f"{n}({c})" for (n, c) in pivot.index],
        colorscale=colorscale, zmin=0, zmax=3, showscale=False,
        hovertemplate='%{y}<br>%{x}<br>%{text}<extra></extra>',
    ))
    fig.update_layout(
        height=max(400, len(pivot) * 18 + 80),
        margin=dict(l=10, r=10, t=10, b=60),
        yaxis=dict(autorange='reversed'),
        xaxis=dict(side='bottom', tickangle=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Spearman & Jaccard matrices side by side ----
    st.divider()
    st.subheader("跨轮一致性矩阵")
    col_s, col_j = st.columns(2)

    with col_s:
        st.caption("**Spearman 秩相关**（个股涨幅排名的跨轮相关性）")
        sp = analysis['spearman_matrix']
        sp_labels = [f"轮{i+1}" for i in sp.columns]
        fig_sp = go.Figure(data=go.Heatmap(
            z=sp.values, x=sp_labels, y=sp_labels,
            colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
            text=sp.values.round(2), texttemplate='%{text}',
            hovertemplate='%{y} × %{x}<br>ρ=%{z:.3f}<extra></extra>',
        ))
        fig_sp.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                             yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_sp, use_container_width=True)

    with col_j:
        st.caption(f"**Jaccard 重合度**（先涨组交集/并集，随机基线={analysis['random_jaccard_baseline']:.3f}）")
        jac_m = analysis['jaccard_matrix']
        jac_labels = [f"轮{i+1}" for i in jac_m.columns]
        fig_jac = go.Figure(data=go.Heatmap(
            z=jac_m.values, x=jac_labels, y=jac_labels,
            colorscale='Greens', zmin=0, zmax=1,
            text=jac_m.values.round(2), texttemplate='%{text}',
            hovertemplate='%{y} × %{x}<br>Jaccard=%{z:.3f}<extra></extra>',
        ))
        fig_jac.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                              yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_jac, use_container_width=True)

    # ---- Stock stability table ----
    st.divider()
    st.subheader("个股节奏稳定性明细")
    st.caption("按平均档位升序。参与轮数少的新股参考价值有限。")

    ss = analysis['stock_stability'].copy()
    ss['lead_rate'] = (ss['lead_count'] / ss['participation']).apply(lambda x: f'{x:.0%}')
    ss['lag_rate'] = (ss['lag_count'] / ss['participation']).apply(lambda x: f'{x:.0%}')
    ss['mean_tier'] = ss['mean_tier'].round(2)
    ss['tier_std'] = ss['tier_std'].round(2)
    ss['mean_return'] = (ss['mean_return'] * 100).round(1).apply(lambda x: f'+{x}%')

    show_cols = ['name', 'code', 'participation', 'lead_count', 'lead_rate',
                 'lag_count', 'lag_rate', 'mean_tier', 'tier_std', 'mean_return']
    ss_display = ss[show_cols].rename(columns={
        'name': '名称', 'code': '代码', 'participation': '参与轮数',
        'lead_count': '先涨次数', 'lead_rate': '先涨率',
        'lag_count': '后涨次数', 'lag_rate': '后涨率',
        'mean_tier': '平均档位', 'tier_std': '档位σ', 'mean_return': '平均窗口涨幅',
    })
    st.dataframe(ss_display, use_container_width=True, hide_index=True,
                 height=min(35 * len(ss_display) + 38, 700))

    # ---- Stable leads / lags ----
    col_sl, col_sla = st.columns(2)
    with col_sl:
        st.subheader("🎯 稳定先涨股")
        st.caption("参与≥3轮 且 先涨率≥50%")
        sl = analysis['stable_leads'][['name', 'code', 'participation', 'lead_count', 'lead_rate', 'mean_tier']].copy()
        if sl.empty:
            st.info("无")
        else:
            st.dataframe(sl.rename(columns={
                'name': '名称', 'code': '代码', 'participation': '参与轮数',
                'lead_count': '先涨次数', 'lead_rate': '先涨率', 'mean_tier': '平均档位'
            }), use_container_width=True, hide_index=True)

    with col_sla:
        st.subheader("🐢 稳定后涨股")
        st.caption("参与≥3轮 且 后涨率≥50%")
        sl_all = analysis['stock_stability']
        stable_lags_df = sl_all[(sl_all['participation'] >= 3) & (sl_all['lag_count'] / sl_all['participation'] >= 0.5)]
        if stable_lags_df.empty:
            st.info("无")
        else:
            lag_display = stable_lags_df[['name', 'code', 'participation', 'lag_count', 'mean_tier']].copy()
            lag_display['lag_rate'] = (lag_display['lag_count'] / lag_display['participation']).apply(lambda x: f'{x:.0%}')
            st.dataframe(lag_display.rename(columns={
                'name': '名称', 'code': '代码', 'participation': '参与轮数',
                'lag_count': '后涨次数', 'lag_rate': '后涨率', 'mean_tier': '平均档位'
            }), use_container_width=True, hide_index=True)


def main():
    st.title("券商板块行情启动信号回测系统")
    st.caption("基于历史行情回测，筛选有效的技术指标买入信号")

    render_sidebar()

    load_odds()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 行情周期总览",
        "🏆 指标回测排行榜",
        "📊 个股回测对比",
        "🔍 行情跟踪",
        "💰 优选赔率",
        "⚡ 领涨/滞涨分析",
    ])

    with tab1:
        render_cycle_overview()

    with tab2:
        render_indicator_rankings()

    with tab3:
        render_stock_backtest()

    with tab4:
        render_live_tracking()

    with tab5:
        render_odds_tab()

    with tab6:
        render_lead_lag_tab()

    st.divider()
    st.caption("⚠️ 本工具仅提供技术指标信号分析，不构成投资建议。历史回测结果不代表未来表现。")


if __name__ == '__main__':
    main()
