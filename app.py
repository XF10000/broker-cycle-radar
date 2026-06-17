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
    fetch_moneyflow, STOCKS, INDEX_CODE, INDEX_CONSTITUENTS,
)
from indicators import get_all_signal_rules, INDICATOR_REGISTRY, filter_signals, get_indicator_lines
from backtest import run_backtest, run_and_save, run_and_save_all, judge_signal, count_false_signals, run_all_stocks_backtest

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
        'stocks_moneyflow': {},
        'backtest_results': None,
        'stock_results': None,
        'cycles_df': None,
        'display_cycles': None,
        'signal_window': 30,
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


@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data_cached():
    """Load all data with caching. Returns dict."""
    from data_fetcher import fetch_index_daily as _fetch_idx, fetch_stock_daily as _fetch_stk
    result = {}
    errors = []

    try:
        result['index_daily'] = _fetch_idx()
        result['index_weekly'] = daily_to_weekly(result['index_daily'])
    except Exception as e:
        errors.append(f'指数数据: {e}')

    result['stocks_daily'] = {}
    result['stocks_weekly'] = {}
    result['stocks_moneyflow'] = {}
    for code in INDEX_CONSTITUENTS:
        ts_code = f'{code}.SH' if code.startswith('6') else f'{code}.SZ'
        try:
            sd = _fetch_stk(ts_code)
            if not sd.empty:
                result['stocks_daily'][code] = sd
                result['stocks_weekly'][code] = daily_to_weekly(sd)
        except Exception as e:
            errors.append(f'个股{code}: {e}')
        try:
            mf = fetch_moneyflow(ts_code)
            if not mf.empty:
                result['stocks_moneyflow'][code] = mf
        except Exception:
            pass

    result['errors'] = errors
    return result


def load_data(force=False):
    if force:
        st.cache_data.clear()
    with st.spinner('加载数据中...'):
        data = load_all_data_cached()
    st.session_state.index_daily = data.get('index_daily')
    st.session_state.index_weekly = data.get('index_weekly')
    st.session_state.stocks_daily = data.get('stocks_daily', {})
    st.session_state.stocks_weekly = data.get('stocks_weekly', {})
    st.session_state.stocks_moneyflow = data.get('stocks_moneyflow', {})
    st.session_state.data_loaded = True
    st.session_state.data_error = '; '.join(data.get('errors', []))
    if st.session_state.index_daily is not None:
        st.session_state.last_data_date = str(
            st.session_state.index_daily['trade_date'].max().date()
        )


def load_cycles():
    path = os.path.join(OUTPUT_DIR, 'cycles.csv')
    if os.path.exists(path):
        all_df = pd.read_csv(path)
        st.session_state.cycles_df = all_df  # keep all for reference calc
        # Display only 2010+ cycles
        st.session_state.display_cycles = all_df[
            pd.to_datetime(all_df['start_date']) >= pd.Timestamp('2010-01-01')
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
        ('OBV底背离', {'lookback': 90}),
        ('CCI脱离超卖', {'period': 20}),
        ('MACD柱线缩短', {'fast': 12, 'slow': 26, 'signal': 9, 'consecutive': 3}),
    ]

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
            sig_count = 0
            for ind_name, params in top_inds:
                cfg = INDICATOR_REGISTRY.get(ind_name)
                if cfg is None:
                    continue
                try:
                    raw = cfg['func'](sdf, **params)
                    filtered = filter_signals(sdf, raw)
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
        return
    data_dfs = {
        'index_daily': st.session_state.index_daily,
        'index_weekly': st.session_state.index_weekly,
        'index_moneyflow': None,
    }
    try:
        results = run_and_save_all(data_dfs, st.session_state.signal_window)
        st.session_state.backtest_results = results
        # Also run stock backtest
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
        cycles = st.session_state.cycles_df.to_dict('records')
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
    if stock_ref_highs_full is None:
        stock_ref_highs_full = [float(full_df.loc[d, 'close']) if d in full_df.index else 0.0 for d in full_df.index]

    # Map ref_highs to compute_seg dates
    stock_ref_highs = []
    for d in compute_seg['trade_date']:
        if d in full_df.index:
            idx = full_df.index.get_loc(d)
            stock_ref_highs.append(stock_ref_highs_full[idx])
        else:
            stock_ref_highs.append(float(compute_seg[compute_seg['trade_date'] == d]['close'].iloc[0]))

    try:
        signals = cfg['func'](compute_seg, **params)
        signals = filter_signals(compute_seg, signals, ref_highs=stock_ref_highs)
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
    ), row=1, col=1)

    # MA250
    full_close = compute_seg['close'].values.astype(float)
    ma250 = talib.SMA(full_close, 250)
    ma250_disp = ma250[disp_mask.values]
    fig.add_trace(go.Scatter(
        x=segment['trade_date'], y=ma250_disp,
        name='年线(250)', line=dict(color='orange', width=1),
        connectgaps=False,
    ), row=1, col=1)

    # Stock-specific decline line (reuse stock_ref_highs from above)
    decline_line = np.array(stock_ref_highs) * 0.80
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
                       height=420, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
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
    rh = []
    for i, d in enumerate(df['trade_date'] if 'trade_date' in df.columns else df.index):
        ref = None
        for pd_, pp in cycle_peaks:
            if pd_ <= d: ref = pp
            else: break
        # After last cycle: use max close since last cycle end
        if ref is None:
            ref = close[i]
        elif d > cycle_peaks[-1][0]:
            since = close[(df['trade_date'] > cycle_peaks[-1][0]) & (df['trade_date'] <= d)]
            since_max = since.max() if len(since) > 0 else ref
            ref = max(ref, since_max)
        rh.append(ref)
    return rh


def _detect_stock_peaks(stock_df):
    """Find cycle end prices for a stock using same algorithm as detect_cycles.py."""
    from scipy.signal import argrelextrema
    close = stock_df['close'].values.astype(float)
    smoothed = pd.Series(close).rolling(20, min_periods=1).mean().values
    peaks = argrelextrema(smoothed, np.greater, order=10)[0]
    troughs = argrelextrema(smoothed, np.less, order=10)[0]
    cycle_highs = []
    for t_idx in troughs:
        later = peaks[peaks > t_idx]
        if len(later) == 0: continue
        p_idx = later[0]
        if int(p_idx - t_idx) < 20: continue
        chg = (close[p_idx] - close[t_idx]) / close[t_idx] * 100
        if chg < 25: continue
        cycle_highs.append((stock_df['trade_date'].iloc[p_idx], close[p_idx]))
    return sorted(cycle_highs)


def render_sidebar():
    st.sidebar.title("券商板块信号回测")

    st.sidebar.subheader("数据状态")
    if st.session_state.data_loaded and st.session_state.index_daily is not None:
        st.sidebar.success(f"已加载 | 最新: {st.session_state.last_data_date}")
    else:
        st.sidebar.warning("数据未加载")

    if st.session_state.data_error:
        st.sidebar.warning(f"⚠️ {st.session_state.data_error}")

    if st.sidebar.button("📥 加载/刷新数据", use_container_width=True):
        load_data(force=True)
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
    ), row=1, col=1)

    # Add MA250 and decline threshold lines for context
    if freq == '日线':
        full_close = compute_seg['close'].values.astype(float)
        ma250 = talib.SMA(full_close, 250)
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
        decline_line = np.array(rh) * 0.80
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
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
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
                    win_end = ws_date + pd.Timedelta(days=30)
                    win_mask = (d_df.index >= ws_date) & (d_df.index <= win_end)
                    if d_sig[win_mask].any():
                        resonance_sig[d_sig[win_mask].index[0]] = True

                for cycle in cycles:
                    start = pd.Timestamp(cycle['start_date'])
                    end = pd.Timestamp(cycle['end_date'])
                    j = judge_signal(resonance_sig, start, end, signal_window=st.session_state.signal_window)
                    if j['hit']:
                        z_row.append(3 if j['days_before'] <= 15 else 2)
                        t_row.append(f"共振命中 提前{j['days_before']}天")
                    elif j['late']:
                        sd = j['signal_date']
                        days_after = (sd - start).days
                        z_row.append(1 if days_after <= 5 else 0.5)
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
                    z_row.append(3 if days <= 15 else 2)
                    t_row.append(f"命中 提前{days}天")
                elif j['late']:
                    sd = j['signal_date']
                    days_after = (sd - start).days
                    z_row.append(1 if days_after <= 5 else 0.5)
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
                '综合得分', format='%.3f', min_value=0, max_value=0.5,
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
    ), row=1, col=1)

    # Add 250MA and decline threshold to K-line
    full_close = compute_seg['close'].values.astype(float)
    ma250 = talib.SMA(full_close, 250)
    # For index: use cycle peaks; for stocks: detect own cycle peaks
    if is_index:
        ref_highs = _build_ref_for_df(compute_seg)
        decline_disp = (np.array(ref_highs) * 0.85)[disp_mask.values] if ref_highs else None
    else:
        stock_peaks = _detect_stock_peaks(compute_seg)
        if stock_peaks:
            last_peak_date = stock_peaks[-1][0]
            rh = []
            for i, d in enumerate(compute_seg['trade_date']):
                ref = None
                for pd_, pp in stock_peaks:
                    if pd_ <= d: ref = pp
                    else: break
                # After last peak: use max close since that peak
                if d > last_peak_date:
                    since = full_close[(compute_seg['trade_date'] > last_peak_date) & (compute_seg['trade_date'] <= d)]
                    since_max = since.max() if len(since) > 0 else ref
                    ref = max(ref, since_max)
                rh.append(ref if ref is not None else full_close[i])
            ref_highs = rh
        else:
            ref_highs = None
        decline_disp = (np.array(ref_highs) * 0.85)[disp_mask.values] if ref_highs else None

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
                    marker=dict(symbol='triangle-up', size=10, color='blue'), name='CCI信号'), row=2, col=1)

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
                    marker=dict(symbol='triangle-up', size=10, color='blue'), name='MACD信号'), row=3, col=1)

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
                marker=dict(symbol='triangle-up', size=10, color='blue'), name='OBV信号'), row=4, col=1)

    fig.update_xaxes(rangebreaks=_build_rangebreaks(seg['trade_date']), tickformat='%Y-%m',
                     rangeslider=dict(visible=True, thickness=0.03), row=4, col=1)
    fig.update_xaxes(rangebreaks=_build_rangebreaks(seg['trade_date']), tickformat='%Y-%m',
                     rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(rangebreaks=_build_rangebreaks(seg['trade_date']), tickformat='%Y-%m',
                     rangeslider_visible=False, row=2, col=1)
    fig.update_layout(
        title=f"{'板块指数' if is_index else label} — 近1年走势",
        height=900, showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_signal_summary(top_inds):
    """Show current signal status for all stocks, sorted by Z score."""
    # Build Z lookup
    z_map = {}
    if st.session_state.odds_df is not None and not st.session_state.odds_df.empty:
        for _, r in st.session_state.odds_df.iterrows():
            raw = r['ts_code'].split('.')[0]
            z = r.get('median_z', -999)
            if pd.isna(z):
                z = -999
            z_map[raw] = float(z)

    rows = []
    for code, df in st.session_state.stocks_daily.items():
        name = INDEX_CONSTITUENTS.get(code, code)
        z_val = z_map.get(code, -999)
        df = df.copy()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date')
        recent_cutoff = df['trade_date'].max() - pd.Timedelta(days=90)
        
        for ind_name in top_inds:
            cfg = INDICATOR_REGISTRY.get(ind_name)
            if cfg is None: continue
            try:
                sig = cfg['func'](df, **cfg['params'][0])
                sig = filter_signals(df, sig)
            except Exception:
                continue
            recent = (df['trade_date'] >= recent_cutoff).values
            recent_sig = sig.values & recent
            last_date = None
            if recent_sig.any():
                last_date = df['trade_date'].iloc[np.where(recent_sig)[0][-1]] if np.where(recent_sig)[0].size > 0 else None
            rows.append({
                '股票': name,
                'Z评分': f"{z_val:+.2f}" if z_val > -999 else '—',
                '指标': ind_name,
                '最近信号': last_date.date() if last_date else '无',
                '_z': z_val,
            })
    
    if rows:
        rows.sort(key=lambda r: r['_z'], reverse=True)
        df_rows = pd.DataFrame(rows).drop(columns=['_z'])
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

    # ---- Compute signals ----
    signal_df = compute_odds_signals()
    signal_map = {}
    if not signal_df.empty:
        signal_map = dict(zip(signal_df['ts_code'], signal_df['signal']))

    # ---- Main ranking table ----
    st.subheader("券商个股赔率排名")
    st.caption("按中位数Z降序排列，点击列头可切换排序。▲ 买点 = 当前触发信号，○ = 无信号")

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
        s = signal_map.get(code, 'no_data')
        if s == 'buy':
            return '▲ 买点'
        elif s == 'no_data':
            return '—'
        return '○ 无'
    display['信号'] = display['ts_code'].apply(fmt_signal)

    display = display.sort_values('median_z', ascending=False).reset_index(drop=True)

    show_cols = ['#', 'name', '中位数Z', 'Z最高值', 'Z正值率', '中位涨幅',
                 '最大涨幅', '跑赢概率', '置信度', '轮数', '信号']

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
        _render_z_bar(display)

    with col_ch2:
        st.subheader(f"中位涨幅%（共同窗口最近{meta.get('N', '—')}轮）")
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


def main():
    st.title("券商板块行情启动信号回测系统")
    st.caption("基于历史行情回测，筛选有效的技术指标买入信号")

    render_sidebar()

    load_odds()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 行情周期总览",
        "🏆 指标回测排行榜",
        "📊 个股回测对比",
        "🔍 行情跟踪",
        "💰 优选赔率",
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

    st.divider()
    st.caption("⚠️ 本工具仅提供技术指标信号分析，不构成投资建议。历史回测结果不代表未来表现。")


if __name__ == '__main__':
    main()
