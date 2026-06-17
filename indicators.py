"""
Technical indicator library for securities sector bull-run signal detection.
Uses TA-Lib for standard indicator computation. Signal detection logic is custom.

Each indicator function:
    Input:  df (pd.DataFrame with columns: trade_date, open, high, low, close, vol, amount)
    Output: pd.Series of bool (True = signal triggered on that day)

Warm-up: TA-Lib returns NaN during the lookback period. Signal code naturally
skips NaN values since they never satisfy trigger conditions.
"""
import numpy as np
import pandas as pd
import talib
from scipy.signal import argrelextrema


# ============================================================
# 5.1 Trend indicators
# ============================================================

def macd_divergence(df, fast=12, slow=26, signal=9, lookback=60):
    """
    MACD bottom divergence: adjacent troughs — price makes lower low
    but DIF makes higher low. Uses local minima detection.
    """
    close = df['close'].values.astype(float)
    dif, dea, hist = talib.MACD(close, fastperiod=fast, slowperiod=slow, signalperiod=signal)

    n = len(close)
    sig = pd.Series(False, index=df.index)

    # Find local price minima (troughs) — order controls min distance between troughs
    order = max(lookback // 4, 5)
    troughs = argrelextrema(close, np.less, order=order)[0]

    # Compare adjacent troughs for divergence
    for j in range(1, len(troughs)):
        t1 = troughs[j-1]  # earlier trough
        t2 = troughs[j]    # later trough

        if np.isnan(dif[t1]) or np.isnan(dif[t2]):
            continue

        # Price: lower low at t2
        if close[t2] >= close[t1]:
            continue

        # DIF: higher low at t2 (divergence!)
        if dif[t2] > dif[t1]:
            sig.iloc[t2] = True

    return sig


def macd_histogram_shrink(df, fast=12, slow=26, signal=9, consecutive=3):
    """
    MACD histogram bars (below zero) shortening for N consecutive days.
    Indicates declining bearish momentum — potential bottom.
    """
    close = df['close'].values.astype(float)
    dif, dea, hist = talib.MACD(close, fastperiod=fast, slowperiod=slow, signalperiod=signal)

    n = len(close)
    sig = pd.Series(False, index=df.index)

    for i in range(consecutive, n):
        ok = True
        for j in range(consecutive):
            if np.isnan(hist[i-consecutive+1+j]):
                ok = False
                break
        if not ok:
            continue
        # All bars below zero and consecutively rising (getting shorter/closer to zero)
        rising = True
        below_zero = True
        for j in range(consecutive):
            if hist[i-consecutive+1+j] >= 0:
                below_zero = False
            if j > 0 and hist[i-consecutive+1+j] <= hist[i-consecutive+j]:
                rising = False
        if rising and below_zero:
            sig.iloc[i] = True

    return sig


def ma_return(df, ma_period=60, below_days=20):
    """
    Price crosses back above MA after running below it for at least below_days.
    """
    close = df['close'].values.astype(float)
    ma = talib.SMA(close, timeperiod=ma_period)

    below = df['close'] < pd.Series(ma, index=df.index)
    sig = pd.Series(False, index=df.index)
    consec = below.astype(int).groupby((below != below.shift()).cumsum()).cumsum()

    for i in range(1, len(df)):
        if np.isnan(ma[i]):
            continue
        if consec.iloc[i-1] >= below_days and below.iloc[i-1] and not below.iloc[i]:
            sig.iloc[i] = True
    return sig


def adx_turn(df, period=14, adx_low=20):
    """
    ADX turns up from below adx_low, with +DI crossing above -DI simultaneously.
    """
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    close = df['close'].values.astype(float)

    adx = talib.ADX(high, low, close, timeperiod=period)
    p_di = talib.PLUS_DI(high, low, close, timeperiod=period)
    m_di = talib.MINUS_DI(high, low, close, timeperiod=period)

    sig = pd.Series(False, index=df.index)
    for i in range(period * 2, len(df)):
        if np.isnan(adx[i]) or np.isnan(adx[i-1]):
            continue
        if (adx[i] > adx[i-1] and
            adx[i-1] <= adx_low and
            p_di[i] > m_di[i] and
            p_di[i-1] <= m_di[i-1]):
            sig.iloc[i] = True
    return sig


# ============================================================
# 5.2 Momentum indicators
# ============================================================

def rsi_divergence(df, period=14, lookback=60):
    """
    RSI bottom divergence: price at new low in lookback but RSI is not.
    TA-Lib RSI uses Wilder's smoothing (standard).
    """
    close = df['close'].values.astype(float)
    rsi = talib.RSI(close, timeperiod=period)

    sig = pd.Series(False, index=df.index)
    for i in range(lookback, len(df)):
        if np.isnan(rsi[i]):
            continue
        wc = close[i-lookback:i+1]
        wr = rsi[i-lookback:i+1]
        wr = wr[~np.isnan(wr)]
        if len(wr) == 0:
            continue
        if (wc[-1] <= wc.min() and
            rsi[i] > wr.min() * 0.95 and
            rsi[i] < 50):
            sig.iloc[i] = True
    return sig


def kdj_j_reversal(df, n=9, m1=3, m2=3):
    """
    KDJ: J value crosses above +20 after being below 0.
    TA-Lib STOCH returns %K and %D. J = 3*%K - 2*%D.
    """
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    close = df['close'].values.astype(float)

    k, d = talib.STOCH(high, low, close,
                       fastk_period=n, slowk_period=m1, slowd_period=m2)
    j = 3 * k - 2 * d

    sig = pd.Series(False, index=df.index)
    for i in range(2, len(df)):
        if np.isnan(j[i-1]) or np.isnan(j[i]):
            continue
        if j[i-1] < 0 and j[i] > 20:
            sig.iloc[i] = True
    return sig


def cci_reversal(df, period=20):
    """
    CCI crosses above -100 from below.
    """
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    close = df['close'].values.astype(float)

    cci = talib.CCI(high, low, close, timeperiod=period)

    sig = pd.Series(False, index=df.index)
    for i in range(1, len(df)):
        if np.isnan(cci[i-1]) or np.isnan(cci[i]):
            continue
        if cci[i-1] < -100 and cci[i] > -100:
            sig.iloc[i] = True
    return sig


# ============================================================
# 5.3 Volume indicators
# ============================================================

def volume_contraction(df, ma_period=20, vol_ratio=0.5, flat_days=5, flat_pct=2.0):
    """Volume shrinks + price flat. Pure pandas — no TA-Lib equivalent."""
    vol = df['vol'].values.astype(float)
    close = df['close'].values.astype(float)

    vol_ma = talib.SMA(vol, timeperiod=ma_period)
    sig = pd.Series(False, index=df.index)

    for i in range(flat_days, len(df)):
        if np.isnan(vol_ma[i]):
            continue
        if vol[i] >= vol_ma[i] * vol_ratio:
            continue
        window = close[i-flat_days:i+1]
        price_range = (window.max() - window.min()) / window.mean() * 100
        if price_range < flat_pct:
            sig.iloc[i] = True
    return sig


def moderate_expansion(df, contract_window=10, expand_ratio=1.3, max_ratio=1.5):
    """Volume expands moderately + small bullish candle."""
    vol = df['vol'].values.astype(float)
    close = df['close'].values.astype(float)
    open_ = df['open'].values.astype(float)

    vol_ma = talib.SMA(vol, timeperiod=20)

    sig = pd.Series(False, index=df.index)
    for i in range(contract_window + 1, len(df)):
        if np.isnan(vol_ma[i]):
            continue
        prior = vol[i-contract_window:i]
        if prior.mean() > vol_ma[i] * 0.8:
            continue
        ratio = vol[i] / vol_ma[i]
        if ratio < expand_ratio or ratio > max_ratio:
            continue
        chg = (close[i] - close[i-1]) / close[i-1] * 100
        if 1.0 <= chg <= 3.0 and close[i] > open_[i]:
            sig.iloc[i] = True
    return sig


def obv_divergence(df, lookback=60):
    """OBV divergence: price declining but OBV rising."""
    close = df['close'].values.astype(float)
    vol = df['vol'].values.astype(float)

    obv = talib.OBV(close, vol)

    sig = pd.Series(False, index=df.index)
    for i in range(lookback, len(df)):
        if np.isnan(obv[i]):
            continue
        wc = close[i-lookback:i+1]
        wo = obv[i-lookback:i+1]
        wo = wo[~np.isnan(wo)]
        if len(wo) == 0:
            continue
        if wc[-1] <= wc.mean() and obv[i] > wo.mean():
            sig.iloc[i] = True
    return sig


# ============================================================
# 5.4 Volatility indicators
# ============================================================

def bollinger_squeeze(df, period=20, std=2, ndays_newlow=60):
    """Bollinger Band width at N-day low — signals impending breakout."""
    close = df['close'].values.astype(float)
    upper, middle, lower = talib.BBANDS(close, timeperiod=period, nbdevup=std, nbdevdn=std)

    bandwidth = (upper - lower) / (middle + 1e-10)

    sig = pd.Series(False, index=df.index)
    for i in range(ndays_newlow, len(df)):
        if np.isnan(bandwidth[i]):
            continue
        if bandwidth[i] <= np.nanmin(bandwidth[i-ndays_newlow:i]):
            sig.iloc[i] = True
    return sig


def bollinger_lower_rebound(df, period=20, std=2):
    """Price crosses above middle band after running near lower band."""
    close = df['close'].values.astype(float)
    low = df['low'].values.astype(float)
    upper, middle, lower = talib.BBANDS(close, timeperiod=period, nbdevup=std, nbdevdn=std)

    sig = pd.Series(False, index=df.index)
    for i in range(period, len(df)):
        if np.isnan(middle[i]) or np.isnan(lower[i]):
            continue
        near_lower = low[i-3:i+1].min() <= np.nanmax(lower[i-3:i+1])
        if (close[i-1] < middle[i-1] and close[i] > middle[i] and near_lower):
            sig.iloc[i] = True
    return sig


# ============================================================
# 5.5 Money flow indicators (no TA-Lib equivalent, kept as-is)
# ============================================================

def main_force_inflow(df, consecutive_days=3):
    sig = pd.Series(False, index=df.index)
    if 'buy_elg_vol' not in df.columns or 'sell_elg_vol' not in df.columns:
        return sig
    net = df['buy_elg_vol'] - df['sell_elg_vol']
    positive = net > 0
    for i in range(consecutive_days - 1, len(df)):
        if positive.iloc[i-consecutive_days+1:i+1].all():
            sig.iloc[i] = True
    return sig


def margin_stabilize(df, window=5, threshold=-1.0):
    sig = pd.Series(False, index=df.index)
    if 'rzye' not in df.columns:
        return sig
    change_pct = df['rzye'].pct_change(periods=window) * 100
    for i in range(window + 1, len(df)):
        if change_pct.iloc[i] > threshold:
            sig.iloc[i] = True
    return sig


# ============================================================
# Indicator registry with parameter search spaces
# ============================================================

INDICATOR_REGISTRY = {
    'MACD底背离': {
        'func': macd_divergence,
        'category': '趋势',
        'params': [
            {'fast': 12, 'slow': 26, 'signal': 9, 'lookback': 60},
            {'fast': 12, 'slow': 26, 'signal': 9, 'lookback': 90},
            {'fast': 5, 'slow': 34, 'signal': 5, 'lookback': 60},
        ],
    },
    'MACD柱线缩短': {
        'func': macd_histogram_shrink,
        'category': '趋势',
        'params': [
            {'fast': 12, 'slow': 26, 'signal': 9, 'consecutive': 2},
            {'fast': 12, 'slow': 26, 'signal': 9, 'consecutive': 3},
            {'fast': 12, 'slow': 26, 'signal': 9, 'consecutive': 5},
            {'fast': 5, 'slow': 34, 'signal': 5, 'consecutive': 3},
        ],
    },
    '均线回归': {
        'func': ma_return,
        'category': '趋势',
        'params': [
            {'ma_period': 60, 'below_days': 20},
            {'ma_period': 60, 'below_days': 30},
            {'ma_period': 120, 'below_days': 30},
        ],
    },
    'ADX拐头': {
        'func': adx_turn,
        'category': '趋势',
        'params': [
            {'period': 14, 'adx_low': 20},
            {'period': 14, 'adx_low': 25},
        ],
    },
    'RSI底背离': {
        'func': rsi_divergence,
        'category': '动量',
        'params': [
            {'period': 14, 'lookback': 60},
            {'period': 14, 'lookback': 90},
            {'period': 9, 'lookback': 60},
        ],
    },
    'KDJ_J值反转': {
        'func': kdj_j_reversal,
        'category': '动量',
        'params': [
            {'n': 9, 'm1': 3, 'm2': 3},
        ],
    },
    'CCI脱离超卖': {
        'func': cci_reversal,
        'category': '动量',
        'params': [
            {'period': 20},
            {'period': 14},
        ],
    },
    '缩量止跌': {
        'func': volume_contraction,
        'category': '成交量',
        'params': [
            {'ma_period': 20, 'vol_ratio': 0.5, 'flat_days': 5, 'flat_pct': 2.0},
            {'ma_period': 20, 'vol_ratio': 0.6, 'flat_days': 10, 'flat_pct': 2.0},
        ],
    },
    '温和放量': {
        'func': moderate_expansion,
        'category': '成交量',
        'params': [
            {'contract_window': 10, 'expand_ratio': 1.3, 'max_ratio': 1.5},
            {'contract_window': 20, 'expand_ratio': 1.3, 'max_ratio': 1.5},
        ],
    },
    'OBV底背离': {
        'func': obv_divergence,
        'category': '成交量',
        'params': [
            {'lookback': 60},
            {'lookback': 90},
        ],
    },
    '布林带收窄': {
        'func': bollinger_squeeze,
        'category': '波动',
        'params': [
            {'period': 20, 'std': 2, 'ndays_newlow': 40},
            {'period': 20, 'std': 2, 'ndays_newlow': 60},
            {'period': 20, 'std': 2, 'ndays_newlow': 120},
        ],
    },
    '布林带下轨反弹': {
        'func': bollinger_lower_rebound,
        'category': '波动',
        'params': [
            {'period': 20, 'std': 2},
        ],
    },
    '主力净流入': {
        'func': main_force_inflow,
        'category': '资金',
        'params': [
            {'consecutive_days': 3},
        ],
        'requires_moneyflow': True,
    },
    '融资余额企稳': {
        'func': margin_stabilize,
        'category': '资金',
        'params': [
            {'window': 5, 'threshold': -1.0},
            {'window': 10, 'threshold': -1.0},
        ],
        'requires_moneyflow': True,
    },
}


def get_all_signal_rules():
    """Expand registry into flat list of dicts."""
    rules = []
    for name, cfg in INDICATOR_REGISTRY.items():
        for p in cfg['params']:
            rules.append({
                'name': name,
                'category': cfg['category'],
                'func': cfg['func'],
                'params': p,
                'requires_moneyflow': cfg.get('requires_moneyflow', False),
            })
    return rules


# ============================================================
# Signal context filter
# ============================================================

def filter_signals(df, signal_series, ma_period=250, decline_pct=20, cooldown=15, state_window=5, ref_highs=None):
    """
    Post-filter signals with trend context and cooldown.
    Only keep signals when market is in correction/downtrend mode.

    Context rule: decline > decline_pct (20%) from reference high, AND close < MA250.
    Once context is met, a state_window of trading days (5) keeps it alive (latching)

    Args:
        decline_pct: minimum decline percentage (default 20).
        state_window: trading days to keep context alive after last met (default 5).
        ref_highs: optional array of reference high prices (same length as df).
                   If None, uses rolling 250-day max.
    """
    close = df['close'].values.astype(float)
    n = len(close)

    ma = talib.SMA(close, timeperiod=ma_period)
    if ref_highs is not None:
        decline = (np.array(ref_highs) - close) / (np.array(ref_highs) + 1e-10) * 100
    else:
        rolling_high = pd.Series(close).rolling(250, min_periods=1).max().values
        decline = (rolling_high - close) / (rolling_high + 1e-10) * 100

    # Per-day context: is market in correction mode?
    context_ok = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isnan(ma[i]) and not np.isnan(decline[i]):
            context_ok[i] = (close[i] < ma[i]) and (decline[i] > decline_pct)

    # Single pass: track last context date and filter signals by proximity
    filtered = signal_series.copy()
    last_context_idx = -state_window - 1
    last_signal_idx = -cooldown - 1
    signal_set = set(np.where(signal_series.values)[0])

    for i in range(n):
        if context_ok[i]:
            last_context_idx = i
        if i in signal_set:
            if i >= n or np.isnan(ma[i]):
                filtered.iloc[i] = False
            elif i - last_context_idx > state_window:
                filtered.iloc[i] = False
            elif i - last_signal_idx < cooldown:
                filtered.iloc[i] = False
            else:
                last_signal_idx = i

    return filtered


# ============================================================
# Indicator display lines (for chart rendering)
# ============================================================

def get_indicator_lines(df, indicator_name, params=None):
    """
    Return display lines for an indicator to show below K-line chart.
    Uses TA-Lib for computation.
    """
    close = df['close'].values.astype(float)
    if params is None:
        cfg = INDICATOR_REGISTRY.get(indicator_name)
        if cfg:
            params = cfg['params'][0]
        else:
            return []

    if indicator_name in ('MACD底背离', 'MACD柱线缩短'):
        fast = params.get('fast', 12)
        slow = params.get('slow', 26)
        sig = params.get('signal', 9)
        dif, dea, hist = talib.MACD(close, fastperiod=fast, slowperiod=slow, signalperiod=sig)
        # Per-bar colors: Chinese convention (red above zero, green below)
        colors = []
        for i in range(len(hist)):
            if np.isnan(hist[i]):
                colors.append('gray')
            elif hist[i] >= 0:
                if i > 0 and not np.isnan(hist[i-1]) and hist[i] > hist[i-1]:
                    colors.append('#ef5350')  # 深红 (accel)
                else:
                    colors.append('#ef9a9a')  # 浅红 (decel)
            else:
                if i > 0 and not np.isnan(hist[i-1]) and hist[i] < hist[i-1]:
                    colors.append('#2e7d32')  # 深绿 (accel)
                else:
                    colors.append('#81c784')  # 浅绿 (decel)
        return [
            {'name': 'DIF', 'values': dif, 'color': 'blue', 'row': 2},
            {'name': 'DEA', 'values': dea, 'color': 'orange', 'row': 2},
            {'name': 'HIST', 'values': hist, 'colors': colors, 'row': 2, 'type': 'bar'},
        ]

    elif indicator_name == 'RSI底背离':
        period = params.get('period', 14)
        rsi = talib.RSI(close, timeperiod=period)
        return [
            {'name': 'RSI', 'values': rsi, 'color': 'purple', 'row': 2},
            {'name': '30', 'values': np.full(len(df), 30), 'color': 'gray', 'row': 2, 'dash': 'dash'},
            {'name': '70', 'values': np.full(len(df), 70), 'color': 'gray', 'row': 2, 'dash': 'dash'},
        ]

    elif indicator_name == 'KDJ_J值反转':
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        n = params.get('n', 9)
        m1 = params.get('m1', 3)
        m2 = params.get('m2', 3)
        k, d = talib.STOCH(high, low, close, fastk_period=n, slowk_period=m1, slowd_period=m2)
        j = 3 * k - 2 * d
        return [
            {'name': 'K', 'values': k, 'color': 'blue', 'row': 2},
            {'name': 'D', 'values': d, 'color': 'orange', 'row': 2},
            {'name': 'J', 'values': j, 'color': 'purple', 'row': 2},
        ]

    elif indicator_name == 'CCI脱离超卖':
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        period = params.get('period', 20)
        cci = talib.CCI(high, low, close, timeperiod=period)
        return [
            {'name': 'CCI', 'values': cci, 'color': 'blue', 'row': 2},
            {'name': '+100', 'values': np.full(len(df), 100), 'color': 'gray', 'row': 2, 'dash': 'dash'},
            {'name': '-100', 'values': np.full(len(df), -100), 'color': 'gray', 'row': 2, 'dash': 'dash'},
        ]

    elif indicator_name in ('布林带收窄', '布林带下轨反弹'):
        period = params.get('period', 20)
        std = params.get('std', 2)
        upper, middle, lower = talib.BBANDS(close, timeperiod=period, nbdevup=std, nbdevdn=std)
        return [
            {'name': '上轨', 'values': upper, 'color': 'gray', 'row': 2, 'dash': 'dash'},
            {'name': '中轨', 'values': middle, 'color': 'orange', 'row': 2},
            {'name': '下轨', 'values': lower, 'color': 'gray', 'row': 2, 'dash': 'dash'},
            {'name': '收盘', 'values': close, 'color': 'blue', 'row': 2},
        ]

    elif indicator_name == '均线回归':
        ma_period = params.get('ma_period', 60)
        ma = talib.SMA(close, timeperiod=ma_period)
        return [
            {'name': f'MA{ma_period}', 'values': ma, 'color': 'orange', 'row': 2},
            {'name': '收盘', 'values': close, 'color': 'blue', 'row': 2},
        ]

    elif indicator_name == 'OBV底背离':
        vol = df['vol'].values.astype(float)
        obv = talib.OBV(close, vol)
        return [
            {'name': 'OBV', 'values': obv, 'color': 'blue', 'row': 2},
        ]

    elif indicator_name == 'ADX拐头':
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        period = params.get('period', 14)
        adx = talib.ADX(high, low, close, timeperiod=period)
        p_di = talib.PLUS_DI(high, low, close, timeperiod=period)
        m_di = talib.MINUS_DI(high, low, close, timeperiod=period)
        return [
            {'name': 'ADX', 'values': adx, 'color': 'blue', 'row': 2},
            {'name': '+DI', 'values': p_di, 'color': 'green', 'row': 2},
            {'name': '-DI', 'values': m_di, 'color': 'red', 'row': 2},
        ]

    elif indicator_name in ('缩量止跌', '温和放量'):
        return [
            {'name': '成交量', 'values': df['vol'].values.astype(float), 'color': 'blue', 'row': 2, 'type': 'bar'},
        ]

    return [
        {'name': '成交量', 'values': df['vol'].values.astype(float), 'color': 'gray', 'row': 2, 'type': 'bar'},
    ]
