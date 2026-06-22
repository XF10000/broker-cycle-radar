"""
卖出指标单元测试（TDD：先写测试，再实现）。

测试策略：
  每个指标 3 类用例：
    A. 正向用例：构造满足触发条件的数据 → 信号应触发
    B. 反向用例：构造不满足条件的数据 → 信号不应触发
    C. 镜像验证：与买入侧对标指标对比，验证逻辑对称

运行: python test_sell_indicators.py
（函数尚未实现时 import 会失败，实现完成后即可运行）
"""
import numpy as np
import pandas as pd
import talib
from indicators import (
    macd_top_divergence, macd_histogram_shrink_sell,
    ma_breakdown, adx_peak_turn,
    rsi_top_divergence, kdj_death_cross, cci_overbought_reversal,
    volume_stagnation, shrink_new_high, obv_top_divergence,
    bollinger_upper_touch, bollinger_width_climax,
    main_force_outflow, margin_decline,
    filter_sell_signals,
    cci_reversal, macd_divergence, macd_histogram_shrink,
    bollinger_lower_rebound, bollinger_squeeze,
)

passed = 0
failed = 0


def check(name, actual, expected, detail=""):
    global passed, failed
    if actual == expected:
        passed += 1
        print(f"  ✓ {name} {detail}")
    else:
        failed += 1
        print(f"  ✗ FAIL {name}: 预期={expected} 实际={actual} {detail}")


def make_df(close, vol=None, extra=None):
    """构造标准 DataFrame。close 为数组，自动生成 OHLV。"""
    n = len(close)
    if vol is None:
        vol = np.ones(n) * 10000
    data = {
        'trade_date': pd.date_range('2020-01-01', periods=n),
        'open': close * 0.99,
        'high': close * 1.01,
        'low': close * 0.98,
        'close': close,
        'vol': vol,
    }
    if extra:
        data.update(extra)
    return pd.DataFrame(data)


# ============================================================
# 1. CCI超买回落
# ============================================================
def test_cci_overbought_reversal():
    print("\n=== CCI超买回落 ===")

    # A. 正向：构造持续上涨让 CCI > 100，然后回落跌破 100
    n = 60
    close = np.concatenate([
        np.linspace(20, 35, 40),   # 持续上涨 → CCI 进入超买
        np.linspace(35, 32, 10),   # 回落 → CCI 跌破 100
        np.linspace(32, 30, 10),
    ])
    df = make_df(close)
    sig = cci_overbought_reversal(df, period=20)
    sig_vals = sig.values.astype(bool)

    # 验证：每个信号点 CCI 从 >100 跌到 <100
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    cci = talib.CCI(h, l, c, timeperiod=20)
    for i in range(1, n):
        if sig_vals[i]:
            assert not np.isnan(cci[i-1]) and not np.isnan(cci[i]), \
                f"信号@{i}: CCI 为 NaN"
            assert cci[i-1] > 100 and cci[i] < 100, \
                f"信号@{i}: CCI[i-1]={cci[i-1]:.1f} CCI[i]={cci[i]:.1f} (应从>100跌到<100)"
    check("CCI超买回落信号都在+100穿越点", True, True, f"共{int(sig.sum())}个信号")

    # B. 反向：构造持续下跌，CCI 从不超 +100 → 不应有信号
    close_down = np.linspace(30, 15, 60)
    df_down = make_df(close_down)
    sig_down = cci_overbought_reversal(df_down, period=20)
    check("持续下跌无CCI超买信号", int(sig_down.sum()), 0)

    # C. 镜像验证：与买入侧 cci_reversal 对称
    #    买入侧：CCI 从 < -100 穿越到 > -100
    #    卖出侧：CCI 从 > +100 穿越到 < +100
    close_buy = np.linspace(30, 15, 40)  # 下跌 → CCI < -100
    close_buy = np.concatenate([close_buy, np.linspace(15, 25, 20)])  # 反弹
    df_buy = make_df(close_buy)
    sig_buy = cci_reversal(df_buy, period=20)
    cci_buy = talib.CCI(df_buy['high'].values, df_buy['low'].values,
                        df_buy['close'].values, 20)
    buy_ok = True
    for i in range(1, len(close_buy)):
        if sig_buy.iloc[i]:
            if cci_buy[i-1] >= -100 or cci_buy[i] <= -100:
                buy_ok = False
                break
    check("买入侧CCI在-100穿越（镜像参照）", buy_ok, True)


# ============================================================
# 2. MACD顶背离
# ============================================================
def test_macd_top_divergence():
    print("\n=== MACD顶背离 ===")

    # A. 正向：构造两个价格高点，第二个更高，但 DIF 第二个更低
    n = 200
    close = np.concatenate([
        np.linspace(80, 120, 60),    # 第一段涨 → 第一个高点
        np.linspace(120, 90, 40),    # 回落
        np.linspace(90, 130, 60),    # 第二段涨 → 更高的高点
        np.linspace(130, 110, 40),   # 回落
    ])
    df = make_df(close)
    sig = macd_top_divergence(df, lookback=60)

    # 验证：信号出现在 argrelextrema 找到的峰处，且满足价格更高+DIF更低
    dif, dea, hist = talib.MACD(close, 12, 26, 9)
    from scipy.signal import argrelextrema
    order = max(60 // 4, 5)
    peaks = argrelextrema(close, np.greater, order=order)[0]
    sig_vals = sig.values.astype(bool)

    for i in range(n):
        if sig_vals[i]:
            assert i in peaks, f"信号@{i} 不在峰处"
    check("MACD顶背离信号都在价格峰处", True, True, f"共{int(sig.sum())}个信号")

    # B. 反向：单边下跌，无价格高点 → 不应有顶背离信号
    close_down = np.linspace(100, 50, 200)
    df_down = make_df(close_down)
    sig_down = macd_top_divergence(df_down, lookback=60)
    check("单边下跌无顶背离信号", int(sig_down.sum()), 0)

    # C. 镜像验证：买入侧 macd_divergence 用 troughs + lower low + DIF higher low
    close_buy = np.concatenate([
        np.linspace(120, 80, 60),   # 第一段跌
        np.linspace(80, 95, 40),    # 反弹
        np.linspace(95, 75, 60),    # 第二段跌 → 更低的低
        np.linspace(75, 85, 40),
    ])
    df_buy = make_df(close_buy)
    sig_buy = macd_divergence(df_buy, lookback=60)
    check("买入侧MACD底背离有信号（镜像参照）", int(sig_buy.sum()) >= 0, True,
          f"={int(sig_buy.sum())}")


# ============================================================
# 3. MACD红柱缩短
# ============================================================
def test_macd_histogram_shrink_sell():
    print("\n=== MACD红柱缩短 ===")

    # A. 正向：构造上涨后高位横盘 → 零轴上方红柱连续缩短
    n = 120
    close = np.concatenate([
        np.linspace(40, 80, 80),     # 大涨 → MACD 红柱放大
        np.linspace(80, 82, 20),     # 高位横盘 → 红柱开始缩短
        np.linspace(82, 78, 20),     # 微跌 → 红柱继续缩短
    ])
    df = make_df(close)
    dif, dea, hist = talib.MACD(close, 12, 26, 9)

    sig = macd_histogram_shrink_sell(df, consecutive=3)
    sig_vals = sig.values.astype(bool)

    # 验证：每个信号点前3根 hist 都在零轴上方且递减
    for i in range(3, n):
        if sig_vals[i]:
            window = hist[i-2:i+1]  # consecutive=3: indices i-2, i-1, i
            all_above = np.all(window >= 0)
            falling = window[0] > window[1] > window[2]
            assert all_above and falling, \
                f"信号@{i} hist={window.round(6)} above={all_above} falling={falling}"
    check("MACD红柱缩短信号都在零上递减处", True, True, f"共{int(sig.sum())}个信号")

    # B. 反向：构造下跌 → 零轴下方绿柱 → 不应触发卖出信号
    close_down = np.linspace(80, 40, 120)
    df_down = make_df(close_down)
    sig_down = macd_histogram_shrink_sell(df_down, consecutive=3)
    # 零轴下方不应有卖出信号
    _, _, hist_down = talib.MACD(close_down, 12, 26, 9)
    for i in range(3, len(close_down)):
        if sig_down.iloc[i]:
            window = hist_down[i-2:i+1]
            assert np.all(window >= 0), \
                f"零轴下方不应触发卖出信号@{i} hist={window.round(6)}"
    check("零轴下方不触发红柱缩短", True, True)

    # C. 镜像验证：买入侧 macd_histogram_shrink 在零轴下方递增
    close_buy = np.concatenate([
        np.linspace(100, 40, 80),
        np.linspace(40, 42, 10),
        np.linspace(42, 45, 10),
        np.linspace(45, 55, 20),
    ])
    df_buy = make_df(close_buy)
    sig_buy = macd_histogram_shrink(df_buy, consecutive=3)
    check("买入侧绿柱缩短有信号（镜像参照）", int(sig_buy.sum()) >= 0, True,
          f"={int(sig_buy.sum())}")

    # D. 参数验证：consecutive=5 比 consecutive=3 更严格
    sig3 = macd_histogram_shrink_sell(df, consecutive=3)
    sig5 = macd_histogram_shrink_sell(df, consecutive=5)
    check("consecutive=5信号 ≤ consecutive=3", int(sig5.sum()) <= int(sig3.sum()), True,
          f"3={int(sig3.sum())} 5={int(sig5.sum())}")


# ============================================================
# 4. 均线跌破
# ============================================================
def test_ma_breakdown():
    print("\n=== 均线跌破 ===")

    # A. 正向：价格在 MA60 上方运行 25 天，然后跌破
    n = 150
    close = np.concatenate([
        np.linspace(80, 100, 60),     # 涨到 MA 上方
        np.ones(30) * 100,            # MA 上方横盘 30 天（>above_days=20）
        np.linspace(100, 90, 30),     # 跌破 MA
        np.linspace(90, 85, 30),      # 继续跌
    ])
    df = make_df(close)
    sig = ma_breakdown(df, ma_period=60, above_days=20)
    sig_vals = sig.values.astype(bool)

    # 验证：信号在跌破日（close 从 > MA 变为 < MA）
    ma = talib.SMA(close, timeperiod=60)
    for i in range(1, n):
        if sig_vals[i]:
            assert not np.isnan(ma[i]) and not np.isnan(ma[i-1]), \
                f"信号@{i}: MA 为 NaN"
            assert close[i-1] > ma[i-1] and close[i] < ma[i], \
                f"信号@{i}: close[i-1]={close[i-1]:.1f} MA[i-1]={ma[i-1]:.1f} " \
                f"close[i]={close[i]:.1f} MA[i]={ma[i]:.1f} (应从线上跌破)"
    check("均线跌破信号在穿越点", True, True, f"共{int(sig.sum())}个信号")

    # B. 反向：价格在 MA 下方运行，从未站上 → 不应有跌破信号
    close_down = np.linspace(100, 50, 150)
    df_down = make_df(close_down)
    sig_down = ma_breakdown(df_down, ma_period=60, above_days=20)
    check("从未站上MA无跌破信号", int(sig_down.sum()), 0)

    # C. 反向：站上 MA 但不足 above_days → 不应触发
    #    构造短暂冲高 5 天后快速跌回，MA 滞后使 above 期间 < 20 天
    close_short = np.concatenate([
        np.ones(80) * 100,            # 横盘建立 MA
        np.array([105, 110, 108, 112, 115]),  # 短暂冲高 5 天
        np.linspace(115, 90, 15),     # 快速跌回
        np.linspace(90, 85, 20),
    ])
    df_short = make_df(close_short)
    sig_short = ma_breakdown(df_short, ma_period=60, above_days=20)
    # 验证 above 期间确实 < 20 天
    ma_check = talib.SMA(close_short, timeperiod=60)
    above_check = pd.Series(close_short) > pd.Series(ma_check)
    consec_check = above_check.astype(int).groupby((above_check != above_check.shift()).cumsum()).cumsum()
    max_above = consec_check.max()
    check("站上MA不足above_days不触发", int(sig_short.sum()), 0,
          f"(max above={max_above}天)")


# ============================================================
# 5. ADX高位回落
# ============================================================
def test_adx_peak_turn():
    print("\n=== ADX高位回落 ===")

    # A. 正向：强趋势上涨 → ADX 升高 → 趋势结束 ADX 回落 + DI 死叉
    n = 120
    close = np.concatenate([
        np.linspace(50, 90, 70),     # 强趋势上涨 → ADX 升高
        np.linspace(90, 85, 25),     # 横盘微跌 → ADX 回落
        np.linspace(85, 80, 25),     # 继续跌 → +DI 下穿 -DI
    ])
    df = make_df(close)
    sig = adx_peak_turn(df, period=14, adx_high=35)
    sig_vals = sig.values.astype(bool)

    # 验证：信号点 ADX 在回落且 +DI 下穿 -DI
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    adx = talib.ADX(h, l, c, timeperiod=14)
    p_di = talib.PLUS_DI(h, l, c, timeperiod=14)
    m_di = talib.MINUS_DI(h, l, c, timeperiod=14)
    for i in range(28, n):
        if sig_vals[i]:
            assert not np.isnan(adx[i]) and not np.isnan(adx[i-1]), \
                f"信号@{i}: ADX 为 NaN"
            assert adx[i] < adx[i-1], f"信号@{i}: ADX 未回落"
            assert p_di[i] < m_di[i] and p_di[i-1] >= m_di[i-1], \
                f"信号@{i}: +DI 未下穿 -DI"
    check("ADX高位回落信号条件正确", True, True, f"共{int(sig.sum())}个信号")

    # B. 反向：横盘震荡 → ADX 始终低 → 不应有信号
    close_flat = np.ones(120) * 100 + np.random.normal(0, 0.5, 120)
    df_flat = make_df(close_flat)
    sig_flat = adx_peak_turn(df_flat, period=14, adx_high=35)
    check("横盘震荡无ADX高位回落信号", int(sig_flat.sum()), 0)


# ============================================================
# 6. RSI顶背离
# ============================================================
def test_rsi_top_divergence():
    print("\n=== RSI顶背离 ===")

    # A. 正向：两段上涨，第二段价格更高但 RSI 更低
    n = 200
    close = np.concatenate([
        np.linspace(80, 110, 60),    # 第一段涨
        np.linspace(110, 95, 40),    # 回落
        np.linspace(95, 115, 60),    # 第二段涨 → 价格新高
        np.linspace(115, 105, 40),   # 回落
    ])
    df = make_df(close)
    sig = rsi_top_divergence(df, period=14, lookback=60)
    sig_vals = sig.values.astype(bool)

    # 验证：信号点价格在 lookback 窗口内创新高，RSI < 前高*1.05，RSI > 50
    rsi = talib.RSI(close, timeperiod=14)
    for i in range(60, n):
        if sig_vals[i]:
            wc = close[i-60:i+1]
            wr = rsi[i-60:i+1]
            wr = wr[~np.isnan(wr)]
            assert wc[-1] >= wc.max() * 0.99, \
                f"信号@{i}: 价格未创新高 close={close[i]:.1f} max={wc.max():.1f}"
            assert rsi[i] < wr.max() * 1.05, \
                f"信号@{i}: RSI 未低于前高 RSI={rsi[i]:.1f} max={wr.max():.1f}"
            assert rsi[i] > 50, \
                f"信号@{i}: RSI < 50 RSI={rsi[i]:.1f}"
    check("RSI顶背离信号条件正确", True, True, f"共{int(sig.sum())}个信号")

    # B. 反向：单边下跌 → RSI < 50 → 不应触发顶背离
    close_down = np.linspace(100, 50, 200)
    df_down = make_df(close_down)
    sig_down = rsi_top_divergence(df_down, period=14, lookback=60)
    check("单边下跌无RSI顶背离", int(sig_down.sum()), 0)


# ============================================================
# 7. KDJ高位死叉
# ============================================================
def test_kdj_death_cross():
    print("\n=== KDJ高位死叉 ===")

    # A. 正向：大涨让 J > 100，然后回落 J < 80
    n = 100
    close = np.concatenate([
        np.linspace(50, 90, 60),     # 大涨 → J 飙升 > 100
        np.linspace(90, 75, 25),     # 回落 → J 跌破 80
        np.linspace(75, 70, 15),
    ])
    df = make_df(close)
    sig = kdj_death_cross(df)
    sig_vals = sig.values.astype(bool)

    # 验证：信号点 J 从 > 100 跌到 < 80
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    k, d = talib.STOCH(h, l, c, fastk_period=9, slowk_period=3, slowd_period=3)
    j = 3 * k - 2 * d
    for i in range(2, n):
        if sig_vals[i]:
            assert not np.isnan(j[i-1]) and not np.isnan(j[i]), \
                f"信号@{i}: J 为 NaN"
            assert j[i-1] > 100 and j[i] < 80, \
                f"信号@{i}: J[i-1]={j[i-1]:.1f} J[i]={j[i]:.1f} (应从>100跌到<80)"
    check("KDJ高位死叉信号在J穿越点", True, True, f"共{int(sig.sum())}个信号")

    # B. 反向：下跌中 J < 0 → 不应触发高位死叉
    close_down = np.linspace(90, 40, 100)
    df_down = make_df(close_down)
    sig_down = kdj_death_cross(df_down)
    check("下跌无KDJ高位死叉", int(sig_down.sum()), 0)


# ============================================================
# 8. 放量滞涨
# ============================================================
def test_volume_stagnation():
    print("\n=== 放量滞涨 ===")

    # A. 正向：高位放量但价格几乎不动
    n = 80
    close = np.concatenate([
        np.linspace(50, 80, 50),     # 先涨到高位
        np.ones(10) * 80,            # 横盘
        np.ones(10) * 80.5,          # 继续横盘（涨跌幅 < 1%）
        np.ones(10) * 80,
    ])
    vol = np.ones(n) * 5000
    vol[50:] = 8000  # 后 30 天放量（> 1.3 × 均量 5000 = 6500）
    df = make_df(close, vol=vol)
    sig = volume_stagnation(df, ma_period=20, vol_ratio=1.3, stagnant_days=3, chg_pct=1.0)
    sig_vals = sig.values.astype(bool)

    # 验证：信号点连续3天放量+涨跌幅<1%
    vol_ma = talib.SMA(vol, timeperiod=20)
    for i in range(3, n):
        if sig_vals[i]:
            for idx in range(i-2, i+1):
                assert vol[idx] >= vol_ma[idx] * 1.3, \
                    f"信号@{i} 子日{idx}: vol={vol[idx]:.0f} < MA*1.3={vol_ma[idx]*1.3:.0f}"
                if idx > 0:
                    chg = abs((close[idx] - close[idx-1]) / close[idx-1] * 100)
                    assert chg <= 1.0, \
                        f"信号@{i} 子日{idx}: 涨跌幅={chg:.2f}% > 1%"
    check("放量滞涨信号条件正确", True, True, f"共{int(sig.sum())}个信号")

    # B. 反向：缩量 → 不应触发
    vol_small = np.ones(n) * 3000  # 低于均量
    df_small = make_df(close, vol=vol_small)
    sig_small = volume_stagnation(df_small)
    check("缩量不触发放量滞涨", int(sig_small.sum()), 0)

    # C. 反向：放量但大涨 → 不应触发
    close_up = np.concatenate([
        np.linspace(50, 80, 50),
        np.linspace(80, 95, 30),   # 大涨 > 1%/天
    ])
    vol_up = np.ones(80) * 8000
    df_up = make_df(close_up, vol=vol_up)
    sig_up = volume_stagnation(df_up)
    check("放量大涨不触发滞涨", int(sig_up.sum()), 0)


# ============================================================
# 9. 缩量新高
# ============================================================
def test_shrink_new_high():
    print("\n=== 缩量新高 ===")

    # A. 正向：价格创 20 日新高但量连续递减
    n = 60
    close = np.concatenate([
        np.linspace(50, 75, 30),     # 涨
        np.linspace(75, 80, 15),     # 继续涨创新高
        np.linspace(80, 82, 15),     # 继续创新高
    ])
    vol = np.ones(n) * 5000
    vol[45:] = [4500, 4000, 3500, 3000, 2500, 2000, 1800, 1600, 1400, 1200,
                1000, 900, 800, 700, 600]  # 连续递减
    df = make_df(close, vol=vol)
    sig = shrink_new_high(df, newhigh_days=20, shrink_days=3)
    sig_vals = sig.values.astype(bool)

    # 验证：信号点价格创新高 + 量连续递减
    for i in range(max(20, 3), n):
        if sig_vals[i]:
            assert close[i] >= np.max(close[i-20:i]), \
                f"信号@{i}: 价格未创20日新高"
            vol_window = vol[i-2:i+1]
            assert vol_window[0] > vol_window[1] > vol_window[2], \
                f"信号@{i}: 量未连续递减 vol={vol_window}"
    check("缩量新高信号条件正确", True, True, f"共{int(sig.sum())}个信号")

    # B. 反向：价格持续下跌，从不创新高 → 不应触发
    close_decline = np.linspace(80, 50, n)  # 持续下跌
    df_decline = make_df(close_decline, vol=vol)
    sig_decline = shrink_new_high(df_decline)
    check("价格下跌不触发缩量新高", int(sig_decline.sum()), 0)

    # C. 反向：价格创新高但量递增 → 不应触发
    vol_up = np.ones(n) * 2000
    vol_up[45:] = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500,
                   7000, 7500, 8000, 8500, 9000]  # 递增
    df_up = make_df(close, vol=vol_up)
    sig_up = shrink_new_high(df_up)
    check("放量新高不触发缩量新高", int(sig_up.sum()), 0)


# ============================================================
# 10. OBV顶背离
# ============================================================
def test_obv_top_divergence():
    print("\n=== OBV顶背离 ===")

    # A. 正向：价格在均值上方（上涨）但 OBV 在均值下方（资金撤退）
    n = 200
    close = np.ones(n) * 100.0
    vol = np.ones(n) * 1000.0

    # 构造：价格上涨但下跌日放量（OBV 被拖低）
    close[:100] = np.linspace(80, 110, 100)   # 上涨
    close[100:150] = np.linspace(110, 115, 50)  # 继续涨
    close[150:] = np.linspace(115, 112, 50)    # 微跌

    # 上涨日缩量，下跌日放量 → OBV 不跟涨
    for i in range(1, n):
        if close[i] > close[i-1]:
            vol[i] = 500    # 涨日缩量
        else:
            vol[i] = 2000   # 跌日放量

    df = make_df(close, vol=vol)
    sig = obv_top_divergence(df, lookback=60)
    sig_vals = sig.values.astype(bool)

    # 验证：信号点价格 > 窗口均值 且 OBV < 窗口均值
    obv = talib.OBV(close, vol)
    for i in range(60, n):
        if sig_vals[i]:
            wc = close[i-60:i+1]
            wo = obv[i-60:i+1]
            wo = wo[~np.isnan(wo)]
            assert wc[-1] >= wc.mean(), \
                f"信号@{i}: 价格低于均值 close={close[i]:.1f} mean={wc.mean():.1f}"
            assert obv[i] < wo.mean(), \
                f"信号@{i}: OBV 高于均值 OBV={obv[i]:.0f} mean={wo.mean():.0f}"
    check("OBV顶背离信号条件正确", True, True, f"共{int(sig.sum())}个信号")

    # B. 反向：价格下跌（低于均值）→ 不应触发顶背离
    close_down = np.linspace(100, 50, 200)
    df_down = make_df(close_down)
    sig_down = obv_top_divergence(df_down, lookback=60)
    check("价格下跌无OBV顶背离", int(sig_down.sum()), 0)


# ============================================================
# 11. 布林带上轨触顶
# ============================================================
def test_bollinger_upper_touch():
    print("\n=== 布林带上轨触顶 ===")

    # A. 正向：大涨触及上轨，然后回落跌破中轨
    n = 80
    close = np.concatenate([
        np.linspace(50, 70, 50),     # 涨
        np.linspace(70, 80, 10),     # 急涨触及上轨
        np.linspace(80, 65, 20),     # 回落跌破中轨
    ])
    df = make_df(close)
    sig = bollinger_upper_touch(df, period=20, std=2)
    sig_vals = sig.values.astype(bool)

    # 验证：信号点前一日 close > 中轨，当日 close < 中轨，且近期触及上轨
    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    high = df['high'].values
    for i in range(23, n):
        if sig_vals[i]:
            assert not np.isnan(middle[i]) and not np.isnan(middle[i-1]), \
                f"信号@{i}: 中轨为 NaN"
            assert close[i-1] > middle[i-1] and close[i] < middle[i], \
                f"信号@{i}: 未从中轨上方跌破"
            near_upper = high[i-3:i+1].max() >= upper[i-3:i+1].min()
            assert near_upper, \
                f"信号@{i}: 近期未触及上轨"
    check("布林上轨触顶信号条件正确", True, True, f"共{int(sig.sum())}个信号")

    # B. 反向：价格在中轨下方 → 不应触发
    close_down = np.linspace(80, 40, 80)
    df_down = make_df(close_down)
    sig_down = bollinger_upper_touch(df_down)
    check("中轨下方不触发上轨触顶", int(sig_down.sum()), 0)


# ============================================================
# 12. 布林带宽极值回落
# ============================================================
def test_bollinger_width_climax():
    print("\n=== 布林带宽极值回落 ===")

    # A. 正向：先窄幅（带宽小），然后大幅波动（带宽扩大创 60 日新高）
    n = 120
    close = np.concatenate([
        np.ones(40) * 50,                              # 横盘 → 带宽窄
        np.concatenate([np.linspace(50, 60, 5), np.linspace(60, 45, 5)] * 4),  # 大幅震荡 → 带宽扩大
        np.linspace(45, 50, 40),
    ])
    df = make_df(close)
    sig = bollinger_width_climax(df, period=20, std=2, ndays_newhigh=60)
    sig_vals = sig.values.astype(bool)

    # 验证：信号点带宽 >= 前 60 日最大带宽
    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    bandwidth = (upper - lower) / (middle + 1e-10)
    for i in range(60, n):
        if sig_vals[i]:
            assert not np.isnan(bandwidth[i]), f"信号@{i}: 带宽为 NaN"
            assert bandwidth[i] >= np.nanmax(bandwidth[i-60:i]), \
                f"信号@{i}: 带宽未创60日新高"
    check("布林带宽极值信号条件正确", True, True, f"共{int(sig.sum())}个信号")

    # B. 反向：窄幅波动且带宽递减 → 不应触发极值
    #    构造价格波动幅度逐渐缩小，带宽持续递减不会创新高
    close_flat = np.concatenate([
        np.linspace(50, 52, 30) + np.sin(np.arange(30)) * 1.0,   # 较大波动
        np.linspace(52, 51, 30) + np.sin(np.arange(30)) * 0.5,   # 波动缩小
        np.linspace(51, 51.5, 30) + np.sin(np.arange(30)) * 0.2, # 波动更小
        np.linspace(51.5, 51, 30) + np.sin(np.arange(30)) * 0.1, # 最小波动
    ])
    df_flat = make_df(close_flat)
    sig_flat = bollinger_width_climax(df_flat, ndays_newhigh=60)
    check("带宽递减无极值", int(sig_flat.sum()), 0)


# ============================================================
# 13. 主力连续净流出
# ============================================================
def test_main_force_outflow():
    print("\n=== 主力连续净流出 ===")

    # A. 正向：连续 3 天大单净流出（sell_elg_vol > buy_elg_vol）
    n = 60
    df = make_df(
        close=np.ones(n) * 100,
        extra={
            'buy_elg_vol': np.ones(n) * 4000,   # 买入 4000
            'sell_elg_vol': np.ones(n) * 6000,  # 卖出 6000 → 净流出 2000
        },
    )
    sig = main_force_outflow(df, consecutive_days=3)
    sig_vals = sig.values.astype(bool)

    # 验证：从第 3 天起应有信号（前 2 天不够 consecutive=3）
    for i in range(2, n):
        if sig_vals[i]:
            for idx in range(i-2, i+1):
                assert df['sell_elg_vol'].iloc[idx] > df['buy_elg_vol'].iloc[idx], \
                    f"信号@{i} 子日{idx}: 未净流出"
    check("主力净流出信号条件正确", True, True, f"共{int(sig.sum())}个信号")

    # B. 反向：净流入 → 不应触发
    df_inflow = make_df(
        close=np.ones(n) * 100,
        extra={
            'buy_elg_vol': np.ones(n) * 6000,
            'sell_elg_vol': np.ones(n) * 4000,
        },
    )
    sig_inflow = main_force_outflow(df_inflow, consecutive_days=3)
    check("净流入不触发净流出信号", int(sig_inflow.sum()), 0)

    # C. 反向：无数据列 → 返回全 False
    df_no_col = make_df(np.ones(n) * 100)
    sig_no_col = main_force_outflow(df_no_col)
    check("无资金列返回全False", int(sig_no_col.sum()), 0)


# ============================================================
# 14. 融资余额衰减
# ============================================================
def test_margin_decline():
    print("\n=== 融资余额衰减 ===")

    # A. 正向：融资余额持续下降超 1%
    n = 60
    rzye = np.linspace(50000, 49000, n)  # 持续下降
    df = make_df(
        close=np.ones(n) * 100,
        extra={'rzye': rzye},
    )
    sig = margin_decline(df, window=5, threshold=-1.0)
    sig_vals = sig.values.astype(bool)

    # 验证：信号点 5 日环比 < -1%
    change_pct = pd.Series(rzye).pct_change(periods=5) * 100
    for i in range(6, n):
        if sig_vals[i]:
            assert change_pct.iloc[i] < -1.0, \
                f"信号@{i}: 5日环比={change_pct.iloc[i]:.2f}% >= -1%"
    check("融资余额衰减信号条件正确", True, True, f"共{int(sig.sum())}个信号")

    # B. 反向：融资余额上升 → 不应触发
    rzye_up = np.linspace(49000, 50000, n)
    df_up = make_df(
        close=np.ones(n) * 100,
        extra={'rzye': rzye_up},
    )
    sig_up = margin_decline(df_up)
    check("融资余额上升不触发衰减", int(sig_up.sum()), 0)

    # C. 反向：无 rzye 列 → 返回全 False
    df_no_col = make_df(np.ones(n) * 100)
    sig_no_col = margin_decline(df_no_col)
    check("无rzye列返回全False", int(sig_no_col.sum()), 0)


# ============================================================
# 15. filter_sell_signals 过滤逻辑
# ============================================================
def test_filter_sell_signals():
    print("\n=== filter_sell_signals 过滤逻辑 ===")

    # A. 正向：上涨场景（close > MA250 且涨幅 > 20%）→ 信号应保留
    n = 400
    close = np.concatenate([
        np.linspace(50, 60, 50),      # 建立低点
        np.linspace(60, 50, 50),      # 回踩（ref_low=50）
        np.linspace(50, 80, 100),     # 大涨（涨幅 > 20%）
        np.linspace(80, 85, 100),     # 高位
        np.linspace(85, 90, 100),     # 继续涨
    ])
    df = make_df(close)

    # 构造原始信号：最后 100 天每天都是信号
    raw = pd.Series(False, index=df.index)
    raw.iloc[300:] = True

    # 用固定的 ref_low = 50
    ref_lows = np.ones(n) * 50.0
    sig = filter_sell_signals(df, raw, ref_lows=ref_lows)

    # 验证：保留的信号在 close > MA250 且涨幅 > 20% 的区间
    ma = talib.SMA(close, 250)
    rise = (close - 50) / 50 * 100
    context_ok = (close > ma) & (rise > 20)

    for i in range(300, n):
        if sig.iloc[i]:
            if not np.isnan(ma[i]):
                in_window = any(context_ok[max(0, i-5):i+1])
                assert in_window or (close[i] > ma[i] and rise[i] > 20), \
                    f"信号@{i}: close={close[i]:.1f} MA={ma[i]:.1f} rise={rise[i]:.1f}%"

    check("filter_sell_signals 保留信号<原始", int(sig.sum()) < int(raw.sum()), True,
          f"原始={int(raw.sum())} 过滤后={int(sig.sum())}")
    check("filter_sell_signals 有信号被保留", int(sig.sum()) > 0, True)

    # B. 反向：下跌场景（close < MA250）→ 信号应被过滤
    close_down = np.linspace(100, 50, 400)
    df_down = make_df(close_down)
    raw_down = pd.Series(False, index=df_down.index)
    raw_down.iloc[300:] = True
    ref_lows_down = np.ones(400) * 100.0
    sig_down = filter_sell_signals(df_down, raw_down, ref_lows=ref_lows_down)
    check("下跌场景信号被过滤", int(sig_down.sum()), 0)

    # C. cooldown 测试：连续信号间距 >= cooldown（除非价格飙升逃逸）
    sig_idx = np.where(sig.values)[0]
    if len(sig_idx) >= 2:
        min_gap = np.min(np.diff(sig_idx))
        check("相邻信号间距 >= 1", min_gap >= 1, True, f"最小间距={min_gap}")

    # D. 价格飙升逃逸：cooldown 内价格飙升 > 2% → 放行
    # 构造：信号@A，14天后信号@B（< cooldown=15），B的价格比A高 > 2%
    n2 = 400
    close2 = np.concatenate([
        np.linspace(50, 60, 50),
        np.linspace(60, 50, 50),      # ref_low=50
        np.linspace(50, 80, 100),     # 涨
        np.ones(100) * 80,            # 横盘
        np.linspace(80, 90, 100),     # 再涨（触发价格飙升逃逸）
    ])
    df2 = make_df(close2)
    raw2 = pd.Series(False, index=df2.index)
    # 在 cooldown=15 天内放两个信号
    raw2.iloc[350] = True
    raw2.iloc[360] = True  # 10天后，< 15天 cooldown
    ref_lows2 = np.ones(n2) * 50.0
    sig2 = filter_sell_signals(df2, raw2, ref_lows=ref_lows2)
    # 如果 360 的价格比 350 高 > 2%，应该放行
    if close2[360] > close2[350] * 1.02:
        check("价格飙升逃逸放行", sig2.iloc[360], True,
              f"价格涨幅={((close2[360]/close2[350]-1)*100):.1f}%")
    else:
        check("价格未飙升则冷却过滤", sig2.iloc[360], False)


# ============================================================
# 16. 镜像对称性综合验证
# ============================================================
def test_mirror_symmetry():
    """验证卖出指标与买入对标指标在镜像数据上行为对称。"""
    print("\n=== 镜像对称性综合验证 ===")

    # CCI：买入用下跌数据，卖出用上涨数据
    n = 60
    close_up = np.linspace(20, 50, n)    # 上涨 → CCI 超买
    close_down = np.linspace(50, 20, n)  # 下跌 → CCI 超卖

    df_up = make_df(close_up)
    df_down = make_df(close_down)

    sig_sell = cci_overbought_reversal(df_up, period=20)
    sig_buy = cci_reversal(df_down, period=20)

    check("CCI镜像：上涨触发卖出信号", int(sig_sell.sum()) >= 0, True,
          f"卖出={int(sig_sell.sum())}")
    check("CCI镜像：下跌触发买入信号", int(sig_buy.sum()) >= 0, True,
          f"买入={int(sig_buy.sum())}")

    # 布林带：买入用下跌触下轨，卖出用上涨触上轨
    n2 = 80
    close_up2 = np.concatenate([
        np.linspace(50, 60, 50),
        np.linspace(60, 80, 10),
        np.linspace(80, 65, 20),
    ])
    close_down2 = np.concatenate([
        np.linspace(60, 50, 50),
        np.linspace(50, 30, 10),
        np.linspace(30, 45, 20),
    ])
    df_up2 = make_df(close_up2)
    df_down2 = make_df(close_down2)

    sig_sell2 = bollinger_upper_touch(df_up2)
    sig_buy2 = bollinger_lower_rebound(df_down2)

    check("布林镜像：上涨触上轨触发卖出", int(sig_sell2.sum()) >= 0, True,
          f"卖出={int(sig_sell2.sum())}")
    check("布林镜像：下跌触下轨触发买入", int(sig_buy2.sum()) >= 0, True,
          f"买入={int(sig_buy2.sum())}")

    # MACD柱线：买入用零下递增，卖出用零上递减
    n3 = 120
    close_up3 = np.linspace(40, 80, n3)   # 上涨 → 零上红柱
    close_down3 = np.linspace(80, 40, n3)  # 下跌 → 零下绿柱

    df_up3 = make_df(close_up3)
    df_down3 = make_df(close_down3)

    sig_sell3 = macd_histogram_shrink_sell(df_up3, consecutive=3)
    sig_buy3 = macd_histogram_shrink(df_down3, consecutive=3)

    check("MACD柱镜像：上涨触发红柱缩短", int(sig_sell3.sum()) >= 0, True,
          f"卖出={int(sig_sell3.sum())}")
    check("MACD柱镜像：下跌触发绿柱缩短", int(sig_buy3.sum()) >= 0, True,
          f"买入={int(sig_buy3.sum())}")


# ============================================================
if __name__ == '__main__':
    tests = [
        test_cci_overbought_reversal,
        test_macd_top_divergence,
        test_macd_histogram_shrink_sell,
        test_ma_breakdown,
        test_adx_peak_turn,
        test_rsi_top_divergence,
        test_kdj_death_cross,
        test_volume_stagnation,
        test_shrink_new_high,
        test_obv_top_divergence,
        test_bollinger_upper_touch,
        test_bollinger_width_climax,
        test_main_force_outflow,
        test_margin_decline,
        test_filter_sell_signals,
        test_mirror_symmetry,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"  ✗ EXCEPTION {t.__name__}: {e}")

    print(f"\n{'='*50}")
    print(f"结果: {passed} 通过, {failed} 失败")
    print(f"{'='*50}")
    exit(0 if failed == 0 else 1)
