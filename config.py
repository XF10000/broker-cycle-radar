"""
项目全局常量定义。

所有跨模块共享的数值阈值集中在此处，消除散落在各文件中的魔法数字。
修改某个阈值只需改这里一处，全项目自动生效。
"""

# ============================================================
# 信号过滤阈值（filter_signals / filter_sell_signals 核心参数）
# ============================================================
MA_PERIOD = 250                  # 年线周期（close < MA_PERIOD 才保留买入信号；close > MA_PERIOD 才保留卖出信号）
DECLINE_PCT = 20                 # 最小跌幅%（ref_high 跌幅 > 此值才视为超跌）
DECLINE_FACTOR = 1 - DECLINE_PCT / 100  # 跌幅乘数（0.80），用于绘图"跌X%线"
RISE_PCT = 30                    # 最小涨幅%（ref_low 涨幅 > 此值才视为超涨）
                                   # 依据：9轮行情涨幅全部 > 30%，20%门槛太低放进了太多虚警
SIGNAL_COOLDOWN = 15             # 信号冷却天数（同股同指标两次信号最小间隔）
STATE_WINDOW = 5                 # 上下文保持天数（超跌/超涨状态消失后仍允许发信号的天数）
PRICE_DROP_THRESHOLD = 2.0       # 价格穿越逃逸阈值%（cooldown 内价格跌/涨超此值则放行）

# ============================================================
# 周期检测参数（detect_cycles / _build_stock_ref_highs / _build_stock_ref_lows 共用）
# ============================================================
SMOOTH_WINDOW = 20               # 平滑均线周期
MIN_AMPLITUDE_PCT = 25           # 最小周期涨幅%（低于此值不算一轮行情）
MIN_DURATION_DAYS = 20           # 最小周期持续交易日数
ARGRELEXTREMA_ORDER = 10         # argrelextrema order（左右各 N 根 K 线比较）

# ============================================================
# 回测参数
# ============================================================
SIGNAL_WINDOW_INDEX = 30         # 指数回测：信号提前窗口默认值
SIGNAL_WINDOW_STOCK = 45         # 个股/共振回测：信号提前窗口默认值
LATE_CUTOFF_DAYS = 10            # 信号晚于周期起点多少天内仍算"迟到"（非错过）
RESONANCE_WINDOW_DAYS = 30       # 共振观察窗口：周线信号后 N 天内日线确认

# 买入评分公式权重（综合得分 = 命中率×WEIGHT_HIT_RATE + 精度×WEIGHT_PRECISION - 天数扣分）
SCORE_IDEAL_DAYS = 5             # 理想提前天数
SCORE_DAYS_NORMALIZE = 30        # 天数归一化基准
WEIGHT_DAYS_SCORE = 0.3          # 天数扣分权重
WEIGHT_HIT_RATE = 0.5            # 命中率权重
WEIGHT_PRECISION = 0.2           # 精度权重
SCORE_MAX = WEIGHT_HIT_RATE + WEIGHT_PRECISION  # 理论最大综合得分（0.7）

# 卖出信号回测参数
SELL_SEARCH_WINDOW = 30           # 峰前后搜索窗口（天）：在此范围内找距峰最近的信号
                                   # 依据：与买入侧 SIGNAL_WINDOW_INDEX=30 对称
SELL_HIT_CAPTURE = 0.90            # 命中捕获率阈值：捕获率 ≥ 此值视为"命中"
                                   # 依据：90% 捕获率意味着抓住了绝大部分涨幅

# 卖出评分公式权重（综合得分 = 命中率×WEIGHT_SELL_HIT_RATE + 捕获率×WEIGHT_SELL_CAPTURE + 精度×WEIGHT_SELL_PRECISION）
WEIGHT_SELL_HIT_RATE = 0.5        # 命中率权重（覆盖多少轮，与买入侧一致，最重要）
WEIGHT_SELL_CAPTURE = 0.3         # 捕获率权重（卖出价/实际峰价，次要）
WEIGHT_SELL_PRECISION = 0.2       # 信号有效率权重（与买入侧 WEIGHT_PRECISION 相同）
SCORE_SELL_MAX = WEIGHT_SELL_HIT_RATE + WEIGHT_SELL_CAPTURE + WEIGHT_SELL_PRECISION  # 1.0

# ============================================================
# 数据范围
# ============================================================
DATA_START_DATE = '2008-01-01'   # 全量拉取起始日期
CYCLE_FILTER_DATE = '2010-01-01' # 回测/显示只取此日期之后的周期

# ============================================================
# 缓存参数
# ============================================================
CACHE_FRESH_HOURS = 12           # 本地 CSV 缓存新鲜度（小时）
CACHE_TTL_SECONDS = 3600         # Streamlit @st.cache_data TTL（秒）

# ============================================================
# 热力图评分分级
# ============================================================
HEATMAP_HIT_FAST = 15            # 命中≤15天 → 3分
HEATMAP_LATE_FAST = 5            # 迟到≤5天 → 1分
