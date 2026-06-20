"""Dual-source (AKShare primary, tushare fallback) data fetching with local CSV caching.

Schema (统一口径，对齐原 tushare 缓存):
    ts_code      str   '600030.SH' / '399975.SZ'
    trade_date   datetime
    open/high/low/close  float
    vol          float  单位：手 (AKShare 股 / 100)
    amount       float  单位：千元 (AKShare 元 / 1000)
"""
import os
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TUSHARE_TOKEN_FILE = os.path.join(BASE_DIR, '.tushare_token')

STOCKS = {
    '601066': '中信建投',
    '600030': '中信证券',
    '600918': '中泰证券',
    '601696': '中银证券',
    '600906': '财达证券',
    '600837': '海通证券',
    '601688': '华泰证券',
    '000776': '广发证券',
    '600999': '招商证券',
    '601788': '光大证券',
}

INDEX_CODE = '399975.SZ'
INDEX_CODE_AK = 'sz399975'  # AKShare 新浪指数源格式

# Hardcoded constituent list (49 stocks) — final fallback when both AKShare & tushare fail
INDEX_CONSTITUENTS = {
    '600909': '华安证券', '600621': '华鑫股份', '600906': '财达证券',
    '601375': '中原证券', '000783': '长江证券', '600999': '招商证券',
    '600030': '中信证券', '601066': '中信建投', '601236': '红塔证券',
    '601198': '东兴证券', '601456': '国联民生', '601211': '国泰海通',
    '601099': '太平洋',  '002736': '国信证券', '002500': '山西证券',
    '601995': '中金公司', '601108': '财通证券', '000728': '国元证券',
    '000166': '申万宏源', '002926': '华西证券', '600369': '西南证券',
    '601059': '信达证券', '000750': '国海证券', '601688': '华泰证券',
    '600155': '华创云信', '601881': '中国银河', '601136': '首创证券',
    '601990': '南京证券', '002673': '西部证券', '601788': '光大证券',
    '000776': '广发证券', '002939': '长城证券', '300059': '东方财富',
    '600958': '东方证券', '601878': '浙商证券', '002797': '第一创业',
    '601377': '兴业证券', '000686': '东北证券', '601901': '方正证券',
    '601555': '东吴证券', '600061': '国投资本', '601162': '天风证券',
    '600109': '国金证券', '600918': '中泰证券', '601696': '中银证券',
    '002945': '华林证券', '002670': '国盛证券', '000712': '锦龙股份',
    '600095': '湘财股份',
}


def _ts_code(code):
    """Convert raw code to ts_code format ('600030' -> '600030.SH')."""
    return f'{code}.SH' if code.startswith('6') else f'{code}.SZ'


def _ak_symbol(ts_code):
    """Convert ts_code to AKShare sina symbol ('600030.SH' -> 'sh600030')."""
    code, market = ts_code.split('.')
    return f'sh{code}' if market == 'SH' else f'sz{code}'


# ---------------------------------------------------------------------------
# Normalization layer
# ---------------------------------------------------------------------------

def _normalize_akshare_stock(df, ts_code):
    """Normalize AKShare stock_zh_a_daily output to unified schema.
    Input columns: date, open, high, low, close, volume, amount, ...
    Output: ts_code, trade_date, open, high, low, close, vol(手), amount(千元)
    """
    if df is None or df.empty:
        return pd.DataFrame()
    required = {'date', 'open', 'high', 'low', 'close', 'volume', 'amount'}
    if not required.issubset(df.columns):
        raise ValueError(f'AKShare stock 缺字段: {set(df.columns) ^ required}')
    out = pd.DataFrame({
        'ts_code': ts_code,
        'trade_date': pd.to_datetime(df['date']),
        'open': df['open'].astype(float),
        'high': df['high'].astype(float),
        'low': df['low'].astype(float),
        'close': df['close'].astype(float),
        'vol': df['volume'].astype(float) / 100.0,        # 股 -> 手
        'amount': df['amount'].astype(float) / 1000.0,    # 元 -> 千元
    })
    return out.sort_values('trade_date').reset_index(drop=True)


def _normalize_akshare_index(df, ts_code):
    """Normalize AKShare stock_zh_index_daily output to unified schema.
    Input columns: date, open, high, low, close, volume
    指数无 amount 列，填 NaN。
    """
    if df is None or df.empty:
        return pd.DataFrame()
    required = {'date', 'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(df.columns):
        raise ValueError(f'AKShare index 缺字段: {set(df.columns) ^ required}')
    out = pd.DataFrame({
        'ts_code': ts_code,
        'trade_date': pd.to_datetime(df['date']),
        'open': df['open'].astype(float),
        'high': df['high'].astype(float),
        'low': df['low'].astype(float),
        'close': df['close'].astype(float),
        'vol': df['volume'].astype(float) / 100.0,        # 股 -> 手
        'amount': float('nan'),                            # 指数新浪源无 amount
    })
    return out.sort_values('trade_date').reset_index(drop=True)


def _normalize_tushare(df):
    """tushare 输出已是目标口径（手/千元），仅做类型规范化。"""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    for c in ('open', 'high', 'low', 'close', 'vol', 'amount'):
        if c in df.columns:
            df[c] = df[c].astype(float)
    return df.sort_values('trade_date').reset_index(drop=True)


# ---------------------------------------------------------------------------
# AKShare source (primary)
# ---------------------------------------------------------------------------

def _ak_get_pro():
    """Lazy import akshare to avoid hard dependency at module import time."""
    import akshare as ak
    return ak


def _akshare_index_daily(start_date=None, end_date=None):
    """Fetch index daily from AKShare. Raises on failure."""
    ak = _ak_get_pro()
    df = ak.stock_zh_index_daily(symbol=INDEX_CODE_AK)
    out = _normalize_akshare_index(df, INDEX_CODE)
    if start_date:
        out = out[out['trade_date'] >= pd.Timestamp(start_date)]
    if end_date:
        out = out[out['trade_date'] <= pd.Timestamp(end_date)]
    if out.empty:
        raise ValueError('AKShare index 返回空')
    return out


def _akshare_stock_daily(ts_code, start_date=None, end_date=None):
    """Fetch stock daily from AKShare (sina source, unadjusted). Raises on failure."""
    ak = _ak_get_pro()
    sym = _ak_symbol(ts_code)
    df = ak.stock_zh_a_daily(symbol=sym, start_date=start_date, end_date=end_date, adjust="")
    out = _normalize_akshare_stock(df, ts_code)
    if out.empty:
        raise ValueError(f'AKShare stock {ts_code} 返回空')
    return out


def _akshare_index_constituents():
    """Fetch 399975 constituents from AKShare. Raises on failure."""
    ak = _ak_get_pro()
    df = ak.index_stock_cons_csindex(symbol='399975')
    if df is None or df.empty:
        raise ValueError('AKShare 成分股返回空')
    # 列: 日期, 指数代码, 指数名称, ..., 成分券代码, 成分券名称, ...
    code_col = '成分券代码' if '成分券代码' in df.columns else df.columns[4]
    name_col = '成分券名称' if '成分券名称' in df.columns else df.columns[5]
    result = [{
        'ts_code': _ts_code(str(r[code_col]).zfill(6)),
        'name': r[name_col],
        'float_mv': None,
        'weight': None,
    } for _, r in df.iterrows()]
    return result


# ---------------------------------------------------------------------------
# tushare source (fallback)
# ---------------------------------------------------------------------------

def _ts_get_pro():
    import tushare as ts
    if not os.path.exists(TUSHARE_TOKEN_FILE):
        raise RuntimeError('tushare token 不存在')
    with open(TUSHARE_TOKEN_FILE) as f:
        token = f.read().strip()
    ts.set_token(token)
    pro = ts.pro_api()
    return pro


def _tushare_index_daily(start_date=None, end_date=None):
    """Fetch index daily from tushare. Raises on failure."""
    pro = _ts_get_pro()
    kwargs = {'ts_code': INDEX_CODE,
              'fields': 'ts_code,trade_date,open,high,low,close,vol,amount'}
    if start_date:
        kwargs['start_date'] = pd.Timestamp(start_date).strftime('%Y%m%d')
    if end_date:
        kwargs['end_date'] = pd.Timestamp(end_date).strftime('%Y%m%d')
    df = pro.index_daily(**kwargs)
    out = _normalize_tushare(df)
    if out.empty:
        raise ValueError('tushare index 返回空')
    return out


def _tushare_stock_daily(ts_code, start_date=None, end_date=None):
    """Fetch stock daily from tushare. Raises on failure."""
    pro = _ts_get_pro()
    kwargs = {'ts_code': ts_code,
              'fields': 'ts_code,trade_date,open,high,low,close,vol,amount'}
    if start_date:
        kwargs['start_date'] = pd.Timestamp(start_date).strftime('%Y%m%d')
    if end_date:
        kwargs['end_date'] = pd.Timestamp(end_date).strftime('%Y%m%d')
    df = pro.daily(**kwargs)
    out = _normalize_tushare(df)
    if out.empty:
        raise ValueError(f'tushare stock {ts_code} 返回空')
    return out


def _tushare_index_constituents():
    """tushare index_weight 在 2000 积分下不可用，直接抛错触发硬编码兜底。"""
    raise RuntimeError('tushare index_weight 2000 积分下不可用')


# ---------------------------------------------------------------------------
# Dual-source dispatchers
# ---------------------------------------------------------------------------

def _try_sources(name, *fetchers):
    """依次尝试每个 fetcher，返回第一个成功的结果；全部失败则抛聚合异常。"""
    last_err = None
    for fetcher in fetchers:
        try:
            return fetcher()
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f'{name} 所有数据源失败: {last_err}')


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _cache_path(name):
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f'{name}.csv')


def _is_fresh(path, max_hours=12):
    if not os.path.exists(path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return (datetime.now() - mtime).total_seconds() < max_hours * 3600


def _cache_usable(path):
    """判断本地缓存是否可直接使用（不走 API）。
    历史K线数据不会变，只要满足以下任一条件即可：
    1. mtime 新鲜（12h内）
    2. 当前不需要盘后更新（非交易日/盘中/缓存已是今天）
    """
    if not os.path.exists(path):
        return False
    if _is_fresh(path):
        return True
    return not _needs_eod_update(path)


# ---------------------------------------------------------------------------
# Holiday / trading-day helpers
# 外部接口: https://api.haoshenqi.top/holiday?date=YYYY-MM-DD
# status: 0 普通工作日 / 1 周末双休 / 2 需要补班的工作日 / 3 法定节假日
# A 股规则：周末一律不交易（即使国家调休补班），仅 status==0 为交易日。
# ---------------------------------------------------------------------------

_HOLIDAY_API = "http://api.haoshenqi.top/holiday"
_holiday_cache = {}  # date_str('YYYY-MM-DD') -> status(int) or None；进程内按天缓存


def _fetch_holiday_status(date_str):
    """查询某天 status。带进程内缓存（同一 date_str 只请求一次）；失败返回 None。
    按天缓存一次：节假日本身不会变，进程内缓存即可；进程重启会重新请求一次。
    """
    if date_str in _holiday_cache:
        return _holiday_cache[date_str]
    status = None
    try:
        import requests
        r = requests.get(_HOLIDAY_API, params={"date": date_str},
                         headers={"content-type": "application/json"},
                         timeout=5)
        if r.status_code == 200:
            data = r.json()
            if data:
                status = data[0].get("status")
    except Exception:
        status = None
    _holiday_cache[date_str] = status
    return status


def is_trading_day(date=None):
    """判断某天是否为 A 股交易日。
    date: datetime / date / str('YYYYMMDD' 或 'YYYY-MM-DD') / None(=今天)
    A 股交易日规则：仅 status==0（普通工作日）为交易日；
    周末（含补班日 status=2）与法定节假日（status=3）一律非交易日。
    API 失败退回 weekday() 判定，保证不比原逻辑更差。
    """
    if date is None:
        d = datetime.now().date()
    elif isinstance(date, datetime):
        d = date.date()
    elif isinstance(date, str):
        d = datetime.strptime(date.replace("-", ""), "%Y%m%d").date()
    else:
        d = date  # datetime.date
    status = _fetch_holiday_status(d.strftime("%Y-%m-%d"))
    if status is None:
        return d.weekday() < 5      # fallback：周一~周五视为交易日
    return status == 0              # 仅普通工作日为交易日；周末补班也算非交易日


def _needs_eod_update(path):
    """盘后更新检查：交易日 15:00 后，缓存最新数据日期 < 今天 → True。
    用于 force=False 时绕过 mtime 新鲜度判定，触发增量拉取当天收盘新数据。
    交易日判定优先走节假日 API（含周末补班/法定节假日），失败退回 weekday()。
    """
    if not os.path.exists(path):
        return False
    now = datetime.now()
    if now.hour < 15:               # 盘中不触发
        return False
    if not is_trading_day(now):     # 非交易日（法定节假日/周末，不含补班）不触发
        return False
    try:
        df = pd.read_csv(path)
        if 'trade_date' not in df.columns or df.empty:
            return False
        last = pd.to_datetime(df['trade_date']).max().date()
        return last < now.date()
    except Exception:
        return False


def _read_csv(path):
    df = pd.read_csv(path)
    if 'trade_date' in df.columns:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df


def _append_and_save(existing, new, path):
    """合并增量数据，去重，按日期排序后落盘。返回合并后的 DataFrame。"""
    if new is None or new.empty:
        return existing.sort_values('trade_date').reset_index(drop=True) if existing is not None else pd.DataFrame()
    if existing is None or existing.empty:
        combined = new
    else:
        combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_index_constituents(force=False):
    """获取 399975.SZ 当前成分股列表。
    主源 AKShare → 备源 tushare → 硬编码兜底。
    返回 list[dict]: {ts_code, name, float_mv, weight}
    缓存到 data/index_constituents.csv，月度新鲜度。
    """
    path = _cache_path('index_constituents')
    if not force and _is_fresh(path, max_hours=24 * 30):
        df = pd.read_csv(path)
        return df.to_dict('records')

    def _ak():
        return _akshare_index_constituents()

    def _ts():
        return _tushare_index_constituents()

    def _hardcoded():
        return [{'ts_code': _ts_code(code), 'name': name, 'float_mv': None, 'weight': None}
                for code, name in INDEX_CONSTITUENTS.items()]

    try:
        result = _try_sources('fetch_index_constituents', _ak, _ts)
    except Exception:
        result = _hardcoded()

    pd.DataFrame(result).to_csv(path, index=False)
    return result


def fetch_index_daily(force=False):
    """获取 399975.SZ 日线。主源 AKShare → 备源 tushare。
    缓存命中：非 force 且 _cache_usable（mtime新鲜 或 非交易日直接用本地）。
    工作日 15:00 后若缓存最新日期 < 今天，自动触发增量拉取。
    """
    path = _cache_path('index_daily')
    if not force and _cache_usable(path):
        return _read_csv(path)

    existing = None
    if os.path.exists(path):
        try:
            existing = _read_csv(path)
        except Exception:
            existing = None

    # 增量模式：缓存存在
    if existing is not None and not existing.empty:
        last_date = existing['trade_date'].max()
        # 缓存最新日期已是今天 → 无需再拉（force=True 也一样）
        if last_date.date() >= datetime.now().date():
            return existing.sort_values('trade_date').reset_index(drop=True)
        start_date = last_date + pd.Timedelta(days=1)

        def _ak():
            return _akshare_index_daily(start_date=start_date)
        def _ts():
            return _tushare_index_daily(start_date=start_date)

        try:
            new = _try_sources('fetch_index_daily(增量)', _ak, _ts)
            return _append_and_save(existing, new, path)
        except Exception:
            return existing.sort_values('trade_date').reset_index(drop=True)

    # 全量拉取
    def _ak():
        return _akshare_index_daily(start_date='2008-01-01')
    def _ts():
        return _tushare_index_daily(start_date='2008-01-01')

    try:
        df = _try_sources('fetch_index_daily(全量)', _ak, _ts)
        df.to_csv(path, index=False)
        return df
    except Exception:
        if existing is not None:
            return existing
        return pd.DataFrame()


def fetch_stock_daily(ts_code, force=False):
    """获取个股日线。主源 AKShare → 备源 tushare。
    缓存命中：非 force 且 _cache_usable（mtime新鲜 或 非交易日直接用本地）。
    工作日 15:00 后若缓存最新日期 < 今天，自动触发增量拉取。
    """
    code = ts_code.split('.')[0]
    path = _cache_path(f'stock_daily_{code}')
    if not force and _cache_usable(path):
        return _read_csv(path)

    existing = None
    if os.path.exists(path):
        try:
            existing = _read_csv(path)
        except Exception:
            existing = None

    if existing is not None and not existing.empty:
        last_date = existing['trade_date'].max()
        if last_date.date() >= datetime.now().date():
            return existing.sort_values('trade_date').reset_index(drop=True)
        start_date = last_date + pd.Timedelta(days=1)

        def _ak():
            return _akshare_stock_daily(ts_code, start_date=start_date.strftime('%Y%m%d'))
        def _ts():
            return _tushare_stock_daily(ts_code, start_date=start_date)

        try:
            new = _try_sources(f'fetch_stock_daily({ts_code}, 增量)', _ak, _ts)
            return _append_and_save(existing, new, path)
        except Exception:
            return existing.sort_values('trade_date').reset_index(drop=True)

    # 全量
    def _ak():
        return _akshare_stock_daily(ts_code, start_date='20080101', end_date=datetime.now().strftime('%Y%m%d'))
    def _ts():
        return _tushare_stock_daily(ts_code, start_date='2008-01-01')

    try:
        df = _try_sources(f'fetch_stock_daily({ts_code}, 全量)', _ak, _ts)
        df.to_csv(path, index=False)
        return df
    except Exception:
        if existing is not None:
            return existing
        return pd.DataFrame()


def fetch_all_stocks_daily(force=False):
    """Fetch all 10 candidate stocks daily data."""
    result = {}
    for code, name in STOCKS.items():
        ts_code = _ts_code(code)
        df = fetch_stock_daily(ts_code, force)
        if not df.empty:
            result[code] = df
    return result


def daily_to_weekly(df):
    """Convert daily DataFrame to weekly by resampling."""
    df = df.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date')
    weekly = df.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'vol': 'sum',
        'amount': 'sum',
    }).dropna().reset_index()
    return weekly
