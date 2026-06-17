"""Shared tushare data fetching with local CSV caching."""
import os
import pandas as pd
import tushare as ts
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TOKEN_FILE = os.path.join(BASE_DIR, '.tushare_token')

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

# Hardcoded constituent list (49 stocks) — fallback when tushare index APIs unavailable
# Source: user-provided 399975.SZ index constituent list as of 2026-06
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
    """Convert raw code to tushare ts_code format."""
    return f'{code}.SH' if code.startswith('6') else f'{code}.SZ'


def fetch_index_constituents(force=False):
    """
    Fetch 399975.SZ current constituent stocks.
    Tries tushare APIs first; falls back to hardcoded INDEX_CONSTITUENTS list.
    Returns list of dicts with ts_code, name (float_mv/weight may be None).
    Cached to data/index_constituents.csv
    """
    path = _cache_path('index_constituents')
    if not force and _is_fresh(path, max_hours=24 * 30):
        df = pd.read_csv(path)
        return df.to_dict('records')

    # Attempt tushare APIs
    pro = _get_pro()
    for api_name, api_call in [
        ('index_weight', lambda: pro.index_weight(
            index_code=INDEX_CODE, trade_date=datetime.now().strftime('%Y%m%d'),
            fields='index_code,con_code,trade_date,weight')),
        ('index_member', lambda: pro.index_member(
            index_code=INDEX_CODE,
            fields='index_code,con_code,con_name,in_date,out_date,is_new')),
    ]:
        try:
            df = api_call()
            if df is not None and not df.empty:
                result = []
                for _, row in df.iterrows():
                    code = row.get('con_code', '')
                    result.append({
                        'ts_code': code,
                        'name': row.get('con_name', INDEX_CONSTITUENTS.get(code.split('.')[0], '')),
                        'float_mv': None,
                        'weight': row.get('weight'),
                    })
                pd.DataFrame(result).to_csv(path, index=False)
                return result
        except Exception:
            continue

    # Fallback: hardcoded list
    result = [{'ts_code': _ts_code(code), 'name': name, 'float_mv': None, 'weight': None}
              for code, name in INDEX_CONSTITUENTS.items()]
    pd.DataFrame(result).to_csv(path, index=False)
    return result


def _get_pro():
    with open(TOKEN_FILE) as f:
        token = f.read().strip()
    ts.set_token(token)
    pro = ts.pro_api()
    pro.__timeout = 10  # 10-second timeout per API call
    return pro


def _cache_path(name):
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f'{name}.csv')


def _is_fresh(path, max_hours=12):
    if not os.path.exists(path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return (datetime.now() - mtime).total_seconds() < max_hours * 3600


def _read_csv(path):
    df = pd.read_csv(path)
    if 'trade_date' in df.columns:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df


def fetch_index_daily(force=False):
    """Fetch 399975.SZ daily K-line. Cached to data/index_daily.csv"""
    path = _cache_path('index_daily')
    if not force and _is_fresh(path):
        return _read_csv(path)

    pro = _get_pro()
    # Incremental: if cache exists, only fetch new data
    if force and os.path.exists(path):
        try:
            existing = pd.read_csv(path)
            existing['trade_date'] = pd.to_datetime(existing['trade_date'])
            last_date = existing['trade_date'].max().strftime('%Y%m%d')
            df = pro.index_daily(ts_code=INDEX_CODE, start_date=last_date,
                                 fields='ts_code,trade_date,open,high,low,close,vol,amount')
            if df is not None and not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                combined = pd.concat([existing, df], ignore_index=True)
                combined = combined.drop_duplicates(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
                combined.to_csv(path, index=False)
                return combined
        except Exception:
            pass  # API failed, fall through to full fetch
        return existing  # API returned nothing new, use cached

    # Full fetch
    try:
        df = pro.index_daily(ts_code=INDEX_CODE, start_date='20080101',
                             fields='ts_code,trade_date,open,high,low,close,vol,amount')
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        df.to_csv(path, index=False)
        return df
    except Exception:
        # Fallback: return cached if available
        if os.path.exists(path):
            existing = pd.read_csv(path)
            existing['trade_date'] = pd.to_datetime(existing['trade_date'])
            return existing
        return pd.DataFrame()


def fetch_stock_daily(ts_code, force=False):
    """Fetch individual stock daily K-line."""
    code = ts_code.split('.')[0]
    path = _cache_path(f'stock_daily_{code}')
    if not force and _is_fresh(path):
        return _read_csv(path)

    pro = _get_pro()
    # Incremental: if cache exists, only fetch new data
    if force and os.path.exists(path):
        try:
            existing = pd.read_csv(path)
            existing['trade_date'] = pd.to_datetime(existing['trade_date'])
            last_date = existing['trade_date'].max().strftime('%Y%m%d')
            df = pro.daily(ts_code=ts_code, start_date=last_date,
                           fields='ts_code,trade_date,open,high,low,close,vol,amount')
            if df is not None and not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                combined = pd.concat([existing, df], ignore_index=True)
                combined = combined.drop_duplicates(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
                combined.to_csv(path, index=False)
                return combined
        except Exception:
            pass  # API failed, use cached as-is
        return existing

    # Full fetch
    try:
        df = pro.daily(ts_code=ts_code, start_date='20080101',
                       fields='ts_code,trade_date,open,high,low,close,vol,amount')
        if df is not None and not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)
            df.to_csv(path, index=False)
            return df
    except Exception:
        pass
    return pd.DataFrame()


def fetch_all_stocks_daily(force=False):
    """Fetch all 5 candidate stocks daily data."""
    result = {}
    for code, name in STOCKS.items():
        ts_code = f'{code}.SH' if code.startswith('6') else f'{code}.SZ'
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


def fetch_moneyflow(ts_code, force=False):
    """
    Fetch moneyflow data for a stock.
    Returns empty DataFrame if permission denied or API unavailable.
    """
    code = ts_code.split('.')[0]
    path = _cache_path(f'moneyflow_{code}')
    if not force and _is_fresh(path):
        df = _read_csv(path)
        if 'trade_date' in df.columns:
            return df
    try:
        pro = _get_pro()
        df = pro.moneyflow_dc(ts_code=ts_code, start_date='20080101')
        if df is not None and not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)
            df.to_csv(path)
            return df
    except Exception:
        pass
    return pd.DataFrame()
