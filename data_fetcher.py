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


def _get_pro():
    with open(TOKEN_FILE) as f:
        token = f.read().strip()
    ts.set_token(token)
    return ts.pro_api()


def _cache_path(name):
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f'{name}.csv')


def _is_fresh(path, max_hours=12):
    if not os.path.exists(path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return (datetime.now() - mtime).total_seconds() < max_hours * 3600


def _read_csv(path):
    df = pd.read_csv(path, index_col=0)
    if 'trade_date' in df.columns:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df


def fetch_index_daily(force=False):
    """Fetch 399975.SZ daily K-line. Cached to data/index_daily.csv"""
    path = _cache_path('index_daily')
    if not force and _is_fresh(path):
        return _read_csv(path)

    pro = _get_pro()
    df = pro.index_daily(ts_code=INDEX_CODE, start_date='20080101',
                         fields='ts_code,trade_date,open,high,low,close,vol,amount')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').reset_index(drop=True)
    df.to_csv(path)
    return df


def fetch_stock_daily(ts_code, force=False):
    """Fetch individual stock daily K-line."""
    code = ts_code.split('.')[0]
    path = _cache_path(f'stock_daily_{code}')
    if not force and _is_fresh(path):
        return _read_csv(path)

    pro = _get_pro()
    df = pro.daily(ts_code=ts_code, start_date='20080101',
                   fields='ts_code,trade_date,open,high,low,close,vol,amount')
    if df is not None and not df.empty:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        df.to_csv(path)
        return df
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
