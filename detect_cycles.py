"""
One-time cycle detection for securities sector index (399975.SZ).
Detects bull market cycles using trough-to-peak analysis on smoothed price.

Usage: python detect_cycles.py
Output: output/cycles.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from data_fetcher import fetch_index_daily

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# ---- Tunable constants ----
SMOOTH_WINDOW = 20       # MA period for smoothing
MIN_AMPLITUDE_PCT = 25   # Minimum rally amplitude (%)
MIN_DURATION_DAYS = 20   # Minimum rally duration (trading days)
MERGE_ORDER = 10          # Order for argrelextrema (half-window size)
# ---------------------------


def detect_cycles(df, smooth_window=SMOOTH_WINDOW,
                  min_amp_pct=MIN_AMPLITUDE_PCT,
                  min_days=MIN_DURATION_DAYS):
    """
    Detect bull cycles from price data.

    Returns list of dicts: [{start_date, end_date, start_price, end_price, change_pct, duration_days}]
    """
    close = df['close'].values
    dates = df['trade_date'].values

    # Smooth the close price
    smoothed = pd.Series(close).rolling(smooth_window, min_periods=1).mean().values

    # Find local minima (troughs) and maxima (peaks)
    troughs = argrelextrema(smoothed, np.less, order=MERGE_ORDER)[0]
    peaks = argrelextrema(smoothed, np.greater, order=MERGE_ORDER)[0]

    # Build cycles: each trough followed by next peak = one bull cycle
    cycles = []
    for t_idx in troughs:
        # Find the next peak after this trough
        later_peaks = peaks[peaks > t_idx]
        if len(later_peaks) == 0:
            continue
        p_idx = later_peaks[0]

        duration = int(p_idx - t_idx)
        if duration < min_days:
            continue

        start_price = close[t_idx]
        end_price = close[p_idx]
        change_pct = (end_price - start_price) / start_price * 100

        if change_pct < min_amp_pct:
            continue

        cycles.append({
            'start_date': pd.Timestamp(dates[t_idx]),
            'end_date': pd.Timestamp(dates[p_idx]),
            'start_price': round(start_price, 2),
            'end_price': round(end_price, 2),
            'change_pct': round(change_pct, 2),
            'duration_days': duration,
        })

    # Remove overlapping cycles (keep the one with larger amplitude when overlap)
    filtered = []
    for c in cycles:
        if not filtered:
            filtered.append(c)
            continue
        last = filtered[-1]
        if c['start_date'] <= last['end_date']:
            # Overlap: keep the one with larger amplitude
            if c['change_pct'] > last['change_pct']:
                filtered[-1] = c
        else:
            filtered.append(c)

    return filtered


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching index data...")
    df = fetch_index_daily()
    print(f"Loaded {len(df)} daily bars from {df['trade_date'].iloc[0].date()} to {df['trade_date'].iloc[-1].date()}")

    print(f"\nDetecting cycles (smooth={SMOOTH_WINDOW}, min_amp={MIN_AMPLITUDE_PCT}%, min_days={MIN_DURATION_DAYS})...")
    cycles = detect_cycles(df)

    print(f"\nFound {len(cycles)} bull cycles (including 2009 for reference):\n")
    print(f"{'#':<4} {'Start':<12} {'End':<12} {'Days':<6} {'Start$':<8} {'End$':<8} {'Change%':<8}")
    print("-" * 68)
    for i, c in enumerate(cycles, 1):
        print(f"{i:<4} {str(c['start_date'].date()):<12} {str(c['end_date'].date()):<12} "
              f"{c['duration_days']:<6} {c['start_price']:<8} {c['end_price']:<8} {c['change_pct']:<8}")

    # Save to CSV
    cycles_df = pd.DataFrame(cycles)
    path = os.path.join(OUTPUT_DIR, 'cycles.csv')
    cycles_df.to_csv(path, index=False)
    print(f"\nSaved to {path}")


if __name__ == '__main__':
    main()
