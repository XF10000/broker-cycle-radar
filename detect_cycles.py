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
from config import (
    SMOOTH_WINDOW, MIN_AMPLITUDE_PCT, MIN_DURATION_DAYS, ARGRELEXTREMA_ORDER,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def detect_cycles(df, smooth_window=SMOOTH_WINDOW,
                  min_amp_pct=MIN_AMPLITUDE_PCT,
                  min_days=MIN_DURATION_DAYS):
    """
    Detect bull cycles from price data.

    Uses smoothed price to find cycle boundaries (trough/peak).
    end_date/end_price = actual highest close within [smoothed_trough, smoothed_peak].
    """
    close = df['close'].values
    dates = df['trade_date'].values

    smoothed = pd.Series(close).rolling(smooth_window, min_periods=1).mean().values
    troughs = argrelextrema(smoothed, np.less, order=ARGRELEXTREMA_ORDER)[0]
    peaks = argrelextrema(smoothed, np.greater, order=ARGRELEXTREMA_ORDER)[0]

    cycles = []
    for t_idx in troughs:
        later_peaks = peaks[peaks > t_idx]
        if len(later_peaks) == 0:
            continue
        p_idx = later_peaks[0]

        duration = int(p_idx - t_idx)
        if duration < min_days:
            continue

        # Smoothed amplitude check (gatekeeping) — original logic,
        # only changed: end_date/price uses actual highest close within window.
        smoothed_amp = (close[p_idx] - close[t_idx]) / close[t_idx] * 100
        if smoothed_amp < min_amp_pct:
            continue

        start_price = close[t_idx]

        # Find actual highest close within [t_idx, p_idx] for end_date/price
        actual_peak_idx = int(t_idx + np.argmax(close[t_idx:p_idx+1]))
        # Require actual peak is at least 10 trading days from trough
        # (half of min_days) to avoid counting noise spikes as peaks
        if actual_peak_idx - t_idx >= 10:
            end_price = close[actual_peak_idx]
            end_date = dates[actual_peak_idx]
            dur = int(actual_peak_idx - t_idx)
        else:
            end_price = close[p_idx]
            end_date = dates[p_idx]
            dur = int(p_idx - t_idx)

        change_pct = (end_price - start_price) / start_price * 100
        if change_pct < min_amp_pct:
            continue

        cycles.append({
            'start_date': pd.Timestamp(dates[t_idx]),
            'end_date': pd.Timestamp(end_date),
            'start_price': round(start_price, 2),
            'end_price': round(end_price, 2),
            'change_pct': round(change_pct, 2),
            'duration_days': dur,
        })

    # Remove overlapping cycles (keep larger amplitude)
    filtered = []
    for c in cycles:
        if not filtered:
            filtered.append(c)
            continue
        last = filtered[-1]
        if c['start_date'] <= last['end_date']:
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

    # ---- Manual overrides for cycles smoothing misses ----

    # 2012-01-16 trough: smoothing masks the true June peak (637.72).
    # First smoothed peak (March) only gives 19.4% amplitude.
    # Replace with the actual cycle spanning to the real June high.
    seg12 = df[(df['trade_date'] >= '2012-01-16') & (df['trade_date'] <= '2012-06-30')]
    if not seg12.empty:
        cycles.append({
            'start_date': pd.Timestamp('2012-01-16'),
            'end_date': pd.Timestamp(seg12.loc[seg12['close'].idxmax(), 'trade_date']),
            'start_price': 447.44,
            'end_price': round(seg12['close'].max(), 2),
            'change_pct': round((seg12['close'].max() - 447.44) / 447.44 * 100, 2),
            'duration_days': len(seg12),
        })

    # 2014-07 to 2015-04: smoothing splits the mega-rally at the Dec→Feb
    # pullback (-12.5%). The Feb trough (1301) is far above the Jul trough (495),
    # so this is one continuous bull run, not two cycles.
    seg14 = df[(df['trade_date'] >= '2014-07-15') & (df['trade_date'] <= '2015-04-22')]
    if not seg14.empty:
        cycles.append({
            'start_date': pd.Timestamp('2014-07-15'),
            'end_date': pd.Timestamp(seg14.loc[seg14['close'].idxmax(), 'trade_date']),
            'start_price': 495.29,
            'end_price': round(seg14['close'].max(), 2),
            'change_pct': round((seg14['close'].max() - 495.29) / 495.29 * 100, 2),
            'duration_days': len(seg14),
        })

    # 2018-10-19 to 2019-03-07: smoothing splits this into two cycles
    # (Oct→Nov and Jan→Mar), but Jan 4 trough (593.74) > Oct 19 trough (488.63)
    # — a higher low within the same recovery. Merge into one.
    seg18 = df[(df['trade_date'] >= '2018-10-19') & (df['trade_date'] <= '2019-03-07')]
    if not seg18.empty:
        cycles.append({
            'start_date': pd.Timestamp('2018-10-19'),
            'end_date': pd.Timestamp(seg18.loc[seg18['close'].idxmax(), 'trade_date']),
            'start_price': 488.63,
            'end_price': round(seg18['close'].max(), 2),
            'change_pct': round((seg18['close'].max() - 488.63) / 488.63 * 100, 2),
            'duration_days': len(seg18),
        })

    # 2025 V-bottom rally missed by smoothing entirely
    cycles.append({
        'start_date': pd.Timestamp('2025-04-07'),
        'end_date': pd.Timestamp('2025-08-25'),
        'start_price': 695.11,
        'end_price': 951.11,
        'change_pct': 36.83,
        'duration_days': 100,
    })

    # Final sort
    cycles.sort(key=lambda c: c['start_date'])

    # Remove natural-detected cycles that overlap with manual overrides
    cycles = [c for c in cycles if not (
        # 2012: natural detection gives short cycle (01-16→03-09), overridden
        (str(c['start_date']).startswith('2012-01') and str(c['end_date']).startswith('2012-03')) or
        # 2014-2015: smoothing splits mega-rally at the Dec→Feb pullback
        (str(c['start_date']).startswith('2014-07') and str(c['end_date']).startswith('2014-12')) or
        (str(c['start_date']).startswith('2015-02') and str(c['end_date']).startswith('2015-04')) or
        # 2018-2019: smoothing splits recovery into two cycles
        (str(c['start_date']).startswith('2018-10') and str(c['end_date']).startswith('2018-11')) or
        (str(c['start_date']).startswith('2019-01') and str(c['end_date']).startswith('2019-')) or
        # 2025: natural detection gives a later start; manual override covers it
        (str(c['start_date']).startswith('2025-05') and str(c['end_date']).startswith('2025-08'))
    )]

    print(f"\nFound {len(cycles)} bull cycles:\n")
    print(f"{'#':<4} {'Start':<12} {'End':<12} {'Days':<6} {'Start$':<8} {'End$':<8} {'Change%':<8}")
    print("-" * 68)
    for i, c in enumerate(cycles, 1):
        print(f"{i:<4} {str(c['start_date'].date()):<12} {str(c['end_date'].date()):<12} "
              f"{c['duration_days']:<6} {c['start_price']:<8} {c['end_price']:<8} {c['change_pct']:<8}")

    cycles_df = pd.DataFrame(cycles)
    path = os.path.join(OUTPUT_DIR, 'cycles.csv')
    cycles_df.to_csv(path, index=False)
    print(f"\nSaved to {path}")


if __name__ == '__main__':
    main()
