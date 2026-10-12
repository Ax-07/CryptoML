"""Coverage and OHLC data quality checks. Never synthesize missing candles."""
from __future__ import annotations

import pandas as pd

INTERVAL = pd.Timedelta(minutes=5)


def inspect_quality(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "row_count": 0, "first_bar_open_at": None, "last_bar_open_at": None,
            "internal_missing_bars": 0, "gap_count": 0, "gaps": [], "zero_volume_bars": 0,
            "duplicate_bars": 0, "status": "EMPTY",
        }
    ordered = df.sort_values("bar_open_at").reset_index(drop=True)
    steps = ordered["bar_open_at"].diff()
    missing = 0
    gaps = []
    for i in range(1, len(ordered)):
        diff = steps.iloc[i]
        if diff > INTERVAL:
            absent = int(diff / INTERVAL) - 1
            missing += absent
            if len(gaps) < 100:  # never return an unbounded response
                gaps.append({
                    "after": ordered.loc[i - 1, "bar_open_at"].isoformat().replace("+00:00", "Z"),
                    "before": ordered.loc[i, "bar_open_at"].isoformat().replace("+00:00", "Z"),
                    "missing_bars": absent,
                })
    duplicates = int(ordered["bar_open_at"].duplicated().sum())
    return {
        "row_count": len(ordered),
        "first_bar_open_at": ordered["bar_open_at"].iloc[0].isoformat().replace("+00:00", "Z"),
        "last_bar_open_at": ordered["bar_open_at"].iloc[-1].isoformat().replace("+00:00", "Z"),
        "internal_missing_bars": missing,
        "gap_count": int((steps > INTERVAL).sum()),
        "gaps": gaps,
        "gaps_truncated": int((steps > INTERVAL).sum()) > len(gaps),
        "zero_volume_bars": int((ordered["volume_base"] == 0).sum()),
        "duplicate_bars": duplicates,
        "status": "GAPS_DETECTED" if missing or duplicates else "OK",
    }
