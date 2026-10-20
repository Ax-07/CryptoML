"""Causal EMA9/EMA26 bullish pullback events, research-only (no orders).

The module deliberately separates feature/event detection at the CLOSED bar t
from hypothetical future markouts. It does not fit models or read P0 itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

M5 = pd.Timedelta(minutes=5)
HORIZONS = (1, 2, 3, 6)
PULLBACK_BARS = 3
TOUCH_TOLERANCE_ATR = 0.25
COOLDOWN_BARS = 8  # Prevent simultaneous H6 hypothetical positions.
COST_BPS = 2.0

# All ML candidate columns available only after a bar has CLOSED.
EVENT_FEATURES = (
    "close_ema9_atr", "ema9_ema26_atr", "ema9_slope_3_atr",
    "prior_low_ema9_atr", "body_atr", "atr_pct", "ret_3",
)


def build_signal_frame(candles: pd.DataFrame) -> pd.DataFrame:
    """Generate long-entry signals using only closed OHLCV through time t.

    Initial fixed rule: EMA9 > EMA26, close > EMA9, EMA9 rising over 3 bars,
    uptrend also established at t-3, a shallow EMA9 touch in t-3..t-1,
    and current green reclaim candle closing above previous close.
    """
    required = {"symbol", "bar_open_at", "bar_close_at", "open", "high", "low", "close"}
    if missing := required - set(candles.columns):
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    if candles.empty or candles.symbol.nunique() != 1:
        raise ValueError("Exactly one non-empty symbol is required")
    d = candles.sort_values("bar_open_at").reset_index(drop=True).copy()
    for column in ("bar_open_at", "bar_close_at"):
        if not isinstance(d[column].dtype, pd.DatetimeTZDtype) or str(d[column].dt.tz) != "UTC":
            raise ValueError(f"{column} must be UTC-aware")
    if d.bar_open_at.duplicated().any() or not (d.bar_close_at - d.bar_open_at).eq(M5).all():
        raise ValueError("Duplicate, malformed, or non-M5 candle timestamps")
    if (d.bar_open_at.diff().dropna() < M5).any():
        raise ValueError("Overlapping candles")
    numbers = d[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(numbers).all() or (numbers <= 0).any():
        raise ValueError("Finite positive prices are mandatory")
    if ((d.low > d.high) | (d.open < d.low) | (d.open > d.high) |
            (d.close < d.low) | (d.close > d.high)).any():
        raise ValueError("Invalid candle OHLC geometry")
    d["segment_id"] = d.bar_open_at.diff().ne(M5).cumsum()
    group = d.groupby("segment_id", sort=False)
    close = d.close.astype(float)
    d["ema9"] = group["close"].transform(
        lambda x: x.ewm(span=9, adjust=False, min_periods=9).mean())
    d["ema26"] = group["close"].transform(
        lambda x: x.ewm(span=26, adjust=False, min_periods=26).mean())
    prev_close = group["close"].shift(1)
    true_range = pd.concat([
        (d.high - d.low), (d.high - prev_close).abs(),
        (d.low - prev_close).abs(),
    ], axis=1).max(axis=1)
    true_range.loc[group.head(1).index] = np.nan
    d["atr14"] = true_range.groupby(d.segment_id, sort=False).transform(
        lambda x: x.rolling(14, min_periods=14).mean())
    lag_ema9 = d.ema9.groupby(d.segment_id, sort=False).shift(3)
    lag_ema26 = d.ema26.groupby(d.segment_id, sort=False).shift(3)
    lag_close = group["close"].shift(3)
    lag_touch = ((d.low <= d.ema9 + TOUCH_TOLERANCE_ATR * d.atr14) &
                 (d.low >= d.ema26 - TOUCH_TOLERANCE_ATR * d.atr14) &
                 d.ema9.gt(d.ema26))
    prior_touch = lag_touch.groupby(d.segment_id, sort=False).transform(
        lambda x: x.shift(1).rolling(PULLBACK_BARS, min_periods=PULLBACK_BARS).max())
    d["prior_ema9_touch"] = prior_touch.fillna(0).astype(bool)
    d["trend"] = d.close.gt(d.ema9) & d.ema9.gt(d.ema26)
    d["trigger"] = (
        d["trend"] & d.ema9.gt(lag_ema9) &
        lag_close.gt(lag_ema9) & lag_ema9.gt(lag_ema26) &
        d.prior_ema9_touch & d.close.gt(d.open) & d.close.gt(prev_close)
    )
    d["signal"] = False
    # A deterministic temporal embargo prevents repeated signals on a single run.
    for _, segment in d.groupby("segment_id", sort=False):
        last = -COOLDOWN_BARS
        for idx in segment.index[segment.trigger]:
            if idx - last >= COOLDOWN_BARS:
                d.at[idx, "signal"] = True
                last = idx
    # All engineered columns are functions of time <= t only.
    prior_low = group["low"].shift(1)
    d["close_ema9_atr"] = (d.close - d.ema9) / d.atr14
    d["ema9_ema26_atr"] = (d.ema9 - d.ema26) / d.atr14
    d["ema9_slope_3_atr"] = (d.ema9 - lag_ema9) / d.atr14
    d["prior_low_ema9_atr"] = (prior_low - d.ema9) / d.atr14
    d["body_atr"] = (d.close - d.open) / d.atr14
    d["atr_pct"] = d.atr14 / d.close
    d["ret_3"] = close / lag_close - 1
    if not np.isfinite(d.loc[d.signal, EVENT_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("A triggered event has incomplete features")
    return d


def label_signal_events(
    frame: pd.DataFrame, *, end_exclusive: pd.Timestamp,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """Score hypothetical delayed fills only if ALL exits mature before boundary.

    Signal after bar close t; skip the next complete bar; assume entry at the
    OPEN of the candle beginning t+5 min, exit after h complete candles at
    t+(1+h)*5 min. Prices are NOT guaranteed executable fills.
    """
    if (not isinstance(end_exclusive, pd.Timestamp) or end_exclusive.tz is None
            or str(end_exclusive.tz) != "UTC"):
        raise ValueError("end_exclusive must be a UTC timestamp")
    if not horizons or tuple(sorted(set(horizons))) != horizons or any(
            not isinstance(h, int) or h < 1 or h > 12 for h in horizons):
        raise ValueError("Horizons must be distinct ascending integers 1..12")
    needed = {"symbol", "bar_close_at", "segment_id", "signal", "open", "close", *EVENT_FEATURES}
    if needed - set(frame):
        raise ValueError("Not a valid signal frame")
    d = frame.sort_values("bar_close_at").reset_index(drop=True)
    if d.symbol.nunique() != 1 or d.bar_close_at.duplicated().any():
        raise ValueError("Require unique timestamps for one asset")
    group = d.groupby("segment_id", sort=False)
    entry = group["open"].shift(-2)  # open of bar beginning t+5, not the next instant open
    entry_open_at = d.bar_close_at.groupby(d.segment_id, sort=False).shift(-2) - M5
    events = d.loc[d.signal].copy()
    events["entry_open_at"] = entry_open_at.loc[events.index]
    events["entry_open"] = entry.loc[events.index]
    max_mature = max(horizons)
    next_close = group["close"].shift(-1)
    next_at = group["bar_close_at"].shift(-1)
    events["next_bar_return_bps"] = 10000 * (
        next_close.loc[events.index] / events.close - 1.0)
    events["next_bar_up"] = events.next_bar_return_bps.gt(0).astype("int8")
    valid = (events.entry_open_at.eq(events.bar_close_at + M5) &
             next_at.loc[events.index].eq(events.bar_close_at + M5))
    for h in horizons:
        exit_close = group["close"].shift(-(h+1))
        exit_at = group["bar_close_at"].shift(-(h+1))
        valid &= exit_at.loc[events.index].eq(events.bar_close_at + (h+1) * M5)
        events[f"gross_h{h}_bps"] = 10000 * (
            exit_close.loc[events.index] / events.entry_open - 1.0)
        events[f"net_h{h}_bps"] = events[f"gross_h{h}_bps"] - COST_BPS
    events["label_mature_at"] = events.bar_close_at + (max_mature+1) * M5
    valid &= events.label_mature_at.lt(end_exclusive)
    valid &= events.entry_open.gt(0)
    events = events.loc[valid].copy().reset_index(drop=True)
    required_scores = [f"net_h{h}_bps" for h in horizons]
    if not np.isfinite(events[required_scores].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite markout")
    # Model training should see EVENT_FEATURES at t, not later prices/labels.
    keep = ["symbol", "bar_close_at", "entry_open_at", "label_mature_at",
            *EVENT_FEATURES, "next_bar_return_bps", "next_bar_up"]
    for h in horizons:
        keep.extend([f"gross_h{h}_bps", f"net_h{h}_bps"])
    return events.loc[:, keep]
