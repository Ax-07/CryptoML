"""Causal EMA9/EMA26 uptrend + first-touch pullback detector (research v1).

One symbol at a time; only completed M5 bars at or before signal time t.
Never reads files, future labels, reserved holdouts, or exchange endpoints.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

M5 = pd.Timedelta(minutes=5)
MIN_EVENT_SPACING_BARS = 8  # no overlapping 2-bar delay + six-bar outcomes


def _validate_closed_m5(candles: pd.DataFrame) -> pd.DataFrame:
    required = {
        "symbol", "bar_open_at", "bar_close_at", "open", "high", "low", "close", "is_final",
    }
    missing = required.difference(candles.columns)
    if missing:
        raise ValueError(f"Missing candle fields: {sorted(missing)}")
    if candles.empty or candles.symbol.nunique() != 1:
        raise ValueError("Require nonempty candles from exactly one symbol")
    bars = candles.loc[:, sorted(required)].copy()
    for column in ("bar_open_at", "bar_close_at"):
        if not isinstance(bars[column].dtype, pd.DatetimeTZDtype) or str(bars[column].dt.tz) != "UTC":
            raise ValueError(f"{column} must be UTC-aware")
        if bars[column].isna().any():
            raise ValueError(f"Null {column}")
    bars = bars.sort_values("bar_open_at").reset_index(drop=True)
    if (bars.bar_open_at.duplicated().any()
            or not (bars.bar_close_at - bars.bar_open_at).eq(M5).all()
            or not bars.is_final.eq(True).all()):
        raise ValueError("Require unique, closed five-minute candles")
    values = bars[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Require finite positive OHLC prices")
    if ((bars.high < bars[["open", "close", "low"]].max(axis=1)).any()
            or (bars.low > bars[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("Invalid OHLC candle geometry")
    return bars


def _segment_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close, high, low = out.close, out.high, out.low
    out["ema9"] = close.ewm(span=9, adjust=False, min_periods=9).mean()
    out["ema26"] = close.ewm(span=26, adjust=False, min_periods=26).mean()
    previous_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    tr.iloc[0] = np.nan
    out["atr14"] = tr.rolling(14, min_periods=14).mean()
    out["price_ema9_atr"] = (close - out.ema9) / out.atr14.replace(0, np.nan)
    out["ema_spread_atr"] = (out.ema9 - out.ema26) / out.atr14.replace(0, np.nan)
    return out


def detect_pullbacks(candles: pd.DataFrame) -> pd.DataFrame:
    """Return chronological closed-bar features and LONG candidate flags.

    t-1 must have close > EMA9 > EMA26, both EMAs rising since t-4,
    and its low strictly above EMA9. At t, low touches EMA9 for the
    first time; close remains at/above EMA26 and does not exceed t-1 close.
    Low may undercut EMA26 by at most 0.25 * ATR14(t). Candidate is known
    strictly after bar t CLOSE, not at the intrabar EMA9 touch.

    Cooldown is deterministic, scoped by contiguous segment, and excludes
    a candidate within 8 bars of the last selected candidate.
    """
    bars = _validate_closed_m5(candles)
    bars["segment_id"] = bars.bar_open_at.diff().ne(M5).cumsum()
    features = pd.concat(
        [_segment_features(part) for _, part in bars.groupby("segment_id", sort=False)],
        ignore_index=True,
    )
    segment = features.groupby("segment_id", sort=False)
    close_prior = segment.close.shift(1)
    low_prior = segment.low.shift(1)
    ema9_prior = segment.ema9.shift(1)
    ema26_prior = segment.ema26.shift(1)
    ema9_four_back = segment.ema9.shift(4)
    ema26_four_back = segment.ema26.shift(4)
    uptrend = (
        close_prior.gt(ema9_prior)
        & ema9_prior.gt(ema26_prior)
        & ema9_prior.gt(ema9_four_back)
        & ema26_prior.gt(ema26_four_back)
    )
    first_touch = low_prior.gt(ema9_prior) & features.low.le(features.ema9)
    intact = (
        features.close.ge(features.ema26)
        & features.close.le(close_prior)
        & features.low.ge(features.ema26 - 0.25 * features.atr14)
        & features.atr14.gt(0)
    )
    features["uptrend_prior"] = uptrend.fillna(False).astype(bool)
    features["first_ema9_touch"] = first_touch.fillna(False).astype(bool)
    features["pullback_raw"] = (uptrend & first_touch & intact).fillna(False).astype(bool)
    chosen = np.zeros(len(features), dtype=bool)
    for _, part in features.groupby("segment_id", sort=False):
        last_event = -MIN_EVENT_SPACING_BARS
        for local_i, (i, is_raw) in enumerate(zip(part.index, part.pullback_raw)):
            if is_raw and local_i - last_event >= MIN_EVENT_SPACING_BARS:
                chosen[i] = True
                last_event = local_i
    features["pullback_signal"] = chosen
    return features[[
        "symbol", "bar_open_at", "bar_close_at", "segment_id", "close",
        "ema9", "ema26", "atr14", "price_ema9_atr", "ema_spread_atr",
        "uptrend_prior", "first_ema9_touch", "pullback_raw", "pullback_signal",
    ]].copy()
