"""Causal, optional P2 market-structure features; NOT part of frozen features_v1.

A pivot at index p needs LEFT=2 prior and confirmation_bars=1/2/3
later closed bars. It is FIRST KNOWN at p+confirmation_bars.
The default 2-bar calculation reproduces the original R2 experiment.
All calculations restart after any noncontiguous M5 bar; no fabricated candles.
This is a mechanical operational definition, not a canonical BOS/CHoCH signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlpipeline.features import FIVE_MINUTES

LEFT = 2
RIGHT = 2
STRUCTURE_SCHEMA = "market_structure_p2_causal_v1"
STRUCTURE_COLUMNS = (
    "structure_hh", "structure_hl", "structure_lh", "structure_ll",
    "structure_bos_up", "structure_bos_down",
    "structure_choch_up", "structure_choch_down",
    "structure_distance_swing_high", "structure_distance_swing_low",
    "structure_age_swing_high", "structure_age_swing_low",
)


def _segment_structure(bars: pd.DataFrame, *, confirmation_bars: int = RIGHT) -> pd.DataFrame:
    """Use only bar-close information known at each row; each segment isolated."""
    high = bars["high"].astype(float).reset_index(drop=True)
    low = bars["low"].astype(float).reset_index(drop=True)
    close = bars["close"].astype(float).reset_index(drop=True)
    if (not np.isfinite(high).all() or not np.isfinite(low).all()
            or not np.isfinite(close).all() or (high < low).any()
            or (close <= 0).any()):
        raise ValueError("Invalid OHLC for structure features")

    # At time t=p+r, all comparisons are to bars <=t: two older
    # (p-2,p-1) and r already-closed newer bars (p+1..p+r).
    # Strict comparison means ties never create a pivot.
    r = confirmation_bars
    ph = high.shift(r)
    pl = low.shift(r)
    confirmed_h = pd.Series(True, index=high.index)
    confirmed_l = pd.Series(True, index=low.index)
    for offset in range(r):
        confirmed_h &= ph.gt(high.shift(offset))
        confirmed_l &= pl.lt(low.shift(offset))
    for offset in range(r + 1, r + LEFT + 1):
        confirmed_h &= ph.gt(high.shift(offset))
        confirmed_l &= pl.lt(low.shift(offset))
    new_h = ph.where(confirmed_h)
    new_l = pl.where(confirmed_l)
    # At event t, compare against the prior CONFIRMED pivot of the same kind.
    prev_h_at_event = new_h.ffill().shift(1).where(confirmed_h)
    prev_l_at_event = new_l.ffill().shift(1).where(confirmed_l)
    high_direction = pd.Series(
        np.where(confirmed_h & prev_h_at_event.notna(),
                 np.where(new_h > prev_h_at_event, 1,
                          np.where(new_h < prev_h_at_event, -1, 0)), np.nan),
        dtype=float,
    ).ffill().fillna(0)
    low_direction = pd.Series(
        np.where(confirmed_l & prev_l_at_event.notna(),
                 np.where(new_l > prev_l_at_event, 1,
                          np.where(new_l < prev_l_at_event, -1, 0)), np.nan),
        dtype=float,
    ).ffill().fillna(0)
    last_h = new_h.ffill()
    last_l = new_l.ffill()
    # Use yesterday's known structural state/levels for breaks on current bar.
    level_h = last_h.shift(1)
    level_l = last_l.shift(1)
    crossed_h = (
        level_h.notna() & close.gt(level_h) & close.shift(1).le(level_h)
    )
    crossed_l = (
        level_l.notna() & close.lt(level_l) & close.shift(1).ge(level_l)
    )
    bull = high_direction.shift(1).eq(1) & low_direction.shift(1).eq(1)
    bear = high_direction.shift(1).eq(-1) & low_direction.shift(1).eq(-1)

    index = pd.Series(np.arange(len(close), dtype=float))
    last_h_index = index.where(confirmed_h).ffill()
    last_l_index = index.where(confirmed_l).ffill()
    age_h = (index - last_h_index) / 48.0
    age_l = (index - last_l_index) / 48.0

    return pd.DataFrame({
        "structure_hh": high_direction.eq(1).astype("float32"),
        "structure_hl": low_direction.eq(1).astype("float32"),
        "structure_lh": high_direction.eq(-1).astype("float32"),
        "structure_ll": low_direction.eq(-1).astype("float32"),
        "structure_bos_up": (crossed_h & bull).astype("float32"),
        "structure_bos_down": (crossed_l & bear).astype("float32"),
        "structure_choch_up": (crossed_h & bear).astype("float32"),
        "structure_choch_down": (crossed_l & bull).astype("float32"),
        "structure_distance_swing_high": ((last_h / close) - 1).astype("float32"),
        "structure_distance_swing_low": ((close / last_l) - 1).astype("float32"),
        "structure_age_swing_high": age_h.astype("float32"),
        "structure_age_swing_low": age_l.astype("float32"),
    })


def build_market_structure_features(
    candles: pd.DataFrame, *, confirmation_bars: int = RIGHT,
) -> pd.DataFrame:
    """Twelve new columns at closed M5 t. The first confirmed pivots are NaN.

    No label reading, no future prices, no ADA access. Strict pivot ties are
    ignored to avoid arbitrary ordering. A pivot at p first appears at
    p+confirmation_bars (default 2). Only 1, 2, 3 are predeclared.
    BOS = close crossing last known high in bullish state (or low in bearish).
    CHoCH = same break AGAINST the immediately previous confirmed structure.
    No ambiguous transition is assigned a BOS/CHoCH; zero means no signal.
    """
    if type(confirmation_bars) is not int or confirmation_bars not in (1, 2, 3):
        raise ValueError("confirmation_bars must be one of 1, 2 or 3")
    required = {"symbol","bar_open_at","bar_close_at","high","low","close"}
    if not required.issubset(candles):
        raise ValueError("Missing structure OHLC columns")
    bars = candles.sort_values("bar_open_at").reset_index(drop=True)
    if bars.empty or bars.symbol.nunique()!=1:
        raise ValueError("Exactly one nonempty symbol per structure calculation")
    if bars.bar_open_at.duplicated().any():
        raise ValueError("Duplicate M5 timestamp")
    if (not isinstance(bars.bar_open_at.dtype, pd.DatetimeTZDtype)
            or not isinstance(bars.bar_close_at.dtype, pd.DatetimeTZDtype)
            or str(bars.bar_open_at.dt.tz)!="UTC"
            or str(bars.bar_close_at.dt.tz)!="UTC"
            or not (bars.bar_close_at-bars.bar_open_at).eq(FIVE_MINUTES).all()):
        raise ValueError("Closed UTC M5 bars required")
    bars["_segment"] = bars.bar_open_at.diff().ne(FIVE_MINUTES).cumsum()
    chunks = []
    for _, group in bars.groupby("_segment", sort=False):
        chunk = _segment_structure(group, confirmation_bars=confirmation_bars)
        chunk.index=group.index
        chunks.append(chunk)
    calculated = pd.concat(chunks,axis=0).sort_index()
    out=bars.loc[:,["symbol","bar_close_at"]].copy()
    for column in STRUCTURE_COLUMNS:
        out[column]=calculated[column].to_numpy()
    if out.duplicated(["symbol","bar_close_at"]).any():
        raise ValueError("Duplicate symbol/timestamp structure rows")
    return out
