"""Optional P2 causal 12-M5 path features; do NOT alter frozen features_v1.

No future bars, no labels. Each time gap resets all lookbacks.
Both features first become available after 12 complete preceding bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlpipeline.features import FIVE_MINUTES

TRAJECTORY_SCHEMA = "trajectory_p2_predeclared_12_v1"
TRAJECTORY_COLUMNS = (
    "trajectory_signed_efficiency_12",
    "trajectory_prior_range_position_12",
)
WINDOW = 12


def build_trajectory_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly causal trajectory descriptors at closed bar t."""
    required = {"symbol", "bar_open_at", "bar_close_at", "high", "low", "close"}
    if not required.issubset(candles.columns):
        raise ValueError("Missing OHLC/time fields")
    bars = candles.sort_values("bar_open_at").reset_index(drop=True)
    if bars.empty or bars.symbol.nunique() != 1:
        raise ValueError("Exactly one nonempty symbol is required")
    if (bars.bar_open_at.duplicated().any()
            or bars.bar_close_at.duplicated().any()
            or not isinstance(bars.bar_open_at.dtype, pd.DatetimeTZDtype)
            or not isinstance(bars.bar_close_at.dtype, pd.DatetimeTZDtype)
            or str(bars.bar_open_at.dt.tz) != "UTC"
            or str(bars.bar_close_at.dt.tz) != "UTC"
            or not (bars.bar_close_at-bars.bar_open_at).eq(FIVE_MINUTES).all()):
        raise ValueError("Closed UTC M5 bars with unique times required")
    ohlc = bars[["high", "low", "close"]].to_numpy(dtype=float)
    if (not np.isfinite(ohlc).all()
            or (ohlc <= 0).any()
            or (bars.high < bars.low).any()
            or (bars.close > bars.high).any()
            or (bars.close < bars.low).any()):
        raise ValueError("Invalid OHLC")
    bars["_segment"] = bars.bar_open_at.diff().ne(FIVE_MINUTES).cumsum()
    chunks = []
    for _, group in bars.groupby("_segment", sort=False):
        close = group.close.astype(float).reset_index(drop=True)
        high = group.high.astype(float).reset_index(drop=True)
        low = group.low.astype(float).reset_index(drop=True)
        logclose = np.log(close)
        diff = logclose.diff()
        distance = diff.abs().rolling(WINDOW, min_periods=WINDOW).sum()
        displacement = logclose - logclose.shift(WINDOW)
        efficiency = displacement.div(distance.where(distance.ne(0)))
        # Exactly flat paths have no directional efficiency.
        efficiency = efficiency.mask(distance.eq(0), 0.0)
        prior_high = high.shift(1).rolling(WINDOW, min_periods=WINDOW).max()
        prior_low = low.shift(1).rolling(WINDOW, min_periods=WINDOW).min()
        width = prior_high - prior_low
        position = (close - prior_low).div(width.where(width.ne(0)))
        position = position.mask(width.eq(0), 0.5)
        result = pd.DataFrame({
            "trajectory_signed_efficiency_12":efficiency.astype("float32"),
            "trajectory_prior_range_position_12":position.astype("float32"),
        })
        result.index = group.index
        chunks.append(result)
    features = pd.concat(chunks).sort_index()
    out = bars[["symbol", "bar_close_at"]].copy()
    for name in TRAJECTORY_COLUMNS:
        out[name] = features[name].to_numpy()
    if out.duplicated(["symbol", "bar_close_at"]).any():
        raise ValueError("Duplicate trajectory keys")
    return out
