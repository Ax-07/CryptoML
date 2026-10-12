"""Direction labels for fixed future H-bar close, independent of trade profitability."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlpipeline.features import FIVE_MINUTES

LABEL_SCHEMA_VERSION = "vol_scaled_v1"
DOWN, NEUTRAL, UP = 0, 1, 2
LABEL_NAMES = {DOWN: "DOWN", NEUTRAL: "NEUTRAL", UP: "UP"}


def add_labels(features: pd.DataFrame, *, horizon: int = 6,
               threshold_multiplier: float = 0.5) -> pd.DataFrame:
    """Labels use future close; their threshold uses trailing volatility at t.

    Future close must be exactly t+H; any missing candle makes sample invalid.
    A final-window future missing is never labeled NEUTRAL by mistake.
    """
    if not (1 <= horizon <= 288):
        raise ValueError("horizon must be 1..288")
    if not (0 < threshold_multiplier <= 10):
        raise ValueError("threshold_multiplier must be >0 and <=10")
    data = features.copy()
    if "segment_id" not in data:
        raise ValueError("Missing contiguous segment_id")
    segment = data.groupby("segment_id", sort=False)
    future_close = segment["close"].shift(-horizon)
    future_at = segment["bar_close_at"].shift(-horizon)
    valid = future_at.eq(data["bar_close_at"] + horizon*FIVE_MINUTES)
    future_return = (future_close/data["close"] - 1).where(valid)
    threshold = (threshold_multiplier * data["volatility_48"] * np.sqrt(horizon))
    labels = pd.Series(np.nan, index=data.index, dtype="float64")
    known = future_return.notna() & threshold.notna() & np.isfinite(threshold)
    labels.loc[known] = NEUTRAL
    labels.loc[known & (future_return > threshold)] = UP
    labels.loc[known & (future_return < -threshold)] = DOWN
    data["future_return"] = future_return
    data["direction_threshold"] = threshold
    data["target"] = labels
    data["label_mature_at"] = future_at.where(valid)
    return data
