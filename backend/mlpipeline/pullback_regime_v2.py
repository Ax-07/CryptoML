"""P4 V2 descriptive regime features, no trade scoring, training or labels.

Computes only closed-candle features at t. Base pullback V1 is unchanged.
V2 feature design is frozen before any V2 outcome evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from mlpipeline.trend_pullback import M5, _validate_closed_m5, detect_pullbacks

FEATURE_COLUMNS = (
    "v2_trend_spread_atr",
    "v2_ema9_slope_5_atr",
    "v2_ema26_slope_5_atr",
    "v2_prior_close_runup_12_atr",
    "v2_pullback_depth_ema9_atr",
    "v2_pullback_depth_ema26_atr",
    "v2_pullback_body_atr",
    "v2_pullback_lower_wick_atr",
    "v2_pullback_close_position",
    "v2_12bar_atr_fraction",
)

def describe_regimes(candles: pd.DataFrame) -> pd.DataFrame:
    """Return V1 events augmented by *past/current only* V2 features.

    Every gap starts a new indicator segment. No cross-gap shifts.
    All features describe t after close(t); no trade labels are read.
    """
    bars = _validate_closed_m5(candles)
    base = detect_pullbacks(bars)
    if (len(base) != len(bars) or
            not base.bar_close_at.equals(bars.bar_close_at)):
        raise AssertionError("Base detector/candle alignment mismatch")
    features = base.copy()
    # Segment ID from the V1 detector, which already resets on any missing M5 bar.
    group = features.groupby("segment_id",sort=False)
    ema9_5 = group.ema9.shift(5)
    ema26_5 = group.ema26.shift(5)
    close12 = group.close.shift(12)
    atr = features.atr14.replace(0,np.nan)
    features["v2_trend_spread_atr"] = (features.ema9-features.ema26)/atr
    features["v2_ema9_slope_5_atr"] = (features.ema9-ema9_5)/atr
    features["v2_ema26_slope_5_atr"] = (features.ema26-ema26_5)/atr
    features["v2_prior_close_runup_12_atr"] = (group.close.shift(1)-close12)/atr
    features["v2_pullback_depth_ema9_atr"] = (features.ema9-bars.low)/atr
    features["v2_pullback_depth_ema26_atr"] = (features.ema26-bars.low)/atr
    features["v2_pullback_body_atr"] = (bars.close-bars.open)/atr
    features["v2_pullback_lower_wick_atr"] = (bars[["open","close"]].min(axis=1)-bars.low)/atr
    spread=(bars.high-bars.low).replace(0,np.nan)
    features["v2_pullback_close_position"] = (bars.close-bars.low)/spread
    features["v2_12bar_atr_fraction"] = atr/bars.close
    return features


def regime_census(features: pd.DataFrame) -> dict:
    """No outcomes: fixed descriptive quartile-free buckets for V1 selected events.

    The absolute ATR units are research categories, not trained thresholds.
    Counts are not strategy results and MUST NOT be used as selection rules
    without a separate, prospectively frozen test.
    """
    required = {"symbol","pullback_signal","bar_close_at",*FEATURE_COLUMNS}
    if not required.issubset(features.columns):
        raise ValueError("Missing V2 feature columns")
    chosen=features.loc[features.pullback_signal]
    if chosen.empty:
        return {"events":0,"strength":{},"depth":{},"unclassified":0}
    matrix=chosen.loc[:,FEATURE_COLUMNS].to_numpy(dtype=float)
    valid=np.isfinite(matrix).all(axis=1)
    eligible=chosen.loc[valid]
    strength=eligible.v2_trend_spread_atr
    depth=eligible.v2_pullback_depth_ema9_atr
    strength_bucket=pd.cut(strength,[-np.inf,0.5,1.5,np.inf],
                           labels=["low","mid","high"],right=False)
    depth_bucket=pd.cut(depth,[-np.inf,0.0,0.5,1.0,np.inf],
                        labels=["no_undercut","shallow","medium","deep"],right=False)
    strength_counts={str(k):int(v) for k,v in strength_bucket.value_counts(sort=False).items()}
    depth_counts={str(k):int(v) for k,v in depth_bucket.value_counts(sort=False).items()}
    if sum(strength_counts.values())!=len(eligible) or sum(depth_counts.values())!=len(eligible):
        raise AssertionError("V2 regimes must reconcile")
    return {"events":int(len(chosen)),"valid_events":int(len(eligible)),
            "unclassified":int(len(chosen)-len(eligible)),
            "strength":strength_counts,"depth":depth_counts}
