"""P4 V6: causal multi-timeframe market context and symmetric LONG/SHORT census.

Strictly UNLABELED. No fills, returns, labels, ML fitting or orders.
Use completed, contiguous M5 -> M15/H1 candles only; do not forward fill
missing data. Historical development must come from 21-month P0 loader.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlpipeline.trend_pullback import M5, _validate_closed_m5, _segment_features

TIMEFRAMES = {"M15": 3, "H1": 12}
REGIMES = ("UP", "DOWN", "RANGE", "UNKNOWN")
MIN_EVENT_SPACING_BARS = 8
# Fixed descriptive thresholds, chosen before any P4 V6 outcome observations.
EFFICIENCY_WINDOW = 12
EFFICIENCY_MIN_TREND = 0.35
EMA_SEPARATION_MIN_ATR = 0.35
SLOPE_LOOKBACK = 3


def closed_higher_candles(bars: pd.DataFrame, factor: int) -> pd.DataFrame:
    """Aggregate only *complete* UTC-aligned consecutive M5 blocks.

    A missing sub-bar invalidates the whole higher-timeframe candle;
    it cannot be replaced by a synthetic bar. A higher candle is available
    at the close of its final M5 member, never earlier.
    """
    if factor not in (3, 12):
        raise ValueError("Only M15 and H1 are approved")
    m5 = _validate_closed_m5(bars)
    ns = factor * 5
    stamp_ns = m5.bar_close_at.astype("int64").to_numpy()
    # A candle ending at 00:15 contains M5 closes 00:05,00:10,00:15.
    ends = ((stamp_ns - 1) // (ns * 60 * 1_000_000_000) + 1) * (ns * 60 * 1_000_000_000)
    temp = m5.copy()
    temp["_bucket"] = pd.to_datetime(ends, utc=True)
    output = []
    for bucket, group in temp.groupby("_bucket", sort=True):
        if (len(group) != factor or not group.bar_close_at.iloc[-1] == bucket
                or not group.bar_close_at.diff().iloc[1:].eq(M5).all()):
            continue
        output.append({
            "symbol": str(group.symbol.iloc[0]),
            "bar_open_at": group.bar_open_at.iloc[0],
            "bar_close_at": bucket,
            "open": float(group.open.iloc[0]),
            "high": float(group.high.max()),
            "low": float(group.low.min()),
            "close": float(group.close.iloc[-1]),
            "is_final": True,
        })
    if not output:
        return pd.DataFrame(columns=[
            "symbol", "bar_open_at", "bar_close_at", "open", "high", "low", "close", "is_final"
        ])
    return pd.DataFrame(output)


def classify_context(higher: pd.DataFrame, factor: int) -> pd.DataFrame:
    """Regime determined only at the end of a complete higher-timeframe bar."""
    result = higher.copy()
    if result.empty:
        result["regime"] = pd.Series(dtype="object")
        return result
    if factor not in (3, 12):
        raise ValueError("Unknown higher timeframe")
    delta = M5 * factor
    result = result.sort_values("bar_close_at").reset_index(drop=True)
    result["_segment"] = result.bar_open_at.diff().ne(delta).cumsum()
    pieces = []
    for _, g in result.groupby("_segment", sort=False):
        f = _segment_features(g.reset_index(drop=True))
        close = f.close
        change = close.diff(EFFICIENCY_WINDOW).abs()
        path = close.diff().abs().rolling(EFFICIENCY_WINDOW, min_periods=EFFICIENCY_WINDOW).sum()
        efficiency = change / path.replace(0, np.nan)
        atr = f.atr14.replace(0, np.nan)
        spread = (f.ema9 - f.ema26) / atr
        positive = (spread >= EMA_SEPARATION_MIN_ATR
                    ) & f.ema9.gt(f.ema9.shift(SLOPE_LOOKBACK)
                    ) & f.ema26.gt(f.ema26.shift(SLOPE_LOOKBACK)
                    ) & close.gt(f.ema26)
        negative = (spread <= -EMA_SEPARATION_MIN_ATR
                    ) & f.ema9.lt(f.ema9.shift(SLOPE_LOOKBACK)
                    ) & f.ema26.lt(f.ema26.shift(SLOPE_LOOKBACK)
                    ) & close.lt(f.ema26)
        enough = (f.ema26.notna() & atr.notna() &
                  f.ema9.shift(SLOPE_LOOKBACK).notna() &
                  f.ema26.shift(SLOPE_LOOKBACK).notna() &
                  efficiency.notna())
        states = np.full(len(f), "UNKNOWN", dtype=object)
        states[enough.to_numpy()] = "RANGE"
        states[(enough & efficiency.ge(EFFICIENCY_MIN_TREND) & positive).to_numpy()] = "UP"
        states[(enough & efficiency.ge(EFFICIENCY_MIN_TREND) & negative).to_numpy()] = "DOWN"
        f["regime"] = states
        f["efficiency"] = efficiency
        f["spread_atr"] = spread
        pieces.append(f)
    return pd.concat(pieces, ignore_index=True)[
        ["bar_close_at","regime","efficiency","spread_atr"]
    ]


def symmetric_candidates(bars: pd.DataFrame) -> pd.DataFrame:
    """Same EMA9/EMA26 first-touch mechanics for LONG and mirrored SHORT.

    This is a new unlabeled discovery, *not* the old V1 trade universe.
    """
    m5=_validate_closed_m5(bars)
    m5["_segment"] = m5.bar_open_at.diff().ne(M5).cumsum()
    chunks=[]
    for _,g in m5.groupby("_segment",sort=False):
        f=_segment_features(g.reset_index(drop=True))
        p=f.shift(1)
        e9_4=f.ema9.shift(4)
        e26_4=f.ema26.shift(4)
        long_prior=(p.close>p.ema9)&(p.ema9>p.ema26)&(p.ema9>e9_4)&(p.ema26>e26_4)
        short_prior=(p.close<p.ema9)&(p.ema9<p.ema26)&(p.ema9<e9_4)&(p.ema26<e26_4)
        long_touch=(p.low>p.ema9)&(f.low<=f.ema9)
        short_touch=(p.high<p.ema9)&(f.high>=f.ema9)
        long_intact=(f.close>=f.ema26)&(f.close<=p.close)&(f.low>=f.ema26-0.25*f.atr14)
        short_intact=(f.close<=f.ema26)&(f.close>=p.close)&(f.high<=f.ema26+0.25*f.atr14)
        valid=f.atr14.gt(0)
        long_raw=(long_prior&long_touch&long_intact&valid).fillna(False).to_numpy(dtype=bool)
        short_raw=(short_prior&short_touch&short_intact&valid).fillna(False).to_numpy(dtype=bool)
        if (long_raw&short_raw).any():
            raise AssertionError("Contradictory LONG/SHORT at one timestamp")
        long=np.zeros(len(f),dtype=bool)
        short=np.zeros(len(f),dtype=bool)
        last_long=-MIN_EVENT_SPACING_BARS
        last_short=-MIN_EVENT_SPACING_BARS
        for j in range(len(f)):
            if long_raw[j] and j-last_long>=MIN_EVENT_SPACING_BARS:
                long[j]=True;last_long=j
            if short_raw[j] and j-last_short>=MIN_EVENT_SPACING_BARS:
                short[j]=True;last_short=j
        f["long_raw"]=long_raw
        f["short_raw"]=short_raw
        f["long_signal"]=long
        f["short_signal"]=short
        chunks.append(f)
    return pd.concat(chunks,ignore_index=True)[
        ["symbol","bar_close_at","long_raw","short_raw","long_signal","short_signal"]
    ]


def classify_m5(bars: pd.DataFrame) -> pd.DataFrame:
    """As-of join: M15/H1 state is always from a fully closed candle at <= t."""
    base=symmetric_candidates(bars)
    times=base.bar_close_at
    for name,factor in TIMEFRAMES.items():
        higher=closed_higher_candles(bars,factor)
        context=classify_context(higher,factor)
        right=context.rename(columns={"bar_close_at":"context_closed_at",
                         "regime":"regime_"+name.lower(),
                         "efficiency":"efficiency_"+name.lower(),
                         "spread_atr":"spread_atr_"+name.lower()})
        if right.empty:
            base["regime_"+name.lower()]="UNKNOWN"
            base["context_closed_at_"+name.lower()]=pd.NaT
            continue
        out=pd.merge_asof(base[["bar_close_at"]],
                          right.sort_values("context_closed_at"),
                          left_on="bar_close_at",
                          right_on="context_closed_at",
                          direction="backward")
        # Disallow stale context after a missing higher-timeframe candle.
        age=out.bar_close_at-out.context_closed_at
        stale=out.context_closed_at.isna() | age.ge(M5*factor)
        regime=out["regime_"+name.lower()].where(~stale,"UNKNOWN")
        base["regime_"+name.lower()]=regime.to_numpy()
        base["context_closed_at_"+name.lower()]=out.context_closed_at
        if (base.loc[~stale,"context_closed_at_"+name.lower()].gt(
                times.loc[~stale])).any():
            raise AssertionError("Future higher-timeframe candle joined")
    m15=base.regime_m15
    h1=base.regime_h1
    base["market_regime"]=np.select(
        [(m15=="UP")&(h1=="UP"),(m15=="DOWN")&(h1=="DOWN"),
         (m15=="UNKNOWN")|(h1=="UNKNOWN")],
        ["UP","DOWN","UNKNOWN"],default="RANGE",
    )
    base["long_with_trend"]=base.long_signal & base.market_regime.eq("UP")
    base["short_with_trend"]=base.short_signal & base.market_regime.eq("DOWN")
    if (base.long_with_trend & base.short_with_trend).any():
        raise AssertionError("Overlapping directions")
    return base
