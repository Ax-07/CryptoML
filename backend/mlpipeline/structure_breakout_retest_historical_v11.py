"""V11 frozen historical RESEARCH clone of V9 detector; NO labels or orders.

Only differences: allowing historical start times and removing CLI.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlpipeline.trend_pullback import M5, _validate_closed_m5, _segment_features
from mlpipeline.market_structure_v6 import closed_higher_candles, classify_context

PROTOCOL="p4_v9_breakout_retest_paper_signal_only_v1"
PROSPECTIVE_START=pd.Timestamp("2026-10-22T00:00:00Z")
SYMBOLS=("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")
BREAKOUT_ATR=0.10
INVALIDATION_ATR=0.75
RETEST_BARS=6
COOLDOWN_BARS=8
STRUCTURE_M15=12


def _asof_regime(times:pd.Series, context:pd.DataFrame, duration:pd.Timedelta)->pd.Series:
    if context.empty:
        return pd.Series("UNKNOWN",index=times.index,dtype=object)
    right=context.rename(columns={"bar_close_at":"known_at"})
    joined=pd.merge_asof(pd.DataFrame({"bar_close_at":times}),
                         right[["known_at","regime"]].sort_values("known_at"),
                         left_on="bar_close_at",right_on="known_at",direction="backward")
    stale=joined.known_at.isna() | (joined.bar_close_at-joined.known_at).ge(duration)
    return joined.regime.where(~stale,"UNKNOWN")


def detect(bars:pd.DataFrame,*,start:pd.Timestamp=pd.Timestamp("2024-10-01T00:00:00Z"))->pd.DataFrame:
    """Returns decision-time detections only; other candles used for warmup.

    The entire dataframe is already-known *offline input*. Passing prospective
    start does NOT attest that any result was live or contemporaneously saved.
    """
    if start.tzinfo is None or str(start.tz)!="UTC":
        raise ValueError("Historical V11 start must be UTC-aware")
    f=_validate_closed_m5(bars)
    symbol=str(f.symbol.iloc[0])
    if symbol not in SYMBOLS:
        raise ValueError("Not an approved V9 symbol")
    if len(f)<400:
        raise ValueError("Require sufficient warmup history, at least 400 M5")
    f["_segment"]=f.bar_open_at.diff().ne(M5).cumsum()
    f=pd.concat([_segment_features(s.reset_index(drop=True))
                 for _,s in f.groupby("_segment",sort=False)],ignore_index=True)
    h15=closed_higher_candles(bars,3)
    h1=closed_higher_candles(bars,12)
    regime=classify_context(h1,12)
    f["h1_regime"]=_asof_regime(f.bar_close_at,regime,pd.Timedelta(hours=1)).to_numpy()
    # The last 12 M15 must themselves be contiguous and already completed BEFORE
    # the current M15 UTC bucket; exclude the bucket containing breakout t.
    h15=h15.sort_values("bar_close_at").reset_index(drop=True)
    levels=[]
    for i in range(STRUCTURE_M15,len(h15)+1):
        previous=h15.iloc[i-STRUCTURE_M15:i]
        if not previous.bar_open_at.diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all():
            continue
        levels.append((previous.bar_close_at.iloc[-1],
                       float(previous.high.max()),float(previous.low.min())))
    if not levels:
        return pd.DataFrame(columns=["symbol","side","breakout_close_at",
            "decision_close_at","frozen_structure_level","atr_at_breakout",
            "h1_regime","status"])
    lev=pd.DataFrame(levels,columns=["level_known_at","structure_high","structure_low"])
    # strict: level must predate current 15-minute bucket's beginning;
    # at 10:15, current M15 close 10:15 is excluded, meaning cutoff 10:00.
    epoch=f.bar_close_at.astype("int64")
    fifteen_ns=15*60*1_000_000_000
    bucket_end=pd.to_datetime(((epoch-1)//fifteen_ns+1)*fifteen_ns,utc=True)
    bound=bucket_end-pd.Timedelta(minutes=15)
    join=pd.merge_asof(pd.DataFrame({"bound":bound}),
                       lev,left_on="bound",right_on="level_known_at",
                       direction="backward")
    f[["structure_high","structure_low"]]=join[["structure_high","structure_low"]]
    f["level_known_at"]=join["level_known_at"]
    events=[]
    cooldown=-COOLDOWN_BARS
    active=None
    for i in range(1,len(f)):
        r=f.iloc[i]
        if active is not None:
            age=i-active["at_idx"]
            if age>RETEST_BARS or f.bar_open_at.iloc[i]-f.bar_open_at.iloc[i-1]!=M5:
                active=None
            else:
                side=active["side"];level=active["level"];atr=active["atr"]
                if ((side=="LONG" and r.close<level-INVALIDATION_ATR*atr)
                    or (side=="SHORT" and r.close>level+INVALIDATION_ATR*atr)
                    or r.h1_regime!=active["h1_regime"]):
                    active=None
                else:
                    touched=(r.low<=level+BREAKOUT_ATR*atr and
                             r.high>=level-BREAKOUT_ATR*atr)
                    confirmed=(r.close>level and r.close>r.open) if side=="LONG" else (
                        r.close<level and r.close<r.open)
                    if touched and confirmed and r.bar_close_at>=start:
                        events.append({"symbol":symbol,"side":side,
                            "breakout_close_at":active["breakout_time"],
                            "decision_close_at":r.bar_close_at,
                            "frozen_structure_level":level,"atr_at_breakout":atr,
                            "h1_regime":active["h1_regime"],
                            "status":"OFFLINE_REPLAY_NOT_LIVE_ATTESTED"})
                        active=None
        if active is not None or i-cooldown<COOLDOWN_BARS:
            continue
        # Ensure input is an uninterrupted segment and indicators valid.
        p=f.iloc[i-1]
        expected_level_close = bucket_end.iloc[i] - pd.Timedelta(minutes=15)
        if (r.bar_open_at-p.bar_open_at!=M5 or
                not np.isfinite(r.atr14) or r.atr14<=0 or
                pd.isna(r.level_known_at) or
                r.level_known_at != expected_level_close):
            # Reject stale structure after a gap or missing complete M15.
            continue
        for side,regime_key,level,cross in (
            ("LONG","UP",float(r.structure_high),
             r.close>r.structure_high+BREAKOUT_ATR*r.atr14 and p.close<=r.structure_high),
            ("SHORT","DOWN",float(r.structure_low),
             r.close<r.structure_low-BREAKOUT_ATR*r.atr14 and p.close>=r.structure_low),
        ):
            if r.bar_close_at>=start and r.h1_regime==regime_key and cross:
                active={"at_idx":i,"side":side,"level":level,"atr":float(r.atr14),
                        "breakout_time":r.bar_close_at,"h1_regime":regime_key}
                cooldown=i
                break
    return pd.DataFrame.from_records(events,columns=["symbol","side","breakout_close_at",
           "decision_close_at","frozen_structure_level","atr_at_breakout","h1_regime","status"])


