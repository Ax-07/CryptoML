"""V9 PAPER ONLY: causal breakout/retest detector, no backtest or orders."""
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


def detect(bars:pd.DataFrame,*,start:pd.Timestamp=PROSPECTIVE_START)->pd.DataFrame:
    """Returns decision-time detections only; other candles used for warmup.

    The entire dataframe is already-known *offline input*. Passing prospective
    start does NOT attest that any result was live or contemporaneously saved.
    """
    if start.tzinfo is None or start<PROSPECTIVE_START:
        raise ValueError("V9 cannot issue historical research signals")
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
        if (r.bar_open_at-p.bar_open_at!=M5 or
                not np.isfinite(r.atr14) or r.atr14<=0 or
                pd.isna(r.level_known_at) or
                r.level_known_at>r.bar_close_at-pd.Timedelta(minutes=15)):
            continue
        for side,regime_key,level,cross in (
            ("LONG","UP",float(r.structure_high),
             r.close>r.structure_high+BREAKOUT_ATR*r.atr14 and p.close<=r.structure_high),
            ("SHORT","DOWN",float(r.structure_low),
             r.close<r.structure_low-BREAKOUT_ATR*r.atr14 and p.close>=r.structure_low),
        ):
            if r.h1_regime==regime_key and cross:
                active={"at_idx":i,"side":side,"level":level,"atr":float(r.atr14),
                        "breakout_time":r.bar_close_at,"h1_regime":regime_key}
                cooldown=i
                break
    return pd.DataFrame.from_records(events,columns=["symbol","side","breakout_close_at",
           "decision_close_at","frozen_structure_level","atr_at_breakout","h1_regime","status"])


def main():
    parser=argparse.ArgumentParser(description="Offline V9 paper-signal replay — NEVER live proof")
    parser.add_argument("--input-csv",required=True,type=Path,
                        help="One asset, real closed M5 UTC candles with symbol/open/high/low/close/is_final")
    parser.add_argument("--output-json",required=True,type=Path)
    args=parser.parse_args()
    src=args.input_csv.resolve()
    dest=args.output_json.resolve()
    if not src.is_file() or dest.exists() or not dest.parent.is_dir() or dest==src:
        parser.error("Existing input CSV and NEW output JSON required")
    f=pd.read_csv(src)
    for c in ("bar_open_at","bar_close_at"):
        if c not in f:
            parser.error("CSV requires UTC bar_open_at and bar_close_at")
        f[c]=pd.to_datetime(f[c],utc=True,errors="raise")
    if "is_final" not in f or not f.is_final.isin((True,False,"True","False","true","false",1,0)).all():
        parser.error("CSV requires explicit is_final flags")
    f["is_final"]=f.is_final.astype(str).str.lower().eq("true")
    signals=detect(f)
    result={"protocol":PROTOCOL,"mode":"OFFLINE_REPLAY_NOT_LIVE_ATTESTED",
            "market":"Binance USD-M P0-style OHLCV; input claimed by operator, not exchange-verified",
            "no_returns_no_order_execution":True,
            "signals":[{k:(v.isoformat() if isinstance(v,pd.Timestamp) else v)
                       for k,v in row.items()} for row in signals.to_dict("records")]}
    with dest.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print("V9 offline paper signals",len(result["signals"]))
    print("NOT LIVE ATTESTED; no PnL, trading or independent performance claim")
    print("Saved",dest)


if __name__=="__main__":
    main()
