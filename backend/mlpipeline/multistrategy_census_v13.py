"""V13 fixed A01-A05 causal, UNLABELED candidate census. No trades or returns."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ingestion.storage import CandleStore
from mlpipeline.dataset import manifest_fingerprint
from mlpipeline.market_structure_v6 import _segment_features
from mlpipeline.pullback_long_history_v3 import SYMBOLS, load_long_development_history
from mlpipeline.structure_breakout_retest_historical_v11 import detect as v9_research
from mlpipeline.trend_pullback import M5, _validate_closed_m5

PROTOCOL="p4_v13_five_strategy_unlabeled_census_v1"
COOLDOWN=12
STRATEGIES=("A01_MOMENTUM_H4","A02_DONCHIAN_H1",
            "A03_COMPRESSION_H1","A04_MEAN_REVERSION_H1","A05_V9_RETEST")
FIELDS=("strategy","symbol","side","timeframe","decision_close_at",
        "reference_price","atr_known")


def aggregate(bars:pd.DataFrame,factor:int)->pd.DataFrame:
    """Complete aligned blocks, never bridge missing M5 sub-bars."""
    if factor not in (12,48):
        raise ValueError("Only H1/H4")
    f=_validate_closed_m5(bars)
    size=factor*5*60*1_000_000_000
    stamps=f.bar_close_at.astype("int64").to_numpy()
    ends=((stamps-1)//size+1)*size
    f=f.copy()
    f["_bucket"]=pd.to_datetime(ends,utc=True)
    result=[]
    for close_time,g in f.groupby("_bucket",sort=True):
        if (len(g)!=factor or g.bar_close_at.iloc[-1]!=close_time
            or not g.bar_open_at.diff().iloc[1:].eq(M5).all()):
            continue
        result.append({"symbol":g.symbol.iloc[0],
          "bar_open_at":g.bar_open_at.iloc[0],"bar_close_at":close_time,
          "open":float(g.open.iloc[0]),"high":float(g.high.max()),
          "low":float(g.low.min()),"close":float(g.close.iloc[-1]),
          "is_final":True})
    return pd.DataFrame.from_records(result,columns=[
        "symbol","bar_open_at","bar_close_at","open","high","low","close","is_final"])


def _features(frame:pd.DataFrame,minutes:int)->pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    x=frame.copy().reset_index(drop=True)
    x["_seg"]=x.bar_open_at.diff().ne(pd.Timedelta(minutes=minutes)).cumsum()
    blocks=[]
    for _,part in x.groupby("_seg",sort=False):
        p=_segment_features(part.reset_index(drop=True))
        c=p.close
        p["sma20"]=c.rolling(20,min_periods=20).mean()
        p["std20"]=c.rolling(20,min_periods=20).std(ddof=0)
        p["sma24"]=c.rolling(24,min_periods=24).mean()
        p["r24"]=np.log(c/c.shift(24))
        ret=np.log(c/c.shift(1))
        p["vol24"]=ret.rolling(24,min_periods=24).std(ddof=0)*np.sqrt(24)
        p["d_high20"]=p.high.shift(1).rolling(20,min_periods=20).max()
        p["d_low20"]=p.low.shift(1).rolling(20,min_periods=20).min()
        p["box_high12"]=p.high.shift(1).rolling(12,min_periods=12).max()
        p["box_low12"]=p.low.shift(1).rolling(12,min_periods=12).min()
        width=(p["box_high12"]-p["box_low12"])/p.atr14.replace(0,np.nan)
        p["prior_width"]=width.shift(1)
        p["prior_width_median48"]=width.shift(2).rolling(48,min_periods=48).median()
        change=c.diff(12).abs()
        path=c.diff().abs().rolling(12,min_periods=12).sum()
        p["er12"]=change/path.replace(0,np.nan)
        blocks.append(p)
    return pd.concat(blocks,ignore_index=True)


def _select(f:pd.DataFrame,strategy:str,tf:str,mode:str)->pd.DataFrame:
    records=[]
    last=-COOLDOWN
    for i in range(len(f)):
        r=f.iloc[i]
        # All tests use only fully closed candles and all state resets at gaps.
        side=None
        if strategy=="A01_MOMENTUM_H4":
            if r.bar_close_at.hour!=0 or r.bar_close_at.minute!=0:
                continue
            score=r.r24/r.vol24 if pd.notna(r.vol24) and r.vol24>0 else np.nan
            if score>=1 and r.close>r.sma24: side="LONG"
            elif score<=-1 and r.close<r.sma24: side="SHORT"
        elif strategy=="A02_DONCHIAN_H1":
            if pd.notna(r.atr14) and r.atr14>0:
                if r.close>r.d_high20+0.1*r.atr14: side="LONG"
                elif r.close<r.d_low20-0.1*r.atr14: side="SHORT"
        elif strategy=="A03_COMPRESSION_H1":
            if (pd.notna(r.prior_width) and pd.notna(r.prior_width_median48)
                and r.prior_width<=0.7*r.prior_width_median48 and
                pd.notna(r.atr14) and r.atr14>0):
                if r.close>r.box_high12+0.1*r.atr14: side="LONG"
                elif r.close<r.box_low12-0.1*r.atr14: side="SHORT"
        else:
            if i==0 or r.bar_open_at-f.bar_open_at.iloc[i-1]!=pd.Timedelta(minutes=60):
                continue
            p=f.iloc[i-1]
            low_band=p.sma20-2*p.std20
            high_band=p.sma20+2*p.std20
            if pd.notna(p.er12) and p.er12<=0.30:
                if (p.close<low_band and r.close>low_band and
                    r.close<p.sma20 and r.close>r.open): side="LONG"
                elif (p.close>high_band and r.close<high_band and
                      r.close>p.sma20 and r.close<r.open): side="SHORT"
        if side is None:continue
        if strategy!="A01_MOMENTUM_H4" and i-last<COOLDOWN:continue
        if strategy!="A01_MOMENTUM_H4":last=i
        records.append((strategy,str(r.symbol),side,tf,r.bar_close_at,
                        float(r.close),float(r.atr14) if pd.notna(r.atr14) else None))
    return pd.DataFrame.from_records(records,columns=FIELDS)


def candidates(bars:pd.DataFrame)->pd.DataFrame:
    f=_validate_closed_m5(bars)
    h1=_features(aggregate(f,12),60)
    h4=_features(aggregate(f,48),240)
    all_frames=[]
    for key,source,tf in (
        ("A01_MOMENTUM_H4",h4,"H4"),
        ("A02_DONCHIAN_H1",h1,"H1"),
        ("A03_COMPRESSION_H1",h1,"H1"),
        ("A04_MEAN_REVERSION_H1",h1,"H1"),
    ):
        if not source.empty:
            all_frames.append(_select(source,key,tf,"both"))
    historical=v9_research(f,start=pd.Timestamp("2024-10-01T00:00:00Z"))
    if not historical.empty:
        times=f.set_index("bar_close_at")
        # V9 entry is known at retest close. Use the close only as a reference.
        close=times.loc[list(historical.decision_close_at),"close"].to_numpy(dtype=float)
        all_frames.append(pd.DataFrame({
            "strategy":"A05_V9_RETEST","symbol":historical.symbol.to_numpy(),
            "side":historical.side.to_numpy(),"timeframe":"M5",
            "decision_close_at":historical.decision_close_at.to_numpy(),
            "reference_price":close,"atr_known":historical.atr_at_breakout.to_numpy(dtype=float)
        }))
    merged=pd.concat(all_frames,ignore_index=True) if all_frames else pd.DataFrame(columns=FIELDS)
    if not merged.empty:
        merged["decision_close_at"]=pd.to_datetime(merged.decision_close_at,utc=True)
        merged=merged.sort_values(["decision_close_at","strategy"]).reset_index(drop=True)
        if merged.duplicated(["strategy","symbol","decision_close_at"]).any():
            raise AssertionError("Multiple signals for strategy/symbol/close")
        if (not merged.side.isin(("LONG","SHORT")).all() or
            not merged.strategy.isin(STRATEGIES).all()):
            raise AssertionError("Invalid candidate")
    return merged


def census(frames:dict[str,pd.DataFrame],fingerprint:str):
    if set(frames)!=set(SYMBOLS) or len(fingerprint)!=64:
        raise ValueError("Four approved assets and fingerprint required")
    all_signals=[];per_asset={};per_month=[]
    for symbol in SYMBOLS:
        sig=candidates(frames[symbol])
        if not sig.empty and not sig.symbol.eq(symbol).all():
            raise AssertionError("Symbol mismatch")
        all_signals.append(sig)
        per_asset[symbol]={s:{side:int((sig.strategy.eq(s)&sig.side.eq(side)).sum())
                                  for side in ("LONG","SHORT")} for s in STRATEGIES}
        for strategy in STRATEGIES:
            for side in ("LONG","SHORT"):
                x=sig.loc[sig.strategy.eq(strategy)&sig.side.eq(side)]
                for month,n in x.decision_close_at.dt.strftime("%Y-%m").value_counts().sort_index().items():
                    per_month.append({"symbol":symbol,"strategy":strategy,
                                      "side":side,"month":month,"count":int(n)})
    combined=pd.concat(all_signals,ignore_index=True)
    totals={s:{side:sum(per_asset[x][s][side] for x in SYMBOLS) for side in ("LONG","SHORT")}
            for s in STRATEGIES}
    return {"protocol":PROTOCOL,"mode":"UNLABELED_NO_BACKTEST",
            "fingerprint_sha256":fingerprint,"assets":list(SYMBOLS),
            "counts":totals,"by_asset":per_asset,
            "total_signals":len(combined),
            "warning":"Historical inspected sample, no returns and no independent OOS"},(
            combined,pd.DataFrame(per_month,columns=["symbol","strategy","side","month","count"]))


def main():
    p=argparse.ArgumentParser(description="V13 5-family causal UNLABELED census")
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--output-json",type=Path,required=True)
    p.add_argument("--output-signals-csv",type=Path,required=True)
    p.add_argument("--output-monthly-csv",type=Path,required=True)
    a=p.parse_args()
    root=a.data_root.resolve()
    outs=[x.resolve() for x in (a.output_json,a.output_signals_csv,a.output_monthly_csv)]
    if len(set(outs))!=3 or any(x.exists() or x.is_relative_to(root) or not x.parent.is_dir() for x in outs):
        p.error("Three NEW outputs outside P0 required")
    bars,manifests=load_long_development_history(CandleStore(root))
    print("V13: evaluating five predeclared families on the four SHA-verified symbols",flush=True)
    report,(signals,monthly)=census(bars,manifest_fingerprint(manifests))
    with outs[1].open("x",encoding="utf-8",newline="") as f:signals.to_csv(f,index=False)
    with outs[2].open("x",encoding="utf-8",newline="") as f:monthly.to_csv(f,index=False)
    with outs[0].open("x",encoding="utf-8") as f:json.dump(report,f,indent=2,ensure_ascii=False)
    for k,v in report["counts"].items():print(k,v,flush=True)
    print("NO ECONOMIC OUTCOMES, NO TRAINING, NO ORDERS",outs[0])


if __name__=="__main__":main()
