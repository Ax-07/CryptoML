"""V11 historical research economics, fixed before looking at V9 outcomes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ingestion.storage import CandleStore
from mlpipeline.dataset import manifest_fingerprint
from mlpipeline.long_short_economics_audit_v8 import _barrier
from mlpipeline.pullback_long_history_v3 import (
    SYMBOLS, CUTOFF, load_long_development_history,
)
from mlpipeline.structure_breakout_retest_historical_v11 import detect
from mlpipeline.trend_pullback import M5, _validate_closed_m5

PROTOCOL="p4_v11_frozen_breakout_retest_exploratory_economics_v1"
HORIZONS=(6,12,24)
PRIMARY=12
COSTS=(2,5,10)
REASONS=("gap_stop","gap_target_capped","both_hit_stop_first","stop","target","timeout")
FIRST=pd.Timestamp("2024-10-01T00:00:00Z")
MATURITY=pd.Timedelta(minutes=120)


def score_symbol(bars:pd.DataFrame, signals:pd.DataFrame):
    """One identical 24-bar-matured population for all prespecified horizons."""
    f=_validate_closed_m5(bars)
    lookup={t:i for i,t in enumerate(f.bar_close_at)}
    output=[]
    exclusions={"candidate_decisions":len(signals),"matured":0,
                "no_contiguous_full_24bar_window":0,"invalid_barrier":0}
    for row in signals.itertuples(index=False):
        if row.side not in ("LONG","SHORT") or row.symbol!=f.symbol.iloc[0]:
            raise ValueError("Incorrect V9 side/symbol")
        i=lookup.get(row.decision_close_at)
        if i is None or i+24>=len(f):
            exclusions["no_contiguous_full_24bar_window"]+=1
            continue
        window=f.iloc[i+1:i+25]
        if (len(window)!=24 or row.decision_close_at+MATURITY>=CUTOFF
            or window.bar_open_at.iloc[0]!=row.decision_close_at
            or window.bar_close_at.iloc[-1]!=row.decision_close_at+MATURITY
            or not window.bar_open_at.diff().iloc[1:].eq(M5).all()):
            exclusions["no_contiguous_full_24bar_window"]+=1
            continue
        entry=float(window.open.iloc[0])
        atr=float(row.atr_at_breakout)
        if not np.isfinite(atr) or atr<=0 or entry<=1.5*atr:
            exclusions["invalid_barrier"]+=1
            continue
        event={"symbol":row.symbol,"side":row.side,
               "breakout_close_at":row.breakout_close_at,
               "decision_close_at":row.decision_close_at,
               "entry_at":window.bar_open_at.iloc[0],"entry_price":entry,
               "atr_at_breakout":atr,
               "regime":row.h1_regime}
        for horizon in HORIZONS:
            sub=window.iloc[:horizon]
            out,reason,gross=_barrier(sub,entry,atr,row.side)
            tag=f"h{horizon}"
            event.update({f"{tag}_exit_price":float(out),
                          f"{tag}_exit_reason":reason,
                          f"{tag}_gross_bps":float(gross),
                          **{f"{tag}_net_{cost}bps":float(gross-cost) for cost in COSTS}})
        output.append(event)
    columns=["symbol","side","breakout_close_at","decision_close_at","entry_at",
             "entry_price","atr_at_breakout","regime"]
    for h in HORIZONS:
        columns += [f"h{h}_exit_price",f"h{h}_exit_reason",f"h{h}_gross_bps"]
        columns += [f"h{h}_net_{cost}bps" for cost in COSTS]
    result=pd.DataFrame.from_records(output,columns=columns)
    exclusions["matured"]=len(result)
    if (sum(exclusions[k] for k in ("matured","no_contiguous_full_24bar_window",
                                   "invalid_barrier")) != exclusions["candidate_decisions"]):
        raise AssertionError("Opportunity exclusions do not reconcile")
    return result,exclusions


def _stats(events:pd.DataFrame,horizon:int)->dict:
    n=len(events)
    key=f"h{horizon}"
    gross=events[f"{key}_gross_bps"].to_numpy(dtype=float)
    if not np.isfinite(gross).all():
        raise ValueError("Nonfinite gross result")
    out={"trades":n,"sum_gross_bps":float(gross.sum()),
         "gross_bps_per_trade":float(gross.mean()) if n else None,
         "gross_bps_median":float(np.median(gross)) if n else None,
         "gross_positive_fraction":float((gross>0).mean()) if n else None,
         "exit_reasons":{r:int(events[f"{key}_exit_reason"].eq(r).sum()) for r in REASONS}}
    if sum(out["exit_reasons"].values())!=n:
        raise AssertionError("Invalid outcome reason")
    for cost in COSTS:
        net=events[f"{key}_net_{cost}bps"].to_numpy(dtype=float)
        if not np.allclose(net,gross-cost,atol=1e-8,rtol=0):
            raise AssertionError("Costs incorrect")
        out.update({f"sum_net_{cost}bps":float(net.sum()),
                    f"net_{cost}bps_per_trade":float(net.mean()) if n else None,
                    f"net_{cost}bps_positive_fraction":float((net>0).mean()) if n else None})
    return out


def report(events:dict, exclusions:dict, fingerprint:str)->tuple[dict,pd.DataFrame]:
    if set(events)!=set(SYMBOLS) or set(exclusions)!=set(SYMBOLS):
        raise ValueError("V11 only four approved symbols")
    all_frames=[];rows=[];detail={}
    for symbol in SYMBOLS:
        group=events[symbol]
        if not group.empty:
            if (not group.symbol.eq(symbol).all() or
                group.decision_close_at.duplicated().any() or
                not group.decision_close_at.is_monotonic_increasing or
                not group.decision_close_at.lt(CUTOFF).all() or
                not group.side.isin(("LONG","SHORT")).all()):
                raise ValueError("Malformed events")
            group=group.copy()
            group["month"]=group.decision_close_at.dt.strftime("%Y-%m")
        all_frames.append(group)
        detail[symbol]={}
        for side in ("LONG","SHORT","BOTH"):
            subset=group if side=="BOTH" or group.empty else group.loc[group.side.eq(side)]
            detail[symbol][side]={str(h):_stats(subset,h) for h in HORIZONS}
            if subset.empty:
                continue
            for month,m in subset.groupby("month",sort=True):
                for h in HORIZONS:
                    stats=_stats(m,h)
                    rows.append({"symbol":symbol,"side":side,"month":month,
                                 "horizon_bars":h,**{k:v for k,v in stats.items() if k!="exit_reasons"}})
    full=pd.concat(all_frames,ignore_index=True)
    combined={}
    for side in ("LONG","SHORT","BOTH"):
        subset=full if side=="BOTH" or full.empty else full.loc[full.side.eq(side)]
        combined[side]={str(h):_stats(subset,h) for h in HORIZONS}
    total_candidates=sum(x["candidate_decisions"] for x in exclusions.values())
    total_matured=sum(x["matured"] for x in exclusions.values())
    if len(full)!=total_matured:
        raise AssertionError("Global matured-event reconciliation failure")
    return ({
        "protocol":PROTOCOL,
        "status":"EXPLORATORY_ALREADY_INSPECTED_HISTORY_NOT_OOS",
        "source_dataset_fingerprint_sha256":fingerprint,
        "symbols":list(SYMBOLS),"period":"2024-10 through 2026-06",
        "candidate_decisions":total_candidates,"common_matured_events":total_matured,
        "primary_horizon_M5_bars":PRIMARY,"fixed_sensitivities":list(HORIZONS),
        "fixed_assumed_roundtrip_cost_bps":list(COSTS),
        "method":"Hypothetical OPEN immediately after confirmed retest; 1ATR stop/1.5ATR target, stop-first, gap-adverse, fixed 24-bar common maturity",
        "limitations":"Overlapping theoretical trades; no funding, spread/slippage, latency, leverage or execution realism; no independent validation",
        "no_trade_baseline_bps":0,
        "by_asset":detail,"exclusions":exclusions,"overall":combined,
    },pd.DataFrame(rows))


def main():
    p=argparse.ArgumentParser(description="V11 frozen V9 economics exploratory only, no orders")
    p.add_argument("--data-root",required=True,type=Path)
    p.add_argument("--output-json",required=True,type=Path)
    p.add_argument("--output-csv",required=True,type=Path)
    a=p.parse_args()
    root=a.data_root.resolve()
    jout=a.output_json.resolve()
    cout=a.output_csv.resolve()
    if (jout==cout or jout.parent!=cout.parent or not jout.parent.is_dir() or
        any(f.exists() or f.is_relative_to(root) for f in (jout,cout))):
        p.error("Need two NEW results in same directory outside P0")
    candles,manifests=load_long_development_history(CandleStore(root))
    per={};counts={}
    for symbol in SYMBOLS:
        print("V11 frozen research:",symbol,flush=True)
        signals=detect(candles[symbol],start=FIRST)
        events,count=score_symbol(candles[symbol],signals)
        per[symbol]=events;counts[symbol]=count
        print(symbol,count,"LONG",int(events.side.eq("LONG").sum()) if not events.empty else 0,
              "SHORT",int(events.side.eq("SHORT").sum()) if not events.empty else 0,flush=True)
    summary,monthly=report(per,counts,manifest_fingerprint(manifests))
    with cout.open("x",encoding="utf-8",newline="") as out:
        monthly.to_csv(out,index=False)
    with jout.open("x",encoding="utf-8") as out:
        json.dump(summary,out,indent=2,ensure_ascii=False,allow_nan=False)
    for side in ("LONG","SHORT","BOTH"):
        for h in HORIZONS:
            s=summary["overall"][side][str(h)]
            print(side,f"h{h}","n",s["trades"],"gross/trade",s["gross_bps_per_trade"],
                  "net5/trade",s["net_5bps_per_trade"])
    print("V11 NOT OOS / NO EXECUTION / NO PARAMETER TUNING")
    print("Saved",jout,cout)


if __name__=="__main__":
    main()
