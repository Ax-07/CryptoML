"""V14 frozen five-strategy exploratory markouts; never trading or ML."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from ingestion.storage import CandleStore
from mlpipeline.dataset import manifest_fingerprint
from mlpipeline.pullback_long_history_v3 import SYMBOLS,CUTOFF,load_long_development_history
from mlpipeline.multistrategy_census_v13 import PROTOCOL as V13_PROTOCOL,STRATEGIES,candidates
from mlpipeline.long_short_economics_audit_v8 import _barrier
from mlpipeline.trend_pullback import M5,_validate_closed_m5

PROTOCOL="p4_v14_frozen_five_family_economics_v1"
HORIZONS=dict(zip(STRATEGIES,(576,144,144,48,12)))
QUALIFY={**HORIZONS,"A05_V9_RETEST":24}
COSTS=(2,5,10)
REASONS=("gap_stop","gap_target_capped","both_hit_stop_first","stop","target","timeout")
EVENT_COLS=("strategy","symbol","side","timeframe","decision_close_at","reference_price","atr_known")


def verify_signals(reference:dict, csv:pd.DataFrame, computed:dict[str,pd.DataFrame], fingerprint:str)->None:
    if (reference.get("protocol")!=V13_PROTOCOL or
        reference.get("fingerprint_sha256")!=fingerprint or
        reference.get("assets")!=list(SYMBOLS) or set(computed)!=set(SYMBOLS)):
        raise ValueError("V13 census identity mismatch")
    expected=[]
    for sym in SYMBOLS:
        x=computed[sym]
        if not x.empty and not x.symbol.eq(sym).all():raise ValueError("Invalid computed symbols")
        expected.append(x.loc[:,EVENT_COLS])
        for strategy in STRATEGIES:
            for side in ("LONG","SHORT"):
                count=int((x.strategy.eq(strategy)&x.side.eq(side)).sum())
                if (count!=reference["by_asset"][sym][strategy][side]):
                    raise AssertionError(f"V13 counts mismatch {sym} {strategy} {side}")
    a=pd.concat(expected,ignore_index=True)
    if len(a)!=reference["total_signals"] or len(csv)!=len(a):
        raise AssertionError("V13 signal count mismatch")
    if csv.duplicated(["strategy","symbol","decision_close_at"]).any():
        raise ValueError("Duplicate signals")
    for f in (a,csv):
        f["decision_close_at"]=pd.to_datetime(f.decision_close_at,utc=True,errors="raise")
    order=["symbol","strategy","decision_close_at","side"]
    a=a.sort_values(order).reset_index(drop=True)
    b=csv.loc[:,EVENT_COLS].sort_values(order).reset_index(drop=True)
    for col in EVENT_COLS:
        if col in ("reference_price","atr_known"):
            aa=pd.to_numeric(a[col],errors="coerce")
            bb=pd.to_numeric(b[col],errors="coerce")
            if not np.allclose(aa,bb,rtol=1e-10,atol=1e-8,equal_nan=True):
                raise AssertionError(f"V13 event mismatch {col}")
        elif not a[col].equals(b[col]):
            raise AssertionError(f"V13 event mismatch {col}")


def score_symbol(bars:pd.DataFrame,events:pd.DataFrame):
    f=_validate_closed_m5(bars)
    loc={t:i for i,t in enumerate(f.bar_close_at)}
    rows=[];excluded={}
    for name in STRATEGIES:
        excluded[name]={"candidate":0,"matured":0,"missing_or_immature":0,"invalid_atr_or_price":0}
    for e in events.itertuples(index=False):
        s=e.strategy
        if s not in HORIZONS or e.symbol!=f.symbol.iloc[0] or e.side not in ("LONG","SHORT"):
            raise ValueError("Unapproved signal")
        count=excluded[s];count["candidate"]+=1
        idx=loc.get(e.decision_close_at)
        q=QUALIFY[s]
        if idx is None or idx+q>=len(f):
            count["missing_or_immature"]+=1;continue
        window=f.iloc[idx+1:idx+q+1]
        if (len(window)!=q or e.decision_close_at+M5*q>=CUTOFF or
            window.bar_open_at.iloc[0]!=e.decision_close_at or
            window.bar_close_at.iloc[-1]!=e.decision_close_at+M5*q or
            not window.bar_open_at.diff().iloc[1:].eq(M5).all()):
            count["missing_or_immature"]+=1;continue
        entry=float(window.open.iloc[0]);atr=float(e.atr_known)
        if (not np.isfinite(atr) or atr<=0 or not np.isfinite(entry) or
            entry<=1.5*atr):
            count["invalid_atr_or_price"]+=1;continue
        actual=window.iloc[:HORIZONS[s]]
        exit_price,reason,gross=_barrier(actual,entry,atr,e.side)
        if reason not in REASONS:raise AssertionError("Unexpected exit")
        row=dict(strategy=s,symbol=e.symbol,side=e.side,
                 decision_close_at=e.decision_close_at,entry_at=window.bar_open_at.iloc[0],
                 assumed_entry=entry,assumed_exit=exit_price,
                 atr_known=atr,horizon_M5=HORIZONS[s],exit_reason=reason,
                 gross_bps=gross,
                 **{f"net_{c}bps":gross-c for c in COSTS})
        rows.append(row);count["matured"]+=1
    for s,c in excluded.items():
        if c["candidate"]!=sum(c[k] for k in ("matured","missing_or_immature","invalid_atr_or_price")):
            raise AssertionError("Counts do not reconcile")
    return pd.DataFrame.from_records(rows,columns=[
        "strategy","symbol","side","decision_close_at","entry_at","assumed_entry",
        "assumed_exit","atr_known","horizon_M5","exit_reason","gross_bps",
        *(f"net_{c}bps" for c in COSTS)]),excluded


def stats(rows:pd.DataFrame)->dict:
    n=len(rows);gross=rows.gross_bps.to_numpy(dtype=float)
    if not np.isfinite(gross).all():raise ValueError("Invalid markouts")
    result={"trades":n,"gross_mean_bps":float(gross.mean()) if n else None,
        "gross_median_bps":float(np.median(gross)) if n else None,
        "sum_gross_bps":float(gross.sum()),
        "exit_reasons":{r:int(rows.exit_reason.eq(r).sum()) for r in REASONS}}
    if sum(result["exit_reasons"].values())!=n:raise AssertionError("Bad exits")
    for c in COSTS:
        net=rows[f"net_{c}bps"].to_numpy(dtype=float)
        if not np.allclose(net,gross-c,rtol=0,atol=1e-8):
            raise AssertionError("Net fee mismatch")
        result.update({f"net_{c}bps_mean":float(net.mean()) if n else None,
                       f"net_{c}bps_sum":float(net.sum()),
                       f"net_{c}bps_positive_fraction":float((net>0).mean()) if n else None})
    return result


def summarize(results:dict[str,pd.DataFrame],exclusions:dict,fingerprint:str):
    if set(results)!=set(SYMBOLS):raise ValueError("Only approved symbols")
    full=pd.concat([results[s] for s in SYMBOLS],ignore_index=True)
    full["month"]=full.decision_close_at.dt.strftime("%Y-%m") if not full.empty else pd.Series(dtype="object")
    by_strategy={};records=[]
    for s in STRATEGIES:
        selected=full.loc[full.strategy.eq(s)]
        by_strategy[s]={"ALL":stats(selected),"LONG":stats(selected.loc[selected.side.eq("LONG")]),
                         "SHORT":stats(selected.loc[selected.side.eq("SHORT")])}
        for (symbol,side,month),group in selected.groupby(["symbol","side","month"],sort=True):
            z=stats(group)
            records.append({"strategy":s,"symbol":symbol,"side":side,"month":month,
                            **{k:v for k,v in z.items() if k!="exit_reasons"}})
    for s in STRATEGIES:
        if sum(exclusions[x][s]["matured"] for x in SYMBOLS)!=by_strategy[s]["ALL"]["trades"]:
            raise AssertionError("Asset/global mismatch")
    return {"protocol":PROTOCOL,"status":"EXPLORATORY_NOT_INDEPENDENT_OOS",
      "fingerprint_sha256":fingerprint,"costs_roundtrip_bps":list(COSTS),
      "strategy_horizons_M5":HORIZONS,"common_qualification_M5":QUALIFY,
      "NO_TRADE_net_bps":0.0,"limitations":"OHLC markouts, overlapping events, no actual fills/funding/slippage or capital",
      "summary":by_strategy,"exclusions":exclusions},pd.DataFrame(records)


def main():
    p=argparse.ArgumentParser(description="V14 fixed five-family historical marks, NO ORDERS")
    for arg in ("data-root","reference-json","reference-signals-csv","output-json",
                "output-events-csv","output-monthly-csv"):
        p.add_argument("--"+arg,required=True,type=Path)
    a=p.parse_args();root=a.data_root.resolve()
    ref=a.reference_json.resolve();csv=a.reference_signals_csv.resolve()
    out=[x.resolve() for x in (a.output_json,a.output_events_csv,a.output_monthly_csv)]
    if (len(set(out))!=3 or not ref.is_file() or not csv.is_file() or
        ref.is_relative_to(root) or csv.is_relative_to(root) or
        any(x.exists() or x.is_relative_to(root) or not x.parent.is_dir() for x in out)):
        p.error("Require existing V13 reference and three NEW outputs outside P0")
    bars,manifests=load_long_development_history(CandleStore(root))
    fingerprint=manifest_fingerprint(manifests)
    with ref.open("r",encoding="utf-8") as f: reference=json.load(f)
    signals=pd.read_csv(csv)
    computed={}
    for sym in SYMBOLS:
        print("Revalidating frozen V13 candidates",sym,flush=True)
        computed[sym]=candidates(bars[sym])
    verify_signals(reference,signals,computed,fingerprint)
    print("V13 FINGERPRINT + EVENTS MATCH — scoring fixed windows",flush=True)
    results={};exclusions={}
    for sym in SYMBOLS:
        results[sym],exclusions[sym]=score_symbol(bars[sym],computed[sym])
        print(sym, {k:v["matured"] for k,v in exclusions[sym].items()},flush=True)
    report,monthly=summarize(results,exclusions,fingerprint)
    events=pd.concat([results[s] for s in SYMBOLS],ignore_index=True)
    with out[1].open("x",encoding="utf-8",newline="") as f:events.to_csv(f,index=False)
    with out[2].open("x",encoding="utf-8",newline="") as f:monthly.to_csv(f,index=False)
    with out[0].open("x",encoding="utf-8") as f:json.dump(report,f,indent=2,allow_nan=False)
    for s,info in report["summary"].items():
        print(s,"trades",info["ALL"]["trades"],"gross_mean",info["ALL"]["gross_mean_bps"],
              "net5_mean",info["ALL"]["net_5bps_mean"],flush=True)
    print("NO ORDERS / NO TRAINING / HISTORIC EXPLORATORY ONLY",flush=True)


if __name__=="__main__":main()
