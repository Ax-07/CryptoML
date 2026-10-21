"""Frozen V17 exploratory A06 economic markouts. Never trading or selecting by outcome."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ingestion.storage import CandleStore
from mlpipeline.dataset import manifest_fingerprint
from mlpipeline.pullback_long_history_v3 import SYMBOLS, CUTOFF
from mlpipeline.volume_funding_readiness_v16 import (
    PROTOCOL as V16_PROTOCOL, FIELDS, _load_full_p0, census_symbol,
)
from mlpipeline.long_short_economics_audit_v8 import _barrier
from mlpipeline.trend_pullback import M5, _validate_closed_m5

PROTOCOL = "p4_v17_a06_taker_volume_economics_v1"
HORIZON = 12
COSTS = (2, 5, 10)
REASONS = ("gap_stop", "gap_target_capped", "both_hit_stop_first", "stop", "target", "timeout")
OUTPUT_COLUMNS = ("symbol", "side", "decision_close_at", "entry_at", "entry_price",
                  "atr14_m15", "exit_price", "exit_reason", "gross_bps",
                  "net_2bps", "net_5bps", "net_10bps")


def validate_v16(reference:dict, csv:pd.DataFrame,
                 recounted:dict[str,pd.DataFrame], fingerprint:str)->None:
    if (reference.get("protocol") != V16_PROTOCOL or
        reference.get("fingerprint_sha256") != fingerprint or
        reference.get("source_partitions") != 84 or
        reference.get("symbols") != list(SYMBOLS) or
        set(recounted) != set(SYMBOLS)):
        raise ValueError("V16 reference identity mismatch")
    actual = pd.concat([recounted[s].loc[:, FIELDS] for s in SYMBOLS], ignore_index=True)
    if len(actual) != reference.get("total_signals") or len(csv) != len(actual):
        raise AssertionError("V16 event count mismatch")
    for s in SYMBOLS:
        x = recounted[s]
        stored = reference["by_symbol"][s]
        if int(x.side.eq("LONG").sum()) != stored["long"] or int(x.side.eq("SHORT").sum()) != stored["short"]:
            raise AssertionError(f"V16 side mismatch {s}")
    if csv.duplicated(["symbol","decision_close_at"]).any():
        raise ValueError("Duplicate V16 event")
    if not csv.empty:
        if not csv.symbol.isin(SYMBOLS).all() or not csv.side.isin(("LONG","SHORT")).all():
            raise ValueError("V16 invalid symbol/side")
    for frame in (actual,csv):
        frame["decision_close_at"] = pd.to_datetime(frame.decision_close_at,utc=True,errors="raise")
    ordering=["symbol","decision_close_at","side"]
    a=actual.sort_values(ordering).reset_index(drop=True)
    b=csv.loc[:,FIELDS].sort_values(ordering).reset_index(drop=True)
    for key in FIELDS:
        if key in ("reference_close","atr14_m15","taker_imbalance","relative_volume"):
            x=pd.to_numeric(a[key],errors="raise").to_numpy(dtype=float)
            y=pd.to_numeric(b[key],errors="raise").to_numpy(dtype=float)
            if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.allclose(x,y,rtol=1e-9,atol=1e-8):
                raise AssertionError(f"V16 mismatch {key}")
        elif not a[key].equals(b[key]):
            raise AssertionError(f"V16 mismatch {key}")


def score_symbol(candles:pd.DataFrame,signals:pd.DataFrame):
    f=_validate_closed_m5(candles)
    lookup={t:i for i,t in enumerate(f.bar_close_at)}
    output=[]
    tally={s:{"signals":0,"matured":0,"missing_or_immature":0,"invalid_atr_or_price":0}
           for s in ("LONG","SHORT")}
    for r in signals.itertuples(index=False):
        if r.symbol != f.symbol.iloc[0] or r.side not in tally:
            raise ValueError("Unexpected A06 event")
        counter=tally[r.side];counter["signals"]+=1
        i=lookup.get(r.decision_close_at)
        if i is None or i+HORIZON>=len(f):
            counter["missing_or_immature"]+=1
            continue
        window=f.iloc[i+1:i+HORIZON+1]
        if (len(window)!=HORIZON or r.decision_close_at+HORIZON*M5>=CUTOFF or
            window.bar_open_at.iloc[0]!=r.decision_close_at or
            window.bar_close_at.iloc[-1]!=r.decision_close_at+HORIZON*M5 or
            not window.bar_open_at.diff().iloc[1:].eq(M5).all()):
            counter["missing_or_immature"]+=1
            continue
        entry=float(window.open.iloc[0]);atr=float(r.atr14_m15)
        if not np.isfinite(entry) or not np.isfinite(atr) or atr<=0 or entry<=1.5*atr:
            counter["invalid_atr_or_price"]+=1
            continue
        exit_price,reason,gross=_barrier(window,entry,atr,r.side)
        if reason not in REASONS or not np.isfinite(gross):
            raise AssertionError("Invalid economics")
        output.append(dict(symbol=r.symbol,side=r.side,decision_close_at=r.decision_close_at,
                           entry_at=window.bar_open_at.iloc[0],entry_price=entry,
                           atr14_m15=atr,exit_price=float(exit_price),
                           exit_reason=reason,gross_bps=float(gross),
                           **{f"net_{fee}bps":float(gross-fee) for fee in COSTS}))
        counter["matured"]+=1
    for c in tally.values():
        if c["signals"] != c["matured"]+c["missing_or_immature"]+c["invalid_atr_or_price"]:
            raise AssertionError("A06 exclusions do not reconcile")
    return pd.DataFrame.from_records(output,columns=OUTPUT_COLUMNS),tally


def summarize(scores:dict, exclusions:dict, fingerprint:str, sha:str):
    if set(scores)!=set(SYMBOLS) or set(exclusions)!=set(SYMBOLS):
        raise ValueError("Only 4 approved assets")
    all_rows=pd.concat([scores[s] for s in SYMBOLS],ignore_index=True)
    def calc(x):
        n=len(x)
        gross=x.gross_bps.to_numpy(dtype=float)
        if not np.isfinite(gross).all():
            raise ValueError("Nonfinite returns")
        r={"count":n,"gross_mean_bps":float(gross.mean()) if n else None,
           "gross_median_bps":float(np.median(gross)) if n else None,
           "exit_reasons":{reason:int(x.exit_reason.eq(reason).sum()) for reason in REASONS}}
        if sum(r["exit_reasons"].values())!=n:
            raise AssertionError("Exit reconciliation")
        for fee in COSTS:
            net=x[f"net_{fee}bps"].to_numpy(dtype=float)
            if not np.allclose(net,gross-fee,rtol=0,atol=1e-8):
                raise AssertionError("Incorrect fees")
            r[f"net_{fee}bps_mean"]=float(net.mean()) if n else None
            r[f"net_{fee}bps_positive_fraction"]=float((net>0).mean()) if n else None
        return r
    by_asset={}
    for sym in SYMBOLS:
        by_asset[sym]={side:calc(scores[sym].loc[scores[sym].side.eq(side)]) for side in ("LONG","SHORT")}
        by_asset[sym]["ALL"]=calc(scores[sym])
    by_side={side:calc(all_rows.loc[all_rows.side.eq(side)]) for side in ("LONG","SHORT")}
    by_side["ALL"]=calc(all_rows)
    if len(all_rows) != sum(exclusions[s][side]["matured"] for s in SYMBOLS for side in ("LONG","SHORT")):
        raise AssertionError("Global event count does not reconcile")
    return {"protocol":PROTOCOL,"status":"EXPLORATORY_ALREADY_INSPECTED_HISTORY_NOT_OOS",
            "fingerprint_sha256":fingerprint,"v16_reference_csv_sha256":sha,
            "symbols":list(SYMBOLS),"signal_family":"A06_VOLUME_TAKER_M15",
            "horizon_m5":HORIZON,"stop_atr":1.,"target_atr":1.5,
            "roundtrip_assumed_cost_bps":list(COSTS),"NO_TRADE_bp":0.,
            "by_symbol":by_asset,"by_side":by_side,"exclusions":exclusions,
            "limitations":"Overlapping theoretical markouts, no funding, changing spreads, slippage, actual fills, leverage, capital or independent OOS"},all_rows


def main():
    p=argparse.ArgumentParser(description="V17 A06 economics fixed before looking at any A06 returns")
    for name in ("data-root","reference-json","reference-csv","output-json","output-csv"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();root=a.data_root.resolve()
    ref=a.reference_json.resolve();reference_csv=a.reference_csv.resolve()
    out=(a.output_json.resolve(),a.output_csv.resolve())
    if (not ref.is_file() or not reference_csv.is_file() or
        ref.is_relative_to(root) or reference_csv.is_relative_to(root) or
        out[0]==out[1] or any(x.exists() or x.is_relative_to(root)
                                or not x.parent.is_dir() for x in out)):
        p.error("Existing V16 references and TWO NEW outputs outside P0 are required")
    candles,manifests=_load_full_p0(CandleStore(root))
    fingerprint=manifest_fingerprint(manifests)
    with ref.open("r",encoding="utf-8") as f: reference=json.load(f)
    csv=pd.read_csv(reference_csv)
    recomputed={}
    for sym in SYMBOLS:
        print("Revalidate V16 volume/taker + signals",sym,flush=True)
        recomputed[sym],_=census_symbol(candles[sym])
    validate_v16(reference,csv,recomputed,fingerprint)
    print("V16 FINGERPRINT + ALL EVENTS MATCH — scoring frozen A06 economics",flush=True)
    score={};counts={}
    for sym in SYMBOLS:
        score[sym],counts[sym]=score_symbol(candles[sym],recomputed[sym])
        print(sym,{side:z["matured"] for side,z in counts[sym].items()},flush=True)
    report,events=summarize(score,counts,fingerprint,hashlib.sha256(reference_csv.read_bytes()).hexdigest())
    with out[1].open("x",encoding="utf-8",newline="") as f:events.to_csv(f,index=False)
    with out[0].open("x",encoding="utf-8") as f:json.dump(report,f,indent=2,allow_nan=False)
    for side,stat in report["by_side"].items():
        print(side,"n",stat["count"],"gross",stat["gross_mean_bps"],
              "net2",stat["net_2bps_mean"],"net5",stat["net_5bps_mean"],
              "net10",stat["net_10bps_mean"],flush=True)
    print("HISTORICAL ONLY / NO ORDERS / NO ML",flush=True)


if __name__=="__main__":main()
