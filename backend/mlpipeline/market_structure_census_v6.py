"""Unlabeled P4 V6 direction/regime census on immutable pre-July history."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from ingestion.storage import CandleStore
from mlpipeline.dataset import manifest_fingerprint
from mlpipeline.pullback_long_history_v3 import (
    CUTOFF, EXPECTED_MONTHS, SYMBOLS, load_long_development_history,
)
from mlpipeline.market_structure_v6 import classify_m5, REGIMES

PROTOCOL="p4_v6_m15_h1_symmetric_long_short_unlabeled_v1"
COLUMNS=("long_raw","short_raw","long_signal","short_signal",
         "long_with_trend","short_with_trend")


def summarize(features_by_symbol: dict[str,pd.DataFrame],fingerprint:str):
    if set(features_by_symbol)!=set(SYMBOLS) or re.fullmatch("[0-9a-f]{64}",fingerprint) is None:
        raise ValueError("Exactly four approved assets and valid fingerprint required")
    rows=[]
    summary={}
    for symbol in SYMBOLS:
        f=features_by_symbol[symbol]
        if (f.empty or not f.symbol.eq(symbol).all()
                or not f.bar_close_at.is_unique or not f.bar_close_at.is_monotonic_increasing
                or not f.bar_close_at.lt(CUTOFF).all()
                or any(not pd.api.types.is_bool_dtype(f[c]) for c in COLUMNS)
                or not f.market_regime.isin(REGIMES).all()):
            raise ValueError(f"{symbol}: invalid feature chronology or categories")
        bymonth={}
        for month,group in f.groupby(f.bar_close_at.dt.strftime("%Y-%m"),sort=True):
            if month not in EXPECTED_MONTHS:
                raise ValueError("Month outside development")
            counts={c:int(group[c].sum()) for c in COLUMNS}
            for name in REGIMES:
                counts[f"regime_{name.lower()}"]=int(group.market_regime.eq(name).sum())
                for side in ("long_signal","short_signal"):
                    counts[f"{side}_in_{name.lower()}"]=int(
                        (group[side]&group.market_regime.eq(name)).sum())
            counts["closed_bars"]=int(len(group))
            if sum(counts[f"regime_{name.lower()}"] for name in REGIMES)!=len(group):
                raise AssertionError("Regimes do not reconcile")
            bymonth[month]=counts
            rows.append({"symbol":symbol,"month":month,**counts})
        if tuple(bymonth)!=EXPECTED_MONTHS:
            raise ValueError("Missing historical months")
        overall={k:sum(item[k] for item in bymonth.values())
                 for k in next(iter(bymonth.values()))}
        for side in ("long_signal","short_signal"):
            if sum(overall[f"{side}_in_{r.lower()}"] for r in REGIMES)!=overall[side]:
                raise AssertionError("Direction×regime breakdown mismatch")
        summary[symbol]={"total":overall,"by_month":bymonth}
    grand={k:sum(summary[s]["total"][k] for s in SYMBOLS)
           for k in summary[SYMBOLS[0]]["total"]}
    return {
        "protocol":PROTOCOL,
        "status":"EXPLORATORY_UNLABELED_NOT_INDEPENDENT_OOS",
        "source_dataset_fingerprint_sha256":fingerprint,
        "assets":list(SYMBOLS),
        "period":"2024-10 through 2026-06",
        "market_type":"Binance USDT perpetual OHLCV; NO EXCHANGE ORDERS",
        "direction":"symmetric LONG/SHORT candidates, context M15+H1",
        "no_labels_no_outcomes_no_ml":True,
        "note":"A regime label is a descriptive model construct, not an established market truth. Not a tradable strategy.",
        "total":grand,"by_asset":summary,
    },pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description="P4 V6 unlabeled LONG/SHORT + regime census only")
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--output-json",type=Path,required=True)
    p.add_argument("--output-csv",type=Path,required=True)
    a=p.parse_args()
    root=a.data_root.resolve()
    jpath,cpath=a.output_json.resolve(),a.output_csv.resolve()
    if (jpath==cpath or not jpath.parent.is_dir() or jpath.parent!=cpath.parent
            or any(path.exists() or path.is_relative_to(root) for path in (jpath,cpath))):
        p.error("Two NEW separate output files in same existing dir outside P0")
    candles,manifests=load_long_development_history(CandleStore(root))
    frames={}
    for symbol in SYMBOLS:
        print("Computing causal M5/M15/H1",symbol,flush=True)
        frames[symbol]=classify_m5(candles[symbol])
    report,table=summarize(frames,manifest_fingerprint(manifests))
    with cpath.open("x",encoding="utf-8",newline="") as stream:
        table.to_csv(stream,index=False)
    with jpath.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print("TOTAL", {k:v for k,v in report["total"].items() if k in COLUMNS})
    print("REGIMES", {k:v for k,v in report["total"].items() if k.startswith("regime_")})
    print("UNLABELED ONLY — NO RETURNS / NO SHORT ORDER / NO LIVE TRADING")
    print("Saved",jpath,cpath)


if __name__=="__main__":
    main()
