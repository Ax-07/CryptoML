"""Read-only fixed A/B historical economics audit; no ML or optimization.

Only BTC/ETH/SOL/XRP development candles before July 2026, verified via
P4 V3 loader. Outcomes: unchanged P4 V1 entry, stop/target, timeout.
All results are descriptive on already inspected research history.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlpipeline.pullback_global_ridge_v3 import prepare_symbol
from mlpipeline.pullback_long_history_v3 import SYMBOLS, load_long_development_history
from mlpipeline.dataset import manifest_fingerprint
from mlpipeline.pullback_baselines import _score

PROTOCOL = "p4_v4_mechanical_AB_economics_frozen_audit"
COSTS = (2, 5, 10)
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
REASONS = ("gap_stop","gap_target_capped","both_hit_stop_first",
           "stop","target","timeout")


def _stats(group: pd.DataFrame, all_opportunities: int) -> dict:
    """Descriptive markout distribution; no parameter or subgroup selection."""
    result = _score(group, total_opportunities=all_opportunities)
    n = len(group)
    gross = group.gross_bps.to_numpy(dtype=float)
    if not np.isfinite(gross).all():
        raise ValueError("Nonfinite gross markout")
    result["gross_bps_mean"] = float(gross.mean()) if n else None
    result["gross_bps_median"] = float(np.median(gross)) if n else None
    result["gross_bps_std"] = float(gross.std(ddof=1)) if n>1 else None
    result["gross_bps_quantiles"] = {
        f"p{int(round(q*100)):02d}": float(np.quantile(gross,q)) if n else None
        for q in QUANTILES
    }
    result["gross_positive_trades"] = int((gross>0).sum())
    result["gross_zero_trades"] = int((gross==0).sum())
    result["gross_negative_trades"] = int((gross<0).sum())
    result["sum_gross_bps"] = float(gross.sum())
    result["exit_reasons_fixed"] = {
        reason: int(group.exit_reason.eq(reason).sum()) for reason in REASONS
    }
    if not group.exit_reason.isin(REASONS).all():
        raise ValueError("Unknown exit reason; do not invent bucket")
    if sum(result["exit_reasons_fixed"].values())!=n:
        raise AssertionError("Exit reasons do not reconcile")
    for cost in COSTS:
        key = f"net_{cost}bps"
        net = group[key].to_numpy(dtype=float)
        if not np.allclose(net, gross-cost, rtol=0, atol=1e-8):
            raise AssertionError(f"Wrong fixed roundtrip cost arithmetic: {cost} bp")
        if not np.isclose(result[f"sum_net_{cost}bps"], gross.sum()-cost*n):
            raise AssertionError("Gross-to-net arithmetic mismatch")
    return result


def economics_audit(events: dict[str,pd.DataFrame],fingerprint: str) -> tuple[dict,pd.DataFrame]:
    import re
    if set(events)!=set(SYMBOLS) or re.fullmatch(r"[0-9a-f]{64}",fingerprint) is None:
        raise ValueError("Four approved symbols and SHA256 fingerprint required")
    by_asset={}
    rows=[]
    overall={"A_all":[],"B_confirmed":[]}
    for symbol in SYMBOLS:
        frame=events[symbol]
        required={"symbol","signal_close_at","mature_at","baseline_b_confirmed",
                  "gross_bps","net_2bps","net_5bps","net_10bps","exit_reason"}
        if not required.issubset(frame.columns) or not frame.symbol.eq(symbol).all():
            raise ValueError("Bad outcome frame or symbol")
        if (not frame.signal_close_at.is_unique
                or not frame.signal_close_at.is_monotonic_increasing
                or not frame.signal_close_at.lt(pd.Timestamp("2026-07-01T00:00:00Z")).all()
                or not frame.mature_at.lt(pd.Timestamp("2026-07-01T00:00:00Z")).all()
                or not pd.api.types.is_bool_dtype(frame.baseline_b_confirmed)):
            raise ValueError("Out-of-scope or malformed outcomes")
        periods={}
        for month,part in frame.groupby(frame.signal_close_at.dt.strftime("%Y-%m"),sort=True):
            report={}
            for name, cohort in (("A_all",part),
                                 ("B_confirmed",part.loc[part.baseline_b_confirmed])):
                stats=_stats(cohort,len(part))
                report[name]=stats
                rows.append({
                    "symbol":symbol,"month":month,"strategy":name,
                    "opportunities":len(part),
                    "trades":stats["trades"],
                    "gross_bps_mean":stats["gross_bps_mean"],
                    "gross_bps_median":stats["gross_bps_median"],
                    **{f"net_{cost}bps_per_trade":stats[f"net_{cost}bps_per_trade"]
                       for cost in COSTS},
                    **{f"net_{cost}bps_per_opportunity":stats[f"net_{cost}bps_per_opportunity"]
                       for cost in COSTS},
                    **{f"exit_{reason}":stats["exit_reasons_fixed"][reason] for reason in REASONS},
                })
            periods[month]=report
        total={}
        for name, cohort in (("A_all",frame),
                             ("B_confirmed",frame.loc[frame.baseline_b_confirmed])):
            total[name]=_stats(cohort,len(frame))
            if sum(periods[m][name]["trades"] for m in periods)!=total[name]["trades"]:
                raise AssertionError("Monthly trade counts mismatch")
            if not np.isclose(sum(periods[m][name]["sum_gross_bps"] for m in periods),
                              total[name]["sum_gross_bps"]):
                raise AssertionError("Monthly gross outcomes mismatch")
            overall[name].append(cohort)
        by_asset[symbol]={"all":total,"by_month":periods}
    combined={}
    all_opps=sum(len(events[s]) for s in SYMBOLS)
    for name,chunks in overall.items():
        combined[name]=_stats(pd.concat(chunks,ignore_index=True),all_opps)
        if combined[name]["trades"]!=sum(by_asset[s]["all"][name]["trades"] for s in SYMBOLS):
            raise AssertionError("Asset trade counts mismatch")
    return {
        "protocol":PROTOCOL,
        "status":"EXPLORATORY_READ_ONLY_ECONOMICS_NOT_INDEPENDENT_OOS",
        "dataset_fingerprint_sha256":fingerprint,
        "symbols":list(SYMBOLS),
        "costs_assumed_roundtrip_bps":list(COSTS),
        "exit_reason_categories":list(REASONS),
        "quantiles_fixed":list(QUANTILES),
        "warning":"Hindsight markouts, fictional exact OPEN(t+2) fills; no optimization or independent OOS.",
        "overall":combined,"by_asset":by_asset,
    },pd.DataFrame(rows)


def main() -> None:
    p=argparse.ArgumentParser(description="P4 fixed read-only economics audit")
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--output-json",type=Path,required=True)
    p.add_argument("--output-csv",type=Path,required=True)
    a=p.parse_args()
    root=a.data_root.resolve()
    outjson,outcsv=a.output_json.resolve(),a.output_csv.resolve()
    if (outjson==outcsv or outjson.parent!=outcsv.parent or
            not outjson.parent.is_dir() or
            any(x.exists() or x.is_relative_to(root) for x in (outjson,outcsv))):
        p.error("Two NEW files in one existing results folder outside P0")
    from ingestion.storage import CandleStore
    candles,manifests=load_long_development_history(CandleStore(root))
    events={}
    for symbol in SYMBOLS:
        events[symbol],census=prepare_symbol(candles[symbol])
        print(f"{symbol}: A={census['matured_A']} B={census['confirmed_B']}",flush=True)
    report,table=economics_audit(events,manifest_fingerprint(manifests))
    with outcsv.open("x",encoding="utf-8",newline="") as f:
        table.to_csv(f,index=False)
    with outjson.open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    for name,stats in report["overall"].items():
        print(f"{name}: trades={stats['trades']} gross/trade={stats['gross_bps_mean']} "
              f"net5/trade={stats['net_5bps_per_trade']} exits={stats['exit_reasons_fixed']}")
    print("NO ML / NO OPTIMIZATION / HISTORICAL EXPLORATORY ONLY")
    print("Saved",outjson,outcsv)


if __name__=="__main__":
    main()
