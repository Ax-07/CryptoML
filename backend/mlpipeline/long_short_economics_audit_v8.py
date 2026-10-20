"""V8 frozen exploratory long/short mechanical economics. No ML or orders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ingestion.storage import CandleStore
from mlpipeline.dataset import manifest_fingerprint
from mlpipeline.market_structure_v6 import classify_m5
from mlpipeline.market_structure_visual_audit_v7 import verify_reference
from mlpipeline.pullback_global_ridge_v3 import prepare_symbol
from mlpipeline.pullback_long_history_v3 import SYMBOLS, CUTOFF, load_long_development_history
from mlpipeline.trend_pullback import M5, _segment_features, _validate_closed_m5

PROTOCOL = "p4_v8_frozen_symmetric_barrier_mtf_economics_v1"
COSTS = (2, 5, 10)
REASONS = ("gap_stop", "gap_target_capped", "both_hit_stop_first", "stop", "target", "timeout")
COHORTS = ("A", "B")
SELECTIONS = ("ALL", "ALIGNED", "REJECTED")


def _features(bars: pd.DataFrame) -> pd.DataFrame:
    """Same EMA9 as V1, with gap resets."""
    data = bars.copy()
    data["_segment"] = data.bar_open_at.diff().ne(M5).cumsum()
    return pd.concat([_segment_features(p.reset_index(drop=True))
                      for _, p in data.groupby("_segment", sort=False)], ignore_index=True)


def _barrier(window: pd.DataFrame, entry: float, atr: float, side: str):
    """Identical LONG semantics to pullback_outcomes, mirrored SHORT."""
    sign = 1 if side == "LONG" else -1
    stop = entry - sign * atr
    target = entry + sign * 1.5 * atr
    exit_price = float(window.close.iloc[-1])
    reason = "timeout"
    for bar in window.itertuples(index=False):
        op = float(bar.open)
        adverse_gap = op <= stop if sign == 1 else op >= stop
        favorable_gap = op >= target if sign == 1 else op <= target
        stop_hit = bar.low <= stop if sign == 1 else bar.high >= stop
        target_hit = bar.high >= target if sign == 1 else bar.low <= target
        if adverse_gap:
            exit_price, reason = op, "gap_stop"
        elif favorable_gap:
            exit_price, reason = target, "gap_target_capped"
        elif stop_hit and target_hit:
            exit_price, reason = stop, "both_hit_stop_first"
        elif stop_hit:
            exit_price, reason = stop, "stop"
        elif target_hit:
            exit_price, reason = target, "target"
        else:
            continue
        break
    gross = 10000 * sign * (exit_price / entry - 1)
    return exit_price, reason, gross


def symbol_events(candles: pd.DataFrame, classification: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    bars = _validate_closed_m5(candles)
    if (len(classification) != len(bars)
            or not classification.bar_close_at.equals(bars.bar_close_at)):
        raise ValueError("V6 signal alignment mismatch")
    f = _features(bars)
    items = []
    candidates = 0
    excluded = 0
    for i in np.flatnonzero((classification.long_signal |
                             classification.short_signal).to_numpy()):
        side = "LONG" if bool(classification.long_signal.iloc[i]) else "SHORT"
        candidates += 1
        end = i + 7
        if end >= len(bars):
            excluded += 1
            continue
        signal_at = bars.bar_close_at.iloc[i]
        mature = signal_at + pd.Timedelta(minutes=35)
        window = bars.iloc[i + 2:end + 1]
        if (mature >= CUTOFF or len(window) != 6
                or bars.bar_close_at.iloc[i + 1] != signal_at + M5
                or window.bar_open_at.iloc[0] != signal_at + M5
                or window.bar_close_at.iloc[-1] != mature
                or not window.bar_open_at.diff().iloc[1:].eq(M5)):
            excluded += 1
            continue
        atr = float(f.atr14.iloc[i])
        entry = float(window.open.iloc[0])
        if not np.isfinite(atr) or atr <= 0 or entry <= 1.5 * atr:
            excluded += 1
            continue
        close1, open1, ema1 = (float(bars.close.iloc[i + 1]),
                               float(bars.open.iloc[i + 1]),
                               float(f.ema9.iloc[i + 1]))
        close0 = float(bars.close.iloc[i])
        confirmed = bool(np.isfinite(ema1) and (
            (close1 > open1 and close1 > ema1 and close1 > close0)
            if side == "LONG" else
            (close1 < open1 and close1 < ema1 and close1 < close0)))
        exit_price, reason, gross = _barrier(window, entry, atr, side)
        aligned = bool(classification.long_with_trend.iloc[i] if side == "LONG"
                       else classification.short_with_trend.iloc[i])
        items.append({
            "symbol": str(bars.symbol.iloc[0]), "side": side,
            "signal_close_at": signal_at, "entry_at":window.bar_open_at.iloc[0],
            "mature_at":mature, "entry_price":entry,
            "exit_price":exit_price, "exit_reason":reason, "atr_at_signal":atr,
            "gross_bps":gross, "confirmed_B":confirmed, "aligned":aligned,
            "regime":str(classification.market_regime.iloc[i]),
            **{f"net_{cost}bps":gross - cost for cost in COSTS},
        })
    return pd.DataFrame.from_records(items), {
        "candidate_signals":candidates, "matured":len(items),
        "excluded_immature_or_gap_or_invalid":excluded,
    }


def verify_legacy_long(bars: pd.DataFrame, events: pd.DataFrame):
    """Exact V1 compatibility, event by event; never permit changed LONG outcomes."""
    prior, _ = prepare_symbol(bars)
    new = events.loc[events.side.eq("LONG")].sort_values("signal_close_at")
    old = prior.sort_values("signal_close_at")
    if len(new) != len(old) or not new.signal_close_at.reset_index(drop=True).equals(
            old.signal_close_at.reset_index(drop=True)):
        raise AssertionError("V8 LONG cohort disagrees with V1 mature universe")
    for column in ("entry_at","mature_at","entry_price","exit_price","atr_at_signal",
                   "exit_reason","gross_bps","net_2bps","net_5bps","net_10bps"):
        a = new[column].reset_index(drop=True)
        b = old[column].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(a.dtype):
            if not np.allclose(a.to_numpy(dtype=float),b.to_numpy(dtype=float),
                               atol=1e-8,rtol=1e-10):
                raise AssertionError(f"LONG parity violated: {column}")
        elif not a.equals(b):
            raise AssertionError(f"LONG parity violated: {column}")
    if not new.confirmed_B.reset_index(drop=True).equals(
            old.baseline_b_confirmed.reset_index(drop=True)):
        raise AssertionError("LONG B confirmation parity violated")


def _stats(items: pd.DataFrame, opportunities: int) -> dict:
    n = len(items)
    if n > opportunities:
        raise ValueError("Trade count exceeds opportunity pool")
    gross = items.gross_bps.to_numpy(dtype=float)
    if not np.isfinite(gross).all():
        raise ValueError("Non-finite gross marks")
    out = {
        "opportunities":int(opportunities), "trades":int(n),
        "coverage": n/opportunities if opportunities else None,
        "gross_mean_bps":float(gross.mean()) if n else None,
        "gross_median_bps":float(np.median(gross)) if n else None,
        "sum_gross_bps":float(gross.sum()),
        "exit_reasons":{r:int(items.exit_reason.eq(r).sum()) for r in REASONS},
    }
    if sum(out["exit_reasons"].values()) != n:
        raise AssertionError("Unknown exit reasons")
    for cost in COSTS:
        arr = items[f"net_{cost}bps"].to_numpy(dtype=float)
        if not np.allclose(arr, gross-cost, atol=1e-8, rtol=0):
            raise AssertionError("Incorrect round-trip fee")
        out.update({
            f"sum_net_{cost}bps":float(arr.sum()),
            f"net_{cost}bps_per_trade":float(arr.mean()) if n else None,
            f"net_{cost}bps_per_opportunity":float(arr.sum()/opportunities) if opportunities else None,
            f"positive_net_{cost}bps_trades":int((arr>0).sum()),
        })
    return out


def _group_report(group: pd.DataFrame) -> dict:
    if group.empty:
        raise ValueError("Must have A opportunities in each reported group")
    result = {}
    for cohort in COHORTS:
        pool = group if cohort == "A" else group.loc[group.confirmed_B]
        masks = {
            "ALL":np.ones(len(pool), dtype=bool),
            "ALIGNED":pool.aligned.to_numpy(dtype=bool),
            "REJECTED":~pool.aligned.to_numpy(dtype=bool),
        }
        scored = {name:_stats(pool.iloc[np.flatnonzero(mask)],len(group))
                  for name,mask in masks.items()}
        for column in ("trades", "sum_gross_bps", *(f"sum_net_{c}bps" for c in COSTS)):
            if not np.isclose(scored["ALL"][column],
                              scored["ALIGNED"][column]+scored["REJECTED"][column],
                              rtol=1e-10, atol=1e-7):
                raise AssertionError(f"ALL partition mismatch {cohort}/{column}")
        result[cohort]=scored
    if result["B"]["ALL"]["trades"] > result["A"]["ALL"]["trades"]:
        raise AssertionError("B must be a subset of A")
    return result


def audit(events: dict[str,pd.DataFrame],fingerprint:str, exclusions:dict) -> tuple[dict,pd.DataFrame]:
    if set(events) != set(SYMBOLS):
        raise ValueError("Four known assets only")
    rows=[]; by_asset={}; merged=[]
    for symbol in SYMBOLS:
        f=events[symbol]
        if (f.empty or not f.symbol.eq(symbol).all() or
                not f.mature_at.lt(CUTOFF).all() or
                not f.signal_close_at.is_unique or
                not f.signal_close_at.is_monotonic_increasing or
                not f.side.isin(("LONG","SHORT")).all()):
            # Opposite-direction candidates *may* share timestamps; use pair uniqueness.
            if (f.empty or not f.symbol.eq(symbol).all() or
                    not f.mature_at.lt(CUTOFF).all() or
                    f.duplicated(["signal_close_at","side"]).any() or
                    not f.signal_close_at.is_monotonic_increasing):
                raise ValueError("Invalid events")
        if f.duplicated(["signal_close_at","side"]).any():
            raise ValueError("Duplicate side+timestamp")
        f=f.copy()
        f["month"]=f.signal_close_at.dt.strftime("%Y-%m")
        merged.append(f)
        by_asset[symbol]={}
        for side in ("LONG","SHORT"):
            group=f.loc[f.side.eq(side)]
            if group.empty:
                raise ValueError("Missing direction")
            by_asset[symbol][side]=_group_report(group)
            for month,subset in group.groupby("month",sort=True):
                blocks=_group_report(subset)
                for cohort in COHORTS:
                    for name in SELECTIONS:
                        rows.append({"symbol":symbol,"side":side,"month":month,
                                     "cohort":cohort,"selection":name,
                                     **{k:v for k,v in blocks[cohort][name].items()
                                        if k!="exit_reasons"}})
    combined=pd.concat(merged,ignore_index=True)
    result={}
    for side in ("LONG","SHORT","BOTH"):
        chosen=combined if side=="BOTH" else combined.loc[combined.side.eq(side)]
        result[side]=_group_report(chosen)
    result["NO_TRADE"]={
        "trades":0,**{f"sum_net_{c}bps":0. for c in COSTS}
    }
    # Count opposite-direction windows that overlap on same asset, no portfolio inference.
    overlap=0
    for symbol in SYMBOLS:
        asset=events[symbol]
        longs=asset.loc[asset.side.eq("LONG")]
        shorts=asset.loc[asset.side.eq("SHORT")]
        for item in longs.itertuples(index=False):
            overlap+=int(((shorts.entry_at<item.mature_at) &
                          (shorts.mature_at>item.entry_at)).sum())
    return {
        "protocol":PROTOCOL,"status":"EXPLORATORY_HISTORICAL_NOT_INDEPENDENT_OOS",
        "dataset_fingerprint_sha256":fingerprint, "assets":list(SYMBOLS),
        "costs_fixed_roundtrip_bps":list(COSTS),
        "strictly_hypothetical":True,"no_orders_no_ml_no_funding":True,
        "no_trade_eligibility_claim":True,
        "gross_bps_definition":"sum is not portfolio return; events may overlap",
        "opposite_side_overlapping_window_pairs":overlap,
        "exclusions_by_asset":exclusions,
        "overall":result,"by_asset":by_asset,
    },pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description="Frozen P4 V8 symmetric economics (historical)")
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--reference-json",type=Path,required=True)
    p.add_argument("--output-json",type=Path,required=True)
    p.add_argument("--output-csv",type=Path,required=True)
    a=p.parse_args()
    root=a.data_root.resolve()
    ref=a.reference_json.resolve()
    output,tablepath=a.output_json.resolve(),a.output_csv.resolve()
    if (not ref.is_file() or ref.is_relative_to(root)
            or output==tablepath or output.parent!=tablepath.parent
            or not output.parent.is_dir() or
            any(x.exists() or x.is_relative_to(root) or x==ref
                for x in (output,tablepath))):
        p.error("Existing V6 JSON and two NEW output files outside P0 required")
    with ref.open("r",encoding="utf-8") as stream:
        reference=json.load(stream)
    candles,manifests=load_long_development_history(CandleStore(root))
    fingerprint=manifest_fingerprint(manifests)
    features={}; events={}; exclusions={}
    for symbol in SYMBOLS:
        print("Computing V8 fixed outcomes:",symbol,flush=True)
        features[symbol]=classify_m5(candles[symbol])
    verify_reference(reference,fingerprint,features)
    for symbol in SYMBOLS:
        event,excluded=symbol_events(candles[symbol],features[symbol])
        verify_legacy_long(candles[symbol],event)
        events[symbol]=event
        exclusions[symbol]=excluded
        print(symbol,excluded,"LONG",int(event.side.eq("LONG").sum()),
              "SHORT",int(event.side.eq("SHORT").sum()),flush=True)
    report,table=audit(events,fingerprint,exclusions)
    # Save CSV first; final JSON is the completion marker.
    with tablepath.open("x",encoding="utf-8",newline="") as stream:
        table.to_csv(stream,index=False)
    with output.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    for side in ("LONG","SHORT","BOTH"):
        for cohort in COHORTS:
            for selection in SELECTIONS:
                s=report["overall"][side][cohort][selection]
                print(side,cohort,selection,
                      "n=",s["trades"],"net5/trade=",s["net_5bps_per_trade"],
                      "net5/opportunity=",s["net_5bps_per_opportunity"])
    print("NO EXECUTION / NO FUNDING / NO ML / NOT OOS — exploratory only")
    print("Saved",output,tablepath)


if __name__=="__main__":
    main()
