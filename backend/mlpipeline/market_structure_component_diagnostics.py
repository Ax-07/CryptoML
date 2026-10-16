"""P2 historical market-structure components: frequency and ablation.

Four groups: confirmed structure state, BOS, CHoCH, swing distance/age.
No ADA, fold OOS checks or final 90-day label scoring; P3 unaffected.
Existing inspected development checks are NOT independent evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ingestion.storage import CandleStore
from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.expanded_horizon_diagnostics import (
    SEEN_SYMBOLS, common_folds, shared_partitions,
)
from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.market_structure_diagnostics import (
    HORIZONS, _fit_and_predict, shared_with_structure,
)
from mlpipeline.market_structure_features import STRUCTURE_COLUMNS, STRUCTURE_SCHEMA

STATE = ("structure_hh", "structure_hl", "structure_lh", "structure_ll")
BOS = ("structure_bos_up", "structure_bos_down")
CHOCH = ("structure_choch_up", "structure_choch_down")
POSITION = (
    "structure_distance_swing_high", "structure_distance_swing_low",
    "structure_age_swing_high", "structure_age_swing_low",
)
COMPONENTS = {"state": STATE, "bos": BOS, "choch": CHOCH,
              "distance_age": POSITION}
EVENTS = (*BOS, *CHOCH)
FLAGS = (*STATE, *EVENTS)
VARIANTS = {
    "baseline_30": tuple(FEATURE_COLUMNS),
    "plus_state_34": (*FEATURE_COLUMNS, *STATE),
    "plus_bos_32": (*FEATURE_COLUMNS, *BOS),
    "plus_choch_32": (*FEATURE_COLUMNS, *CHOCH),
    "plus_distance_age_34": (*FEATURE_COLUMNS, *POSITION),
    "plus_all_structure_42": (*FEATURE_COLUMNS, *STRUCTURE_COLUMNS),
}
PROTOCOL = "cpu_p2_market_structure_components_h2_h6_v1"


def validate_groups() -> None:
    cols = [col for group in COMPONENTS.values() for col in group]
    if (len(cols) != 12 or len(set(cols)) != 12
            or set(cols) != set(STRUCTURE_COLUMNS)
            or len(FEATURE_COLUMNS) != 30
            or set(FEATURE_COLUMNS) & set(STRUCTURE_COLUMNS)
            or len(VARIANTS) != 6
            or any(len(v) != len(set(v)) for v in VARIANTS.values())):
        raise ValueError("Feature component partition or variants changed")


def event_frequency(frame: pd.DataFrame) -> dict:
    """Per-row rates. HH/HL/LH/LL are STABLE states, BOS/CHoCH are pulses."""
    if not set(FLAGS).issubset(frame):
        raise ValueError("Structure flags missing")
    rates = {}
    for feature in FLAGS:
        arr = frame[feature].to_numpy(dtype=float)
        if not np.isfinite(arr).all() or not np.isin(arr, [0, 1]).all():
            raise ValueError(f"Invalid structure binary feature: {feature}")
        count = int(np.count_nonzero(arr))
        rates[feature] = {
            "active_bars": count,
            "fraction": count/len(frame) if len(frame) else None,
        }
    return {
        "rows": int(len(frame)),
        "flags": rates,
        "bos_any": int(frame[list(BOS)].any(axis=1).sum()),
        "choch_any": int(frame[list(CHOCH)].any(axis=1).sum()),
        "break_any": int(frame[list(EVENTS)].any(axis=1).sum()),
    }


def frequency_report(checks: list[tuple[int, pd.DataFrame]]) -> dict:
    if not checks:
        raise ValueError("No validation-check frames")
    slices = []
    per_fold = []
    for fold_id, check in checks:
        if check.empty:
            raise ValueError("Empty validation check")
        per_fold.append({"fold_id": fold_id, **event_frequency(check)})
        slices.append(check[["symbol", "bar_close_at", *FLAGS]])
    merged = pd.concat(slices, ignore_index=True)
    if merged.duplicated(["symbol", "bar_close_at"]).any():
        raise ValueError("Overlapping validation-check keys: cannot count twice")
    primary = merged.bar_close_at.dt.hour.between(12, 17)
    return {
        "all_check_rows": event_frequency(merged),
        "primary_12_17_UTC": event_frequency(merged.loc[primary]),
        "by_symbol": {
            symbol: event_frequency(merged.loc[merged.symbol.eq(symbol)])
            for symbol in SEEN_SYMBOLS
        },
        "per_fold": per_fold,
        "semantics": "HH/HL/LH/LL are persistent states; BOS/CHoCH are one-bar events",
    }


def _mean(vals):
    vals = list(vals)
    if not vals or any(v is None for v in vals):
        return None
    return float(np.mean(vals))


def ablation_summary(records: list[dict]) -> dict:
    if not records:
        raise ValueError("No ablation results")
    out = {}
    for h in HORIZONS:
        key=f"H{h}"
        variants={}
        for name in VARIANTS:
            vals=[r["horizons"][key][name] for r in records]
            good=[v for v in vals if v["feasible"]]
            variants[name]={
                "feasible_folds":len(good), "total_folds":len(vals),
                "mean_uplift_points":_mean(
                    v["check_metrics"]["uplift_points"] for v in good),
                "mean_realized_coverage":_mean(
                    v["check_metrics"]["coverage"] for v in good),
                "mean_log_loss":_mean(
                    v["check_metrics"]["raw_log_loss"] for v in good),
            }
        paired={}
        for name in tuple(VARIANTS)[1:]:
            pairs=[(r["horizons"][key]["baseline_30"],r["horizons"][key][name])
                   for r in records]
            good=[(a,b) for a,b in pairs if a["feasible"] and b["feasible"]
                  and a["check_metrics"]["uplift_points"] is not None
                  and b["check_metrics"]["uplift_points"] is not None]
            delta=[b["check_metrics"]["uplift_points"]-
                   a["check_metrics"]["uplift_points"] for a,b in good]
            paired[name]={
                "paired_feasible_folds":len(good),
                "positive_uplift_folds":sum(d>0 for d in delta),
                "mean_uplift_delta_points":_mean(delta),
                "mean_realized_coverage_delta_points":_mean(
                    100*(b["check_metrics"]["coverage"]-
                         a["check_metrics"]["coverage"]) for a,b in good),
                "mean_log_loss_improvement":_mean(
                    a["check_metrics"]["raw_log_loss"]-
                    b["check_metrics"]["raw_log_loss"] for a,b in good),
            }
        out[key]={"variants":variants,"paired_vs_baseline":paired}
    return out


def run_study(
    shared:pd.DataFrame, *, mode:str, max_folds:int|None=None,
    n_estimators:int=150, seed:int=42,
    train_days:int=120,validation_days:int=14,test_days:int=14,
    step_days:int=14,final_holdout_days:int=90,
) -> dict:
    validate_groups()
    if mode not in ("frequency","ablation"):
        raise ValueError("Unknown study mode")
    if set(shared.symbol)!=set(SEEN_SYMBOLS):
        raise ValueError("ADA excluded: exactly four seen symbols required")
    cols=set((*FEATURE_COLUMNS,*STRUCTURE_COLUMNS))
    if not cols.issubset(shared.columns):
        raise ValueError("Missing feature columns")
    if not np.isfinite(shared[list(cols)].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite columns: use shared_with_structure")
    if n_estimators<20 or (max_folds is not None and max_folds<1):
        raise ValueError("Invalid estimators or max-folds")
    folds=common_folds(
        shared,train_days=train_days,validation_days=validation_days,
        test_days=test_days,step_days=step_days,
        final_holdout_days=final_holdout_days,
    )
    total=len(folds)
    if max_folds is not None:
        folds=folds[:max_folds]
    if not folds:
        raise ValueError("No available folds")
    result_rows=[]
    checks=[]
    for fold in folds:
        parts,bounds=shared_partitions(shared,fold)
        check=parts["validation_check"]
        checks.append((int(fold.fold_id),check))
        if mode=="frequency":
            freq=event_frequency(check)
            print(f"fold={fold.fold_id} M5={freq['rows']} "
                  f"BOS={freq['bos_any']} CHoCH={freq['choch_any']}",flush=True)
            continue
        stop_end=pd.Timestamp(bounds["early_stop_end_exclusive_utc"])
        tune_end=pd.Timestamp(bounds["unused_tuning_end_exclusive_utc"])
        t,mature=shared.bar_close_at,shared.label_mature_at
        tune=shared.loc[t.ge(stop_end)&t.lt(tune_end)&mature.lt(tune_end)]
        if len(tune)<100:
            raise ValueError("Insufficient prior H12-purged tuning rows")
        horizons={}
        for h in HORIZONS:
            variants={}
            for name,features in VARIANTS.items():
                res=_fit_and_predict(
                    parts["train"],parts["early_stopping"],tune,check,
                    cols=list(features),horizon=h,
                    iterations=n_estimators,seed=seed)
                variants[name]=res
                metrics=res["check_metrics"]
                print(f"fold={fold.fold_id} H{h} {name} "
                      f"feasible={res['feasible']} "
                      f"uplift={metrics['uplift_points'] if metrics else None} "
                      f"coverage={metrics['coverage'] if metrics else None}",
                      flush=True)
            horizons[f"H{h}"]=variants
        result_rows.append({
            "fold_id":int(fold.fold_id),**bounds,
            "partition_counts":{k:len(v) for k,v in parts.items()},
            "tuning_rows":len(tune),"horizons":horizons,
        })
    partial=len(folds)!=total
    return {
        "protocol":PROTOCOL,"mode":mode,
        "research_status":"PARTIAL_SMOKE_NOT_FINAL" if partial
                          else "EXPLORATORY_DEVELOPMENT_ONLY",
        "available_folds":total,"analyzed_folds":len(folds),
        "expected_fits":len(folds)*len(HORIZONS)*len(VARIANTS)
                        if mode=="ablation" else 0,
        "feature_schema":STRUCTURE_SCHEMA,
        "component_columns":{k:list(v) for k,v in COMPONENTS.items()},
        "variant_columns":{k:list(v) for k,v in VARIANTS.items()},
        "target_tuning_coverage":0.40,
        "note_coverage":"Only tuning coverage targeted; realized check coverage may differ",
        "H12_label_maturity_purge":True,"ADA_loaded_or_scored":False,
        "fold_OOS_scored":False,"final_90_days_scored":False,
        "P3_frozen_models_unchanged":True,
        "frequency":frequency_report(checks),
        "folds":result_rows,
        "summary":ablation_summary(result_rows) if mode=="ablation" else {},
        "limitations":[
            "Previously inspected development checks, not independent evidence.",
            "Overlapping train folds, multiple hypotheses, descriptive only.",
            "Group addition is not causal importance; no trading or profitability claim.",
        ],
    }


def main():
    p=argparse.ArgumentParser(description="P2 H2/H6 market structure component study")
    p.add_argument("--mode",required=True,choices=("frequency","ablation"))
    p.add_argument("--data-root",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--months",default=24,type=int)
    p.add_argument("--n-estimators",default=150,type=int)
    p.add_argument("--seed",default=42,type=int)
    p.add_argument("--max-folds",default=None,type=int,
                   help="Smoke test only: partial first N folds")
    p.add_argument("--verify-hashes",action="store_true")
    a=p.parse_args()
    if a.months!=24:
        p.error("24-month partition selection required")
    if a.output.exists():
        p.error("Refusing to overwrite an existing research report")
    store=CandleStore(a.data_root)
    candles,manifests={},[]
    for symbol in SEEN_SYMBOLS:
        print("Loading seen asset",symbol,flush=True)
        candles[symbol],entries=load_symbol(
            store,symbol,months=a.months,verify_hashes=a.verify_hashes)
        manifests.extend(entries)
    shared=shared_with_structure(candles)
    del candles
    result=run_study(shared,mode=a.mode,max_folds=a.max_folds,
                     n_estimators=a.n_estimators,seed=a.seed)
    result["dataset_fingerprint_sha256"]=manifest_fingerprint(manifests)
    result["lightgbm_version"]=lgb.__version__
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print("Saved historical component study",a.output.resolve(),flush=True)


if __name__=="__main__":
    main()
