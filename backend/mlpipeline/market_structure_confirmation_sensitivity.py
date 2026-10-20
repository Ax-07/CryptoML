"""Exploratory P2 pivot-confirmation sensitivity: 1 vs 2 vs 3 closed M5 bars.

The original frozen feature schema and default R2 structure are unchanged.
No ADA, fold OOS test, final reserved 90d, or prospective P3 involvement.
Previously inspected dev checks: results are NOT independent validation.
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
    SEEN_SYMBOLS, build_shared_examples, common_folds, shared_partitions,
)
from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.market_structure_diagnostics import (
    HORIZONS, _fit_and_predict,
)
from mlpipeline.market_structure_features import (
    STRUCTURE_COLUMNS, build_market_structure_features,
)

RIGHTS = (1, 2, 3)
VARIANTS = {
    "baseline_30": tuple(FEATURE_COLUMNS),
    **{
        f"structure_r{r}_42": (
            *FEATURE_COLUMNS, *(f"r{r}_{c}" for c in STRUCTURE_COLUMNS)
        ) for r in RIGHTS
    },
}
EVENTS = (
    "structure_bos_up", "structure_bos_down",
    "structure_choch_up", "structure_choch_down",
)


def build_confirmation_shared(candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Same symbol/timestamp rows for all four variants, after all r warmups."""
    if set(candles) != set(SEEN_SYMBOLS):
        raise ValueError("Exactly four seen assets required; ADA forbidden")
    common = build_shared_examples(candles)
    for r in RIGHTS:
        parts = []
        for symbol in SEEN_SYMBOLS:
            s = build_market_structure_features(
                candles[symbol], confirmation_bars=r)
            s = s.rename(columns={c:f"r{r}_{c}" for c in STRUCTURE_COLUMNS})
            parts.append(s)
        extra = pd.concat(parts, ignore_index=True)
        if extra.duplicated(["symbol", "bar_close_at"]).any():
            raise ValueError("Duplicate structure keys")
        previous = len(common)
        common = common.merge(
            extra, on=["symbol", "bar_close_at"], how="left",
            validate="one_to_one", sort=False,
        )
        if len(common) != previous:
            raise AssertionError("Structure join changed example cardinality")
    all_cols = list({c for cols in VARIANTS.values() for c in cols})
    eligible = np.isfinite(common[all_cols].to_numpy(dtype=float)).all(axis=1)
    common = common.loc[eligible].copy()
    if common.empty:
        raise ValueError("No joint R1/R2/R3 eligible M5 examples")
    for r in RIGHTS:
        cols = [f"r{r}_{c}" for c in STRUCTURE_COLUMNS]
        common[cols] = common[cols].astype("float32")
    return common


def signal_rates(frame: pd.DataFrame) -> dict:
    """Event prevalence in validation-check rows, not directional performance."""
    if frame.empty:
        raise ValueError("Empty event frequency slice")
    result = {"rows":len(frame)}
    for r in RIGHTS:
        bos_cols = [f"r{r}_{c}" for c in EVENTS[:2]]
        choch_cols = [f"r{r}_{c}" for c in EVENTS[2:]]
        b = frame[bos_cols].any(axis=1)
        c = frame[choch_cols].any(axis=1)
        result[f"r{r}"] = {
            "bos_count":int(b.sum()),"choch_count":int(c.sum()),
            "any_break_count":int((b|c).sum()),
            "bos_rate":float(b.mean()),"choch_rate":float(c.mean()),
            "any_break_rate":float((b|c).mean()),
        }
    return result


def _mean(values):
    vals=list(values)
    return float(np.mean(vals)) if vals and all(v is not None for v in vals) else None


def summarize(rows:list[dict]) -> dict:
    summary={}
    for h in HORIZONS:
        key=f"H{h}"
        variant_results={}
        for name in VARIANTS:
            samples=[row["horizons"][key][name] for row in rows]
            feasible=[x for x in samples if x["feasible"]]
            variant_results[name]={
                "feasible_folds":len(feasible),
                "total_folds":len(samples),
                "mean_check_uplift_points":_mean(
                    x["check_metrics"]["uplift_points"] for x in feasible),
                "mean_check_coverage":_mean(
                    x["check_metrics"]["coverage"] for x in feasible),
                "mean_check_log_loss":_mean(
                    x["check_metrics"]["raw_log_loss"] for x in feasible),
            }
        paired={}
        for name in tuple(VARIANTS)[1:]:
            base_pairs=[
                (row["horizons"][key]["baseline_30"],
                 row["horizons"][key][name])
                for row in rows
            ]
            comparable=[
                (a,b) for a,b in base_pairs
                if a["feasible"] and b["feasible"]
                and a["check_metrics"]["uplift_points"] is not None
                and b["check_metrics"]["uplift_points"] is not None
            ]
            delta=[
                b["check_metrics"]["uplift_points"]-
                a["check_metrics"]["uplift_points"] for a,b in comparable
            ]
            paired[name]={
                "paired_feasible_folds":len(comparable),
                "positive_uplift_folds":sum(v>0 for v in delta),
                "mean_uplift_delta_points":_mean(delta),
                "mean_realized_coverage_delta_points":_mean(
                    100*(b["check_metrics"]["coverage"]-
                         a["check_metrics"]["coverage"])
                    for a,b in comparable),
                "mean_log_loss_improvement":_mean(
                    a["check_metrics"]["raw_log_loss"]-
                    b["check_metrics"]["raw_log_loss"]
                    for a,b in comparable),
            }
        summary[key]={"variants":variant_results,"paired_vs_baseline":paired}
    return summary


def run_sensitivity(
    shared:pd.DataFrame, *, max_folds:int|None=None,
    n_estimators:int=150,seed:int=42,
    train_days:int=120,validation_days:int=14,
    test_days:int=14,step_days:int=14,final_holdout_days:int=90,
) -> dict:
    if set(shared.symbol)!=set(SEEN_SYMBOLS):
        raise ValueError("ADA forbidden, require four seen assets")
    cols=list({c for v in VARIANTS.values() for c in v})
    if not set(cols).issubset(shared.columns) or (
        not np.isfinite(shared[cols].to_numpy(dtype=float)).all()
    ):
        raise ValueError("Missing/nonfinite common feature rows")
    if n_estimators<20 or (max_folds is not None and max_folds<1):
        raise ValueError("Invalid CPU training parameters")
    folds=common_folds(
        shared,train_days=train_days,validation_days=validation_days,
        test_days=test_days,step_days=step_days,
        final_holdout_days=final_holdout_days,
    )
    total_folds=len(folds)
    if max_folds is not None:
        folds=folds[:max_folds]
    if not folds:
        raise ValueError("No folds")
    records=[]
    event_frames=[]
    for fold in folds:
        chunks,bounds=shared_partitions(shared,fold)
        check=chunks["validation_check"]
        event_frames.append(check.loc[:,["symbol","bar_close_at",*[
            f"r{r}_{c}" for r in RIGHTS for c in EVENTS
        ]] ])
        early_end=pd.Timestamp(bounds["early_stop_end_exclusive_utc"])
        tune_end=pd.Timestamp(bounds["unused_tuning_end_exclusive_utc"])
        t,m=shared.bar_close_at,shared.label_mature_at
        tune=shared.loc[t.ge(early_end)&t.lt(tune_end)&m.lt(tune_end)]
        if len(tune)<100:
            raise ValueError("Insufficient H12-purged earlier tuning data")
        horizon_runs={}
        for h in HORIZONS:
            horizon_runs[f"H{h}"]={}
            for variant,feat in VARIANTS.items():
                result=_fit_and_predict(
                    chunks["train"],chunks["early_stopping"],
                    tune,check,cols=list(feat),horizon=h,
                    iterations=n_estimators,seed=seed)
                horizon_runs[f"H{h}"][variant]=result
                metrics=result["check_metrics"]
                print(f"fold={fold.fold_id} H{h} {variant} "
                      f"feasible={result['feasible']} "
                      f"uplift={metrics['uplift_points'] if metrics else None} "
                      f"coverage={metrics['coverage'] if metrics else None}",
                      flush=True)
        records.append({
            "fold_id":int(fold.fold_id),**bounds,
            "partition_rows":{k:len(v) for k,v in chunks.items()},
            "tuning_rows":len(tune),"horizons":horizon_runs,
        })
    counted=pd.concat(event_frames,ignore_index=True)
    if counted.duplicated(["symbol","bar_close_at"]).any():
        raise ValueError("Overlapping check periods: frequency would double count")
    return {
        "protocol":"P2_R1_R2_R3_confirmation_sensitivity_v1",
        "research_status":"SMOKE_PARTIAL_NOT_INDEPENDENT"
                          if len(folds)<total_folds
                          else "EXPLORATORY_PREVIOUSLY_INSPECTED_DEVELOPMENT",
        "left_pivot_bars":2,"right_confirmation_bars":list(RIGHTS),
        "original_R2_default_unchanged":True,
        "horizons":list(HORIZONS),
        "variant_columns":{k:list(v) for k,v in VARIANTS.items()},
        "total_available_folds":total_folds,"folds_analyzed":len(folds),
        "expected_training_runs":len(folds)*len(HORIZONS)*len(VARIANTS),
        "target_tuning_coverage":0.4,
        "coverage_note":"Tuning target only; realized check coverage not forced",
        "H12_purge":True,"ADA_loaded_or_scored":False,
        "fold_OOS_scored":False,"reserved_90_days_scored":False,
        "P3_models_modified":False,
        "check_event_frequency": {
            "all_unique_rows":signal_rates(counted),
            "12_17_UTC":signal_rates(
                counted.loc[counted.bar_close_at.dt.hour.between(12,17)]),
            "by_symbol":{
                symbol:signal_rates(counted.loc[counted.symbol.eq(symbol)])
                for symbol in SEEN_SYMBOLS
            },
        },
        "folds":records,"summary":summarize(records),
        "limitations":[
            "The 36 historical check slices have been studied previously.",
            "Three comparisons on reused data increase selection bias.",
            "One confirmed pivot first becomes available at p+r after r closed bars.",
            "A small uplift change is not independent evidence or trading profitability.",
        ],
    }


def main():
    p=argparse.ArgumentParser(description="P2 causal R1 R2 R3 structural sensitivity")
    p.add_argument("--data-root",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--months",default=24,type=int)
    p.add_argument("--n-estimators",default=150,type=int)
    p.add_argument("--seed",default=42,type=int)
    p.add_argument("--max-folds",default=None,type=int,
                   help="Smoke only; marked partial, not evidence")
    p.add_argument("--verify-hashes",action="store_true")
    a=p.parse_args()
    if a.months!=24:
        p.error("P2 requires the same 24 monthly partitions")
    if a.output.exists():
        p.error("Refusing to overwrite a previous research report")
    store=CandleStore(a.data_root)
    candles,manifests={},[]
    for symbol in SEEN_SYMBOLS:
        print("Loading",symbol,flush=True)
        candles[symbol],m=load_symbol(
            store,symbol,months=a.months,verify_hashes=a.verify_hashes)
        manifests.extend(m)
    shared=build_confirmation_shared(candles)
    del candles
    report=run_sensitivity(shared,max_folds=a.max_folds,
                           n_estimators=a.n_estimators,seed=a.seed)
    report["dataset_fingerprint_sha256"]=manifest_fingerprint(manifests)
    report["lightgbm_version"]=lgb.__version__
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print("Saved R1 R2 R3 historical report:",a.output.resolve(),flush=True)


if __name__=="__main__":
    main()
