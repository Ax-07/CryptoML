"""Exploratory P2 30-feature ablation at 40% tuning-only directional coverage.

group: baseline versus seven leave-family-out models.
individual: baseline versus optional subset/all 30 leave-one-feature-out models.

Use common H12-purged folds and SAME rows for every candidate. No ADA P0,
fold OOS or last 90 days scored. Dev checks are already inspected.
Does not alter frozen training/P3 models; no automatic feature selection.
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
from mlpipeline.feature_group_ablation import GROUPS, validate_feature_groups
from mlpipeline.features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from mlpipeline.market_structure_diagnostics import _fit_and_predict

BASELINE = "baseline_30"
HORIZONS = (2, 6)
PROTOCOL = "cpu_p2_feature_reduction_40pct_h2_h6_v1"


def variants_for(mode: str, *, only_features: tuple[str, ...] = ()) -> dict:
    validate_feature_groups()
    columns = tuple(FEATURE_COLUMNS)
    if mode == "group":
        if only_features:
            raise ValueError("--only-features only for individual mode")
        drops = {f"drop_group_{name}": tuple(group) for name, group in GROUPS.items()}
    elif mode == "individual":
        if len(set(only_features)) != len(only_features):
            raise ValueError("Duplicate requested features")
        unknown = set(only_features) - set(columns)
        if unknown:
            raise ValueError(f"Unknown features: {sorted(unknown)}")
        selected = only_features or columns
        drops = {f"drop_{col}": (col,) for col in columns if col in selected}
    else:
        raise ValueError("Only group and individual modes")
    variants = {BASELINE: columns}
    for key, excluded in drops.items():
        reduced = tuple(col for col in columns if col not in excluded)
        if not reduced or len(reduced) + len(excluded) != len(columns):
            raise AssertionError("Invalid removal set")
        variants[key] = reduced
    return variants


def _mean(values):
    vals = list(values)
    return float(np.mean(vals)) if vals and all(x is not None for x in vals) else None


def summarize(records: list[dict], variants: dict) -> dict:
    if not records or BASELINE not in variants:
        raise ValueError("Cannot summarize empty or baseline-free runs")
    summary = {}
    for h in HORIZONS:
        k = f"H{h}"
        by_model, paired = {}, {}
        for name, features in variants.items():
            results = [row["horizons"][k][name] for row in records]
            good = [result for result in results if result["feasible"]]
            by_model[name] = {
                "feature_count":len(features),
                "feasible_folds":len(good),"total_folds":len(results),
                "mean_uplift_points":_mean(
                    a["check_metrics"]["uplift_points"] for a in good),
                "mean_realized_coverage":_mean(
                    a["check_metrics"]["coverage"] for a in good),
                "mean_log_loss":_mean(
                    a["check_metrics"]["raw_log_loss"] for a in good),
            }
        for name in tuple(variants)[1:]:
            candidates = [
                (r["horizons"][k][BASELINE],r["horizons"][k][name])
                for r in records
            ]
            good = [
                (baseline, dropped) for baseline, dropped in candidates
                if baseline["feasible"] and dropped["feasible"]
                and baseline["check_metrics"]["uplift_points"] is not None
                and dropped["check_metrics"]["uplift_points"] is not None
            ]
            deltas = [
                dropped["check_metrics"]["uplift_points"]
                - baseline["check_metrics"]["uplift_points"]
                for baseline,dropped in good
            ]
            per_symbol = {}
            for symbol in SEEN_SYMBOLS:
                asset_deltas = [
                    dropped["by_asset"][symbol]["uplift_points"]
                    - baseline["by_asset"][symbol]["uplift_points"]
                    for baseline,dropped in good
                    if (baseline["by_asset"][symbol]["uplift_points"] is not None
                        and dropped["by_asset"][symbol]["uplift_points"] is not None)
                ]
                per_symbol[symbol] = {
                    "comparable_folds":len(asset_deltas),
                    "positive_folds":sum(v > 0 for v in asset_deltas),
                    "mean_uplift_delta_points":_mean(asset_deltas),
                }
            paired[name] = {
                "paired_feasible_folds":len(good),
                "positive_uplift_folds":sum(d > 0 for d in deltas),
                "mean_uplift_delta_points":_mean(deltas),
                "mean_realized_coverage_delta_points":_mean(
                    100*(dropped["check_metrics"]["coverage"]
                         - baseline["check_metrics"]["coverage"])
                    for baseline,dropped in good),
                "mean_log_loss_improvement":_mean(
                    baseline["check_metrics"]["raw_log_loss"]
                    - dropped["check_metrics"]["raw_log_loss"]
                    for baseline,dropped in good),
                "by_symbol":per_symbol,
            }
        summary[k] = {"variants":by_model,"paired_drop_vs_full":paired}
    return summary


def run_feature_reduction(
    shared:pd.DataFrame,*,mode:str,only_features:tuple[str,...]=(),
    max_folds:int|None=None,n_estimators:int=150,seed:int=42,
    train_days:int=120,validation_days:int=14,
    test_days:int=14,step_days:int=14,final_holdout_days:int=90,
) -> dict:
    variants = variants_for(mode,only_features=only_features)
    if set(shared.symbol) != set(SEEN_SYMBOLS):
        raise ValueError("Exactly four seen assets required; ADA forbidden")
    if (not set(FEATURE_COLUMNS).issubset(shared.columns)
            or not np.isfinite(shared[FEATURE_COLUMNS].to_numpy(dtype=float)).all()):
        raise ValueError("Missing or nonfinite 30 features")
    if ((train_days,validation_days,test_days,step_days,final_holdout_days)
            != (120,14,14,14,90)):
        raise ValueError("Fixed H12-purged 120/14/14/14/90 protocol required")
    if n_estimators < 20 or (max_folds is not None and max_folds < 1):
        raise ValueError("Bad estimator count or max_folds")
    folds = common_folds(
        shared,train_days=train_days,validation_days=validation_days,
        test_days=test_days,step_days=step_days,
        final_holdout_days=final_holdout_days,
    )
    available = len(folds)
    if max_folds is not None:
        folds = folds[:max_folds]
    if not folds:
        raise ValueError("No folds")
    rows = []
    for fold in folds:
        chunks,bounds = shared_partitions(shared,fold)
        stop_end = pd.Timestamp(bounds["early_stop_end_exclusive_utc"])
        tune_end = pd.Timestamp(bounds["unused_tuning_end_exclusive_utc"])
        t,maturity = shared.bar_close_at,shared.label_mature_at
        # H12-purged earlier tuning, but target columns MUST NOT be used
        # for threshold selection (_fit_and_predict tunes on scores only).
        tune = shared.loc[t.ge(stop_end)&t.lt(tune_end)&maturity.lt(tune_end)]
        if (len(tune) < 100 or not tune.bar_close_at.lt(tune_end).all()
                or not tune.label_mature_at.lt(tune_end).all()):
            raise ValueError("Non-purged or insufficient tuning rows")
        outcomes={}
        for h in HORIZONS:
            results={}
            for name,features in variants.items():
                result = _fit_and_predict(
                    chunks["train"],chunks["early_stopping"],tune,
                    chunks["validation_check"],cols=list(features),
                    horizon=h,iterations=n_estimators,seed=seed,
                )
                results[name]=result
                m=result["check_metrics"]
                print(f"fold={fold.fold_id} H{h} {name} feasible={result['feasible']} "
                      f"uplift={None if m is None else m['uplift_points']} "
                      f"coverage={None if m is None else m['coverage']}",flush=True)
            outcomes[f"H{h}"]=results
        rows.append({
            "fold_id":int(fold.fold_id),**bounds,
            "partition_rows":{name:len(frame) for name,frame in chunks.items()},
            "tuning_rows":len(tune),"horizons":outcomes,
        })
    return {
        "protocol":PROTOCOL,
        "research_status":"PARTIAL_SMOKE_NOT_FINAL" if len(folds)<available
                          else "EXPLORATORY_PREVIOUSLY_INSPECTED_DEV_CHECKS",
        "mode":mode,
        "feature_schema":FEATURE_SCHEMA_VERSION,
        "feature_count":len(FEATURE_COLUMNS),
        "variant_columns":{k:list(v) for k,v in variants.items()},
        "removed_features":{
            k:[c for c in FEATURE_COLUMNS if c not in cols]
            for k,cols in variants.items()
        },
        "folds_available":available,"folds_analyzed":len(folds),
        "expected_fit_count":len(folds)*len(HORIZONS)*len(variants),
        "target_tuning_coverage":0.4,
        "coverage_note":"Tuning target only; check coverage can drift.",
        "sign_convention":"positive drop-vs-full delta = removal improved uplift",
        "H12_purge":True,"ADA_loaded_or_scored":False,
        "fold_OOS_scored":False,"final_90_days_scored":False,
        "frozen_P3_models_modified":False,
        "limitations":[
            "Checks have already been studied; no independent validation.",
            "Training folds overlap, multiple comparisons may yield chance gains.",
            "Ablation effects are conditional on other retained correlated inputs.",
            "No automatic feature selection, frozen model promotion, or trade returns.",
        ],
        "summary":summarize(rows,variants),
        "folds":rows,
    }


def main() -> None:
    parser=argparse.ArgumentParser(description="P2 historical feature removal H2 H6")
    parser.add_argument("--mode",required=True,choices=("group","individual"))
    parser.add_argument("--only-features",nargs="+",default=None)
    parser.add_argument("--data-root",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--months",default=24,type=int)
    parser.add_argument("--max-folds",default=None,type=int,
                        help="Partial technical smoke only")
    parser.add_argument("--n-estimators",default=150,type=int)
    parser.add_argument("--seed",default=42,type=int)
    parser.add_argument("--verify-hashes",action="store_true")
    a=parser.parse_args()
    if a.months!=24:
        parser.error("24 months required")
    if a.output.exists():
        parser.error("Refusing to overwrite historical report")
    selected=tuple(a.only_features or ())
    variants_for(a.mode,only_features=selected)
    store=CandleStore(a.data_root)
    candles,manifests={},[]
    for symbol in SEEN_SYMBOLS:
        print("Loading seen asset",symbol,flush=True)
        candles[symbol],entries=load_symbol(
            store,symbol,months=a.months,verify_hashes=a.verify_hashes)
        manifests.extend(entries)
    shared=build_shared_examples(candles)
    del candles
    report=run_feature_reduction(
        shared,mode=a.mode,only_features=selected,max_folds=a.max_folds,
        n_estimators=a.n_estimators,seed=a.seed)
    report["dataset_fingerprint_sha256"]=manifest_fingerprint(manifests)
    report["lightgbm_version"]=lgb.__version__
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x",encoding="utf-8") as handle:
        json.dump(report,handle,indent=2,ensure_ascii=False,allow_nan=False)
    print("Saved feature reduction report:",a.output.resolve(),flush=True)


if __name__=="__main__":
    main()
