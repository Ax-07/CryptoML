"""Historical P2 ONLY: paired H2/H6 causal market-structure feature ablation.

Train/evaluate on identical rows/folds; no ADA, OOS test, final 90-day
scoring, threshold tuning on check outcomes, or changes to frozen packs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ingestion.storage import CandleStore
from mlpipeline.confidence_diagnostics import decision_and_margin, directional_metrics
from mlpipeline.coverage_matched_diagnostics import threshold_for_target
from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.evaluation import evaluate
from mlpipeline.expanded_horizon_diagnostics import (
    SEEN_SYMBOLS, build_shared_examples, common_folds, shared_partitions,
)
from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.market_structure_features import (
    STRUCTURE_COLUMNS, STRUCTURE_SCHEMA, build_market_structure_features,
)
from mlpipeline.trainer import _class_probs
from mlpipeline.two_stage_diagnostics import _cpu_model

VARIANTS = {
    "baseline_30": tuple(FEATURE_COLUMNS),
    "baseline_plus_structure_42": (*FEATURE_COLUMNS,*STRUCTURE_COLUMNS),
    "structure_only_12": STRUCTURE_COLUMNS,
}
HORIZONS = (2,6)
TARGET_COVERAGE = 0.40


def shared_with_structure(candles:dict[str,pd.DataFrame]) -> pd.DataFrame:
    if set(candles)!=set(SEEN_SYMBOLS):
        raise ValueError("Four known symbols only, ADA excluded")
    structures=[]
    for symbol in SEEN_SYMBOLS:
        structures.append(build_market_structure_features(candles[symbol]))
    structure=pd.concat(structures,ignore_index=True)
    if structure.duplicated(["symbol","bar_close_at"]).any():
        raise ValueError("Duplicate structure keys")
    shared=build_shared_examples(candles)
    out=shared.merge(structure,on=["symbol","bar_close_at"],how="left",
                     validate="one_to_one",sort=False)
    if len(out)!=len(shared):
        raise AssertionError("Feature join changed shared row count")
    # Fair paired training: identical eligible observations for every variant.
    eligible=np.isfinite(out[list(STRUCTURE_COLUMNS)].to_numpy(dtype="float64")).all(axis=1)
    out=out.loc[eligible].copy()
    if out.empty:
        raise ValueError("No jointly eligible confirmed-swing observations")
    out.loc[:,list(STRUCTURE_COLUMNS)]=out[list(STRUCTURE_COLUMNS)].astype("float32")
    return out


def _fit_and_predict(train,early,tune,check,*,cols,horizon,iterations,seed):
    target=f"target_h{horizon}"
    yt=train[target].to_numpy(dtype=int)
    ye=early[target].to_numpy(dtype=int)
    counts=np.bincount(yt,minlength=3)
    if (counts==0).any() or set(ye)!={0,1,2}:
        raise ValueError("Missing training/early-stopping class")
    m=_cpu_model(binary=False,n_estimators=iterations,seed=seed)
    m.fit(train.loc[:,cols].astype("float32"),yt,
          eval_set=[(early.loc[:,cols].astype("float32"),ye)],
          eval_metric="multi_logloss",
          callbacks=[lgb.early_stopping(stopping_rounds=30,verbose=False)])
    prob_t=_class_probs(m,tune.loc[:,cols].astype("float32"))
    pred_t,margin_t=decision_and_margin(prob_t,counts,alpha=1.)
    fitted=threshold_for_target(pred_t,margin_t,target_coverage=TARGET_COVERAGE)
    prob_c=_class_probs(m,check.loc[:,cols].astype("float32"))
    pred_c,margin_c=decision_and_margin(prob_c,counts,alpha=1.)
    if not fitted["feasible"]:
        return {"feasible":False,"tuning":fitted,"best_iteration":int(m.best_iteration_),
                "check_metrics":None,"by_asset":None}
    threshold=fitted["threshold"]
    truth=check[target].to_numpy(dtype=int)
    metrics=directional_metrics(truth,pred_c,margin_c,threshold=threshold)
    metrics["raw_log_loss"]=float(evaluate(truth,prob_c)["log_loss"])
    per_asset={}
    for symbol in SEEN_SYMBOLS:
        mask=check.symbol.to_numpy()==symbol
        per_asset[symbol]=directional_metrics(
            truth[mask],pred_c[mask],margin_c[mask],threshold=threshold)
    return {"feasible":True,"tuning":fitted,
            "best_iteration":int(m.best_iteration_),
            "check_metrics":metrics,"by_asset":per_asset}


def _mean(values):
    values=list(values)
    if not values or any(v is None for v in values):
        return None
    return float(np.mean(values))


def _summary(records):
    result={}
    for h in HORIZONS:
        hk=f"H{h}"
        result[hk]={}
        for v in VARIANTS:
            cells=[r["horizons"][hk][v] for r in records]
            feasible=[s for s in cells if s["feasible"]]
            result[hk][v]={
                "feasible_folds":len(feasible),
                "total_folds":len(cells),
                "mean_directional_precision":_mean(
                    x["check_metrics"]["directional_precision"] for x in feasible),
                "mean_uplift_points":_mean(
                    x["check_metrics"]["uplift_points"] for x in feasible),
                "mean_realized_coverage":_mean(
                    x["check_metrics"]["coverage"] for x in feasible),
                "mean_log_loss":_mean(
                    x["check_metrics"]["raw_log_loss"] for x in feasible),
            }
        result[hk]["paired_vs_baseline"]={}
        for v in tuple(VARIANTS)[1:]:
            paired=[(r["horizons"][hk]["baseline_30"],
                     r["horizons"][hk][v]) for r in records]
            valid=[(a,b) for a,b in paired if a["feasible"] and b["feasible"]
                   and a["check_metrics"]["uplift_points"] is not None
                   and b["check_metrics"]["uplift_points"] is not None]
            diffs=[b["check_metrics"]["uplift_points"] -
                   a["check_metrics"]["uplift_points"] for a,b in valid]
            result[hk]["paired_vs_baseline"][v]={
                "paired_feasible_folds":len(valid),
                "mean_uplift_delta_points":_mean(diffs),
                "positive_uplift_folds":sum(d>0 for d in diffs),
                "mean_realized_coverage_delta_points":_mean(
                    100*(b["check_metrics"]["coverage"]-a["check_metrics"]["coverage"])
                    for a,b in valid),
                "mean_log_loss_improvement":_mean(
                    a["check_metrics"]["raw_log_loss"]-
                    b["check_metrics"]["raw_log_loss"] for a,b in valid),
            }
    return result


def compare_structure(shared:pd.DataFrame,*,n_estimators=150,seed=42,
                      train_days=120,validation_days=14,test_days=14,
                      step_days=14,final_holdout_days=90):
    if set(shared.symbol)!=set(SEEN_SYMBOLS):
        raise ValueError("ADA is forbidden in P2 structure experiment")
    if n_estimators<20:
        raise ValueError("At least 20 estimators required")
    all_cols=set((*FEATURE_COLUMNS,*STRUCTURE_COLUMNS))
    if not all_cols.issubset(shared.columns) or (
        not np.isfinite(shared[list(all_cols)].to_numpy(dtype=float)).all()
    ):
        raise ValueError("Nonfinite/missing features, use shared_with_structure")
    folds=common_folds(shared,train_days=train_days,validation_days=validation_days,
                       test_days=test_days,step_days=step_days,
                       final_holdout_days=final_holdout_days)
    results=[]
    for fold in folds:
        parts,bounds=shared_partitions(shared,fold)
        # Explicitly reconstruct the untouched 3-day TUNING slice,
        # never using test, check, future-matured or holdout outcomes.
        early_end=pd.Timestamp(bounds["early_stop_end_exclusive_utc"])
        tune_end=pd.Timestamp(bounds["unused_tuning_end_exclusive_utc"])
        t=shared.bar_close_at
        maturity=shared.label_mature_at
        tune=shared.loc[t.ge(early_end)&t.lt(tune_end)&maturity.lt(tune_end)]
        if len(tune)<100:
            raise ValueError("Insufficient H12-purged tuning rows")
        scores={}
        for h in HORIZONS:
            runs={}
            for name,cols in VARIANTS.items():
                runs[name]=_fit_and_predict(
                    parts["train"],parts["early_stopping"],tune,
                    parts["validation_check"],cols=list(cols),horizon=h,
                    iterations=n_estimators,seed=seed)
                m=runs[name]["check_metrics"]
                print(f"fold={fold.fold_id} H{h} {name} "
                      f"tuning_feasible={runs[name]['feasible']} "
                      f"uplift={None if m is None else m['uplift_points']} "
                      f"coverage={None if m is None else m['coverage']}",flush=True)
            scores[f"H{h}"]=runs
        results.append({"fold_id":fold.fold_id,**bounds,
                        "counts":{**{k:len(v) for k,v in parts.items()},
                                  "tuning":len(tune)},"horizons":scores})
    return {
        "protocol":"cpu_p2_h2_h6_market_structure_causal_v1",
        "research_status":"exploratory_development_checks_previously_seen",
        "variant_columns":{v:list(cols) for v,cols in VARIANTS.items()},
        "structure_schema":STRUCTURE_SCHEMA,
        "swing_left_right_bars":[2,2],
        "target_tuning_coverage":TARGET_COVERAGE,
        "coverage_claim":"tuning_target_only_realized_check_coverage_reported",
        "h12_purge":True,"alpha":1.0,
        "ADA_loaded_or_scored":False,
        "fold_oos_test_scored":False,"final_90_days_scored":False,
        "models_frozen_in_P3_modified":False,
        "notes":[
            "Identical train/tune/check rows and folds for 30/42/12 feature variants.",
            "Pivot at p becomes visible only at p+2; confirmed-only and segment-reset.",
            "No new independent validation; historic checks already inspected.",
            "Any selected feature hypothesis requires genuinely new data for confirmation.",
            "No realized returns or trade profitability estimates.",
        ],
        "folds":results,"summary":_summary(results),
    }


def main():
    p=argparse.ArgumentParser(description="P2 CPU causal HH/HL/LH/LL study H2 H6")
    p.add_argument("--data-root",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--months",default=24,type=int)
    p.add_argument("--n-estimators",default=150,type=int)
    p.add_argument("--seed",default=42,type=int)
    p.add_argument("--verify-hashes",action="store_true")
    args=p.parse_args()
    if args.months!=24:
        p.error("Full 24-month P2 experiment required")
    if args.output.exists():
        p.error("Will not overwrite previous research reports")
    store=CandleStore(args.data_root)
    candles,manifests={},[]
    for symbol in SEEN_SYMBOLS:
        print("Loading seen asset",symbol,flush=True)
        candles[symbol],entries=load_symbol(
            store,symbol,months=args.months,verify_hashes=args.verify_hashes)
        manifests.extend(entries)
    shared=shared_with_structure(candles)
    del candles
    report=compare_structure(shared,n_estimators=args.n_estimators,seed=args.seed)
    report["dataset_fingerprint_sha256"]=manifest_fingerprint(manifests)
    report["lightgbm_version"]=lgb.__version__
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print("P2 structure report:",args.output.resolve(),flush=True)


if __name__=="__main__":
    main()
