"""P2 predeclared 30-vs-32 trajectory feature test on reused dev checks ONLY.

H2 primary, H6 secondary. The selection mask is created ONLY from the
unchanged 30-feature model's previous (unlabeled) 3d tuning scores.
New and old predictions are paired on exactly the same rows. NEUTRAL is wrong
for both forced-UP/DOWN directions. A prior 36-fold report is an executable
reproducibility gate: previous baseline counts must match for EVERY fold.

No ADA, fold OOS tests, last 90d, live data, P3 or model pack writes.
This is exploratory reuse of repeatedly inspected historical checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from ingestion.storage import CandleStore
from mlpipeline.confidence_diagnostics import decision_and_margin
from mlpipeline.coverage_matched_diagnostics import threshold_for_target
from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.directional_disagreement_audit import (
    disagreement_counts, verify_baseline_report,
)
from mlpipeline.directional_rule_benchmark import (
    HORIZONS, TARGET_COVERAGE, build_causal_contexts, fixed_mask_metrics,
    majority_direction, parts_for_fold, prepare_fold_examples, rule_predictions,
)
from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS
from mlpipeline.features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, feature_matrix
from mlpipeline.label_stability_audit import load_plan
from mlpipeline.labels import LABEL_SCHEMA_VERSION
from mlpipeline.predictability_decomposition import _prior_from_train, stage_scores
from mlpipeline.trainer import _class_probs
from mlpipeline.trajectory_direction_features import (
    TRAJECTORY_COLUMNS, TRAJECTORY_SCHEMA, build_trajectory_features,
)
from mlpipeline.two_stage_diagnostics import _cpu_model

PROTOCOL = "p2_h2_primary_30_vs_32_trajectory_predeclared_v1"
BASELINE = "baseline_30"
CANDIDATE = "baseline_plus_trajectory_32"
VARIANTS = {BASELINE:tuple(FEATURE_COLUMNS),
            CANDIDATE:(*FEATURE_COLUMNS,*TRAJECTORY_COLUMNS)}
REVERSIONS = {2:"reversion_3", 6:"reversion_6"}
MIN_GAIN_POINTS = 1.5
MIN_POSITIVE_FOLDS = 26
REQUIRED_COMPARABLE_FOLDS = 35
MAX_LOSS_INCREASE = 0.005


def build_trajectory_contexts(candles:dict[str,pd.DataFrame],
                              cutoff:pd.Timestamp)->pd.DataFrame:
    """Clip candles before ANY trajectory construction; do not create labels."""
    if set(candles)!=set(SEEN_SYMBOLS) or cutoff.tzinfo is None:
        raise ValueError("Four seen symbols and UTC cutoff required")
    parts=[]
    for symbol in SEEN_SYMBOLS:
        clipped=candles[symbol].loc[
            candles[symbol].bar_close_at.lt(cutoff)].copy()
        if clipped.empty:
            raise ValueError(f"{symbol}: no allowed historical candles")
        parts.append(build_trajectory_features(clipped))
    out=pd.concat(parts,ignore_index=True)
    if out.duplicated(["symbol","bar_close_at"]).any():
        raise ValueError("Duplicate trajectory keys")
    return out


def align_features(base:pd.DataFrame, trajectory:pd.DataFrame)->pd.DataFrame:
    """Preserve EVERY baseline row, exact order, label and partition geometry."""
    keys=["symbol","bar_close_at"]
    if (base.empty or base.duplicated(keys).any()
            or trajectory.duplicated(keys).any()
            or any(k not in trajectory for k in (*keys,*TRAJECTORY_COLUMNS))
            or any(k in base for k in TRAJECTORY_COLUMNS)):
        raise ValueError("Invalid baseline/trajectory join keys or schema")
    aligned=base.merge(
        trajectory[[*keys,*TRAJECTORY_COLUMNS]],on=keys,how="left",
        sort=False,validate="one_to_one")
    if len(aligned)!=len(base) or not aligned[keys].equals(base[keys]):
        raise AssertionError("Feature join changed baseline rows, order or keys")
    extras=aligned[list(TRAJECTORY_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(extras).all():
        raise ValueError("New feature warmup/NaN/inf would change baseline rows")
    aligned.loc[:,list(TRAJECTORY_COLUMNS)]=extras.astype("float32")
    return aligned


def direction_sign(probabilities:np.ndarray,training_y:np.ndarray)->np.ndarray:
    """Exactly reuse benchmark UP-vs-DOWN sign, even if class-1 is top."""
    p=np.asarray(probabilities,dtype=float)
    if (p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all()
            or (p<0).any() or (p>1).any()
            or not np.allclose(p.sum(axis=1),1,atol=1e-5)):
        raise ValueError("Expected aligned three-class probabilities")
    fallback=majority_direction(training_y)
    return np.where(p[:,2]>p[:,0],2,
                    np.where(p[:,2]<p[:,0],0,fallback)).astype("int8")


def score_pair(truth:np.ndarray,base_proba:np.ndarray,
               candidate_proba:np.ndarray,mask:np.ndarray,
               train_y:np.ndarray, reversion:np.ndarray)->dict|None:
    """Strictly paired outcomes with all neutral outcomes counted as errors."""
    y=np.asarray(truth,dtype=int)
    b=direction_sign(base_proba,train_y)
    c=direction_sign(candidate_proba,train_y)
    rv=np.asarray(reversion,dtype=int)
    chosen=np.asarray(mask)
    if (y.shape!=b.shape or y.shape!=c.shape or y.shape!=rv.shape
            or y.shape!=chosen.shape or chosen.dtype.kind!="b"
            or not np.isin(y,[0,1,2]).all()
            or not np.isin(rv,[0,2]).all()):
        raise ValueError("Mismatched paired samples")
    majority=np.full(len(y),majority_direction(train_y),dtype=np.int8)
    bm=fixed_mask_metrics(y,b,chosen,majority)
    cm=fixed_mask_metrics(y,c,chosen,majority)
    rm=fixed_mask_metrics(y,rv,chosen,majority)
    disagreement=disagreement_counts(y,c,b,chosen)
    if bm is None:
        if cm is not None or rm is not None or disagreement is not None:
            raise AssertionError("Infeasible mask produced paired scores")
        return None
    if (bm["n"]!=cm["n"] or bm["n"]!=rm["n"]
            or cm["correct"]-bm["correct"]!=
            disagreement["disagree_ml_correct"]-
            disagreement["disagree_reversion_correct"]):
        raise AssertionError("Paired mask counts do not reconcile")
    return {
        "n":bm["n"],"baseline":bm,"candidate":cm,"reversion":rm,
        "gain_candidate_points":100*(cm["correct"]-bm["correct"])/bm["n"],
        "gain_candidate_vs_reversion_points":
            100*(cm["correct"]-rm["correct"])/bm["n"],
        "candidate_vs_baseline_disagreements":disagreement,
    }


def verify_original_fold(previous:dict,*,horizon:int,threshold:dict,
                         all_check:dict,selected:dict|None,
                         by_asset:dict)->None:
    """One incorrect reproduced baseline count invalidates the entire run."""
    hk=f"H{horizon}"
    old=previous["horizons"][hk]
    if bool(threshold["feasible"])!=bool(old["ml_tuning"]["feasible"]):
        raise ValueError(f"{hk}: baseline feasibility mismatch")
    old_threshold=old["ml_tuning"]["threshold"]
    if threshold["feasible"] and not np.isclose(
            threshold["threshold"],old_threshold,atol=1e-12,rtol=0):
        raise ValueError(f"{hk}: baseline tuning threshold drift")
    # An infeasible score-only threshold may still have a numeric cutoff
    # (e.g. ties): the prior feasibility flag, not None-ness, is authoritative.
    if (old["best_iteration"]!=all_check["best_iteration"]
            or old["train_counts"]!=all_check["train_counts"]):
        raise ValueError(f"{hk}: baseline training configuration drift")
    for view,observed in (("all_check",all_check),
                          ("shared_ml_selected",selected)):
        expected=old["rules"]["lightgbm_30"][view]
        if observed is None:
            if expected is not None:
                raise ValueError(f"{hk}/{view}: missing baseline score")
            continue
        b=observed["baseline"]
        if (expected is None or
                any(b[k]!=expected[k] for k in (
                    "n","correct","neutral_errors","opposite_errors",
                    "predicted_up","predicted_down"))):
            raise ValueError(f"{hk}/{view}: baseline benchmark count drift")
    for symbol in SEEN_SYMBOLS:
        observed=by_asset[symbol]
        expected=old["by_asset_selected"][symbol]["lightgbm_30"]
        if observed is None:
            if expected is not None:
                raise ValueError(f"{hk}/{symbol}: missing prior baseline result")
        elif (expected is None or
              any(observed["baseline"][k]!=expected[k] for k in (
                  "n","correct","neutral_errors","opposite_errors",
                  "predicted_up","predicted_down"))):
            raise ValueError(f"{hk}/{symbol}: prior per-asset count drift")


def _matrix(frame:pd.DataFrame, columns:tuple[str,...])->pd.DataFrame:
    x=frame.loc[:,list(columns)].astype("float32")
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite model inputs")
    return x


def run_experiment(candles:dict[str,pd.DataFrame], source:dict,
                   baseline_report:dict,*,fingerprint:str,
                   max_folds:int|None=None)->dict:
    plan=load_plan(source)
    old_folds=verify_baseline_report(baseline_report,source)
    if fingerprint!=source.get("dataset_fingerprint_sha256"):
        raise ValueError("P0 dataset fingerprint differs from the original plan")
    if set(candles)!=set(SEEN_SYMBOLS):
        raise ValueError("Only four previously seen assets are authorized")
    if max_folds is not None:
        if type(max_folds) is not int or not 1<=max_folds<=len(plan):
            raise ValueError("Bad technical smoke fold count")
        plan=plan[:max_folds]
    cutoff=plan[-1]["check_end"]
    base_contexts=build_causal_contexts(candles,cutoff)
    trajectory=build_trajectory_contexts(candles,cutoff)
    records=[]
    for fold in plan:
        shared=align_features(
            prepare_fold_examples(base_contexts,fold["check_end"]),
            trajectory)
        parts,bounds=parts_for_fold(
            shared,fold,source["folds"][fold["fold_id"]])
        train,early,tune,check=(parts[k] for k in (
            "train","early_stopping","tuning","validation_check"))
        horizon_results={}
        for h in HORIZONS:
            hk=f"H{h}"
            target=f"target_h{h}"
            ytrain=train[target].to_numpy(dtype=int)
            yearly=early[target].to_numpy(dtype=int)
            truth=check[target].to_numpy(dtype=int)
            counts=np.bincount(ytrain,minlength=3)
            if (np.any(counts==0) or set(yearly.tolist())!={0,1,2}):
                raise ValueError("Missing classes in training or early-stop")
            model={}
            for name,cols in VARIANTS.items():
                clf=_cpu_model(binary=False,n_estimators=150,seed=42)
                clf.fit(
                    _matrix(train,cols),ytrain,
                    eval_set=[(_matrix(early,cols),yearly)],
                    eval_metric="multi_logloss",
                    callbacks=[lgb.early_stopping(stopping_rounds=30,verbose=False)])
                model[name]={
                    "clf":clf,
                    "tune":_class_probs(clf,_matrix(tune,cols)),
                    "check":_class_probs(clf,_matrix(check,cols)),
                    "best_iteration":int(clf.best_iteration_),
                }
            baseline=model[BASELINE]
            candidate=model[CANDIDATE]
            pred_t,margin_t=decision_and_margin(
                baseline["tune"],counts,alpha=1.0)
            threshold=threshold_for_target(
                pred_t,margin_t,target_coverage=TARGET_COVERAGE)
            pred_c,margin_c=decision_and_margin(
                baseline["check"],counts,alpha=1.0)
            selected=((pred_c!=1)&(margin_c>=threshold["threshold"])
                      if threshold["feasible"]
                      else np.zeros(len(check),dtype=bool))
            rev=rule_predictions(check,ytrain,baseline["check"])[REVERSIONS[h]]
            all_paired=score_pair(
                truth,baseline["check"],candidate["check"],
                np.ones(len(check),dtype=bool),ytrain,rev)
            all_paired["best_iteration"]=baseline["best_iteration"]
            all_paired["train_counts"]={
                k:int(counts[i])
                for i,k in enumerate(("DOWN","NEUTRAL","UP"))}
            paired=(score_pair(
                truth,baseline["check"],candidate["check"],selected,ytrain,rev)
                if threshold["feasible"] else None)
            symbols=check.symbol.to_numpy()
            asset={
                sym:(score_pair(
                    truth,baseline["check"],candidate["check"],
                    selected & (symbols==sym),ytrain,rev)
                    if threshold["feasible"] else None)
                for sym in SEEN_SYMBOLS
            }
            verify_original_fold(old_folds[fold["fold_id"]],
                                 horizon=h,threshold=threshold,
                                 all_check=all_paired,selected=paired,
                                 by_asset=asset)
            if paired is not None:
                for counter in ("n","correct"):
                    for variant in ("baseline","candidate","reversion"):
                        n=sum(
                            asset[s][variant][counter] for s in SEEN_SYMBOLS
                            if asset[s] is not None)
                        if n!=paired[variant][counter]:
                            raise AssertionError("Per-asset paired row count drift")
            prior_move,prior_up=_prior_from_train(ytrain)
            probability_metrics={}
            for name,entry in model.items():
                p=entry["check"]
                onehot=np.eye(3,dtype=float)[truth]
                probability_metrics[name]={
                    "log_loss":float(log_loss(truth,p,labels=[0,1,2])),
                    "multiclass_brier_sum":float(
                        np.mean(np.sum((p-onehot)**2,axis=1))),
                    "movement_and_oracle_direction":stage_scores(
                        truth,p,
                        training_movement_prior=prior_move,
                        training_up_given_movement=prior_up),
                    "best_iteration":entry["best_iteration"],
                }
            horizon_results[hk]={
                "comparison_reversion":REVERSIONS[h],
                "mask_source":"baseline_30_alpha1_tuning_score_only",
                "mask_tuning":threshold,
                "realized_check_coverage":(
                    float(selected.mean()) if threshold["feasible"] else None),
                "all_check":all_paired,
                "shared_baseline_selected":paired,
                "by_asset_selected":asset,
                "raw_probability_metrics_all_check":probability_metrics,
            }
            print(
                f"fold={fold['fold_id']} {hk} "
                f"feasible={threshold['feasible']} "
                f"selected_n={0 if paired is None else paired['n']} "
                f"delta_32_vs_30_points={None if paired is None else paired['gain_candidate_points']}",
                flush=True)
        records.append({
            "fold_id":fold["fold_id"],"bounds":bounds,
            "partition_rows":{k:len(v) for k,v in parts.items()},
            "horizons":horizon_results,
        })
        del shared,parts
    result={
        "protocol":PROTOCOL,
        "research_status":"EXPLORATORY_ALREADY_INSPECTED_DEV_CHECKS",
        "source_dataset_fingerprint_sha256":fingerprint,
        "folds_analyzed":len(records),
        "last_dev_check_end_exclusive_utc":cutoff.isoformat(),
        "horizons":list(HORIZONS),
        "baseline_features":list(FEATURE_COLUMNS),
        "candidate_features":list(VARIANTS[CANDIDATE]),
        "base_feature_schema":FEATURE_SCHEMA_VERSION,
        "trajectory_feature_schema":TRAJECTORY_SCHEMA,
        "label_schema":LABEL_SCHEMA_VERSION,
        "label_multiplier":0.5,
        "n_estimators":150,"seed":42,"target_tuning_coverage":TARGET_COVERAGE,
        "previous_baseline_counts_reproduced_each_fold":True,
        "identical_partitions_and_baseline_mask":True,
        "no_check_or_tuning_labels_used_for_mask":True,
        "current_fold_labels_clipped_before_own_check_end":True,
        "ADA_loaded_or_scored":False,
        "fold_OOS_scored":False,
        "final90_scored":False,
        "frozen_P3_or_forward_models_modified":False,
        "warnings":[
            "This candidate and thresholds were predeclared after inspecting the same development checks.",
            "Overlapping folds and correlated assets preclude iid/significance assumptions.",
            "Forced directional sign despite multiclass NEUTRAL; selected mask is baseline-ML-dependent.",
            "True-mover conditional metrics are retrospectively oracle-stratified.",
            "No profitability, calibrated confidence or deployment inference.",
        ],
        "folds":records,
    }
    result["summary"]=summarize(records)
    return result


def _mean(values)->float|None:
    v=[float(x) for x in values if x is not None]
    return float(np.mean(v)) if v else None


def summarize(folds:list[dict])->dict:
    """Means over folds are the PRIMARY measure, not pooled sample counts."""
    if not folds:
        raise ValueError("No folds to summarize")
    summary={}
    for h in HORIZONS:
        hk=f"H{h}"
        records=[f["horizons"][hk] for f in folds]
        good=[r["shared_baseline_selected"] for r in records
              if r["shared_baseline_selected"] is not None]
        differences=[r["gain_candidate_points"] for r in good]
        total=sum(r["n"] for r in good)
        by_asset={}
        for sym in SEEN_SYMBOLS:
            pieces=[r["by_asset_selected"][sym] for r in records
                    if r["by_asset_selected"][sym] is not None]
            by_asset[sym]={
                "comparable_folds":len(pieces),
                "mean_gain_points":_mean(r["gain_candidate_points"] for r in pieces),
                "total_selected_rows":sum(r["n"] for r in pieces),
                "pooled_gain_points":(
                    100*sum(r["candidate"]["correct"]-r["baseline"]["correct"]
                            for r in pieces)/sum(r["n"] for r in pieces)
                    if pieces else None),
            }
        prob=[r["raw_probability_metrics_all_check"] for r in records]
        logloss_deltas=[
            p[CANDIDATE]["log_loss"]-p[BASELINE]["log_loss"] for p in prob
        ]
        summary[hk]={
            "comparable_folds":len(good),
            "total_folds":len(records),
            "selected_rows_across_folds":total,
            "mean_baseline_precision":_mean(
                r["baseline"]["precision"] for r in good),
            "mean_candidate_precision":_mean(
                r["candidate"]["precision"] for r in good),
            "mean_gain_points":_mean(differences),
            "pooled_gain_points":(
                100*sum(r["candidate"]["correct"]-r["baseline"]["correct"]
                        for r in good)/total if total else None),
            "folds_candidate_wins":sum(x>0 for x in differences),
            "folds_baseline_wins":sum(x<0 for x in differences),
            "folds_tied":sum(x==0 for x in differences),
            "by_asset":by_asset,
            "mean_logloss_candidate_minus_baseline":_mean(logloss_deltas),
            "mean_brier_candidate_minus_baseline":_mean(
                p[CANDIDATE]["multiclass_brier_sum"]-
                p[BASELINE]["multiclass_brier_sum"] for p in prob),
            "mean_movement_auc_gain":_mean(
                (p[CANDIDATE]["movement_and_oracle_direction"]
                 ["movement_detection"]["roc_auc"]-
                 p[BASELINE]["movement_and_oracle_direction"]
                 ["movement_detection"]["roc_auc"])
                if (p[CANDIDATE]["movement_and_oracle_direction"]
                    ["movement_detection"]["roc_auc"] is not None
                    and p[BASELINE]["movement_and_oracle_direction"]
                    ["movement_detection"]["roc_auc"] is not None)
                else None for p in prob),
        }
    h2=summary["H2"]
    gates={
        "fixed_35_comparable_folds":
            h2["comparable_folds"]==REQUIRED_COMPARABLE_FOLDS,
        "mean_h2_delta_at_least_1_5_points":
            h2["mean_gain_points"] is not None
            and h2["mean_gain_points"]>=MIN_GAIN_POINTS,
        "at_least_26_positive_h2_folds":
            h2["folds_candidate_wins"]>=MIN_POSITIVE_FOLDS,
        "all_four_asset_h2_gains_positive":all(
            x["mean_gain_points"] is not None and x["mean_gain_points"]>0
            for x in h2["by_asset"].values()),
        "mean_h2_logloss_increase_at_most_0_005":
            h2["mean_logloss_candidate_minus_baseline"] is not None
            and h2["mean_logloss_candidate_minus_baseline"]<=MAX_LOSS_INCREASE,
    }
    summary["predeclared_gates"]=gates
    summary["decision"]=(
        "TECHNICAL_SMOKE_NOT_SCIENTIFIC_RESULT" if len(folds)<36 else
        "CANDIDATE_FOR_FUTURE_INDEPENDENT_CONFIRMATION_NOT_PROMOTION"
        if all(gates.values()) else "EXPLORATORY_NEGATIVE_OR_INCONCLUSIVE"
    )
    return summary


def main()->None:
    p=argparse.ArgumentParser(
        description="P2 predeclared paired baseline30 vs trajectory32 H2 H6")
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--fold-plan",type=Path,required=True)
    p.add_argument("--baseline-report",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--months",type=int,default=24)
    p.add_argument("--max-folds",type=int)
    p.add_argument("--verify-hashes",action="store_true")
    args=p.parse_args()
    if args.months!=24:
        p.error("Fixed 24-month research archive required")
    if args.output.exists():
        p.error("Refusing to overwrite historical research output")
    if args.output.resolve().is_relative_to(args.data_root.resolve().parent):
        p.error("Output must be outside the entire production cryptoML root")
    if args.output.resolve() in (
            args.fold_plan.resolve(),args.baseline_report.resolve()):
        p.error("Output cannot overwrite source reports")
    source_bytes=args.fold_plan.read_bytes()
    baseline_bytes=args.baseline_report.read_bytes()
    source=json.loads(source_bytes)
    baseline=json.loads(baseline_bytes)
    load_plan(source)
    verify_baseline_report(baseline,source)
    store=CandleStore(args.data_root)
    candles={}
    manifests=[]
    for symbol in SEEN_SYMBOLS:
        print("Reading P0 in consultation mode:",symbol,flush=True)
        candles[symbol],parts=load_symbol(
            store,symbol,months=args.months,verify_hashes=args.verify_hashes)
        manifests.extend(parts)
    output=run_experiment(
        candles,source,baseline,fingerprint=manifest_fingerprint(manifests),
        max_folds=args.max_folds)
    output["fold_plan_sha256"]=hashlib.sha256(source_bytes).hexdigest()
    output["original_baseline_report_sha256"]=hashlib.sha256(baseline_bytes).hexdigest()
    output["lightgbm_version"]=lgb.__version__
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as out:
        json.dump(output,out,ensure_ascii=False,indent=2,allow_nan=False)
    print("Saved P2 trajectory comparison:",args.output.resolve(),flush=True)
    for h in HORIZONS:
        s=output["summary"][f"H{h}"]
        print(
            f"H{h} comparable={s['comparable_folds']} "
            f"gain_mean_fold_points={s['mean_gain_points']} "
            f"positive_folds={s['folds_candidate_wins']} "
            f"mean_logloss_delta={s['mean_logloss_candidate_minus_baseline']}",
            flush=True)
    print("Exploratory-only verdict:",output["summary"]["decision"],flush=True)


if __name__=="__main__":
    main()
