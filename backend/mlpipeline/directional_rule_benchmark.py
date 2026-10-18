"""P2 H2/H6 directional benchmark on reused development validation-checks ONLY.

Prespecified rules: train-majority UP/DOWN, momentum 1/3/6 M5, reversion 1/3/6
M5, LightGBM-30. Three DISTINCT views per rule, always same paired rows:
  * all_check: actionable sign each M5 close, NEUTRAL counts as incorrect;
  * oracle_true_movers: conditional accuracy only on true UP/DOWN labels,
    a retrospective non-deployable diagnostic;
  * shared_ml_selected: same LightGBM alpha=1 margin-based 40% tuning-score
    selection mask applied to ALL rules, independent of check labels. Rules
    depend on the ML filter here, not stand-alone trading signals.

Data security: the 36 original fixed P2 fold boundaries are read from an
immutable prior report. Historical candles are CLIPPED at each fold's own
development check end BEFORE add_labels is called. Thus no fold OOS labels
for the current fold are ever constructed. Train 120d / early 7d / score-only
tuning 3d / check 4d; label-maturity purge H12 (60min). No ADA or last90.
Previous checks were inspected; no independent evidence / PnL / promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ingestion.storage import CandleStore
from mlpipeline.confidence_diagnostics import decision_and_margin
from mlpipeline.coverage_matched_diagnostics import threshold_for_target
from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS
from mlpipeline.features import (
    FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, build_symbol_features,
    feature_matrix, join_market_context,
)
from mlpipeline.label_stability_audit import load_plan
from mlpipeline.labels import LABEL_SCHEMA_VERSION, add_labels
from mlpipeline.trainer import _class_probs
from mlpipeline.two_stage_diagnostics import _cpu_model

PROTOCOL = "p2_h2_h6_directional_simple_rules_vs_30_v1"
HORIZONS = (2, 6)
RULES = (
    "majority_train", "momentum_1", "momentum_3", "momentum_6",
    "reversion_1", "reversion_3", "reversion_6", "lightgbm_30",
)
TARGET_COVERAGE = 0.40
MAX_H = 12
BAR = pd.Timedelta(minutes=5)


def build_causal_contexts(candles: dict[str, pd.DataFrame],
                          cutoff: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Causal feature computation only; never create future targets here."""
    if set(candles) != set(SEEN_SYMBOLS):
        raise ValueError("Exactly four previously seen symbols, never ADA")
    if cutoff.tzinfo is None:
        raise ValueError("Cutoff must be UTC-aware")
    features = {}
    for sym in SEEN_SYMBOLS:
        raw = candles[sym]
        safe = raw.loc[raw.bar_close_at.lt(cutoff)].copy()
        if safe.empty:
            raise ValueError(f"{sym}: no allowed historical candles")
        features[sym] = build_symbol_features(safe)
    return {
        sym: join_market_context(features[sym], features["BTCUSDT"],
                                 features["ETHUSDT"])
        for sym in SEEN_SYMBOLS
    }


def prepare_fold_examples(contexts: dict[str, pd.DataFrame],
                          check_end: pd.Timestamp) -> pd.DataFrame:
    """Per-fold temporal CLIP before add_labels; current fold OOS untouched."""
    if set(contexts) != set(SEEN_SYMBOLS) or check_end.tzinfo is None:
        raise ValueError("Invalid four-asset fold or cutoff")
    outputs = []
    for sym in SEEN_SYMBOLS:
        # Important: NOT building labels on full features and then filtering.
        frame = contexts[sym].loc[
            contexts[sym].bar_close_at.lt(check_end)].copy()
        if frame.empty or frame.bar_close_at.ge(check_end).any():
            raise ValueError("Empty or future-contaminated fold features")
        all_h = {
            h: add_labels(frame, horizon=h, threshold_multiplier=0.5)
            for h in (*HORIZONS, MAX_H)
        }
        finite = np.isfinite(frame[FEATURE_COLUMNS].to_numpy(dtype=float)).all(axis=1)
        valid = finite
        for h in (*HORIZONS, MAX_H):
            valid &= all_h[h]["target"].notna().to_numpy()
        cols = ["symbol", "bar_close_at", *FEATURE_COLUMNS,
                "ret_1", "ret_3", "ret_6"]
        selected = frame.loc[valid, list(dict.fromkeys(cols))].copy()
        selected["label_mature_at"] = all_h[MAX_H].loc[
            valid, "label_mature_at"].to_numpy()
        for h in HORIZONS:
            selected[f"target_h{h}"] = all_h[h].loc[
                valid, "target"].to_numpy(dtype="int8")
        selected.loc[:, FEATURE_COLUMNS] = selected[FEATURE_COLUMNS].astype("float32")
        if (selected.label_mature_at.ge(check_end).any()
                or not selected.label_mature_at.eq(
                    selected.bar_close_at + MAX_H*BAR).all()):
            raise ValueError("H12 maturity extends outside allowed fold check")
        outputs.append(selected)
    result = pd.concat(outputs, ignore_index=True).sort_values(
        ["symbol", "bar_close_at"]).reset_index(drop=True)
    if result.duplicated(["symbol", "bar_close_at"]).any():
        raise ValueError("Duplicate row keys")
    return result


def parts_for_fold(shared: pd.DataFrame, fold: dict,
                   source_row: dict) -> tuple[dict[str, pd.DataFrame],dict]:
    """Exact P2 partitions and expected row counts; tune labels unused."""
    end=fold["check_end"]
    tune_start=fold["check_start"]
    stop=tune_start-pd.Timedelta(days=3)
    val=stop-pd.Timedelta(days=7)
    start=val-pd.Timedelta(days=120)
    t=shared.bar_close_at
    mature=shared.label_mature_at
    ranges={
        "train":(start,val),
        "early_stopping":(val,stop),
        "tuning":(stop,tune_start),
        "validation_check":(tune_start,end),
    }
    parts={
        name:shared.loc[
            t.ge(a)&t.lt(b)&mature.lt(b)].copy()
        for name,(a,b) in ranges.items()
    }
    expected=source_row["partition_rows"]
    for name in ("train","early_stopping","validation_check"):
        if len(parts[name])!=expected[name]:
            raise ValueError(
                f"Fold {fold['fold_id']} {name} count mismatch: "
                f"{len(parts[name])} != {expected[name]}")
    if len(parts["tuning"])!=source_row["tuning_rows"]:
        raise ValueError("Tuning rows changed from P2 reference")
    if min(map(len,parts.values()))<100:
        raise ValueError("Insufficient purged samples")
    for name,(a,b) in ranges.items():
        p=parts[name]
        if (not p.bar_close_at.ge(a).all()
                or not p.bar_close_at.lt(b).all()
                or not p.label_mature_at.lt(b).all()
                or set(p.symbol)!=set(SEEN_SYMBOLS)):
            raise ValueError("Nonpurged, nonchronological or partial partition")
    return parts,{
        "train_start":start.isoformat(),"validation_start":val.isoformat(),
        "stop_end":stop.isoformat(),"tune_end":tune_start.isoformat(),
        "check_end":end.isoformat(),
    }


def majority_direction(y_train: np.ndarray) -> int:
    y=np.asarray(y_train,dtype=int)
    if y.ndim!=1 or not len(y) or not np.isin(y,[0,1,2]).all():
        raise ValueError("Bad train targets")
    down=int((y==0).sum())
    up=int((y==2).sum())
    if min(down,up)==0:
        raise ValueError("Both directional train classes required")
    return 2 if up>down else 0  # predeclared DOWN tie fallback


def rule_predictions(check:pd.DataFrame, y_train:np.ndarray,
                     ml_proba:np.ndarray)->dict[str,np.ndarray]:
    """All decisions available at close t; no use of check targets."""
    fallback=majority_direction(y_train)
    n=len(check)
    probs=np.asarray(ml_proba,dtype=float)
    if probs.shape!=(n,3) or not np.isfinite(probs).all():
        raise ValueError("Bad ML probabilities")
    outputs={"majority_train":np.full(n,fallback,dtype=np.int8)}
    for h in (1,3,6):
        ret=check[f"ret_{h}"].to_numpy(dtype=float)
        if not np.isfinite(ret).all():
            raise ValueError("Noncausal/missing trailing return")
        mom=np.where(ret>0,2,np.where(ret<0,0,fallback)).astype("int8")
        rev=np.where(ret<0,2,np.where(ret>0,0,fallback)).astype("int8")
        outputs[f"momentum_{h}"]=mom
        outputs[f"reversion_{h}"]=rev
    # Raw conditional p(UP|movement) hard direction, matches prior stage audit.
    # Do NOT choose true movers to make this decision.
    outputs["lightgbm_30"]=np.where(
        probs[:,2]>probs[:,0],2,np.where(
            probs[:,2]<probs[:,0],0,fallback)).astype("int8")
    if set(outputs)!=set(RULES):
        raise AssertionError("Rules changed")
    return outputs


def fixed_mask_metrics(truth:np.ndarray, prediction:np.ndarray,
                       mask:np.ndarray, majority:np.ndarray)->dict|None:
    """Same samples for every rule; NEUTRAL counts as incorrect."""
    y=np.asarray(truth,dtype=int)
    pred=np.asarray(prediction,dtype=int)
    base=np.asarray(majority,dtype=int)
    subset=np.asarray(mask,dtype=bool)
    if (y.ndim!=1 or pred.shape!=y.shape or subset.shape!=y.shape or
            base.shape!=y.shape or not np.isin(y,[0,1,2]).all() or
            not np.isin(pred,[0,2]).all() or not np.isin(base,[0,2]).all()):
        raise ValueError("Invalid aligned paired comparison")
    n=int(subset.sum())
    if n==0:
        return None
    selected=y[subset]
    predicted=pred[subset]
    ref=base[subset]
    hit=predicted==selected
    ref_hit=ref==selected
    neutral=selected==1
    wrong_dir=(~hit)&(~neutral)
    return {
        "n":n,
        "correct":int(hit.sum()),
        "precision":float(hit.mean()),
        "majority_correct":int(ref_hit.sum()),
        "majority_precision_on_same_rows":float(ref_hit.mean()),
        "gain_vs_majority_points":100*float(hit.mean()-ref_hit.mean()),
        "neutral_errors":int(neutral.sum()),
        "opposite_errors":int(wrong_dir.sum()),
        "predicted_up":int((predicted==2).sum()),
        "predicted_down":int((predicted==0).sum()),
    }


def _mean(seq)->float|None:
    nonnull=[float(x) for x in seq if x is not None]
    return float(np.mean(nonnull)) if nonnull else None


def summarize(folds:list[dict])->dict:
    summary={}
    for h in HORIZONS:
        key=f"H{h}"
        summary[key]={}
        for view in ("all_check","oracle_true_movers","shared_ml_selected"):
            summary[key][view]={}
            for rule in RULES:
                metrics=[
                    f["horizons"][key]["rules"][rule][view]
                    for f in folds
                ]
                ok=[x for x in metrics if x is not None]
                summary[key][view][rule]={
                    "comparable_folds":len(ok),
                    "mean_precision":_mean(x["precision"] for x in ok),
                    "mean_gain_vs_train_majority_points":_mean(
                        x["gain_vs_majority_points"] for x in ok),
                    "folds_beating_train_majority":sum(
                        x["gain_vs_majority_points"]>0 for x in ok),
                    "total_selected_rows_across_folds":sum(x["n"] for x in ok),
                    "pooled_precision_across_folds":(
                        sum(x["correct"] for x in ok)/sum(x["n"] for x in ok)
                        if ok else None),
                }
        summary[key]["ml_mask"]={
            "feasible_folds":sum(f["horizons"][key]["ml_tuning"]["feasible"]
                                 for f in folds),
            "mean_realized_check_coverage":_mean(
                f["horizons"][key]["realized_check_coverage"] for f in folds),
        }
        summary[key]["by_asset_selected"]={}
        for sym in SEEN_SYMBOLS:
            summary[key]["by_asset_selected"][sym]={}
            for rule in RULES:
                metrics=[
                    f["horizons"][key]["by_asset_selected"][sym][rule]
                    for f in folds
                ]
                ok=[x for x in metrics if x is not None]
                summary[key]["by_asset_selected"][sym][rule]={
                    "comparable_folds":len(ok),
                    "mean_gain_vs_majority_points":_mean(
                        x["gain_vs_majority_points"] for x in ok),
                    "mean_precision":_mean(x["precision"] for x in ok),
                }
    return summary


def run_benchmark(candles:dict[str,pd.DataFrame],source:dict,*,
                  fingerprint:str,max_folds:int|None=None,
                  n_estimators:int=150,seed:int=42)->dict:
    plan=load_plan(source)
    if source.get("dataset_fingerprint_sha256")!=fingerprint:
        raise ValueError("Historical P2 dataset fingerprint differs")
    if n_estimators!=150 or seed!=42:
        raise ValueError("Fixed baseline LightGBM: 150 estimators seed 42")
    if max_folds is not None:
        if type(max_folds) is not int or not 1<=max_folds<=len(plan):
            raise ValueError("Bad smoke fold count")
        plan=plan[:max_folds]
    if set(candles)!=set(SEEN_SYMBOLS):
        raise ValueError("Only BTC, ETH, SOL and XRP are authorized")
    cutoff=plan[-1]["check_end"]
    contexts=build_causal_contexts(candles,cutoff)
    results=[]
    for fold in plan:
        shared=prepare_fold_examples(contexts,fold["check_end"])
        src=source["folds"][fold["fold_id"]]
        parts,bounds=parts_for_fold(shared,fold,src)
        train=parts["train"]
        early=parts["early_stopping"]
        tune=parts["tuning"]
        check=parts["validation_check"]
        horizons={}
        for horizon in HORIZONS:
            key=f"H{horizon}"
            target=f"target_h{horizon}"
            train_y=train[target].to_numpy(dtype=int)
            early_y=early[target].to_numpy(dtype=int)
            check_y=check[target].to_numpy(dtype=int)
            counts=np.bincount(train_y,minlength=3)
            if (counts==0).any() or set(early_y.tolist())!={0,1,2}:
                raise ValueError("Missing training/early-stop classes")
            model=_cpu_model(binary=False,n_estimators=n_estimators,seed=seed)
            model.fit(
                feature_matrix(train),train_y,
                eval_set=[(feature_matrix(early),early_y)],
                eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(stopping_rounds=30,verbose=False)],
            )
            probs_t=_class_probs(model,feature_matrix(tune))
            score_t,margin_t=decision_and_margin(probs_t,counts,alpha=1.0)
            fitted=threshold_for_target(
                score_t,margin_t,target_coverage=TARGET_COVERAGE)
            probs_c=_class_probs(model,feature_matrix(check))
            score_c,margin_c=decision_and_margin(probs_c,counts,alpha=1.0)
            selected=(score_c!=1)&(margin_c>=fitted["threshold"]) if fitted[
                "feasible"] else np.zeros(len(check),dtype=bool)
            actual_coverage=float(selected.mean()) if fitted["feasible"] else None
            decisions=rule_predictions(check,train_y,probs_c)
            majority=decisions["majority_train"]
            full_mask=np.ones(len(check),dtype=bool)
            oracle_mask=check_y!=1  # used ONLY for oracle evaluation
            rules={}
            for name,pred in decisions.items():
                rules[name]={
                    "all_check":fixed_mask_metrics(
                        check_y,pred,full_mask,majority),
                    "oracle_true_movers":fixed_mask_metrics(
                        check_y,pred,oracle_mask,majority),
                    "shared_ml_selected":fixed_mask_metrics(
                        check_y,pred,selected,majority)
                        if fitted["feasible"] else None,
                }
            by_asset={}
            symbols=check.symbol.to_numpy()
            for sym in SEEN_SYMBOLS:
                mask=(symbols==sym)&selected
                by_asset[sym]={
                    name:fixed_mask_metrics(check_y,pred,mask,majority)
                          if fitted["feasible"] else None
                    for name,pred in decisions.items()
                }
            # Ensure identical number of evaluations for every rule.
            for view in ("all_check","oracle_true_movers","shared_ml_selected"):
                nums={v[view]["n"] for v in rules.values() if v[view] is not None}
                if len(nums)>1:
                    raise AssertionError("Rules compared on unequal observations")
            horizons[key]={
                "train_counts":{
                    name:int(counts[i])
                    for i,name in enumerate(("DOWN","NEUTRAL","UP"))
                },
                "majority_direction":(
                    "UP" if majority[0]==2 else "DOWN"),
                "best_iteration":int(model.best_iteration_),
                "ml_tuning":fitted,
                "realized_check_coverage":actual_coverage,
                "rules":rules,
                "by_asset_selected":by_asset,
                "warning":"True-mover oracle is retrospective; ML-shared mask "
                          "requires an ML selector at inference.",
            }
            highlight=rules["lightgbm_30"]["shared_ml_selected"]
            baseline=rules["majority_train"]["shared_ml_selected"]
            print(
                f"fold={fold['fold_id']} {key} "
                f"ml_feasible={fitted['feasible']} "
                f"coverage={actual_coverage} "
                f"selected_ML_precision={None if highlight is None else highlight['precision']} "
                f"selected_majority_precision={None if baseline is None else baseline['precision']}",
                flush=True)
        results.append({
            "fold_id":fold["fold_id"],"bounds":bounds,
            "partition_rows":{k:len(v) for k,v in parts.items()},
            "horizons":horizons,
        })
        del shared,parts
    return {
        "protocol":PROTOCOL,
        "research_status":"EXPLORATORY_REPEATEDLY_INSPECTED_DEVELOPMENT_CHECKS",
        "source_dataset_fingerprint_sha256":fingerprint,
        "folds_analyzed":len(results),
        "last_dev_check_end_exclusive_utc":cutoff.isoformat(),
        "horizons":list(HORIZONS),
        "features":list(FEATURE_COLUMNS),
        "feature_schema":FEATURE_SCHEMA_VERSION,
        "label_schema":LABEL_SCHEMA_VERSION,
        "label_multiplier":0.5,
        "model":"unchanged LightGBM 30-feature hyperparameters",
        "n_estimators":n_estimators,
        "seed":seed,
        "rules":list(RULES),
        "ml_tuning_coverage_target":TARGET_COVERAGE,
        "no_check_label_based_threshold_selection":True,
        "current_fold_labels_clipped_before_OOS":True,
        "ADA_loaded_or_scored":False,
        "fold_OOS_test_labels_accessed":False,
        "final90_test_labels_accessed":False,
        "P3_or_forward_models_modified":False,
        "limitations":[
            "Previously examined checks reused; folds overlap in training, not iid.",
            "Each fold's labels are constructed only before its check end, never in its fold OOS test.",
            "Oracle true-mover subset is not observable at inference.",
            "The shared 40% selection mask is generated by the ML model from "
            "3d prior tuning scores alone; simple rules are NOT independent "
            "40%-coverage standalone strategies.",
            "The all-check sign comparison forces a direction each bar; no abstention.",
            "Baseline majority learned from pooled train labels, with DOWN tie rule.",
            "No threshold/feature/label optimization, probabilities for hard rules, "
            "P&L, cost-adjusted detection or model promotion.",
        ],
        "summary":summarize(results),
        "folds":results,
    }


def main()->None:
    p=argparse.ArgumentParser(
        description="P2 H2/H6 directional majority momentum reversion versus LightGBM")
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--fold-plan",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--months",type=int,default=24)
    p.add_argument("--max-folds",type=int)
    p.add_argument("--n-estimators",type=int,default=150)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--verify-hashes",action="store_true")
    a=p.parse_args()
    if a.months!=24 or a.n_estimators!=150 or a.seed!=42:
        p.error("Frozen research protocol: 24 months, 150 estimators, seed=42")
    if a.output.exists():
        p.error("Report exists; refusing overwrite")
    if a.data_root.resolve()==a.output.resolve() or a.data_root.resolve() in a.output.resolve().parents:
        p.error("Cannot write under source P0 data root")
    source_bytes=a.fold_plan.read_bytes()
    source=json.loads(source_bytes)
    load_plan(source)  # validate sealed-period guards BEFORE raw candle loading
    store=CandleStore(a.data_root)
    candles,manifests={},[]
    for symbol in SEEN_SYMBOLS:
        print("Reading seen asset",symbol,flush=True)
        candles[symbol],pieces=load_symbol(
            store,symbol,months=a.months,verify_hashes=a.verify_hashes)
        manifests.extend(pieces)
    report=run_benchmark(
        candles,source,fingerprint=manifest_fingerprint(manifests),
        max_folds=a.max_folds,n_estimators=a.n_estimators,seed=a.seed)
    report["fold_plan_sha256"]=hashlib.sha256(source_bytes).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x",encoding="utf-8") as fd:
        json.dump(report,fd,indent=2,ensure_ascii=False,allow_nan=False)
    print("Saved P2 directional benchmark:",a.output.resolve(),flush=True)
    for h in HORIZONS:
        data=report["summary"][f"H{h}"]
        print(f"H{h}: selected feasible {data['ml_mask']['feasible_folds']}/"
              f"{report['folds_analyzed']}, check coverage="
              f"{data['ml_mask']['mean_realized_check_coverage']}",flush=True)
        for name in RULES:
            stats=data["shared_ml_selected"][name]
            print(f"  {name}: paired_folds={stats['comparable_folds']} "
                  f"precision={stats['mean_precision']} "
                  f"delta_vs_majority_pp={stats['mean_gain_vs_train_majority_points']}",
                  flush=True)


if __name__=="__main__":
    main()
