"""Exploratory H1..H6 EMA9/26 vs 30-feature CPU experiment; old DEV checks only."""
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
from mlpipeline.directional_disagreement_audit import verify_baseline_report
from mlpipeline.directional_rule_benchmark import (
    build_causal_contexts, parts_for_fold, prepare_fold_examples,
    fixed_mask_metrics, majority_direction,
)
from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS
from mlpipeline.features import FEATURE_COLUMNS, FIVE_MINUTES
from mlpipeline.label_stability_audit import load_plan
from mlpipeline.labels import add_labels
from mlpipeline.trainer import _class_probs
from mlpipeline.two_stage_diagnostics import _cpu_model

PROTOCOL="p2_h1_h6_ema9_26_three_baselines_v1"
HORIZONS=tuple(range(1,7))
EMA_COLUMNS=("ema9_price_gap","ema9_ema26_gap")
MODELS=("lightgbm_30","lightgbm_ema2","ema_cross_rule")
COVERAGE=0.40


def ema_features(candles:pd.DataFrame)->pd.DataFrame:
    """EMA with adjust=False and reset on M5 gap; known only at close t."""
    if candles.empty or candles.symbol.nunique()!=1:
        raise ValueError("One nonempty asset expected")
    bars=candles.sort_values("bar_open_at").reset_index(drop=True)
    if (bars.bar_open_at.duplicated().any()
            or str(bars.bar_open_at.dt.tz)!="UTC"
            or str(bars.bar_close_at.dt.tz)!="UTC"
            or not (bars.bar_close_at-bars.bar_open_at).eq(FIVE_MINUTES).all()
            or not np.isfinite(bars.close.to_numpy(dtype=float)).all()
            or (bars.close<=0).any()):
        raise ValueError("Invalid closed UTC M5 data")
    segment=bars.bar_open_at.diff().ne(FIVE_MINUTES).cumsum()
    ema9=bars.groupby(segment,sort=False)["close"].transform(
        lambda s:s.ewm(span=9,adjust=False,min_periods=9).mean())
    ema26=bars.groupby(segment,sort=False)["close"].transform(
        lambda s:s.ewm(span=26,adjust=False,min_periods=26).mean())
    out=bars[["symbol","bar_close_at"]].copy()
    out[EMA_COLUMNS[0]]=(bars.close/ema9-1).astype("float32")
    out[EMA_COLUMNS[1]]=(ema9/ema26-1).astype("float32")
    return out


def build_inputs(candles:dict,cutoff:pd.Timestamp)->tuple[dict,pd.DataFrame]:
    contexts=build_causal_contexts(candles,cutoff)
    em=[ema_features(candles[s].loc[
        candles[s].bar_close_at.lt(cutoff)].copy()) for s in SEEN_SYMBOLS]
    features=pd.concat(em,ignore_index=True)
    if features.duplicated(["symbol","bar_close_at"]).any():
        raise ValueError("Duplicate EMA inputs")
    return contexts,features


def fold_examples(contexts:dict,ema:pd.DataFrame,cutoff:pd.Timestamp)->pd.DataFrame:
    """Reuse H12-purged old exact row keys and construct H1..H6 after clipping."""
    shared=prepare_fold_examples(contexts,cutoff)
    originals=shared[["symbol","bar_close_at"]].copy()
    shared=shared.merge(ema,on=["symbol","bar_close_at"],how="left",
                        sort=False,validate="one_to_one")
    if (len(shared)!=len(originals)
            or not shared[["symbol","bar_close_at"]].equals(originals)
            or not np.isfinite(shared[list(EMA_COLUMNS)].to_numpy(dtype=float)).all()):
        raise ValueError("EMA warmup or join changed original P2 sample rows")
    lookup=[]
    for sym in SEEN_SYMBOLS:
        frame=contexts[sym].loc[contexts[sym].bar_close_at.lt(cutoff)].copy()
        cols=["symbol","bar_close_at"]
        built=frame[cols].copy()
        for h in HORIZONS:
            lbl=add_labels(frame,horizon=h,threshold_multiplier=0.5)
            built[f"target_h{h}"]=lbl["target"].to_numpy()
            if h==1:
                built["future_return_h1"]=lbl["future_return"].to_numpy(dtype=float)
        lookup.append(built)
    truth=pd.concat(lookup,ignore_index=True)
    shared=shared.merge(truth,on=["symbol","bar_close_at"],
                        how="left",sort=False,validate="one_to_one",
                        suffixes=("","_rebuilt"))
    if len(shared)!=len(originals) or not shared[["symbol","bar_close_at"]].equals(originals):
        raise ValueError("Target merge changed sample keys")
    for h in HORIZONS:
        rebuilt=shared[f"target_h{h}_rebuilt"] if h in (2,6) else shared[f"target_h{h}"]
        if rebuilt.isna().any():
            raise ValueError("Horizon target missing from matched H12 rows")
        if h in (2,6):
            if not np.array_equal(rebuilt.to_numpy(dtype=int),
                                  shared[f"target_h{h}"].to_numpy(dtype=int)):
                raise ValueError("Existing H2/H6 labels differ from recomputed labels")
            shared=shared.drop(columns=[f"target_h{h}_rebuilt"])
        else:
            shared[f"target_h{h}"]=rebuilt.astype("int8")
    if not np.isfinite(shared.future_return_h1.to_numpy(dtype=float)).all():
        raise ValueError("Missing next bar for exact H1 sign")
    return shared


def sign(probs:np.ndarray,majority:int)->np.ndarray:
    p=np.asarray(probs,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():
        raise ValueError("Invalid class probabilities")
    return np.where(p[:,2]>p[:,0],2,
                    np.where(p[:,2]<p[:,0],0,majority)).astype("int8")


def rule_sign(gap:np.ndarray,majority:int)->np.ndarray:
    g=np.asarray(gap,dtype=float)
    if g.ndim!=1 or not np.isfinite(g).all():
        raise ValueError("Invalid EMA gap")
    return np.where(g>0,2,np.where(g<0,0,majority)).astype("int8")


def rule_mask(tune_gap:np.ndarray,check_gap:np.ndarray)->tuple[np.ndarray,dict]:
    """Absolute EMA separation, 40% selection from unlabeled prior tuning."""
    t=np.abs(np.asarray(tune_gap,dtype=float))
    c=np.abs(np.asarray(check_gap,dtype=float))
    if (t.ndim!=1 or c.ndim!=1 or not len(t)
            or not np.isfinite(t).all() or not np.isfinite(c).all()):
        raise ValueError("Nonfinite EMA rule selector")
    cutoff=float(np.sort(t)[::-1][min(len(t)-1,max(0,round(.4*len(t))-1))])
    # Conservative: ties at boundary excluded, no peeking at check truth.
    mask=c>cutoff
    return mask,{"threshold_abs_gap":cutoff,
                 "tuning_fraction_strict":float((t>cutoff).mean()),
                 "target_tuning_fraction":.4}


def score_next_bar(raw_return:np.ndarray,pred:np.ndarray,mask:np.ndarray)->dict|None:
    ret=np.asarray(raw_return,dtype=float)
    side=np.asarray(pred,dtype=int)
    selected=np.asarray(mask,dtype=bool)
    if (ret.ndim!=1 or side.shape!=ret.shape or selected.shape!=ret.shape
            or not np.isfinite(ret).all() or not np.isin(side,[0,2]).all()):
        raise ValueError("Bad next-bar sign inputs")
    chosen=ret[selected]
    sign_pred=side[selected]
    if not len(chosen):
        return None
    flat=chosen==0
    actual=np.where(chosen>0,2,0)
    n=int((~flat).sum())
    return {"selected":int(len(chosen)),"flat":int(flat.sum()),
            "nonflat":n,"correct_nonflat":int((sign_pred[~flat]==actual[~flat]).sum()),
            "sign_accuracy":float((sign_pred[~flat]==actual[~flat]).mean())
                            if n else None}


def summary(folds:list[dict])->dict:
    output={}
    for h in HORIZONS:
        hk=f"H{h}"
        results=[f["horizons"][hk] for f in folds]
        pooled={}
        for view in ("all_check","shared_baseline_selected","ema2_autonomous",
                     "ema_rule_autonomous"):
            models=MODELS if view in ("all_check","shared_baseline_selected") else (
                ("lightgbm_ema2",) if view=="ema2_autonomous" else ("ema_cross_rule",))
            pooled[view]={}
            for name in models:
                valid=[r["metrics"][view][name] for r in results
                       if r["metrics"][view].get(name) is not None]
                pooled[view][name]={
                    "folds":len(valid),
                    "mean_precision":float(np.mean([v["precision"] for v in valid]))
                                     if valid else None,
                    "selected_rows":sum(v["n"] for v in valid),
                    "pooled_precision":sum(v["correct"] for v in valid)/
                        sum(v["n"] for v in valid) if valid else None,
                }
        deltas=[
            100*(r["metrics"]["shared_baseline_selected"]["lightgbm_ema2"]["precision"]-
                 r["metrics"]["shared_baseline_selected"]["lightgbm_30"]["precision"])
            for r in results
            if r["metrics"]["shared_baseline_selected"]["lightgbm_30"] is not None
        ]
        output[hk]={
            "minutes_ahead":h*5,"views":pooled,
            "mean_delta_ema2_minus_30_shared_points":float(np.mean(deltas)) if deltas else None,
            "folds_ema2_wins":sum(v>0 for v in deltas),
            "folds_30_wins":sum(v<0 for v in deltas),
            "folds_tied":sum(v==0 for v in deltas),
            "mean_logloss_30":float(np.mean([r["logloss"]["lightgbm_30"] for r in results])),
            "mean_logloss_ema2":float(np.mean([r["logloss"]["lightgbm_ema2"] for r in results])),
        }
        if h==1:
            output[hk]["next_bar_sign"]={}
            for name in MODELS:
                cells=[r["h1_next_bar_sign"][name] for r in results]
                hits=sum(x["correct_nonflat"] for x in cells)
                count=sum(x["nonflat"] for x in cells)
                output[hk]["next_bar_sign"][name]={
                    "nonflat":count,"flat":sum(x["flat"] for x in cells),
                    "pooled_sign_accuracy":hits/count if count else None,
                }
    return output


def run(candles:dict,plan_source:dict,old_report:dict,*,
        fingerprint:str,max_folds:int|None=None)->tuple[dict,pd.DataFrame]:
    plan=load_plan(plan_source)
    old=verify_baseline_report(old_report,plan_source)
    if fingerprint!=plan_source["dataset_fingerprint_sha256"]:
        raise ValueError("P0 data fingerprint mismatch")
    if set(candles)!=set(SEEN_SYMBOLS):
        raise ValueError("Only four seen assets allowed")
    if max_folds is not None:
        if type(max_folds)!=int or not 1<=max_folds<=len(plan):
            raise ValueError("Bad max-folds")
        plan=plan[:max_folds]
    cutoff=plan[-1]["check_end"]
    contexts,ema=build_inputs(candles,cutoff)
    results=[]
    predictions=[]
    for fold in plan:
        shared=fold_examples(contexts,ema,fold["check_end"])
        parts,bounds=parts_for_fold(shared,fold,
                                    plan_source["folds"][fold["fold_id"]])
        train,early,tune,check=(parts[k] for k in (
            "train","early_stopping","tuning","validation_check"))
        horizons={}
        n=len(check)
        symbols=check.symbol.to_numpy()
        stamps=check.bar_close_at.to_numpy()
        for h in HORIZONS:
            hk=f"H{h}"
            key=f"target_h{h}"
            yt=train[key].to_numpy(dtype=int)
            ye=early[key].to_numpy(dtype=int)
            truth=check[key].to_numpy(dtype=int)
            counts=np.bincount(yt,minlength=3)
            if (np.any(counts==0) or set(ye.tolist())!={0,1,2}):
                raise ValueError("Missing class on train or early")
            majority=majority_direction(yt)
            probs={}
            iterations={}
            for name,cols in (("lightgbm_30",FEATURE_COLUMNS),
                              ("lightgbm_ema2",EMA_COLUMNS)):
                clf=_cpu_model(binary=False,n_estimators=150,seed=42)
                clf.fit(train[list(cols)].astype("float32"),yt,
                    eval_set=[(early[list(cols)].astype("float32"),ye)],
                    eval_metric="multi_logloss",
                    callbacks=[lgb.early_stopping(stopping_rounds=30,verbose=False)])
                iterations[name]=int(clf.best_iteration_)
                probs[name]={
                    "tune":_class_probs(clf,tune[list(cols)].astype("float32")),
                    "check":_class_probs(clf,check[list(cols)].astype("float32"))}
            pred_t,margin_t=decision_and_margin(probs["lightgbm_30"]["tune"],counts,alpha=1.)
            baseline_threshold=threshold_for_target(pred_t,margin_t,target_coverage=COVERAGE)
            pred_c,margin_c=decision_and_margin(probs["lightgbm_30"]["check"],counts,alpha=1.)
            baseline_mask=((pred_c!=1)&(margin_c>=baseline_threshold["threshold"])
                           if baseline_threshold["feasible"]
                           else np.zeros(n,dtype=bool))
            pred_t2,margin_t2=decision_and_margin(probs["lightgbm_ema2"]["tune"],counts,alpha=1.)
            ema_threshold=threshold_for_target(pred_t2,margin_t2,target_coverage=COVERAGE)
            pred_c2,margin_c2=decision_and_margin(probs["lightgbm_ema2"]["check"],counts,alpha=1.)
            ema_mask=((pred_c2!=1)&(margin_c2>=ema_threshold["threshold"])
                      if ema_threshold["feasible"] else np.zeros(n,dtype=bool))
            rulemask,rule_settings=rule_mask(
                tune[EMA_COLUMNS[1]].to_numpy(dtype=float),
                check[EMA_COLUMNS[1]].to_numpy(dtype=float))
            directional={k:sign(p["check"],majority) for k,p in probs.items()}
            directional["ema_cross_rule"]=rule_sign(
                check[EMA_COLUMNS[1]].to_numpy(dtype=float),majority)
            ref=np.full(n,majority,dtype="int8")
            allmask=np.ones(n,dtype=bool)
            masks={"all_check":allmask,
                   "shared_baseline_selected":baseline_mask,
                   "ema2_autonomous":ema_mask,
                   "ema_rule_autonomous":rulemask}
            metrics={}
            for view,mask in masks.items():
                names=MODELS if view in ("all_check","shared_baseline_selected") else (
                    ("lightgbm_ema2",) if view=="ema2_autonomous" else ("ema_cross_rule",))
                metrics[view]={
                    name:(fixed_mask_metrics(truth,directional[name],mask,ref)
                          if mask.any() and (view!="shared_baseline_selected"
                                              or baseline_threshold["feasible"]) else None)
                    for name in names}
            if h in (2,6):
                previous=old[fold["fold_id"]]["horizons"][hk]
                if (iterations["lightgbm_30"]!=previous["best_iteration"]
                      or bool(baseline_threshold["feasible"])!=bool(previous["ml_tuning"]["feasible"])):
                    raise ValueError("Prior H2/H6 fit/mask feasibility mismatch")
                for view,oldview in (("all_check","all_check"),
                                     ("shared_baseline_selected","shared_ml_selected")):
                    observed=metrics[view]["lightgbm_30"]
                    expected=previous["rules"]["lightgbm_30"][oldview]
                    if (observed is None)!=(expected is None):
                        raise ValueError("Prior baseline mismatch")
                    if observed is not None and any(
                            observed[k]!=expected[k] for k in (
                                "n","correct","neutral_errors","opposite_errors",
                                "predicted_up","predicted_down")):
                        raise ValueError("H2/H6 baseline reproduction drift")
            ll={name:float(log_loss(truth,prob["check"],labels=[0,1,2]))
                for name,prob in probs.items()}
            row={"mask_tuning":{"baseline":baseline_threshold,
                                "ema2":ema_threshold,"rule":rule_settings},
                 "realized_coverage":{k:float(v.mean()) for k,v in masks.items()},
                 "metrics":metrics,"logloss":ll,
                 "best_iteration":iterations,
                 "observed_class_counts":dict(zip(("DOWN","NEUTRAL","UP"),
                                      map(int,np.bincount(truth,minlength=3))))}
            if h==1:
                row["h1_next_bar_sign"]={
                    name:score_next_bar(check.future_return_h1.to_numpy(dtype=float),
                                        direction,allmask)
                    for name,direction in directional.items()}
            horizons[hk]=row
            data={"fold_id":np.full(n,fold["fold_id"],dtype="int16"),
                  "symbol":symbols,"bar_close_at":stamps,
                  "horizon":np.full(n,h,dtype="int8"),
                  "target_3class":truth.astype("int8"),
                  "baseline_shared_mask":baseline_mask,
                  "ema2_autonomous_mask":ema_mask,
                  "ema_rule_autonomous_mask":rulemask}
            if h==1:
                data["next_bar_return"]=check.future_return_h1.to_numpy(dtype=float)
            else:
                data["next_bar_return"]=np.full(n,np.nan)
            for name in MODELS:
                data[f"{name}_sign"]=directional[name]
                if name in probs:
                    for i,cls in enumerate(("down","neutral","up")):
                        data[f"{name}_p_{cls}"]=probs[name]["check"][:,i]
            predictions.append(pd.DataFrame(data))
            print(f"fold={fold['fold_id']} H{h} baseline_feasible={baseline_threshold['feasible']} "
                  f"n_shared={int(baseline_mask.sum())} "
                  f"all_precision_30={metrics['all_check']['lightgbm_30']['precision']:.5f}",
                  flush=True)
        results.append({"fold_id":fold["fold_id"],"bounds":bounds,
                        "partition_rows":{k:len(v) for k,v in parts.items()},
                        "horizons":horizons})
    pred=pd.concat(predictions,ignore_index=True)
    if pred.duplicated(["fold_id","symbol","bar_close_at","horizon"]).any():
        raise ValueError("Duplicate prediction keys")
    report={"protocol":PROTOCOL,"status":"EXPLORATORY_REUSED_DEV_CHECKS",
            "folds_analyzed":len(results),"source_dataset_fingerprint_sha256":fingerprint,
            "horizons":list(HORIZONS),"model_columns":{
                "lightgbm_30":list(FEATURE_COLUMNS),
                "lightgbm_ema2":list(EMA_COLUMNS)},
            "models":list(MODELS),"base_comparison_H2_H6_reproduced":True,
            "max_horizon_label_maturity_purge":12,
            "per_fold_clipping_before_label_construction":True,
            "H1_raw_sign_flat_excluded_from_accuracy":True,
            "no_ADA_no_OOS_no_final90_no_P3_modification":True,
            "note":"H2..H6 labels are cumulative returns from current close, not each future candle color.",
            "summary":summary(results),"folds":results}
    return report,pred


def main()->None:
    parser=argparse.ArgumentParser(description="P2 H1..H6 EMA9/26 vs 30 baseline")
    for flag in ("data-root","fold-plan","baseline-report","output-json","output-parquet"):
        parser.add_argument("--"+flag,required=True,type=Path)
    parser.add_argument("--months",type=int,default=24)
    parser.add_argument("--max-folds",type=int)
    parser.add_argument("--verify-hashes",action="store_true")
    a=parser.parse_args()
    if a.months!=24:
        parser.error("Fixed 24 months only")
    outputs=(a.output_json.resolve(),a.output_parquet.resolve())
    if outputs[0]==outputs[1] or any(p.exists() for p in outputs):
        parser.error("Outputs must be distinct nonexisting paths")
    protected={a.fold_plan.resolve(),a.baseline_report.resolve()}
    if any(p in protected for p in outputs):
        parser.error("Outputs must not overwrite input reports")
    if any(p.is_relative_to(a.data_root.resolve().parent)
           for p in outputs):
        parser.error("Outputs must remain outside production data/repo")
    raw_source=a.fold_plan.read_bytes()
    raw_baseline=a.baseline_report.read_bytes()
    source=json.loads(raw_source)
    baseline=json.loads(raw_baseline)
    load_plan(source)
    verify_baseline_report(baseline,source)
    store=CandleStore(a.data_root)
    candles,manifests={},[]
    for symbol in SEEN_SYMBOLS:
        print("Loading read-only P0:",symbol,flush=True)
        candles[symbol],files=load_symbol(
            store,symbol,months=a.months,verify_hashes=a.verify_hashes)
        manifests.extend(files)
    report,preds=run(candles,source,baseline,
                     fingerprint=manifest_fingerprint(manifests),
                     max_folds=a.max_folds)
    report["fold_plan_sha256"]=hashlib.sha256(raw_source).hexdigest()
    report["baseline_report_sha256"]=hashlib.sha256(raw_baseline).hexdigest()
    report["lightgbm_version"]=lgb.__version__
    report["predictions_rows"]=len(preds)
    for path in outputs:
        path.parent.mkdir(parents=True,exist_ok=True)
    # Parquet first; JSON is the final completion marker.
    preds.to_parquet(a.output_parquet,index=False)
    with a.output_json.open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print("Saved detailed predictions:",a.output_parquet,flush=True)
    print("Saved summary:",a.output_json,flush=True)
    for h in HORIZONS:
        item=report["summary"][f"H{h}"]
        print(f"H{h}: shared delta EMA2-30={item['mean_delta_ema2_minus_30_shared_points']}",
              flush=True)
    print("SMOKE ONLY" if a.max_folds else "EXPLORATORY CHECKS ONLY",flush=True)


if __name__=="__main__":
    main()
