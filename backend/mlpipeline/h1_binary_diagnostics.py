"""Direct binary next-M5-close sign vs frozen H1 3-class predictions.

Research ONLY, historically reused P2 development checks. Never read ADA,
fold OOS/final90, or edit frozen P3. No online observation, trading or PnL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ingestion.storage import CandleStore
from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.directional_rule_benchmark import parts_for_fold
from mlpipeline.ema_horizon_grid import EMA_COLUMNS, build_inputs, fold_examples
from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS
from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.h1_individual_feature_study import align_check, load_reference
from mlpipeline.label_stability_audit import load_plan
from mlpipeline.two_stage_diagnostics import _cpu_model

PROTOCOL="p2_h1_direct_binary_vs_frozen_ternary_v1"
VARIANTS={
    "binary_ema2":list(EMA_COLUMNS),
    "binary_ema2_close":[*EMA_COLUMNS,"close_location"],
    "binary_30":list(FEATURE_COLUMNS),
    "logistic_ema2":list(EMA_COLUMNS),
}
REFERENCES=("ternary_ema2","ternary_30","always_down","last_reversion")
ALL_METHODS=(*VARIANTS,*REFERENCES)
SCOPES=("all_nonflat","shared30_nonflat","ema2_selected_nonflat")
BINS=((0.0,0.01),(0.01,0.025),(0.025,0.05),(0.05,0.5000001))


def binary_labels(raw_return:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    """Truth known ONLY in purged train/early or already inspected DEV check."""
    r=np.asarray(raw_return,dtype=float)
    if r.ndim!=1 or not np.isfinite(r).all():
        raise ValueError("H1 returns must be one-dimensional and finite")
    return (r>0).astype("int8"),r!=0


def purged_binary_training(part:pd.DataFrame,columns:list[str])->tuple[pd.DataFrame,np.ndarray,int]:
    y,valid=binary_labels(part.future_return_h1.to_numpy(dtype=float))
    x=part.loc[valid,columns].astype("float32")
    if x.empty or not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Empty/nonfinite binary training features")
    if set(np.unique(y[valid]).tolist())!={0,1}:
        raise ValueError("Missing UP/DOWN training class")
    return x,y[valid],int((~valid).sum())


def sign_from_up_probability(prob:np.ndarray)->np.ndarray:
    p=np.asarray(prob,dtype=float)
    if p.ndim!=1 or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError("Invalid P(UP)")
    return np.where(p>0.5,2,0).astype("int8")


def train_binary(train:pd.DataFrame,early:pd.DataFrame,
                 *,name:str)->tuple[object,dict]:
    if name not in VARIANTS:
        raise ValueError("Unknown frozen binary variant")
    columns=VARIANTS[name]
    tx,ty,train_flat=purged_binary_training(train,columns)
    ex,ey,early_flat=purged_binary_training(early,columns)
    if name=="logistic_ema2":
        model=make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0,max_iter=1000,solver="lbfgs",random_state=42))
        model.fit(tx,ty)
        iterations=None
    else:
        model=_cpu_model(binary=True,n_estimators=150,seed=42)
        model.fit(tx,ty,eval_set=[(ex,ey)],
                  eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(stopping_rounds=30,verbose=False)])
        iterations=int(model.best_iteration_)
    return model,{"train_nonflat":len(tx),"train_flats_excluded":train_flat,
                  "early_nonflat":len(ex),"early_flats_excluded":early_flat,
                  "best_iteration":iterations}


def probability_up(model:object,check:pd.DataFrame,name:str)->np.ndarray:
    x=check[VARIANTS[name]].astype("float32")
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite check input")
    classes=np.asarray(model.classes_ if name!="logistic_ema2" else
                       model.named_steps["logisticregression"].classes_,dtype=int)
    if not np.array_equal(classes,[0,1]):
        raise ValueError("Unexpected binary model class ordering")
    p=np.asarray(model.predict_proba(x),dtype=float)
    if p.shape!=(len(check),2):
        raise ValueError("Invalid binary probabilities")
    p_up=p[:,1]
    sign_from_up_probability(p_up)
    return p_up


def prediction_frame(check:pd.DataFrame,fitted:dict[str,object])->pd.DataFrame:
    """Emit forecasts on all rows; no oracle filtering of prospective signals."""
    y,nonflat=binary_labels(check.future_return_h1.to_numpy(dtype=float))
    out=check[["symbol","bar_close_at","target_h1","next_bar_return",
               "baseline_shared_mask","ema2_autonomous_mask"]].copy()
    out["h1_binary_target"]=np.where(nonflat,y,-1).astype("int8")
    out["ternary_ema2_sign"]=check.lightgbm_ema2_sign.astype("int8").to_numpy()
    out["ternary_30_sign"]=check.lightgbm_30_sign.astype("int8").to_numpy()
    out["always_down_sign"]=np.zeros(len(check),dtype="int8")
    # All inputs known at close t; ret_1==0 deterministically falls back DOWN.
    out["last_reversion_sign"]=np.where(
        check.ret_1.to_numpy(dtype=float)<0,2,0).astype("int8")
    for name,model in fitted.items():
        p=probability_up(model,check,name)
        out[f"{name}_p_up"]=p
        out[f"{name}_sign"]=sign_from_up_probability(p)
    if tuple(fitted)!=tuple(VARIANTS):
        raise ValueError("All 4 fixed candidates are required")
    return out


def paired_counts(actual:np.ndarray,a:np.ndarray,b:np.ndarray)->dict:
    """All same nonflat rows; paired correct/incorrect are auditable."""
    if actual.shape!=a.shape or actual.shape!=b.shape:
        raise ValueError("Paired sign arrays differ")
    ca=a==actual
    cb=b==actual
    return {
        "n":len(actual),
        "candidate_correct":int(ca.sum()),
        "reference_correct":int(cb.sum()),
        "delta_points":100*float(ca.mean()-cb.mean()) if len(actual) else None,
        "candidate_only_correct":int((ca&~cb).sum()),
        "reference_only_correct":int((~ca&cb).sum()),
    }


def scope_metrics(pred:pd.DataFrame,mask:np.ndarray)->dict|None:
    """Mask is fixed (old reference mask), plus common future-nonflat filter."""
    if mask.dtype!=bool or mask.shape!=(len(pred),):
        raise ValueError("Bad metrics mask")
    rows=pred.loc[mask]
    if rows.empty:
        return None
    actual=np.where(rows.h1_binary_target.to_numpy(dtype=int)==1,2,0)
    if not rows.h1_binary_target.isin([0,1]).all():
        raise ValueError("Flat bars must never enter directional metric")
    scores={}
    for name in ALL_METHODS:
        s=rows[f"{name}_sign"].to_numpy(dtype=int)
        if not np.isin(s,[0,2]).all():
            raise ValueError("Invalid direction: "+name)
        scores[name]={"correct":int((s==actual).sum()),
                      "accuracy":float((s==actual).mean())}
    pairs={}
    for name in VARIANTS:
        a=rows[f"{name}_sign"].to_numpy(dtype=int)
        for reference in REFERENCES:
            b=rows[f"{reference}_sign"].to_numpy(dtype=int)
            pairs[f"{name}_minus_{reference}"]=paired_counts(actual,a,b)
    diagnostic={}
    true=rows.h1_binary_target.to_numpy(dtype=int)
    for name in VARIANTS:
        p=rows[f"{name}_p_up"].to_numpy(dtype=float)
        margin=np.abs(p-0.5)
        bucket=[]
        for low,high in BINS:
            chosen=(margin>=low)&(margin<high)
            bucket.append({
                "margin_from_0_5":f"[{low:.3f},{high:.3f})",
                "n":int(chosen.sum()),
                "accuracy":float((sign_from_up_probability(p[chosen])==
                                  actual[chosen]).mean()) if chosen.any() else None,
            })
        diagnostic[name]={
            "logloss_nonflat":float(log_loss(true,p,labels=[0,1])),
            "brier_nonflat":float(np.mean((p-true)**2)),
            "auc_nonflat":float(roc_auc_score(true,p))
                           if len(np.unique(true))==2 else None,
            "fixed_confidence_bins":bucket,
        }
    return {"n":len(rows),"real_up":int((actual==2).sum()),
            "real_down":int((actual==0).sum()),"scores":scores,
            "paired":pairs,"binary_probabilities":diagnostic}


def fold_metrics(pred:pd.DataFrame)->dict:
    if pred.empty or not pred.h1_binary_target.isin([-1,0,1]).all():
        raise ValueError("Bad H1 check frame")
    nonflat=pred.h1_binary_target.ne(-1).to_numpy()
    masks={
        "all_nonflat":nonflat,
        "shared30_nonflat":nonflat&pred.baseline_shared_mask.to_numpy(dtype=bool),
        "ema2_selected_nonflat":nonflat&pred.ema2_autonomous_mask.to_numpy(dtype=bool),
    }
    return {
        "rows_all":len(pred),
        "future_flats_excluded":int((~nonflat).sum()),
        "scopes":{k:scope_metrics(pred,mask) for k,mask in masks.items()},
        "by_asset":{sym:scope_metrics(pred,nonflat&(pred.symbol.to_numpy()==sym))
                    for sym in SEEN_SYMBOLS},
    }


def summarize(folds:list[dict])->dict:
    if not folds:
        raise ValueError("Empty folds")
    result={}
    for scope in SCOPES:
        entries=[f["metrics"]["scopes"][scope] for f in folds]
        valid=[x for x in entries if x is not None]
        count=sum(x["n"] for x in valid)
        if not count:
            result[scope]={"n":0}
            continue
        scores={
            name:{
                "correct":sum(x["scores"][name]["correct"] for x in valid),
                "accuracy":sum(x["scores"][name]["correct"] for x in valid)/count,
            } for name in ALL_METHODS
        }
        diffs={}
        for name in VARIANTS:
            for reference in REFERENCES:
                key=f"{name}_minus_{reference}"
                arr=[x["paired"][key]["delta_points"] for x in valid]
                a=sum(x["paired"][key]["candidate_only_correct"] for x in valid)
                b=sum(x["paired"][key]["reference_only_correct"] for x in valid)
                diffs[key]={
                    "pooled_delta_points":100*(a-b)/count,
                    "mean_fold_delta_points":float(np.mean(arr)),
                    "positive_folds":sum(v>0 for v in arr),
                    "negative_folds":sum(v<0 for v in arr),
                    "tied_folds":sum(v==0 for v in arr),
                    "candidate_only_correct":a,
                    "reference_only_correct":b,
                    "comparable_folds":len(arr),
                }
        result[scope]={"n":count,"scores":scores,"paired":diffs,
                       "comparable_folds":len(valid)}
    assets={}
    for symbol in SEEN_SYMBOLS:
        entries=[f["metrics"]["by_asset"][symbol] for f in folds]
        valid=[x for x in entries if x is not None]
        n=sum(x["n"] for x in valid)
        assets[symbol]={
            "n":n,
            "accuracy":{name:(sum(x["scores"][name]["correct"] for x in valid)/n
                               if n else None) for name in ALL_METHODS},
        }
    result["by_asset"]=assets
    result["all_future_flats_excluded"]=sum(
        f["metrics"]["future_flats_excluded"] for f in folds)
    return result


def run(candles:dict,source:dict,reference:pd.DataFrame, *,
        fingerprint:str,max_folds:int|None=None)->tuple[dict,pd.DataFrame]:
    plan=load_plan(source)
    if fingerprint!=source.get("dataset_fingerprint_sha256"):
        raise ValueError("Original P0 fingerprint mismatch")
    if set(candles)!=set(SEEN_SYMBOLS):
        raise ValueError("Only four seen assets permitted")
    if max_folds is not None:
        if type(max_folds)!=int or not 1<=max_folds<=36:
            raise ValueError("Invalid smoke fold limit")
        plan=plan[:max_folds]
    contexts,ema=build_inputs(candles,plan[-1]["check_end"])
    folds=[]
    exports=[]
    for fold in plan:
        shared=fold_examples(contexts,ema,fold["check_end"])
        parts,bounds=parts_for_fold(shared,fold,
                                    source["folds"][fold["fold_id"]])
        train,early,check=(parts[k] for k in (
            "train","early_stopping","validation_check"))
        check=align_check(check,reference,fold["fold_id"])
        models={}
        fit_metadata={}
        for name in VARIANTS:
            fitted,info=train_binary(train,early,name=name)
            models[name]=fitted
            fit_metadata[name]=info
        predictions=prediction_frame(check,models)
        metrics=fold_metrics(predictions)
        if metrics["scopes"]["all_nonflat"] is None:
            raise ValueError("No H1 nonflat samples")
        reference_count=metrics["scopes"]["all_nonflat"]["n"]
        if reference_count!=sum(check.next_bar_return.ne(0)):
            raise AssertionError("H1 denominators changed")
        predictions.insert(0,"fold_id",int(fold["fold_id"]))
        exports.append(predictions)
        folds.append({
            "fold_id":int(fold["fold_id"]),
            "bounds":bounds,
            "partition_rows":{k:len(v) for k,v in parts.items()},
            "fitting":fit_metadata,
            "metrics":metrics,
        })
        scores=metrics["scopes"]["all_nonflat"]["scores"]
        print(f"fold={fold['fold_id']} H1_binary nonflat={reference_count} "
              + " ".join(f"{n}={scores[n]['accuracy']:.5f}" for n in VARIANTS),
              flush=True)
    exported=pd.concat(exports,ignore_index=True)
    if exported.duplicated(["fold_id","symbol","bar_close_at"]).any():
        raise ValueError("Duplicate exported paired predictions")
    return {
        "protocol":PROTOCOL,
        "research_status":"EXPLORATORY_PREVIOUSLY_INSPECTED_DEV_CHECKS",
        "smoke_only":max_folds is not None,
        "folds_analyzed":len(folds),
        "horizon_bars":1,
        "binary_training_excludes_zero_change":True,
        "binary_target_definition":"1 iff next_close > current_close, 0 iff <; exact flats excluded",
        "sign_at_inference":"UP iff p_up>0.5, else DOWN; no oracle nonflat gate",
        "training_and_early_only_purged_H12":True,
        "tuning_and_check_never_fit":True,
        "reference_3class_preds_reused_no_refits":True,
        "no_ADA_no_fold_OOS_no_final90_no_P3_changes":True,
        "variants":VARIANTS,
        "summary":summarize(folds),
        "folds":folds,
        "limitations":[
            "Previously explored DEV checks: not new independent evidence",
            "Overlapping fold training and correlated crypto market observations",
            "Training conditional on next-bar not being flat; not a movement detector",
            "No trading execution, slippage, fees, PnL or model promotion",
        ],
    },exported


def _sha(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):
            digest.update(block)
    return digest.hexdigest()


def main()->None:
    p=argparse.ArgumentParser(description="P2 H1 binary CPU study, no OOS/ADA/P3")
    for flag in ("data-root","fold-plan","h1h6-report","predictions-parquet",
                 "output-json","output-csv","output-parquet"):
        p.add_argument("--"+flag,required=True,type=Path)
    p.add_argument("--months",type=int,default=24)
    p.add_argument("--max-folds",type=int)
    p.add_argument("--verify-hashes",action="store_true")
    a=p.parse_args()
    if a.months!=24:
        p.error("Only fixed 24 original months supported")
    outputs=[a.output_json.resolve(),a.output_csv.resolve(),a.output_parquet.resolve()]
    inputs={a.fold_plan.resolve(),a.h1h6_report.resolve(),a.predictions_parquet.resolve()}
    production=a.data_root.resolve().parent
    if (len(set(outputs))!=3 or any(f.exists() or f in inputs or
             f==production or production in f.parents for f in outputs)):
        p.error("Distinct fresh outputs strictly outside the production root only")
    source=json.loads(a.fold_plan.read_text(encoding="utf-8"))
    old=json.loads(a.h1h6_report.read_text(encoding="utf-8"))
    plan=load_plan(source)
    if (len(plan)!=36 or old.get("source_dataset_fingerprint_sha256")!=
            source.get("dataset_fingerprint_sha256")):
        p.error("Invalid original 36-fold H1 source fingerprint")
    reference=load_reference(a.predictions_parquet,old,plan)
    candles,manifests={},[]
    store=CandleStore(a.data_root)
    for sym in SEEN_SYMBOLS:
        print("Consulting original P0:",sym,flush=True)
        candles[sym],files=load_symbol(store,sym,months=24,
                                       verify_hashes=a.verify_hashes)
        manifests.extend(files)
    report,pred=run(candles,source,reference,
                    fingerprint=manifest_fingerprint(manifests),
                    max_folds=a.max_folds)
    report["source_fold_plan_sha256"]=_sha(a.fold_plan)
    report["source_h1h6_report_sha256"]=_sha(a.h1h6_report)
    report["source_predictions_sha256"]=_sha(a.predictions_parquet)
    report["predictions_rows"]=len(pred)
    summary=report["summary"]["all_nonflat"]
    csv=[]
    for name in ALL_METHODS:
        scored=summary["scores"][name]
        cell={"method":name,"nonflat":summary["n"],
              "pooled_accuracy":scored["accuracy"],
              "pooled_correct":scored["correct"]}
        for reference_name in ("ternary_ema2","ternary_30"):
            compared=summary["paired"].get(f"{name}_minus_{reference_name}")
            cell[f"delta_vs_{reference_name}_points"]=(
                compared["pooled_delta_points"] if compared else None)
            cell[f"positive_folds_vs_{reference_name}"]=(
                compared["positive_folds"] if compared else None)
        csv.append(cell)
    for path in outputs:
        path.parent.mkdir(parents=True,exist_ok=True)
    # Refuse any existing path, including near-simultaneous accidental repeats.
    if any(x.exists() for x in outputs):
        raise ValueError("Output created in meantime; refusing to overwrite")
    pred.to_parquet(a.output_parquet,index=False,engine="pyarrow",compression="zstd")
    pd.DataFrame(csv).to_csv(a.output_csv,index=False)
    with a.output_json.open("x",encoding="utf-8") as f:
        json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
    print("Saved H1 binary JSON:",a.output_json,flush=True)
    print("Saved all-model summary:",a.output_csv,flush=True)
    print("Saved check-only predictions:",a.output_parquet,flush=True)
    print("SMOKE ONLY" if report["smoke_only"] else "FULL EXPLORATORY",flush=True)


if __name__=="__main__":
    main()
