"""P2 H1 direct binary sign benchmark, offline exploratory historical checks ONLY."""
from __future__ import annotations
import argparse
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

PROTOCOL="p2_h1_binary_direct_sign_four_models_v1"
CANDIDATES={
    "binary_ema2":list(EMA_COLUMNS),
    "binary_ema2_close":[*EMA_COLUMNS,"close_location"],
    "binary_30":list(FEATURE_COLUMNS),
    "logistic_ema2":list(EMA_COLUMNS),
}
REFERENCES=("lightgbm_ema2","lightgbm_30","always_down","previous_reversion")
ALL_MODELS=tuple(CANDIDATES)+REFERENCES
CONFIDENCE_EDGES=(0.,.01,.025,.05,float("inf"))


def nonflat_sign(returns:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    """Zero movement is not a binary training observation; never assign it DOWN."""
    r=np.asarray(returns,dtype=float)
    if r.ndim!=1 or not np.isfinite(r).all():
        raise ValueError("Invalid H1 raw returns")
    keep=r!=0
    return (r[keep]>0).astype(np.int8),keep


def fit_candidates(train:pd.DataFrame,early:pd.DataFrame,check:pd.DataFrame)->tuple[dict,dict]:
    yt,mt=nonflat_sign(train.future_return_h1.to_numpy(dtype=float))
    ye,me=nonflat_sign(early.future_return_h1.to_numpy(dtype=float))
    if len(yt)<100 or len(ye)<100 or set(yt)!={0,1} or set(ye)!={0,1}:
        raise ValueError("Insufficient binary nonflat train/early labels")
    result={}
    training={}
    for name,cols in CANDIDATES.items():
        x=train.loc[mt,cols].astype("float32")
        e=early.loc[me,cols].astype("float32")
        c=check[cols].astype("float32")
        if not (np.isfinite(x.to_numpy()).all() and
                np.isfinite(e.to_numpy()).all() and
                np.isfinite(c.to_numpy()).all()):
            raise ValueError("Nonfinite feature values")
        if name=="logistic_ema2":
            model=make_pipeline(StandardScaler(),
                LogisticRegression(C=1.,max_iter=1000,solver="lbfgs",random_state=42))
            model.fit(x,yt)
            best=None
        else:
            model=_cpu_model(binary=True,n_estimators=150,seed=42)
            model.fit(x,yt,eval_set=[(e,ye)],eval_metric="binary_logloss",
                      callbacks=[lgb.early_stopping(stopping_rounds=30,verbose=False)])
            best=int(model.best_iteration_)
        cls=np.asarray(model.classes_ if name!="logistic_ema2"
                       else model.named_steps["logisticregression"].classes_,dtype=int)
        if not np.array_equal(cls,[0,1]):
            raise ValueError("Binary class ordering drift")
        prob=np.asarray(model.predict_proba(c),dtype=float)
        if (prob.shape!=(len(c),2) or not np.isfinite(prob).all()
            or not np.allclose(prob.sum(axis=1),1.,atol=1e-6)):
            raise ValueError("Invalid binary class probabilities")
        result[name]=prob[:,1]
        training[name]={"columns":cols,"best_iteration":best,
                        "train_rows_nonflat":len(yt),"early_rows_nonflat":len(ye),
                        "train_flat_excluded":int((~mt).sum()),
                        "early_flat_excluded":int((~me).sum())}
    return result,training


def check_metrics(check:pd.DataFrame,prob:dict)->tuple[dict,pd.DataFrame]:
    ret=check.future_return_h1.to_numpy(dtype=float)
    y,nonflat=nonflat_sign(ret)
    if not len(y):
        raise ValueError("Empty non-flat H1 check")
    n=len(check)
    previous=check.ret_1.to_numpy(dtype=float)
    if not np.isfinite(previous).all():
        raise ValueError("Missing causal previous-candle return")
    dirs={}
    for name in CANDIDATES:
        p=np.asarray(prob[name],dtype=float)
        if p.shape!=(n,) or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
            raise ValueError("Invalid predicted binary probability")
        dirs[name]=(p>0.5).astype(np.int8)
    dirs["lightgbm_ema2"]=(check.lightgbm_ema2_sign.to_numpy(dtype=int)==2).astype(np.int8)
    dirs["lightgbm_30"]=(check.lightgbm_30_sign.to_numpy(dtype=int)==2).astype(np.int8)
    dirs["always_down"]=np.zeros(n,dtype=np.int8)
    dirs["previous_reversion"]=(previous<0).astype(np.int8)
    if set(dirs)!=set(ALL_MODELS):
        raise AssertionError("Unexpected comparator")
    for name in ALL_MODELS:
        if not np.isin(dirs[name],[0,1]).all():
            raise ValueError("Non-binary guess")
    correctness={k:(v[nonflat]==y) for k,v in dirs.items()}
    acc={k:{"correct":int(h.sum()),"n":len(y),"accuracy":float(h.mean())}
         for k,h in correctness.items()}
    pairs={}
    for candidate in CANDIDATES:
        for reference in REFERENCES:
            a,b=correctness[candidate],correctness[reference]
            pairs[f"{candidate}_minus_{reference}"]={
                "delta_points":100*float(a.mean()-b.mean()),
                "candidate_only_correct":int((a&~b).sum()),
                "reference_only_correct":int((~a&b).sum()),
            }
    probability_metrics={}
    for name in CANDIDATES:
        p=np.asarray(prob[name])[nonflat]
        brier=float(np.mean((p-y)**2))
        loss=float(log_loss(y,np.column_stack([1-p,p]),labels=[0,1]))
        auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None
        bands=[]
        margin=np.abs(p-.5)
        for lower,upper in zip(CONFIDENCE_EDGES[:-1],CONFIDENCE_EDGES[1:]):
            sel=(margin>=lower)&(margin<upper)
            bands.append({"lower":lower,"upper":None if np.isinf(upper) else upper,
                          "n":int(sel.sum()),
                          "accuracy":float(correctness[name][sel].mean())
                          if sel.any() else None})
        probability_metrics[name]={"binary_brier":brier,
                                   "binary_logloss":loss,"auc":auc,
                                   "confidence_bands":bands}
    checks={}
    for mask_name,mask in (
        ("all_nonflat",nonflat),
        ("original_30_selected",nonflat&check.baseline_shared_mask.to_numpy(dtype=bool)),
        ("original_ema2_selected",nonflat&check.ema2_autonomous_mask.to_numpy(dtype=bool)),
    ):
        valid=np.flatnonzero(mask)
        checks[mask_name]={"n":len(valid),"accuracy":{
            name:float((dirs[name][valid]==(ret[valid]>0)).mean()) if len(valid) else None
            for name in ALL_MODELS}}
    per_symbol={}
    for sym in SEEN_SYMBOLS:
        sel=(check.symbol.to_numpy()==sym)&nonflat
        indexes=np.flatnonzero(sel)
        per_symbol[sym]={"n":len(indexes),"accuracy":{
            name:float((dirs[name][indexes]==(ret[indexes]>0)).mean())
            if len(indexes) else None for name in ALL_MODELS}}
    predictions=check[["symbol","bar_close_at"]].copy()
    predictions["fold_id"]=int(check.fold_id.iloc[0]) if "fold_id" in check else -1
    predictions["actual_return_h1"]=ret
    predictions["is_flat"]=~nonflat
    predictions["actual_up_or_null"]=np.where(nonflat,(ret>0).astype(float),np.nan)
    for name in ALL_MODELS:
        predictions[f"{name}_sign"]=dirs[name]
    for name in CANDIDATES:
        predictions[f"{name}_p_up"]=prob[name]
    return {"n_all":n,"n_nonflat":len(y),"flat":int((~nonflat).sum()),
            "accuracy":acc,"paired":pairs,"binary_quality":probability_metrics,
            "views":checks,"by_symbol":per_symbol},predictions


def _summarize(folds:list[dict])->dict:
    names=ALL_MODELS
    counts={name:{"n":0,"correct":0,"fold_accuracies":[]} for name in names}
    paired={f"{cand}_minus_{ref}":[] for cand in CANDIDATES for ref in REFERENCES}
    symbols={sym:{name:{"n":0,"correct":0} for name in names} for sym in SEEN_SYMBOLS}
    for fold in folds:
        m=fold["metrics"]
        for name in names:
            rec=counts[name]; x=m["accuracy"][name]
            rec["n"]+=x["n"];rec["correct"]+=x["correct"]
            rec["fold_accuracies"].append(x["accuracy"])
        for key,x in m["paired"].items():
            paired[key].append(x["delta_points"])
        for sym in SEEN_SYMBOLS:
            row=m["by_symbol"][sym]
            for name in names:
                n=row["n"]; a=row["accuracy"][name]
                if n:
                    symbols[sym][name]["n"]+=n
                    symbols[sym][name]["correct"]+=int(round(a*n))
    return {
        "accuracy":{name:{
            "n":v["n"],"pooled_accuracy":v["correct"]/v["n"],
            "mean_fold_accuracy":float(np.mean(v["fold_accuracies"]))}
            for name,v in counts.items()},
        "paired":{key:{"mean_fold_delta_points":float(np.mean(ds)),
                         "positive_folds":sum(d>0 for d in ds),
                         "negative_folds":sum(d<0 for d in ds),
                         "tied_folds":sum(d==0 for d in ds),
                         "pooled_delta_points":100*(
                             counts[key.split("_minus_")[0]]["correct"]-
                             counts[key.split("_minus_")[1]]["correct"])/counts[
                                key.split("_minus_")[0]]["n"]}
                  for key,ds in paired.items()},
        "by_symbol":{sym:{name:{
            "n":v["n"],"accuracy":v["correct"]/v["n"] if v["n"] else None}
            for name,v in values.items()} for sym,values in symbols.items()},
        "flat_checks":sum(row["metrics"]["flat"] for row in folds),
    }


def run(candles:dict,plan_source:dict,original_report:dict,
        references:pd.DataFrame,*,fingerprint:str,
        max_folds:int|None=None)->tuple[dict,pd.DataFrame]:
    plan=load_plan(plan_source)
    if (fingerprint!=plan_source["dataset_fingerprint_sha256"]
        or fingerprint!=original_report["source_dataset_fingerprint_sha256"]
        or set(candles)!=set(SEEN_SYMBOLS)):
        raise ValueError("Original four-asset P2 fingerprint/scope mismatch")
    if max_folds is not None:
        if type(max_folds)!=int or not 1<=max_folds<=36:
            raise ValueError("Invalid max folds")
        plan=plan[:max_folds]
    contexts,emas=build_inputs(candles,plan[-1]["check_end"])
    folds=[];frames=[]
    for fold in plan:
        shared=fold_examples(contexts,emas,fold["check_end"])
        parts,bounds=parts_for_fold(shared,fold,
                                  plan_source["folds"][fold["fold_id"]])
        train,early,_,check=(parts[k] for k in (
            "train","early_stopping","tuning","validation_check"))
        check=align_check(check,references,fold["fold_id"])
        probs,training=fit_candidates(train,early,check)
        check["fold_id"]=fold["fold_id"]
        metrics,predictions=check_metrics(check,probs)
        folds.append({"fold_id":fold["fold_id"],"bounds":bounds,
                      "partition_rows":{k:len(v) for k,v in parts.items()},
                      "training":training,"metrics":metrics})
        frames.append(predictions)
        print(f"fold={fold['fold_id']} nonflat={metrics['n_nonflat']} "
              f"EMA2_binary={metrics['accuracy']['binary_ema2']['accuracy']:.5f} "
              f"EMA2_3class={metrics['accuracy']['lightgbm_ema2']['accuracy']:.5f}",
              flush=True)
    pred=pd.concat(frames,ignore_index=True)
    if pred.duplicated(["fold_id","symbol","bar_close_at"]).any():
        raise ValueError("Repeated fold predictions")
    return {"protocol":PROTOCOL,
            "research_status":"EXPLORATORY_REUSED_PREVIOUSLY_INSPECTED_P2_CHECKS",
            "smoke_only":max_folds is not None,
            "folds_analyzed":len(folds),"models":CANDIDATES,
            "reference_predictions_reused_no_3class_refit":True,
            "zero_return_excluded_in_train_early_and_primary_scoring":True,
            "no_ADA_no_fold_OOS_no_final90_no_P3_changes":True,
            "summary":_summarize(folds),"folds":folds},pred


def main()->None:
    p=argparse.ArgumentParser(description="P2 direct binary H1 diagnostic, old dev checks only")
    for name in ("data-root","fold-plan","h1h6-report","predictions-parquet",
                 "output-json","output-csv","output-parquet"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--months",type=int,default=24)
    p.add_argument("--max-folds",type=int)
    p.add_argument("--verify-hashes",action="store_true")
    a=p.parse_args()
    if a.months!=24:
        p.error("Only original 24 months")
    outputs=[a.output_json.resolve(),a.output_csv.resolve(),a.output_parquet.resolve()]
    sources=[a.fold_plan.resolve(),a.h1h6_report.resolve(),a.predictions_parquet.resolve()]
    production=a.data_root.resolve().parent
    if (len(set(outputs))!=3 or any(q.exists() or q in sources or
        q==production or production in q.parents for q in outputs)):
        p.error("Output collision or production write forbidden")
    source=json.loads(a.fold_plan.read_text(encoding="utf-8"))
    old=json.loads(a.h1h6_report.read_text(encoding="utf-8"))
    plan=load_plan(source)
    if (len(plan)!=36 or source["dataset_fingerprint_sha256"]!=
        old.get("source_dataset_fingerprint_sha256")):
        raise ValueError("Original P2 identity mismatch")
    existing=load_reference(a.predictions_parquet,old,plan)
    candles={};fingerprints=[]
    store=CandleStore(a.data_root)
    for sym in SEEN_SYMBOLS:
        print("Reading immutable P0",sym,flush=True)
        candles[sym],manifests=load_symbol(store,sym,months=24,
                                           verify_hashes=a.verify_hashes)
        fingerprints.extend(manifests)
    result,pred=run(candles,source,old,existing,
                    fingerprint=manifest_fingerprint(fingerprints),
                    max_folds=a.max_folds)
    for path in outputs:
        path.parent.mkdir(parents=True,exist_ok=True)
    with a.output_json.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    rows=[]
    for name,entry in result["summary"]["accuracy"].items():
        rows.append({"model":name,**entry})
    pd.DataFrame(rows).to_csv(a.output_csv,index=False)
    pred.to_parquet(a.output_parquet,index=False,engine="pyarrow")
    print("Wrote",a.output_json,a.output_csv,a.output_parquet,flush=True)
    print("SMOKE ONLY" if result["smoke_only"] else "EXPLORATORY COMPLETE",flush=True)


if __name__=="__main__":
    main()
