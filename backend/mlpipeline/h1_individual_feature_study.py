"""P2 H1: compare each of the 30 canonical features alone and with EMA9/26.

EXPLORATORY reused development checks ONLY. Reuse immutable H1 baseline predictions.
No tuning on check labels, no ADA, fold OOS/final90, or P3 modifications.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from ingestion.storage import CandleStore
from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.directional_rule_benchmark import parts_for_fold, majority_direction
from mlpipeline.ema_horizon_grid import build_inputs, fold_examples, sign
from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS
from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.h1_sign_reference_audit import audit as _unused_audit
from mlpipeline.label_stability_audit import load_plan
from mlpipeline.trainer import _class_probs
from mlpipeline.two_stage_diagnostics import _cpu_model

PROTOCOL = "p2_h1_individual_solo_and_ema2_plus_one_v1"
EMA_COLS = ("ema9_price_gap", "ema9_ema26_gap")
BASE_COLS = ("lightgbm_30_sign", "lightgbm_ema2_sign")
M5 = pd.Timedelta(minutes=5)


def select_features(max_features:int|None)->tuple[str,...]:
    if max_features is not None and (
            type(max_features) is not int or not 1<=max_features<=30):
        raise ValueError("--max-features must be in 1..30")
    return tuple(FEATURE_COLUMNS[:max_features])


def load_reference(path:Path,report:dict,plan:list[dict])->pd.DataFrame:
    if (report.get("protocol")!="p2_h1_h6_ema9_26_three_baselines_v1"
        or report.get("folds_analyzed")!=36
        or report.get("base_comparison_H2_H6_reproduced") is not True
        or report.get("horizons")!=list(range(1,7))
        or report.get("no_ADA_no_OOS_no_final90_no_P3_modification") is not True):
        raise ValueError("Require verified full previous H1-H6 report")
    cols=["fold_id","symbol","bar_close_at","horizon","target_3class",
          "next_bar_return","baseline_shared_mask","ema2_autonomous_mask",
          "lightgbm_30_sign","lightgbm_ema2_sign",
          "lightgbm_30_p_down","lightgbm_30_p_neutral","lightgbm_30_p_up",
          "lightgbm_ema2_p_down","lightgbm_ema2_p_neutral","lightgbm_ema2_p_up"]
    df=pd.read_parquet(path,columns=cols)
    if len(df)!=report.get("predictions_rows"):
        raise ValueError("Prior prediction row count mismatch")
    keys=["fold_id","symbol","bar_close_at"]
    if (df.duplicated(keys+["horizon"]).any()
        or not df.horizon.isin(range(1,7)).all()
        or not df.symbol.isin(SEEN_SYMBOLS).all()
        or not df.fold_id.isin(range(36)).all()
        or not isinstance(df.bar_close_at.dtype,pd.DatetimeTZDtype)
        or str(df.bar_close_at.dt.tz)!="UTC"
        or df[keys+["horizon"]].isna().any(axis=None)):
        raise ValueError("Invalid reference grid geometry")
    n_h=df.groupby(keys,sort=False).horizon.agg(["size","nunique"])
    if (n_h["size"].ne(6).any() or n_h["nunique"].ne(6).any()):
        raise ValueError("Incomplete six-horizon reference grid")
    h1=df.loc[df.horizon.eq(1)].drop(columns="horizon").copy()
    if not np.isfinite(h1.next_bar_return.to_numpy(dtype=float)).all():
        raise ValueError("H1 raw returns missing")
    for c in BASE_COLS:
        if not h1[c].isin((0,2)).all():
            raise ValueError("Invalid H1 directional baseline")
    for col in ("baseline_shared_mask","ema2_autonomous_mask"):
        if not pd.api.types.is_bool_dtype(h1[col]):
            raise ValueError("Reference selection mask invalid")
    for name in ("lightgbm_30","lightgbm_ema2"):
        p=h1[[f"{name}_p_{k}" for k in ("down","neutral","up")]].to_numpy(dtype=float)
        if (not np.isfinite(p).all() or (p<0).any() or (p>1).any()
            or not np.allclose(p.sum(axis=1),1,atol=1e-6)):
            raise ValueError("Invalid reference multiclass probabilities")
    for fold in plan:
        x=h1.loc[h1.fold_id.eq(fold["fold_id"])]
        if (len(x)!=fold["expected_check_rows"]
            or set(x.symbol)!=set(SEEN_SYMBOLS)
            or not x.bar_close_at.ge(fold["check_start"]).all()
            or not (x.bar_close_at+pd.Timedelta(minutes=60)).lt(fold["check_end"]).all()):
            raise ValueError("Reference H1 escaped fixed P2 development check")
    old=report["summary"]["H1"]["next_bar_sign"]
    nonflat=h1.next_bar_return.ne(0).to_numpy()
    for model,col in (("lightgbm_30","lightgbm_30_sign"),
                      ("lightgbm_ema2","lightgbm_ema2_sign")):
        truth=np.where(h1.next_bar_return.to_numpy()>0,2,0)
        correct=float((h1[col].to_numpy()[nonflat]==truth[nonflat]).mean())
        if (sum(nonflat)!=old[model]["nonflat"] or
            abs(correct-old[model]["pooled_sign_accuracy"])>1e-13):
            raise ValueError("Original H1 reference accuracy mismatch")
    return h1


def align_check(check:pd.DataFrame, reference:pd.DataFrame, fold_id:int)->pd.DataFrame:
    expected=reference.loc[reference.fold_id.eq(fold_id)].copy()
    keys=["symbol","bar_close_at"]
    if len(check)!=len(expected):
        raise ValueError("Fold check count differs from frozen H1 reference")
    aligned=check.merge(expected.drop(columns="fold_id"),on=keys,how="left",
                        sort=False,validate="one_to_one",
                        suffixes=("","_stored"))
    if (len(aligned)!=len(check)
        or not aligned[keys].equals(check[keys].reset_index(drop=True))):
        raise ValueError("H1 check join changed keys or sample order")
    if (not np.array_equal(aligned.target_h1.to_numpy(dtype=int),
                           aligned.target_3class.to_numpy(dtype=int))
        or not np.allclose(aligned.future_return_h1.to_numpy(dtype=float),
                           aligned.next_bar_return.to_numpy(dtype=float),
                           atol=1e-12,rtol=1e-9)):
        raise ValueError("H1 target/future return drift vs stored predictions")
    return aligned


def compare(truth:np.ndarray, prediction:np.ndarray, ref:np.ndarray,
            mask:np.ndarray)->dict|None:
    y=np.asarray(truth,dtype=int)
    p=np.asarray(prediction,dtype=int)
    r=np.asarray(ref,dtype=int)
    m=np.asarray(mask,dtype=bool)
    if (y.shape!=p.shape or y.shape!=r.shape or y.shape!=m.shape
        or not np.isin(y,[0,2]).all() or not np.isin(p,[0,2]).all()
        or not np.isin(r,[0,2]).all()):
        raise ValueError("Invalid paired direction comparison")
    if not m.any():
        return None
    a=p[m]==y[m]
    b=r[m]==y[m]
    return {"n":int(m.sum()),"candidate_correct":int(a.sum()),
            "reference_correct":int(b.sum()),
            "candidate_accuracy":float(a.mean()),
            "reference_accuracy":float(b.mean()),
            "gain_points":100*float(a.mean()-b.mean()),
            "candidate_only_correct":int((a&~b).sum()),
            "reference_only_correct":int((~a&b).sum())}


def score_variant(check:pd.DataFrame,prediction:np.ndarray,
                  probabilities:np.ndarray)->dict:
    n=len(check)
    probs=np.asarray(probabilities,dtype=float)
    if probs.shape!=(n,3) or not np.isfinite(probs).all():
        raise ValueError("Bad candidate probability matrix")
    actual_return=check.future_return_h1.to_numpy(dtype=float)
    truth=np.where(actual_return>0,2,0)
    nonflat=actual_return!=0
    baseline_masks={
        "all_nonflat":nonflat,
        "shared30_nonflat":nonflat&check.baseline_shared_mask.to_numpy(dtype=bool),
        "ema2_nonflat":nonflat&check.ema2_autonomous_mask.to_numpy(dtype=bool),
    }
    views={}
    for name,mask in baseline_masks.items():
        views[name]={
            ref:compare(truth,prediction,check[f"{ref}_sign"].to_numpy(dtype=int),mask)
            for ref in ("lightgbm_ema2","lightgbm_30")
        }
    canonical=check.target_h1.to_numpy(dtype=int)
    mask=check.baseline_shared_mask.to_numpy(dtype=bool)
    canonical_summary={
        "all_check_correct":int((prediction==canonical).sum()),
        "all_check_n":n,
        "shared30_correct":int((prediction[mask]==canonical[mask]).sum()),
        "shared30_n":int(mask.sum()),
    }
    loss=float(log_loss(canonical,probs,labels=[0,1,2]))
    asset={}
    for sym in SEEN_SYMBOLS:
        sub=(check.symbol.to_numpy()==sym)&nonflat
        asset[sym]=compare(truth,prediction,
                            check.lightgbm_ema2_sign.to_numpy(dtype=int),sub)
    return {"views":views,"by_asset":asset,"canonical":canonical_summary,
            "logloss":loss}


def summarize(folds:list[dict], features:tuple[str,...])->dict:
    output={}
    for col in features:
        output[col]={}
        for variant in ("solo","ema2_plus"):
            entries=[f["candidates"][col][variant] for f in folds]
            paired=[e["views"]["all_nonflat"]["lightgbm_ema2"] for e in entries]
            scores=[x["gain_points"] for x in paired if x is not None]
            all_n=sum(x["n"] for x in paired if x is not None)
            all_candidate=sum(x["candidate_correct"] for x in paired if x is not None)
            all_reference=sum(x["reference_correct"] for x in paired if x is not None)
            assets={}
            for sym in SEEN_SYMBOLS:
                rows=[e["by_asset"][sym] for e in entries if e["by_asset"][sym]]
                assets[sym]={
                    "n":sum(r["n"] for r in rows),
                    "candidate_accuracy":sum(r["candidate_correct"] for r in rows)/sum(r["n"] for r in rows) if rows else None,
                    "delta_vs_ema2_points":100*sum(r["candidate_correct"]-r["reference_correct"] for r in rows)/sum(r["n"] for r in rows) if rows else None,
                }
            output[col][variant]={
                "features_count":1 if variant=="solo" else 3,
                "folds":len(entries),"comparable_folds":len(scores),
                "mean_fold_delta_vs_ema2_points":float(np.mean(scores)) if scores else None,
                "pooled_accuracy":all_candidate/all_n if all_n else None,
                "pooled_ema2_reference_accuracy":all_reference/all_n if all_n else None,
                "pooled_delta_vs_ema2_points":100*(all_candidate-all_reference)/all_n if all_n else None,
                "positive_folds":sum(x>0 for x in scores),
                "negative_folds":sum(x<0 for x in scores),
                "tied_folds":sum(x==0 for x in scores),
                "mean_logloss":float(np.mean([e["logloss"] for e in entries])),
                "by_asset":assets,
            }
    return output


def run(candles:dict,source:dict,old_report:dict,old_predictions:pd.DataFrame,
        *,fingerprint:str,max_folds:int|None=None,
        max_features:int|None=None)->dict:
    plan=load_plan(source)
    if (fingerprint!=source["dataset_fingerprint_sha256"]
        or fingerprint!=old_report.get("source_dataset_fingerprint_sha256")):
        raise ValueError("Input P0 fingerprint differs from prior report")
    if set(candles)!=set(SEEN_SYMBOLS):
        raise ValueError("Only BTC ETH SOL XRP authorized")
    selected=select_features(max_features)
    reference=old_predictions
    if max_folds is not None:
        if type(max_folds) is not int or not 1<=max_folds<=36:
            raise ValueError("Bad --max-folds")
        plan=plan[:max_folds]
    cutoff=plan[-1]["check_end"]
    contexts,ema=build_inputs(candles,cutoff)
    records=[]
    for fold in plan:
        shared=fold_examples(contexts,ema,fold["check_end"])
        parts,bounds=parts_for_fold(shared,fold,
                                    source["folds"][fold["fold_id"]])
        train,early,_,check=(parts[k] for k in (
            "train","early_stopping","tuning","validation_check"))
        check=align_check(check,reference,fold["fold_id"])
        yt=train.target_h1.to_numpy(dtype=int)
        ye=early.target_h1.to_numpy(dtype=int)
        if (np.any(np.bincount(yt,minlength=3)==0)
            or set(ye)!={0,1,2}):
            raise ValueError("Missing H1 training/early class")
        majority=majority_direction(yt)
        candidates={}
        for feature in selected:
            choices={}
            for kind,cols in (
                ("solo",[feature]),("ema2_plus",[*EMA_COLS,feature])):
                x=train[cols].astype("float32")
                model=_cpu_model(binary=False,n_estimators=150,seed=42)
                model.fit(x,yt,
                    eval_set=[(early[cols].astype("float32"),ye)],
                    eval_metric="multi_logloss",
                    callbacks=[lgb.early_stopping(stopping_rounds=30,verbose=False)])
                probs=_class_probs(model,check[cols].astype("float32"))
                predicted=sign(probs,majority)
                choices[kind]={
                    **score_variant(check,predicted,probs),
                    "best_iteration":int(model.best_iteration_),
                    "columns":list(cols),
                }
                diff=choices[kind]["views"]["all_nonflat"]["lightgbm_ema2"]
                print(f"fold={fold['fold_id']} {kind} {feature} "
                      f"delta_ema2_points={diff['gain_points']:+.4f}",
                      flush=True)
            candidates[feature]=choices
        records.append({"fold_id":fold["fold_id"],
                        "bounds":bounds,
                        "partition_rows":{k:len(v) for k,v in parts.items()},
                        "candidates":candidates})
    return {
        "protocol":PROTOCOL,
        "research_status":"EXPLORATORY_REUSED_P2_DEV_CHECKS",
        "smoke_only":max_folds is not None or max_features is not None,
        "folds_analyzed":len(records),
        "features_analyzed":list(selected),
        "features_baseline_total":len(FEATURE_COLUMNS),
        "reference_predictions_reused_no_baseline_refit":True,
        "no_model_selection_or_tuning_from_checks":True,
        "H1_future_flat_excluded_from_direction_accuracy":True,
        "no_ADA_no_fold_OOS_no_final90_no_P3_changes":True,
        "summary":summarize(records,selected),
        "folds":records,
    }


def main()->None:
    p=argparse.ArgumentParser(description="P2 H1 solo-feature and EMA2+one CPU study")
    for arg in ("data-root","fold-plan","h1h6-report","predictions-parquet","output-json","output-csv"):
        p.add_argument("--"+arg,required=True,type=Path)
    p.add_argument("--max-folds",type=int)
    p.add_argument("--max-features",type=int)
    p.add_argument("--months",type=int,default=24)
    p.add_argument("--verify-hashes",action="store_true")
    a=p.parse_args()
    if a.months!=24:
        p.error("Requires exact 24 P0 months")
    inputs={a.fold_plan.resolve(),a.h1h6_report.resolve(),a.predictions_parquet.resolve()}
    outputs={a.output_json.resolve(),a.output_csv.resolve()}
    if (len(outputs)!=2 or any(x.exists() or x in inputs for x in outputs)
        or any(x==a.data_root.resolve() or a.data_root.resolve() in x.parents for x in outputs)):
        p.error("Outputs must be distinct nonexisting non-P0 files")
    source=json.loads(a.fold_plan.read_text(encoding="utf-8"))
    old=json.loads(a.h1h6_report.read_text(encoding="utf-8"))
    plan=load_plan(source)
    if len(plan)!=36 or source.get("dataset_fingerprint_sha256")!=old.get("source_dataset_fingerprint_sha256"):
        p.error("Wrong original 36-fold source/fingerprint")
    reference=load_reference(a.predictions_parquet,old,plan)
    store=CandleStore(a.data_root)
    candles,manifests={},[]
    for sym in SEEN_SYMBOLS:
        print("Reading P0 consultation only:",sym,flush=True)
        candles[sym],files=load_symbol(store,sym,months=24,
                                       verify_hashes=a.verify_hashes)
        manifests.extend(files)
    result=run(candles,source,old,reference,
               fingerprint=manifest_fingerprint(manifests),
               max_folds=a.max_folds,max_features=a.max_features)
    rows=[]
    for feature in result["features_analyzed"]:
        for variant in ("solo","ema2_plus"):
            x=result["summary"][feature][variant]
            rows.append({
                "feature":feature,"variant":variant,
                "folds":x["folds"],
                "mean_fold_delta_vs_ema2_points":x["mean_fold_delta_vs_ema2_points"],
                "pooled_accuracy":x["pooled_accuracy"],
                "pooled_ema2_accuracy":x["pooled_ema2_reference_accuracy"],
                "positive_folds":x["positive_folds"],
                "negative_folds":x["negative_folds"],
                "mean_logloss":x["mean_logloss"],
            })
    for path in outputs:
        path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output_csv,index=False)
    with a.output_json.open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print("Saved report:",a.output_json,flush=True)
    print("Saved all-feature table:",a.output_csv,flush=True)
    print("SMOKE ONLY" if result["smoke_only"] else "EXPLORATORY FULL",flush=True)


if __name__=="__main__":
    main()
