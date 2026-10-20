"""P2 delayed-entry H2 return regression with abstention; historical dev checks only."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from ingestion.storage import CandleStore
from mlpipeline.dataset import load_symbol,manifest_fingerprint
from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS
from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.ema_horizon_grid import build_inputs,fold_examples,EMA_COLUMNS
from mlpipeline.directional_rule_benchmark import parts_for_fold
from mlpipeline.label_stability_audit import load_plan
from mlpipeline.h1_individual_feature_study import load_reference

PROTOCOL="p2_h2_delayed_return_selective_devchecks_v1"
MODELS=("regression_30","ridge_ema2")
COSTS=(0,2,5)
EDGE_BPS=3.
M5=pd.Timedelta(minutes=5)

def delayed_truth(raw,cutoff):
    """Price at t+5 open (after waiting a full bar), then t+10 close."""
    frames=[]
    for sym in SEEN_SYMBOLS:
        d=raw[sym].loc[raw[sym].bar_close_at.lt(cutoff)].sort_values("bar_close_at")
        d=d.reset_index(drop=True)
        if (not d.bar_close_at.is_unique or
            not d.bar_open_at.add(M5).eq(d.bar_close_at).all()):
            raise ValueError("Invalid M5 candle timestamps")
        step2=d.bar_close_at.shift(-2)
        continuity=step2.sub(d.bar_close_at).eq(2*M5)
        # t close, then t+5 next bar close, then bar at t+5 open / t+10 close.
        entry=d.open.shift(-2).astype(float)
        exit_=d.close.shift(-2).astype(float)
        ret=(exit_/entry-1.)*10000.
        valid=continuity & entry.gt(0)&exit_.gt(0)
        frames.append(pd.DataFrame({
            "symbol":sym,"bar_close_at":d.bar_close_at,
            "delayed_return_bps":ret.where(valid),
            "delayed_mature_at":step2.where(valid),
        }))
    return pd.concat(frames,ignore_index=True)

def attach(split,truth):
    old=split[["symbol","bar_close_at"]].copy()
    out=split.merge(truth,on=["symbol","bar_close_at"],how="left",
                    sort=False,validate="one_to_one")
    if len(out)!=len(old) or not out[["symbol","bar_close_at"]].equals(old.reset_index(drop=True)):
        raise ValueError("Delayed label key join drift")
    return out

def fit(train,early,check):
    yt=train.delayed_return_bps.to_numpy(dtype=float)
    ye=early.delayed_return_bps.to_numpy(dtype=float)
    if (len(yt)<500 or len(ye)<100 or
        not np.isfinite(yt).all() or not np.isfinite(ye).all()):
        raise ValueError("Missing purged delayed labels")
    outputs={};fits={}
    for name,cols in (("regression_30",FEATURE_COLUMNS),
                      ("ridge_ema2",list(EMA_COLUMNS))):
        xt=train[cols].astype("float32")
        xe=early[cols].astype("float32")
        xc=check[cols].astype("float32")
        if not all(np.isfinite(x.to_numpy()).all() for x in (xt,xe,xc)):
            raise ValueError("Missing feature")
        if name=="regression_30":
            model=lgb.LGBMRegressor(objective="regression",n_estimators=150,
                learning_rate=.04,num_leaves=31,min_child_samples=100,
                reg_lambda=5,max_bin=63,random_state=42,n_jobs=1,
                verbosity=-1,device_type="cpu")
            model.fit(xt,yt,eval_set=[(xe,ye)],eval_metric="l2",
                callbacks=[lgb.early_stopping(stopping_rounds=30,verbose=False)])
            fits[name]={"best_iteration":int(model.best_iteration_)}
        else:
            model=make_pipeline(StandardScaler(),Ridge(alpha=10.))
            model.fit(xt,yt)
            fits[name]={"best_iteration":None}
        outputs[name]=np.asarray(model.predict(xc),dtype=float)
        if outputs[name].shape!=(len(check),) or not np.isfinite(outputs[name]).all():
            raise ValueError("Invalid predicted return")
    return outputs,fits

def policy(pred):
    p=np.asarray(pred,dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("Invalid policy input")
    return np.where(p>EDGE_BPS,1,np.where(p< -EDGE_BPS,-1,0)).astype("int8")

def metrics(actual,side):
    ret=np.asarray(actual,dtype=float)
    direction=np.asarray(side,dtype=int)
    if ret.shape!=direction.shape or not np.isin(direction,[-1,0,1]).all():
        raise ValueError("Invalid scoring geometry")
    selected=direction!=0
    n=len(ret);trades=int(selected.sum())
    gross=direction*ret
    output={"opportunities":n,"trades":trades,"abstentions":n-trades,
            "coverage":trades/n,"gross_bps_per_opportunity":float(gross.mean())}
    output["by_cost"]={}
    for c in COSTS:
        net=gross-selected*c
        output["by_cost"][str(c)]={
            "net_bps_per_opportunity":float(net.mean()),
            "net_bps_per_trade":float(net[selected].mean()) if trades else None}
    return output

def run(candles,source,original,old_predictions,fingerprint):
    plan=load_plan(source)
    if (len(plan)!=36 or set(candles)!=set(SEEN_SYMBOLS) or
        fingerprint!=source["dataset_fingerprint_sha256"] or
        fingerprint!=original["source_dataset_fingerprint_sha256"]):
        raise ValueError("Original P2 dataset or plan drift")
    keys=["fold_id","symbol","bar_close_at"]
    refs=old_predictions.loc[old_predictions.horizon.eq(2),
                             keys+["lightgbm_30_sign"]].copy()
    if refs.duplicated(keys).any():
        raise ValueError("Reference H2 key duplication")
    contexts,ema=build_inputs(candles,plan[-1]["check_end"])
    records=[];outputs=[]
    for fold in plan:
        fid=fold["fold_id"]
        shared=fold_examples(contexts,ema,fold["check_end"])
        parts,_=parts_for_fold(shared,fold,source["folds"][fid])
        truth=delayed_truth(candles,fold["check_end"])
        train=attach(parts["train"],truth)
        early=attach(parts["early_stopping"],truth)
        check=attach(parts["validation_check"],truth)
        for part,bound in ((train,fold["check_start"]-pd.Timedelta(days=14)),
                           (early,fold["check_start"]-pd.Timedelta(days=3)),
                           (check,fold["check_end"])):
            if (part.delayed_return_bps.isna().any() or
                not part.delayed_mature_at.lt(bound).all()):
                raise ValueError("Delayed outcome missing or escaped fold boundary")
        predictions,fits=fit(train,early,check)
        check["fold_id"]=fid
        check=check.merge(refs,on=keys,validate="one_to_one",how="left",
                          sort=False)
        if (len(check)!=fold["expected_check_rows"] or
            check.lightgbm_30_sign.isna().any()):
            raise ValueError("Missing original H2 reference predictions")
        realized=check.delayed_return_bps.to_numpy(dtype=float)
        record={"fold_id":fid,"partition_rows":{k:len(v) for k,v in parts.items()},
                "models":{}}
        result=check[keys+["delayed_return_bps","delayed_mature_at"]].copy()
        baseline_side=np.where(check.lightgbm_30_sign.to_numpy()==2,1,-1)
        record["reference_always_active_3class_H2_30"]=metrics(realized,baseline_side)
        record["reference_always_abstain"]=metrics(realized,np.zeros(len(check),dtype=int))
        result["reference_3class_H2_30_side"]=baseline_side
        for name,pred in predictions.items():
            side=policy(pred)
            record["models"][name]={
                "fit":fits[name],
                "mse_bps2":float(np.mean((pred-realized)**2)),
                "mae_bps":float(np.mean(np.abs(pred-realized))),
                "decision":metrics(realized,side),
            }
            result[name+"_expected_bps"]=pred
            result[name+"_side"]=side
        records.append(record)
        outputs.append(result)
        print("fold",fid,"signals",record["models"]["regression_30"]["decision"]["trades"],
              "of",len(check),flush=True)
    pred=pd.concat(outputs,ignore_index=True)
    if pred.duplicated(keys).any():
        raise ValueError("Duplicate exported fold predictions")
    totals={}
    for name in (*MODELS,"reference_always_active_3class_H2_30","reference_always_abstain"):
        s=pred["reference_3class_H2_30_side"].to_numpy() if name=="reference_always_active_3class_H2_30" else (
            np.zeros(len(pred),dtype=int) if name=="reference_always_abstain" else
            pred[name+"_side"].to_numpy())
        totals[name]=metrics(pred.delayed_return_bps.to_numpy(dtype=float),s)
    by_asset={}
    for sym in SEEN_SYMBOLS:
        sub=pred.loc[pred.symbol.eq(sym)]
        by_asset[sym]={name:metrics(sub.delayed_return_bps.to_numpy(dtype=float),
                                   sub[name+"_side"].to_numpy())
                      for name in MODELS}
    report={"protocol":PROTOCOL,
        "status":"EXPLORATORY_REUSED_P2_DEVELOPMENT_CHECKS_NOT_OOS",
        "folds_analyzed":36,"no_ADA_no_fold_OOS_no_final90_no_P3":True,
        "entry_assumption":"OPEN of candle starting at t+5min",
        "exit_assumption":"CLOSE of same candle at t+10min",
        "signal_available":"after close(t), deliberately wait 5min",
        "fill_guarantee":False,"trading_or_live_orders":False,
        "costs_roundtrip_bps":list(COSTS),
        "fixed_abstention_edge_bps":EDGE_BPS,
        "source_fingerprint_sha256":fingerprint,
        "summary":totals,"by_asset":by_asset,"folds":records}
    return report,pred
