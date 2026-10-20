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
