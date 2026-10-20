"""Read-only paired H1 sign audit using original P0 candles and existing predictions."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from ingestion.storage import CandleStore
from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS
from mlpipeline.label_stability_audit import load_plan

NAMES=("lightgbm_30","lightgbm_ema2","ema_cross_rule","always_down","always_up","last_momentum","last_reversion","inverse_ema_posthoc")
PAIRS=(("lightgbm_30","always_down"),("lightgbm_ema2","always_down"),("lightgbm_30","lightgbm_ema2"),("lightgbm_30","last_momentum"),("lightgbm_ema2","last_momentum"),("lightgbm_30","last_reversion"),("lightgbm_ema2","last_reversion"))

def scope_metrics(frame):
    nonflat=frame.loc[frame.next_bar_return.ne(0)]
    if nonflat.empty: return None
    truth=np.where(nonflat.next_bar_return.to_numpy()>0,2,0)
    previous=nonflat.previous_return.to_numpy()
    ema=nonflat.ema_cross_rule_sign.to_numpy(dtype=int)
    pred={
        "lightgbm_30":nonflat.lightgbm_30_sign.to_numpy(dtype=int),
        "lightgbm_ema2":nonflat.lightgbm_ema2_sign.to_numpy(dtype=int),
        "ema_cross_rule":ema,
        "always_down":np.zeros(len(nonflat),dtype=int),
        "always_up":np.full(len(nonflat),2,dtype=int),
        "last_momentum":np.where(previous>0,2,0),
        "last_reversion":np.where(previous<0,2,0),
        "inverse_ema_posthoc":2-ema,
    }
    correct={name:values==truth for name,values in pred.items()}
    return {"n":len(nonflat),"flat":int(len(frame)-len(nonflat)),
       "up":int((truth==2).sum()),"down":int((truth==0).sum()),
       "previous_flat":int((previous==0).sum()),
       "accuracy":{k:float(v.mean()) for k,v in correct.items()},
       "paired":{f"{a}_minus_{b}":{
           "points":100*float(correct[a].mean()-correct[b].mean()),
           "a_only":int((correct[a]&~correct[b]).sum()),
           "b_only":int((~correct[a]&correct[b]).sum())}
           for a,b in PAIRS}}

def audit(grid,plan,previous,source):
    if source.get("protocol")!="p2_h1_h6_ema9_26_three_baselines_v1" or source.get("folds_analyzed")!=36:
        raise ValueError("Require completed original 36-fold H1-H6 report")
    if len(grid)!=source["predictions_rows"]: raise ValueError("Prediction row count drift")
    fields=["fold_id","symbol","bar_close_at","horizon","next_bar_return",
            "lightgbm_30_sign","lightgbm_ema2_sign","ema_cross_rule_sign",
            "baseline_shared_mask","ema2_autonomous_mask"]
    if any(k not in grid for k in fields): raise ValueError("Missing prediction fields")
    if grid.duplicated(fields[:4]).any() or not grid.horizon.isin(range(1,7)).all():
        raise ValueError("Duplicate/outside horizon")
    if not isinstance(grid.bar_close_at.dtype,pd.DatetimeTZDtype) or str(grid.bar_close_at.dt.tz)!="UTC":
        raise ValueError("Prediction timestamps not UTC-aware")
    counts=grid.groupby(fields[:3],sort=False).horizon.agg(["size","nunique"])
    if counts["size"].ne(6).any() or counts["nunique"].ne(6).any():
        raise ValueError("Incomplete H1-H6 keys")
    h=grid.loc[grid.horizon.eq(1)].copy()
    if not h.symbol.isin(SEEN_SYMBOLS).all() or not h.fold_id.isin(range(36)).all():
        raise ValueError("Unknown asset/fold")
    for f in plan:
        t=h.loc[h.fold_id.eq(f["fold_id"])]
        if len(t)!=f["expected_check_rows"] or set(t.symbol)!=set(SEEN_SYMBOLS) or not t.bar_close_at.ge(f["check_start"]).all() or not (t.bar_close_at+pd.Timedelta(minutes=60)).lt(f["check_end"]).all():
            raise ValueError("Fold geometry/check cutoff mismatch: "+str(f["fold_id"]))
    joined=h.merge(previous,on=["symbol","bar_close_at"],how="left",sort=False,validate="many_to_one")
    if len(joined)!=len(h) or not np.isfinite(joined[["previous_return","verified_next_return","next_bar_return"]].to_numpy(dtype=float)).all():
        raise ValueError("Missing/noncontinuous M5 comparison")
    if not np.allclose(joined.verified_next_return,joined.next_bar_return,rtol=1e-9,atol=1e-12):
        raise ValueError("H1 target differs from immutable P0")
    all_results=scope_metrics(joined)
    old=source["summary"]["H1"]["next_bar_sign"]
    if all_results["n"]!=old["lightgbm_30"]["nonflat"] or all_results["flat"]!=old["lightgbm_30"]["flat"]:
        raise ValueError("Prior H1 denominators differ")
    for name in ("lightgbm_30","lightgbm_ema2","ema_cross_rule"):
        if not np.isclose(all_results["accuracy"][name],old[name]["pooled_sign_accuracy"],atol=1e-13,rtol=0):
            raise ValueError("Original H1 model score mismatch: "+name)
    views={"all_checks":joined,
           "shared_30_selected":joined.loc[joined.baseline_shared_mask],
           "ema2_autonomous":joined.loc[joined.ema2_autonomous_mask]}
    answer={}
    for scope,subset in views.items():
        pooled=scope_metrics(subset)
        by_asset={s:scope_metrics(subset.loc[subset.symbol.eq(s)]) for s in SEEN_SYMBOLS}
        by_fold={str(i):scope_metrics(subset.loc[subset.fold_id.eq(i)]) for i in range(36)}
        fold_wins={}
        for a,b in PAIRS:
            k=f"{a}_minus_{b}"
            deltas=[v["paired"][k]["points"] for v in by_fold.values() if v]
            fold_wins[k]={"positive":sum(v>0 for v in deltas),
                          "negative":sum(v<0 for v in deltas),
                          "ties":sum(v==0 for v in deltas),
                          "mean_fold_points":float(np.mean(deltas)) if deltas else None}
        answer[scope]={"all_rows":len(subset),"pooled":pooled,
                       "by_asset":by_asset,"by_fold":by_fold,"fold_wins":fold_wins}
    return {"protocol":"p2_h1_paired_fixed_references_v1",
            "research_status":"POST_HOC_EXPLORATORY_NOT_INDEPENDENT",
            "no_training":True,"no_OOS_or_ADA_or_final90_scored":True,
            "tie_policy":"DOWN for flat previous candle",
            "inverse_ema_posthoc_only":True,
            "overlapping_folds_not_independent":True,
            "views":answer}

def past_candles(candles,cutoff):
    result=[]
    for sym in SEEN_SYMBOLS:
        df=candles[sym].loc[candles[sym].bar_close_at.lt(cutoff),
            ["bar_open_at","bar_close_at","close"]].sort_values("bar_open_at").reset_index(drop=True)
        before=df.bar_open_at.diff().eq(pd.Timedelta(minutes=5))
        after=df.bar_open_at.shift(-1).sub(df.bar_open_at).eq(pd.Timedelta(minutes=5))
        out=df[["bar_close_at"]].copy()
        out["symbol"]=sym
        out["previous_return"]=(df.close/df.close.shift(1)-1).where(before)
        out["verified_next_return"]=(df.close.shift(-1)/df.close-1).where(after)
        result.append(out)
    return pd.concat(result,ignore_index=True)

def main():
    p=argparse.ArgumentParser()
    for name in ("data-root","fold-plan","h1h6-report","predictions-parquet","output"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--verify-hashes",action="store_true")
    a=p.parse_args()
    output=a.output.resolve()
    if output.exists() or output in {a.fold_plan.resolve(),a.h1h6_report.resolve(),a.predictions_parquet.resolve()} or output==a.data_root.resolve() or a.data_root.resolve() in output.parents:
        p.error("Output exists or collides with P0/input")
    plan_source=json.loads(a.fold_plan.read_text(encoding="utf-8"))
    plan=load_plan(plan_source)
    report=json.loads(a.h1h6_report.read_text(encoding="utf-8"))
    if report.get("source_dataset_fingerprint_sha256")!=plan_source["dataset_fingerprint_sha256"]:
        raise ValueError("Source fingerprint mismatch")
    grid=pd.read_parquet(a.predictions_parquet)
    store=CandleStore(a.data_root)
    candles={}; manifests=[]
    for sym in SEEN_SYMBOLS:
        print("Read-only P0",sym,flush=True)
        candles[sym],m=load_symbol(store,sym,months=24,verify_hashes=a.verify_hashes)
        manifests.extend(m)
    if manifest_fingerprint(manifests)!=plan_source["dataset_fingerprint_sha256"]:
        raise ValueError("P0 SHA manifest fingerprint changed")
    data=audit(grid,plan,past_candles(candles,plan[-1]["check_end"]),report)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
    x=data["views"]["all_checks"]["pooled"]
    print("H1 n=",x["n"],"flats=",x["flat"],flush=True)
    for name,val in x["accuracy"].items(): print(name,round(val*100,3),flush=True)
    print("Saved:",output,flush=True)

if __name__=="__main__":main()
