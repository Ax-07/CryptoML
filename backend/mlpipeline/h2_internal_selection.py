"""P2 H2 internal-only train/early/tuning selection, never old checks."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from ingestion.storage import CandleStore
from mlpipeline.dataset import load_symbol,manifest_fingerprint
from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS
from mlpipeline.ema_horizon_grid import build_inputs,fold_examples
from mlpipeline.h2_delayed_selective import delayed_truth,attach,fit,policy,metrics,PROTOCOL as MODEL_PROTOCOL
from mlpipeline.label_stability_audit import load_plan

PROTOCOL="p2_h2_internal_train_early_tuning_selection_v1"
NAMES=("regression_30","ridge_ema2")
M5=pd.Timedelta(minutes=5)
H12=pd.Timedelta(minutes=60)

def split_purged(shared,fold,original):
    stop=fold["check_start"]-pd.Timedelta(days=3)
    early_start=stop-pd.Timedelta(days=7)
    start=early_start-pd.Timedelta(days=120)
    t=shared.bar_close_at
    mature=shared.label_mature_at
    ranges={"train":(start,early_start),"early_stopping":(early_start,stop),
            "tuning":(stop,fold["check_start"])}
    result={}
    for name,(a,b) in ranges.items():
        part=shared.loc[t.ge(a)&t.lt(b)&mature.lt(b)].copy()
        expected=original["tuning_rows"] if name=="tuning" else original["partition_rows"][name]
        if (len(part)!=expected or len(part)<100 or
            not part.bar_close_at.ge(a).all() or not part.bar_close_at.lt(b).all()
            or not part.label_mature_at.lt(b).all() or
            set(part.symbol)!=set(SEEN_SYMBOLS)):
            raise ValueError("Original train/early/tuning fold geometry drift: "+name)
        result[name]=part
    return result

def prepared_parts(candles,fold,original):
    cutoff=fold["check_start"]
    # Crucial: clips all features and every prospective label before the CHECK starts.
    contexts,ema=build_inputs(candles,cutoff)
    shared=fold_examples(contexts,ema,cutoff)
    partitions=split_purged(shared,fold,original)
    true=delayed_truth(candles,cutoff)
    output={name:attach(part,true) for name,part in partitions.items()}
    for name,part in output.items():
        bound=(fold["check_start"]-pd.Timedelta(days=10) if name=="train"
               else fold["check_start"]-pd.Timedelta(days=3) if name=="early_stopping"
               else cutoff)
        if (part.delayed_return_bps.isna().any() or
            not np.isfinite(part.delayed_return_bps.to_numpy(dtype=float)).all() or
            not part.delayed_mature_at.lt(bound).all()):
            raise ValueError("Unmatured delayed target in "+name)
    return output

def summarize_tune(df,name):
    side=df[name+"_side"].to_numpy(dtype=int)
    ret=df.delayed_return_bps.to_numpy(dtype=float)
    trade=side!=0
    bps=side[trade]*ret[trade]-2.
    folds=[]
    for _,p in df.groupby("fold_id",sort=True):
        m=p[name+"_side"].to_numpy(dtype=int)!=0
        sel=p.loc[m]
        sums=float(np.sum(sel[name+"_side"].to_numpy()*sel.delayed_return_bps.to_numpy()-2))
        folds.append({"fold_id":int(p.fold_id.iloc[0]),"trades":int(m.sum()),"sum_net_bps":sums})
    assets={}
    for sym in SEEN_SYMBOLS:
        p=df.loc[df.symbol.eq(sym)]
        active=p.loc[p[name+"_side"].ne(0)]
        v=active[name+"_side"].to_numpy()*active.delayed_return_bps.to_numpy()-2
        assets[sym]={"trades":len(active),"net_per_trade_bps":float(v.mean()) if len(v) else None}
    active_folds=[f for f in folds if f["trades"]]
    leave=[float(bps.sum()-f["sum_net_bps"]) for f in folds]
    best_five=float(np.sort(bps)[-5:].sum()) if len(bps)>=5 else float(bps.sum())
    rules={
      "min_200_trades_and_max_5pct_coverage":len(bps)>=200 and len(bps)/len(df)<=.05,
      "positive_net_trade_and_opportunity":len(bps)>0 and float(bps.sum())>0,
      "at_least_12_active_folds_and_60pct_positive":
          len(active_folds)>=12 and sum(x["sum_net_bps"]>0 for x in active_folds)*5>=len(active_folds)*3,
      "at_least_three_assets_20_trades_and_positive":sum(
          v["trades"]>=20 and v["net_per_trade_bps"] is not None and v["net_per_trade_bps"]>0
          for v in assets.values())>=3,
      "positive_after_every_fold_and_best_five_removal":
          len(bps)>=5 and min(leave)>0 and float(bps.sum()-best_five)>0,
    }
    return {"metrics":metrics(ret,side),"active_folds":len(active_folds),
            "positive_active_folds":sum(x["sum_net_bps"]>0 for x in active_folds),
            "folds":folds,"by_asset":assets,
            "min_leave_one_fold_out_sum_net_bps":min(leave),
            "without_best_five_sum_net_bps":float(bps.sum()-best_five),
            "eligibility_rules":rules,"eligible":all(rules.values())}

def run(candles,source,fingerprint):
    plan=load_plan(source)
    if len(plan)!=36 or set(candles)!=set(SEEN_SYMBOLS) or fingerprint!=source["dataset_fingerprint_sha256"]:
        raise ValueError("Wrong immutable P2 dataset")
    frames=[]; records=[]
    for fold in plan:
        fid=fold["fold_id"]
        chunks=prepared_parts(candles,fold,source["folds"][fid])
        # Tuning labels are never supplied to fitting, only to final descriptive scoring.
        pred,fits=fit(chunks["train"],chunks["early_stopping"],chunks["tuning"])
        tune=chunks["tuning"]
        out=tune[["symbol","bar_close_at","delayed_return_bps","delayed_mature_at"]].copy()
        out["fold_id"]=fid
        for name in NAMES:
            out[name+"_expected_bps"]=pred[name]
            out[name+"_side"]=policy(pred[name])
        frames.append(out)
        records.append({"fold_id":fid,"train_rows":len(chunks["train"]),
                        "early_rows":len(chunks["early_stopping"]),
                        "tuning_rows":len(tune),"fitting":fits})
        print("fold",fid,"tuning_only",len(tune),
              "reg30_trades",int((out.regression_30_side!=0).sum()),flush=True)
    all_pred=pd.concat(frames,ignore_index=True)
    if all_pred.duplicated(["fold_id","symbol","bar_close_at"]).any():
        raise ValueError("Duplicate tuning forecast")
    scored={name:summarize_tune(all_pred,name) for name in NAMES}
    selected=next((n for n in ("ridge_ema2","regression_30") if scored[n]["eligible"]),None)
    report={"protocol":PROTOCOL,
            "status":"INTERNAL_TUNING_ONLY_NOT_INDEPENDENT_OOS",
            "source_model_protocol":MODEL_PROTOCOL,
            "dataset_fingerprint_sha256":fingerprint,
            "folds_analyzed":36,
            "selection":"NO_ELIGIBLE_CANDIDATE" if selected is None else selected,
            "selection_is_not_validation":True,
            "never_loaded_or_scored_development_checks":True,
            "no_ADA_no_fold_OOS_no_final90_no_P3":True,
            "fixed_abs_signal_threshold_bps":3.0,
            "fixed_roundtrip_cost_bps":2.0,
            "no_executable_fill_guarantee":True,
            "summary":scored,"folds":records}
    return report,all_pred

def main():
    p=argparse.ArgumentParser(description="P2 train-early-tune ONLY H2 internal selection")
    for name in ("data-root","fold-plan","output-json","output-csv","output-parquet"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--verify-hashes",action="store_true")
    a=p.parse_args()
    src=a.fold_plan.resolve()
    outs=[a.output_json.resolve(),a.output_csv.resolve(),a.output_parquet.resolve()]
    forbidden=a.data_root.resolve().parent
    if (len(set(outs))!=3 or any(x.exists() or x==src or x==forbidden or
         forbidden in x.parents or x.parent!=src.parent for x in outs)):
        p.error("Outputs must be new files in original isolated fold-plan results folder")
    source=json.loads(a.fold_plan.read_text(encoding="utf-8"))
    plan=load_plan(source)
    store=CandleStore(a.data_root)
    candles={};files=[]
    for sym in SEEN_SYMBOLS:
        print("Consulting immutable P0",sym,flush=True)
        candles[sym],m=load_symbol(store,sym,months=24,verify_hashes=a.verify_hashes)
        files.extend(m)
    fingerprint=manifest_fingerprint(files)
    report,pred=run(candles,source,fingerprint)
    rows=[]
    for name,x in report["summary"].items():
        d=x["metrics"]
        rows.append({"candidate":name,"eligible":x["eligible"],
            "trades":d["trades"],"opportunities":d["opportunities"],
            "coverage":d["coverage"],
            "net_per_opportunity_bps":d["by_cost"]["2"]["net_bps_per_opportunity"],
            "net_per_trade_bps":d["by_cost"]["2"]["net_bps_per_trade"],
            "active_folds":x["active_folds"],
            "positive_active_folds":x["positive_active_folds"],
            "without_best_five_sum_net_bps":x["without_best_five_sum_net_bps"],
            "min_leave_one_fold_out_sum_net_bps":x["min_leave_one_fold_out_sum_net_bps"]})
    with a.output_json.open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    with a.output_csv.open("x",encoding="utf-8") as f:
        pd.DataFrame(rows).to_csv(f,index=False)
    pred.to_parquet(a.output_parquet,index=False)
    print("INTERNAL TUNING-ONLY selection:",report["selection"],flush=True)
    print("Saved",a.output_json,a.output_csv,a.output_parquet,flush=True)

if __name__=="__main__":
    main()
