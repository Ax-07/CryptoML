"""No-fit forensic audit of existing P2 H2 delayed-selective dev-check results.

Reads ONLY the previously saved JSON/Parquet and original fold plan.
No raw P0, training, live prices, ADA, fold OOS, final90, P3 or trades.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from mlpipeline.h2_delayed_selective import PROTOCOL as SOURCE,policy,metrics
from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS
from mlpipeline.label_stability_audit import load_plan

KEY=["fold_id","symbol","bar_close_at"]
NAMES=("regression_30","ridge_ema2")
COST=2.

def verified(frame,report,plan):
    if (report.get("protocol")!=SOURCE or report.get("folds_analyzed")!=36
        or report.get("no_ADA_no_fold_OOS_no_final90_no_P3") is not True
        or report.get("trading_or_live_orders") is not False
        or report.get("fixed_abstention_edge_bps")!=3.0
        or report.get("costs_roundtrip_bps")!=[0,2,5]
        or report.get("status")!="EXPLORATORY_REUSED_P2_DEVELOPMENT_CHECKS_NOT_OOS"
        or len(report.get("folds",[]))!=36 or len(plan)!=36):
        raise ValueError("Wrong source historical report/scope")
    cols=KEY+["delayed_return_bps","delayed_mature_at",
              "reference_3class_H2_30_side"]
    for name in NAMES: cols.extend([name+"_expected_bps",name+"_side"])
    if (any(c not in frame for c in cols) or frame[KEY].isna().any(axis=None)
        or frame.duplicated(KEY).any() or not frame.symbol.isin(SEEN_SYMBOLS).all()
        or not frame.fold_id.isin(range(36)).all()
        or not isinstance(frame.bar_close_at.dtype,pd.DatetimeTZDtype)
        or str(frame.bar_close_at.dt.tz)!="UTC"
        or not isinstance(frame.delayed_mature_at.dtype,pd.DatetimeTZDtype)
        or str(frame.delayed_mature_at.dt.tz)!="UTC"
        or not frame.delayed_mature_at.eq(frame.bar_close_at+pd.Timedelta(minutes=10)).all()
        or not np.isfinite(frame.delayed_return_bps.to_numpy(dtype=float)).all()
        or not frame.reference_3class_H2_30_side.isin((-1,1)).all()):
        raise ValueError("Corrupt exported predictions/timestamps")
    for name in NAMES:
        if not np.array_equal(policy(frame[name+"_expected_bps"].to_numpy(dtype=float)),
                              frame[name+"_side"].to_numpy(dtype=int)):
            raise ValueError("Fixed threshold prediction drift: "+name)
    for f in plan:
        fid=f["fold_id"];sub=frame.loc[frame.fold_id.eq(fid)]
        if (len(sub)!=f["expected_check_rows"] or
            not sub.bar_close_at.ge(f["check_start"]).all() or
            not sub.delayed_mature_at.lt(f["check_end"]).all()):
            raise ValueError("Fold count/chronology drift")
        prev=report["folds"][fid]
        if prev["fold_id"]!=fid: raise ValueError("Fold order changed")
        for name in NAMES:
            got=metrics(sub.delayed_return_bps.to_numpy(dtype=float),
                        sub[name+"_side"].to_numpy(dtype=int))
            saved=prev["models"][name]["decision"]
            if (got["trades"]!=saved["trades"] or
                abs(got["by_cost"]["2"]["net_bps_per_opportunity"]-
                    saved["by_cost"]["2"]["net_bps_per_opportunity"])>1e-10):
                raise ValueError("Fold net/trade metrics drift")
    for name in NAMES:
        got=metrics(frame.delayed_return_bps.to_numpy(dtype=float),
                    frame[name+"_side"].to_numpy(dtype=int))
        saved=report["summary"][name]
        if (got["trades"]!=saved["trades"] or
            abs(got["by_cost"]["2"]["net_bps_per_opportunity"]-
                saved["by_cost"]["2"]["net_bps_per_opportunity"])>1e-10):
            raise ValueError("Pooled model results drift")
    for sym in SEEN_SYMBOLS:
        s=frame.loc[frame.symbol.eq(sym)]
        for name in NAMES:
            got=metrics(s.delayed_return_bps.to_numpy(dtype=float),
                        s[name+"_side"].to_numpy(dtype=int))
            old=report["by_asset"][sym][name]
            if got["trades"]!=old["trades"]:
                raise ValueError("Asset trade count differs")
            if got["trades"] and abs(
                got["by_cost"]["2"]["net_bps_per_trade"]-
                old["by_cost"]["2"]["net_bps_per_trade"])>1e-10:
                raise ValueError("Asset mean net differs")
    return frame.sort_values(["fold_id","symbol","bar_close_at"]).reset_index(drop=True)

def describe(frame,name="regression_30"):
    x=frame.copy()
    x["side"]=x[name+"_side"].astype(int)
    x=x.loc[x.side.ne(0)].copy()
    x["net_bps"]=x.side*x.delayed_return_bps-2.
    x["gross_bps"]=x.side*x.delayed_return_bps
    return x

def aggregate(x):
    if x.empty:
        return {"trades":0,"sum_net_bps":0.,"mean_net_bps_per_trade":None}
    return {"trades":len(x),"sum_net_bps":float(x.net_bps.sum()),
            "mean_net_bps_per_trade":float(x.net_bps.mean()),
            "wins":int(x.net_bps.gt(0).sum()),"losses":int(x.net_bps.lt(0).sum())}

def extremes(x):
    good=x.sort_values("net_bps",ascending=False)
    bad=x.sort_values("net_bps")
    rows=[]
    for n in (1,5,10):
        rest_good=good.iloc[n:]
        rest_bad=bad.iloc[n:]
        rows.append({"excluded_top":n,
          "top_gain_sum_bps":float(good.head(n).net_bps.sum()),
          "without_best":aggregate(rest_good),
          "without_worst":aggregate(rest_bad)})
    cols=["fold_id","symbol","bar_close_at","side","gross_bps","net_bps"]
    def records(f): 
        z=f[cols].copy();z["bar_close_at"]=z.bar_close_at.astype(str)
        return z.to_dict("records")
    quantiles={str(q):float(x.net_bps.quantile(q)) for q in (0,.01,.05,.25,.5,.75,.95,.99,1)}
    return {"quantiles_net_bps":quantiles,"outlier_sensitivity":rows,
            "best_five":records(good.head(5)),"worst_five":records(bad.head(5))}

def audit(frame):
    x=describe(frame)
    total=aggregate(x)
    folds={str(fid):aggregate(x.loc[x.fold_id.eq(fid)]) for fid in range(36)}
    assets={sym:aggregate(x.loc[x.symbol.eq(sym)]) for sym in SEEN_SYMBOLS}
    by_side={side:aggregate(x.loc[x.side.eq(v)]) for side,v in (("LONG",1),("SHORT",-1))}
    days=x.assign(day=x.bar_close_at.dt.strftime("%Y-%m-%d")).groupby(
         "day",sort=True).net_bps.agg(["size","sum"]).reset_index()
    days=days.sort_values("sum",ascending=False)
    key=["symbol","bar_close_at"]
    duplicates=x.duplicated(key,keep=False)
    unique=x.sort_values(["fold_id","symbol","bar_close_at"]).drop_duplicates(key,keep="first")
    independent=aggregate(unique)
    leave={}
    for fid in range(36):
        remaining=x.loc[x.fold_id.ne(fid)]
        leave[str(fid)]=aggregate(remaining)
    runs=[]
    for (fold,sym),group in x.groupby(["fold_id","symbol"]):
        t=group.sort_values("bar_close_at")
        run=t.bar_close_at.diff().ne(pd.Timedelta(minutes=5)).cumsum()
        for _,part in t.groupby(run):
            if len(part)>1:
                runs.append({"fold_id":int(fold),"symbol":sym,
                             "signals_consecutive":int(len(part)),
                             "sum_net_bps":float(part.net_bps.sum())})
    runs=sorted(runs,key=lambda z:(-z["signals_consecutive"],z["fold_id"]))[:20]
    positive_sum=sum(max(0,v["sum_net_bps"]) for v in folds.values())
    return {"status":"POST_HOC_REUSED_CHECKS_NOT_INDEPENDENT",
      "model":"regression_30","fixed_cost_bps":2,
      "all":total,"fold_1":folds["1"],
      "without_fold_1":leave["1"],
      "positive_folds":sum(v["sum_net_bps"]>0 for v in folds.values()),
      "negative_folds":sum(v["sum_net_bps"]<0 for v in folds.values()),
      "empty_folds":sum(v["trades"]==0 for v in folds.values()),
      "folds":folds,"leave_one_fold_out":leave,
      "fold_1_share_of_all_trades":folds["1"]["trades"]/len(x),
      "fold_1_share_of_positive_fold_contributions":folds["1"]["sum_net_bps"]/positive_sum if positive_sum else None,
      "by_asset":assets,"by_side":by_side,
      "top_positive_days":days.head(10).to_dict("records"),
      "top_negative_days":days.sort_values("sum").head(10).to_dict("records"),
      "same_market_time_trades_across_folds":int(duplicates.sum()),
      "deduplicated_earliest_fold_diagnostic":independent,
      "longest_consecutive_signal_runs":runs,
      "extremes":extremes(x)}

def main():
    p=argparse.ArgumentParser(description="P2 H2 selective retrospective concentration audit: NO fits")
    for name in ("binary-report","predictions-parquet","fold-plan","output-json","output-csv"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    inputs=[a.binary_report.resolve(),a.predictions_parquet.resolve(),a.fold_plan.resolve()]
    outs=[a.output_json.resolve(),a.output_csv.resolve()]
    if (len(set(outs))!=2 or any(o.exists() or o in inputs or
        o.parent!=a.predictions_parquet.resolve().parent for o in outs)):
        p.error("New separate output paths in same isolated results directory only")
    report=json.loads(a.binary_report.read_text(encoding="utf-8"))
    plan_source=json.loads(a.fold_plan.read_text(encoding="utf-8"))
    plan=load_plan(plan_source)
    if report.get("source_fingerprint_sha256")!=plan_source["dataset_fingerprint_sha256"]:
        raise ValueError("Original P2 data fingerprint mismatch")
    raw=pd.read_parquet(a.predictions_parquet)
    clean=verified(raw,report,plan)
    result=audit(clean)
    result["protocol"]="p2_h2_selective_concentration_audit_v1"
    result["no_fit_no_P0_no_ADA_no_reserved_OOS_no_final90_no_P3"]=True
    rows=[{"fold_id":fid,**score} for fid,score in result["folds"].items()]
    with a.output_json.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    with a.output_csv.open("x",encoding="utf-8") as f:
        pd.DataFrame(rows).to_csv(f,index=False)
    print("No-fit concentration audit saved:",a.output_json,a.output_csv,flush=True)
    for key in ("all","fold_1","without_fold_1","deduplicated_earliest_fold_diagnostic"):
        print(key,result[key],flush=True)
    print("POSITIVE/NEGATIVE/EMPTY FOLDS",result["positive_folds"],
          result["negative_folds"],result["empty_folds"],flush=True)
    print("NOT REAL PROFITS OR INDEPENDENT OOS",flush=True)

if __name__=="__main__":
    main()
