"""H2/H3/H6 old P2 checks: offline no-fit, non-executable markouts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ingestion.storage import CandleStore
from mlpipeline.dataset import load_symbol,manifest_fingerprint
from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS
from mlpipeline.features import build_symbol_features
from mlpipeline.h1_individual_feature_study import load_reference
from mlpipeline.label_stability_audit import load_plan
from mlpipeline.horizon_markout_source import (
    HORIZONS,MODELS,KEY,rebuild,phase_rows,evaluate_rows)

PROTOCOL="p2_h2_h3_h6_markout_devchecks_v1"

def verify_grid(grid,report,plan):
    expected=report["predictions_rows"]
    if (len(grid)!=expected or grid.duplicated(KEY).any()
        or not grid.symbol.isin(SEEN_SYMBOLS).all()
        or not grid.horizon.isin(range(1,7)).all()
        or not grid.fold_id.isin(range(36)).all()
        or not isinstance(grid.bar_close_at.dtype,pd.DatetimeTZDtype)
        or str(grid.bar_close_at.dt.tz)!="UTC"):
        raise ValueError("Source prediction grid mismatch")
    h1=grid.loc[grid.horizon.eq(1),KEY[:3]]
    for h in range(2,7):
        sub=grid.loc[grid.horizon.eq(h),KEY[:3]]
        if len(sub)!=len(h1) or len(sub.merge(h1,on=KEY[:3],validate="one_to_one"))!=len(h1):
            raise ValueError("Missing cross-horizon prediction keys")
    if not grid.loc[grid.horizon.ne(1),"next_bar_return"].isna().all():
        raise ValueError("Original H>1 returns must be absent")
    for f in plan:
        sub=h1.loc[h1.fold_id.eq(f["fold_id"])]
        if (len(sub)!=f["expected_check_rows"]
            or not sub.bar_close_at.ge(f["check_start"]).all()
            or not (sub.bar_close_at+pd.Timedelta(minutes=60)).lt(f["check_end"]).all()):
            raise ValueError("Escaped original development folds")
    return grid.loc[grid.horizon.isin(HORIZONS)].copy()

def verify_rebuild(data,report):
    for fid in range(36):
        for h in HORIZONS:
            sub=data.loc[data.fold_id.eq(fid)&data.horizon.eq(h)]
            old=report["folds"][fid]["horizons"][f"H{h}"]["metrics"]["all_check"]
            for model in MODELS:
                if (old[model] is None or old[model]["n"]!=len(sub)
                    or old[model]["correct"]!=int(sub[model+"_sign"].eq(sub.target_3class).sum())):
                    raise ValueError("Original check score drift")

def summarize(data):
    for name,values in {
        "always_down_sign":np.zeros(len(data),dtype=int),
        "reversion_1_sign":np.where(data.previous_return.to_numpy(dtype=float)<0,2,0)
    }.items():
        data[name]=values
    output={}
    for h in HORIZONS:
        subset=data.loc[data.horizon.eq(h)]
        phases=[{"phase":p,"metrics":evaluate_rows(phase_rows(subset,p))}
                for p in range(h)]
        first=phase_rows(subset,0)
        output[f"H{h}"]={
            "minutes":h*5,"dense_overlapping_nonadditive":evaluate_rows(subset),
            "fixed_nonoverlap_phase0":evaluate_rows(first),
            "phase_sensitivity":phases,
            "by_symbol":{s:{"dense":evaluate_rows(subset.loc[subset.symbol.eq(s)]),
                            "phase0":evaluate_rows(first.loc[first.symbol.eq(s)])}
                         for s in SEEN_SYMBOLS},
            "by_fold":{str(fid):{"dense":evaluate_rows(subset.loc[subset.fold_id.eq(fid)]),
                                 "phase0":evaluate_rows(first.loc[first.fold_id.eq(fid)])}
                       for fid in range(36)},
            "duplicate_market_times_across_folds":int(
                len(subset)-len(subset[["symbol","bar_close_at"]].drop_duplicates()))
        }
    return {"protocol":PROTOCOL,"status":"REUSED_DEV_CHECKS_EXPLORATORY_NOT_EXECUTABLE",
            "no_training":True,"no_P3_ADA_OOS_final90":True,
            "theoretical_close_t_entry_not_executable":True,
            "no_portfolio_compounding_or_real_fills":True,
            "fixed_roundtrip_cost_bps":[0,2,5,10],
            "horizons":output}

def main():
    p=argparse.ArgumentParser()
    for k in ("data-root","fold-plan","h1h6-report","predictions-parquet","output-json","output-csv"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--verify-hashes",action="store_true")
    a=p.parse_args()
    outputs=(a.output_json.resolve(),a.output_csv.resolve())
    sources=(a.fold_plan.resolve(),a.h1h6_report.resolve(),a.predictions_parquet.resolve())
    root=a.data_root.resolve().parent
    if (outputs[0]==outputs[1] or any(
        x.exists() or x in sources or x.parent!=a.predictions_parquet.resolve().parent
        or x==root or root in x.parents for x in outputs)):
        p.error("Output collision/non-isolated output")
    source=json.loads(a.fold_plan.read_text(encoding="utf8"))
    report=json.loads(a.h1h6_report.read_text(encoding="utf8"))
    plan=load_plan(source)
    if source["dataset_fingerprint_sha256"]!=report.get("source_dataset_fingerprint_sha256"):
        raise ValueError("Historical manifest fingerprint mismatch")
    load_reference(a.predictions_parquet,report,plan)
    cols=KEY+["target_3class","next_bar_return"]+[m+"_sign" for m in MODELS]
    grid=pd.read_parquet(a.predictions_parquet,columns=cols)
    selected=verify_grid(grid,report,plan)
    store=CandleStore(a.data_root)
    candles={}; manifests=[]
    for s in SEEN_SYMBOLS:
        print("Original P0 read-only",s,flush=True)
        bars,part=load_symbol(store,s,months=24,verify_hashes=a.verify_hashes)
        manifests.extend(part)
        candles[s]=build_symbol_features(bars.loc[bars.bar_close_at.lt(plan[-1]["check_end"])])
    if manifest_fingerprint(manifests)!=source["dataset_fingerprint_sha256"]:
        raise ValueError("P0 hash mismatch")
    data=rebuild(selected,candles,plan,report)
    verify_rebuild(data,report)
    result=summarize(data)

    rows=[]
    for name,part in result["horizons"].items():
        for view in ("dense_overlapping_nonadditive","fixed_nonoverlap_phase0"):
            for model,rec in part[view]["models"].items():
                for cost in (0,2,5,10):
                    rows.append({"horizon":name,"minutes":part["minutes"],
                        "view":view,"model":model,"cost_bps":cost,
                        "signals":part[view]["n"],
                        "accuracy_nonflat":rec["accuracy_nonflat"],
                        "gross_mean_bps":rec["gross_mean_bps"],
                        "net_mean_bps":rec["net_mean_bps"][str(cost)],
                        "executable":False})
    with a.output_json.open("x",encoding="utf8") as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    with a.output_csv.open("x",encoding="utf8") as f:
        pd.DataFrame(rows).to_csv(f,index=False)
    print("Saved",a.output_json,a.output_csv,flush=True)
    for h,part in result["horizons"].items():
        x=part["fixed_nonoverlap_phase0"]["models"]["lightgbm_30"]
        print(h,"n=",part["fixed_nonoverlap_phase0"]["n"],
              "accuracy=",x["accuracy_nonflat"],
              "gross_bps=",x["gross_mean_bps"],
              "net_2bps=",x["net_mean_bps"]["2"],flush=True)
    print("NOT EXECUTABLE, REUSED DEV CHECKS",flush=True)

if __name__=="__main__":
    main()
