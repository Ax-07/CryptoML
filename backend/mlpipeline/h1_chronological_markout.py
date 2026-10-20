"""Offline H1 chronological markout study on ORIGINAL REUSED P2 dev checks only.

This is NOT an executable strategy backtest: the proxy assumes a fill at close(t),
even though predictions can only be computed AFTER close(t). M5 OHLCV cannot
establish latency, bid-ask fills, queue, fees or executable trade prices.

Never refits a model; never accesses P0/ADA/fold-OOS/final90/P3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS
from mlpipeline.h1_binary_diagnostic import (
    ALL_MODELS, CANDIDATES, PROTOCOL as BINARY_PROTOCOL,
)
from mlpipeline.h1_individual_feature_study import load_reference
from mlpipeline.label_stability_audit import load_plan

PROTOCOL = "p2_h1_chronological_nonexecutable_markout_v1"
BAR = pd.Timedelta(minutes=5)
COST_ROUNDTRIP_BPS = (0.0, 2.0, 5.0, 10.0)
PROBABILITY_MARGIN_BINS = ((0.0, 0.01), (0.01, 0.025),
                           (0.025, 0.05), (0.05, None))
MODEL_NAMES = tuple(ALL_MODELS)
COMPARISONS = (
    ("binary_30", "logistic_ema2"),
    ("binary_30", "lightgbm_30"),
    ("logistic_ema2", "lightgbm_ema2"),
    ("binary_ema2", "lightgbm_ema2"),
    ("binary_ema2_close", "binary_ema2"),
)
KEY = ["fold_id", "symbol", "bar_close_at"]


def validate_inputs(grid:pd.DataFrame, binary_report:dict,
                    old_h1:pd.DataFrame, plan:list[dict])->pd.DataFrame:
    """Full, exact H1 original-check keys and counts, sign/probability integrity."""
    if (binary_report.get("protocol")!=BINARY_PROTOCOL
        or binary_report.get("research_status")!=
            "EXPLORATORY_REUSED_PREVIOUSLY_INSPECTED_P2_CHECKS"
        or binary_report.get("smoke_only") is not False
        or binary_report.get("folds_analyzed")!=36
        or binary_report.get("no_ADA_no_fold_OOS_no_final90_no_P3_changes") is not True
        or binary_report.get("zero_return_excluded_in_train_early_and_primary_scoring") is not True
        or tuple(binary_report.get("models",{}))!=tuple(CANDIDATES)
        or len(binary_report.get("folds",[]))!=36 or len(plan)!=36):
        raise ValueError("Full P2 binary report with original scope required")
    required = KEY+["actual_return_h1","is_flat","actual_up_or_null"]
    required += [f"{m}_sign" for m in MODEL_NAMES]
    required += [f"{m}_p_up" for m in CANDIDATES]
    if (any(c not in grid.columns for c in required)
        or len(grid)!=len(old_h1)
        or grid[KEY].isna().any(axis=None)
        or grid[KEY].duplicated().any()
        or not grid.symbol.isin(SEEN_SYMBOLS).all()
        or not isinstance(grid.bar_close_at.dtype,pd.DatetimeTZDtype)
        or str(grid.bar_close_at.dt.tz)!="UTC"
        or not grid.fold_id.isin(range(36)).all()):
        raise ValueError("Invalid/missing H1 prediction rows or keys")
    src=old_h1[KEY+["next_bar_return","lightgbm_ema2_sign",
                    "lightgbm_30_sign"]]
    if src[KEY].duplicated().any():
        raise ValueError("Duplicate original three-class H1 prediction")
    merged=grid[required].merge(src,on=KEY,how="outer",indicator=True,
        sort=False,validate="one_to_one",suffixes=("","_source"))
    if (len(merged)!=len(grid) or not merged._merge.eq("both").all()):
        raise ValueError("Saved binary and original H1 check keys differ")
    future=merged.actual_return_h1.to_numpy(dtype=float)
    recorded=merged.next_bar_return.to_numpy(dtype=float)
    if (not np.isfinite(future).all()
        or not np.allclose(future,recorded,rtol=1e-9,atol=1e-12)):
        raise ValueError("Saved future H1 return differs from original P2")
    if (not pd.api.types.is_bool_dtype(merged["is_flat"])
        or not np.array_equal(merged.is_flat.to_numpy(),future==0)):
        raise ValueError("Corrupt zero-return flag")
    actual=merged.actual_up_or_null.to_numpy(dtype=float)
    if (not np.isnan(actual[future==0]).all()
        or not np.array_equal(actual[future!=0],(future[future!=0]>0).astype(float))):
        raise ValueError("Binary actual-up ground truth drift")
    for name in MODEL_NAMES:
        col=f"{name}_sign"
        if (not merged[col].isin((0,1)).all()):
            raise ValueError("Direction must be binary for "+name)
    for name in CANDIDATES:
        probabilities=merged[f"{name}_p_up"].to_numpy(dtype=float)
        if (not np.isfinite(probabilities).all()
            or (probabilities<0).any() or (probabilities>1).any()
            or not np.array_equal(merged[f"{name}_sign"].to_numpy(),
                                  (probabilities>0.5).astype("int8"))):
            raise ValueError("Probability/prediction drift: "+name)
    for name in ("lightgbm_ema2","lightgbm_30"):
        if not np.array_equal(merged[f"{name}_sign"].to_numpy(dtype=int),
                              (merged[f"{name}_sign_source"].to_numpy(dtype=int)==2).astype(int)):
            raise ValueError("Historical three-class reference drift: "+name)
    if not merged.always_down_sign.eq(0).all():
        raise ValueError("Always-down rule changed")
    full_n=0; full_nonflat=0
    for fold in plan:
        fid=fold["fold_id"]
        part=merged.loc[merged.fold_id.eq(fid)]
        previous=binary_report["folds"][fid]
        if (previous.get("fold_id")!=fid or
            len(part)!=fold["expected_check_rows"] or
            len(part)!=previous["metrics"]["n_all"] or
            part.actual_return_h1.ne(0).sum()!=previous["metrics"]["n_nonflat"] or
            not part.bar_close_at.ge(fold["check_start"]).all() or
            not (part.bar_close_at+pd.Timedelta(minutes=60)).lt(fold["check_end"]).all()):
            raise ValueError("Fold check/purge geometry or count drift: "+str(fid))
        valid=part.actual_return_h1.ne(0).to_numpy()
        truth=part.actual_return_h1.to_numpy()[valid]>0
        for name in MODEL_NAMES:
            correct=int(np.sum(part[f"{name}_sign"].to_numpy()[valid]==truth))
            if correct!=previous["metrics"]["accuracy"][name]["correct"]:
                raise ValueError("Original fold accuracy drift: "+name)
        full_n+=len(part)
        full_nonflat+=int(valid.sum())
    if full_n!=len(merged):
        raise ValueError("Unexpected fold total")
    old_summary=binary_report["summary"]
    if (sum(x["metrics"]["flat"] for x in binary_report["folds"])!=int((future==0).sum())
        or old_summary["flat_checks"]!=int((future==0).sum())):
        raise ValueError("Original flat count differs")
    for name in MODEL_NAMES:
        valid=future!=0
        correct=int(np.sum(merged[f"{name}_sign"].to_numpy()[valid]==(future[valid]>0)))
        expected=old_summary["accuracy"][name]
        if (expected["n"]!=full_nonflat or
            not np.isclose(correct/full_nonflat,expected["pooled_accuracy"],atol=1e-13,rtol=0)):
            raise ValueError("Original pooled accuracy mismatch: "+name)
    return merged[required].sort_values(["bar_close_at","fold_id","symbol"]).reset_index(drop=True)


def chronological_gate(data:pd.DataFrame)->dict:
    """Replay prediction events strictly BEFORE H1 truth events become available.

    The stored parquet already contains post-hoc outcomes (from the old dev
    experiment). Separate event streams ensure no simulated signal uses them.
    This is a chronology diagnostic, NOT proof forecasts were timestamped live.
    """
    predictions=data[KEY+[f"{name}_sign" for name in MODEL_NAMES]].copy()
    truth=data[KEY+["actual_return_h1"]].copy()
    truth["mature_at"]=truth.bar_close_at+BAR
    predictions=predictions.sort_values(["bar_close_at","fold_id","symbol"])
    truth=truth.sort_values(["mature_at","fold_id","symbol"])
    pvals=predictions.itertuples(index=False,name=None)
    tvals=truth.itertuples(index=False,name=None)
    p=next(pvals,None)
    q=next(tvals,None)
    pending={}
    forecast_count=settled_count=0
    while p is not None or q is not None:
        pt=p[2] if p is not None else pd.Timestamp.max.tz_localize("UTC")
        qt=q[4] if q is not None else pd.Timestamp.max.tz_localize("UTC")
        now=min(pt,qt)
        # Maturity events are processed BEFORE predictions newly available at now.
        while q is not None and q[4]==now:
            fid,sym,t,ret,matures=q
            key=(fid,sym,t)
            if key not in pending or not t<matures or not np.isfinite(ret):
                raise ValueError("Settlement without earlier prediction or invalid return")
            pending.pop(key)
            settled_count+=1
            q=next(tvals,None)
        while p is not None and p[2]==now:
            fid,sym,t,*signs=p
            key=(fid,sym,t)
            if key in pending or any(x not in (0,1) for x in signs):
                raise ValueError("Duplicate or invalid forecast")
            pending[key]=tuple(signs)
            forecast_count+=1
            p=next(pvals,None)
    if pending or forecast_count!=settled_count or forecast_count!=len(data):
        raise ValueError("Unsettled or missing M5 observation")
    return {"forecast_events":forecast_count,
            "settlement_events":settled_count,
            "strict_forecast_before_maturity":True,
            "h1_maturity_minutes":5,
            "live_timestamped_prediction_proof":False}


def _score_scope(data:pd.DataFrame)->dict|None:
    if data.empty:
        return None
    ret=data.actual_return_h1.to_numpy(dtype=float)
    nonflat=ret!=0
    truth=ret>0
    n=len(data); k=int(nonflat.sum())
    scored={}
    for name in MODEL_NAMES:
        sign=data[f"{name}_sign"].to_numpy(dtype=int)
        gross=(2*sign-1)*ret*10000.0
        cost_scenarios={}
        for cost in COST_ROUNDTRIP_BPS:
            net=gross-cost
            cost_scenarios[str(int(cost))]={
                "mean_net_proxy_bps_per_signal":float(net.mean()),
                "positive_net_fraction_all_signals":float((net>0).mean()),
            }
        scored[name]={
            "nonflat_accuracy":float(np.mean(sign[nonflat]==truth[nonflat])) if k else None,
            "sign_correct_nonflat":int(np.sum(sign[nonflat]==truth[nonflat])),
            "mean_gross_proxy_bps_per_signal":float(gross.mean()),
            "mean_gross_proxy_bps_correct_direction_nonflat":
                float(gross[nonflat&(sign==truth)].mean()) if np.any(nonflat&(sign==truth)) else None,
            "mean_gross_proxy_bps_wrong_direction_nonflat":
                float(gross[nonflat&(sign!=truth)].mean()) if np.any(nonflat&(sign!=truth)) else None,
            "break_even_hypothetical_roundtrip_cost_bps":float(gross.mean()),
            "cost_scenarios":cost_scenarios,
        }
    paired={}
    for model,ref in COMPARISONS:
        a=data[f"{model}_sign"].to_numpy(dtype=int)
        b=data[f"{ref}_sign"].to_numpy(dtype=int)
        correct_a=nonflat&(a==truth)
        correct_b=nonflat&(b==truth)
        paired[f"{model}_minus_{ref}"]={
            "gross_proxy_difference_bps_per_signal":
                float(((2*a-2*b)*ret*10000.).mean()),
            "direction_accuracy_difference_points":
                100*float((correct_a.sum()-correct_b.sum())/k) if k else None,
            "candidate_only_correct_nonflat":int((correct_a&~correct_b).sum()),
            "reference_only_correct_nonflat":int((correct_b&~correct_a).sum()),
        }
    return {"all_rows":n,"nonflat_rows":k,"flat_rows":n-k,
            "models":scored,"paired":paired}


def confidence_strata(data:pd.DataFrame)->dict:
    ret=data.actual_return_h1.to_numpy(dtype=float)
    output={}
    for name in CANDIDATES:
        p=data[f"{name}_p_up"].to_numpy(dtype=float)
        margin=np.abs(p-.5)
        bands=[]
        for lo,hi in PROBABILITY_MARGIN_BINS:
            mask=margin>=lo
            if hi is not None:
                mask &=margin<hi
            result=_score_scope(data.loc[mask])
            bands.append({"margin_low_inclusive":lo,
                          "margin_high_exclusive":hi,
                          "n":int(mask.sum()),
                          "model_score":result["models"][name] if result else None,
                          "nonflat_rows":result["nonflat_rows"] if result else 0})
        if sum(x["n"] for x in bands)!=len(data):
            raise AssertionError("Confidence bands omitted observations")
        output[name]=bands
    return output


def calculate(data:pd.DataFrame)->dict:
    replay=chronological_gate(data)
    pooled=_score_scope(data)
    by_asset={sym:_score_scope(data.loc[data.symbol.eq(sym)])
              for sym in SEEN_SYMBOLS}
    by_fold={str(fid):_score_scope(data.loc[data.fold_id.eq(fid)])
             for fid in range(36)}
    for name in MODEL_NAMES:
        total=sum(x["nonflat_rows"] for x in by_fold.values())
        hits=sum(x["models"][name]["sign_correct_nonflat"] for x in by_fold.values())
        if (total!=pooled["nonflat_rows"] or
            hits!=pooled["models"][name]["sign_correct_nonflat"]):
            raise AssertionError("Fold-level accuracy fails pooled reconciliation")
    pairs={}
    for model,ref in COMPARISONS:
        key=f"{model}_minus_{ref}"
        values=[x["paired"][key]["gross_proxy_difference_bps_per_signal"] for x in by_fold.values()]
        pairs[key]={
            "pooled":pooled["paired"][key],
            "mean_fold_gross_difference_bps":float(np.mean(values)),
            "positive_folds_gross_proxy":sum(v>0 for v in values),
            "negative_folds_gross_proxy":sum(v<0 for v in values),
            "tied_folds_gross_proxy":sum(v==0 for v in values),
        }
    return {
        "protocol":PROTOCOL,
        "research_status":"EXPLORATORY_OLD_DEVELOPMENT_CHECKS_NOT_INDEPENDENT",
        "execution_assumption":"NON_EXECUTABLE_CLOSE_T_TO_CLOSE_T_PLUS_5M_PROXY",
        "fill_guarantee":False,"actual_trades":False,
        "training_performed":False,
        "no_P0_read_no_ADA_no_fold_OOS_no_final90_no_P3":True,
        "flat_bars_included_in_gross_and_cost_denominator":True,
        "flat_bars_excluded_only_for_directional_accuracy":True,
        "simultaneous_assets_no_portfolio_capital_or_compounding":True,
        "cost_scenarios_roundtrip_bps":list(COST_ROUNDTRIP_BPS),
        "hypothetical_costs_cover":"Illustrative fixed round-trip cost: fees plus spread plus slippage; not exchange quotes",
        "time_replay":replay,"pooled":pooled,"by_asset":by_asset,
        "by_fold":by_fold,"fold_differences":pairs,
        "fixed_confidence_bands":confidence_strata(data),
        "limitations":[
            "Forecasts were generated historically with labeled dev checks, not logged live",
            "A prediction available after candle close t cannot be guaranteed filled at close(t)",
            "M5 OHLCV does not identify executable spread, latency or order-book depth",
            "Cost scenarios are hypothetical fixed roundtrip bps, not historical actual fees",
            "One theoretical roundtrip per asset per M5, all rows, no capital constraints",
            "No position continuity, leverage, funding, liquidation or liquidation costs",
            "No portfolio compounding, fill simulation or exchange order placement",
            "Overlapping historical fold training, multiple prior comparisons, not independent evidence",
        ],
    }


def main()->None:
    parser=argparse.ArgumentParser(description="P2 offline chronological H1 NONEXECUTABLE markout study")
    for flag in ("binary-report","binary-predictions","fold-plan",
                 "h1h6-report","h1h6-predictions","output-json","output-csv"):
        parser.add_argument("--"+flag,required=True,type=Path)
    args=parser.parse_args()
    source_paths=[args.binary_report.resolve(),args.binary_predictions.resolve(),
                  args.fold_plan.resolve(),args.h1h6_report.resolve(),
                  args.h1h6_predictions.resolve()]
    outputs=[args.output_json.resolve(),args.output_csv.resolve()]
    if (len(set(outputs))!=2 or any(p.exists() or p in source_paths for p in outputs)
        or any(p.parent!=args.binary_predictions.resolve().parent for p in outputs)):
        parser.error("Both fresh output paths must be inside same isolated results folder as binary predictions")
    plan_source=json.loads(args.fold_plan.read_text(encoding="utf-8"))
    plan=load_plan(plan_source)
    old=json.loads(args.h1h6_report.read_text(encoding="utf-8"))
    if old.get("source_dataset_fingerprint_sha256")!=plan_source["dataset_fingerprint_sha256"]:
        raise ValueError("Historical P2 fingerprint mismatch")
    original=load_reference(args.h1h6_predictions,old,plan)
    binary_report=json.loads(args.binary_report.read_text(encoding="utf-8"))
    raw=pd.read_parquet(args.binary_predictions)
    print("Validating original 36-fold prediction keys and scores...",flush=True)
    data=validate_inputs(raw,binary_report,original,plan)
    print("Offline time-ordered replay; no orders/no P0/no real fills...",flush=True)
    result=calculate(data)
    result["inputs"]={
        "binary_report":args.binary_report.name,
        "binary_predictions":args.binary_predictions.name,
        "h1h6_report":args.h1h6_report.name,
        "h1h6_predictions":args.h1h6_predictions.name,
        "fold_plan":args.fold_plan.name,
    }
    # No writing into production, P0 or any input; both result files must be new.
    with args.output_json.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    rows=[]
    for name in MODEL_NAMES:
        src=result["pooled"]["models"][name]
        for cost in COST_ROUNDTRIP_BPS:
            cost_str=str(int(cost))
            rows.append({
                "model":name,"roundtrip_cost_bps":int(cost),
                "rows":result["pooled"]["all_rows"],
                "nonflat":result["pooled"]["nonflat_rows"],
                "nonflat_accuracy":src["nonflat_accuracy"],
                "gross_proxy_mean_bps":src["mean_gross_proxy_bps_per_signal"],
                "net_proxy_mean_bps":src["cost_scenarios"][cost_str]["mean_net_proxy_bps_per_signal"],
                "fraction_positive_net":src["cost_scenarios"][cost_str]["positive_net_fraction_all_signals"],
                "EXECUTABLE":False,
            })
    with args.output_csv.open("x",encoding="utf-8") as f:
        pd.DataFrame(rows).to_csv(f,index=False)
    print("Historical non-executable markout:",args.output_json,flush=True)
    print("Fixed-cost scenarios:",args.output_csv,flush=True)
    for name in ("binary_30","logistic_ema2","lightgbm_30","always_down"):
        x=result["pooled"]["models"][name]
        print(name,"accuracy=",round(x["nonflat_accuracy"]*100,4),
              "gross proxy bps=",round(x["mean_gross_proxy_bps_per_signal"],5),
              "5bps net proxy=",round(x["cost_scenarios"]["5"]["mean_net_proxy_bps_per_signal"],5),
              flush=True)


if __name__=="__main__":
    main()
