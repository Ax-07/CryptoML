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
            if (got["trades"]!=old["trades"] or
                abs(got["by_cost"]["2"]["net_bps_per_trade"]-
                    old["by_cost"]["2"]["net_bps_per_trade"])>1e-10
                if got["trades"] else old["trades"]!=0):
                raise ValueError("Asset summary differs")
    return frame.sort_values(["fold_id","symbol","bar_close_at"]).reset_index(drop=True)
