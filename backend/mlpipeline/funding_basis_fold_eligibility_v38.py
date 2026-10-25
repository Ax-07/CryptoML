"""V38: sealed, metadata-only theoretical fold eligibility; NO ML or data reads.

Reads only SHA-pinned V35/V37 JSON reports. Builds no targets, feature
matrices, predictions, or price rows; never accesses any market partition.
Counts are per-fold upper bounds, not unique labeled examples.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import pandas as pd
from mlpipeline.funding_basis_unified_dev_reader_v34 import sha_file, APPROVED_FREEZE_SHA256, BEGIN
from mlpipeline.funding_basis_walkforward_preflight_v35 import (
    PROTOCOL as V35_PROTOCOL, V35_POLICY, make_calendar_folds,
    INTERNAL_RESERVE_START, CUTOFF, LABEL_HORIZON_BARS, EMBARGO_BARS, M5,
)
from mlpipeline.funding_basis_cross_month_features_v37 import (
    PROTOCOL as V37_PROTOCOL, RESULT_STATUS as V37_STATUS,
    FEATURE_SCHEMA_VERSION, EXPECTED_STITCHED_COMPLETE,
)
from mlpipeline.pullback_long_history_v3 import SYMBOLS

PROTOCOL="p4_v38_sealed_theoretical_fold_eligibility_v1"
FIRST_FULL_FEATURE_CLOSE=BEGIN+49*M5
EXPECTED_FOLDS=33


def read_sha_json(path: Path, digest: str, expected_protocol: str):
    if not re.fullmatch(r"[0-9a-f]{64}",digest):
        raise ValueError("Malformed frozen report SHA")
    if not path.is_file() or sha_file(path)!=digest:
        raise ValueError("Frozen report missing/changed")
    obj=json.loads(path.read_bytes())
    if obj.get("protocol")!=expected_protocol:
        raise ValueError("Frozen report protocol differs")
    return obj


def eligibility(folds):
    if len(folds)!=EXPECTED_FOLDS: raise ValueError("Unexpected fold count")
    rows=[]
    first=FIRST_FULL_FEATURE_CLOSE
    for fold in folds:
        for subset in ("train","validation","test"):
            entry=fold["windows"][subset]
            t0=pd.Timestamp(entry["first_candidate_close_utc"])
            t1=pd.Timestamp(entry["last_candidate_close_utc"])
            end=pd.Timestamp(fold[
                {"train":"train_end_exclusive_utc",
                 "validation":"validation_end_exclusive_utc",
                 "test":"test_end_exclusive_utc"}[subset]])
            start=max(t0,first)
            count=0 if start>t1 else int((t1-start)//M5)+1
            if (count < 1 or t1+LABEL_HORIZON_BARS*M5>=end
                    or end>INTERNAL_RESERVE_START
                    or count>entry["theoretical_upper_bound_per_symbol"]):
                raise ValueError("Invalid fold eligibility or label maturity")
            rows.append({
                "fold":fold["fold"],"subset":subset,
                "first_eligible_close_utc":start.isoformat(),
                "last_eligible_close_utc":t1.isoformat(),
                "theoretical_upper_bound_per_symbol":count,
                "theoretical_upper_bound_all_four_symbols":count*len(SYMBOLS),
                "last_hypothetical_label_maturity_utc":(t1+LABEL_HORIZON_BARS*M5).isoformat(),
            })
    return rows


def run(v35_path:Path,v35_sha:str,v37_path:Path,v37_sha:str,
        json_out:Path,csv_out:Path):
    a,b=json_out.resolve(),csv_out.resolve()
    if a==b or a.exists() or b.exists() or not a.parent.is_dir() or not b.parent.is_dir():
        raise ValueError("Require two distinct NEW metadata reports")
    v35=read_sha_json(v35_path,v35_sha,V35_PROTOCOL)
    v37=read_sha_json(v37_path,v37_sha,V37_PROTOCOL)
    if (v35.get("source_provenance_sha256")!=APPROVED_FREEZE_SHA256
            or v35.get("status")!="CALENDAR_PREFLIGHT_PASS_ML_PIPELINE_BLOCKED"
            or v35.get("fold_count")!=EXPECTED_FOLDS
            or v35.get("policy")!=V35_POLICY
            or v37.get("v35_report_sha256")!=v35_sha
            or v37.get("v33_freeze_sha256")!=APPROVED_FREEZE_SHA256
            or v37.get("status")!=V37_STATUS
            or v37.get("feature_schema_version")!=FEATURE_SCHEMA_VERSION
            or v37.get("source_partitions_read")!=80
            or v37.get("no_june_2026_features") is not True
            or v37.get("no_external_holdout_access") is not True
            or set(v37.get("by_symbol",{}))!=set(SYMBOLS)
            or any(v37["by_symbol"][s].get("stitched_complete")!=EXPECTED_STITCHED_COMPLETE for s in SYMBOLS)):
        raise ValueError("Frozen V35/V37 contract mismatch")
    if a in (v35_path.resolve(),v37_path.resolve()) or b in (v35_path.resolve(),v37_path.resolve()):
        raise ValueError("Cannot overwrite evidence")
    folds=make_calendar_folds()
    rows=eligibility(folds)
    if len(rows)!=EXPECTED_FOLDS*3:
        raise ValueError("Unexpected fold schedule")
    counts={subset:sum(r["theoretical_upper_bound_all_four_symbols"] for r in rows if r["subset"]==subset)
            for subset in ("train","validation","test")}
    if counts!={"train":4560996,"validation":530640,"test":530640}:
        raise ValueError("Theoretical bounds differ from frozen calendar")
    report={
        "protocol":PROTOCOL,"v35_report_sha256":v35_sha,"v37_report_sha256":v37_sha,
        "v33_freeze_sha256":APPROVED_FREEZE_SHA256,"universe":list(SYMBOLS),
        "fold_count":EXPECTED_FOLDS,"feature_schema_version":FEATURE_SCHEMA_VERSION,
        "label_horizon_bars_theoretical_only":LABEL_HORIZON_BARS,
        "embargo_bars":EMBARGO_BARS,"earliest_fully_causal_feature_close_utc":FIRST_FULL_FEATURE_CLOSE.isoformat(),
        "internal_blind_start_utc":INTERNAL_RESERVE_START.isoformat(),
        "external_holdout_start_utc":CUTOFF.isoformat(),
        "fold_sample_upper_bound_counts_not_unique":counts,
        "interpretation":"Overlapping rolling train windows: do NOT sum as unique examples. No actual labels built.",
        "status":"THEORETICAL_FOLD_ELIGIBILITY_PASS_LABELS_AND_ML_BLOCKED",
        "no_market_data_read_no_holdout_no_labels_no_training_no_pnl_no_orders":True,
    }
    with b.open("x",encoding="utf-8",newline="") as f:pd.DataFrame(rows).to_csv(f,index=False)
    with a.open("x",encoding="utf-8") as f:json.dump(report,f,indent=2)
    return report


def main():
    p=argparse.ArgumentParser(description="V38 sealed fold metadata audit (no market reads)")
    for arg in ("v35-json","v35-sha256","v37-json","v37-sha256","output-json","output-csv"):
        p.add_argument("--"+arg,required=True,
                       type=Path if arg.endswith("json") or arg=="output-csv" else str)
    x=p.parse_args()
    result=run(x.v35_json,x.v35_sha256,x.v37_json,x.v37_sha256,x.output_json,x.output_csv)
    print("V38",result["status"],"folds",result["fold_count"])
    print("NO MARKET FILES / NO HOLDOUT / NO LABELS / NO ML / NO PNL")

if __name__=="__main__":
    main()
