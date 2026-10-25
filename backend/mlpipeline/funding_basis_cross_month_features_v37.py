"""P4 V37: offline cross-month causal feature continuity, diagnostics only.

Requires the original immutable V33 freeze, the V35 calendar and the V36
feature diagnostics. Uses only Oct 2024--May 2026 development market candles.
Never reads June 2026 market partitions, July-Sep held-out history, funding
rates, labels, training examples, predictions or PnL. No feature-row exports.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mlpipeline.funding_basis_unified_dev_reader_v34 import (
    APPROVED_FREEZE_SHA256, MARKET_FIELDS, M5,
    UnifiedDevelopmentReader, sha_file,
)
from mlpipeline.funding_basis_causal_native_features_v36 import (
    PROTOCOL as V36_PROTOCOL,
    FEATURE_SCHEMA_VERSION, FEATURE_COLUMNS,
    ELIGIBLE_MONTHS, LOOKBACK_BARS,
    INTERNAL_RESERVE_START,
    build_causal_features, validate_contract, verify_v35_report,
)
from mlpipeline.pullback_long_history_v3 import SYMBOLS

PROTOCOL = "p4_v37_cross_month_causal_feature_continuity_audit_v1"
RESULT_STATUS = "CROSS_MONTH_CAUSAL_CONTINUITY_PASS_ML_AND_FUNDING_BLOCKED"
SOURCE_MONTHS = 20
SOURCE_BARS_PER_ASSET = 175103
V36_PER_MONTH_COMPLETE = 174143
EXPECTED_STITCHED_COMPLETE = 175055
EXPECTED_GAIN = 912
V36_STATUS = "CAUSAL_NATIVE_FEATURE_DIAGNOSTICS_PASS_FUNDING_AND_ML_BLOCKED"
EXPECTED_FEATURES = 19


def _hash_matches(path: Path, expected_sha: str, description: str):
    path = path.resolve()
    if (not isinstance(expected_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}",expected_sha) is None
            or not path.is_file() or sha_file(path) != expected_sha):
        raise ValueError(description+" SHA-256 mismatch or missing")
    return path


def verify_v36_report(v36_path: Path, sha: str, *, v35_sha: str):
    path = _hash_matches(v36_path, sha, "V36 report")
    r = json.loads(path.read_bytes())
    if (r.get("protocol") != V36_PROTOCOL
            or r.get("status") != V36_STATUS
            or r.get("v33_freeze_sha256") != APPROVED_FREEZE_SHA256
            or r.get("v35_report_sha256") != v35_sha
            or r.get("frozen_folds") != 33
            or r.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
            or r.get("feature_columns") != list(FEATURE_COLUMNS)
            or r.get("feature_count") != EXPECTED_FEATURES
            or r.get("source_partitions_used_for_feature_diagnostics") != 80
            or r.get("funding_features_status") != "EXCLUDED_NOT_ASOF_ATTESTED"
            or r.get("no_network_no_holdout_access_no_labels_no_training_no_backtest_no_pnl_no_orders") is not True):
        raise ValueError("V36 feature/provenance/holdout contract mismatch")
    by_symbol = r.get("by_symbol", {})
    if set(by_symbol) != set(SYMBOLS):
        raise ValueError("V36 source universe unexpectedly changed")
    for symbol, data in by_symbol.items():
        if (data.get("native_m5_before_internal_blind") != SOURCE_BARS_PER_ASSET
                or data.get("all_features_complete_rows") != V36_PER_MONTH_COMPLETE
                or data.get("ineligible_rows") != SOURCE_BARS_PER_ASSET-V36_PER_MONTH_COMPLETE):
            raise ValueError("V36 per-month feature counts changed for "+symbol)
    return {"sha256":sha, "report":r}


def stitch_verified_months(month_frames, *, verify_prefix=True):
    """Stitch already verified native month frames without opening other paths.

    Return metadata and per-month diagnostics. Full feature data remain in RAM
    and are never returned or written. Future-suffix invariance is checked at
    exact month boundaries, so no later candle can revise an earlier feature.
    """
    validate_contract()
    if not month_frames:
        raise ValueError("No verified months to stitch")
    offsets = []
    frames = []
    next_open = None
    total = 0
    for month, frame in month_frames:
        if (month not in ELIGIBLE_MONTHS or
                not isinstance(frame,pd.DataFrame) or
                list(frame.columns) != list(MARKET_FIELDS) or frame.empty):
            raise ValueError("Invalid development month or market schema")
        opens = pd.to_datetime(frame.bar_open_at,utc=True)
        closes = pd.to_datetime(frame.bar_close_at,utc=True)
        if (opens.isna().any() or closes.isna().any()
                or not opens.dt.strftime("%Y-%m").eq(month).all()
                or not closes.eq(opens + M5).all()
                or opens.duplicated().any()
                or not opens.diff().iloc[1:].eq(M5).all()
                or closes.ge(INTERNAL_RESERVE_START).any()):
            raise ValueError("Noncausal month, incomplete grid or blind access")
        if next_open is not None and opens.iloc[0] != next_open:
            raise ValueError("M5 gap/overlap across month boundary")
        next_open = closes.iloc[-1]
        offsets.append((month,total,len(frame)))
        total+=len(frame)
        frames.append(frame)
    joined = pd.concat(frames,ignore_index=True)
    if len(joined) != total:
        raise AssertionError("Stitching lost market rows")
    whole = build_causal_features(joined)
    complete = np.isfinite(whole[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
    if len(complete) != len(joined):
        raise AssertionError("Feature/source size mismatch")
    by_month = []
    for month,start,n in offsets:
        month_frame = joined.iloc[start:start+n].reset_index(drop=True)
        local = build_causal_features(month_frame)
        local_ok = np.isfinite(local[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
        stitched_ok = complete[start:start+n]
        # Stitching may replace monthly warm-up NaNs, never create new invalid
        # values in a month that previously had complete local native features.
        if np.any(local_ok & ~stitched_ok):
            raise ValueError("Cross-month feature stitching lost valid rows")
        by_month.append({
            "month":month,
            "native_closed_m5":n,
            "monthly_reset_complete":int(local_ok.sum()),
            "stitched_complete":int(stitched_ok.sum()),
            "newly_available_from_native_carryover":int(stitched_ok.sum()-local_ok.sum()),
        })
    if verify_prefix and len(frames) >= 3:
        # Distinct early and middle month boundaries. The full future suffix
        # must not alter any past feature, including EMA/rolling calculations.
        boundaries={offsets[1][1], offsets[len(offsets)//2][1]}
        for boundary in sorted(boundaries):
            if boundary < LOOKBACK_BARS+1 or boundary >= len(joined):
                raise AssertionError("Bad prefix test boundary")
            past=build_causal_features(joined.iloc[:boundary].reset_index(drop=True))
            observed=whole.iloc[:boundary].reset_index(drop=True)
            pd.testing.assert_frame_equal(past,observed,check_exact=True)
    return {
        "native_closed_m5":len(joined),
        "monthly_reset_complete":sum(x["monthly_reset_complete"] for x in by_month),
        "stitched_complete":int(complete.sum()),
        "newly_available_from_native_carryover":
            int(complete.sum())-sum(x["monthly_reset_complete"] for x in by_month),
        "first_close_utc":joined.bar_close_at.iloc[0].isoformat(),
        "last_close_utc":joined.bar_close_at.iloc[-1].isoformat(),
        "prefix_invariance_boundaries_checked":2 if verify_prefix and len(frames)>=3 else 0,
    }, by_month


def run(freeze_path: Path, freeze_sha: str,
        v35_path: Path, v35_sha: str,
        v36_path: Path, v36_sha: str,
        output_json: Path, output_csv: Path):
    output_json, output_csv = output_json.resolve(), output_csv.resolve()
    if (output_json == output_csv or output_json.exists() or output_csv.exists()
            or not output_json.parent.is_dir() or not output_csv.parent.is_dir()):
        raise ValueError("Require two distinct NEW metadata report paths")
    if freeze_sha != APPROVED_FREEZE_SHA256:
        raise ValueError("Unapproved frozen provenance for V37")
    validate_contract()
    verify_v35_report(v35_path,v35_sha)
    approved_v36 = verify_v36_report(v36_path,v36_sha,v35_sha=v35_sha)
    reader = UnifiedDevelopmentReader.preflight(freeze_path,freeze_sha)
    roots = reader._roots
    for path in (output_json,output_csv):
        if (path in (freeze_path.resolve(),v35_path.resolve(),v36_path.resolve())
                or any(path==root or path.is_relative_to(root) for root in roots[:4])):
            raise ValueError("V37 output overlaps immutable source or provenance")
    rows=[]
    summary={}
    for symbol in SYMBOLS:
        month_frames=[]
        for month in ELIGIBLE_MONTHS:
            market, _unused_funding, status = reader.read_month(symbol,month)
            if status not in ("VERIFIED_V21","VERIFIED_V23"):
                raise ValueError("Unexpected V33 source in pre-June feature development")
            if month == ELIGIBLE_MONTHS[-1]:
                # The May 31 23:55 candle closes precisely at June 1 00:00.
                # Verify its existence via V34, then EXCLUDE before features.
                at_border = market.bar_close_at.ge(INTERNAL_RESERVE_START)
                if (int(at_border.sum())!=1
                        or market.bar_close_at.iloc[-1]!=INTERNAL_RESERVE_START):
                    raise ValueError("Missing/changed May-June blind cutoff candle")
                market=market.loc[~at_border].copy().reset_index(drop=True)
            if market.empty or market.bar_close_at.ge(INTERNAL_RESERVE_START).any():
                raise ValueError("Source overlaps internal June blind")
            month_frames.append((month,market))
        counts,per_month=stitch_verified_months(month_frames)
        expected=approved_v36["report"]["by_symbol"][symbol]
        if (len(per_month)!=SOURCE_MONTHS
                or counts["native_closed_m5"]!=SOURCE_BARS_PER_ASSET
                or counts["monthly_reset_complete"]!=expected["all_features_complete_rows"]
                or counts["stitched_complete"]!=EXPECTED_STITCHED_COMPLETE
                or counts["newly_available_from_native_carryover"]!=EXPECTED_GAIN
                or counts["prefix_invariance_boundaries_checked"]!=2
                or counts["last_close_utc"]!="2026-05-31T23:55:00+00:00"
                or per_month[0]["newly_available_from_native_carryover"]!=0
                or any(x["newly_available_from_native_carryover"]!=LOOKBACK_BARS
                       for x in per_month[1:])):
            raise ValueError("Stitched feature counts/continuity diverge from frozen V36")
        summary[symbol]=counts
        rows.extend({"symbol":symbol,**row} for row in per_month)
    if len(rows)!=80 or sum(x["newly_available_from_native_carryover"] for x in rows)!=4*EXPECTED_GAIN:
        raise ValueError("Cross-symbol V37 feature inventory incomplete")
    result={
        "protocol":PROTOCOL,
        "v33_freeze_sha256":freeze_sha,
        "v35_report_sha256":v35_sha,
        "v36_report_sha256":v36_sha,
        "feature_schema_version":FEATURE_SCHEMA_VERSION,
        "feature_columns":list(FEATURE_COLUMNS),
        "native_market_months_per_symbol":SOURCE_MONTHS,
        "source_partitions_read":len(rows),
        "by_symbol":summary,
        "total_additional_causally_complete_rows":4*EXPECTED_GAIN,
        "no_june_2026_features":True,
        "no_external_holdout_access":True,
        "funding_features_status":"EXCLUDED_NOT_ASOF_ATTESTED",
        "labels_and_training_status":"BLOCKED_NOT_PERFORMED",
        "feature_storage":"IN_MEMORY_ONLY_METADATA_REPORTS",
        "status":RESULT_STATUS,
        "no_network_no_labels_no_training_no_backtest_no_pnl_no_orders":True,
    }
    with output_csv.open("x",encoding="utf-8",newline="") as f:
        pd.DataFrame(rows).to_csv(f,index=False)
    with output_json.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2)
    return result


def main():
    parser=argparse.ArgumentParser(description="V37 full-month causal feature continuity audit, NO ML")
    for name in ("freeze-json","freeze-sha256","v35-json","v35-sha256",
                 "v36-json","v36-sha256","output-json","output-csv"):
        parser.add_argument("--"+name,required=True,
                            type=Path if name.endswith("json") else str)
    a=parser.parse_args()
    report=run(a.freeze_json,a.freeze_sha256,a.v35_json,a.v35_sha256,
               a.v36_json,a.v36_sha256,a.output_json,a.output_csv)
    print("V37",report["status"],"rows regained",
          report["total_additional_causally_complete_rows"],flush=True)
    print("NO JUN 2026 / NO OOS / NO FUNDING FEATURES / NO LABELS / NO ML / NO PNL",flush=True)


if __name__=="__main__":
    main()
