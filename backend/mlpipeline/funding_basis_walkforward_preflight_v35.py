"""P4 V35: immutable offline walk-forward *methodology* preflight.

Reads the V33 provenance freeze through the verified V34 reader, but NEVER
builds labels, feature matrices, forecasts, orders, a model, or PnL.
The calendar is preregistered; no sampling from or touching reserved 2026-07+.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mlpipeline.features import FEATURE_SCHEMA_VERSION
from mlpipeline.labels import LABEL_SCHEMA_VERSION
from mlpipeline.pullback_long_history_v3 import SYMBOLS, EXPECTED_MONTHS, CUTOFF
from mlpipeline.funding_basis_unified_dev_reader_v34 import (
    APPROVED_FREEZE_SHA256, BEGIN, MARKET_FIELDS, FUNDING_FIELDS, M5,
    UnifiedDevelopmentReader, readiness,
)

PROTOCOL = "p4_v35_frozen_walkforward_methodology_preflight_v1"

# Fixed research design only, NOT the legacy ADA unseen-symbol experiment.
TRAIN_DAYS = 120
VALIDATION_DAYS = 14
TEST_DAYS = 14
STEP_DAYS = 14
LABEL_HORIZON_BARS = 6  # labels.py defaults: 30 minutes
EMBARGO_BARS = 6        # conservative additional 30-minute block on val/test starts
INTERNAL_RESERVE_START = pd.Timestamp("2026-06-01T00:00:00Z")
M5_PER_DAY = 288
REQUIRED_LEGACY_FEATURE_INPUTS = frozenset({
    "bar_close_at", "open", "high", "low", "close",
    "volume_base", "volume_quote", "trade_count", "taker_buy_volume_base",
})
# The V34 unified market data is a different schema (spot_/perp_/mark_);
# it does NOT provide quote_volume/trade_count/taker_buy activity.
MISSING_LEGACY_FEATURE_INPUTS = (
    "volume_quote", "trade_count", "taker_buy_volume_base",
)
FUNDING_PUBLICATION_TIMESTAMP_FIELD = "funding_published_at"


def validate_design() -> None:
    if (len(SYMBOLS) != 4 or len(EXPECTED_MONTHS) != 21
            or CUTOFF != pd.Timestamp("2026-07-01T00:00:00Z")
            or INTERNAL_RESERVE_START != pd.Timestamp("2026-06-01T00:00:00Z")
            or BEGIN != pd.Timestamp("2024-10-01T00:00:00Z")
            or TRAIN_DAYS < 1 or VALIDATION_DAYS < 1 or TEST_DAYS < 1
            or STEP_DAYS < TEST_DAYS or LABEL_HORIZON_BARS < 1
            or EMBARGO_BARS < LABEL_HORIZON_BARS
            or M5 != pd.Timedelta(minutes=5)
            or LABEL_SCHEMA_VERSION != "vol_scaled_v1"
            or FEATURE_SCHEMA_VERSION != "features_v1"):
        raise ValueError("Frozen V35 research calendar/feature/label contract changed")
    if not BEGIN < INTERNAL_RESERVE_START < CUTOFF:
        raise ValueError("Invalid reserved development and OOS boundaries")


def _utc(ts: pd.Timestamp) -> pd.Timestamp:
    if not isinstance(ts, pd.Timestamp) or ts.tzinfo is None or str(ts.tz) != "UTC":
        raise ValueError("Require explicit UTC timestamp")
    return ts


def _expected_closes(begin: pd.Timestamp, end: pd.Timestamp, *, embargo_bars: int):
    """Only theoretical sample-time geometry; NEVER creates labels or touches candles.

    Timestamp t denotes CLOSED M5 bar. Its hypothetical label would mature
    t + LABEL_HORIZON_BARS*M5. Strictly require maturity < end.
    """
    begin, end = _utc(begin), _utc(end)
    if embargo_bars < 0 or end <= begin:
        raise ValueError("Invalid embargo/window interval")
    first = begin + max(1, embargo_bars) * M5
    exclusive_last = end - LABEL_HORIZON_BARS * M5
    if first >= exclusive_last:
        raise ValueError("Empty causally eligible window")
    if (first - BEGIN) % M5 != pd.Timedelta(0):
        raise ValueError("Sample close off M5 grid")
    if (exclusive_last - first) % M5 != pd.Timedelta(0):
        raise ValueError("Window endpoint off M5 grid")
    count = int((exclusive_last - first) // M5)
    final = first + (count - 1) * M5
    if count < 1 or final + LABEL_HORIZON_BARS*M5 >= end:
        raise AssertionError("Potential label maturity leakage")
    return {
        "first_candidate_close_utc": first.isoformat(),
        "last_candidate_close_utc": final.isoformat(),
        "theoretical_upper_bound_per_symbol": count,
        "last_label_maturity_utc": (final + LABEL_HORIZON_BARS*M5).isoformat(),
    }


def make_calendar_folds():
    """Expanding NO, rolling 120-day train; 14-day val/test, 14-day test step."""
    validate_design()
    folds = []
    train_start = BEGIN
    previous_test_end = None
    for i in range(1000):
        validation_start = train_start + pd.Timedelta(days=TRAIN_DAYS)
        test_start = validation_start + pd.Timedelta(days=VALIDATION_DAYS)
        test_end = test_start + pd.Timedelta(days=TEST_DAYS)
        if test_end > INTERNAL_RESERVE_START:
            break
        if previous_test_end is not None and test_start < previous_test_end:
            raise AssertionError("Overlapping development test intervals")
        windows = {
            "train": _expected_closes(
                train_start, validation_start, embargo_bars=0
            ),
            "validation": _expected_closes(
                validation_start, test_start, embargo_bars=EMBARGO_BARS
            ),
            "test": _expected_closes(
                test_start, test_end, embargo_bars=EMBARGO_BARS
            ),
        }
        if any(pd.Timestamp(x["last_label_maturity_utc"]) >= INTERNAL_RESERVE_START
               for x in windows.values()):
            raise AssertionError("June 2026 development reserve contaminated")
        folds.append({
            "fold": i,
            "train_start_utc": train_start.isoformat(),
            "train_end_exclusive_utc": validation_start.isoformat(),
            "validation_start_utc": validation_start.isoformat(),
            "validation_end_exclusive_utc": test_start.isoformat(),
            "test_start_utc": test_start.isoformat(),
            "test_end_exclusive_utc": test_end.isoformat(),
            "windows": windows,
        })
        previous_test_end = test_end
        train_start += pd.Timedelta(days=STEP_DAYS)
    if not folds or len(folds) >= 1000:
        raise ValueError("No valid finite preregistered development folds")
    return folds


def compatibility_report():
    # Predeclare the distinction; never claim the V34 market is the old P0 schema.
    source_columns = set(MARKET_FIELDS) | set(FUNDING_FIELDS)
    if any(x in source_columns for x in MISSING_LEGACY_FEATURE_INPUTS):
        raise ValueError("Legacy feature dependency assumption no longer true")
    if FUNDING_PUBLICATION_TIMESTAMP_FIELD in source_columns:
        raise ValueError("Funding publication-time contract needs a new review")
    return {
        "legacy_feature_schema": FEATURE_SCHEMA_VERSION,
        "legacy_label_schema": LABEL_SCHEMA_VERSION,
        "feature_matrix_status": "BLOCKED_ADAPTER_AND_CAUSALITY_REVIEW_REQUIRED",
        "missing_legacy_feature_inputs": list(MISSING_LEGACY_FEATURE_INPUTS),
        "funding_asof_status": "BLOCKED_PUBLICATION_TIMESTAMP_NOT_ATTESTED",
        "no_feature_or_label_creation": True,
        "reason": (
            "V34 markets expose spot/perp/mark M5 OHLC and base volumes; "
            "the existing features_v1 pipeline expects P0 quote volume, "
            "trade count and taker-buy fields. Funding event time alone does "
            "not certify when the funding value was available to a model."
        ),
    }


def schedule_report(folds):
    if not folds:
        raise ValueError("Empty folds")
    out = []
    for f in folds:
        for subset in ("train", "validation", "test"):
            out.append({
                "fold": f["fold"],
                "subset": subset,
                "window_start_utc": (
                    f["train_start_utc"] if subset == "train" else
                    f["validation_start_utc"] if subset == "validation" else
                    f["test_start_utc"]
                ),
                "window_end_exclusive_utc": (
                    f["train_end_exclusive_utc"] if subset == "train" else
                    f["validation_end_exclusive_utc"] if subset == "validation" else
                    f["test_end_exclusive_utc"]
                ),
                **f["windows"][subset],
            })
    return out


def run(freeze: Path, freeze_sha256: str,
        output_json: Path, output_csv: Path):
    output_json, output_csv = output_json.resolve(), output_csv.resolve()
    if (output_json == output_csv or output_json.exists() or output_csv.exists()
            or not output_json.parent.is_dir() or not output_csv.parent.is_dir()):
        raise ValueError("Require two distinct NEW report outputs")
    if freeze_sha256 != APPROVED_FREEZE_SHA256:
        raise ValueError("V35 freeze differs from approved V33 snapshot")
    validate_design()
    folds = make_calendar_folds()
    compatibility = compatibility_report()
    reader = UnifiedDevelopmentReader.preflight(freeze, freeze_sha256)
    roots = reader._roots
    for path in (output_json,output_csv):
        if any(path == root or path.is_relative_to(root) for root in roots[:4]):
            raise ValueError("V35 reports must be outside immutable source roots")
        if path == freeze.resolve():
            raise ValueError("Refuse overwrite of provenance freeze")
    dev_rows = readiness(reader)
    if len(dev_rows) != 84:
        raise ValueError("V34 returned an incomplete development calendar")
    by_symbol = {}
    for symbol in SYMBOLS:
        rows = [r for r in dev_rows if r["symbol"] == symbol]
        if (len(rows) != len(EXPECTED_MONTHS)
                or [r["month"] for r in rows] != list(EXPECTED_MONTHS)
                or sum(r["market_m5_dev"] for r in rows) != 183743
                or sum(r["excluded_cutoff_bar"] for r in rows) != 1):
            raise ValueError("V34 time geometry differs from approved full development history")
        by_symbol[symbol] = {
            "monthly_partitions": len(rows),
            "closed_m5_before_external_holdout": sum(r["market_m5_dev"] for r in rows),
            "funding_events_total": sum(r["funding_events_month"] for r in rows),
        }
    if any(f["test_end_exclusive_utc"] > INTERNAL_RESERVE_START.isoformat()
           for f in folds):
        raise ValueError("A fold uses internal blind June reserve")
    result = {
        "protocol": PROTOCOL,
        "source_provenance_sha256": freeze_sha256,
        "universe": list(SYMBOLS),
        "calendar": {"start_utc": BEGIN.isoformat(),
                     "development_end_exclusive_utc": CUTOFF.isoformat(),
                     "internal_blind_start_utc": INTERNAL_RESERVE_START.isoformat(),
                     "external_holdout_start_utc": CUTOFF.isoformat()},
        "policy": {"train_days": TRAIN_DAYS, "validation_days": VALIDATION_DAYS,
                   "test_days": TEST_DAYS, "step_days": STEP_DAYS,
                   "label_horizon_bars": LABEL_HORIZON_BARS,
                   "embargo_bars_after_validation_and_test_starts": EMBARGO_BARS,
                   "purging": "label_maturity_strictly_before_next_window_boundary",
                   "rolling_train": True,
                   "internal_june_blind_reserved": True},
        "fold_count": len(folds),
        "first_fold": {k:v for k,v in folds[0].items() if k!="windows"},
        "last_fold": {k:v for k,v in folds[-1].items() if k!="windows"},
        "source_data": by_symbol,
        "compatibility": compatibility,
        "status": "CALENDAR_PREFLIGHT_PASS_ML_PIPELINE_BLOCKED",
        "no_network_no_holdout_access_no_training_no_labels_no_pnl_no_orders": True,
        "no_price_export_no_imputation": True,
    }
    report_rows = schedule_report(folds)
    # Atomic-ish non-overwrite creation; failures before writing emit no audit files.
    with output_csv.open("x",encoding="utf-8",newline="") as output:
        pd.DataFrame(report_rows).to_csv(output,index=False)
    with output_json.open("x",encoding="utf-8") as output:
        json.dump(result,output,indent=2)
    return result


def main():
    p = argparse.ArgumentParser(description="V35 sealed walk-forward methodology preflight ONLY")
    p.add_argument("--freeze-json",type=Path,required=True)
    p.add_argument("--freeze-sha256",required=True)
    p.add_argument("--output-json",type=Path,required=True)
    p.add_argument("--output-csv",type=Path,required=True)
    args = p.parse_args()
    output=run(args.freeze_json,args.freeze_sha256,args.output_json,args.output_csv)
    print("V35",output["status"],"folds",output["fold_count"],flush=True)
    print("DEV ONLY / JUNE INTERNAL RESERVE / JUL-SEP EXTERNAL HOLDOUT SEALED",flush=True)
    print("NO NETWORK / NO TRAINING / NO FEATURES / NO LABELS / NO PNL / NO ORDERS",flush=True)


if __name__=="__main__":
    main()
