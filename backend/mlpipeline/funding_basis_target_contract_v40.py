"""P4 V40: approved causal target CONTRACT + pure in-memory synthetic label tests.

CLI ONLY reads V39 normative markdown and a V38 metadata report, verifies
their provenance, and emits ONE JSON approval audit. It cannot open source
market files, the June 2026 internal reserve, or Jul-Sep external holdout.
No historical labels, training, predictions, PnL or orders.

The in-memory label function exists for synthetic regression fixtures only.
It is NOT connected to any production dataset builder or trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mlpipeline.funding_basis_causal_native_features_v36 import (
    FEATURE_SCHEMA_VERSION, build_causal_features,
)
from mlpipeline.funding_basis_unified_dev_reader_v34 import (
    APPROVED_FREEZE_SHA256, BEGIN, M5, MARKET_FIELDS, sha_file,
)
from mlpipeline.funding_basis_fold_eligibility_v38 import (
    PROTOCOL as V38_PROTOCOL,
)
from mlpipeline.funding_basis_walkforward_preflight_v35 import (
    LABEL_HORIZON_BARS, EMBARGO_BARS, INTERNAL_RESERVE_START,
)
from mlpipeline.pullback_long_history_v3 import CUTOFF, SYMBOLS

PROTOCOL = "p4_v40_approved_directional_target_synthetic_only_v1"
TARGET_SCHEMA_VERSION = "p4_directional_perp_close_30m_v40_v1"
TARGET_STATUS = "TARGET_CONTRACT_APPROVED_SYNTHETIC_ONLY_HISTORICAL_LABELS_BLOCKED"
APPROVED_V39_BLOB_SHA1 = "48445865b9ae7c19f25be0927a375e5188374d32"
V39_DOCUMENT_NAME = "135_P4_V39_PROPOSED_CAUSAL_TARGET_CONTRACT.md"

# Verified in the user's V38 output; these are provenance anchors, NOT
# a claim that the yet-unseen V38 JSON file has any particular SHA-256.
PINNED_V35_REPORT_SHA256 = "ad478769e6b3d58f3df7143fb8d5f917937a3f10eba090f5747700a2cd8e2abe"
PINNED_V37_REPORT_SHA256 = "33308282142445e912faa1a8ac215e38e0e9764d5ff97f7e64859410f1235d08"
EXPECTED_V38_COUNTS = {"train": 4560804, "validation": 530640, "test": 530640}

REFERENCE_MARKET = "perp"
REFERENCE_PRICE_COLUMN = "perp_close"
REFERENCE_VOLATILITY_COLUMN = "perp_volatility_48"
HORIZON_BARS = 6
VOLATILITY_LOOKBACK_RETURNS = 48
THRESHOLD_MULTIPLIER = 0.5
DOWN, NEUTRAL, UP = 0, 1, 2
TARGET_FIELDS = ("bar_close_at", "label_mature_at", "target")


def validate_frozen_policy():
    if (
        HORIZON_BARS != LABEL_HORIZON_BARS
        or HORIZON_BARS != 6
        or EMBARGO_BARS != 6
        or VOLATILITY_LOOKBACK_RETURNS != 48
        or THRESHOLD_MULTIPLIER != 0.5
        or REFERENCE_PRICE_COLUMN != "perp_close"
        or REFERENCE_VOLATILITY_COLUMN != "perp_volatility_48"
        or FEATURE_SCHEMA_VERSION != "p4_native_closed_m5_features_v36_v1"
        or len(SYMBOLS) != 4
        or INTERNAL_RESERVE_START != pd.Timestamp("2026-06-01T00:00:00Z")
        or CUTOFF != pd.Timestamp("2026-07-01T00:00:00Z")
    ):
        raise ValueError("V40 preregistered V35/V36/V39 target contract changed")


def git_blob_sha1(blob: bytes) -> str:
    """Validate the exact normative V39 Git blob, not only its display name."""
    header = b"blob " + str(len(blob)).encode("ascii") + b"\0"
    return hashlib.sha1(header + blob).hexdigest()


def verify_normative_v39(document: Path) -> str:
    validate_frozen_policy()
    path = document.resolve()
    if path.name != V39_DOCUMENT_NAME or not path.is_file():
        raise ValueError("Expected approved V39 normative document")
    blob = path.read_bytes()
    if git_blob_sha1(blob) != APPROVED_V39_BLOB_SHA1:
        raise ValueError("V39 normative text differs from frozen Git blob")
    return hashlib.sha256(blob).hexdigest()


def verify_v38_report(path: Path, expected_sha: str) -> dict:
    validate_frozen_policy()
    if (
        not isinstance(expected_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
        or not path.is_file()
        or sha_file(path) != expected_sha
    ):
        raise ValueError("V38 report SHA-256 missing or different")
    report = json.loads(path.read_bytes())
    if (
        report.get("protocol") != V38_PROTOCOL
        or report.get("status") != "THEORETICAL_FOLD_ELIGIBILITY_PASS_LABELS_AND_ML_BLOCKED"
        or report.get("v33_freeze_sha256") != APPROVED_FREEZE_SHA256
        or report.get("v35_report_sha256") != PINNED_V35_REPORT_SHA256
        or report.get("v37_report_sha256") != PINNED_V37_REPORT_SHA256
        or report.get("fold_count") != 33
        or report.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
        or report.get("label_horizon_bars_theoretical_only") != HORIZON_BARS
        or report.get("embargo_bars") != EMBARGO_BARS
        or report.get("universe") != list(SYMBOLS)
        or report.get("internal_blind_start_utc") != INTERNAL_RESERVE_START.isoformat()
        or report.get("external_holdout_start_utc") != CUTOFF.isoformat()
        or report.get("fold_sample_upper_bound_counts_not_unique") != EXPECTED_V38_COUNTS
        or report.get("no_market_data_read_no_holdout_no_labels_no_training_no_pnl_no_orders") is not True
    ):
        raise ValueError("V38 temporal, label, provenance or no-ML contract mismatch")
    return report


def _check_utc_time(ts: pd.Timestamp, role: str):
    if (
        not isinstance(ts, pd.Timestamp)
        or ts.tzinfo is None
        or str(ts.tz) != "UTC"
    ):
        raise ValueError(role + " must be UTC-aware")
    return ts


def classify_in_memory_test_fixture(
    market: pd.DataFrame,
    *,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    subset: str,
) -> pd.DataFrame:
    """Pure test-fixture helper; never invoked by the V40 CLI.

    Predictive X(t) is computed solely from closes <=t. The supervised
    evaluation target may look ahead exactly six observed contiguous M5
    bars but no target/return/threshold enters the feature schema.
    """
    validate_frozen_policy()
    start = _check_utc_time(window_start, "window_start")
    end = _check_utc_time(window_end, "window_end")
    if (
        subset not in ("train", "validation", "test")
        or start < BEGIN
        or not start < end <= INTERNAL_RESERVE_START
        or (start - BEGIN) % M5 != pd.Timedelta(0)
        or (end - BEGIN) % M5 != pd.Timedelta(0)
    ):
        raise ValueError("Unauthorized synthetic fixture window or holdout period")
    if (
        not isinstance(market, pd.DataFrame)
        or list(market.columns) != list(MARKET_FIELDS)
        or not isinstance(market["bar_close_at"].dtype, pd.DatetimeTZDtype)
        or not isinstance(market["bar_open_at"].dtype, pd.DatetimeTZDtype)
        or str(market["bar_close_at"].dt.tz) != "UTC"
        or str(market["bar_open_at"].dt.tz) != "UTC"
    ):
        raise ValueError("Synthetic test fixture requires exact UTC V34 schema")
    # This rejects any incomplete M5 segment, no imputation and all bars
    # at/after the June blind boundary, BEFORE creating any target.
    features = build_causal_features(market)
    closed = features["bar_close_at"]
    future_close = pd.to_numeric(market[REFERENCE_PRICE_COLUMN], errors="raise").shift(-HORIZON_BARS)
    future_time = closed.shift(-HORIZON_BARS)
    current_close = pd.to_numeric(market[REFERENCE_PRICE_COLUMN], errors="raise")
    sig = features[REFERENCE_VOLATILITY_COLUMN].astype(float)
    threshold = THRESHOLD_MULTIPLIER * sig * np.sqrt(HORIZON_BARS)
    future_return = future_close.div(current_close).sub(1.0)
    embargo = EMBARGO_BARS if subset != "train" else 1
    maturity = closed + HORIZON_BARS * M5
    valid = (
        closed.ge(start + embargo * M5)
        & closed.lt(end)
        & future_time.eq(maturity)
        & maturity.lt(end)  # STRICT, not <=
        & threshold.notna()
        & np.isfinite(threshold)
        & np.isfinite(future_return)
    )
    labels = pd.Series(np.full(len(features), NEUTRAL), index=features.index, dtype="int64")
    labels.loc[valid & future_return.lt(-threshold)] = DOWN
    labels.loc[valid & future_return.gt(threshold)] = UP
    out = pd.DataFrame({
        "bar_close_at": closed.loc[valid],
        "label_mature_at": future_time.loc[valid],
        "target": labels.loc[valid],
    }).reset_index(drop=True)
    if (
        list(out.columns) != list(TARGET_FIELDS)
        or (not out.empty and (
            not out.label_mature_at.lt(end).all()
            or out.target.isna().any()
            or not set(out.target.unique()).issubset({DOWN, NEUTRAL, UP})
        ))
    ):
        raise AssertionError("Unexpected synthetic label output contract")
    return out


def audit_contract(v39_doc: Path, v38_json: Path, v38_sha: str, output: Path, *,
                   synthetic_only_approval: bool) -> dict:
    output = output.resolve()
    if not synthetic_only_approval:
        raise ValueError("Explicit --approve-synthetic-only flag required")
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Require a NEW V40 metadata-only JSON output")
    v39_path, v38_path = v39_doc.resolve(), v38_json.resolve()
    if output in (v39_path, v38_path):
        raise ValueError("Cannot overwrite V39/V38 evidence")
    normative_sha = verify_normative_v39(v39_path)
    verify_v38_report(v38_path, v38_sha)
    report = {
        "protocol": PROTOCOL,
        "status": TARGET_STATUS,
        "approval_scope": "target_mathematics_and_synthetic_fixtures_only",
        "v39_normative_document_sha256": normative_sha,
        "v39_normative_git_blob_sha1": APPROVED_V39_BLOB_SHA1,
        "v38_report_sha256": v38_sha,
        "v33_freeze_sha256": APPROVED_FREEZE_SHA256,
        "upstream_folds": 33,
        "upstream_fold_slots_not_unique": EXPECTED_V38_COUNTS,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "target_schema_version": TARGET_SCHEMA_VERSION,
        "reference_market": REFERENCE_MARKET,
        "reference_price_column": REFERENCE_PRICE_COLUMN,
        "horizon_m5_bars": HORIZON_BARS,
        "threshold": {
            "volatility_series": REFERENCE_VOLATILITY_COLUMN,
            "trailing_log_return_count": VOLATILITY_LOOKBACK_RETURNS,
            "multiplier": THRESHOLD_MULTIPLIER,
            "sqrt_horizon": True,
        },
        "classes": {"DOWN": DOWN, "NEUTRAL": NEUTRAL, "UP": UP},
        "class_edges": {
            "DOWN": "future_return < -threshold",
            "NEUTRAL": "-threshold <= future_return <= threshold",
            "UP": "future_return > threshold",
        },
        "purging": "label_mature_at < window_end_exclusive",
        "validation_test_embargo_m5_bars": EMBARGO_BARS,
        "internal_blind_start_utc": INTERNAL_RESERVE_START.isoformat(),
        "external_holdout_start_utc": CUTOFF.isoformat(),
        "funding_features": "BLOCKED_UNATTESTED_ASOF",
        "historical_label_creation": "NOT_AUTHORIZED",
        "training": "NOT_AUTHORIZED",
        "no_market_data_read_no_holdout_no_label_export_no_training_no_pnl_no_orders": True,
    }
    with output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="V40 approved mathematical target contract; synthetic tests only"
    )
    parser.add_argument("--v39-document", required=True, type=Path)
    parser.add_argument("--v38-json", required=True, type=Path)
    parser.add_argument("--v38-sha256", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--approve-synthetic-only", action="store_true")
    a = parser.parse_args()
    report = audit_contract(
        a.v39_document, a.v38_json, a.v38_sha256, a.output_json,
        synthetic_only_approval=a.approve_synthetic_only,
    )
    print("V40", report["status"], flush=True)
    print("NO HISTORICAL LABELS / NO MARKET FILES / NO ML / NO PNL / NO ORDERS", flush=True)


if __name__ == "__main__":
    main()
