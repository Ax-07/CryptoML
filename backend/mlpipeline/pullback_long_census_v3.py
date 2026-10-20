"""P4 V3 causal, unlabeled EMA9/EMA26 census on 21 development months.

Reads SHA-verified 2024-10 through 2026-06 P0 for exactly BTC/ETH/SOL/XRP.
Neither July-September nor ADA candle files are opened. No future returns,
model fitting, profitability rankings or new OOS claims.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mlpipeline.pullback_long_history_v3 import (
    CUTOFF, EXPECTED_MONTHS, FIRST_MONTH, SYMBOLS,
    load_long_development_history,
)
from mlpipeline.pullback_regime_joint_census_v2 import cross_counts
from mlpipeline.pullback_regime_census_v2 import STRENGTH, DEPTH
from mlpipeline.pullback_regime_v2 import describe_regimes

PROTOCOL = "p4_v3_21month_unlabeled_regime_census_v1"
OVERLAP_BEGIN = "2025-12"
OVERLAP_MONTHS = tuple(month for month in EXPECTED_MONTHS
                       if month >= OVERLAP_BEGIN)
OLDER_MONTHS = tuple(month for month in EXPECTED_MONTHS
                     if month < OVERLAP_BEGIN)


def _counts(frame: pd.DataFrame) -> dict:
    """Explicit zero cells, also when a month contains zero V1 signals."""
    result = cross_counts(frame)
    if (not {"pullback_signal", "pullback_raw"}.issubset(frame.columns)
            or not pd.api.types.is_bool_dtype(frame.pullback_signal)
            or not pd.api.types.is_bool_dtype(frame.pullback_raw)):
        raise ValueError("Invalid or missing V1 detector flags")
    n = len(frame)
    signals = int(frame.pullback_signal.sum())
    raw = int(frame.pullback_raw.sum())
    if (signals != result["events"] or raw < signals
            or (frame.pullback_signal & ~frame.pullback_raw).any()):
        raise AssertionError("V1 signals do not reconcile")
    return {
        "closed_bars": int(n),
        "raw_candidates": raw,
        "signals": signals,
        "suppressed_by_cooldown": raw - signals,
        "classified": result["valid_events"],
        "unclassified": result["unclassified"],
        "joint": result["joint"],
    }


def _aggregate(items: list[dict]) -> dict:
    return {
        key: sum(item[key] for item in items)
        for key in ("closed_bars", "raw_candidates", "signals",
                    "suppressed_by_cooldown", "classified", "unclassified")
    } | {
        "joint": {
            s: {d: sum(item["joint"][s][d] for item in items)
                for d in DEPTH}
            for s in STRENGTH
        }
    }


def _reconcile(total: dict, parts: list[dict]) -> None:
    if total != _aggregate(parts):
        raise AssertionError("Monthly/asset-to-total census mismatch")
    if (total["signals"] != total["classified"] + total["unclassified"]
            or total["raw_candidates"] != total["signals"] + total["suppressed_by_cooldown"]
            or total["classified"] != sum(
                total["joint"][s][d] for s in STRENGTH for d in DEPTH
            )):
        raise AssertionError("Signals/joint classes do not reconcile")


def summarize_long_history(
    features_by_symbol: dict[str, pd.DataFrame],
    dataset_fingerprint: str,
) -> tuple[dict, pd.DataFrame]:
    """Pure analysis of existing CLOSED-CANDLE features; no labels."""
    if set(features_by_symbol) != set(SYMBOLS):
        raise ValueError("Only BTC/ETH/SOL/XRP are approved, not ADA")
    if re.fullmatch(r"[0-9a-f]{64}", dataset_fingerprint) is None:
        raise ValueError("Invalid dataset fingerprint SHA256")

    assets: dict[str, dict] = {}
    rows: list[dict] = []
    for symbol in SYMBOLS:
        features = features_by_symbol[symbol]
        required = {"symbol", "bar_close_at", "pullback_signal", "pullback_raw"}
        if (features.empty or not required.issubset(features.columns)
                or features.symbol.isna().any()
                or not features.symbol.eq(symbol).all()):
            raise ValueError(f"{symbol}: invalid features or symbol")
        times = features.bar_close_at
        if (not isinstance(times.dtype, pd.DatetimeTZDtype)
                or str(times.dt.tz) != "UTC" or times.isna().any()
                or not times.is_unique or not times.is_monotonic_increasing
                or not times.ge(pd.Timestamp("2024-10-01T00:00:00Z")).all()
                or not times.lt(CUTOFF).all()):
            raise ValueError(f"{symbol}: UTC/scope/chronology violation")
        by_month: dict[str, dict] = {}
        for month, part in features.groupby(
            times.dt.strftime("%Y-%m"), sort=True
        ):
            if month not in EXPECTED_MONTHS:
                raise ValueError(f"{symbol}: month outside development")
            counts = _counts(part)
            by_month[month] = counts
            rows.append({
                "symbol": symbol, "month": month,
                **{k: counts[k] for k in (
                    "closed_bars", "raw_candidates", "signals",
                    "suppressed_by_cooldown", "classified", "unclassified")},
                "signals_per_1000_bars": (1000.0 * counts["signals"] / counts["closed_bars"]),
                **{f"strength_{s}_depth_{d}": counts["joint"][s][d]
                   for s in STRENGTH for d in DEPTH},
            })
        if tuple(by_month) != EXPECTED_MONTHS:
            raise ValueError(f"{symbol}: incomplete 21-month development timeline")

        overall = _counts(features)
        _reconcile(overall, list(by_month.values()))
        prior = _aggregate([by_month[m] for m in OLDER_MONTHS])
        overlap = _aggregate([by_month[m] for m in OVERLAP_MONTHS])
        _reconcile(overall, [prior, overlap])
        assets[symbol] = {
            "all_21months": overall,
            "older_14months": prior,
            "overlap_7months": overlap,
            "by_month": by_month,
        }
    grand = _aggregate([assets[s]["all_21months"] for s in SYMBOLS])
    _reconcile(grand, [assets[s]["all_21months"] for s in SYMBOLS])
    return {
        "protocol": PROTOCOL,
        "status": "EXPLORATORY_UNLABELED_21MONTHS_NOT_OOS",
        "source_dataset_fingerprint_sha256": dataset_fingerprint,
        "assets": list(SYMBOLS),
        "scope_first_partition": FIRST_MONTH,
        "scope_last_partition": EXPECTED_MONTHS[-1],
        "exclusive_bar_close_cutoff_utc": CUTOFF.isoformat(),
        "total_monthly_partitions": len(SYMBOLS) * len(EXPECTED_MONTHS),
        "earlier_months": list(OLDER_MONTHS),
        "seven_month_overlap": list(OVERLAP_MONTHS),
        "regime_bins": {
            "strength_atr": ["<0.5", "[0.5,1.5)", ">=1.5"],
            "depth_atr": ["<0", "[0,0.5)", "[0.5,1)", ">=1"],
        },
        "seven_month_comparison_warning": (
            "Prior V2 recomputed indicators starting only in December 2025. "
            "V3 indicators include earlier history, so initialization of EMA "
            "and cooldown near the overlap boundary can change a few events. "
            "Do not require equality of old and new 7-month event counts."
        ),
        "no_labels_no_returns_no_model_training": True,
        "independent_out_of_sample": False,
        "total": grand,
        "by_asset": assets,
    }, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P4 V3 21-month UNLABELED regime census"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    opts = parser.parse_args()
    root = opts.data_root.resolve()
    output_json = opts.output_json.resolve()
    output_csv = opts.output_csv.resolve()
    if (output_json == output_csv
            or output_json.parent != output_csv.parent
            or not output_json.parent.is_dir()
            or any(p.exists() or p == root or p.is_relative_to(root)
                   for p in (output_json, output_csv))):
        parser.error("Two NEW different output files, same existing dir outside P0")
    from ingestion.storage import CandleStore
    from mlpipeline.dataset import manifest_fingerprint

    candles, manifests = load_long_development_history(CandleStore(root))
    # Convert each symbol's full 21 months to causal event features before
    # dividing by month: do not reset EMA at arbitrary month boundaries.
    features = {s: describe_regimes(candles[s]) for s in SYMBOLS}
    report, csv = summarize_long_history(features, manifest_fingerprint(manifests))
    with output_csv.open("x", encoding="utf-8", newline="") as f:
        csv.to_csv(f, index=False)
    with output_json.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    for symbol in SYMBOLS:
        part = report["by_asset"][symbol]
        print(f"{symbol}: all={part['all_21months']['signals']} "
              f"older14={part['older_14months']['signals']} "
              f"overlap7={part['overlap_7months']['signals']} "
              f"unclassified={part['all_21months']['unclassified']}")
    print("TOTAL", report["total"]["signals"], "signals;",
          report["total"]["closed_bars"], "closed bars")
    print("UNLABELED; HISTORICAL DEVELOPMENT ONLY; NO RETURNS/TRAINING/OOS")
    print("Saved", output_json, output_csv)


if __name__ == "__main__":
    main()
