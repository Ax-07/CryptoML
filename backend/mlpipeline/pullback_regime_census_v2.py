"""P4 V2 read-only census of trend-strength and pullback-depth buckets.

Only unlabeled selected EMA9 first-touch events, pre-July 2026 P0 partitions.
No trade outcomes, model fitting, portfolio metrics, or future candles.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from mlpipeline.pullback_frequency import CUTOFF, START, SYMBOLS, count_features
from mlpipeline.pullback_regime_v2 import describe_regimes, regime_census

PROTOCOL = "p4_v2_unlabeled_regime_census_fixed_v1"
STRENGTH = ("low", "mid", "high")
DEPTH = ("no_undercut", "shallow", "medium", "deep")
MONTHS = ("2025-12", "2026-01", "2026-02", "2026-03",
          "2026-04", "2026-05", "2026-06")


def _normalized(counts: dict) -> dict:
    """Preserve explicit zeros even when a month has no signals."""
    if not set(counts.get("strength", {})).issubset(STRENGTH):
        raise ValueError("Unexpected trend-strength bucket")
    if not set(counts.get("depth", {})).issubset(DEPTH):
        raise ValueError("Unexpected pullback-depth bucket")
    normalized = {
        "events": int(counts["events"]),
        "valid_events": int(counts.get("valid_events", 0)),
        "unclassified": int(counts["unclassified"]),
        "strength": {x: int(counts.get("strength", {}).get(x, 0)) for x in STRENGTH},
        "depth": {x: int(counts.get("depth", {}).get(x, 0)) for x in DEPTH},
    }
    if (normalized["events"] != normalized["valid_events"] + normalized["unclassified"]
            or normalized["events"] < 0
            or normalized["valid_events"] < 0
            or normalized["unclassified"] < 0
            or any(x < 0 for x in normalized["strength"].values())
            or any(x < 0 for x in normalized["depth"].values())
            or sum(normalized["strength"].values()) != normalized["valid_events"]
            or sum(normalized["depth"].values()) != normalized["valid_events"]):
        raise AssertionError("Counts do not reconcile")
    return normalized


def _sum_counts(groups: list[dict]) -> dict:
    return {
        "events": sum(x["events"] for x in groups),
        "valid_events": sum(x["valid_events"] for x in groups),
        "unclassified": sum(x["unclassified"] for x in groups),
        "strength": {key: sum(x["strength"][key] for x in groups) for key in STRENGTH},
        "depth": {key: sum(x["depth"][key] for x in groups) for key in DEPTH},
    }


def summarize_regime_features(features_by_asset: dict[str, pd.DataFrame],
                              dataset_fingerprint: str) -> tuple[dict, pd.DataFrame]:
    """Summarize already computed CAUSAL features; accepts no outcome columns.

    Results are descriptive, not statistically independent tests.
    """
    if set(features_by_asset) != set(SYMBOLS):
        raise ValueError("Expected exactly four seen symbols")
    if not re.fullmatch(r"[0-9a-f]{64}", dataset_fingerprint):
        raise ValueError("Expected lowercase SHA-256 dataset fingerprint")
    by_asset: dict[str, dict] = {}
    csv_rows: list[dict] = []
    for symbol in SYMBOLS:
        features = features_by_asset[symbol]
        # Existing P4 read-only validator enforces UTC chronology and P0 scope.
        signal_counts = count_features(features)
        if signal_counts["symbol"] != symbol:
            raise ValueError(f"Wrong symbol for {symbol}")
        months: dict[str, dict] = {}
        for month, frame in features.groupby(features.bar_close_at.dt.strftime("%Y-%m"),
                                              sort=True):
            if month not in MONTHS:
                raise ValueError(f"Month outside approved P0 scope: {month}")
            row = _normalized(regime_census(frame))
            months[month] = row
            csv_rows.append({
                "symbol": symbol,
                "month": month,
                **{k: row[k] for k in ("events", "valid_events", "unclassified")},
                **{f"strength_{k}": row["strength"][k] for k in STRENGTH},
                **{f"depth_{k}": row["depth"][k] for k in DEPTH},
            })
        if tuple(months) != MONTHS:
            raise ValueError(f"{symbol}: expected each of seven full P0 months")
        overall = _normalized(regime_census(features))
        if _sum_counts(list(months.values())) != overall:
            raise AssertionError(f"Monthly-to-total mismatch: {symbol}")
        if overall["events"] != signal_counts["all"]["cooldown_selected"]:
            raise AssertionError(f"V1/V2 candidate mismatch: {symbol}")
        by_asset[symbol] = {"total": overall, "by_month": months}
    grand_total = _sum_counts([item["total"] for item in by_asset.values()])
    report = {
        "protocol": PROTOCOL,
        "status": "EXPLORATORY_UNLABELED_REGIME_CENSUS_NOT_OOS",
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "period_start_utc_inclusive": START.isoformat(),
        "period_end_utc_exclusive": CUTOFF.isoformat(),
        "assets": list(SYMBOLS),
        "fixed_strength_atr_bins": ["<0.5", "[0.5,1.5)", ">=1.5"],
        "fixed_depth_atr_bins": ["<0.0", "[0.0,0.5)", "[0.5,1.0)", ">=1.0"],
        "rule": "V1 closed-bar pullback_signal only; V2 feature maturity at close(t)",
        "no_outcomes_no_training_no_heldout_data": True,
        "note": "Unlabeled descriptions on previously investigated pre-July data; no economic or predictive conclusion.",
        "total": grand_total,
        "by_asset": by_asset,
    }
    return report, pd.DataFrame(csv_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="P4 V2 regime counts ONLY (unlabeled)")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    jpath, cpath = args.output_json.resolve(), args.output_csv.resolve()
    if (jpath == cpath or jpath.parent != cpath.parent
            or not jpath.parent.is_dir()
            or any(path.exists() or path.is_relative_to(root)
                   for path in (jpath, cpath))):
        parser.error("Require two NEW output files in the same existing folder outside P0")

    from ingestion.storage import CandleStore
    from mlpipeline.dataset import manifest_fingerprint
    from mlpipeline.prospective_model_pack import load_clean_historic

    # Manifest hashes verified; only Dec 2025--Jun 2026, four seen symbols.
    candles, manifests = load_clean_historic(CandleStore(root), verify_hashes=True)
    if set(candles) != set(SYMBOLS):
        raise ValueError("Unexpected symbol returned by approved P0 loader")
    features = {symbol: describe_regimes(candles[symbol]) for symbol in SYMBOLS}
    report, csv_table = summarize_regime_features(
        features, manifest_fingerprint(manifests)
    )
    # Prevent overwrite; completed JSON written last as a completion marker.
    with cpath.open("x", encoding="utf-8", newline="") as output:
        csv_table.to_csv(output, index=False)
    with jpath.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False, allow_nan=False)
    for symbol, details in report["by_asset"].items():
        item = details["total"]
        print(f"{symbol}: events={item['events']}, "
              f"categorized={item['valid_events']}, "
              f"unclassified={item['unclassified']}, "
              f"strength={item['strength']}, depth={item['depth']}")
    print("TOTAL", report["total"])
    print("UNLABELED ONLY - NO RETURNS / NO MODEL TRAINING / NO OOS")
    print("Saved", jpath, cpath)


if __name__ == "__main__":
    main()
