"""Read-only P2 horizon stability post-analysis: assets, time and sides.

Consumes an ALREADY computed seven-horizon JSON. Does NOT retrain, read
Parquet, score ADA/final/OOS, select an operational horizon or use new data.
Existing checks have been repeatedly inspected: all findings exploratory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

HORIZONS = ("1", "2", "3", "6")
CANDIDATES = ("1", "2", "3")
REFERENCE = "6"
ASSETS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
SIDES = ("DOWN", "UP")
PROTOCOL = "cpu_p2_7_horizons_shared_samples_h12_purge_v1"


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _metric(item: dict, symbol: str | None) -> dict:
    if symbol is None:
        return item["metrics"]["directional"]
    return item["per_symbol_directional"][symbol]


def _validate(report: dict) -> list[dict]:
    if report.get("protocol") != PROTOCOL:
        raise ValueError("Expected original seven-horizon H12-purged P2 report")
    if (report.get("ADA_P0_loaded_or_scored") is not False
            or report.get("fold_oos_tests_scored") is not False
            or report.get("final_90_days_scored") is not False):
        raise ValueError("Report is not protected from reserved-data use")
    if tuple(report.get("horizons_predeclared", ())) != (1, 2, 3, 5, 6, 9, 12):
        raise ValueError("Unexpected horizon set")
    if set(report.get("seen_symbols_only", ())) != set(ASSETS):
        raise ValueError("Unexpected seen-asset set")
    folds = report.get("folds")
    if not isinstance(folds, list) or len(folds) < 6:
        raise ValueError("Need at least six historical development folds")
    prior_date = None
    ids = set()
    for fold in folds:
        fold_id = fold["fold_id"]
        if fold_id in ids:
            raise ValueError("Repeated fold ID")
        ids.add(fold_id)
        date = datetime.fromisoformat(fold["validation_start_utc"])
        if date.tzinfo is None or (prior_date is not None and date <= prior_date):
            raise ValueError("Folds must be ordered by UTC validation start")
        prior_date = date
        for symbol in ASSETS:
            n = None
            for horizon in HORIZONS:
                obj = fold["horizons"][horizon]
                metric = _metric(obj, symbol)
                size = metric["n_total"]
                if size <= 0 or (n is not None and size != n):
                    raise ValueError("Misaligned same-date samples across horizons")
                n = size
                if not (0 <= metric["selected"] <= size):
                    raise ValueError("Invalid selected count")
                if metric["selected"] > 0 and (
                    metric["directional_precision"] is None
                    or metric["uplift_points"] is None
                ):
                    raise ValueError("Missing directional metrics")
    return folds


def _summarize(rows: list[dict], *, symbol: str | None) -> dict:
    """Paired fold summaries; null-precision folds excluded by pair.

    Precision and uplift are both reported as arithmetic means over
    comparable folds; pooled precision is independently computed as
    total correct / total directional predictions, not mean precision.
    """
    comparisons = {}
    for h in CANDIDATES:
        precision_deltas = []
        uplift_deltas = []
        coverage_deltas = []
        wins_precision = ties_precision = wins_uplift = ties_uplift = 0
        chosen_count = correct_count = base_chosen = base_correct = 0
        total_count = 0
        valid = 0
        for fold in rows:
            a = _metric(fold["horizons"][h], symbol)
            b = _metric(fold["horizons"][REFERENCE], symbol)
            total_count += a["n_total"]
            chosen_count += a["selected"]
            correct_count += a["correct_direction_class"]
            base_chosen += b["selected"]
            base_correct += b["correct_direction_class"]
            if a["directional_precision"] is None or b["directional_precision"] is None:
                continue
            valid += 1
            dp = 100 * (a["directional_precision"] - b["directional_precision"])
            du = a["uplift_points"] - b["uplift_points"]
            dc = 100 * (a["coverage"] - b["coverage"])
            precision_deltas.append(dp)
            uplift_deltas.append(du)
            coverage_deltas.append(dc)
            wins_precision += dp > 1e-12
            ties_precision += abs(dp) <= 1e-12
            wins_uplift += du > 1e-12
            ties_uplift += abs(du) <= 1e-12

        comparisons[f"H{h}_vs_H6"] = {
            "folds": len(rows),
            "folds_with_both_precisions_defined": valid,
            "wins_precision": wins_precision,
            "ties_precision": ties_precision,
            "wins_uplift": wins_uplift,
            "ties_uplift": ties_uplift,
            "mean_precision_delta_points": _average(precision_deltas),
            "mean_uplift_delta_points": _average(uplift_deltas),
            "mean_coverage_delta_points": _average(coverage_deltas),
            "pooled_precision_candidate": (
                correct_count / chosen_count if chosen_count else None
            ),
            "pooled_precision_h6": (
                base_correct / base_chosen if base_chosen else None
            ),
            "pooled_coverage_candidate": (
                chosen_count / total_count if total_count else None
            ),
            "pooled_coverage_h6": (
                base_chosen / total_count if total_count else None
            ),
        }
    # This is a diagnostic only, not a prediction conditioned on true moves.
    sides = {}
    for h in HORIZONS:
        sides[h] = {}
        for side in SIDES:
            predicted = correct = 0
            for fold in rows:
                metric = _metric(fold["horizons"][h], symbol)
                direction = metric["per_direction"][side]
                predicted += direction["predicted"]
                correct += direction["correct"]
            sides[h][side] = {
                "predicted": predicted,
                "correct": correct,
                "precision": correct / predicted if predicted else None,
            }
    return {"comparisons": comparisons, "sides": sides}


def analyze_horizon_stability(report: dict) -> dict:
    folds = _validate(report)
    middle = len(folds) // 2
    periods = {
        "earlier_half": folds[:middle],
        "later_half": folds[middle:],
    }
    for offset in range(0, len(folds), 6):
        periods[f"chronological_block_{offset // 6 + 1:02d}"] = folds[
            offset: offset + 6
        ]
    by_asset = {
        asset: _summarize(folds, symbol=asset)
        for asset in ASSETS
    }
    by_time = {
        key: {
            "folds": len(part),
            "start_validation_utc": part[0]["validation_start_utc"],
            "end_validation_utc": part[-1]["validation_start_utc"],
            **_summarize(part, symbol=None),
        }
        for key, part in periods.items()
    }
    return {
        "protocol": "cpu_p2_posthoc_7_horizon_asset_time_stability_v1",
        "source_protocol": report["protocol"],
        "development_folds": len(folds),
        "model_retrained": False,
        "source_data_reloaded": False,
        "ADA_fold_OOS_final_scored": False,
        "independent_confirmation": False,
        "selected_horizon_is_not_confirmed": True,
        "notes": [
            "Pure read-only analysis of repeatedly inspected development results.",
            "Horizons have different future labels and class prevalences.",
            "Fold precision averages and pooled precision are different statistics.",
            "Overlapping training periods and time-correlated market data prevent treating fold wins as independent trials.",
            "Wins do not account for multiplicity from exploring several horizons.",
            "No per-trade profitability or executable strategy is assessed.",
        ],
        "overall": _summarize(folds, symbol=None),
        "by_asset": by_asset,
        "by_time": by_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze existing 7-horizon P2 JSON by asset and time; no training"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output files must be distinct")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing diagnostic report")
    source = args.input.read_bytes()
    report = json.loads(source)
    result = analyze_horizon_stability(report)
    result["source_report_sha256"] = hashlib.sha256(source).hexdigest()
    result["source_report"] = str(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)

    print("By asset — H2 vs H6 (development only)")
    for asset in ASSETS:
        item = result["by_asset"][asset]["comparisons"]["H2_vs_H6"]
        print(
            f"{asset}: precision_delta_pp={item['mean_precision_delta_points']:.3f} "
            f"uplift_delta_pp={item['mean_uplift_delta_points']:.3f} "
            f"coverage_delta_pp={item['mean_coverage_delta_points']:.3f} "
            f"precision_wins={item['wins_precision']}/{item['folds_with_both_precisions_defined']}",
            flush=True,
        )
    print("Chronological periods — H2 vs H6 (development only)")
    for key, period in result["by_time"].items():
        if key not in ("earlier_half", "later_half"):
            continue
        item = period["comparisons"]["H2_vs_H6"]
        print(
            f"{key}: precision_delta_pp={item['mean_precision_delta_points']:.3f} "
            f"uplift_delta_pp={item['mean_uplift_delta_points']:.3f} "
            f"precision_wins={item['wins_precision']}/{item['folds_with_both_precisions_defined']}",
            flush=True,
        )
    print(f"Read-only stability report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
