"""P2 CPU study: compare 120/240/365 training days on EXACT SAME validation dates.

Use the longest window to define a common set of folds. Every shorter
window ends at the SAME validation boundary, with identical 7d early-stop,
3d unused tuning buffer, and 4d validation check. Strict label purging at
each boundary; no ADA, fold OOS tests or 90-day temporal final evaluation.
Previously inspected development validation is NOT an independent test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ingestion.storage import CandleStore
from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.features import feature_matrix
from mlpipeline.trainer import DEFAULT_SYMBOLS, _class_probs, prepare_training_data
from mlpipeline.two_stage_diagnostics import ALPHA_FIXED, _fit_model, _scored_candidate
from mlpipeline.walkforward import make_walk_forward_folds

TRAIN_WINDOWS = (120, 240, 365)
REFERENCE_TRAIN_DAYS = 120


def common_fold_partitions(
    examples: pd.DataFrame, fold, *,
    train_windows: tuple[int, ...] = TRAIN_WINDOWS,
    validation_days: int = 14,
    holdout_symbol: str = "ADAUSDT",
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame, pd.DataFrame, dict]:
    """Partition using common UTC boundaries and strict label-maturity purge."""
    if (not train_windows or len(set(train_windows)) != len(train_windows)
            or any(type(d) is not int or d < 1 for d in train_windows)):
        raise ValueError("Training durations must be distinct positive integer days")
    if validation_days < 4 or holdout_symbol != "ADAUSDT":
        raise ValueError("Requires validation >=4 days and ADAUSDT excluded")
    seen = examples.loc[examples.symbol.ne(holdout_symbol)]
    t, mature = seen.bar_close_at, seen.label_mature_at
    stop_end = fold.validation_start + pd.Timedelta(days=validation_days // 2)
    tune_end = stop_end + pd.Timedelta(
        days=(validation_days - validation_days // 2) // 2
    )
    trains = {}
    for days in train_windows:
        start = fold.validation_start - pd.Timedelta(days=days)
        if start < fold.train_start:
            raise AssertionError("Training period exceeds common-fold history")
        frame = seen.loc[
            t.ge(start) & t.lt(fold.validation_start)
            & mature.lt(fold.validation_start)
        ]
        if len(frame) < 100:
            raise ValueError(f"Fold {fold.fold_id}, train_days={days}: too few rows")
        if (frame.bar_close_at.min() < start
                or not frame.label_mature_at.lt(fold.validation_start).all()):
            raise AssertionError("Training leaks into validation")
        trains[days] = frame

    early = seen.loc[
        t.ge(fold.validation_start) & t.lt(stop_end) & mature.lt(stop_end)
    ]
    check = seen.loc[
        t.ge(tune_end) & t.lt(fold.test_start)
        & mature.lt(fold.test_start)
    ]
    if min(len(early), len(check)) < 100:
        raise ValueError(f"Fold {fold.fold_id}: insufficient early/check samples")
    if not early.label_mature_at.lt(stop_end).all():
        raise AssertionError("Early-stop labels leak into tuning")
    if not check.label_mature_at.lt(fold.test_start).all():
        raise AssertionError("Check labels leak into OOS test")
    if not (fold.validation_start < stop_end < tune_end < fold.test_start):
        raise AssertionError("Invalid chronological validation boundaries")
    boundaries = {
        "validation_start_utc": fold.validation_start.isoformat(),
        "early_stop_end_exclusive_utc": stop_end.isoformat(),
        "unused_tuning_buffer_end_exclusive_utc": tune_end.isoformat(),
        "validation_check_end_exclusive_utc": fold.test_start.isoformat(),
    }
    return trains, early, check, boundaries


def compare_training_windows(
    examples: pd.DataFrame, *,
    train_windows: tuple[int, ...] = TRAIN_WINDOWS,
    holdout_symbol: str = "ADAUSDT",
    validation_days: int = 14,
    test_days: int = 14,
    step_days: int = 14,
    final_holdout_days: int = 90,
    n_estimators: int = 150,
    seed: int = 42,
) -> dict:
    """Evaluate the same dates for each window; no test/ADA/final scoring."""
    if holdout_symbol != "ADAUSDT":
        raise ValueError("ADAUSDT is the reserved asset")
    if (tuple(train_windows) != TRAIN_WINDOWS
            or validation_days != 14
            or test_days != 14
            or step_days != 14
            or final_holdout_days != 90):
        raise ValueError("P2 predeclared protocol: 120/240/365, 14/14/14, final 90")
    if n_estimators < 20:
        raise ValueError("At least 20 estimators required")
    folds = make_walk_forward_folds(
        examples, holdout_symbol=holdout_symbol,
        train_days=max(train_windows), validation_days=validation_days,
        test_days=test_days, step_days=step_days,
        final_holdout_days=final_holdout_days,
    )
    records = []
    for fold in folds:
        trains, early, check, boundaries = common_fold_partitions(
            examples, fold, train_windows=train_windows,
            validation_days=validation_days, holdout_symbol=holdout_symbol,
        )
        y_early = early.target.to_numpy(dtype=int)
        y_check = check.target.to_numpy(dtype=int)
        if len(set(y_early)) != 3:
            raise ValueError(f"Fold {fold.fold_id}: missing early-stop classes")
        X_check = feature_matrix(check)
        fold_models = {}
        for days in train_windows:
            train = trains[days]
            y_train = train.target.to_numpy(dtype=int)
            counts = np.bincount(y_train, minlength=3)
            if (counts == 0).any():
                raise ValueError(f"Fold {fold.fold_id}, days={days}: missing train class")
            model = _fit_model(
                train, early,
                train_target=y_train, early_target=y_early,
                binary=False, n_estimators=n_estimators, seed=seed,
            )
            proba = _class_probs(model, X_check)
            scored = _scored_candidate(y_check, proba, counts)
            per_symbol = {}
            for symbol in sorted(check.symbol.unique()):
                mask = check.symbol.to_numpy() == symbol
                per_symbol[symbol] = _scored_candidate(
                    y_check[mask], proba[mask], counts
                )["directional"]
            fold_models[str(days)] = {
                "train_start_utc": (fold.validation_start -
                                    pd.Timedelta(days=days)).isoformat(),
                "train_rows": len(train),
                "train_class_counts": {
                    label: int(counts[i])
                    for i, label in enumerate(("DOWN", "NEUTRAL", "UP"))
                },
                "best_iteration": int(model.best_iteration_),
                "metrics": scored,
                "per_symbol_directional": per_symbol,
            }
            direction = scored["directional"]
            raw = scored["raw_probabilities"]
            print(
                f"fold={fold.fold_id} train_days={days} rows={len(train)} "
                f"precision={direction['directional_precision'] if direction['directional_precision'] is not None else float('nan'):.4f} "
                f"coverage={direction['coverage']:.4f} "
                f"macro_f1={scored['fixed_alpha_decisions']['macro_f1']:.4f} "
                f"log_loss={raw['log_loss']:.4f}",
                flush=True,
            )
        records.append({
            "fold_id": fold.fold_id,
            **boundaries,
            "early_stopping_rows": len(early),
            "validation_check_rows": len(check),
            "models": fold_models,
        })

    def average(values) -> float | None:
        seq = list(values)
        if not seq or any(value is None for value in seq):
            return None
        return float(np.mean(seq))

    summary = {}
    for days in train_windows:
        key = str(days)
        summary[key] = {
            "folds": len(records),
            "train_rows_mean": average(
                f["models"][key]["train_rows"] for f in records
            ),
            "directional_precision_mean": average(
                f["models"][key]["metrics"]["directional"]["directional_precision"]
                for f in records
            ),
            "coverage_mean": average(
                f["models"][key]["metrics"]["directional"]["coverage"]
                for f in records
            ),
            "random_reference_precision_mean": average(
                f["models"][key]["metrics"]["directional"]["random_reference_precision"]
                for f in records
            ),
            "directional_uplift_points_mean": average(
                f["models"][key]["metrics"]["directional"]["uplift_points"]
                for f in records
            ),
            "conditional_uplift_points_mean": average(
                f["models"][key]["metrics"]["directional"]["conditional_uplift_points"]
                for f in records
            ),
            "macro_f1_fixed_alpha_mean": average(
                f["models"][key]["metrics"]["fixed_alpha_decisions"]["macro_f1"]
                for f in records
            ),
            "balanced_accuracy_fixed_alpha_mean": average(
                f["models"][key]["metrics"]["fixed_alpha_decisions"]["balanced_accuracy"]
                for f in records
            ),
            "raw_log_loss_mean": average(
                f["models"][key]["metrics"]["raw_probabilities"]["log_loss"]
                for f in records
            ),
            "raw_multiclass_brier_mean": average(
                f["models"][key]["metrics"]["raw_probabilities"]["multiclass_brier"]
                for f in records
            ),
        }

    comparisons = {}
    for days in train_windows:
        if days == REFERENCE_TRAIN_DAYS:
            continue
        key = str(days)
        base_key = str(REFERENCE_TRAIN_DAYS)
        comparisons[f"{days}_vs_{REFERENCE_TRAIN_DAYS}"] = {
            "folds": len(records),
            "precision_delta_points_mean": average(
                100 * (
                    f["models"][key]["metrics"]["directional"]["directional_precision"]
                    - f["models"][base_key]["metrics"]["directional"]["directional_precision"]
                ) for f in records
            ),
            "coverage_delta_points_mean": average(
                100 * (
                    f["models"][key]["metrics"]["directional"]["coverage"]
                    - f["models"][base_key]["metrics"]["directional"]["coverage"]
                ) for f in records
            ),
            "log_loss_improvement_mean": average(
                f["models"][base_key]["metrics"]["raw_probabilities"]["log_loss"]
                - f["models"][key]["metrics"]["raw_probabilities"]["log_loss"]
                for f in records
            ),
            "folds_higher_directional_precision": sum(
                f["models"][key]["metrics"]["directional"]["directional_precision"]
                > f["models"][base_key]["metrics"]["directional"]["directional_precision"]
                for f in records
            ),
            "folds_better_raw_log_loss": sum(
                f["models"][key]["metrics"]["raw_probabilities"]["log_loss"]
                < f["models"][base_key]["metrics"]["raw_probabilities"]["log_loss"]
                for f in records
            ),
        }

    return {
        "protocol": "cpu_p2_common_dates_training_window_comparison_v1",
        "engine": "cpu",
        "train_windows_days": list(train_windows),
        "common_folds_defined_by_longest_window_days": max(train_windows),
        "reference_train_days": REFERENCE_TRAIN_DAYS,
        "alpha_fixed": ALPHA_FIXED,
        "features_and_h6_labels_unchanged": True,
        "tuning_days_unused": True,
        "holdout_symbol_excluded": holdout_symbol,
        "fold_oos_tests_scored": False,
        "final_90_days_scored": False,
        "previously_inspected_development_validation_not_independent": True,
        "warnings": [
            "Fewer common folds than the former 120-day experiment: do not compare "
            "the averages directly with older 36-fold reports.",
            "Prior adjustment uses separate train-only priors for each train length.",
            "Overlapping M5 labels and rolling train windows are not independent.",
            "Development comparison only, not evidence of a profitable strategy.",
        ],
        "parameters": {
            "validation_days": validation_days,
            "test_days": test_days,
            "step_days": step_days,
            "final_holdout_days": final_holdout_days,
            "n_estimators": n_estimators,
            "seed": seed,
        },
        "summary": summary,
        "paired_comparisons": comparisons,
        "folds": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU P2 compare 120, 240 and 365 training days on common dev dates"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    if args.months != 24 or args.horizon != 6:
        parser.error("This preregistered P2 comparison requires months=24 and horizon=6")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing report")
    store = CandleStore(args.data_root)
    candles, manifests = {}, []
    for symbol in DEFAULT_SYMBOLS:
        print(f"Reading {symbol}...", flush=True)
        candles[symbol], source_manifests = load_symbol(
            store, symbol, months=args.months, verify_hashes=args.verify_hashes,
        )
        manifests.extend(source_manifests)
    examples = prepare_training_data(candles, horizon=args.horizon)
    del candles
    report = compare_training_windows(
        examples, n_estimators=args.n_estimators, seed=args.seed,
    )
    report["dataset_fingerprint_sha256"] = manifest_fingerprint(manifests)
    report["lightgbm_version"] = lgb.__version__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(f"Training-window P2 report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
