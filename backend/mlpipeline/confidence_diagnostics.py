"""P2 CPU confidence/abstention diagnostics: development validation only.

Per fold: 120d train -> 7d early stop -> 3d threshold tune -> 4d check.
Alpha is fixed at the frozen reference (1.0), NOT retuned.
No fold OOS-test labels, final 90-day holdout, or ADA targets are evaluated.
Development check periods have been seen in earlier experiments and are NOT a
new untouched independent test. Confidence scores are NOT probabilities.
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
from mlpipeline.walkforward import make_walk_forward_folds

CLASS_NAMES = ("DOWN", "NEUTRAL", "UP")
ALPHA_FIXED = 1.0
THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3)
MIN_TUNE_COVERAGE = 0.20
MIN_TUNE_PER_DIRECTION = 50


def decision_and_margin(
    probabilities: np.ndarray, train_counts: np.ndarray, *,
    alpha: float = ALPHA_FIXED,
) -> tuple[np.ndarray, np.ndarray]:
    """Prior-adjusted argmax and normalized top-two score gap.

    Normalization here defines a bounded, relative margin for RANKING only.
    This is not probability calibration and must never be scored as log loss.
    """
    p = np.asarray(probabilities, dtype=float)
    counts = np.asarray(train_counts, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all():
        raise ValueError("Require nonempty finite N x 3 probability matrix")
    if ((p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Invalid class probabilities")
    if counts.shape != (3,) or not np.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("Require positive training counts for all three classes")
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be in [0,1]")
    prior = counts / counts.sum()
    scores = p * prior[None, :] ** (-alpha)
    scores = scores / scores.sum(axis=1, keepdims=True)
    predictions = scores.argmax(axis=1)
    ordered = np.sort(scores, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    return predictions, margins


def directional_metrics(
    truth: np.ndarray, predictions: np.ndarray,
    margins: np.ndarray, *, threshold: float,
) -> dict:
    """Direction precision across all selected UP/DOWN predictions, NEUTRAL wrong.

    Random-reference precision holds predicted DOWN/UP counts fixed and uses
    the ACTUAL class prevalences of the full evaluation period. A second,
    conditional reference uses prevalences within the selected subset.
    """
    y = np.asarray(truth, dtype=int)
    pred = np.asarray(predictions, dtype=int)
    gap = np.asarray(margins, dtype=float)
    if (y.ndim != 1 or pred.shape != y.shape or gap.shape != y.shape
            or len(y) == 0 or not np.isfinite(gap).all()
            or not np.isin(y, [0, 1, 2]).all()
            or not np.isin(pred, [0, 1, 2]).all()):
        raise ValueError("Invalid truth/prediction/margin arrays")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must lie in [0,1]")
    chosen = (pred != 1) & (gap >= threshold)
    n = int(chosen.sum())
    counts = np.bincount(pred[chosen], minlength=3)
    hits = int(((pred == y) & chosen).sum())
    prevalence = np.bincount(y, minlength=3) / len(y)
    per_class = {}
    for cls, label in ((0, "DOWN"), (2, "UP")):
        side = chosen & (pred == cls)
        side_n = int(side.sum())
        right = int((side & (y == cls)).sum())
        support = int((y == cls).sum())
        per_class[label] = {
            "predicted": side_n,
            "correct": right,
            "precision": right / side_n if side_n else None,
            "recall_of_true_class": right / support if support else None,
        }
    random_reference = (
        float((counts[0] * prevalence[0] + counts[2] * prevalence[2]) / n)
        if n else None
    )
    conditional_prevalence = np.bincount(y[chosen], minlength=3) / n if n else None
    conditional_reference = (
        float((counts[0] * conditional_prevalence[0]
               + counts[2] * conditional_prevalence[2]) / n)
        if n else None
    )
    precision = hits / n if n else None
    return {
        "n_total": len(y),
        "selected": n,
        "coverage": n / len(y),
        "correct_direction_class": hits,
        "directional_precision": precision,
        "random_reference_precision": random_reference,
        "conditional_random_reference_precision": conditional_reference,
        "uplift_points": 100 * (precision - random_reference) if n else None,
        "conditional_uplift_points": (
            100 * (precision - conditional_reference) if n else None
        ),
        "neutral_among_selected": int((chosen & (y == 1)).sum()),
        "per_direction": per_class,
    }


def select_threshold(
    truth: np.ndarray, predictions: np.ndarray, margins: np.ndarray, *,
    thresholds: tuple[float, ...] = THRESHOLDS,
    min_coverage: float = MIN_TUNE_COVERAGE,
    min_each_direction: int = MIN_TUNE_PER_DIRECTION,
) -> tuple[float, dict]:
    """Tune only from the chronologically earlier decision-tuning partition.

    Maximize uplift vs the class-frequency random reference, subject to
    minimum overall coverage and each direction having enough predictions.
    If no threshold qualifies, fall back explicitly to the no-filter policy.
    """
    if not thresholds or thresholds[0] != 0.0 or len(set(thresholds)) != len(thresholds):
        raise ValueError("Threshold grid must start with 0 and contain unique values")
    if not 0 < min_coverage <= 1 or min_each_direction < 1:
        raise ValueError("Invalid selection constraints")
    trials = {}
    eligible = []
    for value in thresholds:
        metrics = directional_metrics(truth, predictions, margins, threshold=value)
        trials[str(value)] = metrics
        if (metrics["coverage"] >= min_coverage
                and all(metrics["per_direction"][side]["predicted"] >= min_each_direction
                        for side in ("DOWN", "UP"))):
            eligible.append((value, metrics))
    if not eligible:
        return 0.0, {
            "eligible_count": 0,
            "selection_status": "fallback_no_eligible_threshold",
            "tuning_trials": trials,
        }
    best_value, _ = max(
        eligible,
        key=lambda item: (
            item[1]["uplift_points"],
            item[1]["directional_precision"],
            -item[0],
        ),
    )
    return float(best_value), {
        "eligible_count": len(eligible),
        "selection_status": "selected_on_tuning_only",
        "tuning_trials": trials,
    }


def confidence_diagnostics(
    examples: pd.DataFrame, *, holdout_symbol: str = "ADAUSDT",
    train_days: int = 120, validation_days: int = 14,
    test_days: int = 14, step_days: int = 14,
    final_holdout_days: int = 90, n_estimators: int = 150,
    seed: int = 42,
) -> dict:
    """36 chronological development checks on 24 months, no sealed data scored."""
    if validation_days < 4 or n_estimators < 20:
        raise ValueError("Need >=4 validation days and >=20 estimators")
    folds = make_walk_forward_folds(
        examples, holdout_symbol=holdout_symbol, train_days=train_days,
        validation_days=validation_days, test_days=test_days,
        step_days=step_days, final_holdout_days=final_holdout_days,
    )
    # No reference to the OOS-test partitions or unseen-symbol label arrays.
    seen = examples.loc[examples.symbol.ne(holdout_symbol)]
    times = seen.bar_close_at
    mature = seen.label_mature_at
    results = []
    for fold in folds:
        stop_end = fold.validation_start + pd.Timedelta(days=validation_days // 2)
        tune_end = stop_end + pd.Timedelta(days=(validation_days - validation_days // 2) // 2)
        train = seen.loc[
            times.ge(fold.train_start) & times.lt(fold.validation_start)
            & mature.lt(fold.validation_start)
        ]
        early = seen.loc[
            times.ge(fold.validation_start) & times.lt(stop_end) & mature.lt(stop_end)
        ]
        tune = seen.loc[
            times.ge(stop_end) & times.lt(tune_end) & mature.lt(tune_end)
        ]
        check = seen.loc[
            times.ge(tune_end) & times.lt(fold.test_start)
            & mature.lt(fold.test_start)
        ]
        if min(len(x) for x in (train, early, tune, check)) < 100:
            raise ValueError(f"Fold {fold.fold_id}: insufficient chronological samples")
        counts = np.bincount(train.target.to_numpy(dtype=int), minlength=3)
        if ((counts == 0).any()
                or len(set(early.target.astype(int))) < 3
                or len(set(tune.target.astype(int))) < 3):
            raise ValueError(f"Fold {fold.fold_id}: missing training/validation class")

        model = lgb.LGBMClassifier(
            device_type="cpu", objective="multiclass", num_class=3,
            n_estimators=n_estimators, learning_rate=0.04, num_leaves=31,
            max_depth=-1, min_child_samples=100, reg_lambda=5.0,
            colsample_bytree=0.8, subsample=0.9, subsample_freq=1,
            max_bin=63, random_state=seed, n_jobs=8, verbosity=-1,
        )
        model.fit(
            feature_matrix(train), train.target.to_numpy(dtype=int),
            eval_set=[(feature_matrix(early), early.target.to_numpy(dtype=int))],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        predicted_tune, margin_tune = decision_and_margin(
            _class_probs(model, feature_matrix(tune)), counts,
        )
        selected_threshold, selection = select_threshold(
            tune.target.to_numpy(dtype=int), predicted_tune, margin_tune,
        )

        predicted_check, margin_check = decision_and_margin(
            _class_probs(model, feature_matrix(check)), counts,
        )
        check_truth = check.target.to_numpy(dtype=int)
        no_filter = directional_metrics(
            check_truth, predicted_check, margin_check, threshold=0.0,
        )
        filtered = directional_metrics(
            check_truth, predicted_check, margin_check,
            threshold=selected_threshold,
        )
        # A diagnostic curve, NOT used to change the chosen threshold or rank
        # candidate methods after observing validation check labels.
        curve = {
            str(threshold): directional_metrics(
                check_truth, predicted_check, margin_check, threshold=threshold
            )
            for threshold in THRESHOLDS
        }
        per_symbol = {}
        for symbol in sorted(check.symbol.unique()):
            symbol_mask = check.symbol.to_numpy() == symbol
            per_symbol[symbol] = {
                "unfiltered": directional_metrics(
                    check_truth[symbol_mask], predicted_check[symbol_mask],
                    margin_check[symbol_mask], threshold=0.0
                ),
                "selected_threshold": directional_metrics(
                    check_truth[symbol_mask], predicted_check[symbol_mask],
                    margin_check[symbol_mask], threshold=selected_threshold
                ),
            }
        results.append({
            "fold_id": fold.fold_id,
            "train_start_utc": fold.train_start.isoformat(),
            "early_stop_start_utc": fold.validation_start.isoformat(),
            "decision_tune_start_utc": stop_end.isoformat(),
            "validation_check_start_utc": tune_end.isoformat(),
            "validation_check_end_exclusive_utc": fold.test_start.isoformat(),
            "partition_counts": {
                "train": len(train), "early_stopping": len(early),
                "decision_tuning": len(tune), "validation_check": len(check),
            },
            "best_iteration": int(model.best_iteration_),
            "train_class_counts": dict(zip(CLASS_NAMES, [int(x) for x in counts])),
            "fixed_alpha": ALPHA_FIXED,
            "tuning_only_threshold": selected_threshold,
            "tuning": selection,
            "check_no_filter": no_filter,
            "check_selected_threshold": filtered,
            "check_descriptive_threshold_curve": curve,
            "check_per_symbol": per_symbol,
        })
        print(
            f"fold={fold.fold_id} threshold={selected_threshold:g} "
            f"precision_base={no_filter['directional_precision']:.4f} "
            f"precision_filtered={filtered['directional_precision'] if filtered['directional_precision'] is not None else float('nan'):.4f} "
            f"coverage={filtered['coverage']:.4f}",
            flush=True,
        )
    return {
        "protocol": "cpu_p2_confidence_development_validation_only_v1",
        "engine": "cpu",
        "class_order": CLASS_NAMES,
        "alpha_fixed": ALPHA_FIXED,
        "confidence_definition": "top1_minus_top2_prior_adjusted_normalized_scores_not_probabilities",
        "threshold_grid": THRESHOLDS,
        "min_tune_coverage": MIN_TUNE_COVERAGE,
        "min_tune_each_direction": MIN_TUNE_PER_DIRECTION,
        "holdout_symbol_excluded": holdout_symbol,
        "fold_oos_tests_scored": False,
        "final_90_days_scored": False,
        "note": (
            "Research only; these validation-check dates have previously been "
            "consulted in other development experiments, so NOT an untouched "
            "independent holdout. No recalibration, trading or performance claims."
        ),
        "parameters": {
            "train_days": train_days, "validation_days": validation_days,
            "test_days": test_days, "step_days": step_days,
            "final_holdout_days": final_holdout_days,
            "n_estimators": n_estimators, "seed": seed,
        },
        "folds": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU P2 direction-confidence diagnostics (no final or ADA scoring)"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--holdout-symbol", choices=DEFAULT_SYMBOLS, default="ADAUSDT")
    parser.add_argument("--train-days", type=int, default=120)
    parser.add_argument("--validation-days", type=int, default=14)
    parser.add_argument("--test-days", type=int, default=14)
    parser.add_argument("--step-days", type=int, default=14)
    parser.add_argument("--final-holdout-days", type=int, default=90)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    if args.holdout_symbol != "ADAUSDT":
        parser.error("P2 development protocol reserves ADAUSDT only")
    if args.output.exists():
        parser.error("Output file exists: refusing to overwrite earlier diagnostics")
    store = CandleStore(args.data_root)
    candles, manifests = {}, []
    for symbol in DEFAULT_SYMBOLS:
        print(f"Reading {symbol}...", flush=True)
        candles[symbol], entries = load_symbol(
            store, symbol, months=args.months, verify_hashes=args.verify_hashes,
        )
        manifests.extend(entries)
    examples = prepare_training_data(candles, horizon=args.horizon)
    del candles
    report = confidence_diagnostics(
        examples, holdout_symbol=args.holdout_symbol,
        train_days=args.train_days, validation_days=args.validation_days,
        test_days=args.test_days, step_days=args.step_days,
        final_holdout_days=args.final_holdout_days,
        n_estimators=args.n_estimators, seed=args.seed,
    )
    report["horizon_bars"] = args.horizon
    report["dataset_fingerprint_sha256"] = manifest_fingerprint(manifests)
    report["lightgbm_version"] = lgb.__version__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(f"P2 confidence report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
