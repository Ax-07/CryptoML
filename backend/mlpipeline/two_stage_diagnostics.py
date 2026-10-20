"""P2 CPU experiment: one 3-class LightGBM versus two binary LightGBMs.

Two-stage model:
    p_movement = P(target != NEUTRAL | X)
    p_up_if_movement = P(UP | target != NEUTRAL, X)
    p(DOWN) = p_movement * (1 - p_up_if_movement)
    p(NEUTRAL) = 1 - p_movement
    p(UP) = p_movement * p_up_if_movement

Each fold uses 120d train, then 7d early-stop, 3d unused tuning
buffer, then 4d validation check. Existing fold OOS, the final 90d and
ADA are never scored. Alpha=1.0 is fixed before this experiment.
Development validation periods have already been seen, NOT fresh holdouts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score,
)

from ingestion.storage import CandleStore
from mlpipeline.confidence_diagnostics import directional_metrics
from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.decision_diagnostics import decide, decision_metrics
from mlpipeline.evaluation import evaluate
from mlpipeline.features import feature_matrix
from mlpipeline.trainer import DEFAULT_SYMBOLS, _class_probs, prepare_training_data
from mlpipeline.walkforward import make_walk_forward_folds

ALPHA_FIXED = 1.0
CLASS_NAMES = ("DOWN", "NEUTRAL", "UP")


def combine_stage_probabilities(
    movement: np.ndarray, up_given_movement: np.ndarray,
) -> np.ndarray:
    """Compose proper 3-class probabilities; no hard gate or oracle routing."""
    move = np.asarray(movement, dtype=float)
    up = np.asarray(up_given_movement, dtype=float)
    if (move.ndim != 1 or up.shape != move.shape or len(move) == 0
            or not np.isfinite(move).all() or not np.isfinite(up).all()
            or (move < 0).any() or (move > 1).any()
            or (up < 0).any() or (up > 1).any()):
        raise ValueError("Expected equal nonempty, finite probabilities in [0,1]")
    return np.column_stack((
        move * (1.0 - up),
        1.0 - move,
        move * up,
    ))


def _binary_positive_probability(model: lgb.LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """Positive-class proba independent of class-column assumptions."""
    classes = np.asarray(model.classes_, dtype=int)
    if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
        raise ValueError("Expected fitted binary LightGBM classes 0 and 1")
    probs = np.asarray(model.predict_proba(X), dtype=float)
    if probs.shape != (len(X), 2):
        raise ValueError("Unexpected binary probability shape")
    return probs[:, int(np.flatnonzero(classes == 1)[0])]


def _cpu_model(*, binary: bool, n_estimators: int, seed: int) -> lgb.LGBMClassifier:
    settings = dict(
        device_type="cpu",
        n_estimators=n_estimators,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,
        reg_lambda=5.0,
        colsample_bytree=0.8,
        subsample=0.9,
        subsample_freq=1,
        max_bin=63,
        random_state=seed,
        n_jobs=8,
        verbosity=-1,
    )
    if binary:
        return lgb.LGBMClassifier(objective="binary", **settings)
    return lgb.LGBMClassifier(objective="multiclass", num_class=3, **settings)


def _fit_model(
    train: pd.DataFrame, early: pd.DataFrame, *,
    train_target: np.ndarray, early_target: np.ndarray,
    binary: bool, n_estimators: int, seed: int,
) -> lgb.LGBMClassifier:
    if set(np.unique(train_target).tolist()) != ({0, 1} if binary else {0, 1, 2}):
        raise ValueError("Incomplete train class set")
    if set(np.unique(early_target).tolist()) != ({0, 1} if binary else {0, 1, 2}):
        raise ValueError("Incomplete early-stopping class set")
    model = _cpu_model(binary=binary, n_estimators=n_estimators, seed=seed)
    model.fit(
        feature_matrix(train),
        train_target,
        eval_set=[(feature_matrix(early), early_target)],
        eval_metric="binary_logloss" if binary else "multi_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    return model


def _scored_candidate(
    actual: np.ndarray, probabilities: np.ndarray, train_counts: np.ndarray,
) -> dict:
    raw = evaluate(actual, probabilities)
    predicted = decide(probabilities, train_counts, ALPHA_FIXED)
    # A zero margin, and threshold zero, means NO confidence filtering.
    direction = directional_metrics(
        actual, predicted, np.zeros(len(actual)), threshold=0.0,
    )
    chosen = predicted != 1
    correct = int(((predicted == actual) & chosen).sum())
    neutral_errors = int((chosen & (actual == 1)).sum())
    opposite_errors = int(chosen.sum()) - correct - neutral_errors
    direction["error_breakdown"] = {
        "correct": correct,
        "neutral": neutral_errors,
        "opposite": opposite_errors,
        "neutral_fraction_among_selected": (
            neutral_errors / int(chosen.sum()) if chosen.any() else None
        ),
        "opposite_fraction_among_selected": (
            opposite_errors / int(chosen.sum()) if chosen.any() else None
        ),
    }
    return {
        "raw_probabilities": {
            "accuracy": raw["accuracy"],
            "macro_f1": raw["macro_f1"],
            "balanced_accuracy": raw["balanced_accuracy"],
            "log_loss": raw["log_loss"],
            "multiclass_brier": raw["multiclass_brier"],
        },
        "fixed_alpha_decisions": decision_metrics(actual, predicted),
        "directional": direction,
    }


def compare_two_stage(
    examples: pd.DataFrame, *,
    holdout_symbol: str = "ADAUSDT",
    train_days: int = 120, validation_days: int = 14,
    test_days: int = 14, step_days: int = 14,
    final_holdout_days: int = 90, n_estimators: int = 150,
    seed: int = 42,
) -> dict:
    """Same-fold fair comparison, exclusively on already-used dev validation."""
    if holdout_symbol != "ADAUSDT":
        raise ValueError("P2 only supports excluded ADAUSDT")
    if validation_days < 4 or n_estimators < 20:
        raise ValueError("Require >=4 validation days and >=20 estimators")
    folds = make_walk_forward_folds(
        examples, holdout_symbol=holdout_symbol,
        train_days=train_days, validation_days=validation_days,
        test_days=test_days, step_days=step_days,
        final_holdout_days=final_holdout_days,
    )
    seen = examples.loc[examples.symbol.ne(holdout_symbol)]
    t, mature = seen.bar_close_at, seen.label_mature_at
    records: list[dict] = []

    for fold in folds:
        stop_end = fold.validation_start + pd.Timedelta(days=validation_days // 2)
        tune_end = stop_end + pd.Timedelta(
            days=(validation_days - validation_days // 2) // 2
        )
        # No evaluation, training or threshold tuning on the reserved 3-day buffer.
        train = seen.loc[
            t.ge(fold.train_start) & t.lt(fold.validation_start)
            & mature.lt(fold.validation_start)
        ]
        early = seen.loc[
            t.ge(fold.validation_start) & t.lt(stop_end)
            & mature.lt(stop_end)
        ]
        check = seen.loc[
            t.ge(tune_end) & t.lt(fold.test_start)
            & mature.lt(fold.test_start)
        ]
        if min(map(len, (train, early, check))) < 100:
            raise ValueError(f"Fold {fold.fold_id}: too few chronological samples")
        if not (
            train.label_mature_at.lt(fold.validation_start).all()
            and early.label_mature_at.lt(stop_end).all()
            and check.label_mature_at.lt(fold.test_start).all()
        ):
            raise AssertionError("Invalid chronological label purge")
        y_train = train.target.to_numpy(dtype=int)
        y_early = early.target.to_numpy(dtype=int)
        y_check = check.target.to_numpy(dtype=int)
        counts = np.bincount(y_train, minlength=3)
        if (counts == 0).any() or len(set(y_early)) != 3:
            raise ValueError(f"Fold {fold.fold_id}: missing 3-class targets")

        multiclass = _fit_model(
            train, early, train_target=y_train, early_target=y_early,
            binary=False, n_estimators=n_estimators, seed=seed,
        )
        movement = _fit_model(
            train, early,
            train_target=(y_train != 1).astype(int),
            early_target=(y_early != 1).astype(int),
            binary=True, n_estimators=n_estimators, seed=seed,
        )
        # Stage 2 is fitted ONLY on realized non-neutral training labels.
        # Evaluation never routes samples using their future class.
        train_move = train.loc[train.target.ne(1)]
        early_move = early.loc[early.target.ne(1)]
        direction = _fit_model(
            train_move, early_move,
            train_target=train_move.target.eq(2).to_numpy(dtype=int),
            early_target=early_move.target.eq(2).to_numpy(dtype=int),
            binary=True, n_estimators=n_estimators, seed=seed,
        )
        x_check = feature_matrix(check)
        standard_probs = _class_probs(multiclass, x_check)
        p_move = _binary_positive_probability(movement, x_check)
        p_up_conditional = _binary_positive_probability(direction, x_check)
        cascade_probs = combine_stage_probabilities(p_move, p_up_conditional)

        models = {
            "three_class": _scored_candidate(y_check, standard_probs, counts),
            "two_stage": _scored_candidate(y_check, cascade_probs, counts),
        }
        real_move = (y_check != 1).astype(int)
        conditional_mask = real_move.astype(bool)
        direction_true = y_check[conditional_mask] == 2
        conditional_direction_correct = (
            (p_up_conditional[conditional_mask] >= 0.5) == direction_true
        )
        models["two_stage"]["stage_diagnostics"] = {
            "movement_binary_roc_auc": float(roc_auc_score(real_move, p_move)),
            "movement_binary_balanced_accuracy": float(
                balanced_accuracy_score(real_move, p_move >= 0.5)
            ),
            "movement_binary_log_loss": float(log_loss(real_move, p_move, labels=[0, 1])),
            "movement_binary_brier": float(brier_score_loss(real_move, p_move)),
            "direction_accuracy_given_TRUE_movement_oracle_diagnostic_only": float(
                conditional_direction_correct.mean()
            ),
            "true_movement_count": int(conditional_mask.sum()),
            "note": (
                "Conditional direction accuracy uses realized movement labels; "
                "it is an ORACLE diagnostic, not an actionable accuracy."
            ),
        }
        per_symbol = {}
        for symbol in sorted(check.symbol.unique()):
            mask = check.symbol.to_numpy() == symbol
            per_symbol[symbol] = {
                name: _scored_candidate(
                    y_check[mask],
                    probabilities[mask],
                    counts,
                )["directional"]
                for name, probabilities in (
                    ("three_class", standard_probs),
                    ("two_stage", cascade_probs),
                )
            }
        records.append({
            "fold_id": fold.fold_id,
            "train_start_utc": fold.train_start.isoformat(),
            "early_stop_end_exclusive_utc": stop_end.isoformat(),
            "unused_tuning_buffer_end_exclusive_utc": tune_end.isoformat(),
            "validation_check_end_exclusive_utc": fold.test_start.isoformat(),
            "counts": {
                "train": len(train),
                "early_stopping": len(early),
                "stage2_train_movers": len(train_move),
                "stage2_early_movers": len(early_move),
                "validation_check": len(check),
            },
            "train_class_counts": dict(zip(CLASS_NAMES, [int(c) for c in counts])),
            "best_iteration": {
                "three_class": int(multiclass.best_iteration_),
                "movement": int(movement.best_iteration_),
                "direction_conditional": int(direction.best_iteration_),
            },
            "models": models,
            "directional_per_symbol": per_symbol,
        })
        a = models["three_class"]["directional"]
        b = models["two_stage"]["directional"]
        print(
            f"fold={fold.fold_id} three_class_precision={a['directional_precision']:.4f} "
            f"two_stage_precision={b['directional_precision']:.4f} "
            f"three_class_coverage={a['coverage']:.4f} "
            f"two_stage_coverage={b['coverage']:.4f} "
            f"three_class_logloss={models['three_class']['raw_probabilities']['log_loss']:.4f} "
            f"two_stage_logloss={models['two_stage']['raw_probabilities']['log_loss']:.4f}",
            flush=True,
        )

    def average(model: str, group: str, metric: str) -> float | None:
        values = [f["models"][model][group][metric] for f in records]
        present = [float(x) for x in values if x is not None]
        return float(np.mean(present)) if len(present) == len(values) else None

    mean_summary = {}
    for name in ("three_class", "two_stage"):
        mean_summary[name] = {
            "directional_precision_mean": average(name, "directional", "directional_precision"),
            "directional_coverage_mean": average(name, "directional", "coverage"),
            "directional_uplift_points_mean": average(name, "directional", "uplift_points"),
            "macro_f1_fixed_alpha_mean": average(name, "fixed_alpha_decisions", "macro_f1"),
            "balanced_accuracy_fixed_alpha_mean": average(
                name, "fixed_alpha_decisions", "balanced_accuracy"
            ),
            "raw_log_loss_mean": average(name, "raw_probabilities", "log_loss"),
            "raw_multiclass_brier_mean": average(
                name, "raw_probabilities", "multiclass_brier"
            ),
        }
    return {
        "protocol": "cpu_p2_two_stage_development_comparison_v1",
        "engine": "cpu",
        "frozen_reference_model_unchanged": True,
        "fixed_alpha": ALPHA_FIXED,
        "labels_and_features_unchanged": True,
        "horizon_bars": 6,
        "holdout_symbol_excluded": holdout_symbol,
        "fold_oos_tests_scored": False,
        "final_90_days_scored": False,
        "tuning_buffer_used": False,
        "validation_check_not_new_independent_test": True,
        "probabilities_note": (
            "Log loss and Brier use raw proper 3-class probabilities. "
            "Fixed-alpha class decisions rescale scores and are NOT calibrated probabilities."
        ),
        "warning": (
            "These development windows were consulted by previous experiments. "
            "Do not infer a fresh independent confirmation or trading profitability."
        ),
        "parameters": {
            "train_days": train_days,
            "validation_days": validation_days,
            "test_days": test_days,
            "step_days": step_days,
            "final_holdout_days": final_holdout_days,
            "n_estimators": n_estimators,
            "seed": seed,
        },
        "folds": records,
        "fold_mean_summary": mean_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU P2 compare three-class vs two-stage on development validation"
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
        parser.error("P2 development excludes ADAUSDT")
    if args.horizon != 6:
        parser.error("This controlled comparison keeps horizon 6 fixed")
    if args.output.exists():
        parser.error("Report already exists: refusing to overwrite")
    store = CandleStore(args.data_root)
    candles, manifests = {}, []
    for symbol in DEFAULT_SYMBOLS:
        print(f"Reading {symbol}...", flush=True)
        candles[symbol], included = load_symbol(
            store, symbol, months=args.months, verify_hashes=args.verify_hashes,
        )
        manifests.extend(included)
    examples = prepare_training_data(candles, horizon=args.horizon)
    del candles
    report = compare_two_stage(
        examples,
        holdout_symbol=args.holdout_symbol,
        train_days=args.train_days,
        validation_days=args.validation_days,
        test_days=args.test_days,
        step_days=args.step_days,
        final_holdout_days=args.final_holdout_days,
        n_estimators=args.n_estimators,
        seed=args.seed,
    )
    report["dataset_fingerprint_sha256"] = manifest_fingerprint(manifests)
    report["lightgbm_version"] = lgb.__version__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"P2 two-stage comparison report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
