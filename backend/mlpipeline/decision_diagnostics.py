"""Tune class decisions without changing LightGBM's raw class probabilities.

Chronological order: train -> early stopping -> decision tuning -> validation
check -> untouched OOS test. Never tune on observed fold-OOS labels or ADA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_recall_fscore_support,
)

from ingestion.storage import CandleStore
from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.evaluation import evaluate
from mlpipeline.features import feature_matrix
from mlpipeline.trainer import DEFAULT_SYMBOLS, _class_probs, prepare_training_data
from mlpipeline.walkforward import make_walk_forward_folds, split_walk_forward_fold

CLASSES = ("DOWN", "NEUTRAL", "UP")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def decide(probs: np.ndarray, train_counts: np.ndarray, alpha: float) -> np.ndarray:
    """Return only predicted classes: raw probabilities are not recalibrated."""
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("Expected three class probabilities")
    counts = np.asarray(train_counts, dtype=float)
    if counts.shape != (3,) or (counts <= 0).any():
        raise ValueError("All three training classes are required")
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be within [0,1]")
    prior = counts / counts.sum()
    scores = probs * (prior ** (-alpha))[None, :]
    return scores.argmax(axis=1)


def decision_metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    if len(y_true) == 0 or len(y_true) != len(predicted):
        raise ValueError("Nonempty and equal-size truth/prediction arrays required")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, labels=[0, 1, 2], zero_division=0
    )
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(
            y_true, predicted, labels=[0, 1, 2], average="macro", zero_division=0
        )),
        "predicted_counts": {
            label: int((predicted == idx).sum())
            for idx, label in enumerate(CLASSES)
        },
        "confusion": confusion_matrix(y_true, predicted, labels=[0, 1, 2]).tolist(),
        "class_metrics": {
            label: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, label in enumerate(CLASSES)
        },
    }


def compare_decision_rules(
    examples: pd.DataFrame, *, holdout_symbol: str = "ADAUSDT",
    train_days: int = 120, validation_days: int = 14,
    test_days: int = 14, step_days: int = 14,
    final_holdout_days: int = 90, n_estimators: int = 150,
    seed: int = 42,
) -> dict:
    """Choose alpha on past validation; evaluate on later validation only."""
    if validation_days < 4 or n_estimators < 20:
        raise ValueError("Require >=4 validation days and >=20 estimators")
    folds = make_walk_forward_folds(
        examples, holdout_symbol=holdout_symbol, train_days=train_days,
        validation_days=validation_days, test_days=test_days,
        step_days=step_days, final_holdout_days=final_holdout_days,
    )
    results = []
    for fold in folds:
        parts = split_walk_forward_fold(
            examples, fold=fold, holdout_symbol=holdout_symbol
        )
        train, val = parts["train"], parts["validation"]
        stop_end = fold.validation_start + pd.Timedelta(days=validation_days // 2)
        tune_end = stop_end + pd.Timedelta(
            days=(validation_days - validation_days // 2) // 2
        )
        early_stop = val.loc[
            val.bar_close_at.lt(stop_end) & val.label_mature_at.lt(stop_end)
        ]
        tune = val.loc[
            val.bar_close_at.ge(stop_end) & val.bar_close_at.lt(tune_end)
            & val.label_mature_at.lt(tune_end)
        ]
        check = val.loc[val.bar_close_at.ge(tune_end)]
        if min(map(len, (train, early_stop, tune, check))) < 100:
            raise ValueError(f"Fold {fold.fold_id}: insufficient split observations")
        if not train.label_mature_at.lt(fold.validation_start).all():
            raise AssertionError("Train labels overlap validation")
        if not early_stop.label_mature_at.lt(stop_end).all():
            raise AssertionError("Early-stop labels overlap tuning")
        if not tune.label_mature_at.lt(tune_end).all():
            raise AssertionError("Tuning labels overlap validation check")
        if not check.label_mature_at.lt(fold.test_start).all():
            raise AssertionError("Validation labels overlap OOS test")
        if any(
            part.symbol.eq(holdout_symbol).any()
            for part in (train, early_stop, tune, check)
        ):
            raise AssertionError("Unseen asset leaked")
        y_train = train.target.to_numpy(dtype=int)
        y_stop = early_stop.target.to_numpy(dtype=int)
        y_tune = tune.target.to_numpy(dtype=int)
        y_check = check.target.to_numpy(dtype=int)
        counts = np.bincount(y_train, minlength=3)
        if (counts == 0).any() or len(set(y_stop)) < 3:
            raise ValueError(f"Fold {fold.fold_id}: expected all three classes")
        model = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, device_type="cpu",
            n_estimators=n_estimators, learning_rate=0.04, num_leaves=31,
            max_depth=-1, min_child_samples=100, reg_lambda=5.0,
            colsample_bytree=0.8, subsample=0.9, subsample_freq=1,
            max_bin=63, random_state=seed, n_jobs=8, verbosity=-1,
        )
        model.fit(
            feature_matrix(train), y_train,
            eval_set=[(feature_matrix(early_stop), y_stop)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        tuning_probs = _class_probs(model, feature_matrix(tune))
        # Selection ONLY on the designated calibration/tuning days.
        tune_scores = {
            str(alpha): decision_metrics(
                y_tune, decide(tuning_probs, counts, alpha)
            )["macro_f1"]
            for alpha in ALPHAS
        }
        best_alpha = max(ALPHAS, key=lambda a: (tune_scores[str(a)], -a))
        check_probs = _class_probs(model, feature_matrix(check))
        # Actual probability quality is computed from original model probabilities.
        probability_metrics = evaluate(y_check, check_probs)
        scores = {
            "standard": decision_metrics(y_check, decide(check_probs, counts, 0.0)),
            "adjusted": decision_metrics(y_check, decide(check_probs, counts, best_alpha)),
        }
        results.append({
            "fold_id": fold.fold_id,
            "train_start": fold.train_start.isoformat(),
            "early_stop_end_exclusive": stop_end.isoformat(),
            "decision_tune_end_exclusive": tune_end.isoformat(),
            "validation_check_end_exclusive": fold.test_start.isoformat(),
            "counts": {
                "train": len(train), "early_stopping": len(early_stop),
                "decision_tuning": len(tune), "validation_check": len(check),
            },
            "train_class_counts": {
                label: int(counts[idx]) for idx, label in enumerate(CLASSES)
            },
            "best_iteration": int(model.best_iteration_),
            "alpha_selected_from_tuning_only": best_alpha,
            "tuning_macro_f1": tune_scores,
            "raw_probability_metrics_on_validation_check": {
                "log_loss": probability_metrics["log_loss"],
                "multiclass_brier": probability_metrics["multiclass_brier"],
            },
            "validation_check_decisions": scores,
        })
        print(
            f"fold={fold.fold_id} selected_alpha={best_alpha:g} "
            f"check_F1_standard={scores['standard']['macro_f1']:.4f} "
            f"check_F1_adjusted={scores['adjusted']['macro_f1']:.4f} "
            f"raw_logloss={probability_metrics['log_loss']:.4f}",
            flush=True,
        )
    return {
        "protocol": "cpu_decision_bias_separate_validation_tuning_v1",
        "engine": "cpu",
        "holdout_symbol_reserved": holdout_symbol,
        "oos_fold_tests_scored": False,
        "final_time_holdout_scored": False,
        "probabilities_unchanged": True,
        "note": (
            "Class decisions use reweighted scores; they are NOT calibrated "
            "probabilities. Raw-probability log loss/Brier remain unchanged."
        ),
        "parameters": {
            "train_days": train_days, "validation_days": validation_days,
            "test_days": test_days, "step_days": step_days,
            "final_holdout_days": final_holdout_days,
            "n_estimators": n_estimators, "seed": seed,
            "alphas": ALPHAS,
        },
        "folds": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune CPU class decision on validation only")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--months", type=int, default=9)
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
    store = CandleStore(args.data_root)
    candles, manifests = {}, []
    for symbol in DEFAULT_SYMBOLS:
        print(f"Reading {symbol}...", flush=True)
        candles[symbol], included = load_symbol(
            store, symbol, months=args.months, verify_hashes=args.verify_hashes
        )
        manifests.extend(included)
    examples = prepare_training_data(candles, horizon=args.horizon)
    report = compare_decision_rules(
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
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(f"Decision-rule report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
