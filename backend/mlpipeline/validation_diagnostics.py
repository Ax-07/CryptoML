"""Exploratory CPU class-weight comparison on validation data only.

Do not use OOS/test/final-holdout labels for model or weight selection.
Each fold's validation is divided into early-stopping and selection periods.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from ingestion.storage import CandleStore
from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.evaluation import evaluate
from mlpipeline.features import feature_matrix
from mlpipeline.trainer import DEFAULT_SYMBOLS, _class_probs, prepare_training_data
from mlpipeline.walkforward import make_walk_forward_folds, split_walk_forward_fold


def compare_class_weights_on_validation(
    examples: pd.DataFrame, *, holdout_symbol: str = "ADAUSDT",
    train_days: int = 120, validation_days: int = 14,
    test_days: int = 14, step_days: int = 14,
    final_holdout_days: int = 90, n_estimators: int = 150,
    seed: int = 42,
) -> dict:
    """Report neutral vs balanced class weighting on unseen validation dates.

    Crucially, neither the OOS fold test nor the unseen asset is scored here.
    Models stop on the first half of validation; candidates are compared on
    its second half, with a purged boundary between these two portions.
    """
    if validation_days < 2 or n_estimators < 20:
        raise ValueError("Require >=2 validation days and >=20 estimators")
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
        train = parts["train"]
        validation = parts["validation"]
        mid = fold.validation_start + (fold.test_start - fold.validation_start) / 2
        stop = validation.loc[
            (validation.bar_close_at < mid)
            & (validation.label_mature_at < mid)
        ]
        selection = validation.loc[validation.bar_close_at >= mid]
        if min(len(train), len(stop), len(selection)) < 100:
            raise ValueError(f"Fold {fold.fold_id}: insufficient train/validation observations")
        if any(x.symbol.eq(holdout_symbol).any() for x in (train, stop, selection)):
            raise AssertionError("Unseen symbol leaked into development")
        if not train.label_mature_at.lt(fold.validation_start).all():
            raise AssertionError("Train labels overlap validation")
        if not stop.label_mature_at.lt(mid).all():
            raise AssertionError("Early stopping labels overlap selection")
        if not selection.label_mature_at.lt(fold.test_start).all():
            raise AssertionError("Selection labels overlap test")

        y_train = train.target.to_numpy(dtype=int)
        y_stop = stop.target.to_numpy(dtype=int)
        y_select = selection.target.to_numpy(dtype=int)
        counts = np.bincount(y_train, minlength=3)
        if (counts == 0).any() or len(set(y_stop)) < 3 or len(set(y_select)) < 3:
            raise ValueError(f"Fold {fold.fold_id}: all classes required in partitions")
        x_train = feature_matrix(train)
        x_stop = feature_matrix(stop)
        x_select = feature_matrix(selection)

        baseline = DummyClassifier(strategy="prior")
        baseline.fit(x_train, y_train)
        candidates = {
            "majority_baseline": {
                "metrics": evaluate(y_select, _class_probs(baseline, x_select)),
                "best_iteration": None,
            }
        }
        # Counts are calculated exclusively on training labels.
        balanced_weights = len(y_train) / (3.0 * counts[y_train])
        for name, sample_weight in (
            ("standard", None),
            ("balanced_train_only", balanced_weights),
        ):
            model = lgb.LGBMClassifier(
                objective="multiclass", num_class=3, device_type="cpu",
                n_estimators=n_estimators, learning_rate=0.04,
                num_leaves=31, max_depth=-1, min_child_samples=100,
                reg_lambda=5.0, colsample_bytree=0.8, subsample=0.9,
                subsample_freq=1, max_bin=63, random_state=seed,
                n_jobs=8, verbosity=-1,
            )
            model.fit(
                x_train, y_train, sample_weight=sample_weight,
                eval_set=[(x_stop, y_stop)], eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )
            probabilities = _class_probs(model, x_select)
            candidates[name] = {
                "metrics": evaluate(y_select, probabilities),
                "best_iteration": int(model.best_iteration_),
                "predicted_counts": {
                    label: int((probabilities.argmax(axis=1) == idx).sum())
                    for idx, label in enumerate(("DOWN", "NEUTRAL", "UP"))
                },
            }
        results.append({
            "fold_id": fold.fold_id,
            "train_start": fold.train_start.isoformat(),
            "validation_start": fold.validation_start.isoformat(),
            "early_stopping_end_exclusive": mid.isoformat(),
            "selection_end_exclusive": fold.test_start.isoformat(),
            "counts": {
                "train": len(train), "early_stopping": len(stop),
                "model_selection": len(selection),
            },
            "train_class_counts": {
                label: int(counts[idx])
                for idx, label in enumerate(("DOWN", "NEUTRAL", "UP"))
            },
            "candidates": candidates,
        })
        print(
            f"fold={fold.fold_id} selection={len(selection)} "
            f"F1 standard={candidates['standard']['metrics']['macro_f1']:.4f} "
            f"balanced={candidates['balanced_train_only']['metrics']['macro_f1']:.4f}",
            flush=True,
        )
    return {
        "protocol": "exploratory_validation_only_weights_v1",
        "engine": "cpu",
        "holdout_symbol_reserved": holdout_symbol,
        "fold_oos_and_final_test_scored": False,
        "parameters": {
            "train_days": train_days, "validation_days": validation_days,
            "test_days": test_days, "step_days": step_days,
            "final_holdout_days": final_holdout_days,
            "n_estimators": n_estimators, "seed": seed,
        },
        "folds": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CPU class weighting on validation only")
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
        candles[symbol], current = load_symbol(
            store, symbol, months=args.months, verify_hashes=args.verify_hashes
        )
        manifests.extend(current)
    examples = prepare_training_data(candles, horizon=args.horizon)
    report = compare_class_weights_on_validation(
        examples, holdout_symbol=args.holdout_symbol,
        train_days=args.train_days, validation_days=args.validation_days,
        test_days=args.test_days, step_days=args.step_days,
        final_holdout_days=args.final_holdout_days,
        n_estimators=args.n_estimators, seed=args.seed,
    )
    report["manifest_fingerprint_sha256"] = manifest_fingerprint(manifests)
    report["horizon_bars"] = args.horizon
    report["lightgbm_version"] = lgb.__version__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Fail rather than silently overwrite an earlier experimental report.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(f"Validation-only report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
