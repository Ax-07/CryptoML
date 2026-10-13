"""CPU-only LightGBM walk-forward evaluation; final dates/assets stay untouched."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.evaluation import evaluate
from mlpipeline.features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, feature_matrix
from mlpipeline.labels import LABEL_SCHEMA_VERSION
from mlpipeline.trainer import DEFAULT_SYMBOLS, _class_probs, prepare_training_data
from mlpipeline.walkforward import make_walk_forward_folds, split_walk_forward_fold


def evaluate_walk_forward(
    examples: pd.DataFrame, *, holdout_symbol: str = "ADAUSDT",
    train_days: int = 120, validation_days: int = 14,
    test_days: int = 14, step_days: int = 14, final_holdout_days: int = 90,
    n_estimators: int = 350, seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    """Fit a new CPU model per fold; score *seen* OOS only.

    The unseen-symbol and final-time holdouts must remain sealed for the
    separate final evaluation. Do not tune models from their scores.
    """
    if n_estimators < 20:
        raise ValueError("n_estimators must be >= 20")
    folds = make_walk_forward_folds(
        examples, holdout_symbol=holdout_symbol, train_days=train_days,
        validation_days=validation_days, test_days=test_days,
        step_days=step_days, final_holdout_days=final_holdout_days,
    )
    scores: list[dict] = []
    predictions: list[pd.DataFrame] = []
    for fold in folds:
        parts = split_walk_forward_fold(examples, fold=fold, holdout_symbol=holdout_symbol)
        train, validation, test = (parts[name] for name in ("train", "validation", "seen_test"))
        if min(map(len, (train, validation, test))) < 100:
            raise ValueError(f"Fold {fold.fold_id}: fewer than 100 observations in a partition")
        if not train.label_mature_at.lt(fold.validation_start).all():
            raise AssertionError("Training label leakage")
        if not validation.label_mature_at.lt(fold.test_start).all():
            raise AssertionError("Validation label leakage")
        if not test.label_mature_at.lt(fold.test_end).all():
            raise AssertionError("OOS label leakage")
        if any(part.symbol.eq(holdout_symbol).any() for part in (train, validation, test)):
            raise AssertionError("Unseen-symbol leakage")
        y_train = train.target.to_numpy(dtype=int)
        y_validation = validation.target.to_numpy(dtype=int)
        if len(np.unique(y_train)) != 3 or len(np.unique(y_validation)) != 3:
            raise ValueError(f"Fold {fold.fold_id}: all 3 classes required in train/validation")
        x_train, x_validation = feature_matrix(train), feature_matrix(validation)
        model = lgb.LGBMClassifier(
            device_type="cpu", objective="multiclass", num_class=3,
            n_estimators=n_estimators, learning_rate=0.04, num_leaves=31,
            max_depth=-1, min_child_samples=100, reg_lambda=5.0,
            colsample_bytree=0.8, subsample=0.9, subsample_freq=1,
            max_bin=63, random_state=seed, n_jobs=8, verbosity=-1,
        )
        model.fit(
            x_train, y_train, eval_set=[(x_validation, y_validation)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        baseline = DummyClassifier(strategy="prior")
        baseline.fit(x_train, y_train)
        x_test = feature_matrix(test)
        probabilities = _class_probs(model, x_test)
        baseline_probabilities = _class_probs(baseline, x_test)
        truth = test.target.to_numpy(dtype=int)
        scores.append({
            "fold_id": fold.fold_id,
            "train_start": fold.train_start.isoformat(),
            "validation_start": fold.validation_start.isoformat(),
            "test_start": fold.test_start.isoformat(),
            "test_end_exclusive": fold.test_end.isoformat(),
            "counts": {k: len(v) for k, v in parts.items()},
            "best_iteration": int(model.best_iteration_),
            "seen_oos_lightgbm": evaluate(truth, probabilities),
            "seen_oos_majority_baseline": evaluate(truth, baseline_probabilities),
            "seen_oos_per_symbol": {
                symbol: evaluate(truth[test.symbol.to_numpy() == symbol],
                                 probabilities[test.symbol.to_numpy() == symbol])
                for symbol in sorted(test.symbol.unique())
            },
        })
        out = test[["symbol", "bar_close_at", "label_mature_at", "target"]].copy()
        out = out.rename(columns={"target": "actual_class"})
        out["fold_id"] = fold.fold_id
        for idx, column in enumerate(("p_down", "p_neutral", "p_up")):
            out[column] = probabilities[:, idx]
        out["predicted_class"] = probabilities.argmax(axis=1)
        for idx, column in enumerate(("baseline_p_down", "baseline_p_neutral", "baseline_p_up")):
            out[column] = baseline_probabilities[:, idx]
        predictions.append(out)
        print(f"fold={fold.fold_id} train={len(train)} val={len(validation)} "
              f"seen_oos={len(test)} best_iter={model.best_iteration_}", flush=True)

    oos = pd.concat(predictions, ignore_index=True)
    if oos.duplicated(["symbol", "bar_close_at"]).any():
        raise AssertionError("Repeated OOS symbol/time: a test interval overlapped")
    if oos.symbol.eq(holdout_symbol).any():
        raise AssertionError("Sealed unseen-symbol evaluated in development")
    oos = oos.sort_values(["bar_close_at", "symbol"]).reset_index(drop=True)
    y = oos.actual_class.to_numpy(dtype=int)
    model_probs = oos[["p_down", "p_neutral", "p_up"]].to_numpy()
    baseline_probs = oos[["baseline_p_down", "baseline_p_neutral", "baseline_p_up"]].to_numpy()
    final_start = examples.loc[
        examples.symbol.ne(holdout_symbol), "bar_close_at"
    ].max() + pd.Timedelta(minutes=5) - pd.Timedelta(days=final_holdout_days)
    if not oos.label_mature_at.lt(final_start).all():
        raise AssertionError("Final temporal holdout leaked into OOS evaluation")
    report = {
        "protocol": "development_walkforward_seen_symbols_only_v1",
        "engine": "cpu",
        "holdout_symbol_reserved": holdout_symbol,
        "final_time_holdout_start": final_start.isoformat(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "parameters": {
            "train_days": train_days, "validation_days": validation_days,
            "test_days": test_days, "step_days": step_days,
            "final_holdout_days": final_holdout_days,
            "n_estimators": n_estimators, "seed": seed,
        },
        "folds": scores,
        "aggregate_seen_oos": {
            "lightgbm": evaluate(y, model_probs),
            "majority_baseline": evaluate(y, baseline_probs),
        },
        "disclaimer": "Research only; test final and unseen-symbol data are not evaluated here.",
    }
    return report, oos


def run_walk_forward(
    store, *, symbols=DEFAULT_SYMBOLS, holdout_symbol="ADAUSDT",
    horizon: int = 6, months: int | None = None,
    threshold_multiplier: float = 0.5, n_estimators: int = 350,
    seed: int = 42, train_days: int = 120, validation_days: int = 14,
    test_days: int = 14, step_days: int = 14,
    final_holdout_days: int = 90,
    artifacts_root: str | Path | None = None,
    verify_hashes: bool = False,
) -> Path:
    if holdout_symbol not in symbols or len(symbols) < 4:
        raise ValueError("Need holdout within >= 4 symbols")
    if not {"BTCUSDT", "ETHUSDT"}.issubset(symbols):
        raise ValueError("BTC/ETH market context required")
    candles, manifests = {}, []
    for symbol in symbols:
        print(f"Reading verified P0 data for {symbol}...", flush=True)
        candles[symbol], associated = load_symbol(
            store, symbol, months=months, verify_hashes=verify_hashes
        )
        manifests.extend(associated)
    examples = prepare_training_data(
        candles, horizon=horizon, threshold_multiplier=threshold_multiplier
    )
    del candles
    report, predictions = evaluate_walk_forward(
        examples, holdout_symbol=holdout_symbol, train_days=train_days,
        validation_days=validation_days, test_days=test_days,
        step_days=step_days, final_holdout_days=final_holdout_days,
        n_estimators=n_estimators, seed=seed,
    )
    report["horizon_bars"] = horizon
    report["threshold_multiplier"] = threshold_multiplier
    report["manifest_fingerprint_sha256"] = manifest_fingerprint(manifests)
    report["source_partitions"] = [
        {"dataset_id": item["dataset_id"], "sha256": item["parquet_sha256"]}
        for item in manifests
    ]
    report["versions"] = {"lightgbm": lgb.__version__, "pandas": pd.__version__}
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("walkforward_%Y%m%dT%H%M%SZ_") + uuid4().hex[:6]
    report["run_id"] = run_id
    # Write to the calling worktree by default, not to the original data storage.
    root = Path(artifacts_root) if artifacts_root else Path(__file__).resolve().parents[2] / "artifacts"
    folder = root / "walkforward_runs" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    predictions.to_parquet(folder / "oos_predictions.parquet", index=False)
    print(f"CPU walk-forward report: {folder.resolve()}", flush=True)
    return folder
