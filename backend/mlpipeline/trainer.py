"""First reproducible LightGBM H6 multi-asset, strictly held-out directional evaluation."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier

from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.evaluation import ALL_LABELS, evaluate
from mlpipeline.features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, build_symbol_features, feature_matrix, join_market_context
from mlpipeline.labels import LABEL_NAMES, LABEL_SCHEMA_VERSION, add_labels
from mlpipeline.split import boundaries, split_with_purge

DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT")
REQUIRED_CONTEXT = ("BTCUSDT", "ETHUSDT")


def prepare_training_data(candles: dict[str, pd.DataFrame], *, horizon: int = 6,
                          threshold_multiplier: float = 0.5) -> pd.DataFrame:
    """Feature/label builders cannot peek at another symbol's future."""
    if not set(REQUIRED_CONTEXT).issubset(candles):
        raise ValueError("Need BTCUSDT and ETHUSDT histories for context")
    features = {symbol: build_symbol_features(df) for symbol, df in candles.items()}
    outputs = []
    keep = ["symbol", "bar_close_at", "label_mature_at", "future_return", "target", *FEATURE_COLUMNS]
    for symbol, frame in features.items():
        with_context = join_market_context(frame, features["BTCUSDT"], features["ETHUSDT"])
        with_labels = add_labels(with_context, horizon=horizon,
                                 threshold_multiplier=threshold_multiplier)
        # Hard block: incomplete features, missing context, missing future price.
        good_features = np.isfinite(with_labels[FEATURE_COLUMNS].to_numpy(dtype="float64")).all(axis=1)
        eligible = good_features & with_labels["target"].notna() & with_labels["label_mature_at"].notna()
        eligible_frame = with_labels.loc[eligible, keep].copy()
        eligible_frame.loc[:, FEATURE_COLUMNS] = eligible_frame[FEATURE_COLUMNS].astype("float32")
        eligible_frame["target"] = eligible_frame["target"].astype("int8")
        outputs.append(eligible_frame)
    out = pd.concat(outputs, ignore_index=True)
    if out.empty:
        raise ValueError("No valid contiguous examples after calculating features/labels")
    return out


def _class_probs(model: object, X: pd.DataFrame) -> np.ndarray:
    if X.empty:
        return np.empty((0, 3), dtype=float)
    raw = model.predict_proba(X)
    probs = np.zeros((len(X), 3), dtype=float)
    probs[:, np.asarray(model.classes_, dtype=int)] = raw
    return probs


def fit_validate(data: pd.DataFrame, *, holdout_symbol: str = "ADAUSDT",
                 n_estimators: int = 350, seed: int = 42) -> tuple[dict, dict, object]:
    """Single purged chronological train/validation/test + completely unseen symbol.

    Test labels and unseen-symbol labels MUST NOT select features/hyperparameters.
    Walk-forward with multiple folds is reserved for the next milestone.
    """
    symbols = sorted(data.symbol.unique().tolist())
    if holdout_symbol not in symbols:
        raise ValueError("Holdout symbol missing")
    if holdout_symbol in REQUIRED_CONTEXT:
        raise ValueError("Holdout must not be BTC/ETH context pair for this P1 protocol")
    seen_symbols = [s for s in symbols if s != holdout_symbol]
    if len(seen_symbols) < 2:
        raise ValueError("Need at least two training symbols")
    dates = boundaries(data, seen_symbols)
    parts = split_with_purge(data, boundaries=dates, holdout_symbol=holdout_symbol)
    if min(len(parts["train"]), len(parts["validation"]), len(parts["seen_test"]),
           len(parts["unseen_test"])) < 100:
        raise ValueError("Insufficient observations in at least one partition; import more history")
    y_train = parts["train"].target.astype(int).to_numpy()
    y_val = parts["validation"].target.astype(int).to_numpy()
    if len(set(y_train)) != 3 or len(set(y_val)) != 3:
        raise ValueError("All three classes must be present in train and validation")
    X_train = feature_matrix(parts["train"])
    X_val = feature_matrix(parts["validation"])
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=n_estimators, learning_rate=0.04, num_leaves=31,
        max_depth=-1, min_child_samples=100, reg_lambda=5.0,
        colsample_bytree=0.80, subsample=0.90, subsample_freq=1,
        random_state=seed, n_jobs=-1, verbosity=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
    baseline = DummyClassifier(strategy="prior")
    baseline.fit(X_train, y_train)
    evaluations = {}
    predicted_rows = []
    for name in ("train", "validation", "seen_test", "unseen_test"):
        frame = parts[name]
        X = feature_matrix(frame)
        truth = frame.target.astype(int).to_numpy()
        probabilities = _class_probs(model, X)
        baseline_probabilities = _class_probs(baseline, X)
        evaluations[name] = {
            "lightgbm": evaluate(truth, probabilities),
            "majority_baseline": evaluate(truth, baseline_probabilities),
            "per_symbol": {
                symbol: evaluate(
                    frame.loc[frame.symbol == symbol, "target"].to_numpy(dtype=int),
                    probabilities[frame.symbol.to_numpy() == symbol],
                ) for symbol in sorted(frame.symbol.unique().tolist())
            },
        }
        if name in ("seen_test", "unseen_test"):
            out = frame[["symbol", "bar_close_at", "label_mature_at", "future_return", "target"]].copy()
            out = out.rename(columns={"target": "actual_class", "future_return": "actual_return"})
            out["p_down"] = probabilities[:, 0]
            out["p_neutral"] = probabilities[:, 1]
            out["p_up"] = probabilities[:, 2]
            out["predicted_class"] = probabilities.argmax(axis=1)
            out["split"] = name
            predicted_rows.append(out)
    # Assert the anti-leakage properties at runtime, not just in tests.
    assert (parts["train"].label_mature_at < dates.validation_start).all()
    assert (parts["validation"].label_mature_at < dates.test_start).all()
    assert (parts["unseen_test"].symbol == holdout_symbol).all()
    assert not parts["train"].symbol.eq(holdout_symbol).any()
    feature_importance = sorted(
        [{"name": name, "split_importance": int(value)}
         for name, value in zip(FEATURE_COLUMNS, model.feature_importances_)],
        key=lambda item: item["split_importance"], reverse=True,
    )
    summary = {
        "model_family": "LightGBM", "task": "direction_classification", "market": "usdt_perpetual",
        "timeframe": "5m", "horizon_bars": None, "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION, "classes": LABEL_NAMES,
        "holdout_symbol": holdout_symbol, "train_symbols": seen_symbols,
        "boundaries_utc": dates.as_dict(),
        "purging": "train/validation labels mature strictly before the next segment",
        "counts": {k: len(v) for k,v in parts.items()},
        "best_iteration": int(model.best_iteration_), "n_estimators_max": n_estimators,
        "lightgbm_seed": seed, "feature_importance": feature_importance,
        "metrics": evaluations,
        "disclaimer": "Research only. No trades, PnL or profitability estimates. Test metrics are not validation feedback.",
    }
    return summary, {"predictions": pd.concat(predicted_rows, ignore_index=True)}, model


def run_training(store, *, symbols=DEFAULT_SYMBOLS, holdout_symbol="ADAUSDT",
                 horizon: int = 6, months: int | None = None,
                 threshold_multiplier: float = 0.5,
                 n_estimators: int = 350, seed: int = 42,
                 artifacts_root: str | Path | None = None,
                 verify_hashes: bool = False) -> Path:
    if holdout_symbol not in symbols or len(symbols) < 4:
        raise ValueError("Holdout symbol must be among at least four symbols")
    if not set(REQUIRED_CONTEXT).issubset(symbols):
        raise ValueError("BTCUSDT/ETHUSDT context required")
    candles = {}
    manifest_all = []
    for symbol in symbols:
        print(f"Reading P0 Parquet: {symbol}", flush=True)
        df, manifests = load_symbol(store, symbol, months=months, verify_hashes=verify_hashes)
        candles[symbol] = df
        manifest_all.extend(manifests)
    print("Building features and future labels...", flush=True)
    examples = prepare_training_data(candles, horizon=horizon,
                                     threshold_multiplier=threshold_multiplier)
    # Release original raw inputs early to keep full 24-month footprint lower.
    del candles
    print(f"Fitting LightGBM on {len(examples):,} eligible examples...", flush=True)
    summary, extras, model = fit_validate(examples, holdout_symbol=holdout_symbol,
                                          n_estimators=n_estimators, seed=seed)
    summary["horizon_bars"] = horizon
    summary["threshold_multiplier"] = threshold_multiplier
    summary["manifest_fingerprint_sha256"] = manifest_fingerprint(manifest_all)
    summary["source_partitions"] = [{"dataset_id": x["dataset_id"],
                                     "sha256": x["parquet_sha256"]} for x in manifest_all]
    summary["environment"] = {"python": platform.python_version(),
                              "lightgbm": lgb.__version__, "sklearn": sklearn.__version__,
                              "pandas": pd.__version__}
    summary["created_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_") + uuid4().hex[:6]
    summary["run_id"] = run_id
    root = Path(artifacts_root) if artifacts_root else store.root.parent / "artifacts"
    run_dir = root / "training_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "evaluation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    extras["predictions"].to_csv(run_dir / "predictions_test.csv.gz", index=False, compression="gzip")
    # Only load internally created/trusted model artifacts with joblib.
    joblib.dump({"estimator": model, "features": FEATURE_COLUMNS,
                 "horizon_bars": horizon, "feature_schema_version": FEATURE_SCHEMA_VERSION,
                 "label_schema_version": LABEL_SCHEMA_VERSION,
                 "manifest_fingerprint_sha256": summary["manifest_fingerprint_sha256"]},
                run_dir / "model.joblib", compress=3)
    print(f"Done: {run_dir.resolve()}", flush=True)
    for split in ("validation", "seen_test", "unseen_test"):
        v = summary["metrics"][split]["lightgbm"]
        base = summary["metrics"][split]["majority_baseline"]
        print(f"{split}: n={v['n']}, balanced_acc={v['balanced_accuracy']}, "
              f"baseline={base['balanced_accuracy']}, macro_f1={v['macro_f1']}", flush=True)
    return run_dir
