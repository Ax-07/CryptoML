"""Rebuild and freeze the exact July-2026 CPU LightGBM final candidate.

No post-final labels or ADA labels may influence training or alpha selection.
Requires matching pre-existing final report and data hashes. Model export is
offline and does not simulate/predict any retrospective holdout outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ingestion.storage import CandleStore
from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.decision_diagnostics import decide, decision_metrics
from mlpipeline.features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, feature_matrix
from mlpipeline.final_holdout import FINAL_SETTINGS, _partitions, audit_final_protocol
from mlpipeline.labels import LABEL_SCHEMA_VERSION
from mlpipeline.trainer import DEFAULT_SYMBOLS, _class_probs, prepare_training_data


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rebuild_frozen_lightgbm(
    examples: pd.DataFrame, final_report: dict, fingerprint: str, *,
    settings: dict = FINAL_SETTINGS,
) -> tuple[lgb.Booster, dict]:
    """Rebuild under exactly the pre-final boundaries; check frozen metrics."""
    if final_report.get("status") != "FINAL_UNSEALED_EVALUATED":
        raise ValueError("Expected a completed frozen final evaluation report")
    expected_audit = audit_final_protocol(examples, fingerprint, settings)
    if final_report.get("audit") != expected_audit:
        raise ValueError("Dataset fingerprint, settings or boundaries differ from frozen audit")
    b, parts = _partitions(examples, settings)
    train = parts["train"]
    stop = parts["early_stopping"]
    tune = parts["decision_tuning"]
    y_train = train.target.to_numpy(dtype=int)
    y_stop = stop.target.to_numpy(dtype=int)
    y_tune = tune.target.to_numpy(dtype=int)
    counts = np.bincount(y_train, minlength=3)
    if (counts <= 0).any() or len(set(y_stop)) != 3 or len(set(y_tune)) != 3:
        raise ValueError("Missing class in training, early stopping or tuning")
    model = lgb.LGBMClassifier(
        device_type="cpu", objective="multiclass", num_class=3,
        n_estimators=settings["n_estimators"], learning_rate=0.04,
        num_leaves=31, max_depth=-1, min_child_samples=100,
        reg_lambda=5.0, colsample_bytree=0.8, subsample=0.9,
        subsample_freq=1, max_bin=63, random_state=settings["seed"],
        n_jobs=8, verbosity=-1,
    )
    model.fit(
        feature_matrix(train), y_train,
        eval_set=[(feature_matrix(stop), y_stop)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    if int(model.best_iteration_) != int(final_report["lightgbm_best_iteration"]):
        raise ValueError("Rebuilt model does not match frozen best iteration")
    expected_alpha = float(
        final_report["chosen_alphas_on_pre_final_tuning_only"]["lightgbm"]["alpha"]
    )
    p_tune = _class_probs(model, feature_matrix(tune))
    scores = {
        str(alpha): decision_metrics(
            y_tune, decide(p_tune, counts, alpha)
        )["macro_f1"]
        for alpha in settings["alpha_grid"]
    }
    selected_alpha = max(
        settings["alpha_grid"], key=lambda alpha: (scores[str(alpha)], -alpha)
    )
    if float(selected_alpha) != expected_alpha:
        raise ValueError("Rebuilt model does not match the frozen selected alpha")
    historical_scores = final_report[
        "chosen_alphas_on_pre_final_tuning_only"
    ]["lightgbm"]["selection_f1"]
    if not all(abs(scores[str(a)] - historical_scores[str(a)]) < 1e-8
               for a in settings["alpha_grid"]):
        raise ValueError("Rebuilt model differs from frozen tuning metrics")
    booster = model.booster_
    if list(booster.feature_name()) != list(FEATURE_COLUMNS):
        raise ValueError("Feature order differs from frozen LightGBM feature schema")
    prior = counts.astype(float) / counts.sum()
    metadata = {
        "protocol": "frozen_cpu_lightgbm_export_v1",
        "source_protocol": final_report["audit"]["protocol"],
        "model_family": "lightgbm",
        "model_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_final_dataset_sha256": fingerprint,
        "alpha": float(expected_alpha),
        "class_names": ["DOWN", "NEUTRAL", "UP"],
        "class_prior": [float(x) for x in prior],
        "train_class_counts": counts.tolist(),
        "training_start_utc": b["train_start"].isoformat(),
        "fitted_labels_before_utc": b["validation_start"].isoformat(),
        "model_selection_labels_before_utc": b["decision_tune_end"].isoformat(),
        "final_holdout_start_utc": b["final_start"].isoformat(),
        "final_holdout_end_utc": b["final_end"].isoformat(),
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "horizon_bars": settings["horizon"],
        "threshold_multiplier": settings["threshold_multiplier"],
        "best_iteration": int(model.best_iteration_),
        "seed": int(settings["seed"]),
        "lightgbm_version": lgb.__version__,
    }
    return booster, metadata


def export_bundle(
    booster: lgb.Booster, metadata: dict, outdir: Path,
    final_report_bytes: bytes,
) -> dict:
    """Write a new, content-identified bundle without overwriting any model."""
    if outdir.exists():
        raise FileExistsError(f"Model export already exists: {outdir}")
    blob = booster.model_to_string(num_iteration=metadata["best_iteration"]).encode("utf-8")
    model_hash = sha256(blob)
    meta = dict(metadata)
    meta["model_sha256"] = model_hash
    meta["frozen_final_report_sha256"] = sha256(final_report_bytes)
    meta["model_id"] = "cryptoml-cpu-lgbm-" + model_hash[:16]
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "model.txt").write_bytes(blob)
    (outdir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


def load_bundle(folder: Path) -> tuple[lgb.Booster, dict]:
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("protocol") != "frozen_cpu_lightgbm_export_v1":
        raise ValueError("Unrecognized frozen model protocol")
    blob = (folder / "model.txt").read_bytes()
    if sha256(blob) != meta["model_sha256"]:
        raise ValueError("Frozen model hash mismatch")
    if meta["model_id"] != "cryptoml-cpu-lgbm-" + sha256(blob)[:16]:
        raise ValueError("Frozen model ID mismatch")
    if meta["feature_columns"] != list(FEATURE_COLUMNS):
        raise ValueError("Frozen feature schema mismatch")
    if meta["model_family"] != "lightgbm":
        raise ValueError("Unsupported model family")
    if meta["class_names"] != ["DOWN", "NEUTRAL", "UP"]:
        raise ValueError("Frozen class order mismatch")
    prior = np.asarray(meta["class_prior"], dtype=float)
    if prior.shape != (3,) or (prior <= 0).any() or not np.isclose(prior.sum(), 1):
        raise ValueError("Invalid frozen training prior")
    booster = lgb.Booster(model_str=blob.decode("utf-8"))
    if booster.feature_name() != list(FEATURE_COLUMNS):
        raise ValueError("Saved model feature order mismatch")
    return booster, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Export final LightGBM without changing frozen protocol")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--frozen-final-report", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if args.outdir.exists():
        parser.error(f"Model directory already exists: {args.outdir}")
    report_bytes = args.frozen_final_report.read_bytes()
    report = json.loads(report_bytes)
    store = CandleStore(args.data_root)
    candles, manifests = {}, []
    for symbol in DEFAULT_SYMBOLS:
        print(f"Loading verified frozen history: {symbol}", flush=True)
        candles[symbol], entries = load_symbol(
            store, symbol, months=FINAL_SETTINGS["months"], verify_hashes=True
        )
        manifests.extend(entries)
    examples = prepare_training_data(
        candles, horizon=FINAL_SETTINGS["horizon"],
        threshold_multiplier=FINAL_SETTINGS["threshold_multiplier"],
    )
    del candles
    booster, metadata = rebuild_frozen_lightgbm(
        examples, report, manifest_fingerprint(manifests)
    )
    saved = export_bundle(booster, metadata, args.outdir, report_bytes)
    print(f"Frozen model saved: {args.outdir.resolve()}")
    print(f"model_id: {saved['model_id']}")
    print(f"alpha: {saved['alpha']}")
    print(f"best_iteration: {saved['best_iteration']}")
    print("No final holdout or ADA outcomes were scored during export.")


if __name__ == "__main__":
    main()
