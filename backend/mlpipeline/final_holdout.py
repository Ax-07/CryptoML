"""Frozen CPU holdout protocol: audit first, score final 90 days only on explicit unseal.

Development results MUST NOT be used to revise these settings after unsealing.
ADA labels are never used for fitting, early stopping, or decision selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ingestion.storage import CandleStore
from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.decision_diagnostics import ALPHAS, decide, decision_metrics
from mlpipeline.evaluation import evaluate
from mlpipeline.features import feature_matrix
from mlpipeline.trainer import DEFAULT_SYMBOLS, _class_probs, prepare_training_data


PROTOCOL = "frozen_cpu_final_holdout_v1"
FINAL_SETTINGS = {
    "months": 24,
    "horizon": 6,  # Six 5-minute bars: 30 minutes.
    "threshold_multiplier": 0.5,
    "holdout_symbol": "ADAUSDT",
    "train_days": 120,
    "validation_days": 14,
    "final_holdout_days": 90,
    "n_estimators": 150,
    "seed": 42,
    "alpha_grid": list(ALPHAS),
    "logistic_C": 1.0,
    "logistic_max_iter": 1000,
}


def _boundaries(examples: pd.DataFrame, *, train_days: int, validation_days: int,
                final_holdout_days: int, holdout_symbol: str) -> dict:
    if validation_days < 4:
        raise ValueError("Need at least 4 validation days for distinct partitions")
    seen = examples.loc[examples.symbol != holdout_symbol]
    if seen.empty or not examples.symbol.eq(holdout_symbol).any():
        raise ValueError("Missing seen or unseen-symbol examples")
    if not isinstance(examples.bar_close_at.dtype, pd.DatetimeTZDtype):
        raise ValueError("UTC timezone is required")
    if str(examples.bar_close_at.dt.tz) != "UTC":
        raise ValueError("UTC timezone is required")
    final_end = seen.bar_close_at.max() + pd.Timedelta(minutes=5)
    final_start = final_end - pd.Timedelta(days=final_holdout_days)
    validation_start = final_start - pd.Timedelta(days=validation_days)
    train_start = validation_start - pd.Timedelta(days=train_days)
    stop_end = validation_start + pd.Timedelta(days=validation_days // 2)
    tune_end = stop_end + pd.Timedelta(
        days=(validation_days - validation_days // 2) // 2
    )
    return {
        "train_start": train_start, "validation_start": validation_start,
        "early_stop_end": stop_end, "decision_tune_end": tune_end,
        "final_start": final_start, "final_end": final_end,
    }


def _partitions(examples: pd.DataFrame, settings: dict) -> tuple[dict, dict]:
    b = _boundaries(
        examples, train_days=settings["train_days"],
        validation_days=settings["validation_days"],
        final_holdout_days=settings["final_holdout_days"],
        holdout_symbol=settings["holdout_symbol"],
    )
    t, m = examples.bar_close_at, examples.label_mature_at
    seen = examples.symbol.ne(settings["holdout_symbol"])
    masks = {
        "train": seen & t.ge(b["train_start"]) & t.lt(b["validation_start"])
        & m.lt(b["validation_start"]),
        "early_stopping": seen & t.ge(b["validation_start"])
        & t.lt(b["early_stop_end"]) & m.lt(b["early_stop_end"]),
        "decision_tuning": seen & t.ge(b["early_stop_end"])
        & t.lt(b["decision_tune_end"]) & m.lt(b["decision_tune_end"]),
        "validation_check": seen & t.ge(b["decision_tune_end"])
        & t.lt(b["final_start"]) & m.lt(b["final_start"]),
        "seen_final": seen & t.ge(b["final_start"]) & t.lt(b["final_end"])
        & m.lt(b["final_end"]),
        "ada_final": ~seen & t.ge(b["final_start"]) & t.lt(b["final_end"])
        & m.lt(b["final_end"]),
    }
    partitions = {
        key: examples.loc[mask].copy().reset_index(drop=True)
        for key, mask in masks.items()
    }
    if min(map(len, partitions.values())) < 100:
        raise ValueError("Fewer than 100 observations in at least one partition")
    if any(part.symbol.eq(settings["holdout_symbol"]).any()
           for name, part in partitions.items() if name != "ada_final"):
        raise AssertionError("ADA leaked into development")
    for name, end in [
        ("train", b["validation_start"]),
        ("early_stopping", b["early_stop_end"]),
        ("decision_tuning", b["decision_tune_end"]),
        ("validation_check", b["final_start"]),
        ("seen_final", b["final_end"]),
        ("ada_final", b["final_end"]),
    ]:
        if not partitions[name].label_mature_at.lt(end).all():
            raise AssertionError(f"{name}: a label matures outside its partition")
    if partitions["seen_final"].symbol.eq(settings["holdout_symbol"]).any():
        raise AssertionError("Unseen symbol leaked into seen final")
    if not partitions["ada_final"].symbol.eq(settings["holdout_symbol"]).all():
        raise AssertionError("Unseen final test contains seen symbols")
    if not pd.concat(
        [partitions["seen_final"], partitions["ada_final"]]
    ).duplicated(["symbol", "bar_close_at"]).sum() == 0:
        raise AssertionError("Duplicate final symbol/timestamp")
    return b, partitions


def audit_final_protocol(examples: pd.DataFrame, dataset_fingerprint: str,
                         settings: dict = FINAL_SETTINGS) -> dict:
    """Describe final partitions WITHOUT computing any holdout metric."""
    boundaries, parts = _partitions(examples, settings)
    return {
        "protocol": PROTOCOL,
        "status": "SEALED_AUDIT_ONLY",
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "parameters": dict(settings),
        "boundaries_utc": {
            key: timestamp.isoformat() for key, timestamp in boundaries.items()
        },
        "partition_counts": {key: len(part) for key, part in parts.items()},
        "notes": [
            "No final labels are scored in this audit.",
            "Final seen and ADA are disjoint; ADA never trains or tunes a model.",
            "Train-only class priors; 7/3/4-day chronological validation for 14-day settings.",
            "Raw probabilities are scored without decision correction.",
            "No model/parameter changes permitted after final unseal.",
        ],
    }


def _candidate(model, frame: pd.DataFrame, alpha: float,
               train_counts: np.ndarray) -> dict:
    probs = _class_probs(model, feature_matrix(frame))
    actual = frame.target.to_numpy(dtype=int)
    return {
        "probability_metrics": evaluate(actual, probs),
        "standard_decision": decision_metrics(
            actual, decide(probs, train_counts, 0.0)
        ),
        "adjusted_decision": decision_metrics(
            actual, decide(probs, train_counts, alpha)
        ),
    }


def evaluate_frozen_final(examples: pd.DataFrame, audit: dict,
                          dataset_fingerprint: str,
                          settings: dict = FINAL_SETTINGS) -> dict:
    """Unseal once after matching a prior audit and frozen settings."""
    expected = audit_final_protocol(examples, dataset_fingerprint, settings)
    if audit != expected:
        raise ValueError("Audit differs from dataset, boundaries or frozen parameters")
    boundaries, p = _partitions(examples, settings)
    train = p["train"]
    x_train = feature_matrix(train)
    y_train = train.target.to_numpy(dtype=int)
    y_stop = p["early_stopping"].target.to_numpy(dtype=int)
    y_tune = p["decision_tuning"].target.to_numpy(dtype=int)
    counts = np.bincount(y_train, minlength=3)
    if (counts == 0).any() or len(set(y_stop)) < 3 or len(set(y_tune)) < 3:
        raise ValueError("Missing training/early-stop/tuning class")
    cfg = settings
    lgbm = lgb.LGBMClassifier(
        device_type="cpu", objective="multiclass", num_class=3,
        n_estimators=cfg["n_estimators"], learning_rate=0.04,
        num_leaves=31, max_depth=-1, min_child_samples=100,
        reg_lambda=5.0, colsample_bytree=0.8, subsample=0.9,
        subsample_freq=1, max_bin=63, random_state=cfg["seed"],
        n_jobs=8, verbosity=-1,
    )
    lgbm.fit(
        x_train, y_train,
        eval_set=[(feature_matrix(p["early_stopping"]), y_stop)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    prior = DummyClassifier(strategy="prior")
    prior.fit(x_train, y_train)
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=cfg["logistic_C"], max_iter=cfg["logistic_max_iter"],
            solver="lbfgs", random_state=cfg["seed"],
        ),
    )
    logistic.fit(x_train, y_train)

    # Independently tune the SAME predeclared decision grid for both models.
    # No final labels are touched until both decision rules are selected.
    tuned = {}
    for name, model in (("lightgbm", lgbm), ("logistic", logistic)):
        probabilities = _class_probs(model, feature_matrix(p["decision_tuning"]))
        alphas = {
            str(alpha): decision_metrics(
                y_tune, decide(probabilities, counts, alpha)
            )["macro_f1"]
            for alpha in cfg["alpha_grid"]
        }
        chosen = max(cfg["alpha_grid"], key=lambda a: (alphas[str(a)], -a))
        tuned[name] = {"alpha": chosen, "selection_f1": alphas}

    # Validation check is observational only; it does not change either alpha.
    validation_check = {
        name: _candidate(model, p["validation_check"], tuned[name]["alpha"], counts)
        for name, model in (("lightgbm", lgbm), ("logistic", logistic))
    }
    output = {}
    for split in ("seen_final", "ada_final"):
        part = p[split]
        output[split] = {
            "lightgbm": _candidate(lgbm, part, tuned["lightgbm"]["alpha"], counts),
            "logistic": _candidate(logistic, part, tuned["logistic"]["alpha"], counts),
            "class_prior_baseline": {
                "probability_metrics": evaluate(
                    part.target.to_numpy(dtype=int),
                    _class_probs(prior, feature_matrix(part)),
                ),
            },
            "per_symbol": {
                symbol: _candidate(
                    lgbm, part.loc[part.symbol.eq(symbol)],
                    tuned["lightgbm"]["alpha"], counts
                )
                for symbol in sorted(part.symbol.unique())
            },
        }
    return {
        "protocol": PROTOCOL,
        "status": "FINAL_UNSEALED_EVALUATED",
        "audit": expected,
        "lightgbm_best_iteration": int(lgbm.best_iteration_),
        "chosen_alphas_on_pre_final_tuning_only": tuned,
        "pre_final_validation_check": validation_check,
        "final_metrics": output,
        "disclaimer": (
            "One-off retrospective holdout evaluation, not proven tradability. "
            "Class decisions are based on prior-adjusted scores, not calibrated probabilities."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit or explicitly unseal frozen CPU final holdouts")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-hashes", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--unseal-final", action="store_true")
    parser.add_argument("--audit-report", type=Path)
    args = parser.parse_args()
    if args.unseal_final and not args.audit_report:
        parser.error("--unseal-final requires --audit-report")
    if args.audit_only and args.audit_report:
        parser.error("--audit-report is not needed for --audit-only")
    # Avoid a partial run consuming the holdout then failing to persist its report.
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    store = CandleStore(args.data_root)
    candles, manifests = {}, []
    for symbol in DEFAULT_SYMBOLS:
        print(f"Reading verified data: {symbol}", flush=True)
        candles[symbol], included = load_symbol(
            store, symbol, months=FINAL_SETTINGS["months"],
            verify_hashes=args.verify_hashes,
        )
        manifests.extend(included)
    examples = prepare_training_data(
        candles, horizon=FINAL_SETTINGS["horizon"],
        threshold_multiplier=FINAL_SETTINGS["threshold_multiplier"],
    )
    fingerprint = manifest_fingerprint(manifests)
    if args.audit_only:
        report = audit_final_protocol(examples, fingerprint)
    else:
        prior_audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
        report = evaluate_frozen_final(examples, prior_audit, fingerprint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(f"Status: {report['status']}")
    if args.audit_only:
        print("Final window (UTC):", report["boundaries_utc"]["final_start"],
              "through", report["boundaries_utc"]["final_end"])
        print("Partitions:", report["partition_counts"])
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
