"""P2 CPU comparison of H3/H5/H6 with identical observations and fold dates.

The target definitions differ, so raw precision/log-loss are not directly
interchangeable. Report horizon-specific prior baselines, prevalence, uplift,
skill vs train-prior log loss, and direction coverage. Labels mature strictly
before every phase boundary, with conservative purge at the LONGEST horizon.
No fold-OOS scoring, final 90-day scoring, or ADA training/evaluation.
Previously inspected development checks are NOT fresh independent validation.
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
from mlpipeline.evaluation import evaluate
from mlpipeline.features import FEATURE_COLUMNS, feature_matrix
from mlpipeline.trainer import DEFAULT_SYMBOLS, _class_probs, prepare_training_data
from mlpipeline.two_stage_diagnostics import ALPHA_FIXED, _fit_model, _scored_candidate
from mlpipeline.walkforward import make_walk_forward_folds

HORIZONS = (3, 5, 6)
MAX_HORIZON = 6


def align_horizon_examples(examples_by_horizon: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Inner-join targets on identical (symbol, close UTC), with common H6 features.

    The latest label maturity is used to purge all three horizons identically.
    ADA is retained only as an anchor for original fold geometry; later code
    drops the entire ADA slice without training or scoring any ADA label.
    """
    if set(examples_by_horizon) != set(HORIZONS):
        raise ValueError("Expected exactly H3, H5 and H6 datasets")
    required = {"symbol", "bar_close_at", "target", "label_mature_at", *FEATURE_COLUMNS}
    h6 = examples_by_horizon[MAX_HORIZON]
    if h6.empty:
        raise ValueError("Empty H6 data")
    common = h6[["symbol", "bar_close_at", *FEATURE_COLUMNS]].copy()
    for h in HORIZONS:
        frame = examples_by_horizon[h]
        if missing := required.difference(frame.columns):
            raise ValueError(f"H{h}: missing {sorted(missing)}")
        if frame.duplicated(["symbol", "bar_close_at"]).any():
            raise ValueError(f"H{h}: duplicate sample keys")
        if frame.target.isna().any() or frame.label_mature_at.isna().any():
            raise ValueError(f"H{h}: incomplete labels")
        if not isinstance(frame.bar_close_at.dtype, pd.DatetimeTZDtype):
            raise ValueError("UTC-aware close timestamps required")
        if str(frame.bar_close_at.dt.tz) != "UTC":
            raise ValueError("UTC-aware close timestamps required")
        if not isinstance(frame.label_mature_at.dtype, pd.DatetimeTZDtype):
            raise ValueError("UTC-aware label maturity required")
        if str(frame.label_mature_at.dt.tz) != "UTC":
            raise ValueError("UTC-aware label maturity required")
        # The horizon label is exactly H future M5 closes.
        if not (frame.label_mature_at ==
                frame.bar_close_at + pd.Timedelta(minutes=5 * h)).all():
            raise ValueError(f"H{h}: invalid label maturation")
        labels = frame[["symbol", "bar_close_at", "target"]].rename(
            columns={"target": f"target_h{h}"}
        )
        common = common.merge(
            labels, on=["symbol", "bar_close_at"], how="inner",
            validate="one_to_one", sort=False,
        )
    if common.empty or not common.symbol.eq("ADAUSDT").any():
        raise ValueError("No common rows or missing reserved ADA anchor")
    common["label_mature_at"] = (
        common.bar_close_at + pd.Timedelta(minutes=5 * MAX_HORIZON)
    )
    common["target"] = common[f"target_h{MAX_HORIZON}"]
    for h in HORIZONS:
        common[f"target_h{h}"] = common[f"target_h{h}"].astype("int8")
    common["target"] = common["target"].astype("int8")
    if common.duplicated(["symbol", "bar_close_at"]).any():
        raise AssertionError("Duplicate common observation")
    return common


def common_horizon_partitions(
    aligned: pd.DataFrame, fold, *,
    holdout_symbol: str = "ADAUSDT",
    validation_days: int = 14,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Shared chronological masks and H6 maturity purging for ALL horizons."""
    if holdout_symbol != "ADAUSDT" or validation_days != 14:
        raise ValueError("Protocol requires reserved ADA and 14-day validation")
    seen = aligned.loc[aligned.symbol.ne(holdout_symbol)]
    t, mature = seen.bar_close_at, seen.label_mature_at
    stop_end = fold.validation_start + pd.Timedelta(days=7)
    tune_end = stop_end + pd.Timedelta(days=3)
    masks = {
        "train": t.ge(fold.train_start) & t.lt(fold.validation_start)
                 & mature.lt(fold.validation_start),
        "early_stop": t.ge(fold.validation_start) & t.lt(stop_end)
                      & mature.lt(stop_end),
        "check": t.ge(tune_end) & t.lt(fold.test_start)
                 & mature.lt(fold.test_start),
    }
    parts = {name: seen.loc[mask] for name, mask in masks.items()}
    if min(map(len, parts.values())) < 100:
        raise ValueError(f"Fold {fold.fold_id}: missing shared observations")
    for key, end in (
        ("train", fold.validation_start),
        ("early_stop", stop_end),
        ("check", fold.test_start),
    ):
        if not parts[key].label_mature_at.lt(end).all():
            raise AssertionError(f"{key}: label maturity leak")
    if parts["train"].bar_close_at.max() >= fold.validation_start:
        raise AssertionError("Training includes validation observations")
    return parts, {
        "train_start_utc": fold.train_start.isoformat(),
        "validation_start_utc": fold.validation_start.isoformat(),
        "early_stop_end_exclusive_utc": stop_end.isoformat(),
        "unused_tuning_end_exclusive_utc": tune_end.isoformat(),
        "validation_check_end_exclusive_utc": fold.test_start.isoformat(),
    }


def compare_horizons(
    aligned: pd.DataFrame, *,
    holdout_symbol: str = "ADAUSDT",
    train_days: int = 120,
    validation_days: int = 14,
    test_days: int = 14,
    step_days: int = 14,
    final_holdout_days: int = 90,
    n_estimators: int = 150,
    seed: int = 42,
) -> dict:
    """Same folds and sample keys, horizon-specific labels and priors."""
    if (holdout_symbol != "ADAUSDT"
            or (train_days, validation_days, test_days, step_days,
                final_holdout_days) != (120, 14, 14, 14, 90)
            or n_estimators < 20):
        raise ValueError("P2 fixed protocol: 120/14/14/14 and final 90 days")
    folds = make_walk_forward_folds(
        aligned, holdout_symbol=holdout_symbol,
        train_days=train_days, validation_days=validation_days,
        test_days=test_days, step_days=step_days,
        final_holdout_days=final_holdout_days,
    )
    results = []
    for fold in folds:
        parts, boundaries = common_horizon_partitions(aligned, fold)
        train, early, check = (
            parts["train"], parts["early_stop"], parts["check"]
        )
        X_check = feature_matrix(check)
        scores = {}
        for h in HORIZONS:
            column = f"target_h{h}"
            y_train = train[column].to_numpy(dtype=int)
            y_early = early[column].to_numpy(dtype=int)
            y_check = check[column].to_numpy(dtype=int)
            counts = np.bincount(y_train, minlength=3)
            if (counts == 0).any() or len(set(y_early)) != 3:
                raise ValueError(f"Fold {fold.fold_id}, H{h}: missing classes")

            model = _fit_model(
                train, early,
                train_target=y_train, early_target=y_early,
                binary=False, n_estimators=n_estimators, seed=seed,
            )
            predictions = _class_probs(model, X_check)
            metrics = _scored_candidate(y_check, predictions, counts)
            train_prior = counts.astype(float) / counts.sum()
            prior_probs = np.broadcast_to(train_prior, predictions.shape)
            prior_eval = evaluate(y_check, prior_probs)
            model_ll = metrics["raw_probabilities"]["log_loss"]
            prior_ll = prior_eval["log_loss"]
            skills = {
                "train_prior_log_loss": prior_ll,
                "log_loss_skill_vs_train_prior": (
                    1.0 - model_ll / prior_ll if prior_ll > 0 else None
                ),
                "train_prior_multiclass_brier": prior_eval["multiclass_brier"],
                "brier_improvement_vs_train_prior": (
                    prior_eval["multiclass_brier"] -
                    metrics["raw_probabilities"]["multiclass_brier"]
                ),
            }
            per_symbol = {}
            for symbol in sorted(check.symbol.unique()):
                mask = check.symbol.to_numpy() == symbol
                per_symbol[symbol] = _scored_candidate(
                    y_check[mask], predictions[mask], counts
                )["directional"]
            scores[str(h)] = {
                "train_class_counts": {
                    label: int(counts[i])
                    for i, label in enumerate(("DOWN", "NEUTRAL", "UP"))
                },
                "check_class_counts": {
                    label: int((y_check == i).sum())
                    for i, label in enumerate(("DOWN", "NEUTRAL", "UP"))
                },
                "best_iteration": int(model.best_iteration_),
                "metrics": metrics,
                "prior_baseline": skills,
                "per_symbol_directional": per_symbol,
            }
            d = metrics["directional"]
            print(
                f"fold={fold.fold_id} H{h} "
                f"precision={d['directional_precision'] if d['directional_precision'] is not None else float('nan'):.4f} "
                f"coverage={d['coverage']:.4f} "
                f"uplift_points={d['uplift_points'] if d['uplift_points'] is not None else float('nan'):.3f} "
                f"logloss_skill={skills['log_loss_skill_vs_train_prior']:.5f}",
                flush=True,
            )
        results.append({
            "fold_id": fold.fold_id,
            **boundaries,
            "common_partition_counts": {k: len(v) for k, v in parts.items()},
            "horizons": scores,
        })

    def mean(values) -> float | None:
        seq = list(values)
        if not seq or any(x is None for x in seq):
            return None
        return float(np.mean(seq))

    summary = {}
    for h in HORIZONS:
        key = str(h)
        items = [f["horizons"][key] for f in results]
        summary[key] = {
            "folds": len(items),
            "minutes_ahead": h * 5,
            "directional_precision_mean": mean(
                x["metrics"]["directional"]["directional_precision"]
                for x in items
            ),
            "directional_coverage_mean": mean(
                x["metrics"]["directional"]["coverage"]
                for x in items
            ),
            "directional_uplift_points_mean": mean(
                x["metrics"]["directional"]["uplift_points"]
                for x in items
            ),
            "conditional_directional_uplift_points_mean": mean(
                x["metrics"]["directional"]["conditional_uplift_points"]
                for x in items
            ),
            "macro_f1_mean": mean(
                x["metrics"]["fixed_alpha_decisions"]["macro_f1"]
                for x in items
            ),
            "balanced_accuracy_mean": mean(
                x["metrics"]["fixed_alpha_decisions"]["balanced_accuracy"]
                for x in items
            ),
            "raw_log_loss_mean": mean(
                x["metrics"]["raw_probabilities"]["log_loss"]
                for x in items
            ),
            "train_prior_log_loss_mean": mean(
                x["prior_baseline"]["train_prior_log_loss"]
                for x in items
            ),
            "log_loss_skill_vs_train_prior_mean": mean(
                x["prior_baseline"]["log_loss_skill_vs_train_prior"]
                for x in items
            ),
            "brier_improvement_vs_train_prior_mean": mean(
                x["prior_baseline"]["brier_improvement_vs_train_prior"]
                for x in items
            ),
            "class_distribution_mean": {
                cls: mean(
                    x["check_class_counts"][cls] /
                    sum(x["check_class_counts"].values())
                    for x in items
                )
                for cls in ("DOWN", "NEUTRAL", "UP")
            },
        }
    return {
        "protocol": "cpu_p2_h3_h5_h6_shared_observations_v1",
        "engine": "cpu",
        "horizons_predeclared": HORIZONS,
        "alpha_fixed": ALPHA_FIXED,
        "feature_count": len(FEATURE_COLUMNS),
        "training_days_fixed": train_days,
        "label_threshold_multiplier_fixed": 0.5,
        "holdout_symbol_excluded_from_all_scoring": holdout_symbol,
        "fold_oos_tests_scored": False,
        "final_90_days_scored": False,
        "tuning_days_unused": True,
        "previously_inspected_dev_check_not_independent": True,
        "comparability_note": (
            "All horizons use identical keys, feature vectors, and partitions. "
            "Target distributions and tasks differ: compare frequency-adjusted "
            "directional uplift and log-loss skill against each horizon's own "
            "train-prior baseline, not just raw precision or raw log loss."
        ),
        "warnings": [
            "Three horizons are research candidates; no production horizon change.",
            "Historical M5 samples and folds are correlated; no independent confirmation.",
            "No trading execution or economic backtest.",
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
        "folds": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P2 CPU compare H3/H5/H6 on exactly shared development observations"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    if args.months != 24:
        parser.error("P2 frozen-history comparison requires 24 months")
    if args.output.exists():
        parser.error("Refusing to overwrite prior research results")
    store = CandleStore(args.data_root)
    candles, manifests = {}, []
    for symbol in DEFAULT_SYMBOLS:
        print(f"Reading {symbol}...", flush=True)
        candles[symbol], entries = load_symbol(
            store, symbol, months=args.months, verify_hashes=args.verify_hashes,
        )
        manifests.extend(entries)
    horizon_examples = {}
    for h in HORIZONS:
        print(f"Building H{h} labels/features...", flush=True)
        examples = prepare_training_data(candles, horizon=h)
        # Keep features only for H6, thus avoiding 3 complete feature copies.
        if h == MAX_HORIZON:
            horizon_examples[h] = examples
        else:
            horizon_examples[h] = examples[[
                "symbol", "bar_close_at", "target", "label_mature_at", *FEATURE_COLUMNS,
            ]]
        # Previous epochs can be GC'd after alignment, not scored or persisted.
    del candles
    aligned = align_horizon_examples(horizon_examples)
    del horizon_examples
    report = compare_horizons(
        aligned, n_estimators=args.n_estimators, seed=args.seed
    )
    report["dataset_fingerprint_sha256"] = manifest_fingerprint(manifests)
    report["lightgbm_version"] = lgb.__version__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(f"Multi-horizon report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
