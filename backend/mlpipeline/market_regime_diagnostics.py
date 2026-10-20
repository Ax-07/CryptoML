"""P2 CPU market-regime diagnostics for H2 and H6 on shared dev checks.

Describe UTC hours/6h blocks, weekday vs weekend and train-fitted
asset-specific trailing-volatility terciles. All grouping uses only
timestamp or information available at prediction time. Models are trained
on 120d and early-stopped on 7d; 3d tune unused, 4d checks descriptive.
Identical H12-purged rows, no ADA, fold OOS or final-90 scoring.
Previously seen dev checks are not independent evidence or trading rules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ingestion.storage import CandleStore
from mlpipeline.confidence_diagnostics import directional_metrics
from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.decision_diagnostics import decide
from mlpipeline.expanded_horizon_diagnostics import (
    SEEN_SYMBOLS, build_shared_examples, common_folds, shared_partitions,
)
from mlpipeline.features import FEATURE_COLUMNS, feature_matrix
from mlpipeline.trainer import _class_probs
from mlpipeline.two_stage_diagnostics import ALPHA_FIXED, _fit_model

HORIZONS = (2, 6)
SYMBOLS = ("ALL", *SEEN_SYMBOLS)
VOLATILITY_FEATURE = "volatility_48"
VOL_LABELS = ("low", "medium", "high")
REGIMES = {
    "hour_utc": tuple(f"{h:02d}" for h in range(24)),
    "utc_block_6h": ("00-05", "06-11", "12-17", "18-23"),
    "day_type": ("weekday", "weekend"),
    "volatility_48_train_tercile": VOL_LABELS,
}
MIN_SELECTED_PER_FOLD = 20


def fit_volatility_cutoffs(train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Fit each asset's 1/3 and 2/3 volatility quantiles on TRAIN alone."""
    if (set(train.symbol.unique()) != set(SEEN_SYMBOLS)
            or VOLATILITY_FEATURE not in train.columns):
        raise ValueError("Expected four seen assets and volatility_48")
    result = {}
    for symbol in SEEN_SYMBOLS:
        vol = train.loc[
            train.symbol.eq(symbol), VOLATILITY_FEATURE
        ].to_numpy(dtype=float)
        if len(vol) < 100 or not np.isfinite(vol).all():
            raise ValueError(f"Invalid past volatility samples for {symbol}")
        q1, q2 = np.quantile(vol, [1 / 3, 2 / 3])
        if not 0 <= q1 < q2:
            raise ValueError(f"Indistinguishable volatility terciles: {symbol}")
        result[symbol] = (float(q1), float(q2))
    return result


def assign_regimes(
    check: pd.DataFrame, cutoffs: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    """No check labels or hindsight quantiles are used to assign a regime."""
    if set(cutoffs) != set(SEEN_SYMBOLS):
        raise ValueError("All four train-only volatility cutoffs required")
    if (check.empty or check.symbol.eq("ADAUSDT").any()
            or not isinstance(check.bar_close_at.dtype, pd.DatetimeTZDtype)
            or str(check.bar_close_at.dt.tz) != "UTC"):
        raise ValueError("Expected nonempty seen-only UTC check")
    hours = check.bar_close_at.dt.hour.to_numpy()
    dow = check.bar_close_at.dt.dayofweek.to_numpy()
    symbol_ids = check.symbol.to_numpy()
    vol = check[VOLATILITY_FEATURE].to_numpy(dtype=float)
    if not np.isfinite(vol).all():
        raise ValueError("Check volatility must be finite")
    bins = np.empty(len(check), dtype=object)
    for symbol in SEEN_SYMBOLS:
        mask = symbol_ids == symbol
        boundaries = cutoffs[symbol]
        if not 0 <= boundaries[0] < boundaries[1]:
            raise ValueError("Invalid volatility boundaries")
        bins[mask] = np.asarray(VOL_LABELS, dtype=object)[
            np.searchsorted(boundaries, vol[mask], side="right")
        ]
    if not np.isin(symbol_ids, SEEN_SYMBOLS).all():
        raise ValueError("Unknown symbol in check")
    return {
        "hour_utc": np.asarray([f"{h:02d}" for h in hours], dtype=object),
        "utc_block_6h": np.asarray(
            [REGIMES["utc_block_6h"][h // 6] for h in hours], dtype=object
        ),
        "day_type": np.where(dow < 5, "weekday", "weekend"),
        "volatility_48_train_tercile": bins,
    }


def slice_scores(
    truth: np.ndarray, predicted: np.ndarray, included: np.ndarray,
) -> dict | None:
    """Observed labels read ONLY to compute descriptive check metrics.

    Random reference is recalculated inside each regime (not global).
    No selection or threshold tuning is performed from these metrics.
    """
    if included.dtype != bool or included.shape != truth.shape:
        raise ValueError("Invalid regime mask")
    if not included.any():
        return None
    y = truth[included]
    p = predicted[included]
    metric = directional_metrics(
        y, p, np.zeros(len(y)), threshold=0.0,
    )
    neutral = metric["neutral_among_selected"]
    metric["neutral_fraction_among_selected"] = (
        neutral / metric["selected"] if metric["selected"] else None
    )
    metric["check_class_counts"] = {
        cls: int((y == idx).sum())
        for idx, cls in enumerate(("DOWN", "NEUTRAL", "UP"))
    }
    return metric


def score_regime_cells(
    check: pd.DataFrame,
    assignments: dict[str, np.ndarray],
    truth: np.ndarray, predicted: np.ndarray,
) -> dict:
    """Partition each regime for all assets pooled and each asset separately."""
    n = len(check)
    if truth.shape != (n,) or predicted.shape != (n,):
        raise ValueError("Truth/prediction length mismatch")
    result = {}
    symbols = check.symbol.to_numpy()
    for dimension, labels in REGIMES.items():
        groups = {}
        current = assignments[dimension]
        if current.shape != (n,) or not np.isin(current, labels).all():
            raise ValueError(f"Invalid {dimension} categories")
        for category in labels:
            category_mask = current == category
            by_symbol = {}
            for symbol in SYMBOLS:
                mask = category_mask if symbol == "ALL" else (
                    category_mask & (symbols == symbol)
                )
                by_symbol[symbol] = slice_scores(truth, predicted, mask)
            groups[category] = by_symbol
        result[dimension] = groups
    return result


def summarize_cell(
    folds: list[dict], dimension: str, category: str, symbol: str,
) -> dict:
    """Micro aggregation plus paired per-fold comparisons with count guards.

    Micro random reference is built from EACH fold's subgroup prevalence;
    averaging raw class prevalence across heterogeneous regimes would not
    be an equivalent baseline.
    """
    summed = {}
    for h in HORIZONS:
        key = str(h)
        total = selected = correct = neutral = 0
        expected_random_correct = 0.0
        qualifying_folds = 0
        for fold in folds:
            cell = fold["horizons"][key]["regimes"][dimension][category][symbol]
            if cell is None:
                continue
            total += cell["n_total"]
            selected += cell["selected"]
            correct += cell["correct_direction_class"]
            neutral += cell["neutral_among_selected"]
            if cell["selected"]:
                expected_random_correct += (
                    cell["selected"] * cell["random_reference_precision"]
                )
            if cell["selected"] >= MIN_SELECTED_PER_FOLD:
                qualifying_folds += 1
        summed[key] = {
            "folds_with_at_least_20_directional_predictions": qualifying_folds,
            "observations": total,
            "selected": selected,
            "correct": correct,
            "pooled_coverage": selected / total if total else None,
            "pooled_precision": correct / selected if selected else None,
            "fold_local_random_reference": (
                expected_random_correct / selected if selected else None
            ),
            "pooled_uplift_points": (
                100 * (correct - expected_random_correct) / selected
                if selected else None
            ),
            "neutral_fraction_among_selected": (
                neutral / selected if selected else None
            ),
        }

    paired_precision = []
    paired_uplift = []
    paired_coverage = []
    better_precision = better_uplift = 0
    for fold in folds:
        left = fold["horizons"]["2"]["regimes"][dimension][category][symbol]
        right = fold["horizons"]["6"]["regimes"][dimension][category][symbol]
        if (left is None or right is None
                or left["selected"] < MIN_SELECTED_PER_FOLD
                or right["selected"] < MIN_SELECTED_PER_FOLD):
            continue
        dp = 100 * (
            left["directional_precision"] - right["directional_precision"]
        )
        du = left["uplift_points"] - right["uplift_points"]
        dc = 100 * (left["coverage"] - right["coverage"])
        paired_precision.append(dp)
        paired_uplift.append(du)
        paired_coverage.append(dc)
        better_precision += dp > 0
        better_uplift += du > 0
    def mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    return {
        "H2": summed["2"],
        "H6": summed["6"],
        "paired": {
            "folds_with_both_min_20_selected": len(paired_precision),
            "H2_higher_precision_folds": better_precision,
            "H2_higher_uplift_folds": better_uplift,
            "H2_minus_H6_precision_points_mean": mean(paired_precision),
            "H2_minus_H6_uplift_points_mean": mean(paired_uplift),
            "H2_minus_H6_coverage_points_mean": mean(paired_coverage),
        },
    }


def summarize_regimes(folds: list[dict]) -> dict:
    return {
        dimension: {
            category: {
                symbol: summarize_cell(folds, dimension, category, symbol)
                for symbol in SYMBOLS
            }
            for category in categories
        }
        for dimension, categories in REGIMES.items()
    }


def compare_market_regimes(
    shared: pd.DataFrame, *,
    train_days: int = 120, validation_days: int = 14,
    test_days: int = 14, step_days: int = 14,
    final_holdout_days: int = 90,
    n_estimators: int = 150, seed: int = 42,
) -> dict:
    """No post hoc bin tuning; compare original full-feature H2/H6 fits."""
    if (train_days, validation_days, test_days, step_days,
            final_holdout_days) != (120, 14, 14, 14, 90):
        raise ValueError("Fixed P2 protocol required")
    if n_estimators < 20:
        raise ValueError("At least 20 boosting iterations required")
    if (set(shared.symbol.unique()) != set(SEEN_SYMBOLS)
            or not all(f"target_h{h}" in shared for h in HORIZONS)
            or not (shared.label_mature_at ==
                    shared.bar_close_at + pd.Timedelta(minutes=60)).all()):
        raise ValueError("Expected seen-only shared H12-matured feature rows")

    folds = common_folds(
        shared, train_days=train_days, validation_days=validation_days,
        test_days=test_days, step_days=step_days,
        final_holdout_days=final_holdout_days,
    )
    result = []
    for fold in folds:
        partitions, boundaries = shared_partitions(shared, fold)
        train, early, check = (
            partitions["train"],
            partitions["early_stopping"],
            partitions["validation_check"],
        )
        cutoffs = fit_volatility_cutoffs(train)
        assignments = assign_regimes(check, cutoffs)
        scored = {}
        for horizon in HORIZONS:
            label = f"target_h{horizon}"
            y_train = train[label].to_numpy(dtype=int)
            y_early = early[label].to_numpy(dtype=int)
            y_check = check[label].to_numpy(dtype=int)
            counts = np.bincount(y_train, minlength=3)
            if not np.all(counts > 0) or len(np.unique(y_early)) != 3:
                raise ValueError("All train/early classes must be present")
            model = _fit_model(
                train, early,
                train_target=y_train, early_target=y_early,
                binary=False, n_estimators=n_estimators, seed=seed,
            )
            probs = _class_probs(model, feature_matrix(check))
            predicted = decide(probs, counts, ALPHA_FIXED)
            all_check = slice_scores(
                y_check, predicted, np.ones(len(check), dtype=bool),
            )
            scored[str(horizon)] = {
                "train_counts": counts.tolist(),
                "best_iteration": int(model.best_iteration_),
                "overall": all_check,
                "regimes": score_regime_cells(
                    check, assignments, y_check, predicted
                ),
            }
            print(
                f"fold={fold.fold_id} H{horizon} "
                f"precision={all_check['directional_precision'] if all_check['directional_precision'] is not None else float('nan'):.4f} "
                f"coverage={all_check['coverage']:.4f}",
                flush=True,
            )
        result.append({
            "fold_id": fold.fold_id,
            **boundaries,
            "partition_counts": {
                key: len(frame) for key, frame in partitions.items()
            },
            "past_training_volatility_terciles": {
                symbol: list(pair) for symbol, pair in cutoffs.items()
            },
            "horizons": scored,
        })
    summary = summarize_regimes(result)
    return {
        "protocol": "cpu_p2_h2_h6_market_regime_descriptive_v1",
        "engine": "cpu",
        "horizons": list(HORIZONS),
        "alpha_fixed": ALPHA_FIXED,
        "training_days": train_days,
        "maximum_maturity_purge": "H12_60_minutes",
        "feature_count": len(FEATURE_COLUMNS),
        "regime_definitions": {
            "hour_utc": "24 fixed UTC one-hour buckets",
            "utc_block_6h": list(REGIMES["utc_block_6h"]),
            "day_type": list(REGIMES["day_type"]),
            "volatility_48_train_tercile": (
                "Per asset and per fold, 33.3% / 66.7% quantiles from train only. "
                "Current closed-bar volatility_48 determines check membership."
            ),
        },
        "min_directional_predictions_for_fold_comparison": MIN_SELECTED_PER_FOLD,
        "seen_assets": list(SEEN_SYMBOLS),
        "ADA_P0_loaded_or_scored": False,
        "fold_oos_tests_scored": False,
        "final_90_days_scored": False,
        "tuning_buffer_used": False,
        "previously_inspected_development_checks_not_independent": True,
        "regime_selection_from_check_prohibited": True,
        "interpretation": [
            "Descriptive regime and direction performance, NOT recommended trading hours.",
            "H2/H6 have different targets; reference class prevalences are measured within each regime and fold.",
            "Calendar timing, volatility and label threshold are interdependent; this does NOT establish causality.",
            "Low subgroup counts and repeated multi-group comparisons invite noise and selection bias.",
            "UTC clock buckets do not imply a stable exchange trading-session effect.",
            "Overlapping training windows and related market observations are NOT independent tests.",
            "Do not promote a model or claim profitability from these previously seen development labels.",
        ],
        "summary": summary,
        "folds": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P2 CPU H2/H6 market regime diagnostics, already-seen development only"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    if args.months != 24:
        parser.error("Fixed development archive: --months 24 required")
    if args.output.exists():
        parser.error("Refusing to overwrite existing diagnostic JSON")
    store = CandleStore(args.data_root)
    candles, manifests = {}, []
    for symbol in SEEN_SYMBOLS:
        print(f"Loading {symbol} (read-only P0)...", flush=True)
        candles[symbol], entries = load_symbol(
            store, symbol, months=args.months, verify_hashes=args.verify_hashes,
        )
        manifests.extend(entries)
    shared = build_shared_examples(candles)
    del candles
    report = compare_market_regimes(
        shared, n_estimators=args.n_estimators, seed=args.seed,
    )
    report["data_sha256_seen_assets"] = manifest_fingerprint(manifests)
    report["lightgbm_version"] = lgb.__version__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"Market-regime P2 report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
