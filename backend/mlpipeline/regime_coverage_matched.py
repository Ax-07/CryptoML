"""P2 matched tuning coverage by broad market regime, H2 versus H6, CPU only.

Each fold: 120d train, 7d early stop, 3d TUNING, 4d development
check. A separate score-only threshold is fit for each (horizon, regime)
using ONLY the earlier tuning scores, with 40% target directional coverage.
No using check labels/coverage to set thresholds. Unfeasible cells are
explicitly excluded, never replaced by an unmarked global fallback.

Same closed M5 observations, H12 conservative label-maturity purge,
full 30 features, fixed prior adjustment alpha=1.0, unseen ADA excluded.
Fold OOS / final 90d are not evaluated. Historical development checks
have been repeatedly inspected: results are descriptive, NOT independent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ingestion.storage import CandleStore
from mlpipeline.confidence_diagnostics import (
    decision_and_margin, directional_metrics,
)
from mlpipeline.coverage_matched_diagnostics import threshold_for_target
from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.expanded_horizon_diagnostics import (
    SEEN_SYMBOLS, build_shared_examples, common_folds, shared_partitions,
)
from mlpipeline.features import FEATURE_COLUMNS, feature_matrix
from mlpipeline.market_regime_diagnostics import (
    REGIMES, assign_regimes, fit_volatility_cutoffs,
)
from mlpipeline.trainer import _class_probs
from mlpipeline.two_stage_diagnostics import ALPHA_FIXED, _fit_model

HORIZONS = (2, 6)
TARGET_COVERAGE = 0.40
REGIME_AXES = ("utc_block_6h", "day_type", "volatility_48_train_tercile")
MIN_TUNE_ROWS = 100
MIN_CHECK_SELECTED_FOR_FOLD = 20
SCORE_NAMES = ("H2", "H6")


def tuning_partition(shared: pd.DataFrame, fold) -> pd.DataFrame:
    """Only the 3 days after early stopping, purged using H12 maturity."""
    stop_end = fold.validation_start + pd.Timedelta(days=7)
    tune_end = stop_end + pd.Timedelta(days=3)
    t = shared.bar_close_at
    maturity = shared.label_mature_at
    mask = t.ge(stop_end) & t.lt(tune_end) & maturity.lt(tune_end)
    tune = shared.loc[mask]
    if len(tune) < 100 or tune.symbol.eq("ADAUSDT").any():
        raise ValueError("Not enough seen-only tuning data")
    if not (
        tune.bar_close_at.ge(stop_end).all()
        and tune.bar_close_at.lt(tune_end).all()
        and tune.label_mature_at.lt(tune_end).all()
        and stop_end < tune_end < fold.test_start
    ):
        raise AssertionError("Tuning time leak: H12 labels must mature before check")
    return tune


def fit_cell_threshold(
    decisions: np.ndarray, margins: np.ndarray,
    regime_membership: np.ndarray, *, target: float = TARGET_COVERAGE,
) -> dict:
    """Fit only on UNLABELED tuning predictions and known current regimes."""
    mask = np.asarray(regime_membership, dtype=bool)
    if (mask.ndim != 1 or mask.shape != decisions.shape
            or mask.shape != margins.shape):
        raise ValueError("Tuning membership and score shape mismatch")
    n = int(mask.sum())
    if n < MIN_TUNE_ROWS:
        return {
            "feasible": False,
            "reason": "insufficient_tuning_rows",
            "tuning_rows": n,
            "threshold": None,
            "tuning_selected": None,
            "tuning_selected_coverage": None,
            "target_coverage": target,
        }
    return threshold_for_target(
        decisions[mask], margins[mask], target_coverage=target,
    )


def filtered_metrics(
    frame: pd.DataFrame, truth: np.ndarray, decisions: np.ndarray,
    margins: np.ndarray, cohort: np.ndarray, threshold: float,
) -> dict:
    """Check labels are consulted only AFTER tuning threshold is frozen."""
    mask = np.asarray(cohort, dtype=bool)
    if (len(frame) != len(mask) or truth.shape != mask.shape
            or decisions.shape != mask.shape or margins.shape != mask.shape
            or not mask.any()):
        raise ValueError("Empty or inconsistent development-check cohort")
    score = directional_metrics(
        truth[mask], decisions[mask], margins[mask], threshold=threshold,
    )
    score["check_class_counts"] = {
        name: int((truth[mask] == idx).sum())
        for idx, name in enumerate(("DOWN", "NEUTRAL", "UP"))
    }
    return score


def _check_comparison(a: dict, b: dict) -> dict:
    """Paired on identical cohort, not necessarily on achieved coverage."""
    if a["n_total"] != b["n_total"]:
        raise AssertionError("Horizon check keys not aligned")
    pa, pb = a["directional_precision"], b["directional_precision"]
    ua, ub = a["uplift_points"], b["uplift_points"]
    return {
        "precision_delta_points": (
            100 * (pa - pb) if pa is not None and pb is not None else None
        ),
        "uplift_delta_points": (
            ua - ub if ua is not None and ub is not None else None
        ),
        "coverage_delta_points": 100 * (a["coverage"] - b["coverage"]),
        "actual_coverage_h2": a["coverage"],
        "actual_coverage_h6": b["coverage"],
        "both_at_least_20_selected": (
            a["selected"] >= MIN_CHECK_SELECTED_FOR_FOLD
            and b["selected"] >= MIN_CHECK_SELECTED_FOR_FOLD
        ),
    }


def evaluate_cell(
    check: pd.DataFrame, membership: np.ndarray,
    check_predictions: dict, policies: dict,
) -> dict:
    """Unfeasible tuning cells never peek at check labels for a fallback."""
    if not all(policies[str(h)]["feasible"] for h in HORIZONS):
        return {
            "paired_tuning_feasible": False,
            "reason": "infeasible_tuning_coverage_or_insufficient_rows",
            "check_rows": int(membership.sum()),
            "policies": policies,
            "check": {},
            "by_asset": {},
            "comparison": None,
        }
    n = int(membership.sum())
    if n == 0:
        return {
            "paired_tuning_feasible": True,
            "reason": "empty_check_cohort",
            "check_rows": 0,
            "policies": policies,
            "check": {},
            "by_asset": {},
            "comparison": None,
        }
    checks = {}
    by_asset = {symbol: {} for symbol in SEEN_SYMBOLS}
    symbols = check.symbol.to_numpy()
    for h in HORIZONS:
        key = str(h)
        predictions = check_predictions[key]
        threshold = policies[key]["threshold"]
        checks[key] = filtered_metrics(
            check, predictions["truth"], predictions["decisions"],
            predictions["margins"], membership, threshold,
        )
        for symbol in SEEN_SYMBOLS:
            own = membership & (symbols == symbol)
            by_asset[symbol][key] = (
                filtered_metrics(
                    check, predictions["truth"], predictions["decisions"],
                    predictions["margins"], own, threshold,
                ) if own.any() else None
            )
    asset_comparisons = {
        symbol: (
            _check_comparison(scores["2"], scores["6"])
            if scores["2"] is not None and scores["6"] is not None else None
        )
        for symbol, scores in by_asset.items()
    }
    return {
        "paired_tuning_feasible": True,
        "reason": "check_scored_with_tuning_only_thresholds",
        "check_rows": n,
        "policies": policies,
        "check": checks,
        "by_asset": {
            symbol: {
                "checks": by_asset[symbol],
                "comparison": asset_comparisons[symbol],
            }
            for symbol in SEEN_SYMBOLS
        },
        "comparison": _check_comparison(checks["2"], checks["6"]),
    }


def _mean(values) -> float | None:
    seq = list(values)
    return float(np.mean(seq)) if seq else None


def summarize_regime_cells(folds: list[dict]) -> dict:
    """Aggregate only paired feasible cells with both selected counts >=20.

    Denominator is reported separately for every regime/asset. A tuning
    policy only constrains tuning coverage, never realized check coverage.
    The local prevalence-based reference is recomputed inside EACH
    fold+regime+asset; no global-prior substitution.
    """
    result = {}
    for axis in REGIME_AXES:
        result[axis] = {}
        for category in REGIMES[axis]:
            cells = [fold["cells"][axis][category] for fold in folds]
            feasible = [cell for cell in cells if cell["paired_tuning_feasible"]]
            scored = [cell for cell in feasible if cell["comparison"] is not None]
            qualified = [
                cell for cell in scored
                if cell["comparison"]["both_at_least_20_selected"]
            ]
            per_horizon = {}
            for h in HORIZONS:
                key = str(h)
                metric_list = [cell["check"][key] for cell in qualified]
                total = sum(m["n_total"] for m in metric_list)
                selected = sum(m["selected"] for m in metric_list)
                correct = sum(m["correct_direction_class"] for m in metric_list)
                expected = sum(
                    m["selected"] * m["random_reference_precision"]
                    for m in metric_list
                )
                per_horizon[key] = {
                    "observations": total,
                    "selected": selected,
                    "precision_pooled": correct / selected if selected else None,
                    "coverage_pooled": selected / total if total else None,
                    "uplift_points_fold_local_pooled": (
                        100 * (correct - expected) / selected if selected else None
                    ),
                    "precision_fold_mean": _mean(
                        m["directional_precision"] for m in metric_list
                    ),
                    "coverage_fold_mean": _mean(
                        m["coverage"] for m in metric_list
                    ),
                    "uplift_points_fold_mean": _mean(
                        m["uplift_points"] for m in metric_list
                    ),
                }
            pair = [cell["comparison"] for cell in qualified]
            by_asset = {}
            for symbol in SEEN_SYMBOLS:
                local_pairs = [
                    cell["by_asset"][symbol]["comparison"] for cell in qualified
                    if cell["by_asset"][symbol]["comparison"] is not None
                    and cell["by_asset"][symbol]["comparison"]["both_at_least_20_selected"]
                ]
                by_asset[symbol] = {
                    "paired_folds_eligible_min20": len(local_pairs),
                    "h2_better_precision_folds": sum(
                        p["precision_delta_points"] > 0 for p in local_pairs
                    ),
                    "h2_better_uplift_folds": sum(
                        p["uplift_delta_points"] > 0 for p in local_pairs
                    ),
                    "precision_delta_points_fold_mean": _mean(
                        p["precision_delta_points"] for p in local_pairs
                    ),
                    "uplift_delta_points_fold_mean": _mean(
                        p["uplift_delta_points"] for p in local_pairs
                    ),
                    "coverage_delta_points_fold_mean": _mean(
                        p["coverage_delta_points"] for p in local_pairs
                    ),
                }
            result[axis][category] = {
                "total_folds": len(folds),
                "paired_tuning_feasible_folds": len(feasible),
                "paired_check_nonempty_folds": len(scored),
                "paired_folds_eligible_min20": len(qualified),
                "excluded_tuning_or_small_check_folds": len(folds) - len(qualified),
                "h2_better_precision_folds": sum(
                    p["precision_delta_points"] > 0 for p in pair
                ),
                "h2_better_uplift_folds": sum(
                    p["uplift_delta_points"] > 0 for p in pair
                ),
                "precision_delta_points_fold_mean": _mean(
                    p["precision_delta_points"] for p in pair
                ),
                "uplift_delta_points_fold_mean": _mean(
                    p["uplift_delta_points"] for p in pair
                ),
                "realized_coverage_delta_points_fold_mean": _mean(
                    p["coverage_delta_points"] for p in pair
                ),
                "horizons": per_horizon,
                "by_asset_same_pooled_regime_threshold": by_asset,
            }
    return result


def compare_matched_regimes(
    shared: pd.DataFrame, *, train_days: int = 120,
    validation_days: int = 14, test_days: int = 14,
    step_days: int = 14, final_holdout_days: int = 90,
    n_estimators: int = 150, seed: int = 42,
) -> dict:
    if (train_days, validation_days, test_days,
            step_days, final_holdout_days) != (120, 14, 14, 14, 90):
        raise ValueError("P2 fixed 120/14/14/14 and reserved final 90 days")
    if n_estimators < 20:
        raise ValueError("Need >=20 estimators")
    if (set(shared.symbol.unique()) != set(SEEN_SYMBOLS)
            or not all(f"target_h{h}" in shared for h in HORIZONS)
            or not (shared.label_mature_at ==
                    shared.bar_close_at + pd.Timedelta(minutes=60)).all()):
        raise ValueError("Shared seen-only H12-matured features required")

    folds = common_folds(
        shared, train_days=train_days, validation_days=validation_days,
        test_days=test_days, step_days=step_days,
        final_holdout_days=final_holdout_days,
    )
    records = []
    for fold in folds:
        parts, boundaries = shared_partitions(shared, fold)
        train, early, check = (
            parts["train"], parts["early_stopping"], parts["validation_check"]
        )
        tune = tuning_partition(shared, fold)
        if not (
            train.label_mature_at.lt(fold.validation_start).all()
            and early.label_mature_at.lt(
                fold.validation_start + pd.Timedelta(days=7)
            ).all()
            and tune.label_mature_at.lt(
                fold.validation_start + pd.Timedelta(days=10)
            ).all()
            and check.label_mature_at.lt(fold.test_start).all()
        ):
            raise AssertionError("H12 purge check failed")
        cutoffs = fit_volatility_cutoffs(train)
        tune_bins = assign_regimes(tune, cutoffs)
        check_bins = assign_regimes(check, cutoffs)

        tuned = {}
        checked = {}
        iterations = {}
        for h in HORIZONS:
            key = str(h)
            target = f"target_h{h}"
            y_train = train[target].to_numpy(dtype=int)
            y_early = early[target].to_numpy(dtype=int)
            y_check = check[target].to_numpy(dtype=int)
            counts = np.bincount(y_train, minlength=3)
            if np.any(counts <= 0) or len(np.unique(y_early)) != 3:
                raise ValueError("Three train/early classes required")
            model = _fit_model(
                train, early, train_target=y_train, early_target=y_early,
                binary=False, n_estimators=n_estimators, seed=seed,
            )
            tune_pred, tune_margins = decision_and_margin(
                _class_probs(model, feature_matrix(tune)), counts,
                alpha=ALPHA_FIXED,
            )
            check_pred, check_margins = decision_and_margin(
                _class_probs(model, feature_matrix(check)), counts,
                alpha=ALPHA_FIXED,
            )
            tuned[key] = {
                "decisions": tune_pred,
                "margins": tune_margins,
            }
            checked[key] = {
                "decisions": check_pred,
                "margins": check_margins,
                "truth": y_check,
            }
            iterations[key] = int(model.best_iteration_)

        fold_cells = {}
        for axis in REGIME_AXES:
            axis_cells = {}
            for category in REGIMES[axis]:
                tune_mask = tune_bins[axis] == category
                check_mask = check_bins[axis] == category
                policies = {}
                for h in HORIZONS:
                    key = str(h)
                    policies[key] = fit_cell_threshold(
                        tuned[key]["decisions"], tuned[key]["margins"],
                        tune_mask, target=TARGET_COVERAGE,
                    )
                axis_cells[category] = evaluate_cell(
                    check, check_mask, checked, policies,
                )
            fold_cells[axis] = axis_cells
        records.append({
            "fold_id": fold.fold_id,
            **boundaries,
            "tuning_start_utc": (
                fold.validation_start + pd.Timedelta(days=7)
            ).isoformat(),
            "tuning_end_exclusive_utc": (
                fold.validation_start + pd.Timedelta(days=10)
            ).isoformat(),
            "sample_counts": {
                **{k: len(v) for k, v in parts.items()},
                "tuning": len(tune),
            },
            "best_iterations": iterations,
            "past_only_volatility_terciles": {
                symbol: list(cutoffs[symbol]) for symbol in SEEN_SYMBOLS
            },
            "cells": fold_cells,
        })
        diagnostics = fold_cells["volatility_48_train_tercile"]["medium"]
        comparison = diagnostics["comparison"]
        print(
            f"fold={fold.fold_id} "
            f"vol_medium_feasible={diagnostics['paired_tuning_feasible']} "
            f"H2_minus_H6_uplift_pp={comparison['uplift_delta_points'] if comparison else None} "
            f"coverage_delta_pp={comparison['coverage_delta_points'] if comparison else None}",
            flush=True,
        )

    return {
        "protocol": "cpu_p2_h2_h6_regime_coverage40_tuning_matched_v1",
        "engine": "cpu",
        "horizons_predeclared": list(HORIZONS),
        "target_tuning_coverage": TARGET_COVERAGE,
        "target_tuning_coverage_max_shortfall_points": 2.0,
        "regime_axes_predeclared": list(REGIME_AXES),
        "regime_categories": {
            axis: list(REGIMES[axis]) for axis in REGIME_AXES
        },
        "feature_count": len(FEATURE_COLUMNS),
        "alpha_fixed": ALPHA_FIXED,
        "h12_purge_for_every_horizon": True,
        "volatility_terciles_from_training_only": True,
        "min_tuning_rows_per_regime": MIN_TUNE_ROWS,
        "min_check_directional_selected_per_horizon": MIN_CHECK_SELECTED_FOR_FOLD,
        "threshold_tuning_uses_only_unlabeled_scores": True,
        "regime_threshold_per_horizon_and_axis_category": True,
        "NO_joint_hour_volatility_filters": True,
        "ADA_P0_loaded_or_scored": False,
        "fold_oos_tests_scored": False,
        "final_90_days_scored": False,
        "development_check_previously_inspected_not_independent": True,
        "model_promotion_forbidden": True,
        "critical_caveats": [
            "H2 and H6 have different future target labels, even on identical M5 rows.",
            "Thresholds are fitted separately on unlabeled 3d tuning scores for each horizon x broad regime.",
            "A 40% tuning target DOES NOT guarantee 40% realized check coverage or equal H2/H6 coverage.",
            "Regime cells missing on the 3d tuning buffer (notably some weekends) are EXCLUDED, not silently backfilled.",
            "Asset breakdown uses pooled regime-level thresholds: no claim of target 40% coverage per asset.",
            "All findings are post hoc on repeatedly studied development checks and overlapping folds.",
            "References for uplift are fold/regime/asset observed class frequencies, not production-time known values.",
            "No jointly optimized hour-volatility strategy, no future labels for threshold selection, no profitability study.",
        ],
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
        "summary": summarize_regime_cells(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P2 H2/H6 40%-tuning-coverage broad-regime CPU comparison"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    if args.months != 24:
        parser.error("P2 only supports --months 24")
    if args.output.exists():
        parser.error("Existing report: refusing overwrite")
    store = CandleStore(args.data_root)
    candles, manifests = {}, []
    for symbol in SEEN_SYMBOLS:
        print(f"Reading immutable P0 {symbol}", flush=True)
        candles[symbol], entries = load_symbol(
            store, symbol, months=args.months, verify_hashes=args.verify_hashes,
        )
        manifests.extend(entries)
    shared = build_shared_examples(candles)
    del candles
    report = compare_matched_regimes(
        shared, n_estimators=args.n_estimators, seed=args.seed,
    )
    report["dataset_fingerprint_sha256_seen_symbols"] = manifest_fingerprint(manifests)
    report["lightgbm_version"] = lgb.__version__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Matched-regime P2 report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
