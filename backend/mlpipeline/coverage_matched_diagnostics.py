"""P2 CPU paired comparison at predeclared *tuning* directional coverage.

Every fold: 120d train, 7d early stopping, 3d threshold tuning,
4d validation check. No OOS fold tests, final 90d or ADA outcomes.
Targets 30% (primary) and 40% (secondary), fixed *before* check labels.
Coverage may drift on the check period; always report that drift rather
than claiming identical realized check coverage.
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
from mlpipeline.dataset import load_symbol, manifest_fingerprint
from mlpipeline.features import feature_matrix
from mlpipeline.trainer import DEFAULT_SYMBOLS, _class_probs, prepare_training_data
from mlpipeline.two_stage_diagnostics import (
    ALPHA_FIXED, _binary_positive_probability, _fit_model,
    combine_stage_probabilities,
)
from mlpipeline.walkforward import make_walk_forward_folds

TARGET_COVERAGES = (0.30, 0.40)
PRIMARY_TARGET_COVERAGE = 0.30
MAX_TUNING_COVERAGE_SHORTFALL = 0.02
CANDIDATES = ("three_class", "two_stage")


def threshold_for_target(
    predicted: np.ndarray, margins: np.ndarray, *,
    target_coverage: float,
) -> dict:
    """Fit an abstention threshold from unlabeled *tuning* scores alone.

    Only prior-adjusted predicted UP/DOWN can be selected. Never turn
    NEUTRAL predictions into directional decisions to force coverage.
    Exact tuning coverage is not always achievable owing to ties or too
    few predicted directions; those cases are explicitly marked infeasible.
    """
    p = np.asarray(predicted, dtype=int)
    m = np.asarray(margins, dtype=float)
    if (p.ndim != 1 or m.shape != p.shape or len(p) == 0
            or not np.isin(p, [0, 1, 2]).all()
            or not np.isfinite(m).all()
            or (m < 0).any() or (m > 1).any()):
        raise ValueError("Expected 1-D class decisions and bounded finite margins")
    if not (0 < target_coverage < 1):
        raise ValueError("Coverage target must be within (0,1)")
    n = len(p)
    k = max(1, int(round(target_coverage * n)))
    eligible_margins = np.sort(m[p != 1])[::-1]
    available = int(len(eligible_margins))
    details = {
        "target_coverage": target_coverage,
        "tuning_rows": n,
        "target_count": k,
        "available_directional": available,
        "available_directional_coverage": available / n,
        "feasible": False,
        "threshold": None,
        "tuning_selected": None,
        "tuning_selected_coverage": None,
        "reason": None,
    }
    if available < k:
        details["reason"] = "not_enough_directional_predictions_on_tuning"
        return details

    if k == available:
        threshold = 0.0
    elif eligible_margins[k - 1] > eligible_margins[k]:
        # Midpoint between the desired last and the first rejected score.
        threshold = float(
            eligible_margins[k] +
            (eligible_margins[k - 1] - eligible_margins[k]) / 2.0
        )
    else:
        # A score-only threshold cannot break identical-score ties.
        # Conservatively exclude the entire tied boundary group.
        threshold = float(np.nextafter(eligible_margins[k - 1], np.inf))

    selected = int(((p != 1) & (m >= threshold)).sum())
    achieved = selected / n
    details.update(
        threshold=threshold,
        tuning_selected=selected,
        tuning_selected_coverage=achieved,
        feasible=(achieved >= target_coverage - MAX_TUNING_COVERAGE_SHORTFALL),
        reason=(
            "tuning_threshold_selected" if
            achieved >= target_coverage - MAX_TUNING_COVERAGE_SHORTFALL else
            "threshold_ties_prevent_target_coverage"
        ),
    )
    return details


def _model_pair_probabilities(
    train: pd.DataFrame, early: pd.DataFrame,
    tune: pd.DataFrame, check: pd.DataFrame, *,
    n_estimators: int, seed: int,
) -> tuple[dict, dict, dict]:
    y_train = train.target.to_numpy(dtype=int)
    y_early = early.target.to_numpy(dtype=int)
    train_counts = np.bincount(y_train, minlength=3)
    if (np.any(train_counts == 0) or len(np.unique(y_early)) != 3):
        raise ValueError("Missing full training/early-stopping class set")

    flat = _fit_model(
        train, early, train_target=y_train, early_target=y_early,
        binary=False, n_estimators=n_estimators, seed=seed,
    )
    move = _fit_model(
        train, early,
        train_target=(y_train != 1).astype(int),
        early_target=(y_early != 1).astype(int),
        binary=True, n_estimators=n_estimators, seed=seed,
    )
    moving_train = train.loc[train.target.ne(1)]
    moving_early = early.loc[early.target.ne(1)]
    direction = _fit_model(
        moving_train, moving_early,
        train_target=moving_train.target.eq(2).to_numpy(dtype=int),
        early_target=moving_early.target.eq(2).to_numpy(dtype=int),
        binary=True, n_estimators=n_estimators, seed=seed,
    )
    probabilities = {}
    for part_name, frame in (("tuning", tune), ("check", check)):
        X = feature_matrix(frame)
        probabilities[part_name] = {
            "three_class": _class_probs(flat, X),
            "two_stage": combine_stage_probabilities(
                _binary_positive_probability(move, X),
                _binary_positive_probability(direction, X),
            ),
        }
    iterations = {
        "three_class": int(flat.best_iteration_),
        "movement": int(move.best_iteration_),
        "conditional_direction": int(direction.best_iteration_),
    }
    return probabilities, iterations, {
        "train_class_counts": [int(x) for x in train_counts],
        "stage2_train_movers": int(len(moving_train)),
        "stage2_early_movers": int(len(moving_early)),
    }


def matched_coverage_comparison(
    examples: pd.DataFrame, *,
    holdout_symbol: str = "ADAUSDT",
    train_days: int = 120, validation_days: int = 14,
    test_days: int = 14, step_days: int = 14,
    final_holdout_days: int = 90, n_estimators: int = 150,
    seed: int = 42,
) -> dict:
    """Paired fold comparison; check labels never choose score thresholds."""
    if holdout_symbol != "ADAUSDT":
        raise ValueError("ADAUSDT is reserved and excluded from development")
    if validation_days < 4 or n_estimators < 20:
        raise ValueError("Need at least 4 validation days and 20 estimators")
    folds = make_walk_forward_folds(
        examples, holdout_symbol=holdout_symbol, train_days=train_days,
        validation_days=validation_days, test_days=test_days,
        step_days=step_days, final_holdout_days=final_holdout_days,
    )
    seen = examples.loc[examples.symbol.ne(holdout_symbol)]
    t, mature = seen.bar_close_at, seen.label_mature_at
    records = []
    for fold in folds:
        stop_end = fold.validation_start + pd.Timedelta(days=validation_days // 2)
        tune_end = stop_end + pd.Timedelta(
            days=(validation_days - validation_days // 2) // 2
        )
        masks = {
            "train": t.ge(fold.train_start) & t.lt(fold.validation_start)
                     & mature.lt(fold.validation_start),
            "early_stop": t.ge(fold.validation_start) & t.lt(stop_end)
                          & mature.lt(stop_end),
            "tuning": t.ge(stop_end) & t.lt(tune_end)
                      & mature.lt(tune_end),
            "check": t.ge(tune_end) & t.lt(fold.test_start)
                     & mature.lt(fold.test_start),
        }
        parts = {name: seen.loc[mask] for name, mask in masks.items()}
        if min(len(part) for part in parts.values()) < 100:
            raise ValueError(f"Fold {fold.fold_id}: not enough purged observations")
        train, early = parts["train"], parts["early_stop"]
        tune, check = parts["tuning"], parts["check"]
        probs, iterations, training = _model_pair_probabilities(
            train, early, tune, check,
            n_estimators=n_estimators, seed=seed,
        )
        counts = np.asarray(training["train_class_counts"], dtype=int)
        predictions = {}
        for part_name in ("tuning", "check"):
            predictions[part_name] = {
                candidate: decision_and_margin(probs[part_name][candidate], counts,
                                               alpha=ALPHA_FIXED)
                for candidate in CANDIDATES
            }

        target_results = {}
        for target in TARGET_COVERAGES:
            policies = {
                candidate: threshold_for_target(
                    *predictions["tuning"][candidate], target_coverage=target,
                )
                for candidate in CANDIDATES
            }
            paired = all(policies[candidate]["feasible"] for candidate in CANDIDATES)
            checked = {}
            if paired:
                truth = check.target.to_numpy(dtype=int)
                for candidate in CANDIDATES:
                    pred, gap = predictions["check"][candidate]
                    checked[candidate] = directional_metrics(
                        truth, pred, gap,
                        threshold=policies[candidate]["threshold"],
                    )
                x, y = (checked[c] for c in CANDIDATES)
                precision_difference = (
                    100 * (y["directional_precision"] - x["directional_precision"])
                    if y["directional_precision"] is not None
                    and x["directional_precision"] is not None else None
                )
                coverage_difference = 100 * (y["coverage"] - x["coverage"])
            else:
                precision_difference = None
                coverage_difference = None
            target_results[str(target)] = {
                "paired_tuning_feasible": paired,
                "policies_fit_only_on_tuning": policies,
                "validation_check_metrics": checked,
                "two_stage_minus_three_class_precision_points": precision_difference,
                "two_stage_minus_three_class_coverage_points": coverage_difference,
            }
        records.append({
            "fold_id": fold.fold_id,
            "train_start_utc": fold.train_start.isoformat(),
            "early_stop_end_exclusive_utc": stop_end.isoformat(),
            "tuning_end_exclusive_utc": tune_end.isoformat(),
            "validation_check_end_exclusive_utc": fold.test_start.isoformat(),
            "partition_counts": {name: len(x) for name, x in parts.items()},
            "best_iterations": iterations,
            "training": training,
            "target_results": target_results,
        })
        primary = target_results[str(PRIMARY_TARGET_COVERAGE)]
        print(
            f"fold={fold.fold_id} primary_pair_feasible={primary['paired_tuning_feasible']} "
            f"delta_precision_points={primary['two_stage_minus_three_class_precision_points']} "
            f"delta_coverage_points={primary['two_stage_minus_three_class_coverage_points']}",
            flush=True,
        )

    summary = {}
    for target in TARGET_COVERAGES:
        rows = [r["target_results"][str(target)] for r in records]
        feasible = [r for r in rows if r["paired_tuning_feasible"]]
        def mean(vals: list) -> float | None:
            if not vals or any(v is None for v in vals):
                return None
            return float(np.mean(vals))
        summary[str(target)] = {
            "total_folds": len(records),
            "paired_folds_with_feasible_tuning": len(feasible),
            "excluded_due_to_tuning_only": len(records) - len(feasible),
            "two_stage_higher_precision_folds": sum(
                r["two_stage_minus_three_class_precision_points"] is not None
                and r["two_stage_minus_three_class_precision_points"] > 0
                for r in feasible
            ),
            "precision_delta_points_mean": mean([
                r["two_stage_minus_three_class_precision_points"] for r in feasible
            ]),
            "realized_coverage_delta_points_mean": mean([
                r["two_stage_minus_three_class_coverage_points"] for r in feasible
            ]),
            "check_precision_mean": {
                candidate: mean([
                    r["validation_check_metrics"][candidate]["directional_precision"]
                    for r in feasible
                ])
                for candidate in CANDIDATES
            },
            "check_coverage_mean": {
                candidate: mean([
                    r["validation_check_metrics"][candidate]["coverage"]
                    for r in feasible
                ])
                for candidate in CANDIDATES
            },
            "mean_conditional_uplift_points": {
                candidate: mean([
                    r["validation_check_metrics"][candidate]["conditional_uplift_points"]
                    for r in feasible
                ])
                for candidate in CANDIDATES
            },
        }
    return {
        "protocol": "cpu_p2_paired_tuning_matched_coverage_v1",
        "engine": "cpu",
        "targets_predeclared": TARGET_COVERAGES,
        "primary_target_coverage": PRIMARY_TARGET_COVERAGE,
        "max_tuning_coverage_shortfall": MAX_TUNING_COVERAGE_SHORTFALL,
        "fixed_alpha": ALPHA_FIXED,
        "label_horizon_bars": 6,
        "holdout_symbol_excluded": holdout_symbol,
        "fold_oos_tests_scored": False,
        "final_90_days_scored": False,
        "preexisting_development_validation_not_independent": True,
        "coverage_caveat": (
            "Thresholds match target on earlier tuning observations only; "
            "later check coverage is allowed to differ, and realized coverage "
            "is NOT forced to match using check-period information."
        ),
        "note": (
            "No parameter, threshold or model selection on check labels; "
            "check periods were already inspected in previous P2 work. "
            "This is not evidence of production profitability."
        ),
        "parameters": {
            "train_days": train_days, "validation_days": validation_days,
            "test_days": test_days, "step_days": step_days,
            "final_holdout_days": final_holdout_days,
            "n_estimators": n_estimators, "seed": seed,
        },
        "folds": records,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU matched tuning-coverage comparison, P2 dev only"
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
    if args.holdout_symbol != "ADAUSDT" or args.horizon != 6:
        parser.error("P2 comparison requires reserved ADAUSDT and fixed horizon=6")
    if args.output.exists():
        parser.error("Report already exists: refusing to overwrite")
    store = CandleStore(args.data_root)
    candles, manifests = {}, []
    for symbol in DEFAULT_SYMBOLS:
        print(f"Reading {symbol}...", flush=True)
        candles[symbol], items = load_symbol(
            store, symbol, months=args.months, verify_hashes=args.verify_hashes,
        )
        manifests.extend(items)
    examples = prepare_training_data(candles, horizon=args.horizon)
    del candles
    report = matched_coverage_comparison(
        examples, holdout_symbol=args.holdout_symbol,
        train_days=args.train_days, validation_days=args.validation_days,
        test_days=args.test_days, step_days=args.step_days,
        final_holdout_days=args.final_holdout_days,
        n_estimators=args.n_estimators, seed=args.seed,
    )
    report["dataset_fingerprint_sha256"] = manifest_fingerprint(manifests)
    report["lightgbm_version"] = lgb.__version__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(f"P2 matched-coverage report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
