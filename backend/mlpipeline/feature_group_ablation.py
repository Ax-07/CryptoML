"""P2 CPU group-ablation audit for H2 and H6 on common H12-purged dev rows.

Predeclared seven disjoint groups partition the existing 30 causal features.
For each horizon/fold, fit the unmodified 3-class LightGBM and seven
leave-one-family-out variants on IDENTICAL examples. Only model input
columns change; early stopping uses earlier validation, while 3d tuning is
not used and the subsequent 4d check is descriptive only.

Development checks have been repeatedly examined. No independent
confirmation, trade returns, ADA, fold OOS test or final 90-day scoring.
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
from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.trainer import _class_probs
from mlpipeline.two_stage_diagnostics import (
    ALPHA_FIXED, _cpu_model, _scored_candidate,
)
from mlpipeline.expanded_horizon_diagnostics import (
    SEEN_SYMBOLS, build_shared_examples, common_folds, shared_partitions,
)

HORIZONS = (2, 6)
FULL = "full"
GROUPS = {
    "returns": (
        "ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "ret_48",
    ),
    "trend_momentum": (
        "ema_gap_12", "ema_gap_48", "rsi_14",
    ),
    "volatility": (
        "volatility_12", "volatility_48", "atr_pct_14",
    ),
    "volume_flow": (
        "volume_relative_12", "volume_relative_48",
        "quote_volume_relative_48", "trade_count_relative_48",
        "taker_buy_share",
    ),
    "candle_shape": (
        "body_relative", "upper_wick_relative", "lower_wick_relative",
        "close_location", "range_pct",
    ),
    "calendar": (
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ),
    "market_context": (
        "btc_ret_3", "btc_volatility_12",
        "eth_ret_3", "eth_volatility_12",
    ),
}
VARIANTS = (FULL, *GROUPS)


def validate_feature_groups() -> None:
    """Fail loudly if new/renamed features invalidate a controlled ablation."""
    grouped = [name for group in GROUPS.values() for name in group]
    if (len(grouped) != len(FEATURE_COLUMNS)
            or len(set(grouped)) != len(grouped)
            or set(grouped) != set(FEATURE_COLUMNS)):
        raise ValueError("Feature groups must partition ALL 30 features exactly once")
    if len(FEATURE_COLUMNS) != 30 or len(GROUPS) != 7:
        raise ValueError("Predeclared experiment requires 30 features in 7 groups")


def columns_for_variant(variant: str) -> tuple[str, ...]:
    validate_feature_groups()
    if variant == FULL:
        return tuple(FEATURE_COLUMNS)
    if variant not in GROUPS:
        raise ValueError(f"Unknown feature family: {variant}")
    excluded = set(GROUPS[variant])
    remaining = tuple(col for col in FEATURE_COLUMNS if col not in excluded)
    if not remaining:
        raise AssertionError("Ablation would leave no features")
    return remaining


def _train_and_evaluate(
    train: pd.DataFrame, early: pd.DataFrame, check: pd.DataFrame,
    *, horizon: int, variant: str, n_estimators: int, seed: int,
) -> dict:
    if horizon not in HORIZONS:
        raise ValueError("Only H2 and H6 are predeclared")
    if n_estimators < 20:
        raise ValueError("n_estimators must be >=20")
    cols = columns_for_variant(variant)
    label = f"target_h{horizon}"
    y_train = train[label].to_numpy(dtype=int)
    y_early = early[label].to_numpy(dtype=int)
    y_check = check[label].to_numpy(dtype=int)
    counts = np.bincount(y_train, minlength=3)
    if (counts == 0).any() or set(np.unique(y_early)) != {0, 1, 2}:
        raise ValueError("Three classes required in training and early stopping")

    model = _cpu_model(binary=False, n_estimators=n_estimators, seed=seed)
    model.fit(
        train.loc[:, cols].astype("float32"),
        y_train,
        eval_set=[(early.loc[:, cols].astype("float32"), y_early)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    probabilities = _class_probs(model, check.loc[:, cols].astype("float32"))
    scored = _scored_candidate(y_check, probabilities, counts)
    per_asset = {}
    symbols = check.symbol.to_numpy()
    for symbol in SEEN_SYMBOLS:
        mask = symbols == symbol
        if not mask.any():
            raise ValueError(f"Missing {symbol} from common validation check")
        per_asset[symbol] = _scored_candidate(
            y_check[mask], probabilities[mask], counts
        )["directional"]
    # Split/gain importances are descriptive; ablation deltas provide
    # the stronger grouped counterfactual comparison.
    gains = model.booster_.feature_importance(importance_type="gain")
    splits = model.booster_.feature_importance(importance_type="split")
    if len(gains) != len(cols):
        raise AssertionError("LightGBM feature importance order mismatch")
    return {
        "variant": variant,
        "removed_columns": list(GROUPS[variant]) if variant != FULL else [],
        "feature_count": len(cols),
        "train_rows": len(train),
        "early_stop_rows": len(early),
        "validation_check_rows": len(check),
        "best_iteration": int(model.best_iteration_),
        "metrics": scored,
        "per_symbol_directional": per_asset,
        "feature_gain_importance": {
            name: float(value) for name, value in zip(cols, gains)
        },
        "feature_split_importance": {
            name: int(value) for name, value in zip(cols, splits)
        },
    }


def _average(seq) -> float | None:
    vals = list(seq)
    if not vals or any(value is None for value in vals):
        return None
    return float(np.mean(vals))


def _measure(variant: dict, name: str) -> float | None:
    if name in ("directional_precision", "coverage", "uplift_points",
                "conditional_uplift_points"):
        return variant["metrics"]["directional"][name]
    if name == "macro_f1":
        return variant["metrics"]["fixed_alpha_decisions"]["macro_f1"]
    if name == "balanced_accuracy":
        return variant["metrics"]["fixed_alpha_decisions"]["balanced_accuracy"]
    if name in ("log_loss", "multiclass_brier"):
        return variant["metrics"]["raw_probabilities"][name]
    raise ValueError(f"Unknown metric: {name}")


def _effects(full: dict, removed: dict) -> dict:
    """Positive = retaining the feature group is beneficial.

    Abstention/coverage also changes, so precision effects alone are NOT
    evidence that an input family improves true direction discrimination.
    """
    outcomes = {}
    for name in (
        "directional_precision", "coverage", "uplift_points",
        "conditional_uplift_points", "macro_f1", "balanced_accuracy",
        "log_loss", "multiclass_brier",
    ):
        base, drop = _measure(full, name), _measure(removed, name)
        multiplier = 100 if name in ("directional_precision", "coverage") else 1
        if name in ("log_loss", "multiclass_brier"):
            # Dropping a helpful family raises a loss; lower is better.
            outcomes[f"{name}_benefit"] = (
                drop - base if base is not None and drop is not None else None
            )
        else:
            outcomes[f"{name}_benefit"] = (
                multiplier * (base - drop)
                if base is not None and drop is not None else None
            )
    return outcomes


def _summarize(folds: list[dict]) -> dict:
    if not folds:
        raise ValueError("Cannot summarize empty folds")
    report = {"baseline": {}, "group_ablation": {}}
    for h in HORIZONS:
        hk = str(h)
        fulls = [fold["horizons"][hk][FULL] for fold in folds]
        report["baseline"][hk] = {
            "folds": len(folds),
            "directional_precision_pct": _average(
                100 * _measure(row, "directional_precision")
                for row in fulls
            ),
            "coverage_pct": _average(
                100 * _measure(row, "coverage") for row in fulls
            ),
            "uplift_points": _average(
                _measure(row, "uplift_points") for row in fulls
            ),
            "macro_f1": _average(
                _measure(row, "macro_f1") for row in fulls
            ),
            "log_loss": _average(
                _measure(row, "log_loss") for row in fulls
            ),
        }
        report["group_ablation"][hk] = {}
        for family in GROUPS:
            effects = [
                _effects(fold["horizons"][hk][FULL],
                         fold["horizons"][hk][family])
                for fold in folds
            ]
            names = effects[0]
            per_symbol = {}
            for symbol in SEEN_SYMBOLS:
                pairs = [
                    (
                        fold["horizons"][hk][FULL]["per_symbol_directional"][symbol],
                        fold["horizons"][hk][family]["per_symbol_directional"][symbol],
                    ) for fold in folds
                ]
                per_symbol[symbol] = {
                    "uplift_benefit_points_mean": _average(
                        a["uplift_points"] - b["uplift_points"]
                        if a["uplift_points"] is not None and b["uplift_points"] is not None
                        else None for a, b in pairs
                    ),
                    "precision_benefit_points_mean": _average(
                        100 * (a["directional_precision"] - b["directional_precision"])
                        if a["directional_precision"] is not None
                        and b["directional_precision"] is not None else None
                        for a, b in pairs
                    ),
                }
            report["group_ablation"][hk][family] = {
                "removed_feature_count": len(GROUPS[family]),
                "folds": len(folds),
                "full_better_uplift_folds": sum(
                    x["uplift_points_benefit"] is not None
                    and x["uplift_points_benefit"] > 0 for x in effects
                ),
                "full_better_log_loss_folds": sum(
                    x["log_loss_benefit"] is not None
                    and x["log_loss_benefit"] > 0 for x in effects
                ),
                "mean_benefit": {
                    name: _average(e[name] for e in effects)
                    for name in names
                },
                "by_symbol": per_symbol,
            }
    return report


def compare_feature_groups(
    shared: pd.DataFrame, *,
    train_days: int = 120, validation_days: int = 14,
    test_days: int = 14, step_days: int = 14,
    final_holdout_days: int = 90,
    n_estimators: int = 150, seed: int = 42,
) -> dict:
    validate_feature_groups()
    if ((train_days, validation_days, test_days, step_days,
         final_holdout_days) != (120, 14, 14, 14, 90)
            or n_estimators < 20):
        raise ValueError("P2 experiment fixed to 120/14/14/14, final 90 days")
    if set(shared.symbol) != set(SEEN_SYMBOLS):
        raise ValueError("Only four seen assets are allowed")
    folds = common_folds(
        shared, train_days=train_days, validation_days=validation_days,
        test_days=test_days, step_days=step_days,
        final_holdout_days=final_holdout_days,
    )
    records = []
    for fold in folds:
        partitions, boundaries = shared_partitions(shared, fold)
        train = partitions["train"]
        early = partitions["early_stopping"]
        check = partitions["validation_check"]
        per_horizon = {}
        for h in HORIZONS:
            variants = {}
            for variant in VARIANTS:
                scored = _train_and_evaluate(
                    train, early, check, horizon=h, variant=variant,
                    n_estimators=n_estimators, seed=seed,
                )
                variants[variant] = scored
                d = scored["metrics"]["directional"]
                print(
                    f"fold={fold.fold_id} H{h} removed={variant} "
                    f"direction_precision={d['directional_precision'] if d['directional_precision'] is not None else float('nan'):.4f} "
                    f"coverage={d['coverage']:.4f} "
                    f"log_loss={scored['metrics']['raw_probabilities']['log_loss']:.5f}",
                    flush=True,
                )
            per_horizon[str(h)] = variants
        records.append({
            "fold_id": fold.fold_id,
            **boundaries,
            "shared_partition_counts": {
                name: len(frame) for name, frame in partitions.items()
            },
            "horizons": per_horizon,
        })
    return {
        "protocol": "cpu_p2_group_ablation_h2_h6_h12_purged_v1",
        "engine": "cpu",
        "horizons_predeclared": list(HORIZONS),
        "group_names_predeclared": list(GROUPS),
        "groups": {k: list(v) for k, v in GROUPS.items()},
        "feature_schema_count": len(FEATURE_COLUMNS),
        "feature_groups_disjoint_cover_all_features": True,
        "alpha_fixed": ALPHA_FIXED,
        "h12_conservative_maturity_purge": True,
        "holdout_symbol_excluded": "ADAUSDT",
        "ADA_P0_loaded_or_scored": False,
        "fold_oos_tests_scored": False,
        "final_90_days_scored": False,
        "tuning_buffer_used": False,
        "preexisting_dev_check_not_independent": True,
        "expected_fit_count": len(folds) * len(HORIZONS) * len(VARIANTS),
        "important_notes": [
            "Positive group benefit means the full model beat a model refitted without that group.",
            "Correlated/redundant features mean leave-one-group-out is not causal importance.",
            "Directional precision and coverage co-vary; read uplift and log loss too.",
            "The validation checks were previously inspected; differences are descriptive, not independent evidence.",
            "Multiple families, assets and horizons are explored; no model promotion or profitability inference.",
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
        "summary": _summarize(records),
        "folds": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU P2 leave-one-feature-family-out on H2/H6 dev folds"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    if args.months != 24:
        parser.error("This P2 experiment requires --months 24")
    if args.output.exists():
        parser.error("Refusing to overwrite existing research output")
    store = CandleStore(args.data_root)
    candles, manifests = {}, []
    for symbol in SEEN_SYMBOLS:
        print(f"Reading seen symbol {symbol}", flush=True)
        candles[symbol], entries = load_symbol(
            store, symbol, months=args.months, verify_hashes=args.verify_hashes
        )
        manifests.extend(entries)
    shared = build_shared_examples(candles)
    del candles
    report = compare_feature_groups(
        shared, n_estimators=args.n_estimators, seed=args.seed
    )
    report["dataset_fingerprint_sha256_seen_symbols"] = manifest_fingerprint(manifests)
    report["lightgbm_version"] = lgb.__version__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Feature group P2 report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
