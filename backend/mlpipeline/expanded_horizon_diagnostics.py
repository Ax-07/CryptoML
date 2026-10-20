"""P2 CPU extended horizons (H1/H2/H3/H5/H6/H9/H12), development only.

One causal feature build, all 7 label sets from the canonical label builder.
Identical symbol/time/feature observations, identical purged fold boundaries
using H12 maturity (60 minutes), and fixed 120-day LightGBM 3-class training.
Class-distribution adjusted directional uplift and train-prior log-loss skill
are primary descriptive comparisons: horizons have DIFFERENT targets.

Never read ADA P0, train/evaluate ADA, score fold OOS, or score final 90 days.
The development checks were already studied; these are NOT independent tests.
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
from mlpipeline.features import (
    FEATURE_COLUMNS, build_symbol_features, feature_matrix, join_market_context,
)
from mlpipeline.labels import add_labels
from mlpipeline.trainer import DEFAULT_SYMBOLS, _class_probs
from mlpipeline.two_stage_diagnostics import ALPHA_FIXED, _fit_model, _scored_candidate
from mlpipeline.walkforward import make_walk_forward_folds

HORIZONS = (1, 2, 3, 5, 6, 9, 12)
MAX_HORIZON = max(HORIZONS)
REFERENCE_HORIZON = 6
PREVIOUS_BEST_HORIZON = 3
RESERVED_SYMBOL = "ADAUSDT"
SEEN_SYMBOLS = tuple(s for s in DEFAULT_SYMBOLS if s != RESERVED_SYMBOL)


def build_shared_examples(candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute features once per seen asset and targets on their common M5 keys.

    ADA is not accepted or accessed at all. add_labels respects contiguous
    segments and future-close validity. No partial/missing future bar is labeled.
    """
    if set(candles) != set(SEEN_SYMBOLS):
        raise ValueError("Supply exactly the four seen assets, never ADA")
    features = {
        symbol: build_symbol_features(candles[symbol])
        for symbol in SEEN_SYMBOLS
    }
    outputs = []
    for symbol in SEEN_SYMBOLS:
        context = join_market_context(
            features[symbol], features["BTCUSDT"], features["ETHUSDT"]
        )
        cols = ["symbol", "bar_close_at", *FEATURE_COLUMNS]
        clean = np.isfinite(
            context[FEATURE_COLUMNS].to_numpy(dtype="float64")
        ).all(axis=1)
        shared = context.loc[:, cols].copy()
        for h in HORIZONS:
            labeled = add_labels(context, horizon=h, threshold_multiplier=0.5)
            target = labeled["target"]
            clean &= target.notna().to_numpy()
            shared[f"target_h{h}"] = target.to_numpy()
            if h == MAX_HORIZON:
                shared["label_mature_at"] = labeled["label_mature_at"].to_numpy()
            del labeled
        shared = shared.loc[clean].copy()
        if shared.empty:
            raise ValueError(f"{symbol}: no intersection of valid horizon labels")
        for h in HORIZONS:
            shared[f"target_h{h}"] = shared[f"target_h{h}"].astype("int8")
        shared.loc[:, FEATURE_COLUMNS] = shared[FEATURE_COLUMNS].astype("float32")
        if not (
            shared["label_mature_at"] ==
            shared["bar_close_at"] + pd.Timedelta(minutes=5 * MAX_HORIZON)
        ).all():
            raise AssertionError("H12 label-maturity alignment invalid")
        outputs.append(shared)
        print(
            f"Shared horizons {symbol}: {len(shared)} eligible M5 observations",
            flush=True,
        )
    out = pd.concat(outputs, ignore_index=True).sort_values(
        ["symbol", "bar_close_at"]
    ).reset_index(drop=True)
    if out.duplicated(["symbol", "bar_close_at"]).any():
        raise AssertionError("Duplicate M5 keys")
    for name in ("bar_close_at", "label_mature_at"):
        if (not isinstance(out[name].dtype, pd.DatetimeTZDtype)
                or str(out[name].dt.tz) != "UTC"):
            raise ValueError("M5 close and future-maturity dates must be UTC-aware")
    return out


def common_folds(shared: pd.DataFrame, *,
                 train_days: int = 120,
                 validation_days: int = 14,
                 test_days: int = 14,
                 step_days: int = 14,
                 final_holdout_days: int = 90):
    """Use the existing fold engine without reading ANY ADA price or label.

    Its API requires the unseen symbol to be present. A synthetic *metadata*
    sentinel with the first seen UTC timestamp satisfies this check, and is
    immediately discarded. Only the four seen symbols define fold geometry.
    """
    first = shared["bar_close_at"].min()
    geometry = pd.concat([
        shared[["symbol", "bar_close_at", "label_mature_at"]],
        pd.DataFrame({
            "symbol": [RESERVED_SYMBOL],
            "bar_close_at": [first],
            "label_mature_at": [first + pd.Timedelta(minutes=60)],
        }),
    ], ignore_index=True)
    return make_walk_forward_folds(
        geometry,
        holdout_symbol=RESERVED_SYMBOL,
        train_days=train_days, validation_days=validation_days,
        test_days=test_days, step_days=step_days,
        final_holdout_days=final_holdout_days,
    )


def shared_partitions(shared: pd.DataFrame, fold) -> tuple[dict, dict]:
    """Strict H12 maturity purge, common to all seven horizon targets."""
    if shared.symbol.eq(RESERVED_SYMBOL).any():
        raise ValueError("ADA rows must never reach training or scoring")
    t, maturity = shared.bar_close_at, shared.label_mature_at
    stop_end = fold.validation_start + pd.Timedelta(days=7)
    tune_end = stop_end + pd.Timedelta(days=3)
    masks = {
        "train": t.ge(fold.train_start) & t.lt(fold.validation_start)
                 & maturity.lt(fold.validation_start),
        "early_stopping": t.ge(fold.validation_start) & t.lt(stop_end)
                          & maturity.lt(stop_end),
        "validation_check": t.ge(tune_end) & t.lt(fold.test_start)
                            & maturity.lt(fold.test_start),
    }
    chunks = {name: shared.loc[mask] for name, mask in masks.items()}
    if min(map(len, chunks.values())) < 100:
        raise ValueError(f"Fold {fold.fold_id}: too few samples")
    if not (fold.validation_start < stop_end < tune_end < fold.test_start):
        raise AssertionError("Invalid dev chronology")
    for label, end in (
        ("train", fold.validation_start),
        ("early_stopping", stop_end),
        ("validation_check", fold.test_start),
    ):
        if not chunks[label].label_mature_at.lt(end).all():
            raise AssertionError(f"{label}: invalid H12 temporal purge")
    boundaries = {
        "train_start_utc": fold.train_start.isoformat(),
        "validation_start_utc": fold.validation_start.isoformat(),
        "early_stop_end_exclusive_utc": stop_end.isoformat(),
        "unused_tuning_end_exclusive_utc": tune_end.isoformat(),
        "validation_check_end_exclusive_utc": fold.test_start.isoformat(),
    }
    return chunks, boundaries


def _fold_means(records: list[dict]) -> tuple[dict, dict, dict]:
    """Summarize common-date folds and per-asset means (not independent)."""
    if not records:
        raise ValueError("No development fold results")

    def mean(values) -> float | None:
        seq = list(values)
        if not seq or any(v is None for v in seq):
            return None
        return float(np.mean(seq))

    summary, per_symbol_summary = {}, {}
    for h in HORIZONS:
        key = str(h)
        items = [row["horizons"][key] for row in records]
        summary[key] = {
            "folds": len(items),
            "minutes_ahead": h * 5,
            "directional_precision_mean": mean(
                x["metrics"]["directional"]["directional_precision"] for x in items
            ),
            "directional_coverage_mean": mean(
                x["metrics"]["directional"]["coverage"] for x in items
            ),
            "directional_uplift_points_mean": mean(
                x["metrics"]["directional"]["uplift_points"] for x in items
            ),
            "conditional_uplift_points_mean": mean(
                x["metrics"]["directional"]["conditional_uplift_points"] for x in items
            ),
            "macro_f1_mean": mean(
                x["metrics"]["fixed_alpha_decisions"]["macro_f1"] for x in items
            ),
            "balanced_accuracy_mean": mean(
                x["metrics"]["fixed_alpha_decisions"]["balanced_accuracy"] for x in items
            ),
            "raw_log_loss_mean": mean(
                x["metrics"]["raw_probabilities"]["log_loss"] for x in items
            ),
            "train_prior_log_loss_mean": mean(
                x["prior_baseline"]["train_prior_log_loss"] for x in items
            ),
            "log_loss_skill_vs_train_prior_mean": mean(
                x["prior_baseline"]["log_loss_skill_vs_train_prior"] for x in items
            ),
            "brier_improvement_vs_train_prior_mean": mean(
                x["prior_baseline"]["brier_improvement_vs_train_prior"] for x in items
            ),
            "class_distribution_mean": {
                name: mean(
                    x["check_class_counts"][name] /
                    sum(x["check_class_counts"].values()) for x in items
                ) for name in ("DOWN", "NEUTRAL", "UP")
            },
        }
        per_symbol_summary[key] = {}
        for symbol in SEEN_SYMBOLS:
            per_symbol_summary[key][symbol] = {
                "directional_precision_mean": mean(
                    x["per_symbol_directional"][symbol]["directional_precision"]
                    for x in items
                ),
                "directional_coverage_mean": mean(
                    x["per_symbol_directional"][symbol]["coverage"]
                    for x in items
                ),
                "directional_uplift_points_mean": mean(
                    x["per_symbol_directional"][symbol]["uplift_points"]
                    for x in items
                ),
            }

    paired = {}
    for h in HORIZONS:
        if h == REFERENCE_HORIZON:
            continue
        k = str(h)
        ref = str(REFERENCE_HORIZON)
        pairs = [
            (r["horizons"][k], r["horizons"][ref])
            for r in records
        ]
        paired[f"H{h}_vs_H{REFERENCE_HORIZON}"] = {
            "folds": len(pairs),
            "precision_delta_points_mean": mean(
                100 * (a["metrics"]["directional"]["directional_precision"]
                       - b["metrics"]["directional"]["directional_precision"])
                for a, b in pairs
            ),
            "uplift_delta_points_mean": mean(
                a["metrics"]["directional"]["uplift_points"]
                - b["metrics"]["directional"]["uplift_points"]
                for a, b in pairs
            ),
            "coverage_delta_points_mean": mean(
                100 * (a["metrics"]["directional"]["coverage"]
                       - b["metrics"]["directional"]["coverage"])
                for a, b in pairs
            ),
            "log_loss_skill_delta_points_mean": mean(
                100 * (
                    a["prior_baseline"]["log_loss_skill_vs_train_prior"] -
                    b["prior_baseline"]["log_loss_skill_vs_train_prior"]
                ) for a, b in pairs
            ),
            "folds_higher_directional_precision": sum(
                a["metrics"]["directional"]["directional_precision"]
                > b["metrics"]["directional"]["directional_precision"]
                for a, b in pairs
            ),
            "folds_higher_uplift": sum(
                a["metrics"]["directional"]["uplift_points"]
                > b["metrics"]["directional"]["uplift_points"]
                for a, b in pairs
            ),
            "folds_higher_log_loss_skill": sum(
                a["prior_baseline"]["log_loss_skill_vs_train_prior"]
                > b["prior_baseline"]["log_loss_skill_vs_train_prior"]
                for a, b in pairs
            ),
        }
    return summary, per_symbol_summary, paired


def compare_extended_horizons(
    shared: pd.DataFrame, *,
    train_days: int = 120,
    validation_days: int = 14,
    test_days: int = 14,
    step_days: int = 14,
    final_holdout_days: int = 90,
    n_estimators: int = 150,
    seed: int = 42,
) -> dict:
    if ((train_days, validation_days, test_days,
         step_days, final_holdout_days) != (120, 14, 14, 14, 90)
         or n_estimators < 20):
        raise ValueError("Predeclared P2: 120/14/14/14 days, 90 final, >=20 estimators")
    if set(shared.symbol) != set(SEEN_SYMBOLS):
        raise ValueError("Only the four previously seen assets are permitted")
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
        X_check = feature_matrix(check)
        scores = {}
        for h in HORIZONS:
            target = f"target_h{h}"
            y_train = train[target].to_numpy(dtype=int)
            y_early = early[target].to_numpy(dtype=int)
            y_check = check[target].to_numpy(dtype=int)
            counts = np.bincount(y_train, minlength=3)
            if (counts == 0).any() or len(set(y_early)) != 3:
                raise ValueError(f"Fold {fold.fold_id}, H{h}: not all 3 classes present")
            model = _fit_model(
                train, early,
                train_target=y_train, early_target=y_early,
                binary=False, n_estimators=n_estimators, seed=seed,
            )
            probs = _class_probs(model, X_check)
            scored = _scored_candidate(y_check, probs, counts)
            prior = counts.astype(float) / counts.sum()
            prior_eval = evaluate(
                y_check, np.broadcast_to(prior, probs.shape)
            )
            ll = scored["raw_probabilities"]["log_loss"]
            prior_ll = prior_eval["log_loss"]
            baseline = {
                "train_prior_log_loss": prior_ll,
                "log_loss_skill_vs_train_prior": (
                    1.0 - ll / prior_ll if prior_ll > 0 else None
                ),
                "train_prior_multiclass_brier": prior_eval["multiclass_brier"],
                "brier_improvement_vs_train_prior": (
                    prior_eval["multiclass_brier"] -
                    scored["raw_probabilities"]["multiclass_brier"]
                ),
            }
            per_symbol = {}
            symbols = check.symbol.to_numpy()
            for symbol in SEEN_SYMBOLS:
                mask = symbols == symbol
                if not mask.any():
                    raise ValueError(f"Fold {fold.fold_id}: missing {symbol}")
                per_symbol[symbol] = _scored_candidate(
                    y_check[mask], probs[mask], counts
                )["directional"]
            scores[str(h)] = {
                "best_iteration": int(model.best_iteration_),
                "train_class_counts": {
                    name: int(counts[i])
                    for i, name in enumerate(("DOWN", "NEUTRAL", "UP"))
                },
                "check_class_counts": {
                    name: int((y_check == i).sum())
                    for i, name in enumerate(("DOWN", "NEUTRAL", "UP"))
                },
                "metrics": scored,
                "prior_baseline": baseline,
                "per_symbol_directional": per_symbol,
            }
            direction = scored["directional"]
            skill = baseline["log_loss_skill_vs_train_prior"]
            print(
                f"fold={fold.fold_id} H{h} "
                f"precision={direction['directional_precision']:.4f} "
                f"coverage={direction['coverage']:.4f} "
                f"uplift_points={direction['uplift_points']:.3f} "
                f"log_loss_skill={skill:.5f}", flush=True,
            )
        records.append({
            "fold_id": fold.fold_id,
            **boundaries,
            "common_partition_counts": {k: len(v) for k, v in parts.items()},
            "horizons": scores,
        })
    summary, by_symbol, paired = _fold_means(records)
    return {
        "protocol": "cpu_p2_7_horizons_shared_samples_h12_purge_v1",
        "engine": "cpu",
        "horizons_predeclared": list(HORIZONS),
        "maximum_label_maturity_horizon": MAX_HORIZON,
        "reference_horizon": REFERENCE_HORIZON,
        "previously_promising_horizon": PREVIOUS_BEST_HORIZON,
        "alpha_fixed": ALPHA_FIXED,
        "feature_count": len(FEATURE_COLUMNS),
        "training_days_fixed": train_days,
        "label_threshold_multiplier_fixed": 0.5,
        "seen_symbols_only": list(SEEN_SYMBOLS),
        "ADA_P0_loaded_or_scored": False,
        "fold_oos_tests_scored": False,
        "final_90_days_scored": False,
        "tuning_days_unused": True,
        "previously_inspected_dev_check_not_independent": True,
        "comparability_note": (
            "Identical M5 rows, causal features, periods, and H12 purge. "
            "Each horizon has DIFFERENT targets and class base rates; compare "
            "own-class uplift and log-loss skill, not raw precision alone. "
            "Previously observed H3 performance means horizon choice is adaptive."
        ),
        "limitations": [
            "Repeated inspection of overlapping development windows is not an independent confirmation.",
            "Overlapping horizons, correlated candles and walk-forward folds invalidate naive independent win-rate tests.",
            "No final 90-day or ADA evaluation, model promotion, PnL or execution.",
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
        "by_symbol_summary": by_symbol,
        "paired_vs_h6": paired,
        "folds": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P2 CPU compare H1/H2/H3/H5/H6/H9/H12 (no ADA/final scoring)"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    if args.months != 24:
        parser.error("P2 frozen archive comparison requires --months 24")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing research report")
    store = CandleStore(args.data_root)
    candles, manifests = {}, []
    for symbol in SEEN_SYMBOLS:
        print(f"Reading immutable P0: {symbol}", flush=True)
        candles[symbol], entries = load_symbol(
            store, symbol, months=args.months, verify_hashes=args.verify_hashes,
        )
        manifests.extend(entries)
    shared = build_shared_examples(candles)
    del candles
    report = compare_extended_horizons(
        shared, n_estimators=args.n_estimators, seed=args.seed,
    )
    report["dataset_fingerprint_sha256"] = manifest_fingerprint(manifests)
    report["lightgbm_version"] = lgb.__version__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"Extended P2 horizon report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
