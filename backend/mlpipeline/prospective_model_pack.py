"""Fit and freeze genuine pre-reserve H2/H6 CPU model packs for P3.

Reads ONLY archive MONTHS ending 2026-06 inclusive: never opens a
July-September (previously inspected final holdout) partition, never ADA.
120d training, 7d early stopping, 3d unlabeled margin tuning; shared H12
purge and feature schema. No new outcomes, test folds or historical checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ingestion.storage import CandleStore
from mlpipeline.confidence_diagnostics import decision_and_margin
from mlpipeline.coverage_matched_diagnostics import threshold_for_target
from mlpipeline.dataset import REQUIRED, manifest_fingerprint
from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS, build_shared_examples
from mlpipeline.features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, feature_matrix
from mlpipeline.labels import LABEL_SCHEMA_VERSION
from mlpipeline.prospective_protocol import (
    FEATURE_SCHEMA, LABEL_SCHEMA, RESERVED_FINAL_WINDOW_START_UTC,
    verify,
)
from mlpipeline.trainer import _class_probs
from mlpipeline.two_stage_diagnostics import _fit_model

# End before *any* July 2026 monthly data is opened.
CUTOFF = pd.Timestamp("2026-07-01T00:00:00Z")
TRAIN_DAYS = 120
EARLY_DAYS = 7
TUNING_DAYS = 3
HORIZONS = (2, 6)
TARGET = 0.40
SEED = 42
N_ESTIMATORS = 150


def exclude_july_boundary_close(frame: pd.DataFrame, *, period: str) -> pd.DataFrame:
    """Drop ONLY the June candle closing exactly at the July cutoff.

    Binance M5 partitions are assigned by bar_open_at. Thus the
    2026-06-30 23:55 UTC candle belongs to June but bar_close_at is
    precisely 2026-07-01 00:00 UTC (the exclusive P3 history cutoff).
    Any other crossing is a data-integrity failure, never silently dropped.
    """
    crossing = frame["bar_close_at"].ge(CUTOFF)
    if not crossing.any():
        return frame
    legitimate = (
        period == "2026-06"
        and int(crossing.sum()) == 1
        and frame.loc[crossing, "bar_open_at"].eq(
            CUTOFF - pd.Timedelta(minutes=5)
        ).all()
        and frame.loc[crossing, "bar_close_at"].eq(CUTOFF).all()
    )
    if not legitimate:
        raise AssertionError("Historical reserve crossover beyond expected June boundary")
    return frame.loc[~crossing].copy()


def load_clean_historic(store: CandleStore, *, verify_hashes: bool = True):
    """Only whole December 2025--June 2026 P0 month partitions, seen assets."""
    candles, manifests = {}, []
    all_manifests = store.list_manifests()
    for symbol in SEEN_SYMBOLS:
        parts = sorted(
            (m for m in all_manifests
             if m["symbol"] == symbol and "2025-12" <= m["period"] <= "2026-06"),
            key=lambda m: m["period"],
        )
        if len(parts) != 7:
            raise ValueError(f"{symbol}: need all seven Dec 2025--Jun 2026 months")
        if [x["period"] for x in parts] != [
            "2025-12", "2026-01", "2026-02", "2026-03",
            "2026-04", "2026-05", "2026-06",
        ]:
            raise ValueError(f"{symbol}: missing/duplicate historic month")
        frames = []
        for part in parts:
            if part["quality_status"] != "OK":
                raise ValueError(f"{symbol}: P0 quality != OK")
            yyyy, mm = map(int, part["period"].split("-"))
            path = store.partition_dir(symbol, yyyy, mm) / "part-000.parquet"
            if verify_hashes and hashlib.sha256(path.read_bytes()).hexdigest() != part["parquet_sha256"]:
                raise ValueError(f"Manifest checksum changed: {path}")
            frame = pd.read_parquet(path, columns=list(REQUIRED))
            if not frame["is_final"].all():
                raise ValueError(f"{symbol}: unfinalized input")
            for col in ("bar_open_at", "bar_close_at"):
                frame[col] = pd.to_datetime(frame[col], utc=True)
            frame = exclude_july_boundary_close(frame, period=part["period"])
            if frame.empty or not frame.bar_close_at.lt(CUTOFF).all():
                raise AssertionError("No eligible strictly pre-July history")
            frame["symbol"] = symbol
            frames.append(frame)
            manifests.append(part)
        data = pd.concat(frames, ignore_index=True).sort_values(
            "bar_open_at"
        ).reset_index(drop=True)
        if (data.bar_open_at.duplicated().any()
                or not (data.bar_close_at - data.bar_open_at ==
                        pd.Timedelta(minutes=5)).all()
                or data.bar_open_at.diff().dropna().lt(
                    pd.Timedelta(minutes=5)
                ).any()):
            raise ValueError(f"{symbol}: invalid/duplicate M5 candles")
        candles[symbol] = data
    return candles, manifests


def fixed_partitions(shared: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """No use of retrospective P2 checks or reserved time labels."""
    early_end = CUTOFF - pd.Timedelta(days=TUNING_DAYS)
    train_end = early_end - pd.Timedelta(days=EARLY_DAYS)
    train_start = train_end - pd.Timedelta(days=TRAIN_DAYS)
    t, m = shared.bar_close_at, shared.label_mature_at
    partitions = {
        "train": shared.loc[
            t.ge(train_start) & t.lt(train_end) & m.lt(train_end)
        ],
        "early": shared.loc[
            t.ge(train_end) & t.lt(early_end) & m.lt(early_end)
        ],
        "tuning": shared.loc[
            t.ge(early_end) & t.lt(CUTOFF) & m.lt(CUTOFF)
        ],
    }
    if any(len(frame) < 1000 for frame in partitions.values()):
        raise ValueError("Insufficient complete train/early/tuning rows")
    if any(frame.symbol.eq("ADAUSDT").any() for frame in partitions.values()):
        raise AssertionError("ADA accidentally present")
    if (not partitions["train"].label_mature_at.lt(train_end).all()
            or not partitions["early"].label_mature_at.lt(early_end).all()
            or not partitions["tuning"].label_mature_at.lt(CUTOFF).all()):
        raise AssertionError("H12 maturity leak")
    return partitions


def train_locked_packs(
    shared: pd.DataFrame, *, dataset_fingerprint: str,
) -> tuple[dict, dict]:
    """Return fitted, frozen packs and an activation-compatible policy."""
    partitions = fixed_partitions(shared)
    train, early, tune = (
        partitions["train"], partitions["early"], partitions["tuning"]
    )
    packages, policies, provenance = {}, {}, {}
    for h in HORIZONS:
        key = f"H{h}"
        column = f"target_h{h}"
        y_train = train[column].to_numpy(dtype=int)
        y_early = early[column].to_numpy(dtype=int)
        train_counts = np.bincount(y_train, minlength=3)
        if np.any(train_counts < 1) or set(np.unique(y_early)) != {0, 1, 2}:
            raise ValueError(f"{key}: all three classes required")
        model = _fit_model(
            train, early, train_target=y_train, early_target=y_early,
            binary=False, n_estimators=N_ESTIMATORS, seed=SEED,
        )
        p = _class_probs(model, feature_matrix(tune))
        decisions, margins = decision_and_margin(p, train_counts, alpha=1.0)
        hours = tune.bar_close_at.dt.hour.to_numpy()
        scores = {}
        for cohort, membership in (
            ("12-17_UTC", (hours >= 12) & (hours < 18)),
            ("all_day_UTC", np.ones(len(tune), dtype=bool)),
        ):
            result = threshold_for_target(
                decisions[membership], margins[membership],
                target_coverage=TARGET,
            )
            if not result["feasible"]:
                raise ValueError(
                    f"{key} {cohort}: 40% tuning target infeasible: {result['reason']}"
                )
            scores[cohort] = float(result["threshold"])
            print(
                f"{key} {cohort} tune_coverage={result['tuning_selected_coverage']:.3f}",
                flush=True,
            )
        provenance[key] = {
            "horizon_bars": h,
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "feature_count": len(FEATURE_COLUMNS),
            "feature_columns": list(FEATURE_COLUMNS),
            "label_schema": LABEL_SCHEMA_VERSION,
            "alpha": 1.0,
            "train_prior_counts": train_counts.tolist(),
            "training_data_last_close_utc": train.bar_close_at.max().isoformat(),
            "early_stopping_last_close_utc": early.bar_close_at.max().isoformat(),
            "tuning_last_close_utc": tune.bar_close_at.max().isoformat(),
            "historical_partition_cutoff_exclusive_utc": CUTOFF.isoformat(),
            "dataset_fingerprint_sha256": dataset_fingerprint,
            "training_rows": len(train),
            "early_rows": len(early),
            "tuning_rows": len(tune),
            "model_best_iteration": int(model.best_iteration_),
        }
        packages[key] = {
            "estimator": model,
            "horizon_bars": h,
            "features": list(FEATURE_COLUMNS),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "train_prior_counts": train_counts.tolist(),
            "dataset_fingerprint_sha256": dataset_fingerprint,
        }
        policies[key] = scores
    if FEATURE_SCHEMA != FEATURE_SCHEMA_VERSION or LABEL_SCHEMA != LABEL_SCHEMA_VERSION:
        raise ValueError("Frozen schema mismatch")
    if train.bar_close_at.max() >= pd.Timestamp(RESERVED_FINAL_WINDOW_START_UTC):
        raise AssertionError("Training overlaps previously inspected time")
    policy = {
        "thresholds": policies,
        "fitted_before_utc": datetime.now(timezone.utc).isoformat(),
        "training_provenance": provenance,
    }
    return packages, policy


def save_packs(output: Path, packs: dict, policy: dict) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite model pack: {output}")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("H2", "H6"):
        with (output / f"{name.lower()}_model.joblib").open("xb") as stream:
            joblib.dump(packs[name], stream, compress=3)
    with (output / "threshold_policy.json").open("x", encoding="utf-8") as stream:
        json.dump(policy, stream, indent=2, ensure_ascii=False)
    print(f"Saved frozen packs for operator verification: {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU P3 model packs (historical pre-July data ONLY)"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    verify(args.protocol_root)
    if args.output.exists():
        parser.error("Output already exists; do not overwrite previous model pack")
    store = CandleStore(args.data_root)
    candles, manifests = load_clean_historic(
        store, verify_hashes=args.verify_hashes,
    )
    shared = build_shared_examples(candles)
    del candles
    packs, policy = train_locked_packs(
        shared, dataset_fingerprint=manifest_fingerprint(manifests),
    )
    save_packs(args.output, packs, policy)


if __name__ == "__main__":
    main()
