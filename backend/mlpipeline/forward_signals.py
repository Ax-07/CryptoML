"""Append-only, causally timestamped forward CPU M5 predictions. No orders.

One JSON snapshot per signal, written only after the candle closes, with
exclusive-create semantics. Never reconstruct or backfill an earlier signal.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from mlpipeline.features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from mlpipeline.frozen_model import load_bundle
from mlpipeline.paper_replay import FIVE_MINUTES


def predict_snapshot(
    model_dir: Path, features: pd.DataFrame, *,
    recorded_at: pd.Timestamp, fresh_within_minutes: int = 5,
) -> list[dict]:
    """Predict only the just-closed M5 bar; fail closed on stale data."""
    booster, meta = load_bundle(model_dir)
    if meta.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Feature schema mismatch")
    if not isinstance(recorded_at, pd.Timestamp) or recorded_at.tzinfo is None:
        raise ValueError("recorded_at must be timezone-aware")
    now = recorded_at.tz_convert("UTC")
    if fresh_within_minutes < 1 or fresh_within_minutes > 5:
        raise ValueError("Freshness must be 1..5 min")
    needed = {"symbol", "bar_close_at", *FEATURE_COLUMNS}
    if not needed.issubset(features):
        raise ValueError(f"Missing closed-bar features: {sorted(needed - set(features))}")
    if features.empty or features.symbol.duplicated().any():
        raise ValueError("Expected one just-closed bar per symbol")
    if not isinstance(features.bar_close_at.dtype, pd.DatetimeTZDtype) or str(features.bar_close_at.dt.tz) != "UTC":
        raise ValueError("bar_close_at must be explicitly UTC-aware")
    if features.bar_close_at.isna().any() or features.symbol.isna().any():
        raise ValueError("Null timestamps/symbols forbidden")
    if features.symbol.astype(str).str.fullmatch(r"[A-Z0-9]+USDT").eq(False).any():
        raise ValueError("Invalid symbol format")
    if features.bar_close_at.nunique() != 1:
        raise ValueError("All market snapshots must refer to the same bar close")
    close = features.bar_close_at.iloc[0]
    if close < pd.Timestamp(meta["final_holdout_end_utc"]):
        raise ValueError("Retrospective or final-holdout signal forbidden")
    if (now < close or now >= close + FIVE_MINUTES
            or now - close > pd.Timedelta(minutes=fresh_within_minutes)):
        raise ValueError("Stale, future, or late-arriving signal forbidden")
    if not close < close + FIVE_MINUTES:
        raise AssertionError("Invalid M5 close")
    matrix = features.loc[:, FEATURE_COLUMNS].astype("float32")
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("NaN / infinity in incoming features")
    p = np.asarray(booster.predict(matrix, num_threads=8), dtype=float)
    if p.shape != (len(features), 3) or not np.isfinite(p).all():
        raise ValueError("Invalid model probabilities")
    if (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Invalid probability normalization")
    p = p / p.sum(axis=1, keepdims=True)
    prior = meta["class_prior"]
    return [{
        "symbol": str(features.symbol.iloc[i]),
        "bar_close_at": close.isoformat(),
        "recorded_at": now.isoformat(),
        "model_trained_until": meta["fitted_labels_before_utc"],
        "model_id": meta["model_id"],
        "model_family": "lightgbm",
        "p_down": float(p[i, 0]), "p_neutral": float(p[i, 1]),
        "p_up": float(p[i, 2]),
        "prior_down": float(prior[0]), "prior_neutral": float(prior[1]),
        "prior_up": float(prior[2]),
    } for i in range(len(features))]


def write_snapshots_once(rows: list[dict], log_dir: Path) -> list[Path]:
    """Write-once signal snapshots. Never overwrite prior records."""
    if not rows:
        raise ValueError("No predictions")
    log_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    identifiers = set()
    for row in rows:
        symbol = row["symbol"]
        stamp = pd.Timestamp(row["bar_close_at"])
        if stamp.tzinfo is None:
            raise ValueError("UTC timestamp required")
        name = stamp.tz_convert("UTC").strftime("%Y%m%dT%H%M%SZ")
        key = (symbol, name)
        if key in identifiers:
            raise ValueError("Duplicate signal in batch")
        identifiers.add(key)
        paths.append(log_dir / f"{symbol}_{name}.json")
    if any(p.exists() for p in paths):
        raise FileExistsError("Previously logged signal; snapshots are immutable")
    # Individual files are exclusive-created; duplicate process fails, not overwrites.
    created = []
    try:
        for path, row in zip(paths, rows):
            with path.open("x", encoding="utf-8") as out:
                created.append(path)
                json.dump(row, out, ensure_ascii=False, indent=2)
    except Exception:
        # Do not erase partial data: if two processes race, existing snapshots
        # may be valid. Caller should investigate, not automatically retry.
        raise
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Journal genuine forward M5 model predictions only")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--closed-features", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.closed_features.suffix.lower() != ".parquet":
        parser.error("Expected closed-features Parquet from causal live feature pipeline")
    # Current system time is deliberately captured by the program, not user input.
    observed = pd.Timestamp(datetime.now(timezone.utc))
    features = pd.read_parquet(args.closed_features)
    snapshots = predict_snapshot(args.model_dir, features, recorded_at=observed)
    paths = write_snapshots_once(snapshots, args.log_dir)
    print(f"Logged {len(paths)} current, write-once CPU predictions.")
    print(f"First snapshot: {paths[0]}")


if __name__ == "__main__":
    main()
