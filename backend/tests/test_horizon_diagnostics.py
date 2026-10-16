"""H3/H5/H6 shared-observation and chronology checks, synthetic only."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.horizon_diagnostics import align_horizon_examples, common_horizon_partitions
from mlpipeline.walkforward import WalkForwardFold


def synthetic():
    times = pd.date_range("2025-01-01T00:05:00Z", periods=35 * 288, freq="5min")
    datasets = {}
    for h in (3, 5, 6):
        frames = []
        for i, symbol in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT")):
            frame = pd.DataFrame({
                "symbol": symbol,
                "bar_close_at": times,
                "label_mature_at": times + pd.Timedelta(minutes=5 * h),
                "target": np.arange(len(times)) % 3,
            })
            for col in FEATURE_COLUMNS:
                frame[col] = np.full(len(times), i + h / 10, dtype=np.float32)
            frames.append(frame)
        datasets[h] = pd.concat(frames, ignore_index=True)
    return datasets


def test_align_matches_samples_and_uses_h6_feature_rows():
    raw = synthetic()
    aligned = align_horizon_examples(raw)
    assert len(aligned) == len(raw[6])
    assert aligned[["target_h3", "target_h5", "target_h6"]].notna().all().all()
    assert (aligned.label_mature_at ==
            aligned.bar_close_at + pd.Timedelta(minutes=30)).all()
    assert (aligned.loc[aligned.symbol.eq("BTCUSDT"), FEATURE_COLUMNS[0]] == 0.6).all()


def test_horizon_union_purge_and_reserved_ada():
    aligned = align_horizon_examples(synthetic())
    start = pd.Timestamp("2025-01-10T00:05:00Z")
    fold = WalkForwardFold(
        0, start - pd.Timedelta(days=9), start,
        start + pd.Timedelta(days=14), start + pd.Timedelta(days=15),
    )
    parts, bounds = common_horizon_partitions(aligned, fold)
    assert all(not p.symbol.eq("ADAUSDT").any() for p in parts.values())
    assert (parts["train"].label_mature_at < fold.validation_start).all()
    assert (parts["early_stop"].label_mature_at < pd.Timestamp(
        bounds["early_stop_end_exclusive_utc"])).all()
    assert (parts["check"].label_mature_at < fold.test_start).all()
    assert parts["train"].bar_close_at.max() < fold.validation_start
    assert parts["early_stop"].bar_close_at.max() < parts["check"].bar_close_at.min()


def test_bad_future_maturity_is_rejected():
    raw = synthetic()
    raw[3].loc[0, "label_mature_at"] += pd.Timedelta(minutes=5)
    with pytest.raises(ValueError, match="invalid label maturation"):
        align_horizon_examples(raw)
