"""Synthetic-only tests of prespecified trajectory features; no P0 access."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.trajectory_direction_features import (
    TRAJECTORY_COLUMNS, WINDOW, build_trajectory_features,
)


def candles(closes, *, gap_at=None, constant_range=False):
    vals = np.asarray(closes, dtype=float)
    times = pd.date_range("2025-01-01T00:00:00Z", periods=len(vals), freq="5min")
    if gap_at is not None:
        times = pd.DatetimeIndex([
            t + (pd.Timedelta(minutes=5) if i >= gap_at else pd.Timedelta(0))
            for i, t in enumerate(times)
        ])
    spread = 0.0 if constant_range else 0.2
    return pd.DataFrame({
        "symbol":"BTCUSDT", "bar_open_at":times,
        "bar_close_at":times + pd.Timedelta(minutes=5),
        "high":vals + spread, "low":vals - spread, "close":vals,
    })


def test_schema_and_monotone_path():
    assert WINDOW == 12
    assert len(FEATURE_COLUMNS) == 30
    assert not set(TRAJECTORY_COLUMNS) & set(FEATURE_COLUMNS)
    frame = candles(np.arange(100.0, 122.0))
    got = build_trajectory_features(frame)
    assert got[list(TRAJECTORY_COLUMNS)].iloc[:12].isna().all().all()
    assert got.loc[12, "trajectory_signed_efficiency_12"] == pytest.approx(1.0)
    assert got.loc[12, "trajectory_prior_range_position_12"] > 1.0


def test_flat_zero_denominator_conventions():
    frame = candles(np.full(20, 100.0), constant_range=True)
    got = build_trajectory_features(frame)
    assert got.loc[12, "trajectory_signed_efficiency_12"] == 0.0
    assert got.loc[12, "trajectory_prior_range_position_12"] == 0.5


def test_prefix_invariance_and_later_spike():
    rng = np.random.default_rng(10)
    vals = 120 + np.cumsum(rng.normal(size=80))
    full = candles(vals)
    before = build_trajectory_features(full)
    for n in (1, 7, 12, 13, 32, 55, 79):
        shorter = build_trajectory_features(full.iloc[:n].copy())
        pd.testing.assert_frame_equal(
            shorter.reset_index(drop=True),
            before.iloc[:n].reset_index(drop=True))
    changed = full.copy()
    changed.loc[79, ["high", "low", "close"]] = [1000., 999., 999.5]
    after = build_trajectory_features(changed)
    pd.testing.assert_frame_equal(before.iloc[:79], after.iloc[:79])


def test_gap_resets_windows():
    vals = np.linspace(100.0, 120.0, 40)
    original = build_trajectory_features(candles(vals))
    changed = build_trajectory_features(candles(vals, gap_at=20))
    assert np.isfinite(original.loc[20, list(TRAJECTORY_COLUMNS)].to_numpy(dtype=float)).all()
    assert changed.loc[20:31, list(TRAJECTORY_COLUMNS)].isna().all().all()
    assert np.isfinite(changed.loc[32, list(TRAJECTORY_COLUMNS)].to_numpy(dtype=float)).all()


def test_refuses_noncausal_or_invalid_ohlc():
    good = candles(np.linspace(100, 110, 25))
    bad = good.copy()
    bad.loc[5, "close"] = bad.loc[5, "high"] + 5
    with pytest.raises(ValueError, match="Invalid OHLC"):
        build_trajectory_features(bad)
    bad = good.copy()
    bad.loc[5, "bar_close_at"] = bad.loc[5, "bar_close_at"] + pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="Closed UTC M5"):
        build_trajectory_features(bad)
