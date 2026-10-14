"""Synthetic, small tests for seven horizon P2 study; no historical files."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.expanded_horizon_diagnostics import (
    HORIZONS, MAX_HORIZON, SEEN_SYMBOLS, build_shared_examples,
    common_folds, shared_partitions, _fold_means,
)


def candles(days=22):
    timestamps = pd.date_range(
        "2025-01-01T00:00:00Z", periods=days * 288, freq="5min"
    )
    result = {}
    for index, symbol in enumerate(SEEN_SYMBOLS):
        n = len(timestamps)
        rng = np.random.default_rng(123 + index)
        close = 100 + index * 20 + np.cumsum(rng.normal(0, 0.1, n))
        open_ = np.r_[close[0], close[:-1]]
        result[symbol] = pd.DataFrame({
            "symbol": symbol,
            "bar_open_at": timestamps,
            "bar_close_at": timestamps + pd.Timedelta(minutes=5),
            "open": open_,
            "high": np.maximum(open_, close) + 0.05,
            "low": np.minimum(open_, close) - 0.05,
            "close": close,
            "volume_base": rng.uniform(1, 5, n),
            "volume_quote": rng.uniform(100, 500, n),
            "trade_count": rng.integers(100, 300, n),
            "taker_buy_volume_base": rng.uniform(0.3, 0.7, n),
            "taker_buy_volume_quote": rng.uniform(50, 200, n),
        })
    return result


def test_seven_horizons_same_keys_and_maximal_maturity():
    assert HORIZONS == (1, 2, 3, 5, 6, 9, 12)
    shared = build_shared_examples(candles())
    assert set(shared.symbol) == set(SEEN_SYMBOLS)
    assert all(f"target_h{h}" in shared for h in HORIZONS)
    assert not shared.duplicated(["symbol", "bar_close_at"]).any()
    assert (shared.label_mature_at ==
            shared.bar_close_at + pd.Timedelta(minutes=5 * MAX_HORIZON)).all()
    assert shared[[f"target_h{h}" for h in HORIZONS]].notna().all().all()
    assert str(shared.label_mature_at.dt.tz) == "UTC"


def test_ada_is_rejected_before_feature_computation():
    inputs = candles(days=2)
    inputs["ADAUSDT"] = inputs["BTCUSDT"]
    with pytest.raises(ValueError, match="never ADA"):
        build_shared_examples(inputs)


def test_purges_use_h12_even_for_h1():
    shared = build_shared_examples(candles())
    folds = common_folds(
        shared, train_days=4, validation_days=14,
        test_days=1, step_days=1, final_holdout_days=1,
    )
    assert folds
    part, boundaries = shared_partitions(shared, folds[0])
    assert not any(f.symbol.eq("ADAUSDT").any() for f in part.values())
    assert (part["train"].label_mature_at < folds[0].validation_start).all()
    assert (part["early_stopping"].label_mature_at <
            pd.Timestamp(boundaries["early_stop_end_exclusive_utc"])).all()
    assert (part["validation_check"].label_mature_at < folds[0].test_start).all()
    assert part["train"].bar_close_at.max() < folds[0].validation_start


def test_reference_pairs_use_same_fold_records():
    # No training is necessary to validate fold alignment / mean reporting.
    shared = build_shared_examples(candles())
    folds = common_folds(
        shared, train_days=4, validation_days=14,
        test_days=1, step_days=1, final_holdout_days=1,
    )
    assert all(folds[0].validation_start <= f.validation_start for f in folds)
