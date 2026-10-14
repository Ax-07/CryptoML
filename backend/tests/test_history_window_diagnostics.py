"""Synthetic tests for fixed-date training-history comparison."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.history_window_diagnostics import (
    common_fold_partitions, compare_training_windows,
)
from mlpipeline.walkforward import make_walk_forward_folds


def synthetic_examples():
    dates = pd.date_range("2025-01-01T00:05:00Z", periods=23 * 288, freq="5min")
    frames = []
    for i, symbol in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT")):
        rng = np.random.default_rng(110 + i)
        frame = pd.DataFrame({
            "symbol": symbol, "bar_close_at": dates,
            "label_mature_at": dates + pd.Timedelta(minutes=30),
            "target": rng.integers(0, 3, len(dates)),
        })
        for feature in FEATURE_COLUMNS:
            frame[feature] = rng.normal(size=len(dates)).astype("float32")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_common_dates_nested_train_and_purge():
    data = synthetic_examples()
    fold = make_walk_forward_folds(
        data, holdout_symbol="ADAUSDT", train_days=9, validation_days=14,
        test_days=1, step_days=1, final_holdout_days=1,
    )[0]
    trains, early, check, boundaries = common_fold_partitions(
        data, fold, train_windows=(4, 7, 9), validation_days=14,
    )
    assert len(trains[4]) < len(trains[7]) < len(trains[9])
    assert trains[4].bar_close_at.max() == trains[9].bar_close_at.max()
    assert trains[4].symbol.eq("ADAUSDT").sum() == 0
    assert (trains[9].label_mature_at < fold.validation_start).all()
    assert (early.label_mature_at < pd.Timestamp(
        boundaries["early_stop_end_exclusive_utc"])).all()
    assert (check.label_mature_at < fold.test_start).all()
    assert early.bar_close_at.max() < check.bar_close_at.min()


def test_rejects_changed_protocol():
    with pytest.raises(ValueError, match="predeclared protocol"):
        compare_training_windows(synthetic_examples(), train_windows=(4, 7, 9))


def test_missing_ada_labels_do_not_change_common_fold_geometry():
    data = synthetic_examples()
    modified = data.copy()
    modified.loc[modified.symbol.eq("ADAUSDT"), "target"] = 1
    options = dict(holdout_symbol="ADAUSDT", train_days=9,
                   validation_days=14, test_days=1, step_days=1,
                   final_holdout_days=1)
    a = make_walk_forward_folds(data, **options)
    b = make_walk_forward_folds(modified, **options)
    assert a == b
