"""Validation-only experiment tests: no labels from OOS or unseen asset used."""
import numpy as np
import pandas as pd

from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.validation_diagnostics import compare_class_weights_on_validation


def make_data():
    times = pd.date_range("2025-01-01T00:05:00Z", periods=14 * 288, freq="5min")
    frames = []
    for i, symbol in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT")):
        rng = np.random.default_rng(70 + i)
        frame = pd.DataFrame({
            "symbol": symbol,
            "bar_close_at": times,
            "label_mature_at": times + pd.Timedelta(minutes=30),
            "target": rng.integers(0, 3, size=len(times)),
        })
        for feature in FEATURE_COLUMNS:
            frame[feature] = rng.normal(size=len(times)).astype("float32")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_validation_only_uses_training_class_counts_and_seals_holdouts():
    report = compare_class_weights_on_validation(
        make_data(), holdout_symbol="ADAUSDT",
        train_days=4, validation_days=2, test_days=2, step_days=2,
        final_holdout_days=2, n_estimators=20,
    )
    assert report["fold_oos_and_final_test_scored"] is False
    assert len(report["folds"]) == 3
    for fold in report["folds"]:
        assert fold["counts"]["train"] == sum(fold["train_class_counts"].values())
        assert fold["counts"]["model_selection"] > 100
        assert set(fold["candidates"]) == {
            "majority_baseline", "standard", "balanced_train_only"
        }
        for name in ("standard", "balanced_train_only"):
            assert fold["candidates"][name]["metrics"]["n"] == fold["counts"]["model_selection"]
            assert sum(fold["candidates"][name]["predicted_counts"].values()) == fold["counts"]["model_selection"]
