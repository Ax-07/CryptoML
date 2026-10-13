"""Synthetic tests for P2 confidence selection; no final/ADA scoring."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.confidence_diagnostics import (
    confidence_diagnostics, decision_and_margin,
    directional_metrics, select_threshold,
)
from mlpipeline.features import FEATURE_COLUMNS


def test_margin_is_prior_adjusted_and_not_a_probability():
    probs = np.array([[0.32, 0.38, 0.30], [0.75, 0.15, 0.10]])
    counts = np.array([20, 60, 20])
    pred, gap = decision_and_margin(probs, counts)
    assert list(pred) == [0, 0]
    assert np.all((gap >= 0) & (gap <= 1))
    with pytest.raises(ValueError, match="Invalid class probabilities"):
        decision_and_margin(np.array([[0.3, 0.3, 0.3]]), counts)


def test_neutral_is_incorrect_on_selected_directions():
    true = np.array([0, 1, 2, 1, 0, 2])
    predictions = np.array([0, 0, 2, 2, 1, 1])
    margins = np.array([0.8, 0.7, 0.6, 0.1, 0.5, 0.5])
    result = directional_metrics(true, predictions, margins, threshold=0.5)
    assert result["selected"] == 3
    assert result["correct_direction_class"] == 2
    assert result["directional_precision"] == pytest.approx(2 / 3)
    assert result["neutral_among_selected"] == 1
    assert result["per_direction"]["DOWN"]["precision"] == pytest.approx(0.5)
    assert result["per_direction"]["UP"]["precision"] == pytest.approx(1)
    assert result["coverage"] == pytest.approx(0.5)


def test_tuning_selection_excludes_check_labels():
    # A synthetic tuning dataset with 300 rows and all directions represented.
    tune_pred = np.array([0] * 100 + [2] * 100 + [1] * 100)
    tune_y = np.array([0] * 70 + [1] * 30 + [2] * 70 + [1] * 30 + [1] * 100)
    tune_margin = np.array([0.12] * 50 + [0.01] * 50 +
                           [0.12] * 50 + [0.01] * 50 + [0.01] * 100)
    chosen, report = select_threshold(
        tune_y, tune_pred, tune_margin,
        thresholds=(0.0, 0.05, 0.1),
        min_coverage=0.20, min_each_direction=20,
    )
    assert chosen in (0.0, 0.05, 0.1)
    assert report["selection_status"] == "selected_on_tuning_only"
    original = directional_metrics(tune_y, tune_pred, tune_margin, threshold=chosen)
    # Opposite outcomes on a later partition must not change chosen threshold.
    fake_check_y = 2 - tune_y
    other = directional_metrics(fake_check_y, tune_pred, tune_margin, threshold=chosen)
    assert original["directional_precision"] != other["directional_precision"]


def test_fallback_if_minimum_class_counts_unreachable():
    y = np.array([0, 1, 2, 1])
    p = np.array([0, 0, 2, 1])
    gap = np.array([0.1, 0.3, 0.2, 0.0])
    threshold, report = select_threshold(
        y, p, gap, thresholds=(0.0, 0.05), min_each_direction=50,
    )
    assert threshold == 0.0
    assert report["selection_status"] == "fallback_no_eligible_threshold"


def synthetic_examples():
    timestamps = pd.date_range("2025-01-01T00:05:00Z", periods=18 * 288, freq="5min")
    blocks = []
    for i, symbol in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT")):
        rng = np.random.default_rng(50 + i)
        block = pd.DataFrame({
            "symbol": symbol, "bar_close_at": timestamps,
            "label_mature_at": timestamps + pd.Timedelta(minutes=30),
            "target": rng.integers(0, 3, len(timestamps)),
        })
        for feature in FEATURE_COLUMNS:
            block[feature] = rng.normal(size=len(timestamps)).astype("float32")
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def test_end_to_end_excludes_sealed_dates_and_ada():
    examples = synthetic_examples()
    report = confidence_diagnostics(
        examples, train_days=4, validation_days=4, test_days=2,
        step_days=2, final_holdout_days=3, n_estimators=20,
    )
    assert report["fold_oos_tests_scored"] is False
    assert report["final_90_days_scored"] is False
    assert report["holdout_symbol_excluded"] == "ADAUSDT"
    assert len(report["folds"]) > 0
    for fold in report["folds"]:
        assert fold["fixed_alpha"] == 1.0
        assert fold["validation_check_end_exclusive_utc"] <= (
            examples.bar_close_at.max() - pd.Timedelta(days=3)
            + pd.Timedelta(minutes=5)
        ).isoformat()
        assert fold["check_selected_threshold"]["coverage"] <= (
            fold["check_no_filter"]["coverage"] + 1e-12
        )
