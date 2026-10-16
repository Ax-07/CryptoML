"""Tests of tuning-only, paired directional coverage comparisons."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.coverage_matched_diagnostics import (
    matched_coverage_comparison, threshold_for_target,
)
from mlpipeline.features import FEATURE_COLUMNS


def test_tuning_threshold_matches_target_without_labels():
    predictions = np.array([0, 2, 0, 2, 0, 2, 1, 1, 1, 1])
    margins = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.2, 0.2, 0.2])
    fitted = threshold_for_target(predictions, margins, target_coverage=0.3)
    assert fitted["feasible"]
    assert fitted["tuning_selected"] == 3
    assert fitted["tuning_selected_coverage"] == pytest.approx(0.3)


def test_insufficient_directional_predictions_is_explicit():
    predictions = np.array([1, 1, 1, 1, 0, 2, 1, 1, 1, 1])
    margins = np.full(10, 0.2)
    fitted = threshold_for_target(predictions, margins, target_coverage=0.4)
    assert fitted["feasible"] is False
    assert fitted["reason"] == "not_enough_directional_predictions_on_tuning"
    assert fitted["threshold"] is None


def test_boundary_ties_never_silently_break_with_future_outcomes():
    pred = np.array([0, 2, 0, 2, 0, 2, 0, 2, 1, 1])
    margins = np.array([0.9, 0.8, 0.7, 0.7, 0.7, 0.2, 0.1, 0.1, 0.0, 0.0])
    fitted = threshold_for_target(pred, margins, target_coverage=0.3)
    assert fitted["tuning_selected"] == 2
    assert fitted["feasible"] is False
    assert fitted["reason"] == "threshold_ties_prevent_target_coverage"


def test_rejects_invalid_score_arrays():
    with pytest.raises(ValueError):
        threshold_for_target(np.array([0, 1]), np.array([np.nan, 0.2]), target_coverage=0.3)


def synthetic_examples():
    times = pd.date_range("2025-01-01T00:05:00Z", periods=18 * 288, freq="5min")
    rows = []
    for i, symbol in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT")):
        rng = np.random.default_rng(100 + i)
        frame = pd.DataFrame({
            "symbol": symbol, "bar_close_at": times,
            "label_mature_at": times + pd.Timedelta(minutes=30),
            "target": rng.integers(0, 3, len(times)),
        })
        for feature in FEATURE_COLUMNS:
            frame[feature] = rng.normal(size=len(times)).astype("float32")
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def test_integration_dev_only_and_paired_targets():
    data = synthetic_examples()
    report = matched_coverage_comparison(
        data, train_days=4, validation_days=4, test_days=2,
        step_days=2, final_holdout_days=3, n_estimators=20,
    )
    assert report["holdout_symbol_excluded"] == "ADAUSDT"
    assert report["fold_oos_tests_scored"] is False
    assert report["final_90_days_scored"] is False
    assert report["primary_target_coverage"] == 0.3
    assert set(report["summary"]) == {"0.3", "0.4"}
    assert len(report["folds"]) > 0
    holdout_start = (
        data.loc[data.symbol.ne("ADAUSDT"), "bar_close_at"].max()
        + pd.Timedelta(minutes=5) - pd.Timedelta(days=3)
    )
    for fold in report["folds"]:
        assert pd.Timestamp(fold["validation_check_end_exclusive_utc"]) <= holdout_start
        for target in ("0.3", "0.4"):
            row = fold["target_results"][target]
            policies = row["policies_fit_only_on_tuning"]
            assert set(policies) == {"three_class", "two_stage"}
            if row["paired_tuning_feasible"]:
                assert set(row["validation_check_metrics"]) == {
                    "three_class", "two_stage"
                }
                for model in ("three_class", "two_stage"):
                    metrics = row["validation_check_metrics"][model]
                    assert metrics["n_total"] == fold["partition_counts"]["check"]
                    assert 0 <= metrics["coverage"] <= 1
