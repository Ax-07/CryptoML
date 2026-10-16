"""Synthetic checks for hierarchical movement/conditional-direction CPU research."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.two_stage_diagnostics import (
    combine_stage_probabilities, compare_two_stage, _scored_candidate,
)


def test_exact_composition_with_no_oracle_gate():
    move = np.array([0.0, 0.4, 1.0])
    up = np.array([0.8, 0.75, 0.25])
    p = combine_stage_probabilities(move, up)
    np.testing.assert_allclose(
        p, [[0, 1, 0], [0.1, 0.6, 0.3], [0.75, 0, 0.25]], atol=1e-12,
    )
    np.testing.assert_allclose(p.sum(axis=1), 1)
    with pytest.raises(ValueError):
        combine_stage_probabilities(np.array([1.2]), np.array([0.3]))


def test_directional_neutral_and_opposite_errors_sum_to_selected():
    truth = np.array([0, 1, 2, 1, 0, 2])
    p = np.array([
        [0.80, 0.10, 0.10],
        [0.80, 0.10, 0.10],
        [0.10, 0.10, 0.80],
        [0.10, 0.10, 0.80],
        [0.10, 0.10, 0.80],
        [0.10, 0.10, 0.80],
    ])
    metrics = _scored_candidate(truth, p, np.array([1, 1, 1]))
    selected = metrics["directional"]["selected"]
    breakdown = metrics["directional"]["error_breakdown"]
    assert selected == 6
    assert breakdown == {
        "correct": 3, "neutral": 2, "opposite": 1,
        "neutral_fraction_among_selected": 2 / 6,
        "opposite_fraction_among_selected": 1 / 6,
    }
    assert metrics["raw_probabilities"]["log_loss"] is not None


def synthetic_examples():
    times = pd.date_range("2025-01-01T00:05:00Z", periods=18 * 288, freq="5min")
    rows = []
    for i, symbol in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT")):
        rng = np.random.default_rng(100 + i)
        frame = pd.DataFrame({
            "symbol": symbol,
            "bar_close_at": times,
            "label_mature_at": times + pd.Timedelta(minutes=30),
            "target": rng.integers(0, 3, len(times)),
        })
        for feature in FEATURE_COLUMNS:
            frame[feature] = rng.normal(size=len(times)).astype("float32")
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def test_fold_comparison_remains_within_dev_validation():
    data = synthetic_examples()
    report = compare_two_stage(
        data, train_days=4, validation_days=4, test_days=2,
        step_days=2, final_holdout_days=3, n_estimators=20,
    )
    assert report["holdout_symbol_excluded"] == "ADAUSDT"
    assert report["final_90_days_scored"] is False
    assert report["fold_oos_tests_scored"] is False
    assert report["tuning_buffer_used"] is False
    assert report["fixed_alpha"] == 1.0
    assert len(report["folds"]) > 0
    holdout_start = (
        data.loc[data.symbol.ne("ADAUSDT"), "bar_close_at"].max()
        + pd.Timedelta(minutes=5) - pd.Timedelta(days=3)
    )
    for fold in report["folds"]:
        assert pd.Timestamp(fold["validation_check_end_exclusive_utc"]) <= holdout_start
        assert fold["counts"]["stage2_train_movers"] < fold["counts"]["train"]
        assert fold["counts"]["stage2_early_movers"] < fold["counts"]["early_stopping"]
        for model in ("three_class", "two_stage"):
            assert (fold["models"][model]["raw_probabilities"]["log_loss"] > 0)
            assert fold["models"][model]["directional"]["n_total"] == fold["counts"]["validation_check"]
    assert set(report["fold_mean_summary"]) == {"three_class", "two_stage"}


def test_no_ada_is_fitted_even_if_ada_targets_mutate():
    data = synthetic_examples()
    reduced = data.loc[data.symbol.ne("ADAUSDT")]
    assert len(reduced) > 0
    mutated = data.copy()
    mutated.loc[mutated.symbol.eq("ADAUSDT"), "target"] = 1
    # Demonstrate fold geometry depends on seen dates rather than ADA labels.
    from mlpipeline.walkforward import make_walk_forward_folds
    a = make_walk_forward_folds(
        data, holdout_symbol="ADAUSDT", train_days=4,
        validation_days=4, test_days=2, step_days=2,
        final_holdout_days=3,
    )
    b = make_walk_forward_folds(
        mutated, holdout_symbol="ADAUSDT", train_days=4,
        validation_days=4, test_days=2, step_days=2,
        final_holdout_days=3,
    )
    assert a == b
