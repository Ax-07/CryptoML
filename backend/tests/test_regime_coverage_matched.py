"""P2 broad-regime 40%-coverage matching checks, synthetic data only."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.regime_coverage_matched import (
    MIN_TUNE_ROWS, TARGET_COVERAGE, _check_comparison, evaluate_cell,
    fit_cell_threshold, summarize_regime_cells, tuning_partition,
)
from mlpipeline.walkforward import WalkForwardFold


def test_tuning_fits_without_truth_and_insufficient_cells_stay_excluded():
    n = 500
    p = np.tile([0, 2], n // 2)
    margins = np.linspace(0.01, 0.99, n)
    mask = np.ones(n, dtype=bool)
    result = fit_cell_threshold(p, margins, mask)
    assert result["feasible"] is True
    assert result["target_coverage"] == TARGET_COVERAGE
    assert result["tuning_selected_coverage"] == pytest.approx(0.4)
    insufficient = fit_cell_threshold(p, margins, np.arange(n) < MIN_TUNE_ROWS - 1)
    assert insufficient["feasible"] is False
    assert insufficient["reason"] == "insufficient_tuning_rows"
    assert insufficient["threshold"] is None


def test_no_forced_coverage_when_direction_predictions_are_too_few():
    decisions = np.r_[np.ones(420, dtype=int), np.zeros(80, dtype=int)]
    margins = np.linspace(0, 1, 500)
    result = fit_cell_threshold(
        decisions, margins, np.ones(500, dtype=bool),
    )
    assert result["feasible"] is False
    assert result["threshold"] is None


def test_infeasible_cells_never_evaluate_check_outcomes():
    frame = pd.DataFrame({"symbol": ["BTCUSDT"] * 4})
    policies = {str(h): {"feasible": False, "threshold": None} for h in (2, 6)}
    assert evaluate_cell(frame, np.ones(4, bool), {}, policies)["check"] == {}


def test_check_coverage_is_measured_not_retuned():
    frame = pd.DataFrame({"symbol": ["BTCUSDT"] * 100})
    checks = {}
    for h in (2, 6):
        checks[str(h)] = {
            "truth": np.arange(100) % 3,
            "decisions": np.r_[np.zeros(70, dtype=int), np.ones(30, dtype=int)],
            "margins": np.linspace(0, 1, 100),
        }
    policies = {
        "2": {"feasible": True, "threshold": 0.5},
        "6": {"feasible": True, "threshold": 0.8},
    }
    result = evaluate_cell(frame, np.ones(100, bool), checks, policies)
    assert result["paired_tuning_feasible"] is True
    assert result["check"]["2"]["coverage"] != TARGET_COVERAGE
    assert result["comparison"]["coverage_delta_points"] == pytest.approx(20)
    assert result["check"]["6"]["coverage"] == 0
    assert result["comparison"]["precision_delta_points"] is None


def test_tuning_period_is_strictly_h12_purged():
    start = pd.Timestamp("2025-01-11T00:00:00Z")
    close = pd.date_range(start, periods=14 * 288, freq="5min")
    frame = pd.DataFrame({
        "symbol": "BTCUSDT", "bar_close_at": close,
        "label_mature_at": close + pd.Timedelta(minutes=60),
    })
    fold = WalkForwardFold(
        0, start - pd.Timedelta(days=120), start,
        start + pd.Timedelta(days=14),
        start + pd.Timedelta(days=28),
    )
    tune = tuning_partition(frame, fold)
    stop = start + pd.Timedelta(days=7)
    end = start + pd.Timedelta(days=10)
    assert tune.bar_close_at.min() == stop
    assert tune.bar_close_at.max() < end - pd.Timedelta(minutes=60)
    assert (tune.label_mature_at < end).all()


def test_pair_requires_identical_check_sample_count():
    sample = {
        "n_total": 100, "selected": 40,
        "directional_precision": 0.35, "uplift_points": 4,
        "coverage": 0.4,
    }
    bad = dict(sample, n_total=90)
    with pytest.raises(AssertionError, match="not aligned"):
        _check_comparison(sample, bad)
