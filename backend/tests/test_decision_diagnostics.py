"""Tests for validation-only class decision tuning."""
import numpy as np
import pandas as pd

from mlpipeline.decision_diagnostics import decide, compare_decision_rules
from mlpipeline.features import FEATURE_COLUMNS


def test_decide_does_not_mutate_raw_probabilities():
    probs = np.array([[0.31, 0.42, 0.27], [0.65, 0.20, 0.15]])
    original = probs.copy()
    counts = np.array([20, 60, 20])
    assert np.array_equal(decide(probs, counts, 0.0), probs.argmax(axis=1))
    assert np.array_equal(probs, original)
    assert decide(probs, counts, 1.0).shape == (2,)


def test_decision_rules_only_evaluate_validation_check():
    times = pd.date_range("2025-01-01T00:05:00Z", periods=14 * 288, freq="5min")
    frames = []
    for i, symbol in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT")):
        rng = np.random.default_rng(100 + i)
        frame = pd.DataFrame({
            "symbol": symbol,
            "bar_close_at": times,
            "label_mature_at": times + pd.Timedelta(minutes=30),
            "target": rng.integers(0, 3, len(times)),
        })
        for name in FEATURE_COLUMNS:
            frame[name] = rng.normal(size=len(times)).astype("float32")
        frames.append(frame)
    report = compare_decision_rules(
        pd.concat(frames, ignore_index=True), holdout_symbol="ADAUSDT",
        train_days=4, validation_days=4, test_days=2,
        step_days=2, final_holdout_days=2, n_estimators=20,
    )
    assert report["oos_fold_tests_scored"] is False
    assert report["final_time_holdout_scored"] is False
    assert report["probabilities_unchanged"] is True
    assert len(report["folds"]) == 2
    for fold in report["folds"]:
        scores = fold["validation_check_decisions"]
        assert set(scores) == {"standard", "adjusted"}
        assert scores["standard"]["n"] == fold["counts"]["validation_check"]
        assert scores["adjusted"]["n"] == fold["counts"]["validation_check"]
        assert 0 <= fold["alpha_selected_from_tuning_only"] <= 1
