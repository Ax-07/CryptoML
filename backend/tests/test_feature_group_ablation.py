"""P2 feature-group ablation protocol tests on synthetic data."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.feature_group_ablation import (
    GROUPS, VARIANTS, columns_for_variant, validate_feature_groups,
    _effects, _train_and_evaluate,
)
from mlpipeline.features import FEATURE_COLUMNS


def test_feature_groups_partition_30_columns_exactly():
    validate_feature_groups()
    assert len(GROUPS) == 7
    assert len(columns_for_variant("full")) == 30
    for group in GROUPS:
        columns = columns_for_variant(group)
        assert len(columns) == len(FEATURE_COLUMNS) - len(GROUPS[group])
        assert not set(columns).intersection(GROUPS[group])


def test_unknown_group_rejected():
    with pytest.raises(ValueError):
        columns_for_variant("fake")


def test_effect_sign_convention():
    def scores(precision, coverage, uplift, loss):
        return {
            "metrics": {
                "directional": {
                    "directional_precision": precision,
                    "coverage": coverage,
                    "uplift_points": uplift,
                    "conditional_uplift_points": uplift,
                },
                "fixed_alpha_decisions": {
                    "macro_f1": 0.4, "balanced_accuracy": 0.4,
                },
                "raw_probabilities": {
                    "log_loss": loss, "multiclass_brier": loss * 0.6,
                },
            }
        }
    better = scores(0.35, 0.6, 5, 1.05)
    ablated = scores(0.33, 0.58, 4, 1.08)
    effect = _effects(better, ablated)
    assert effect["directional_precision_benefit"] == pytest.approx(2)
    assert effect["coverage_benefit"] == pytest.approx(2)
    assert effect["uplift_points_benefit"] == pytest.approx(1)
    assert effect["log_loss_benefit"] == pytest.approx(0.03)


def test_one_fold_training_runs_with_changed_feature_columns():
    rng = np.random.default_rng(10)
    def part(n):
        symbols = np.asarray(("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"))
        frame = pd.DataFrame({
            "symbol": np.tile(symbols, n // 4),
            "target_h2": np.arange(n) % 3,
            "target_h6": (np.arange(n) + 1) % 3,
        })
        for col in FEATURE_COLUMNS:
            frame[col] = rng.normal(size=n).astype("float32")
        return frame
    train, early, check = part(600), part(152), part(120)
    full = _train_and_evaluate(
        train, early, check, horizon=2, variant="full",
        n_estimators=20, seed=42,
    )
    ablated = _train_and_evaluate(
        train, early, check, horizon=2, variant="market_context",
        n_estimators=20, seed=42,
    )
    assert full["feature_count"] == 30
    assert ablated["feature_count"] == 26
    assert full["metrics"]["directional"]["n_total"] == 120
    assert not set(ablated["feature_gain_importance"]).intersection(
        GROUPS["market_context"]
    )
