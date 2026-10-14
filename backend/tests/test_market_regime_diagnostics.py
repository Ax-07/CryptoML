"""Synthetic P2 regime diagnostics: bins, chronology, subgroup metrics."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS
from mlpipeline.market_regime_diagnostics import (
    REGIMES, assign_regimes, fit_volatility_cutoffs,
    score_regime_cells, slice_scores, summarize_cell,
)


def fake_train(n=150):
    rows = []
    for i, sym in enumerate(SEEN_SYMBOLS):
        rows.append(pd.DataFrame({
            "symbol": sym,
            "volatility_48": np.linspace(0.001, 0.01, n) * (i + 1),
        }))
    return pd.concat(rows, ignore_index=True)


def fake_check():
    rows = []
    for sym in SEEN_SYMBOLS:
        rows.append(pd.DataFrame({
            "symbol": [sym] * 4,
            "bar_close_at": pd.to_datetime([
                "2025-04-04T02:00:00Z", "2025-04-04T10:00:00Z",
                "2025-04-05T16:00:00Z", "2025-04-06T22:00:00Z",
            ], utc=True),
            "volatility_48": [0.0001, 0.004, 0.009, 0.1],
        }))
    return pd.concat(rows, ignore_index=True)


def test_quantiles_are_train_only_and_per_asset():
    train = fake_train()
    bins = fit_volatility_cutoffs(train)
    check = fake_check()
    assigned = assign_regimes(check, bins)
    assert bins["BTCUSDT"][0] < bins["ETHUSDT"][0]
    assert assigned["hour_utc"].tolist()[:4] == ["02", "10", "16", "22"]
    assert assigned["utc_block_6h"].tolist()[:4] == [
        "00-05", "06-11", "12-17", "18-23",
    ]
    assert assigned["day_type"].tolist()[:4] == [
        "weekday", "weekday", "weekend", "weekend",
    ]
    assert assigned["volatility_48_train_tercile"][0] == "low"
    assert assigned["volatility_48_train_tercile"][3] == "high"
    check["volatility_48"] = np.full(len(check), 1000000.0)
    assert fit_volatility_cutoffs(train) == bins


def test_regime_groups_partition_every_timestamp():
    check = fake_check()
    assignments = assign_regimes(check, fit_volatility_cutoffs(fake_train()))
    actual = np.tile([0, 1, 2, 1], 4)
    predicted = np.tile([0, 0, 2, 1], 4)
    scored = score_regime_cells(check, assignments, actual, predicted)
    for dimension, categories in REGIMES.items():
        assert sum(
            scored[dimension][name]["ALL"]["n_total"]
            for name in categories if scored[dimension][name]["ALL"] is not None
        ) == len(check)
    one = scored["hour_utc"]["02"]["BTCUSDT"]
    assert one["selected"] == 1
    assert one["directional_precision"] == 1
    assert scored["hour_utc"]["03"]["ALL"] is None


def test_empty_selected_does_not_create_fake_precision():
    truth = np.array([1, 0, 2])
    predicted = np.ones(3, dtype=int)
    result = slice_scores(truth, predicted, np.ones(3, dtype=bool))
    assert result["selected"] == 0
    assert result["directional_precision"] is None
    assert result["neutral_fraction_among_selected"] is None


def test_aggregate_keeps_fold_local_random_reference_and_pairs():
    def metric(precision, selected):
        correct = int(precision * selected)
        return {
            "n_total": 100, "selected": selected,
            "correct_direction_class": correct,
            "neutral_among_selected": selected // 3,
            "random_reference_precision": 0.3,
            "directional_precision": correct / selected,
            "uplift_points": 100 * (correct / selected - 0.3),
            "coverage": selected / 100,
        }
    folds = []
    for _ in range(2):
        folds.append({"horizons": {
            "2": {"regimes": {"day_type": {"weekday": {
                "BTCUSDT": metric(0.4, 50),
            }}}},
            "6": {"regimes": {"day_type": {"weekday": {
                "BTCUSDT": metric(0.3, 40),
            }}}},
        }})
    summary = summarize_cell(folds, "day_type", "weekday", "BTCUSDT")
    assert summary["paired"]["folds_with_both_min_20_selected"] == 2
    assert summary["paired"]["H2_higher_uplift_folds"] == 2
    assert summary["paired"]["H2_minus_H6_precision_points_mean"] == pytest.approx(10)
    assert summary["H2"]["pooled_uplift_points"] == pytest.approx(10)


def test_incomplete_asset_volatility_fit_is_rejected():
    with pytest.raises(ValueError):
        fit_volatility_cutoffs(fake_train().query("symbol != 'BTCUSDT'"))
