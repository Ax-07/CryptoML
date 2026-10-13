"""P1 train/holdout evaluation smoke tests with synthetic price series."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.evaluation import evaluate
from mlpipeline.split import boundaries, split_with_purge
from mlpipeline.trainer import prepare_training_data, fit_validate
from test_ml_features_labels import candles


def test_split_purges_overlap_and_never_fits_holdout():
    source = {
        "BTCUSDT": candles("BTCUSDT", n=650, seed=1),
        "ETHUSDT": candles("ETHUSDT", n=650, seed=2),
        "SOLUSDT": candles("SOLUSDT", n=650, seed=3),
        "XRPUSDT": candles("XRPUSDT", n=650, seed=4),
        "ADAUSDT": candles("ADAUSDT", n=650, seed=5),
    }
    prepared = prepare_training_data(source, horizon=6)
    b = boundaries(prepared, ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
    parts = split_with_purge(prepared, boundaries=b, holdout_symbol="ADAUSDT")
    assert set(parts["train"].symbol) == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}
    assert set(parts["unseen_test"].symbol) == {"ADAUSDT"}
    assert (parts["train"].label_mature_at < b.validation_start).all()
    assert (parts["validation"].label_mature_at < b.test_start).all()
    assert (parts["seen_test"].bar_close_at >= b.test_start).all()
    assert len(parts["train"]) > 100
    assert len(parts["validation"]) > 100
    assert len(parts["unseen_test"]) > 100


def test_model_can_fit_and_evaluate_without_parquet():
    source = {s: candles(s, n=650, seed=i+7) for i,s in enumerate(
        ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"])}
    prepared = prepare_training_data(source, horizon=6)
    report, outputs, fitted = fit_validate(prepared, holdout_symbol="ADAUSDT", n_estimators=30)
    assert report["counts"]["train"] > 100
    assert report["metrics"]["seen_test"]["lightgbm"]["n"] > 100
    assert report["metrics"]["unseen_test"]["lightgbm"]["n"] > 100
    assert len(outputs["predictions"]) == report["counts"]["seen_test"] + report["counts"]["unseen_test"]
    assert set(outputs["predictions"].split.unique()) == {"seen_test", "unseen_test"}
    assert len(fitted.feature_importances_) == 30


def test_evaluator_rejects_wrong_probs():
    with pytest.raises(ValueError):
        evaluate(np.array([1]), np.array([[0.1, 0.1, 0.1]]))
    one = evaluate(np.array([2]), np.array([[0.1, 0.1, 0.8]]))
    assert one["balanced_accuracy"] is None
    assert one["macro_f1"] is None


def test_evaluator_normalizes_tiny_row_sum_roundoff():
    import warnings

    y = np.array([0, 1, 2])
    p = np.array([
        [0.2, 0.3, 0.5 + 2e-7],
        [0.1, 0.8, 0.1 - 3e-7],
        [0.2, 0.2, 0.6 + 1e-7],
    ])
    normalized = p / p.sum(axis=1, keepdims=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score = evaluate(y, p)
    assert not any("do not sum to one" in str(w.message) for w in caught)
    assert score["log_loss"] == pytest.approx(evaluate(y, normalized)["log_loss"])
    assert score["multiclass_brier"] == pytest.approx(
        evaluate(y, normalized)["multiclass_brier"]
    )
