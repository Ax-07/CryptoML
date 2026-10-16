"""End-to-end CPU walk-forward smoke test on synthetic labeled feature rows."""
import numpy as np
import pandas as pd

from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.walkforward_train import evaluate_walk_forward


def test_cpu_walkforward_runs_sealed_oos():
    times = pd.date_range("2025-01-01T00:05:00Z", periods=14 * 288, freq="5min")
    frames = []
    for i, symbol in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT")):
        rng = np.random.default_rng(42 + i)
        labels = rng.integers(0, 3, size=len(times))
        frame = pd.DataFrame({
            "symbol": symbol,
            "bar_close_at": times,
            "label_mature_at": times + pd.Timedelta(minutes=30),
            "target": labels,
        })
        for j, feature in enumerate(FEATURE_COLUMNS):
            frame[feature] = rng.normal(size=len(times)).astype("float32")
        frames.append(frame)
    report, predictions = evaluate_walk_forward(
        pd.concat(frames, ignore_index=True),
        holdout_symbol="ADAUSDT", train_days=4,
        validation_days=2, test_days=2, step_days=2,
        final_holdout_days=2, n_estimators=20,
    )
    assert report["engine"] == "cpu"
    assert len(report["folds"]) == 3
    assert not predictions.empty
    assert set(predictions.symbol) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    assert not predictions.duplicated(["symbol", "bar_close_at"]).any()
    assert (predictions[["p_down", "p_neutral", "p_up"]].sum(axis=1) - 1).abs().max() < 1e-5
    assert report["aggregate_seen_oos"]["lightgbm"]["n"] == len(predictions)
