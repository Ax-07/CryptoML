"""Tests for immutable, forward-only inference logging with a stub booster."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mlpipeline.features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from mlpipeline.forward_signals import predict_snapshot, write_snapshots_once


class FakeBooster:
    def predict(self, x, num_threads=None):
        return np.tile([0.2, 0.3, 0.5], (len(x), 1))


def fake_bundle(_):
    return FakeBooster(), {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "final_holdout_end_utc": "2026-09-30T23:35:00+00:00",
        "fitted_labels_before_utc": "2026-06-18T23:35:00+00:00",
        "model_id": "frozen-test-sha",
        "class_prior": [0.3, 0.45, 0.25],
    }


def sample_features():
    df = pd.DataFrame({
        "symbol": ["BTCUSDT", "ETHUSDT"],
        "bar_close_at": pd.to_datetime([
            "2026-10-13T09:05:00Z", "2026-10-13T09:05:00Z"
        ]),
    })
    for name in FEATURE_COLUMNS:
        df[name] = 0.1
    return df


def test_fresh_predictions_recorded_once(tmp_path):
    with patch("mlpipeline.forward_signals.load_bundle", side_effect=fake_bundle):
        rows = predict_snapshot(
            Path("bundle"), sample_features(),
            recorded_at=pd.Timestamp("2026-10-13T09:05:03Z"),
        )
    assert len(rows) == 2
    assert rows[0]["model_id"] == "frozen-test-sha"
    assert rows[0]["p_up"] == pytest.approx(0.5)
    paths = write_snapshots_once(rows, tmp_path)
    assert len(paths) == 2
    assert json.loads(paths[0].read_text())["symbol"] == "BTCUSDT"
    with pytest.raises(FileExistsError):
        write_snapshots_once(rows, tmp_path)


def test_reject_stale_and_unsafe_snapshots():
    with patch("mlpipeline.forward_signals.load_bundle", side_effect=fake_bundle):
        with pytest.raises(ValueError, match="Stale"):
            predict_snapshot(
                Path("bundle"), sample_features(),
                recorded_at=pd.Timestamp("2026-10-13T09:10:00Z"),
            )
        old = sample_features()
        old["bar_close_at"] = pd.Timestamp("2026-09-30T23:30:00Z")
        with pytest.raises(ValueError, match="Retrospective"):
            predict_snapshot(
                Path("bundle"), old,
                recorded_at=pd.Timestamp("2026-09-30T23:30:01Z"),
            )
        duplicate = sample_features()
        duplicate.loc[1, "symbol"] = "BTCUSDT"
        with pytest.raises(ValueError, match="one just-closed"):
            predict_snapshot(
                Path("bundle"), duplicate,
                recorded_at=pd.Timestamp("2026-10-13T09:05:01Z"),
            )
