"""Synthetic-only tests: no P0, P2, P3 or reserved OOS access."""
import numpy as np
import pandas as pd
import pytest
from mlpipeline.pullback_frequency import count_features, frequency_report


def sample():
    stamps = pd.to_datetime([
        "2026-01-31T23:50:00Z", "2026-01-31T23:55:00Z",
        "2026-02-01T00:00:00Z", "2026-02-01T00:05:00Z",
    ], utc=True)
    return pd.DataFrame({
        "symbol": ["BTCUSDT"] * len(stamps), "bar_close_at": stamps,
        "ema9": [np.nan, 100., 101., 102.],
        "ema26": [np.nan, 99., 99., 100.],
        "atr14": [np.nan, 1., 1., 1.],
        "uptrend_prior": [False, True, True, True],
        "first_ema9_touch": [False, False, True, True],
        "pullback_raw": [False, False, True, True],
        "pullback_signal": [False, False, True, False],
    })


def test_monthly_census_and_reconciliation():
    report = count_features(sample())
    assert report["all"]["closed_bars"] == 4
    assert report["all"]["valid_indicators"] == 3
    assert report["all"]["raw_candidates"] == 2
    assert report["all"]["cooldown_selected"] == 1
    assert report["all"]["suppressed_by_cooldown"] == 1
    assert [item["month"] for item in report["by_month"]] == ["2026-01", "2026-02"]
    assert [item["cooldown_selected"] for item in report["by_month"]] == [0, 1]


def test_reject_held_out_or_future_symbol():
    frame = sample(); frame["symbol"] = "ADAUSDT"
    with pytest.raises(ValueError, match="Unapproved"):
        count_features(frame)
    frame = sample(); frame.loc[3, "bar_close_at"] = pd.Timestamp("2026-07-01T00:00:00Z")
    with pytest.raises(ValueError, match="out-of-scope"):
        count_features(frame)


def test_reject_impossible_selection_or_invalid_indicator():
    frame = sample(); frame.loc[1, "pullback_signal"] = True
    with pytest.raises(ValueError, match="outside raw events"):
        count_features(frame)
    frame = sample(); frame.loc[2, "atr14"] = np.nan
    with pytest.raises(ValueError, match="incomplete indicators"):
        count_features(frame)


def test_reject_unsorted_or_non_utc():
    with pytest.raises(ValueError, match="Nonchronological"):
        count_features(sample().iloc[::-1].reset_index(drop=True))
    frame = sample(); frame["bar_close_at"] = frame.bar_close_at.dt.tz_localize(None)
    with pytest.raises(ValueError, match="Nonchronological"):
        count_features(frame)


def test_fingerprint_and_four_asset_gate():
    with pytest.raises(ValueError, match="Wrong approved"):
        frequency_report({}, "a"*64)
    with pytest.raises(ValueError, match="Wrong approved"):
        frequency_report({x: sample() for x in ("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")}, "BAD")
