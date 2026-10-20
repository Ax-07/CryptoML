"""Synthetic-only tests. No data P0, labels, ML fitting or reserved holdouts."""
import pandas as pd
import pytest

from mlpipeline.pullback_regime_census_v2 import summarize_regime_features, SYMBOLS
from mlpipeline.pullback_regime_v2 import describe_regimes


def synthetic(symbol):
    # Make every month present; 65+ bars in each month for EMA warm-up.
    dates = pd.DatetimeIndex([
        item for month in range(12, 14) for item in []
    ], tz="UTC")
    ranges = [
        pd.date_range(f"2025-12-05T00:00:00Z",periods=75,freq="5min"),
        *[pd.date_range(f"2026-{m:02d}-05T00:00:00Z",periods=75,freq="5min")
          for m in range(1, 7)],
    ]
    dates = ranges[0].append(ranges[1:])
    import numpy as np
    close = 100 + 0.01 * np.arange(len(dates))
    return pd.DataFrame({
        "symbol": symbol,
        "bar_open_at": dates,
        "bar_close_at": dates + pd.Timedelta(minutes=5),
        "open": close - 0.02,
        "high": close + 0.10,
        "low": close - 0.10,
        "close": close,
        "is_final": True,
    })


def test_four_assets_reconcile_synthetic():
    features = {s: describe_regimes(synthetic(s)) for s in SYMBOLS}
    report, rows = summarize_regime_features(features, "a" * 64)
    assert report["total"]["events"] == sum(
        v["total"]["events"] for v in report["by_asset"].values()
    )
    assert len(rows) == 4 * 7
    assert (rows["unclassified"] >= 0).all()
    assert report["no_outcomes_no_training_no_heldout_data"] is True
    assert set(rows.symbol) == set(SYMBOLS)


def test_immutability_of_past_census_rows():
    frame = synthetic("BTCUSDT")
    first = describe_regimes(frame)
    new = frame.copy()
    new.loc[200:, ["open", "high", "low", "close"]] *= 1.4
    later = describe_regimes(new)
    pd.testing.assert_frame_equal(first.iloc[:200], later.iloc[:200])


def test_bad_fingerprint_fails_closed():
    features = {s: describe_regimes(synthetic(s)) for s in SYMBOLS}
    with pytest.raises(ValueError, match="fingerprint"):
        summarize_regime_features(features,"not-a-hash")


def test_ada_and_reserved_dates_rejected():
    features = {s: describe_regimes(synthetic(s)) for s in SYMBOLS}
    features["ADAUSDT"] = features.pop("BTCUSDT")
    with pytest.raises(ValueError, match="four seen"):
        summarize_regime_features(features, "a" * 64)

    features = {s: describe_regimes(synthetic(s)) for s in SYMBOLS}
    features["BTCUSDT"].loc[0, "bar_close_at"] = pd.Timestamp("2026-07-01T00:00:00Z")
    with pytest.raises(ValueError, match="Nonchronological"):
        summarize_regime_features(features, "a" * 64)
