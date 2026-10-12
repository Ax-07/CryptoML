"""P1 tests: features cannot see future candles, holes cannot be bridged."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.features import FEATURE_COLUMNS, build_symbol_features, join_market_context
from mlpipeline.labels import add_labels, NEUTRAL, UP, DOWN


def candles(symbol="BTCUSDT", n=180, seed=1, offset=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    close = (100 + offset) * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    opened = close * np.exp(rng.normal(0, 0.0003, n))
    high = np.maximum(close, opened) * 1.0005
    low = np.minimum(close, opened) * 0.9995
    volume = rng.uniform(10, 50, n)
    return pd.DataFrame({
        "symbol": symbol, "bar_open_at": t, "bar_close_at": t + pd.Timedelta(minutes=5),
        "open": opened, "high": high, "low": low, "close": close,
        "volume_base": volume, "volume_quote": volume*close,
        "trade_count": rng.integers(100, 1000, n), "taker_buy_volume_base": volume*.53,
        "taker_buy_volume_quote": volume*.53*close,
    })


def test_future_mutation_cannot_change_features_at_t():
    source = candles()
    f1 = build_symbol_features(source)
    modified = source.copy()
    modified.loc[130:, ["open", "high", "low", "close"]] *= 2
    f2 = build_symbol_features(modified)
    np.testing.assert_allclose(f1.loc[120, FEATURE_COLUMNS[:26]].to_numpy(dtype=float),
                               f2.loc[120, FEATURE_COLUMNS[:26]].to_numpy(dtype=float),
                               rtol=0, atol=1e-12, equal_nan=True)


def test_missing_bar_resets_windows_and_invalidates_boundary_labels():
    source = candles()
    source = source.drop(index=130).reset_index(drop=True)
    f = build_symbol_features(source)
    assert f.loc[130, "segment_id"] != f.loc[129, "segment_id"]
    assert np.isnan(f.loc[130, "ret_1"])
    y = add_labels(f, horizon=6)
    assert y.loc[125, "target"] != y.loc[125, "target"]  # NaN, cannot cross the gap
    assert pd.isna(y.loc[len(y)-1, "target"])


def test_labels_use_correct_exclusive_maturity():
    f = build_symbol_features(candles())
    labeled = add_labels(f, horizon=6)
    row = labeled.iloc[100]
    assert row.label_mature_at == row.bar_close_at + pd.Timedelta(minutes=30)
    expected_ret = f.iloc[106]["close"] / f.iloc[100]["close"] - 1
    assert np.isclose(row.future_return, expected_ret)
    assert row.target in (DOWN, NEUTRAL, UP)


def test_context_strict_timestamp_matching():
    btc = build_symbol_features(candles())
    eth = build_symbol_features(candles("ETHUSDT", seed=2))
    target = build_symbol_features(candles("SOLUSDT", seed=3))
    eth = eth.drop(index=100)
    joined = join_market_context(target, btc, eth)
    assert pd.isna(joined.loc[100, "eth_ret_3"])
    assert pd.notna(joined.loc[101, "eth_ret_3"])


def test_invalid_label_settings():
    f = build_symbol_features(candles())
    with pytest.raises(ValueError):
        add_labels(f, horizon=0)
    with pytest.raises(ValueError):
        add_labels(f, threshold_multiplier=-0.2)
