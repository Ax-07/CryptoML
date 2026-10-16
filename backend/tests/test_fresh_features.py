"""Synthetic-only causal fresh M5 feature adapter tests."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.fresh_features import prepare_fresh_features


def candles(n=250):
    t = pd.date_range("2026-10-13T08:15:00Z", periods=n, freq="5min")
    rows = []
    for i, symbol in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT")):
        rng = np.random.default_rng(10+i)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
        opened = close * 0.9999
        high = np.maximum(opened, close) * 1.001
        low = np.minimum(opened, close) * 0.999
        vol = rng.uniform(10, 50, n)
        rows.append(pd.DataFrame({
            "symbol": symbol, "bar_open_at": t,
            "bar_close_at": t + pd.Timedelta(minutes=5),
            "is_final": True,
            "ingested_at": t + pd.Timedelta(minutes=5, seconds=1),
            "open": opened, "high": high, "low": low, "close": close,
            "volume_base": vol, "volume_quote": vol * close,
            "trade_count": rng.integers(100, 1000, n),
            "taker_buy_volume_base": vol * 0.53,
            "taker_buy_volume_quote": vol * 0.53 * close,
        }))
    return pd.concat(rows, ignore_index=True)


def timestamp(x):
    return x.bar_close_at.max() + pd.Timedelta(seconds=10)


def test_produces_five_synced_feature_vectors():
    df = candles()
    features = prepare_fresh_features(df, observed_at=timestamp(df))
    assert set(features.symbol) == {
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"
    }
    assert features.bar_close_at.nunique() == 1
    assert len(features) == 5
    assert np.isfinite(features[FEATURE_COLUMNS].to_numpy(dtype=float)).all()


def test_future_prices_cannot_influence_features():
    df = candles()
    now = timestamp(df)
    a = prepare_fresh_features(df, observed_at=now)
    altered = df.copy()
    altered["close"] *= 10
    altered["open"] *= 10
    altered["high"] *= 10
    altered["low"] *= 10
    # Scaling all prices leaves relative OHLCV features invariant; a real
    # future candle must be rejected outright.
    next_bars = altered.loc[altered.bar_close_at.eq(altered.bar_close_at.max())].copy()
    next_bars["bar_open_at"] += pd.Timedelta(minutes=5)
    next_bars["bar_close_at"] += pd.Timedelta(minutes=5)
    next_bars["ingested_at"] += pd.Timedelta(minutes=5)
    with pytest.raises(ValueError, match="Future candle"):
        prepare_fresh_features(
            pd.concat([df, next_bars], ignore_index=True),
            observed_at=now
        )
    assert len(a) == 5


def test_fails_closed_on_stale_missing_and_unfinalized_data():
    df = candles()
    with pytest.raises(ValueError, match="stale"):
        prepare_fresh_features(df, observed_at=timestamp(df) + pd.Timedelta(minutes=10))
    incomplete = df.copy()
    incomplete.loc[incomplete.index[-1], "is_final"] = False
    with pytest.raises(ValueError, match="Unfinalized"):
        prepare_fresh_features(incomplete, observed_at=timestamp(incomplete))
    missing = df.loc[~(
        df.symbol.eq("ETHUSDT") &
        df.bar_close_at.eq(df.bar_close_at.max())
    )]
    with pytest.raises(ValueError, match="Asynchronous"):
        prepare_fresh_features(missing, observed_at=timestamp(df))
    gap = df.loc[~(
        df.symbol.eq("BTCUSDT") &
        df.bar_close_at.eq(df.bar_close_at.max() - pd.Timedelta(minutes=15))
    )]
    with pytest.raises(ValueError, match="Insufficient"):
        prepare_fresh_features(gap, observed_at=timestamp(gap))


def test_rejects_future_label_input():
    df = candles()
    df["target"] = 1
    with pytest.raises(ValueError, match="Future labels"):
        prepare_fresh_features(df, observed_at=timestamp(df))
