"""Synthetic-only V3 long-census tests; never open market or reserved data."""
import numpy as np
import pandas as pd
import pytest
from mlpipeline.pullback_long_census_v3 import (
    summarize_long_history, EXPECTED_MONTHS, SYMBOLS,
)
from mlpipeline.pullback_regime_v2 import describe_regimes

def synthetic(symbol):
    segments = [pd.date_range(f"{m}-05T00:00:00Z", periods=70, freq="5min")
                for m in EXPECTED_MONTHS]
    dates = segments[0].append(segments[1:])
    close = 100 + 0.001*np.arange(len(dates))
    return pd.DataFrame({
        "symbol": symbol,
        "bar_open_at": dates,
        "bar_close_at": dates + pd.Timedelta(minutes=5),
        "open": close - 0.01,
        "high": close + 0.10,
        "low": close - 0.10,
        "close": close,
        "is_final": True,
    })

def test_full_21_month_census_reconciles():
    f = {s:describe_regimes(synthetic(s)) for s in SYMBOLS}
    result, table = summarize_long_history(f, "a"*64)
    assert result["total_monthly_partitions"] == 84
    assert len(table) == 84
    assert result["total"]["signals"] == sum(
        result["by_asset"][s]["all_21months"]["signals"] for s in SYMBOLS)
    assert result["total"]["signals"] == result["total"]["classified"] + result["total"]["unclassified"]
    assert result["no_labels_no_returns_no_model_training"]

def test_no_signals_month_is_retained():
    f = {s:describe_regimes(synthetic(s)) for s in SYMBOLS}
    report, table = summarize_long_history(f, "a"*64)
    assert (table["signals"] == 0).all()
    assert (table.filter(regex="^strength_").sum(axis=1) == 0).all()

def test_temporal_reserve_rejected():
    f = {s:describe_regimes(synthetic(s)) for s in SYMBOLS}
    f["BTCUSDT"].loc[0,"bar_close_at"] = pd.Timestamp("2026-07-01T00:00:00Z")
    with pytest.raises(ValueError, match="UTC/scope/chronology"):
        summarize_long_history(f,"a"*64)

def test_unseen_asset_rejected():
    f = {s:describe_regimes(synthetic(s)) for s in SYMBOLS}
    f["ADAUSDT"] = f.pop("BTCUSDT")
    with pytest.raises(ValueError, match="BTC/ETH/SOL/XRP"):
        summarize_long_history(f,"a"*64)

def test_invalid_dataset_fingerprint_rejected():
    f = {s:describe_regimes(synthetic(s)) for s in SYMBOLS}
    with pytest.raises(ValueError, match="fingerprint"):
        summarize_long_history(f, "wrong")
