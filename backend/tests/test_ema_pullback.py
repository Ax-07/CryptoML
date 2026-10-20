"""EMA pullback synthetic tests: no historical holdout and no external calls."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.ema_pullback import (
    HORIZONS, EVENT_FEATURES, build_signal_frame, label_signal_events,
)


def candles(n=240):
    t = pd.date_range("2025-02-01T00:00:00Z", periods=n, freq="5min")
    drift = np.linspace(100, 150, n)
    oscillation = 0.9 * np.sin(np.arange(n) * 0.6)
    c = drift + oscillation
    o = c - 0.12
    return pd.DataFrame({
        "symbol": "BTCUSDT", "bar_open_at": t, "bar_close_at": t + pd.Timedelta(minutes=5),
        "open": o, "close": c,
        "high": np.maximum(o, c) + 0.6,
        "low": np.minimum(o, c) - 0.6,
    })


def test_signals_cannot_see_future_candles():
    base = candles()
    original = build_signal_frame(base)
    changed = base.copy()
    cutoff = 135
    future = changed.index > cutoff
    for col in ("open", "high", "low", "close"):
        changed.loc[future, col] *= 2.0
    later = build_signal_frame(changed)
    cols = ["ema9", "ema26", "atr14", "trend", "trigger", "signal", *EVENT_FEATURES]
    pd.testing.assert_frame_equal(original.loc[:cutoff, cols], later.loc[:cutoff, cols])


def test_utc_and_ohlc_geometry_mandatory():
    bad = candles()
    bad["bar_close_at"] = bad.bar_close_at.dt.tz_localize(None)
    with pytest.raises(ValueError, match="UTC"):
        build_signal_frame(bad)
    bad = candles()
    bad.loc[11, "high"] = bad.loc[11, "low"] - 1
    with pytest.raises(ValueError, match="geometry"):
        build_signal_frame(bad)


def test_gap_resets_ema_and_prevents_labels_crossing_gap():
    data = candles()
    data.loc[150:, ["bar_open_at", "bar_close_at"]] += pd.Timedelta(hours=1)
    f = build_signal_frame(data)
    assert f.loc[150, "segment_id"] != f.loc[149, "segment_id"]
    assert np.isnan(f.loc[150, "ema9"])
    assert not f.loc[150, "signal"]
    f["signal"] = False
    f.loc[149, "signal"] = True
    scores = label_signal_events(f, end_exclusive=pd.Timestamp("2025-03-01", tz="UTC"))
    assert scores.empty


def test_entry_and_exit_are_delayed_and_purged():
    f = build_signal_frame(candles())
    f["signal"] = False
    f.loc[100, "signal"] = True
    f.loc[237, "signal"] = True
    end = pd.Timestamp("2025-02-02T00:00:00Z")
    result = label_signal_events(f, end_exclusive=end)
    assert len(result) == 1
    e = result.iloc[0]
    assert e.entry_open_at == f.loc[100, "bar_close_at"] + pd.Timedelta(minutes=5)
    # i+2 bar open; i+2 closes 10 minutes after the signal for H1.
    expected = 10000 * (f.loc[102, "close"] / f.loc[102, "open"] - 1)
    assert e.gross_h1_bps == pytest.approx(expected)
    assert e.net_h1_bps == pytest.approx(expected - 2)
    expected_next = 10000 * (f.loc[101, "close"] / f.loc[100, "close"] - 1)
    assert e.next_bar_return_bps == pytest.approx(expected_next)
    assert e.next_bar_up == int(expected_next > 0)
    assert e.label_mature_at == f.loc[100, "bar_close_at"] + pd.Timedelta(minutes=35)
    assert e.label_mature_at < end


def test_cooldown_between_signals_and_clean_feature_rows():
    f = build_signal_frame(candles())
    index = np.flatnonzero(f.signal.to_numpy())
    assert all(b - a >= 8 for a, b in zip(index, index[1:]))
    assert np.isfinite(f.loc[f.signal, EVENT_FEATURES].to_numpy(dtype=float)).all()


def test_end_boundary_is_strict():
    f = build_signal_frame(candles())
    f["signal"] = False
    f.loc[100, "signal"] = True
    end = f.loc[100, "bar_close_at"] + pd.Timedelta(minutes=35)
    assert label_signal_events(f, end_exclusive=end).empty
    assert len(label_signal_events(f, end_exclusive=end + pd.Timedelta(minutes=5))) == 1
