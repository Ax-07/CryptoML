"""Offline tests for a one-shot public Binance M5 observer."""
import math
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from ingestion.schema import normalize_klines as real_normalize_klines
from mlpipeline.fresh_features import prepare_fresh_features
from mlpipeline.live_observer import collect_current_closed_m5, observe_once
from mlpipeline.trainer import DEFAULT_SYMBOLS


NOW = pd.Timestamp(datetime.now(timezone.utc))
CLOSE = NOW.floor("5min")


def normalize_at_test_clock(*args, **kwargs):
    """Fake the REST reception clock, not the production leakage guards.

    The real normalizer stamps pd.Timestamp.now(UTC). During a frozen-clock
    test, that real stamp can occur AFTER the mocked observed_at timestamp.
    Simulate consistent reception timestamps in the *test data* instead.
    """
    result = real_normalize_klines(*args, **kwargs)
    result["ingested_at"] = NOW
    return result


class FakeClient:
    def __init__(self, *, missing_latest=None, missing_internal=None):
        self.missing_latest = missing_latest
        self.missing_internal = missing_internal
        self.calls = []

    def fetch_rest(self, *, symbol, start, end, as_of):
        self.calls.append((symbol, start, end, as_of))
        opens = pd.date_range(
            pd.Timestamp(start), pd.Timestamp(end), freq="5min", inclusive="left"
        )
        rows = []
        for i, t in enumerate(opens):
            if symbol == self.missing_latest and t == opens[-1]:
                continue
            if symbol == self.missing_internal and t == opens[-15]:
                continue
            # Non-flat, deterministic candles: RSI needs both rising and
            # falling closes; otherwise 0/0 yields NaN by design.
            close = 100.0 + 0.01 * i + 0.4 * math.sin(i / 5.0)
            opening = close - 0.03 * math.cos(i / 3.0)
            high = max(opening, close) + 0.1
            low = min(opening, close) - 0.1
            volume = 20.0 + i % 17
            taker_buy_volume = 0.5 * volume
            ms = int(t.timestamp() * 1000)
            rows.append([
                ms, str(opening), str(high), str(low), str(close),
                str(volume), ms + 299999, str(volume * close),
                200 + i % 23, str(taker_buy_volume),
                str(taker_buy_volume * close), "0"
            ])
        return rows


def test_public_rest_only_and_last_closed_window():
    cli = FakeClient()
    data = collect_current_closed_m5(cli, started_at=NOW)
    assert set(data.symbol) == set(DEFAULT_SYMBOLS)
    assert all(len(data[data.symbol.eq(sym)]) == 300 for sym in DEFAULT_SYMBOLS)
    assert data.bar_close_at.max() == CLOSE
    assert data.bar_open_at.max() == CLOSE - pd.Timedelta(minutes=5)
    assert all(end == CLOSE.to_pydatetime() for _, _, end, _ in cli.calls)
    assert all(end <= as_of for _, _, end, as_of in cli.calls)


def test_missing_newest_candle_fails_closed():
    with pytest.raises(ValueError, match="Missing latest"):
        collect_current_closed_m5(
            FakeClient(missing_latest="ETHUSDT"), started_at=NOW
        )


def test_check_only_does_not_load_model_or_write_any_signals():
    cli = FakeClient()
    with patch("mlpipeline.live_observer.datetime") as clock, \
            patch("mlpipeline.live_observer.normalize_klines",
                  side_effect=normalize_at_test_clock), \
            patch("mlpipeline.live_observer.predict_snapshot") as pred:
        clock.now.return_value = NOW.to_pydatetime()
        result = observe_once(
            cli, model_dir=None, log_dir=None, check_only=True
        )
    assert result["status"] == "PUBLIC_M5_CHECK_ONLY"
    assert result["feature_rows"] == 5
    pred.assert_not_called()


def test_missing_internal_candle_fails_feature_warmup():
    cli = FakeClient(missing_internal="BTCUSDT")
    with patch("mlpipeline.live_observer.normalize_klines",
               side_effect=normalize_at_test_clock):
        data = collect_current_closed_m5(cli, started_at=NOW)
    with pytest.raises(ValueError, match="Insufficient contiguous"):
        prepare_fresh_features(data, observed_at=NOW)


def test_future_ingestion_timestamp_remains_strictly_rejected():
    with patch("mlpipeline.live_observer.normalize_klines",
               side_effect=normalize_at_test_clock):
        data = collect_current_closed_m5(FakeClient(), started_at=NOW)
    # If even one candle claims to have arrived AFTER observed_at, fail closed.
    data.loc[data.index[0], "ingested_at"] = NOW + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="Future candle or future ingestion"):
        prepare_fresh_features(data, observed_at=NOW)
