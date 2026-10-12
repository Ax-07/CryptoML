from datetime import datetime, timezone

import pandas as pd
import pytest

from ingestion.schema import normalize_klines, validate_symbol
from ingestion.quality import inspect_quality

NOW = datetime(2026, 8, 1, 0, 20, tzinfo=timezone.utc)
BASE_MS = 1785542400000  # 2026-08-01 00:00 UTC


def test_timestamp_close_exclusive_and_open_candle_filtered(sample_row):
    samples = [sample_row(BASE_MS), sample_row(BASE_MS + 300_000), sample_row(BASE_MS + 1_200_000)]
    df = normalize_klines(samples, symbol="btcusdt", source="rest", as_of=NOW)
    assert len(df) == 2  # at 00:20, bar opening at 00:20 is not final
    assert df.bar_close_at.iloc[0] == pd.Timestamp("2026-08-01T00:05:00Z")
    before = datetime(2026, 8, 1, 0, 24, tzinfo=timezone.utc)
    df = normalize_klines(samples, symbol="BTCUSDT", source="rest", as_of=before)
    assert len(df) == 2
    assert df.bar_open_at.iloc[0].timestamp() == BASE_MS / 1000


def test_timestamp_microsecond_archive_supported(sample_row):
    item = sample_row(BASE_MS)
    item[0] *= 1000
    item[6] *= 1000
    df = normalize_klines([item], symbol="BTCUSDT", source="archive_monthly", as_of=NOW)
    assert df.bar_open_at.iloc[0] == pd.Timestamp("2026-08-01T00:00:00Z")


def test_invalid_values_and_alignment_rejected(sample_row):
    bad = sample_row(BASE_MS)
    bad[2] = "98"  # high lower than open
    with pytest.raises(ValueError, match="OHLC"):
        normalize_klines([bad], symbol="BTCUSDT", source="rest", as_of=NOW)
    bad = sample_row(BASE_MS + 10_000)
    with pytest.raises(ValueError, match="aligned"):
        normalize_klines([bad], symbol="BTCUSDT", source="rest", as_of=NOW)
    with pytest.raises(ValueError):
        validate_symbol("BTCUSDT/../../")


def test_gap_detection_does_not_fill_rows(sample_row):
    df = normalize_klines([sample_row(BASE_MS), sample_row(BASE_MS + 900_000)],
                          symbol="SOLUSDT", source="rest", as_of=NOW)
    report = inspect_quality(df)
    assert len(df) == 2
    assert report["internal_missing_bars"] == 2
    assert report["gap_count"] == 1
    assert report["status"] == "GAPS_DETECTED"


def test_as_of_requires_timezone(sample_row):
    with pytest.raises(ValueError, match="timezone"):
        normalize_klines([sample_row(BASE_MS)], symbol="BTCUSDT", source="rest",
                         as_of=datetime(2026, 8, 1))
