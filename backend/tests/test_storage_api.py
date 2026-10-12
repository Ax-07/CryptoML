import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.main import app
from ingestion.schema import normalize_klines
from ingestion.storage import CandleStore

BASE_MS = 1785542400000
NOW = datetime(2026, 8, 2, tzinfo=timezone.utc)


def _make_store(tmp_path, sample_row):
    store = CandleStore(tmp_path)
    data = normalize_klines([sample_row(BASE_MS), sample_row(BASE_MS + 300_000)],
                             symbol="BTCUSDT", source="rest", as_of=NOW)
    return store, data


def test_store_idempotence_conflicts_and_quality(tmp_path, sample_row, mock_parquet):
    store, df = _make_store(tmp_path, sample_row)
    first = store.write_batch(df)
    second = store.write_batch(df)
    assert first[0]["row_count"] == second[0]["row_count"] == 2
    report = store.quality_by_dataset_id(first[0]["dataset_id"])
    assert report["row_count"] == 2
    assert report["status"] == "OK"
    damaged = df.copy()
    damaged.loc[0, "close"] = 101.25
    with pytest.raises(ValueError, match="Conflicting"):
        store.write_batch(damaged)
    assert len(store.read_candles("BTCUSDT", NOW.replace(day=1), NOW)) == 2


def test_api_returns_seconds_and_validates_times(tmp_path, sample_row, mock_parquet, monkeypatch):
    store, df = _make_store(tmp_path, sample_row)
    store.write_batch(df)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    response = client.get("/api/v1/market/candles", params={
        "symbol": "BTCUSDT", "from": "2026-08-01T00:00:00Z", "to": "2026-08-01T00:15:00Z",
    })
    assert response.status_code == 200, response.text
    items = response.json()
    assert len(items) == 2
    assert items[0]["time"] == BASE_MS // 1000
    assert items[0]["volume"] == 10
    symbols = client.get("/api/v1/markets/symbols").json()
    assert symbols["total"] == 1
    manifest = client.get("/api/v1/datasets").json()["items"][0]
    assert client.get(f"/api/v1/datasets/{manifest['dataset_id']}/quality").json()["status"] == "OK"
    assert client.get("/api/v1/datasets/not-a-real-id/quality").status_code == 404
    bad = client.get("/api/v1/market/candles", params={
        "symbol": "BTCUSDT", "from": "2026-08-01T00:00:00", "to": "2026-08-02T00:00:00Z",
    })
    assert bad.status_code == 422
    assert "error" in bad.json()


def test_immutable_raw_zip(tmp_path):
    store = CandleStore(tmp_path)
    store.store_archive("BTCUSDT", "2026-08", b"archive-one", "0"*64)
    store.store_archive("BTCUSDT", "2026-08", b"archive-one", "0"*64)
    with pytest.raises(ValueError, match="changed"):
        store.store_archive("BTCUSDT", "2026-08", b"archive-two", "1"*64)


def test_verified_archive_preferred_over_overlapping_rest(tmp_path, sample_row, mock_parquet):
    from ingestion.schema import normalize_klines

    store = CandleStore(tmp_path)
    rest = normalize_klines([sample_row(BASE_MS)], symbol="BTCUSDT", source="rest", as_of=NOW)
    archive = normalize_klines([sample_row(BASE_MS)], symbol="BTCUSDT", source="archive_monthly", as_of=NOW)
    store.write_batch(rest)
    store.write_batch(archive)
    data = store.read_candles("BTCUSDT", NOW.replace(day=1), NOW)
    assert len(data) == 1
    assert data.iloc[0]["source"] == "archive_monthly"
