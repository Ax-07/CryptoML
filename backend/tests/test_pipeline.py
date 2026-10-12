import hashlib
import io
import zipfile
from datetime import datetime, timezone

import httpx
import pytest

from ingestion.pipeline import ingest_month, ingest_rest
from ingestion.sources import BinanceClient
from ingestion.storage import CandleStore

BASE_MS = 1785542400000


def test_end_to_end_monthly_ingest_and_repeat(tmp_path, mock_parquet, sample_row):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as zipf:
        zipf.writestr("BTCUSDT-5m-2026-08.csv", "\n".join(
            ",".join(map(str, sample_row(BASE_MS + 300_000 * n))) for n in range(3)))
    archive = payload.getvalue()
    sha = hashlib.sha256(archive).hexdigest()

    def handler(req):
        if str(req.url).endswith(".CHECKSUM"):
            return httpx.Response(200, text=sha + "  BTCUSDT-5m-2026-08.zip")
        return httpx.Response(200, content=archive)

    store = CandleStore(tmp_path)
    with BinanceClient(transport=httpx.MockTransport(handler)) as client:
        first = ingest_month(client, store, symbol="BTCUSDT", month="2026-08",
                             as_of=datetime(2026, 9, 1, tzinfo=timezone.utc))
        second = ingest_month(client, store, symbol="BTCUSDT", month="2026-08",
                              as_of=datetime(2026, 9, 1, tzinfo=timezone.utc))
    assert first[0]["row_count"] == second[0]["row_count"] == 3
    assert first[0]["sources"] == ["archive_monthly"]
    assert store.archive_path("BTCUSDT", "2026-08").exists()
    assert store.quality_by_dataset_id(first[0]["dataset_id"])["status"] == "OK"


def test_sync_rest_adds_new_candle_without_duplicates(tmp_path, mock_parquet, sample_row):
    store = CandleStore(tmp_path)
    # The second fetch repeats the same first candle and also includes a new bar.
    def handler(req):
        rows = [sample_row(BASE_MS), sample_row(BASE_MS + 300_000)]
        cursor = int(req.url.params["startTime"])
        return httpx.Response(200, json=[r for r in rows if int(r[0]) >= cursor])

    start = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 8, 1, 0, 10, tzinfo=timezone.utc)
    with BinanceClient(transport=httpx.MockTransport(handler)) as client:
        x = ingest_rest(client, store, symbol="BTCUSDT", start=start, end=end,
                        as_of=datetime(2026, 8, 2, tzinfo=timezone.utc))
        y = ingest_rest(client, store, symbol="BTCUSDT", start=start, end=end,
                        as_of=datetime(2026, 8, 2, tzinfo=timezone.utc))
    assert x[0]["row_count"] == y[0]["row_count"] == 2


def test_reject_invalid_month_without_network(tmp_path):
    class NoNetwork:
        def download_month(self, *args, **kwargs):
            raise AssertionError("network should not be called")

    with pytest.raises(ValueError, match="YYYY-MM"):
        ingest_month(NoNetwork(), CandleStore(tmp_path), symbol="BTCUSDT", month="2026/08")
