import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone

import httpx
import pytest

from ingestion.sources import BinanceClient

BASE_MS = 1785542400000


def zipcsv(data: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as f:
        f.writestr("BTCUSDT-5m-2026-08.csv", data)
    return buf.getvalue()


def test_monthly_zip_verified_and_header_removed(sample_row):
    content = ",".join(str(x) for x in sample_row(BASE_MS)) + "\n"
    binary = zipcsv("open_time,open,high,low,close,volume,close_time,quote,trades,takerbase,takerquote,ignore\n" + content)
    checksum = hashlib.sha256(binary).hexdigest()

    def responder(request):
        if request.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, text=checksum + "  BTCUSDT-5m-2026-08.zip\n")
        return httpx.Response(200, content=binary)

    with BinanceClient(transport=httpx.MockTransport(responder)) as client:
        downloaded, expected, rows = client.download_month(symbol="BTCUSDT", month="2026-08")
    assert downloaded == binary
    assert expected.startswith(checksum)
    assert len(rows) == 1
    assert rows[0][0] == str(BASE_MS)


def test_corrupted_checksum_fails(sample_row):
    binary = zipcsv(",".join(str(x) for x in sample_row(BASE_MS)))

    def responder(request):
        if request.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, text="0" * 64 + "  file.zip")
        return httpx.Response(200, content=binary)

    with BinanceClient(transport=httpx.MockTransport(responder)) as client:
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            client.download_month(symbol="BTCUSDT", month="2026-08")


def test_rest_pagination_and_exclusive_end(sample_row):
    requests = []
    whole = [sample_row(BASE_MS + i * 300_000) for i in range(1001)]

    def responder(request):
        cursor = int(request.url.params["startTime"])
        requests.append(cursor)
        chunk = [r for r in whole if cursor <= int(r[0]) < BASE_MS + 1001 * 300_000]
        return httpx.Response(200, json=chunk[:1000])

    with BinanceClient(transport=httpx.MockTransport(responder)) as client:
        rows = client.fetch_rest(symbol="BTCUSDT",
            start=datetime.fromtimestamp(BASE_MS / 1000, timezone.utc),
            end=datetime.fromtimestamp((BASE_MS + 1001 * 300_000) / 1000, timezone.utc),
            as_of=datetime.fromtimestamp((BASE_MS + 1002 * 300_000) / 1000, timezone.utc))
    assert len(rows) == 1001
    assert len(requests) == 2
    assert requests[1] == BASE_MS + 1000 * 300_000
