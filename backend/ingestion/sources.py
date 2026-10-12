"""Official public Binance monthly archives and Futures REST client."""
from __future__ import annotations

import csv
import hashlib
import io
import time
import zipfile
from datetime import datetime, timezone

import httpx

from ingestion.schema import validate_symbol


class BinanceClient:
    def __init__(
        self,
        archive_base: str = "https://data.binance.vision",
        rest_base: str = "https://fapi.binance.com",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.archive_base = archive_base.rstrip("/")
        self.rest_base = rest_base.rstrip("/")
        self.http = httpx.Client(timeout=30.0, follow_redirects=True, transport=transport,
                                 headers={"User-Agent": "CryptoML-P0/0.1"})

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> "BinanceClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _get(self, url: str, *, params: dict | None = None) -> httpx.Response:
        for attempt in range(4):
            try:
                response = self.http.get(url, params=params)
                if response.status_code == 404:
                    raise FileNotFoundError(f"Not found on Binance: {response.url}")
                if response.status_code in (418, 429, 500, 502, 503, 504):
                    if attempt == 3:
                        response.raise_for_status()
                    delay = min(1.5 * (2 ** attempt), 12)
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = min(max(float(retry_after), delay), 60)
                        except ValueError:
                            pass
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == 3:
                    raise
                time.sleep(1.5 * (2 ** attempt))
        raise RuntimeError("Unreachable retry loop")

    def download_month(self, *, symbol: str, month: str) -> tuple[bytes, str, list[list]]:
        symbol = validate_symbol(symbol)
        name = f"{symbol}-5m-{month}.zip"
        base = f"{self.archive_base}/data/futures/um/monthly/klines/{symbol}/5m/{name}"
        binary = self._get(base).content
        checksum_text = self._get(base + ".CHECKSUM").text.strip()
        expected = checksum_text.split()[0].lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"Invalid archive checksum for {name}")
        actual = hashlib.sha256(binary).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {name}: expected {expected}, got {actual}")
        with zipfile.ZipFile(io.BytesIO(binary)) as archive:
            members = [m for m in archive.infolist() if m.filename.endswith(".csv") and not m.is_dir()]
            if len(members) != 1:
                raise ValueError(f"Expected exactly one CSV inside {name}")
            if members[0].file_size > 100_000_000:
                raise ValueError("CSV uncompressed size exceeds 100 MB safety limit")
            with archive.open(members[0]) as stream:
                reader = csv.reader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline=""))
                rows = list(reader)
        if rows and not rows[0][0].strip().lstrip("-").isdigit():
            rows = rows[1:]  # Some files include a header row
        return binary, checksum_text, rows

    def fetch_rest(
        self, *, symbol: str, start: datetime, end: datetime, as_of: datetime
    ) -> list[list]:
        """Bounded pagination; end is an exclusive 5m boundary in UTC."""
        symbol = validate_symbol(symbol)
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(min(end, as_of).timestamp() * 1000)
        if end_ms <= start_ms:
            return []
        result: list[list] = []
        cursor = start_ms
        while cursor < end_ms:
            payload = self._get(f"{self.rest_base}/fapi/v1/klines", params={
                "symbol": symbol, "interval": "5m", "startTime": cursor,
                "endTime": end_ms - 1, "limit": 1000,
            }).json()
            if not isinstance(payload, list):
                raise ValueError("Unexpected Binance REST response")
            if not payload:
                break
            result.extend(payload)
            last_open = int(payload[-1][0])
            # Enforce strictly forward REST pagination to avoid endless loops.
            if last_open < cursor:
                raise ValueError("Binance API returned out-of-order candles")
            cursor = last_open + 300_000
        return result
