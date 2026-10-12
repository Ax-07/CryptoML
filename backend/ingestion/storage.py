"""Parquet monthly partitions, manifests, local read-only query helpers."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ingestion.quality import inspect_quality
from ingestion.schema import COLUMNS, VALUE_COLUMNS, validate_symbol


def stamp(value: datetime | pd.Timestamp) -> str:
    return pd.Timestamp(value).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmpname = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(obj, stream, ensure_ascii=False, indent=2)
        os.replace(tmpname, path)
    finally:
        Path(tmpname).unlink(missing_ok=True)


class CandleStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def partition_dir(self, symbol: str, year: int, month: int) -> Path:
        symbol = validate_symbol(symbol)
        return (self.root / "processed" / "ohlcv" / "exchange=binance" /
                "market=usdt_perpetual" / "timeframe=5m" / f"symbol={symbol}" /
                f"year={year:04d}" / f"month={month:02d}")

    def archive_path(self, symbol: str, month: str) -> Path:
        return self.root / "raw" / "binance" / "usdt_perpetual" / "5m" / validate_symbol(symbol) / f"{month}.zip"

    def store_archive(self, symbol: str, month: str, binary: bytes, checksum_text: str) -> Path:
        """Raw ZIP immutable: corrected exchange archives require explicit operator review."""
        path = self.archive_path(symbol, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        sha = hashlib.sha256(binary).hexdigest()
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError(f"Archive {path} changed on Binance; manual review required")
        else:
            fd, tmpname = tempfile.mkstemp(dir=path.parent, suffix=".zip.tmp")
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(binary)
                os.replace(tmpname, path)
            finally:
                Path(tmpname).unlink(missing_ok=True)
        sidecar = path.with_suffix(".zip.CHECKSUM")
        if not sidecar.exists():
            sidecar.write_text(checksum_text.strip() + "\n", encoding="utf-8")
        return path

    def write_batch(self, df: pd.DataFrame) -> list[dict]:
        if df.empty:
            return []
        if df["symbol"].nunique() != 1:
            raise ValueError("Write only one symbol per batch")
        output = []
        for (year, month), batch in df.groupby([df.bar_open_at.dt.year, df.bar_open_at.dt.month]):
            path = self.partition_dir(str(batch.symbol.iloc[0]), int(year), int(month))
            path.mkdir(parents=True, exist_ok=True)
            file = path / "part-000.parquet"
            batch = batch[COLUMNS].copy()
            if file.exists():
                existing = pd.read_parquet(file)
                both = pd.concat([existing, batch], ignore_index=True)
                for _, dup in both.groupby("bar_open_at", sort=False):
                    if len(dup) <= 1:
                        continue
                    base = dup.iloc[0]
                    for _, row in dup.iloc[1:].iterrows():
                        for field in VALUE_COLUMNS:
                            if not np.isclose(float(row[field]), float(base[field]), rtol=1e-10, atol=1e-10):
                                raise ValueError(
                                    f"Conflicting candle at {row['bar_open_at']} ({field}); manual review required"
                                )
                # When two sources agree, prefer the checksum-verified monthly archive.
                priority = {"archive_monthly": 0, "archive_daily": 1, "rest": 2}
                both["_priority"] = both["source"].map(priority).fillna(99)
                both = both.sort_values(["bar_open_at", "_priority"], kind="stable")
                combined = both.drop_duplicates(subset=["bar_open_at"], keep="first").drop(columns="_priority")
            else:
                combined = batch
            combined = combined.sort_values("bar_open_at").reset_index(drop=True)
            if combined.bar_open_at.duplicated().any():
                raise ValueError("Duplicate bars remain after reconciliation")
            # Preserve the existing partition on a write failure.
            fd, temporary = tempfile.mkstemp(dir=path, suffix=".parquet.tmp")
            os.close(fd)
            try:
                combined.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
                os.replace(temporary, file)
            finally:
                Path(temporary).unlink(missing_ok=True)
            quality = inspect_quality(combined)
            dataset_id = f"ohlcv_binance_usdt_perpetual_5m_{batch.symbol.iloc[0]}_{year:04d}-{month:02d}"
            manifest = {
                "dataset_id": dataset_id,
                "symbol": str(batch.symbol.iloc[0]),
                "exchange": "binance",
                "market": "usdt_perpetual",
                "timeframe": "5m",
                "period": f"{year:04d}-{month:02d}",
                "schema_version": "ohlcv_v1",
                "updated_at": stamp(datetime.now(timezone.utc)),
                "row_count": int(len(combined)),
                "parquet_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                "sources": sorted(combined.source.unique().tolist()),
                "quality_status": quality["status"],
            }
            atomic_json(path / "manifest.json", manifest)
            atomic_json(path / "quality_report.json", quality)
            output.append(manifest)
        return output

    def _paths(self, symbol: str | None = None) -> list[Path]:
        if symbol is None:
            return sorted((self.root / "processed" / "ohlcv").glob(
                "exchange=binance/market=usdt_perpetual/timeframe=5m/symbol=*/year=*/month=*/part-000.parquet"))
        symbol = validate_symbol(symbol)
        base = (self.root / "processed" / "ohlcv" / "exchange=binance" /
                "market=usdt_perpetual" / "timeframe=5m" / f"symbol={symbol}")
        return sorted(base.glob("year=*/month=*/part-000.parquet"))

    def list_manifests(self) -> list[dict]:
        manifests = []
        for path in self._paths():
            f = path.parent / "manifest.json"
            if f.exists():
                manifests.append(json.loads(f.read_text(encoding="utf-8")))
        return sorted(manifests, key=lambda x: (x["symbol"], x["period"]))

    def quality_by_dataset_id(self, dataset_id: str) -> dict | None:
        # Use manifest lookup, never construct a local path from user input.
        for manifest in self.list_manifests():
            if manifest["dataset_id"] == dataset_id:
                period = manifest["period"]
                folder = self.partition_dir(manifest["symbol"], int(period[:4]), int(period[5:7]))
                f = folder / "quality_report.json"
                if f.exists():
                    return {"dataset_id": dataset_id, **json.loads(f.read_text(encoding="utf-8"))}
        return None

    def read_candles(self, symbol: str, start: datetime, end: datetime, limit: int = 500) -> pd.DataFrame:
        symbol = validate_symbol(symbol)
        start = pd.Timestamp(start).tz_convert("UTC")
        end = pd.Timestamp(end).tz_convert("UTC")
        # Only open monthly files intersecting the requested window.
        candidate = []
        for path in self._paths(symbol):
            year = int(path.parent.parent.name.split("=")[1])
            month = int(path.parent.name.split("=")[1])
            lower = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
            upper = lower + pd.offsets.MonthBegin(1)
            if lower < end and upper > start:
                candidate.append(path)
        if not candidate:
            return pd.DataFrame(columns=COLUMNS)
        chunks = []
        for path in candidate:
            chunk = pd.read_parquet(path)
            mask = (chunk.bar_open_at >= start) & (chunk.bar_open_at < end)
            chunks.append(chunk.loc[mask])
        return pd.concat(chunks, ignore_index=True).sort_values("bar_open_at").head(limit)
