"""CLI for reproducible, public-only Binance Futures M5 ingestion."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from app.core.config import get_settings
from ingestion.pipeline import ingest_month, ingest_rest
from ingestion.schema import require_utc
from ingestion.sources import BinanceClient
from ingestion.storage import CandleStore

PILOT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT")


def parse_date(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return require_utc(dt)


def months_back(n: int, now: datetime) -> list[str]:
    year, month = now.year, now.month
    result = []
    for _ in range(n):
        month -= 1
        if month == 0:
            month, year = 12, year - 1
        result.append(f"{year:04d}-{month:02d}")
    return list(reversed(result))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CryptoML: public Binance USD-M M5 ingestion")
    sub = parser.add_subparsers(dest="command", required=True)
    monthly = sub.add_parser("import-month", help="Verified monthly official ZIP -> Parquet")
    monthly.add_argument("--symbol", required=True)
    monthly.add_argument("--month", required=True, help="YYYY-MM")
    pilot = sub.add_parser("import-pilot", help="Previous N complete months, five pilot pairs")
    pilot.add_argument("--months", type=int, default=2)
    rest = sub.add_parser("sync-rest", help="REST completion, bounded to 31 days")
    rest.add_argument("--symbol", required=True)
    rest.add_argument("--start", required=True, help="ISO-8601 with timezone")
    rest.add_argument("--end", required=True, help="ISO-8601 with timezone, exclusive")
    sub.add_parser("list-data", help="List local Parquet datasets and quality summaries")
    args = parser.parse_args(argv)
    settings = get_settings()
    store = CandleStore(settings.data_root)
    now = datetime.now(timezone.utc)
    try:
        if args.command == "list-data":
            print(json.dumps(store.list_manifests(), ensure_ascii=False, indent=2))
            return 0
        with BinanceClient(settings.binance_data_base_url, settings.binance_fapi_base_url) as client:
            if args.command == "import-month":
                results = ingest_month(client, store, symbol=args.symbol, month=args.month, as_of=now)
            elif args.command == "sync-rest":
                results = ingest_rest(client, store, symbol=args.symbol, start=parse_date(args.start),
                                      end=parse_date(args.end), as_of=now)
            else:
                if not 1 <= args.months <= 36:
                    raise ValueError("--months must be between 1 and 36")
                results = []
                errors = []
                for symbol in PILOT_SYMBOLS:
                    for month in months_back(args.months, now):
                        try:
                            print(f"Importing {symbol} {month}...", file=sys.stderr)
                            results.extend(ingest_month(client, store, symbol=symbol, month=month, as_of=now))
                        except FileNotFoundError as exc:
                            errors.append(str(exc))
                            print(f"SKIPPED: {exc}", file=sys.stderr)
                if errors:
                    print(f"Missing archives: {len(errors)} (see stderr)", file=sys.stderr)
                if not results:
                    raise RuntimeError("No archives could be imported")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
