"""V18 optional public-source availability probe ONLY; no trades or data ingestion."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import pandas as pd

SYMBOL="BTCUSDT"  # Discovery sample only; never claim four-symbol coverage.
MONTHS=("2024-10","2025-06","2026-06")
KINDS={
    "spot_5m":"spot/monthly/klines/{symbol}/5m/{symbol}-5m-{month}.zip",
    "perp_5m":"futures/um/monthly/klines/{symbol}/5m/{symbol}-5m-{month}.zip",
    "perp_mark_5m":"futures/um/monthly/markPriceKlines/{symbol}/5m/{symbol}-5m-{month}.zip",
    "funding_archive":"futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip",
}
BASE="https://data.binance.vision/data/"
PROTOCOL="p4_v18_public_funding_basis_availability_no_trading_v1"


def _head_status(url:str)->dict:
    """HEAD only: check reachability without pulling a price, funding or ZIP payload."""
    try:
        with urlopen(Request(url,method="HEAD",
                             headers={"User-Agent":"CryptoML-V18-source-audit"}),timeout=12) as response:
            return {"http":response.status,"state":"HEAD_OK_NOT_INTEGRITY_VERIFIED",
                    "content_length":response.headers.get("Content-Length")}
    except HTTPError as exc:
        return {"http":exc.code,"state":"NOT_CONFIRMED"}
    except (OSError,URLError,TimeoutError) as exc:
        return {"http":None,"state":"NOT_CONFIRMED","error_type":type(exc).__name__}


def archive_samples(symbol:str=SYMBOL,months=MONTHS):
    if symbol!=SYMBOL or tuple(months)!=MONTHS:
        raise ValueError("V18 scope is exactly BTCUSDT and the three predeclared months")
    return [{"kind":kind,"symbol":symbol,"month":month,
             **_head_status(BASE+template.format(symbol=symbol,month=month))}
            for kind,template in KINDS.items() for month in months]


def _funding_sample():
    start=int(pd.Timestamp("2025-06-01T00:00:00Z").timestamp()*1000)
    end=int(pd.Timestamp("2025-06-03T00:00:00Z").timestamp()*1000)
    url="https://fapi.binance.com/fapi/v1/fundingRate?"+urlencode(
        {"symbol":SYMBOL,"startTime":start,"endTime":end,"limit":100})
    try:
        with urlopen(Request(url,headers={"User-Agent":"CryptoML-V18-source-audit"}),
                     timeout=12) as response:
            if response.status!=200:raise ValueError("Funding HTTP non-200")
            rows=json.loads(response.read(250000))
        if not isinstance(rows,list):raise ValueError("Not a funding array")
        timestamps=[]
        for r in rows:
            if r.get("symbol")!=SYMBOL:raise ValueError("Wrong symbol in funding")
            ts=int(r["fundingTime"]);rate=float(r["fundingRate"])
            if not start<=ts<=end or not -1<rate<1:raise ValueError("Bad timestamp/rate")
            timestamps.append(ts)
        if timestamps!=sorted(set(timestamps)):raise ValueError("Unsorted/duplicate funding")
        return {"state":"HISTORICAL_SAMPLE_RETRIEVED_NOT_FULL_COVERAGE" if rows else "EMPTY_SAMPLE",
                "count":len(rows),"first_ms":timestamps[0] if timestamps else None,
                "last_ms":timestamps[-1] if timestamps else None}
    except HTTPError as exc:
        return {"state":"NOT_CONFIRMED","http":exc.code}
    except (OSError,URLError,ValueError,KeyError,TypeError,TimeoutError) as exc:
        return {"state":"NOT_CONFIRMED","error_type":type(exc).__name__}


def main():
    p=argparse.ArgumentParser(description="Public Binance historical source HEAD + funding GET only")
    p.add_argument("--output-json",required=True,type=Path)
    p.add_argument("--probe-public-sources",action="store_true",
                   help="Explicit permission to issue up to 12 HEAD and one bounded GET")
    args=p.parse_args()
    if not args.probe_public_sources:
        p.error("No network by default: add --probe-public-sources explicitly")
    out=args.output_json.resolve()
    if out.exists() or not out.parent.is_dir():
        p.error("Require a new output file in an existing directory")
    samples=archive_samples()
    funding=_funding_sample()
    result={"protocol":PROTOCOL,"as_of_utc":pd.Timestamp.now(tz="UTC").isoformat(),
            "mode":"PUBLIC_METADATA_SAMPLE_ONLY","symbols_probed":[SYMBOL],
            "months_probed":list(MONTHS),"archive_head_results":samples,
            "funding_rest_sample":funding,
            "no_assumed_21month_coverage":True,"no_downloads":True,
            "no_execution":True,
            "limitations":"HEAD status neither proves valid ZIP nor immutable checksums nor 4-symbol/21-month availability; REST funding paid after decision; no tradable spot prices"}
    with out.open("x",encoding="utf-8") as file:
        json.dump(result,file,indent=2,ensure_ascii=False)
    for item in samples:print(item["kind"],item["month"],item["state"],item.get("http"),flush=True)
    print("Funding sample:",funding,flush=True)
    print("NO BACKTEST / NO SIGNALS / NO TRADING",flush=True)


if __name__=="__main__":main()
