"""V19 bounded read-only archive integrity pilot; no prices exported or strategies."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from urllib.request import Request,urlopen

import pandas as pd

SYMBOL="BTCUSDT"
MONTH="2025-06"
BASE="https://data.binance.vision/data/"
PATHS={
 "spot":"spot/monthly/klines/{s}/5m/{s}-5m-{m}.zip",
 "perp":"futures/um/monthly/klines/{s}/5m/{s}-5m-{m}.zip",
 "mark":"futures/um/monthly/markPriceKlines/{s}/5m/{s}-5m-{m}.zip",
 "funding":"futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip",
}
ZIP_LIMIT=40_000_000
UNCOMPRESSED_LIMIT=100_000_000
PROTOCOL="p4_v19_btc_june2025_archive_integrity_pilot_v1"


def fetch(url:str,limit:int)->bytes:
    with urlopen(Request(url,headers={"User-Agent":"CryptoML-V19-archive-integrity"}),timeout=20) as response:
        if response.status!=200:raise ValueError("Unexpected HTTP status")
        declared=response.headers.get("Content-Length")
        if declared is not None and int(declared)>limit:
            raise ValueError("Content-Length exceeds limit")
        blob=response.read(limit+1)
    if len(blob)>limit:raise ValueError("Download exceeds limit")
    return blob


def checksum(raw:bytes)->str:
    text=raw.decode("utf-8").strip()
    found=re.fullmatch(r"([0-9a-fA-F]{64})(?:\s+\*?[^\r\n]+)?",text)
    if not found:raise ValueError("Unrecognized CHECKSUM layout")
    return found.group(1).lower()


def classify_timestamp(raw:pd.Series)->tuple[pd.Series,str]:
    values=pd.to_numeric(raw,errors="raise")
    if values.isna().any() or (values!=values.astype("int64")).any():
        raise ValueError("Invalid timestamps")
    v=values.astype("int64")
    magnitude=int(v.abs().median())
    unit="us" if magnitude>=10**14 else "ms" if magnitude>=10**11 else "s"
    t=pd.to_datetime(v,unit=unit,utc=True,errors="raise")
    if not t.between(pd.Timestamp("2025-06-01T00:00:00Z"),
                     pd.Timestamp("2025-07-01T00:00:00Z"),inclusive="left").all():
        raise ValueError("Archive timestamps outside prescribed June 2025")
    if t.duplicated().any() or not t.is_monotonic_increasing:
        raise ValueError("Duplicate or non-monotonic timestamps")
    return t,unit


def inspect(blob:bytes,kind:str)->dict:
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        entries=archive.infolist()
        if len(entries)!=1 or entries[0].is_dir() or not entries[0].filename.endswith(".csv") or (
                "/" in entries[0].filename or "\\" in entries[0].filename):
            raise ValueError("Unsafe/nonunique ZIP CSV member")
        member=entries[0]
        if member.file_size>UNCOMPRESSED_LIMIT or (
                member.compress_size>0 and member.file_size/member.compress_size>200):
            raise ValueError("ZIP decompression limit")
        with archive.open(member) as stream:
            data=stream.read(UNCOMPRESSED_LIMIT+1)
        if len(data)>UNCOMPRESSED_LIMIT:raise ValueError("Expanded data too large")
    f=pd.read_csv(io.BytesIO(data),header=None,low_memory=False)
    if f.empty:raise ValueError("Empty archive")
    # kline data begins with integer epoch time; optional CSV header starts with string.
    if kind!="funding":
        first=str(f.iloc[0,0]).strip()
        if not first.isdigit():
            f=f.iloc[1:].reset_index(drop=True)
        if f.shape[1]<5:raise ValueError("Missing kline columns")
        times,unit=classify_timestamp(f.iloc[:,0])
        if not times.dt.minute.mod(5).eq(0).all() or not times.dt.second.eq(0).all():
            raise ValueError("Unaligned kline")
        if not times.diff().dropna().eq(pd.Timedelta(minutes=5)).all():
            raise ValueError("Missing / irregular bars")
        for i in range(1,5):
            price=pd.to_numeric(f.iloc[:,i],errors="raise")
            if price.isna().any() or not (price>0).all():raise ValueError("Invalid OHLC")
        return {"status":"CHECKSUM_AND_SCHEMA_OK","count":len(f),"timestamp_unit":unit,
                "first_open":times.iloc[0].isoformat(),"last_open":times.iloc[-1].isoformat(),
                "open_ms":[int(x.value//1_000_000) for x in times]}
    # Funding archives have varied headers/schemas; don't guess timestamp columns.
    head=pd.read_csv(io.BytesIO(data),nrows=3)
    return {"status":"SCHEMA_REVIEW_REQUIRED","count_estimate":len(f),
            "columns_observed":[str(c) for c in head.columns[:12]]}


def main():
    p=argparse.ArgumentParser(description="V19 four bounded BTC June-2025 verified archives")
    p.add_argument("--inspect-public-archives",action="store_true")
    p.add_argument("--output-json",type=Path,required=True)
    a=p.parse_args()
    if not a.inspect_public_archives:p.error("Explicit --inspect-public-archives required")
    dst=a.output_json.resolve()
    if dst.exists() or not dst.parent.is_dir():p.error("Require new output outside P0")
    rows={};errors={}
    for kind,template in PATHS.items():
        url=BASE+template.format(s=SYMBOL,m=MONTH)
        try:
            blob=fetch(url,ZIP_LIMIT)
            check=checksum(fetch(url+".CHECKSUM",4096))
            if hashlib.sha256(blob).hexdigest()!=check:raise ValueError("SHA256 CHECKSUM mismatch")
            item=inspect(blob,kind)
            item["sha256_verified"]=True
            rows[kind]=item
            print(kind,item["status"],"records",item.get("count"),flush=True)
        except (OSError,ValueError,KeyError,TypeError,zipfile.BadZipFile,UnicodeError) as exc:
            errors[kind]={"status":"NOT_VERIFIED","error":type(exc).__name__,"detail":str(exc)[:180]}
            print(kind,"NOT_VERIFIED",errors[kind]["detail"],flush=True)
    overlap={}
    if all(k in rows and rows[k]["status"]=="CHECKSUM_AND_SCHEMA_OK" for k in ("spot","perp","mark")):
        sets={k:set(rows[k].pop("open_ms")) for k in ("spot","perp","mark")}
        overlap={"common_m5_opens":len(set.intersection(*sets.values())),
                 "counts":{k:len(v) for k,v in sets.items()}}
    else:
        for v in rows.values():v.pop("open_ms",None)
    output={"protocol":PROTOCOL,"sample":"BTCUSDT June 2025","files":rows,
            "errors":errors,"spot_perp_mark_alignment":overlap,
            "no_21month_completeness_claim":True,"no_trade_or_backtest":True}
    with dst.open("x",encoding="utf-8") as o:json.dump(output,o,indent=2,ensure_ascii=False)
    print("NO BACKTEST / NO ORDER / NO FULL-HISTORY CLAIM")


if __name__=="__main__":main()
