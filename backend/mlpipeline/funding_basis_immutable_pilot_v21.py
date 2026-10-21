"""V21 immutable BTCUSDT June 2025 source pilot. No backtest or orders."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from ingestion.storage import CandleStore
from mlpipeline.funding_basis_archive_integrity_v19 import (
    BASE, PATHS, SYMBOL, MONTH, ZIP_LIMIT, checksum, fetch, classify_timestamp,
)
from mlpipeline.funding_archive_schema_v20 import verified_csv, funding_schema

N=8640
START=pd.Timestamp("2025-06-01T00:00:00Z")
END=pd.Timestamp("2025-07-01T00:00:00Z")
M5=pd.Timedelta(minutes=5)
PROTOCOL="p4_v21_immutable_btc_june2025_source_pilot_v1"
CHECK_FIELDS=("spot","perp","mark","funding")


def _reference(v19_path:Path,v20_path:Path):
    old19=v19_path.read_bytes();old20=v20_path.read_bytes()
    v19=json.loads(old19);v20=json.loads(old20)
    if (v19.get("protocol")!="p4_v19_btc_june2025_archive_integrity_pilot_v1" or
        v20.get("protocol")!="p4_v20_funding_archive_schema_only_v1" or
        v20.get("v19_report_sha256")!=hashlib.sha256(old19).hexdigest() or
        v20.get("funding",{}).get("status")!="FUNDING_SCHEMA_AND_CHECKSUM_OK" or
        v20.get("funding",{}).get("count")!=90 or
        v19.get("spot_perp_mark_alignment",{}).get("common_m5_opens")!=N):
        raise ValueError("V19/V20 lineage or observed shape mismatch")
    for kind in CHECK_FIELDS:
        item=v19.get("files",{}).get(kind,{})
        if item.get("sha256_verified") is not True:
            raise ValueError("V19 source not checksum-verified: "+kind)
    for kind in ("spot","perp","mark"):
        if v19["files"][kind].get("status")!="CHECKSUM_AND_SCHEMA_OK" or v19["files"][kind].get("count")!=N:
            raise ValueError("V19 kline source not verified")
    if v20.get("funding_zip_sha256") is None:
        raise ValueError("V20 funding digest missing")
    return {"v19":hashlib.sha256(old19).hexdigest(),
            "v20":hashlib.sha256(old20).hexdigest(),
            "v20_funding_zip_sha256":v20["funding_zip_sha256"]}


def _kline(blob:bytes,kind:str):
    data=verified_csv(blob)
    frame=pd.read_csv(io.BytesIO(data),header=None,low_memory=False)
    if frame.empty:raise ValueError("Empty "+kind)
    if not str(frame.iloc[0,0]).strip().isdigit():
        frame=frame.iloc[1:].reset_index(drop=True)
    if len(frame)!=N or frame.shape[1]<6:
        raise ValueError(kind+" unexpected M5 size/schema")
    ts,unit=classify_timestamp(frame.iloc[:,0])
    if (ts.iloc[0]!=START or ts.iloc[-1]!=END-M5 or
        not ts.diff().iloc[1:].eq(M5).all() or
        not ts.dt.second.eq(0).all() or not ts.dt.minute.mod(5).eq(0).all()):
        raise ValueError(kind+" timestamps incomplete")
    values={}
    # Binance markPriceKlines column 5 is an ignored placeholder, not
    # traded volume. Keep real volume ONLY for spot and perp klines.
    labels=("open","high","low","close") if kind=="mark" else (
        "open","high","low","close","volume")
    for i,label in enumerate(labels,start=1):
        x=pd.to_numeric(frame.iloc[:,i],errors="raise").to_numpy(dtype=float)
        if not np.isfinite(x).all() or (x<0).any() or (label!="volume" and (x<=0).any()):
            raise ValueError(kind+" bad "+label)
        values[label]=x
    if ((values["high"]<np.maximum.reduce([values["open"],values["close"],values["low"]])).any()
        or (values["low"]>np.minimum.reduce([values["open"],values["close"],values["high"]])).any()):
        raise ValueError(kind+" bad OHLC")
    # Close-time, when supplied in Binance kline CSV, must be the inclusive last
    # timestamp of the period. Headerless archives use column 6 for close time.
    if frame.shape[1]>=7:
        end_times=pd.to_numeric(frame.iloc[:,6],errors="raise").astype("int64").to_numpy()
        step={"s":300,"ms":300_000,"us":300_000_000}[unit]
        opens=pd.to_numeric(frame.iloc[:,0],errors="raise").astype("int64").to_numpy()
        if not np.array_equal(end_times,opens+step-1):
            raise ValueError(kind+" invalid inclusive close_time")
    return ts,unit,values


def _funding(blob:bytes):
    data=verified_csv(blob)
    schema=funding_schema(data)
    if schema.get("status")!="FUNDING_SCHEMA_AND_CHECKSUM_OK" or schema.get("count")!=90:
        raise ValueError("Funding schema/size mismatch")
    frame=pd.read_csv(io.BytesIO(data),dtype=str)
    t,_=classify_timestamp(frame["calc_time"])
    rate=pd.to_numeric(frame["last_funding_rate"],errors="raise").astype(float)
    interval=pd.to_numeric(frame["funding_interval_hours"],errors="raise").astype(int)
    out=pd.DataFrame({"funding_at":t,"funding_rate_fraction":rate,
                      "funding_interval_hours":interval})
    return out


def _check_p0(store:CandleStore,ts:pd.Series,prices:dict):
    path=store.partition_dir(SYMBOL,2025,6)
    manifest=json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    parquet=path/"part-000.parquet"
    digest=hashlib.sha256(parquet.read_bytes()).hexdigest()
    if (manifest.get("quality_status")!="OK" or manifest.get("parquet_sha256")!=digest
        or manifest.get("row_count")!=N):
        raise ValueError("P0 immutable source mismatch")
    p0=pd.read_parquet(parquet,columns=["bar_open_at","open","high","low","close"])
    opens=pd.to_datetime(p0.bar_open_at,utc=True)
    if len(p0)!=N or not opens.reset_index(drop=True).equals(ts.reset_index(drop=True)):
        raise ValueError("P0 and Binance perp time disagreement")
    for field in ("open","high","low","close"):
        if not np.allclose(p0[field].to_numpy(dtype=float),prices[field],atol=1e-8,rtol=1e-10):
            raise ValueError("P0 and Binance perp OHLC conflict: "+field)
    return digest


def run(v19:Path,v20:Path,root:Path,destination:Path):
    # Destination safety is a preflight invariant: refuse overwrite or P0
    # access BEFORE reading lineage files or issuing any network requests.
    if destination.exists() or not destination.parent.is_dir() or (
        destination==root or destination.is_relative_to(root) or root.is_relative_to(destination)):
        raise ValueError("New destination outside protected P0 required")
    lineage=_reference(v19,v20)
    with tempfile.TemporaryDirectory(prefix="v21-staging-",dir=destination.parent) as stagedir:
        stage=Path(stagedir)
        raw_dir=stage/"raw"
        raw_dir.mkdir()
        decoded={};archives={}
        for kind in CHECK_FIELDS:
            name=PATHS[kind].format(s=SYMBOL,m=MONTH)
            url=BASE+name
            blob=fetch(url,ZIP_LIMIT)
            checkblob=fetch(url+".CHECKSUM",4096)
            digest=hashlib.sha256(blob).hexdigest()
            if checksum(checkblob)!=digest:raise ValueError("Archive checksum mismatch: "+kind)
            if kind=="funding" and digest!=lineage["v20_funding_zip_sha256"]:
                raise ValueError("Funding ZIP changed since V20")
            (raw_dir/f"{kind}.zip").write_bytes(blob)
            (raw_dir/f"{kind}.zip.CHECKSUM").write_bytes(checkblob)
            archives[kind]={"zip_sha256":digest,"checksum_sha256":hashlib.sha256(checkblob).hexdigest()}
            decoded[kind]=_funding(blob) if kind=="funding" else _kline(blob,kind)
            print(kind,"verified",flush=True)
        a,b,c=(decoded[k] for k in ("spot","perp","mark"))
        if not a[0].reset_index(drop=True).equals(b[0].reset_index(drop=True)) or not a[0].reset_index(drop=True).equals(c[0].reset_index(drop=True)):
            raise ValueError("Spot/perp/mark M5 timeline divergence")
        p0sha=_check_p0(CandleStore(root),b[0],b[2])
        markets=pd.DataFrame({"bar_open_at":a[0],"bar_close_at":a[0]+M5})
        for kind,(ts,unit,v) in zip(("spot","perp","mark"),(a,b,c)):
            for key,arr in v.items():markets[f"{kind}_{key}"]=arr
            archives[kind]["timestamp_unit"]=unit
        markets_file=stage/"markets_5m.parquet"
        funding_file=stage/"funding_events.parquet"
        markets.to_parquet(markets_file,index=False,engine="pyarrow",compression="zstd")
        decoded["funding"].to_parquet(funding_file,index=False,engine="pyarrow",compression="zstd")
        manifest={"protocol":PROTOCOL,"symbol":SYMBOL,"month":MONTH,
                  "lineage":lineage,"p0_perp_parquet_sha256":p0sha,
                  "archives":archives,"market_bars":len(markets),
                  "funding_events":len(decoded["funding"]),
                  "markets_sha256":hashlib.sha256(markets_file.read_bytes()).hexdigest(),
                  "funding_sha256":hashlib.sha256(funding_file.read_bytes()).hexdigest(),
                  "status":"SOURCE_PILOT_ONLY_NO_PNL_NO_FILLS_NO_21MONTH_COVERAGE"}
        (stage/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        # Move staging atomically within the same parent filesystem; never overwrite.
        stage.rename(destination)
        return manifest


def main():
    parser=argparse.ArgumentParser(description="V21 4-archive immutable pilot, NO PNL")
    parser.add_argument("--ingest-verified-pilot",action="store_true")
    for name in ("reference-v19-json","reference-v20-json","data-root","destination"):
        parser.add_argument("--"+name,required=True,type=Path)
    args=parser.parse_args()
    if not args.ingest_verified_pilot:parser.error("Explicit --ingest-verified-pilot required")
    result=run(args.reference_v19_json.resolve(),args.reference_v20_json.resolve(),
               args.data_root.resolve(),args.destination.resolve())
    print("V21 CHECKED:",result["market_bars"],"M5",result["funding_events"],"funding")
    print("NO BACKTEST / NO TRADING / NO FULL-HISTORY CLAIM")


if __name__=="__main__":main()
