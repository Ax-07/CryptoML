"""V23: bounded, single-symbol/month immutable Binance source ingestion. No trading."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from ingestion.storage import CandleStore
from mlpipeline.funding_basis_archive_integrity_v19 import BASE, PATHS, ZIP_LIMIT, fetch, checksum
from mlpipeline.funding_archive_schema_v20 import verified_csv, funding_schema
from mlpipeline.funding_basis_archive_coverage_v22 import PROTOCOL as V22_PROTOCOL, KINDS
from mlpipeline.pullback_long_history_v3 import SYMBOLS, EXPECTED_MONTHS

PROTOCOL="p4_v23_immutable_monthly_source_v1"
M5=pd.Timedelta(minutes=5)


def epoch(raw,begin,end):
    value=pd.to_numeric(raw,errors="raise")
    if value.isna().any() or not (value==value.astype("int64")).all():
        raise ValueError("Invalid epoch")
    v=value.astype("int64")
    size=v.abs().astype(str).str.len()
    if size.nunique()!=1:raise ValueError("Mixed timestamp units")
    digits=int(size.iloc[0])
    units={10:"s",13:"ms",16:"us"}
    if digits not in units:raise ValueError("Unknown timestamp unit")
    unit=units[digits]
    ts=pd.to_datetime(v,unit=unit,utc=True,errors="raise")
    if not ts.between(begin,end,inclusive="left").all() or ts.duplicated().any() or not ts.is_monotonic_increasing:
        raise ValueError("Timestamp outside month / duplicates / unordered")
    return ts,unit


def parse_kline(data,kind,begin,end):
    f=pd.read_csv(io.BytesIO(data),header=None,low_memory=False)
    if f.empty:raise ValueError("Empty kline")
    if not str(f.iloc[0,0]).strip().isdigit():
        f=f.iloc[1:].reset_index(drop=True)
    count=int((end-begin)/M5)
    if len(f)!=count or f.shape[1]<7:raise ValueError("Incorrect M5 count/shape")
    t,unit=epoch(f.iloc[:,0],begin,end)
    if t.iloc[0]!=begin or t.iloc[-1]!=end-M5 or not t.diff().iloc[1:].eq(M5).all():
        raise ValueError("M5 timeline gap/misalignment")
    names=("open","high","low","close") if kind=="mark" else ("open","high","low","close","volume")
    values={}
    for index,name in enumerate(names,start=1):
        a=pd.to_numeric(f.iloc[:,index],errors="raise").to_numpy(dtype=float)
        if not np.isfinite(a).all() or (a<0).any() or (name!="volume" and (a<=0).any()):
            raise ValueError("Invalid kline "+name)
        values[name]=a
    if ((values["high"]<np.maximum.reduce([values["open"],values["close"],values["low"]])).any() or
        (values["low"]>np.minimum.reduce([values["open"],values["close"],values["high"]])).any()):
        raise ValueError("OHLC geometry invalid")
    step={"s":300,"ms":300000,"us":300000000}[unit]
    opens=pd.to_numeric(f.iloc[:,0],errors="raise").to_numpy(dtype=np.int64)
    closes=pd.to_numeric(f.iloc[:,6],errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(opens+step-1,closes):
        raise ValueError("Inclusive close timestamp mismatch")
    return t,unit,values


def parse_funding(data,begin,end):
    schema=funding_schema_for_month(data,begin,end)
    if schema["status"]!="FUNDING_SCHEMA_AND_CHECKSUM_OK":
        raise ValueError("Unknown funding CSV schema")
    frame=pd.read_csv(io.BytesIO(data),dtype=str)
    ts,_=epoch(frame["calc_time"],begin,end)
    rate=pd.to_numeric(frame["last_funding_rate"],errors="raise").to_numpy(dtype=float)
    hours=pd.to_numeric(frame["funding_interval_hours"],errors="raise").to_numpy(dtype=int)
    return pd.DataFrame({"funding_at":ts,"funding_rate_fraction":rate,
                         "funding_interval_hours":hours})


def funding_schema_for_month(data,begin,end):
    # V20's funding_schema hard-codes June 2025 through its original timestamp
    # helper. V23 retains the exact known schema but validates generic month times.
    frame=pd.read_csv(io.BytesIO(data),dtype=str,keep_default_na=False)
    expected={"calc_time","last_funding_rate","funding_interval_hours"}
    if not expected.issubset(frame.columns):
        return {"status":"SCHEMA_REVIEW_REQUIRED","columns":list(frame.columns)}
    if frame.empty:raise ValueError("Empty funding")
    raw=frame["calc_time"].str.strip()
    if not raw.str.fullmatch(r"\d{10}|\d{13}|\d{16}").all():raise ValueError("Bad funding epoch")
    epoch(raw,begin,end)
    rates=frame["last_funding_rate"].str.strip()
    if not rates.str.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?").all():
        raise ValueError("Funding rate invalid")
    numeric=pd.to_numeric(rates,errors="raise").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (np.abs(numeric)>0.1).any():
        raise ValueError("Funding rate invalid")
    hours=frame["funding_interval_hours"].str.strip()
    if not hours.str.fullmatch(r"\d+").all() or not pd.to_numeric(hours).between(1,24).all():
        raise ValueError("Funding interval invalid")
    return {"status":"FUNDING_SCHEMA_AND_CHECKSUM_OK","count":len(frame)}


def validate_coverage(json_path,csv_path):
    obj=json.loads(json_path.read_bytes())
    csv=pd.read_csv(csv_path,dtype={"symbol":str,"month":str,"kind":str,"status":str})
    if (obj.get("protocol")!=V22_PROTOCOL or obj.get("mode")!="full" or
        obj.get("expected")!=336 or obj.get("head_200")!=336 or len(csv)!=336):
        raise ValueError("Full V22 coverage not confirmed")
    expected={(s,m,k) for s in SYMBOLS for m in EXPECTED_MONTHS for k in KINDS}
    observed=set(zip(csv.symbol,csv.month,csv.kind))
    if len(observed)!=336 or observed!=expected or not csv.status.eq("HEAD_OK_UNVERIFIED").all():
        raise ValueError("V22 CSV coverage mismatch")
    return {"json_sha256":hashlib.sha256(json_path.read_bytes()).hexdigest(),
            "csv_sha256":hashlib.sha256(csv_path.read_bytes()).hexdigest()}


def compare_p0(root,symbol,month,t,perp):
    year,mo=map(int,month.split("-"))
    directory=CandleStore(root).partition_dir(symbol,year,mo)
    m=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    parquet=directory/"part-000.parquet"
    sha=hashlib.sha256(parquet.read_bytes()).hexdigest()
    if m.get("parquet_sha256")!=sha or m.get("quality_status")!="OK" or m.get("row_count")!=len(t):
        raise ValueError("P0 integrity mismatch")
    frame=pd.read_parquet(parquet,columns=["bar_open_at","open","high","low","close"])
    dates=pd.to_datetime(frame.bar_open_at,utc=True).reset_index(drop=True)
    if not dates.equals(t.reset_index(drop=True)):
        raise ValueError("P0 timeline mismatch")
    for column in ("open","high","low","close"):
        if not np.allclose(frame[column].to_numpy(dtype=float),perp[column],rtol=1e-10,atol=1e-8):
            raise ValueError("P0 OHLC mismatch: "+column)
    return sha


def ingest(symbol,month,root,output_root,coverage_json,coverage_csv):
    if symbol not in SYMBOLS or month not in EXPECTED_MONTHS:
        raise ValueError("Outside approved development universe")
    destination=output_root/f"symbol={symbol}"/f"month={month}"
    if (output_root==root or output_root.is_relative_to(root) or
        root.is_relative_to(output_root) or not output_root.is_dir() or destination.exists()):
        raise ValueError("Require output root outside P0 and NEW partition")
    lineage=validate_coverage(coverage_json,coverage_csv)
    begin=pd.Timestamp(month+"-01T00:00:00Z")
    end=begin+pd.offsets.MonthBegin(1)
    with tempfile.TemporaryDirectory(dir=output_root,prefix="v23-staging-") as name:
        stage=Path(name)
        raw=stage/"raw";raw.mkdir()
        decoded={};provenance={}
        for kind in KINDS:
            url=BASE+PATHS[kind].format(s=symbol,m=month)
            blob=fetch(url,ZIP_LIMIT)
            check=fetch(url+".CHECKSUM",4096)
            sha=hashlib.sha256(blob).hexdigest()
            if sha!=checksum(check):raise ValueError("SHA mismatch: "+kind)
            (raw/(kind+".zip")).write_bytes(blob)
            (raw/(kind+".zip.CHECKSUM")).write_bytes(check)
            csv=verified_csv(blob)
            decoded[kind]=(parse_funding(csv,begin,end) if kind=="funding"
                           else parse_kline(csv,kind,begin,end))
            provenance[kind]={"zip_sha256":sha,
                              "checksum_sha256":hashlib.sha256(check).hexdigest()}
            print(kind,"verified",flush=True)
        a,b,c=(decoded[k] for k in ("spot","perp","mark"))
        if not a[0].reset_index(drop=True).equals(b[0].reset_index(drop=True)) or not a[0].reset_index(drop=True).equals(c[0].reset_index(drop=True)):
            raise ValueError("Markets not synchronized")
        p0sha=compare_p0(root,symbol,month,b[0],b[2])
        markets=pd.DataFrame({"bar_open_at":a[0],"bar_close_at":a[0]+M5})
        for kind,(t,unit,data) in zip(("spot","perp","mark"),(a,b,c)):
            for field,value in data.items():markets[kind+"_"+field]=value
            provenance[kind]["timestamp_unit"]=unit
        markets_path=stage/"markets_5m.parquet"
        funding_path=stage/"funding_events.parquet"
        markets.to_parquet(markets_path,index=False,engine="pyarrow",compression="zstd")
        decoded["funding"].to_parquet(funding_path,index=False,engine="pyarrow",compression="zstd")
        manifest={"protocol":PROTOCOL,"symbol":symbol,"month":month,
                  "v22":lineage,"p0_sha256":p0sha,"archives":provenance,
                  "market_bars":len(markets),"funding_events":len(decoded["funding"]),
                  "markets_sha256":hashlib.sha256(markets_path.read_bytes()).hexdigest(),
                  "funding_sha256":hashlib.sha256(funding_path.read_bytes()).hexdigest(),
                  "status":"CHECKSUM_AND_P0_VERIFIED_NO_PNL"}
        (stage/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        destparent=destination.parent
        destparent.mkdir(exist_ok=True)
        if destination.exists():raise ValueError("Destination already exists")
        stage.rename(destination)
        return manifest


def main():
    parser=argparse.ArgumentParser(description="V23 one-month source ingest, no economics")
    parser.add_argument("--ingest-one-partition",action="store_true")
    parser.add_argument("--symbol",choices=SYMBOLS,required=True)
    parser.add_argument("--month",choices=EXPECTED_MONTHS,required=True)
    for item in ("data-root","output-root","coverage-json","coverage-csv"):
        parser.add_argument("--"+item,type=Path,required=True)
    args=parser.parse_args()
    if not args.ingest_one_partition:parser.error("Explicit ingest flag required")
    result=ingest(args.symbol,args.month,args.data_root.resolve(),args.output_root.resolve(),
                  args.coverage_json.resolve(),args.coverage_csv.resolve())
    print("V23 VERIFIED",result["symbol"],result["month"],result["market_bars"],
          "M5",result["funding_events"],"funding")
    print("NO BACKTEST / NO TRAINING / NO ORDERS")


if __name__=="__main__":main()
