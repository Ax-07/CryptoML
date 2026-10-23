"""V28 read-only, checksum-confirmed diagnostic for BTCUSDT June 2026 spot ZIP."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from mlpipeline.funding_basis_archive_integrity_v19 import BASE,PATHS,ZIP_LIMIT,fetch,checksum
from mlpipeline.funding_archive_schema_v20 import verified_csv
from mlpipeline.funding_basis_verified_partition_v23 import parse_kline

PROTOCOL="p4_v28_btc_june2026_spot_diagnostic_v1"
SYMBOL="BTCUSDT"
MONTH="2026-06"
START=pd.Timestamp("2026-06-01T00:00:00Z")
END=pd.Timestamp("2026-07-01T00:00:00Z")


def examine(data:bytes)->dict:
    frame=pd.read_csv(io.BytesIO(data),header=None,low_memory=False,dtype=str,keep_default_na=False)
    if frame.empty:raise ValueError("Empty spot CSV")
    header=not frame.iat[0,0].strip().isdigit()
    rows=frame.iloc[1:].reset_index(drop=True) if header else frame
    expected=pd.date_range(START,END-pd.Timedelta(minutes=5),freq="5min",tz="UTC")
    result={"raw_csv_rows":len(frame),"header_detected":header,"data_rows":len(rows),
            "columns":int(frame.shape[1]),"expected_m5":len(expected)}
    try:
        values=pd.to_numeric(rows.iloc[:,0],errors="raise")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError("Bad epoch")
        digits=values.astype("int64").abs().astype(str).str.len()
        if digits.nunique()!=1:raise ValueError("Mixed epoch units")
        unit={10:"s",13:"ms",16:"us"}.get(int(digits.iloc[0]))
        if unit is None:raise ValueError("Unsupported epoch unit")
        stamps=pd.to_datetime(values.astype("int64"),unit=unit,utc=True,errors="raise")
        in_month=stamps.between(START,END,inclusive="left")
        unique=pd.DatetimeIndex(stamps[in_month].drop_duplicates())
        missing=expected.difference(unique)
        outside_grid=unique.difference(expected)
        result.update({"epoch_unit":unit,"first_utc":stamps.iloc[0].isoformat(),
                       "last_utc":stamps.iloc[-1].isoformat(),
                       "duplicate_count":int(stamps.duplicated().sum()),
                       "out_of_month_count":int((~in_month).sum()),
                       "off_grid_count":len(outside_grid),
                       "missing_m5_count":len(missing),
                       "first_missing_utc":[t.isoformat() for t in missing[:10]]})
    except (ValueError,TypeError,IndexError,OverflowError) as exc:
        result["timeline_error"]={"type":type(exc).__name__,"message":str(exc)[:240]}
    try:
        parse_kline(data,"spot",START,END)
        result["v23_parser_status"]="PASS"
    except (ValueError,TypeError,IndexError,KeyError) as exc:
        result["v23_parser_status"]="FAIL"
        result["v23_parser_error"]={"type":type(exc).__name__,"message":str(exc)[:240]}
    return result


def run(output:Path,allow=False)->dict:
    if not allow:raise ValueError("Explicit --permit-downloads required")
    output=output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Require new output JSON")
    url=BASE+PATHS["spot"].format(s=SYMBOL,m=MONTH)
    blob=fetch(url,ZIP_LIMIT)
    proof=fetch(url+".CHECKSUM",4096)
    digest=hashlib.sha256(blob).hexdigest()
    if digest!=checksum(proof):raise ValueError("Spot ZIP SHA256 checksum mismatch")
    csv=verified_csv(blob)
    data={"protocol":PROTOCOL,"symbol":SYMBOL,"month":MONTH,"kind":"spot",
          "url":url,"zip_sha256":digest,
          "checksum_sha256":hashlib.sha256(proof).hexdigest(),
          "sha256_verified":True,"geometry":examine(csv),
          "no_p0_write_no_partition_promotion":True,
          "no_pnl_no_training_no_orders":True}
    with output.open("x",encoding="utf-8") as f:json.dump(data,f,indent=2)
    return data


def main():
    p=argparse.ArgumentParser(description="V28 single spot archive diagnostic; no ingest or trading")
    p.add_argument("--permit-downloads",action="store_true")
    p.add_argument("--output-json",type=Path,required=True)
    a=p.parse_args()
    result=run(a.output_json,a.permit_downloads)
    g=result["geometry"]
    print("V28 BTCUSDT 2026-06 spot SHA256 VERIFIED",flush=True)
    print("raw",g["raw_csv_rows"],"data",g["data_rows"],
          "columns",g["columns"],"expected",g["expected_m5"],flush=True)
    print("timeline",g.get("epoch_unit"),"missing",g.get("missing_m5_count"),
          "duplicates",g.get("duplicate_count"),flush=True)
    print("V23 parser",g["v23_parser_status"],g.get("v23_parser_error"),flush=True)
    print("NO PARTITION WRITE / NO ORDERS / NO PNL")

if __name__=="__main__":main()
