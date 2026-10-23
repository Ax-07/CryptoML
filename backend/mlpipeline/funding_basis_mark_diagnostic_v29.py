"""V29 read-only, SHA-confirmed June 2026 BTCUSDT mark-price ZIP diagnostic."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from mlpipeline.funding_basis_archive_integrity_v19 import BASE,PATHS,ZIP_LIMIT,fetch,checksum
from mlpipeline.funding_archive_schema_v20 import verified_csv
from mlpipeline.funding_basis_verified_partition_v23 import parse_kline
from mlpipeline.funding_basis_spot_diagnostic_v28 import examine,START,END

PROTOCOL="p4_v29_btc_june2026_mark_diagnostic_v1"
SYMBOL="BTCUSDT"
MONTH="2026-06"


def diagnose_csv(data:bytes)->dict:
    geometry=examine(data)
    geometry.pop("v23_parser_status",None)
    geometry.pop("v23_parser_error",None)
    try:
        parse_kline(data,"mark",START,END)
        geometry["v23_mark_parser_status"]="PASS"
    except (ValueError,TypeError,IndexError,KeyError) as exc:
        geometry["v23_mark_parser_status"]="FAIL"
        geometry["v23_mark_parser_error"]={"type":type(exc).__name__,"message":str(exc)[:240]}
    return geometry


def run(output:Path,allow=False)->dict:
    if not allow:raise ValueError("Explicit --permit-downloads required")
    output=output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Require new output JSON")
    url=BASE+PATHS["mark"].format(s=SYMBOL,m=MONTH)
    blob=fetch(url,ZIP_LIMIT)
    proof=fetch(url+".CHECKSUM",4096)
    digest=hashlib.sha256(blob).hexdigest()
    if digest!=checksum(proof):raise ValueError("Mark ZIP SHA256 checksum mismatch")
    data={"protocol":PROTOCOL,"symbol":SYMBOL,"month":MONTH,"kind":"mark",
          "url":url,"zip_sha256":digest,
          "checksum_sha256":hashlib.sha256(proof).hexdigest(),
          "sha256_verified":True,"geometry":diagnose_csv(verified_csv(blob)),
          "no_p0_write_no_partition_promotion":True,
          "no_pnl_no_training_no_orders":True}
    with output.open("x",encoding="utf-8") as out:json.dump(data,out,indent=2)
    return data


def main():
    p=argparse.ArgumentParser(description="V29 mark archive diagnostic only; no ingestion")
    p.add_argument("--permit-downloads",action="store_true")
    p.add_argument("--output-json",type=Path,required=True)
    a=p.parse_args()
    result=run(a.output_json,a.permit_downloads)
    g=result["geometry"]
    print("V29 BTCUSDT 2026-06 mark SHA256 VERIFIED")
    print("raw",g["raw_csv_rows"],"data",g["data_rows"],
          "columns",g["columns"],"expected",g["expected_m5"])
    print("missing",g.get("missing_m5_count"),"duplicate",g.get("duplicate_count"),
          "epoch_unit",g.get("epoch_unit"))
    print("V23 mark parser",g["v23_mark_parser_status"],g.get("v23_mark_parser_error"))
    print("NO PARTITION WRITE / NO PNL / NO ORDERS")

if __name__=="__main__":main()
