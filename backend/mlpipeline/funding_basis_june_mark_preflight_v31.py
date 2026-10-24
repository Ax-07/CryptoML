"""V31 read-only SHA-backed June 2026 mark archive preflight for ETH/SOL/XRP."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from mlpipeline.funding_basis_archive_integrity_v19 import BASE, PATHS
from mlpipeline.funding_basis_daily_mark_reconciliation_v30 import verified_archive
from mlpipeline.funding_basis_mark_diagnostic_v29 import diagnose_csv

PROTOCOL="p4_v31_june2026_mark_preflight_v1"
SYMBOLS=("ETHUSDT","SOLUSDT","XRPUSDT")
MONTH="2026-06"


def run(output:Path,allow=False)->dict:
    if not allow:raise ValueError("Explicit --permit-downloads required")
    output=output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Require new output JSON")
    rows=[]
    for symbol in SYMBOLS:
        url=BASE+PATHS["mark"].format(s=symbol,m=MONTH)
        csv,source=verified_archive(url)
        geometry=diagnose_csv(csv)
        rows.append({"symbol":symbol,"month":MONTH,
                     "archive":source,"geometry":geometry})
    result={"protocol":PROTOCOL,"rows":rows,
            "no_partition_written":True,"no_imputation_no_pnl_no_orders":True}
    with output.open("x",encoding="utf-8") as file:
        json.dump(result,file,indent=2)
    return result


def main():
    p=argparse.ArgumentParser(description="V31 mark archive diagnostic for remaining June assets")
    p.add_argument("--permit-downloads",action="store_true")
    p.add_argument("--output-json",required=True,type=Path)
    a=p.parse_args()
    result=run(a.output_json,a.permit_downloads)
    for row in result["rows"]:
        g=row["geometry"]
        print(row["symbol"],"mark checksum verified",
              "bars",g["data_rows"],"expected",g["expected_m5"],
              "missing",g.get("missing_m5_count"),
              "parse",g["v23_mark_parser_status"],flush=True)
    print("NO PARTITION WRITE / NO PNL / NO ORDERS")

if __name__=="__main__":main()
