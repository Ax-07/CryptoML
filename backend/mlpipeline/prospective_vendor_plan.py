"""P3 provider procurement/readiness planning, read-only and fail-closed.

Displays volume for 56 days at one signed receipt per M5 (NOT eight receipts
per symbol/horizon), provider protocol mismatch, and registration state.
Does not buy services, access internet, activate P3 or inspect predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlpipeline.prospective_protocol import _date, verify

BAR_MINUTES=5
DAY_MINUTES=1440
EXPECTED_POLICIES=("12-17_UTC","all_day_UTC")
CODE_ANCHOR_CONTRACT="HTTPS_JSON_ED25519_SIGNED_RECEIPT_PLUS_DIGEST_GET"
CANDIDATE_PROVIDER="Certigna"
CANDIDATE_CONTRACT="RFC3161_TSR_SIGNED_BY_TSA"
NEEDS_INDEPENDENT_ARCHIVE=True


def provider_plan(registration:dict)->dict:
    start=_date(registration["prospective_start_utc"])
    end=_date(registration["prospective_end_exclusive_utc"])
    duration=end-start
    if duration.total_seconds()%(BAR_MINUTES*60)!=0:
        raise ValueError("Prospective window is not aligned to M5")
    bars=int(duration.total_seconds()//(BAR_MINUTES*60))
    if bars<=0 or bars!=(end-start).days*DAY_MINUTES//BAR_MINUTES:
        raise ValueError("Invalid prospective duration")
    if registration["protocol"]["bar_timeframe"]!="M5_closed_only":
        raise ValueError("P3 horizon incompatible with M5 procurement assumptions")
    if registration["protocol"]["horizon_bars"]!=[2,6]:
        raise ValueError("P3 expected H2/H6 policy mismatch")
    return {
        "protocol_id":registration["protocol"]["protocol_id"],
        "start_utc":registration["prospective_start_utc"],
        "end_exclusive_utc":registration["prospective_end_exclusive_utc"],
        "period_days":duration.days,
        "expected_m5_batches":bars,
        "receipts_if_one_per_batch":bars,
        "forecast_rows_per_horizon":bars*len(registration["protocol"]["symbols"]),
        "directional_policies":list(EXPECTED_POLICIES),
        "candidate_provider":CANDIDATE_PROVIDER,
        "candidate_protocol":CANDIDATE_CONTRACT,
        "implemented_anchor_contract":CODE_ANCHOR_CONTRACT,
        "contracts_compatible_as_is":False,
        "third_party_credentials_available":False,
        "independent_append_only_retention_verified":False,
        "supervised_windows_host_verified":False,
        "activation_authorized":False,
        "next_step":"Obtain commercial RFC3161 TSA access and proof-retention terms, "
                    "build/test a separate verified RFC3161 adapter, plus independent audit store.",
    }


def main():
    p=argparse.ArgumentParser(description="P3 vendor capacity and incompatible API gate")
    p.add_argument("--root",type=Path,required=True)
    args=p.parse_args()
    plan=provider_plan(verify(args.root))
    print(json.dumps(plan,indent=2,ensure_ascii=False))
    print("STOP: this tool never activates a P3 registration.",flush=True)


if __name__=="__main__":
    main()
