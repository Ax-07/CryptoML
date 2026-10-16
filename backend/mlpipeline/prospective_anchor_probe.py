"""Synthetic operator-side preflight against an independent receipt provider.

Does not activate P3, load models, read P0, write a journal or submit
actual model forecasts. Remote provider must independently retain proofs.
"""
from __future__ import annotations

import argparse
import secrets
from datetime import datetime, timezone
from pathlib import Path

from mlpipeline.prospective_anchor import TrustedAnchor
from mlpipeline.prospective_protocol import _iso


def probe(anchor: TrustedAnchor) -> dict:
    # Unique NON-PREDICTION digest, never used in actual P3 scoring.
    now=datetime.now(timezone.utc)
    core={
        "bar_close_utc":_iso(now),
        "recorded_at_utc":_iso(now),
        "protocol_sha256":"0"*64,
        "activation_sha256":"0"*64,
        "previous_sha256":"0"*64,
        "staging_probe_nonce":secrets.token_hex(32),
        "forecasts":{},
    }
    receipt=anchor.attest(core,close=now)
    return {
        "independent_receipt_signature_valid":True,
        "immediate_retrieval_identical":True,
        "receipt_service_id":receipt["signed"]["service_id"],
        "probe_only_no_model_forecasts":True,
        "note":"Retention, provider independence and future lookup must be audited separately",
    }


def main():
    p=argparse.ArgumentParser(description="P3 external anchor probe, no activation")
    p.add_argument("--anchor-url",required=True)
    p.add_argument("--anchor-lookup-url",required=True)
    p.add_argument("--anchor-public-key",required=True,type=Path)
    args=p.parse_args()
    a=TrustedAnchor(url=args.anchor_url,lookup_url_template=args.anchor_lookup_url,
                    public_key_path=args.anchor_public_key)
    result=probe(a)
    for key,value in result.items():
        print(f"{key}: {value}",flush=True)


if __name__=="__main__":
    main()
