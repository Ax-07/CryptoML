"""P3 PERSONAL V2 M5 scheduler (no paid TSA and no retrospective retries).

Run under an operator supervised Windows process. No service installation,
no scoring, no historical P0 data, no ADA, no trading.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from mlpipeline.prospective_attempts import claim_attempt
from mlpipeline.prospective_live_capture import (
    _load_packs, fetch_latest_closed, latest_common_features, make_forecasts,
)
from mlpipeline.prospective_personal_journal import append_local
from mlpipeline.prospective_personal_protocol import verify, verify_activation
from mlpipeline.prospective_protocol import ASSETS, _date
from mlpipeline.prospective_scheduler import due_bar_epoch


def capture_personal(root: Path, *, expected_close: datetime):
    registration = verify(root)
    activation = verify_activation(root)
    now = datetime.now(timezone.utc)
    if not (_date(registration["prospective_start_utc"]) <= expected_close
            < _date(registration["prospective_end_exclusive_utc"])):
        raise ValueError("Claimed M5 close outside V2 registered window")
    # Joblib from operator-pinned SHA256 artifacts ONLY.
    packs = _load_packs(activation)
    candles = {symbol: fetch_latest_closed(symbol, now=now) for symbol in ASSETS}
    rows, close = latest_common_features(candles, now=datetime.now(timezone.utc))
    if close != expected_close:
        raise ValueError("Exchange M5 close differs from preclaimed UTC time")
    predictions = make_forecasts(rows, packs, activation)
    saved = append_local(root, close=close, forecasts=predictions)
    print("PERSONAL V2 local batch", saved["bar_close_utc"],
          saved["entry_sha256"],flush=True)
    return saved


def run(root: Path):
    registration = verify(root)
    verify_activation(root)
    start = _date(registration["prospective_start_utc"])
    end = _date(registration["prospective_end_exclusive_utc"])
    attempted = None
    print("PERSONAL V2 local M5 scheduler; PC must remain awake and synchronized;"
          " missing bars are not reconstructed", flush=True)
    while True:
        now = datetime.now(timezone.utc)
        if now >= end:
            print("PERSONAL V2 window complete. No automatic outcome scoring.", flush=True)
            return
        if now < start:
            time.sleep(min(10, max(0.1, (start - now).total_seconds())))
            continue
        epoch = due_bar_epoch(now)
        if epoch is None or epoch == attempted:
            time.sleep(1)
            continue
        attempted = epoch
        close = datetime.fromtimestamp(epoch, tz=timezone.utc)
        try:
            # Persistent pre-claim makes crashes/missed attempts explicitly missing.
            if not claim_attempt(root,close=close,now=datetime.now(timezone.utc)):
                print("ALREADY ATTEMPTED personal M5", close.isoformat(),flush=True)
                continue
            capture_personal(root,expected_close=close)
        except Exception as exc:
            print(f"PERSONAL V2 MISSING {close.isoformat()}: "
                  f"{type(exc).__name__}: {exc}",flush=True)
        time.sleep(1)


def main():
    p=argparse.ArgumentParser(description="P3 PERSONAL V2 local M5 capture")
    p.add_argument("--root",required=True,type=Path)
    args=p.parse_args()
    run(args.root)


if __name__=="__main__":
    main()
