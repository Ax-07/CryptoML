"""P3 single-process 5-minute UTC scheduler, without backfill.

Runs in the user's foreground/OS-managed process, NEVER an OpenAI
background task. Strictly one attempt per closed M5. A failed candle
remains missing; errors are logged but cannot trigger an old-bar retry.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from mlpipeline.prospective_anchor import TrustedAnchor
from mlpipeline.prospective_live_capture import capture_once
from mlpipeline.prospective_protocol import _date, verify
from mlpipeline.prospective_attempts import claim_attempt
from mlpipeline.prospective_protocol import verify_activation

INTERVAL_SECONDS=300
START_DELAY_SECONDS=15


def due_bar_epoch(now: datetime) -> int | None:
    if now.tzinfo is None:
        raise ValueError("UTC-aware scheduler clock required")
    epoch=int(now.timestamp())
    expected=(epoch // INTERVAL_SECONDS)*INTERVAL_SECONDS
    if epoch < expected + START_DELAY_SECONDS:
        return None
    if epoch >= expected + 270:
        return None
    return expected


def run(root: Path, *, anchor: TrustedAnchor) -> None:
    registration=verify(root)
    verify_activation(root)  # fail before any attempt if packs were never locked
    start=_date(registration["prospective_start_utc"])
    end=_date(registration["prospective_end_exclusive_utc"])
    last_attempted=None
    print("P3 UTC M5 scheduler started; local time needs independent synchronization",
          flush=True)
    while True:
        now=datetime.now(timezone.utc)
        if now>=end:
            print("P3 collection window ended; no automatic scoring",flush=True)
            return
        if now<start:
            time.sleep(min(10,(start-now).total_seconds()))
            continue
        due=due_bar_epoch(now)
        if due is None or due==last_attempted:
            time.sleep(1)
            continue
        last_attempted=due
        try:
            from datetime import datetime as _datetime
            bar_close=_datetime.fromtimestamp(due,tz=timezone.utc)
            if not claim_attempt(root,close=bar_close,now=datetime.now(timezone.utc)):
                print(f"SKIPPED already attempted M5 close_epoch={due}",flush=True)
                continue
            capture_once(root,anchor=anchor,expected_close=bar_close)
        except Exception as exc:
            print(f"MISSING P3 candle close_epoch={due}: "
                  f"{type(exc).__name__}: {exc}",flush=True)
            # Never retry old bar after failure; loss counts in completeness.
        time.sleep(1)


def main():
    p=argparse.ArgumentParser(description="P3 M5 scheduler; no backfill or scoring")
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--anchor-url",required=True)
    p.add_argument("--anchor-public-key",type=Path,required=True)
    args=p.parse_args()
    anchor=TrustedAnchor(url=args.anchor_url,public_key_path=args.anchor_public_key)
    run(args.root,anchor=anchor)


if __name__=="__main__":
    main()
