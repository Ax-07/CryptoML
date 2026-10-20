"""P3 persistent M5 attempt claims: never retry an M5 after restart.

A claim is fsynced BEFORE the network/model work; it is a *local* operational
record, not independent timestamp proof or an observation/performance score.
A crashed or failed attempt is left missing and may not be replayed.
"""
from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from pathlib import Path

from mlpipeline.prospective_journal import exclusive_writer
from mlpipeline.prospective_protocol import _canonical, _date, _iso, _sha

SLOT_SECONDS=300
MAX_AGE_SECONDS=270
GENESIS="0"*64


def verify_attempts(path:Path):
    previous=GENESIS
    last=None
    if not path.exists():
        return previous,last
    with path.open("rb") as f:
        for index,line in enumerate(f,start=1):
            if not line.endswith(b"\n"):
                raise ValueError(f"Incomplete P3 attempt journal row {index}")
            row=json.loads(line)
            signed={k:v for k,v in row.items() if k!="entry_sha256"}
            if (set(row)!={"bar_close_utc","attempted_at_utc",
                          "previous_sha256","entry_sha256"}
                    or _canonical(row)!=line
                    or row["previous_sha256"]!=previous
                    or _sha(_canonical(signed))!=row["entry_sha256"]):
                raise ValueError(f"Attempt journal corruption at row {index}")
            close=_date(row["bar_close_utc"])
            attempted=_date(row["attempted_at_utc"])
            if (last is not None and close<=last
                    or not 0<=(attempted-close).total_seconds()<MAX_AGE_SECONDS
                    or int(close.timestamp())%SLOT_SECONDS!=0):
                raise ValueError(f"Invalid/retroactive attempt journal row {index}")
            previous=row["entry_sha256"]
            last=close
    return previous,last


def claim_attempt(root:Path, *, close:datetime, now:datetime) -> bool:
    """Claim exactly once per M5, atomically across process restarts.

    Returns False for a previously claimed close, never appends backdated
    attempts; fail-closed on corruption. Requires a real UTC-aware clock.
    """
    if close.tzinfo is None or now.tzinfo is None:
        raise ValueError("UTC-aware time required")
    close=close.astimezone(timezone.utc)
    now=now.astimezone(timezone.utc)
    if (int(close.timestamp())%SLOT_SECONDS!=0
            or not 0<=(now-close).total_seconds()<MAX_AGE_SECONDS):
        raise ValueError("Attempt would be stale, future or non-M5 aligned")
    with exclusive_writer(root):
        path=root/"attempts.jsonl"
        previous,last=verify_attempts(path)
        if last is not None and close<=last:
            return False
        row={
            "bar_close_utc":_iso(close),
            "attempted_at_utc":_iso(now),
            "previous_sha256":previous,
        }
        row["entry_sha256"]=_sha(_canonical(row))
        fd=os.open(str(path),os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o600)
        with os.fdopen(fd,"ab") as stream:
            stream.write(_canonical(row))
            stream.flush()
            os.fsync(stream.fileno())
    return True
