"""P3 PERSONAL V2 local-only append: no paid anchoring, no backfill, no scoring.

Not independent trusted timestamp evidence. V1 strict external journal untouched.
"""
from __future__ import annotations
import os
from datetime import datetime, timezone
from pathlib import Path

from mlpipeline.prospective_journal import exclusive_writer
from mlpipeline.prospective_protocol import ASSETS, _canonical, _date, _iso, _sha
from mlpipeline.prospective_personal_protocol import verify, verify_activation

DEADLINE_SECONDS=270

def verify_local_chain(path:Path):
    import json
    previous="0"*64
    last=None
    if not path.exists():
        return previous,last
    with path.open("rb") as f:
        for i,line in enumerate(f,1):
            if not line.endswith(b"\n"):
                raise ValueError(f"Incomplete local prediction row {i}")
            row=json.loads(line)
            core={k:v for k,v in row.items() if k!="entry_sha256"}
            if (set(row)!={"bar_close_utc","recorded_at_utc","protocol_sha256",
                "activation_sha256","previous_sha256","forecasts",
                "evidence_mode","entry_sha256"}
                or row["previous_sha256"]!=previous or _canonical(row)!=line
                or _sha(_canonical(core))!=row["entry_sha256"]
                or row["evidence_mode"]!="personal_local_no_external_anchor"):
                raise ValueError(f"Tampered personal prediction journal row {i}")
            close=_date(row["bar_close_utc"])
            recorded=_date(row["recorded_at_utc"])
            if (int(close.timestamp())%300 != 0
                    or last is not None and close<=last
                    or not 0<=(recorded-close).total_seconds()<DEADLINE_SECONDS):
                raise ValueError(f"Out-of-order or stale prediction at row {i}")
            last=close
            previous=row["entry_sha256"]
    return previous,last


def append_local(root:Path, *, close:datetime, forecasts:dict,
                 clock=None):
    from mlpipeline.prospective_live_capture import _check_forecasts
    reg=verify(root)
    verify_activation(root)
    now=(clock or (lambda:datetime.now(timezone.utc)))()
    if now.tzinfo is None or close.tzinfo is None:
        raise ValueError("UTC-aware time required")
    close=close.astimezone(timezone.utc)
    now=now.astimezone(timezone.utc)
    if (int(close.timestamp())%300 != 0
            or not (_date(reg["prospective_start_utc"])<=close
                    <_date(reg["prospective_end_exclusive_utc"]))
            or not 0<=(now-close).total_seconds()<DEADLINE_SECONDS):
        raise ValueError("Outside registered window, stale or retroactive forecast")
    _check_forecasts(forecasts)
    with exclusive_writer(root):
        file=root/"predictions.jsonl"
        previous,last=verify_local_chain(file)
        if last is not None and close<=last:
            raise ValueError("Duplicate/retroactive prediction")
        row={
            "bar_close_utc":_iso(close),
            "recorded_at_utc":_iso(now),
            "protocol_sha256":(root/"protocol.sha256").read_text("ascii").strip(),
            "activation_sha256":(root/"activation.sha256").read_text("ascii").strip(),
            "previous_sha256":previous,
            "evidence_mode":"personal_local_no_external_anchor",
            "forecasts":forecasts,
        }
        row["entry_sha256"]=_sha(_canonical(row))
        # clock checked again after construction: bound actual disk-write time.
        final=(clock or (lambda:datetime.now(timezone.utc)))()
        if not 0<=(final-close).total_seconds()<DEADLINE_SECONDS:
            raise ValueError("Expired before local append")
        fd=os.open(str(file),os.O_APPEND|os.O_CREAT|os.O_WRONLY,0o600)
        with os.fdopen(fd,"ab") as stream:
            stream.write(_canonical(row))
            stream.flush()
            os.fsync(stream.fileno())
        return row
