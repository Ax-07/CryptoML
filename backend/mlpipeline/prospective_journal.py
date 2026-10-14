"""P3 locked append-only local journal verifier (not independent evidence).

Windows msvcrt.locking / POSIX flock lock the SAME persistent lock inode;
never unlink it. Readers require valid complete newline-terminated chain.
External receipts are verified in the capture flow; verifier optionally
checks signatures when provided a configured TrustedAnchor.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from mlpipeline.prospective_protocol import _canonical, _date, _sha


@contextmanager
def exclusive_writer(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".predictions.writer.lock"
    with path.open("a+b") as lock:
        lock.seek(0)
        if lock.read(1) == b"":
            lock.seek(0)
            lock.write(b"X")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another prediction writer holds the Windows lock") from exc
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Another prediction writer holds the lock") from exc
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def verify_chain(path: Path, *, anchor=None):
    previous = "0" * 64
    last = None
    if not path.exists():
        return previous, last
    with path.open("rb") as file:
        for i, line in enumerate(file, start=1):
            if not line.endswith(b"\n"):
                raise ValueError(f"Incomplete journal row {i}; do not repair/backfill")
            row = json.loads(line)
            signature = row.get("entry_sha256")
            core = {k: v for k, v in row.items() if k != "entry_sha256"}
            if (row.get("previous_sha256") != previous
                    or _sha(_canonical(core)) != signature
                    or _canonical(row) != line):
                raise ValueError(f"Journal chain corruption at row {i}")
            close = _date(row["bar_close_utc"])
            issued = _date(row["recorded_at_utc"])
            if last is not None and close <= last:
                raise ValueError(f"Duplicate/backfilled journal close {i}")
            if not (0 <= (issued-close).total_seconds() < 270):
                raise ValueError(f"Late recorded forecast at journal row {i}")
            if "external_receipt" not in row:
                raise ValueError(f"Missing independent evidence at row {i}")
            if anchor is not None:
                evidence_core={k:v for k,v in core.items() if k!="external_receipt"}
                anchor.verify(row["external_receipt"],evidence_core,close=close)
            last = close
            previous = signature
    return previous,last
