"""Restart/late/corruption guards for P3 attempt journal (no network)."""
from datetime import datetime, timedelta, timezone
import pytest
from mlpipeline.prospective_attempts import claim_attempt, verify_attempts

def t(minute=0,second=20):
    return datetime(2026,10,16,12,minute,second,tzinfo=timezone.utc)

def test_claim_survives_restart_and_refuses_retry(tmp_path):
    root=tmp_path/"p3"
    close=t(second=0)
    assert claim_attempt(root,close=close,now=t()) is True
    assert claim_attempt(root,close=close,now=t(second=25)) is False
    assert len((root/"attempts.jsonl").read_bytes().splitlines())==1
    assert verify_attempts(root/"attempts.jsonl")[1]==close

def test_reject_future_and_stale_claim(tmp_path):
    close=t(second=0)
    with pytest.raises(ValueError,match="stale"):
        claim_attempt(tmp_path,close=close,now=close+timedelta(seconds=271))
    with pytest.raises(ValueError,match="stale"):
        claim_attempt(tmp_path,close=close,now=close-timedelta(seconds=1))
    assert not (tmp_path/"attempts.jsonl").exists()

def test_corrupted_attempt_file_blocks_future_claim(tmp_path):
    root=tmp_path/"p3"
    assert claim_attempt(root,close=t(second=0),now=t())
    with (root/"attempts.jsonl").open("ab") as fp:
        fp.write(b'{"incomplete":')
    with pytest.raises(ValueError,match="Incomplete"):
        claim_attempt(root,close=t(minute=5,second=0),now=t(minute=5))
