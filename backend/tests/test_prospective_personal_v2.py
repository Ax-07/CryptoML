"""Personal V2 protocol and journal guards. All synthetic; no live network."""
from datetime import datetime, timedelta, timezone
import json
import pytest
from mlpipeline.prospective_personal_protocol import (
    make_registration, register, verify, spec, PROTOCOL_ID,
)
from mlpipeline.prospective_personal_journal import verify_local_chain
from mlpipeline.prospective_protocol import spec as v1spec


def now():
    return datetime(2026,10,14,19,30,tzinfo=timezone.utc)

def start():
    return datetime(2026,10,18,0,0,tzinfo=timezone.utc)

def test_new_personal_v2_research_primary_unchanged():
    local=spec()
    original=v1spec()
    assert local["protocol_id"]==PROTOCOL_ID
    assert local["evidence"]["external_timestamp"] is False
    assert local["primary"]==original["primary"]
    assert local["secondary"]==original["secondary"]
    assert local["evaluation"]==original["evaluation"]
    assert local["symbols"]==original["symbols"]


def test_personal_v2_future_registration_and_hash_locked(tmp_path):
    root=tmp_path/"personal-v2"
    reg=register(root,start=start(),now=now())
    assert verify(root)==reg
    assert reg["evaluation_not_before_utc"]=="2026-12-13T01:00:00Z"
    with pytest.raises(FileExistsError):
        register(root,start=start(),now=now())
    with (root/"protocol.json").open("ab") as f:
        f.write(b" ")
    with pytest.raises(ValueError,match="SHA256"):
        verify(root)


def test_personal_v2_requires_two_full_days_notice(tmp_path):
    with pytest.raises(ValueError,match="48h"):
        make_registration(start=datetime(2026,10,16,tzinfo=timezone.utc),now=now())
    with pytest.raises(ValueError,match="midnight"):
        make_registration(start=start()+timedelta(minutes=5),now=now())


def test_personal_v2_refuses_old_v1_registration(tmp_path):
    from mlpipeline.prospective_protocol import register as register_v1
    root=tmp_path/"v1"
    register_v1(root,now=now())
    with pytest.raises(ValueError,match="personal P3|frozen protocol"):
        verify(root)


def test_local_journal_missing_and_corrupted(tmp_path):
    path=tmp_path/"predictions.jsonl"
    assert verify_local_chain(path)==("0"*64,None)
    path.write_bytes(b'{"bar_close_utc":"2026-10-18T00:05:00Z"}')
    with pytest.raises(ValueError,match="Incomplete"):
        verify_local_chain(path)


def test_v1_activate_verifier_hook_still_defaults_to_v1(tmp_path):
    from mlpipeline.prospective_protocol import register as register_v1,activate
    root=tmp_path/"v1"
    register_v1(root,now=now())
    # No models should be read until V1 registration and date bounds are verified.
    with pytest.raises(FileNotFoundError):
        activate(root,model_h2=tmp_path/"missing-h2",model_h6=tmp_path/"missing-h6",
                 threshold_file=tmp_path/"missing-policy",now=now())
