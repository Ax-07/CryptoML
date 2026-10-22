"""V25 offline tests for immutable target plans, stop, and resumption."""
import json
from pathlib import Path
import pandas as pd
import pytest
import mlpipeline.funding_basis_batch_resume_v25 as m


def _fixtures(tmp_path,monkeypatch,statuses=None):
    root=tmp_path/"p0";v21=tmp_path/"v21";v23=tmp_path/"v23"
    for p in (root,v21,v23):p.mkdir()
    cj=tmp_path/"coverage.json";cc=tmp_path/"coverage.csv"
    cj.write_text("{}");cc.write_text("key\n")
    monkeypatch.setattr(m,"validate_coverage",lambda a,b:{"json_sha256":"a"*64,"csv_sha256":"b"*64})
    def fake_registry(root,v21,v23):
        states=statuses or {}
        return pd.DataFrame([{"symbol":s,"month":month,
                              "status":states.get((s,month),"MISSING")}
                 for s in m.SYMBOLS for month in m.EXPECTED_MONTHS])
    monkeypatch.setattr(m,"registry",fake_registry)
    return root,v21,v23,cj,cc


def test_offline_plan_fixed_targets(tmp_path,monkeypatch):
    values=_fixtures(tmp_path,monkeypatch)
    saved=tmp_path/"plan.json"
    result=m.plan(*values,saved)
    assert result["targets"]==[{"symbol":s,"month":mo} for s,mo in m.TARGETS]
    assert result["initial_missing"]==84
    assert saved.exists()
    with pytest.raises(ValueError,match="NEW"):
        m.plan(*values,saved)


def test_plan_refuses_failed_integrity(tmp_path,monkeypatch):
    state={("BTCUSDT","2025-06"):"FAILED_INTEGRITY"}
    values=_fixtures(tmp_path,monkeypatch,state)
    with pytest.raises(ValueError,match="FAILED_INTEGRITY"):
        m.plan(*values,tmp_path/"out.json")


def test_execute_rejects_without_download_permission(tmp_path):
    with pytest.raises(ValueError,match="permit-downloads"):
        m.execute(tmp_path/"absent",tmp_path/"log",tmp_path/"summary")


def test_execute_rejects_tampered_targets_before_download(tmp_path,monkeypatch):
    values=_fixtures(tmp_path,monkeypatch)
    saved=tmp_path/"plan.json"
    result=m.plan(*values,saved)
    result["targets"][0]["symbol"]="XRPUSDT"
    saved.write_text(json.dumps(result))
    def no_ingest(*args,**kwargs):
        raise AssertionError("No network")
    monkeypatch.setattr(m,"ingest",no_ingest)
    with pytest.raises(ValueError,match="target mismatch"):
        m.execute(saved,tmp_path/"log",tmp_path/"summary",True)


def test_resume_skips_existing_and_verifies_new(tmp_path,monkeypatch):
    states={("ETHUSDT","2024-10"):"VERIFIED_V23"}
    values=_fixtures(tmp_path,monkeypatch,states)
    saved=tmp_path/"plan.json"
    # Plan was frozen when the two targets were missing.
    monkeypatch.setattr(m,"registry",lambda *args:pd.DataFrame(
        [{"symbol":s,"month":mo,"status":"MISSING"}
         for s in m.SYMBOLS for mo in m.EXPECTED_MONTHS]))
    m.plan(*values,saved)
    monkeypatch.setattr(m,"registry",lambda *args:pd.DataFrame(
        [{"symbol":s,"month":mo,"status":states.get((s,mo),"MISSING")}
         for s in m.SYMBOLS for mo in m.EXPECTED_MONTHS]))
    events=[]
    def fake_ingest(s,mo,*args):
        events.append((s,mo))
        states[(s,mo)]="VERIFIED_V23"
    monkeypatch.setattr(m,"ingest",fake_ingest)
    monkeypatch.setattr(m,"verify",lambda *args,**kwargs:{"m5":8928})
    result=m.execute(saved,tmp_path/"journal.jsonl",tmp_path/"summary.json",True)
    assert events==[("SOLUSDT","2024-10")]
    assert result["newly_verified"]==1
    assert result["failed"] is None
    logs=[json.loads(line)["event"] for line in (tmp_path/"journal.jsonl").read_text().splitlines()]
    assert logs==["PRECHECK_OK","ALREADY_VERIFIED_SKIPPED","STARTED","VERIFIED","BATCH_DONE"]


def test_failure_stops_second_ingest_and_journals(tmp_path,monkeypatch):
    values=_fixtures(tmp_path,monkeypatch)
    saved=tmp_path/"plan.json"
    m.plan(*values,saved)
    seen=[]
    def broken(s,mo,*args):
        seen.append((s,mo))
        raise ValueError("Synthetic ZIP SHA mismatch")
    monkeypatch.setattr(m,"ingest",broken)
    with pytest.raises(RuntimeError,match="halted"):
        m.execute(saved,tmp_path/"journal.jsonl",tmp_path/"summary.json",True)
    assert seen==[m.TARGETS[0]]
    summary=json.loads((tmp_path/"summary.json").read_text())
    assert summary["failed"]["error_type"]=="ValueError"
    logs=[json.loads(line)["event"] for line in (tmp_path/"journal.jsonl").read_text().splitlines()]
    assert logs==["PRECHECK_OK","STARTED","FAILED","BATCH_FAILED"]
