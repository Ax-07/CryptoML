"""V26 offline tests: strict targets, planning, digest approval, stop and resume."""
import json
import pandas as pd
import pytest
import mlpipeline.funding_basis_flexible_microbatch_v26 as m


def setup(tmp_path,monkeypatch):
    roots=[tmp_path/x for x in ("p0","v21","v23")]
    for p in roots:p.mkdir()
    cj=tmp_path/"coverage.json";cc=tmp_path/"coverage.csv"
    cj.write_text("{}");cc.write_text("x\n")
    monkeypatch.setattr(m,"validate_coverage",lambda *args:{"json_sha256":"a"*64,"csv_sha256":"b"*64})
    states={}
    def fake(*args):
        return pd.DataFrame([{"symbol":s,"month":mo,
            "status":states.get((s,mo),"MISSING")}
            for s in m.SYMBOLS for mo in m.EXPECTED_MONTHS])
    monkeypatch.setattr(m,"audit",fake)
    return (*roots,cj,cc),states


def test_target_validation():
    assert m.targets(["ETHUSDT:2024-10","SOLUSDT:2024-10"])==[
        ("ETHUSDT","2024-10"),("SOLUSDT","2024-10")]
    for sample in ([],["ADAUSDT:2024-10"],
                   ["BTCUSDT:2026-07"],["ETHUSDT:2024-10"]*2,
                   ["BTCUSDT:2024-10"]*3):
        with pytest.raises(ValueError):
            m.targets(sample)


def test_plan_offline_and_immutable(tmp_path,monkeypatch):
    src,_=setup(tmp_path,monkeypatch)
    p=tmp_path/"plan.json"
    plan=m.make_plan(src,["ETHUSDT:2024-10","SOLUSDT:2024-10"],p)
    assert plan["max_new"]==2 and plan["initial_missing"]==84
    with pytest.raises(ValueError,match="New plan"):
        m.make_plan(src,["XRPUSDT:2024-10"],p)


def test_plan_refuses_corrupt_registry(tmp_path,monkeypatch):
    src,states=setup(tmp_path,monkeypatch)
    states[("BTCUSDT","2025-06")]="FAILED_INTEGRITY"
    def strict_audit(*args):
        raise ValueError("FAILED_INTEGRITY")
    monkeypatch.setattr(m,"audit",strict_audit)
    with pytest.raises(ValueError,match="FAILED_INTEGRITY"):
        m.make_plan(src,["XRPUSDT:2024-10"],tmp_path/"plan.json")


def test_sha_approval_before_network(tmp_path,monkeypatch):
    src,_=setup(tmp_path,monkeypatch)
    p=tmp_path/"plan.json"
    m.make_plan(src,["XRPUSDT:2024-10"],p)
    monkeypatch.setattr(m,"ingest",lambda *args: (_ for _ in ()).throw(
        AssertionError("Network must not be contacted")))
    with pytest.raises(ValueError,match="SHA256"):
        m.execute(p,"f"*64,tmp_path/"journal.jsonl",tmp_path/"summary.json",True)


def test_execute_resumes_and_skips_verified(tmp_path,monkeypatch):
    src,states=setup(tmp_path,monkeypatch)
    p=tmp_path/"plan.json"
    m.make_plan(src,["ETHUSDT:2024-10","SOLUSDT:2024-10"],p)
    states[("ETHUSDT","2024-10")]="VERIFIED_V23"
    ingested=[]
    def fake(s,mo,*args):
        ingested.append((s,mo))
        states[(s,mo)]="VERIFIED_V23"
    monkeypatch.setattr(m,"ingest",fake)
    monkeypatch.setattr(m,"verify",lambda *args,**kwargs:{"m5":8928})
    report=m.execute(p,m.sha(p),tmp_path/"journal.jsonl",tmp_path/"summary.json",True)
    assert ingested==[("SOLUSDT","2024-10")]
    assert report["newly_verified"]==1
    events=[json.loads(x)["event"] for x in (tmp_path/"journal.jsonl").read_text().splitlines()]
    assert events==["PRECHECK_OK","ALREADY_VERIFIED_SKIPPED","STARTED","VERIFIED","BATCH_DONE"]


def test_failure_stops_and_writes_summary(tmp_path,monkeypatch):
    src,_=setup(tmp_path,monkeypatch)
    p=tmp_path/"plan.json"
    m.make_plan(src,["XRPUSDT:2024-10","ETHUSDT:2024-11"],p)
    calls=[]
    def fail(s,mo,*args):
        calls.append((s,mo))
        raise ValueError("Synthetic checksum mismatch")
    monkeypatch.setattr(m,"ingest",fail)
    with pytest.raises(RuntimeError,match="stopped"):
        m.execute(p,m.sha(p),tmp_path/"journal.jsonl",tmp_path/"summary.json",True)
    assert calls==[("XRPUSDT","2024-10")]
    report=json.loads((tmp_path/"summary.json").read_text())
    assert report["failed"]["error_type"]=="ValueError"
    events=[json.loads(x)["event"] for x in (tmp_path/"journal.jsonl").read_text().splitlines()]
    assert events==["PRECHECK_OK","STARTED","FAILED","BATCH_FAILED"]
