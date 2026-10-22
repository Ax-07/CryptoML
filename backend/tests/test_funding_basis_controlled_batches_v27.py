"""V27 offline gating, integrity, restart and failure persistence."""
import json
import pandas as pd
import pytest
import mlpipeline.funding_basis_controlled_batches_v27 as m


def setup(tmp_path,monkeypatch):
    dirs=[tmp_path/name for name in ("p0","v21","v23")]
    for p in dirs:p.mkdir()
    cj=tmp_path/"coverage.json";cc=tmp_path/"coverage.csv"
    cj.write_text("{}");cc.write_text("test\n")
    monkeypatch.setattr(m,"validate_coverage",lambda *x:{"json_sha256":"a"*64,"csv_sha256":"b"*64})
    state={}
    def fake_registry(*x):
        return pd.DataFrame([{"symbol":s,"month":month,"status":state.get((s,month),"MISSING")}
                             for s in m.SYMBOLS for month in m.EXPECTED_MONTHS])
    # Mock audit's output; it is regression tested independently by V24.
    monkeypatch.setattr(m,"audit",fake_registry)
    return (*dirs,cj,cc),state


def test_targets_are_strictly_bounded_and_unique():
    assert len(m.parse_targets(["BTCUSDT:2024-11","ETHUSDT:2024-12",
                                "XRPUSDT:2024-11","SOLUSDT:2024-11"]))==4
    for bad in ([],["ETHUSDT:2024-10"]*2,
                ["ETHUSDT:2024-10"]*5,["ADAUSDT:2024-10"],
                ["BTCUSDT:2026-07"],["BTCUSDT:2025-06"]):
        with pytest.raises(ValueError):m.parse_targets(bad)


def test_plan_is_offline_and_refuses_overwrite(tmp_path,monkeypatch):
    paths,_=setup(tmp_path,monkeypatch)
    out=tmp_path/"plan.json"
    report=m.plan(paths,["BTCUSDT:2024-11","ETHUSDT:2024-12"],out)
    assert report["initial_missing"]==84 and report["max_new"]==2
    with pytest.raises(ValueError,match="NEW"):
        m.plan(paths,["SOLUSDT:2024-11"],out)


def test_execute_requires_exact_sha(tmp_path,monkeypatch):
    paths,_=setup(tmp_path,monkeypatch)
    out=tmp_path/"plan.json"
    m.plan(paths,["XRPUSDT:2024-11"],out)
    with pytest.raises(ValueError,match="--permit-downloads"):
        m.execute(out,"bad",tmp_path/"journal",tmp_path/"summary")
    with pytest.raises(ValueError,match="SHA256"):
        m.execute(out,"0"*64,tmp_path/"journal",tmp_path/"summary",True)


def test_resume_skips_verified_and_ingests_next(tmp_path,monkeypatch):
    paths,state=setup(tmp_path,monkeypatch)
    out=tmp_path/"plan.json"
    m.plan(paths,["ETHUSDT:2024-11","SOLUSDT:2024-11"],out)
    state[("ETHUSDT","2024-11")]="VERIFIED_V23"
    # The synthetic already-verified partition must contain a manifest:
    # V27 hashes it during the preflight to detect tampering.
    v23=paths[2]
    existing=v23/"symbol=ETHUSDT"/"month=2024-11"
    existing.mkdir(parents=True)
    (existing/"manifest.json").write_text(
        json.dumps({"symbol":"ETHUSDT","month":"2024-11"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(m.shutil,"disk_usage",lambda *args:type("Usage",(),{"free":m.MIN_FREE+1})())
    events=[]
    def ingest(s,month,*args):
        events.append((s,month))
        folder=v23/f"symbol={s}"/f"month={month}"
        folder.mkdir(parents=True)
        (folder/"manifest.json").write_text(
            json.dumps({"symbol":s,"month":month}),encoding="utf-8",
        )
        state[(s,month)]="VERIFIED_V23"
    monkeypatch.setattr(m,"ingest",ingest)
    # Only the full parquet/P0 validator is mocked; the manifest hashes stay real.
    monkeypatch.setattr(m,"verify",lambda *args,**kwargs:None)
    result=m.execute(out,m.sha(out),tmp_path/"journal",tmp_path/"summary",True)
    assert events==[("SOLUSDT","2024-11")]
    assert result["newly_verified"]==1
    kinds=[json.loads(x)["event"] for x in (tmp_path/"journal").read_text().splitlines()]
    assert kinds==["PRECHECK_OK","ALREADY_VERIFIED_SKIPPED","STARTED","VERIFIED","BATCH_DONE"]


def test_disk_low_stops_before_any_get(tmp_path,monkeypatch):
    paths,_=setup(tmp_path,monkeypatch)
    out=tmp_path/"plan.json";m.plan(paths,["SOLUSDT:2024-11"],out)
    monkeypatch.setattr(m.shutil,"disk_usage",lambda *args:type("Usage",(),{"free":0})())
    monkeypatch.setattr(m,"ingest",lambda *args: (_ for _ in ()).throw(AssertionError("must not GET")))
    with pytest.raises(RuntimeError,match="stopped"):
        m.execute(out,m.sha(out),tmp_path/"journal",tmp_path/"summary",True)
    result=json.loads((tmp_path/"summary").read_text())
    assert result["failed"]["error_type"]=="ValueError"


def test_first_ingestion_failure_stops_second(tmp_path,monkeypatch):
    paths,_=setup(tmp_path,monkeypatch)
    out=tmp_path/"plan.json";m.plan(paths,["SOLUSDT:2024-11","XRPUSDT:2024-11"],out)
    monkeypatch.setattr(m.shutil,"disk_usage",lambda *args:type("Usage",(),{"free":m.MIN_FREE+1})())
    calls=[]
    def broken(s,month,*args):
        calls.append((s,month))
        raise ValueError("Synthetic checksum mismatch")
    monkeypatch.setattr(m,"ingest",broken)
    with pytest.raises(RuntimeError,match="stopped"):
        m.execute(out,m.sha(out),tmp_path/"journal",tmp_path/"summary",True)
    assert calls==[("SOLUSDT","2024-11")]
    assert json.loads((tmp_path/"summary").read_text())["failed"]["error_type"]=="ValueError"


def test_ingest_without_manifest_fails_closed_and_preserves_journal(tmp_path,monkeypatch):
    paths,state=setup(tmp_path,monkeypatch)
    planned=tmp_path/"plan.json"
    m.plan(paths,["XRPUSDT:2024-11"],planned)
    monkeypatch.setattr(
        m.shutil,"disk_usage",
        lambda *args:type("Usage",(),{"free":m.MIN_FREE+1})(),
    )
    calls=[]
    def incomplete_ingest(symbol,month,*args):
        calls.append((symbol,month))
        # Explicitly do NOT publish a manifest: validation must reject this.
    monkeypatch.setattr(m,"ingest",incomplete_ingest)
    monkeypatch.setattr(m,"verify",lambda *args,**kwargs:None)
    with pytest.raises(RuntimeError,match="Batch stopped"):
        m.execute(planned,m.sha(planned),tmp_path/"journal",tmp_path/"summary",True)
    assert calls==[("XRPUSDT","2024-11")]
    result=json.loads((tmp_path/"summary").read_text(encoding="utf-8"))
    assert result["failed"]["error_type"]=="FileNotFoundError"
    assert result["newly_verified"]==0
    events=[json.loads(x)["event"] for x in
            (tmp_path/"journal").read_text(encoding="utf-8").splitlines()]
    assert events==["PRECHECK_OK","STARTED","FAILED","BATCH_FAILED"]
