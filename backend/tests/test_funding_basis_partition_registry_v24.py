"""V24 offline audit and bounded-ingestion guards. No network."""
import json
import pandas as pd
import pytest
import mlpipeline.funding_basis_partition_registry_v24 as m


def test_registry_84_partitions_without_files(tmp_path):
    root=tmp_path/"p0"
    v21=tmp_path/"v21"  # Nonexistent V21 directory: no legacy partition.
    v23=tmp_path/"v23"
    for p in (root,v23):p.mkdir()
    rows=m.registry(root,v21,v23)
    assert len(rows)==84
    assert rows.status.eq("MISSING").all(), rows.loc[rows.status.ne("MISSING")].to_string(index=False)
    assert rows[["symbol","month"]].duplicated().sum()==0


def test_existing_but_empty_legacy_v21_directory_fails_integrity(tmp_path):
    # A present directory with no manifest must NOT be silently marked MISSING.
    root=tmp_path/"p0"
    v21=tmp_path/"v21"
    v23=tmp_path/"v23"
    for p in (root,v21,v23):p.mkdir()
    rows=m.registry(root,v21,v23)
    bad=rows.loc[(rows.symbol=="BTCUSDT") & (rows.month=="2025-06")].iloc[0]
    assert bad.status=="FAILED_INTEGRITY"
    assert bad.error_type=="FileNotFoundError"
    assert rows.status.eq("FAILED_INTEGRITY").sum()==1
    assert rows.status.eq("MISSING").sum()==83


def test_registry_does_not_trust_existing_folder(tmp_path):
    root=tmp_path/"p0";root.mkdir()
    v21=tmp_path/"v21";v21.mkdir()
    v23=tmp_path/"v23";v23.mkdir()
    (v23/"symbol=ETHUSDT"/"month=2025-06").mkdir(parents=True)
    x=m.registry(root,v21,v23)
    row=x[(x.symbol=="ETHUSDT")&(x.month=="2025-06")].iloc[0]
    assert row.status=="FAILED_INTEGRITY"
    assert row.error_type=="FileNotFoundError"


def test_write_report_is_complete_and_no_overwrite(tmp_path):
    rows=pd.DataFrame([{"symbol":symbol,"month":month,"status":"MISSING",
                        "m5":None,"funding_events":None,"error_type":None}
                       for symbol in m.SYMBOLS for month in m.EXPECTED_MONTHS])
    path=tmp_path/"status.json";csv=tmp_path/"status.csv"
    summary=m.write_report(rows,path,csv)
    assert summary["expected"]==84
    assert summary["counts"]["MISSING"]==84
    assert len(pd.read_csv(csv))==84
    with pytest.raises(ValueError,match="NEW"):
        m.write_report(rows,path,csv)


def test_verify_missing_manifest_fails_closed(tmp_path):
    root=tmp_path/"p0";root.mkdir()
    folder=tmp_path/"fake";folder.mkdir()
    with pytest.raises(FileNotFoundError):
        m.verify(folder,"BTCUSDT","2025-06",root,legacy=True)
