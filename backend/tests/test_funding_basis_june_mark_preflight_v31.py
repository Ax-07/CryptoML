"""V31 diagnostics use mocked archived CSVs; never touch real datasets."""
import hashlib
import json
import pytest
from mlpipeline import funding_basis_june_mark_preflight_v31 as m
from test_funding_basis_spot_diagnostic_v28 import csv_bytes


def test_permission(tmp_path):
    with pytest.raises(ValueError,match="permit-downloads"):
        m.run(tmp_path/"report.json")


def test_three_symbols_report_read_only(tmp_path,monkeypatch):
    def fake_archive(url):
        assert "/monthly/markPriceKlines/" in url
        name=url.rsplit("/",1)[-1].split("-")[0]
        gap=(20,) if name=="SOLUSDT" else ()
        blob=csv_bytes(drop=gap)
        return blob,{"url":url,"zip_sha256":hashlib.sha256(blob).hexdigest(),
                     "checksum_verified":True}
    monkeypatch.setattr(m,"verified_archive",fake_archive)
    path=tmp_path/"report.json"
    result=m.run(path,True)
    assert [r["symbol"] for r in result["rows"]]==list(m.SYMBOLS)
    assert result["rows"][0]["geometry"]["v23_mark_parser_status"]=="PASS"
    assert result["rows"][1]["geometry"]["missing_m5_count"]==1
    assert result["rows"][2]["geometry"]["v23_mark_parser_status"]=="PASS"
    assert json.loads(path.read_text())["no_partition_written"] is True
    with pytest.raises(ValueError,match="new output"):
        m.run(path,True)


def test_failure_does_not_publish_partial_report(tmp_path,monkeypatch):
    state={"n":0}
    def failing(url):
        state["n"]+=1
        if state["n"]==2:raise ValueError("bad sha")
        return csv_bytes(),{"checksum_verified":True}
    monkeypatch.setattr(m,"verified_archive",failing)
    path=tmp_path/"report.json"
    with pytest.raises(ValueError,match="bad sha"):
        m.run(path,True)
    assert not path.exists()
