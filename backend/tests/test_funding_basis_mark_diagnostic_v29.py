"""V29 uses synthetic mark kline CSVs; does not access Binance."""
import hashlib
import json
import pytest
from mlpipeline import funding_basis_mark_diagnostic_v29 as m
from test_funding_basis_spot_diagnostic_v28 import csv_bytes


def test_mark_complete_passes():
    d=m.diagnose_csv(csv_bytes())
    assert d["data_rows"]==8640
    assert d["missing_m5_count"]==0
    assert d["v23_mark_parser_status"]=="PASS"


def test_mark_missing_bars_fail_closed():
    d=m.diagnose_csv(csv_bytes(drop=(100,)))
    assert d["data_rows"]==8639
    assert d["missing_m5_count"]==1
    assert d["v23_mark_parser_status"]=="FAIL"
    assert d["v23_mark_parser_error"]["message"]=="Incorrect M5 count/shape"


def test_permission_required(tmp_path):
    with pytest.raises(ValueError,match="permit-downloads"):
        m.run(tmp_path/"report.json")


def test_invalid_checksum_no_output(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"fetch",lambda url,limit:b"invalid")
    with pytest.raises(ValueError):
        m.run(tmp_path/"report.json",True)
    assert not (tmp_path/"report.json").exists()


def test_valid_checksum_diagnostic_only(tmp_path,monkeypatch):
    blob=b"fakezip"
    digest=hashlib.sha256(blob).hexdigest()
    monkeypatch.setattr(m,"fetch",lambda url,limit:
                        digest.encode() if url.endswith(".CHECKSUM") else blob)
    monkeypatch.setattr(m,"verified_csv",lambda blob:csv_bytes(drop=(0,1)))
    result=m.run(tmp_path/"report.json",True)
    assert result["geometry"]["missing_m5_count"]==2
    assert result["geometry"]["v23_mark_parser_status"]=="FAIL"
    assert json.loads((tmp_path/"report.json").read_text())["sha256_verified"]
