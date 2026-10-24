"""No-network V32 tests."""
import hashlib
import json
import pytest
from mlpipeline import funding_basis_remaining_daily_mark_reconciliation_v32 as m
from test_funding_basis_daily_mark_reconciliation_v30 import klines, month_minus_day
from mlpipeline import funding_basis_daily_mark_reconciliation_v30 as v30


def test_urls_exact():
    monthly,daily=m.urls("ETHUSDT")
    assert "/monthly/markPriceKlines/ETHUSDT/" in monthly
    assert daily.endswith("/ETHUSDT-5m-2026-06-29.zip")
    with pytest.raises(ValueError,match="outside"):
        m.urls("BTCUSDT")


def test_permission_and_new_path(tmp_path):
    with pytest.raises(ValueError,match="permit-downloads"):
        m.run(tmp_path/"report.json")


def test_three_verified_reconciliations(tmp_path,monkeypatch):
    def fake_verified(url):
        is_daily="/daily/" in url
        symbol=next(s for s in m.SYMBOLS if f"/{s}/" in url)
        blob=(klines(v30.DAY_BEGIN,v30.DAY_END) if is_daily
              else month_minus_day())
        sha=(hashlib.sha256(blob).hexdigest() if is_daily
             else m.EXPECTED_MONTHLY_SHA256[symbol])
        return blob,{"url":url,"zip_sha256":sha,"checksum_verified":True}
    monkeypatch.setattr(m,"verified_archive",fake_verified)
    path=tmp_path/"report.json"
    out=m.run(path,True)
    assert len(out["reports"])==3
    assert out["all_source_candidates_eligible"] is True
    assert all(x["reconciliation"]["monthly_missing_count"]==288 for x in out["reports"])
    assert json.loads(path.read_text())["no_partition_written"] is True
    with pytest.raises(ValueError,match="NEW"):
        m.run(path,True)


def test_archive_revision_fails_no_report(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"verified_archive",lambda url:(b"",{"zip_sha256":"invalid"}))
    path=tmp_path/"report.json"
    with pytest.raises(ValueError,match="changed since V31"):
        m.run(path,True)
    assert not path.exists()


def test_one_missing_day_archive_fails_without_output(tmp_path,monkeypatch):
    def fake_verified(url):
        symbol=next(s for s in m.SYMBOLS if f"/{s}/" in url)
        is_daily="/daily/" in url
        blob=(klines(v30.DAY_BEGIN,v30.DAY_END,omit=[v30.DAY_BEGIN])
              if is_daily else month_minus_day())
        sha=("daily" if is_daily else m.EXPECTED_MONTHLY_SHA256[symbol])
        return blob,{"url":url,"zip_sha256":sha}
    monkeypatch.setattr(m,"verified_archive",fake_verified)
    path=tmp_path/"report.json"
    with pytest.raises(ValueError,match="Incorrect M5"):
        m.run(path,True)
    assert not path.exists()
