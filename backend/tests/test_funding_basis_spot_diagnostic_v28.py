"""V28 diagnostic geometry tests; synthetic CSVs, no network."""
import io
import json
import hashlib
import pandas as pd
import pytest
import mlpipeline.funding_basis_spot_diagnostic_v28 as m


def csv_bytes(drop=(),header=False):
    dates=pd.date_range(m.START,m.END-pd.Timedelta(minutes=5),freq="5min",tz="UTC")
    skip=set(drop)
    buf=io.StringIO()
    if header:buf.write("open_time,open,high,low,close,volume,close_time\n")
    for i,t in enumerate(dates):
        if i in skip:continue
        epoch=int(t.timestamp()*1000)
        buf.write(f"{epoch},1,2,0.5,1.5,10,{epoch+300000-1}\n")
    return buf.getvalue().encode()


def test_complete_spot_pass():
    d=m.examine(csv_bytes())
    assert d["expected_m5"]==8640
    assert d["data_rows"]==8640
    assert d["missing_m5_count"]==0
    assert d["v23_parser_status"]=="PASS"


def test_missing_bars_reported_not_filled():
    d=m.examine(csv_bytes(drop=(3,4),header=True))
    assert d["header_detected"] is True
    assert d["data_rows"]==8638
    assert d["missing_m5_count"]==2
    assert d["v23_parser_status"]=="FAIL"
    assert d["v23_parser_error"]["message"]=="Incorrect M5 count/shape"


def test_duplicate_bar_has_no_false_missing():
    original=csv_bytes()
    data=original+original.splitlines(keepends=True)[-1]
    d=m.examine(data)
    assert d["duplicate_count"]==1
    assert d["v23_parser_status"]=="FAIL"


def test_requires_permission(tmp_path):
    with pytest.raises(ValueError,match="permit-downloads"):
        m.run(tmp_path/"report.json")


def test_sha_guard_does_not_write(tmp_path,monkeypatch):
    import mlpipeline.funding_basis_spot_diagnostic_v28 as diag
    monkeypatch.setattr(diag,"fetch",lambda u,limit:b"not-valid-hash" if u.endswith(".CHECKSUM") else b"testzip")
    with pytest.raises(ValueError):
        diag.run(tmp_path/"report.json",True)
    assert not (tmp_path/"report.json").exists()


def test_run_fake_verified_archive(tmp_path,monkeypatch):
    blob=b"fakezip";digest=hashlib.sha256(blob).hexdigest()
    monkeypatch.setattr(m,"fetch",lambda url,limit:
                        digest.encode() if url.endswith(".CHECKSUM") else blob)
    monkeypatch.setattr(m,"verified_csv",lambda blob:csv_bytes(drop=(10,)))
    result=m.run(tmp_path/"report.json",True)
    assert result["geometry"]["missing_m5_count"]==1
    assert result["geometry"]["v23_parser_status"]=="FAIL"
    assert result["no_p0_write_no_partition_promotion"] is True
    assert json.loads((tmp_path/"report.json").read_text())["sha256_verified"] is True
