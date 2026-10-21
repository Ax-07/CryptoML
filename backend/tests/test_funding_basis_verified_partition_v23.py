"""V23 synthetic offline tests, no REST or P0 mutations."""
import io
import json
import zipfile
import numpy as np
import pandas as pd
import pytest
import mlpipeline.funding_basis_verified_partition_v23 as m


def archive(payload):
    buff=io.BytesIO()
    with zipfile.ZipFile(buff,"w",compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("example.csv",payload)
    return m.verified_csv(buff.getvalue())


@pytest.mark.parametrize("month,unit",[
    ("2024-10","ms"),("2025-06","us"),("2026-06","ms")])
def test_month_aware_m5_parsing(month,unit):
    first=pd.Timestamp(month+"-01T00:00:00Z")
    last=first+pd.offsets.MonthBegin(1)
    dates=pd.date_range(first,last,freq="5min",inclusive="left")
    mult={"ms":1000,"us":1000000}[unit]
    lines="".join(f"{int(t.timestamp()*mult)},100,101,99,100,10,"
                  f"{int(t.timestamp()*mult)+300*mult-1}\n" for t in dates)
    times,seen,prices=m.parse_kline(archive(lines),"perp",first,last)
    assert len(times)==len(dates)
    assert seen==unit
    assert np.all(prices["close"]==100)


def test_variable_funding_count_month_independent():
    first=pd.Timestamp("2024-10-01T00:00:00Z")
    last=pd.Timestamp("2024-11-01T00:00:00Z")
    text="calc_time,funding_interval_hours,last_funding_rate\n"
    for t in pd.date_range(first,periods=4,freq="8h"):
        text+=f"{int(t.timestamp()*1000)},8,-0.0002\n"
    frame=m.parse_funding(archive(text),first,last)
    assert len(frame)==4
    assert frame.funding_rate_fraction.iloc[0]==pytest.approx(-0.0002)


def test_reject_mixed_ms_us_and_missing_m5():
    begin=pd.Timestamp("2025-06-01T00:00:00Z")
    end=pd.Timestamp("2025-07-01T00:00:00Z")
    with pytest.raises(ValueError,match="Mixed"):
        m.epoch(pd.Series([1748736000000,1748736300000000]),begin,end)
    with pytest.raises(ValueError,match="count"):
        m.parse_kline(archive("1748736000000,100,101,99,100,10,1748736299999\n"),
                      "spot",begin,end)


def test_coverage_csv_identity_and_reject_incomplete(tmp_path):
    from mlpipeline.funding_basis_archive_coverage_v22 import inventory,PROTOCOL
    rows=[{"symbol":s,"month":mo,"kind":kind,"status":"HEAD_OK_UNVERIFIED"}
          for s,mo,kind,_ in inventory("full")]
    csv=tmp_path/"coverage.csv"
    pd.DataFrame(rows).to_csv(csv,index=False)
    ref=tmp_path/"coverage.json"
    ref.write_text(json.dumps({"protocol":PROTOCOL,"mode":"full",
                               "expected":336,"head_200":336}),encoding="utf-8")
    assert "csv_sha256" in m.validate_coverage(ref,csv)
    pd.DataFrame(rows[:-1]).to_csv(csv,index=False)
    with pytest.raises(ValueError):
        m.validate_coverage(ref,csv)


def test_output_rejected_before_coverage_or_download(tmp_path,monkeypatch):
    def deny(*args,**kwargs):
        raise AssertionError("No metadata read or network on unsafe output")
    monkeypatch.setattr(m,"validate_coverage",deny)
    monkeypatch.setattr(m,"fetch",deny)
    root=tmp_path/"p0";root.mkdir()
    with pytest.raises(ValueError,match="output root"):
        m.ingest("ETHUSDT","2025-06",root,root,tmp_path/"missing",tmp_path/"missing")
    with pytest.raises(ValueError,match="Outside"):
        m.ingest("ADAUSDT","2025-06",root,tmp_path/"out",tmp_path/"missing",tmp_path/"missing")
