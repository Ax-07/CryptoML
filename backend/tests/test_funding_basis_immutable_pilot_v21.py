"""V21 offline validation tests; no public API, no orders, no market outcomes."""
import hashlib
import io
import json
import zipfile
import numpy as np
import pandas as pd
import pytest

from mlpipeline.funding_basis_immutable_pilot_v21 import _kline,_funding,_reference,run,N


def zipped(text):
    b=io.BytesIO()
    with zipfile.ZipFile(b,"w",compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("sample.csv",text)
    return b.getvalue()


def test_market_parses_spot_microseconds_and_perp_milliseconds():
    dates=pd.date_range("2025-06-01T00:00:00Z",periods=N,freq="5min")
    for unit in ("ms","us"):
        mul=1_000 if unit=="ms" else 1_000_000
        period=300*mul
        text="".join(f"{int(t.timestamp()*mul)},100,101,99,100,12,{int(t.timestamp()*mul)+period-1}\n"
                     for t in dates)
        time,parsed,prices=_kline(zipped(text),"spot")
        assert parsed==unit
        assert len(time)==N
        assert np.all(prices["close"]==100)


def test_reject_wrong_inclusive_close_timestamp():
    dates=pd.date_range("2025-06-01T00:00:00Z",periods=N,freq="5min")
    text="".join(f"{int(t.timestamp()*1000)},100,101,99,100,12,{int(t.timestamp()*1000)+300000}\n"
                 for t in dates)
    with pytest.raises(ValueError,match="close_time"):
        _kline(zipped(text),"perp")


def test_funding_rates_and_month_kept_as_events():
    times=pd.date_range("2025-06-01T00:00:00Z",periods=90,freq="8h")
    data="calc_time,funding_interval_hours,last_funding_rate\n"
    data+="".join(f"{int(t.timestamp()*1000)},8,0.0001\n" for t in times)
    result=_funding(zipped(data))
    assert len(result)==90
    assert result.funding_rate_fraction.iloc[0]==pytest.approx(0.0001)
    assert result.funding_at.iloc[0]==pd.Timestamp("2025-06-01T00:00:00Z")


def test_reference_report_chain_guards(tmp_path):
    v19={"protocol":"p4_v19_btc_june2025_archive_integrity_pilot_v1",
         "sample":"BTCUSDT June 2025",
         "spot_perp_mark_alignment":{"common_m5_opens":N},
         "files":{k:{"sha256_verified":True,"status":"CHECKSUM_AND_SCHEMA_OK","count":N}
                  for k in ("spot","perp","mark")}}
    v19["files"]["funding"]={"sha256_verified":True,"status":"SCHEMA_REVIEW_REQUIRED"}
    a=tmp_path/"v19.json";a.write_text(json.dumps(v19),encoding="utf-8")
    v20={"protocol":"p4_v20_funding_archive_schema_only_v1",
         "v19_report_sha256":hashlib.sha256(a.read_bytes()).hexdigest(),
         "funding":{"status":"FUNDING_SCHEMA_AND_CHECKSUM_OK","count":90},
         "funding_zip_sha256":"a"*64}
    b=tmp_path/"v20.json";b.write_text(json.dumps(v20),encoding="utf-8")
    assert _reference(a,b)["v19"]==hashlib.sha256(a.read_bytes()).hexdigest()
    v20["funding"]["count"]=89
    b.write_text(json.dumps(v20),encoding="utf-8")
    with pytest.raises(ValueError,match="lineage"):
        _reference(a,b)


def test_refuse_to_overwrite_or_write_inside_p0(tmp_path):
    dest=tmp_path/"existing";dest.mkdir()
    with pytest.raises(ValueError,match="New destination"):
        run(tmp_path/"none",tmp_path/"none",tmp_path/"p0",dest)


def test_bad_destination_is_rejected_before_any_report_access_or_network(tmp_path, monkeypatch):
    import mlpipeline.funding_basis_immutable_pilot_v21 as m
    def forbidden(*args, **kwargs):
        raise AssertionError("No lineage access or network for invalid destination")
    monkeypatch.setattr(m, "_reference", forbidden)
    monkeypatch.setattr(m, "fetch", forbidden)
    root=tmp_path/"p0"
    root.mkdir()
    with pytest.raises(ValueError, match="New destination"):
        m.run(tmp_path/"absent19", tmp_path/"absent20", root, tmp_path/"p0"/"child")
    with pytest.raises(ValueError, match="New destination"):
        m.run(tmp_path/"absent19", tmp_path/"absent20", root, tmp_path)


def test_mark_price_has_no_trade_volume_field():
    dates=pd.date_range("2025-06-01T00:00:00Z", periods=N, freq="5min")
    text="".join(
        f"{int(t.timestamp()*1000)},100,101,99,100,0,"
        f"{int(t.timestamp()*1000)+299999}\n" for t in dates
    )
    _,_,values=_kline(zipped(text),"mark")
    assert set(values)=={"open","high","low","close"}
