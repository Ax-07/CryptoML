"""V20 offline, synthetic schema, integrity, and network-gating tests."""
import io
import json
import zipfile
import hashlib
import pandas as pd
import pytest

import mlpipeline.funding_archive_schema_v20 as m


def rows(data, columns=("calc_time","funding_interval_hours","last_funding_rate")):
    csv=",".join(columns)+"\n"
    csv+="1748736000000,8,0.0001\n1748764800000,8,-0.0002\n"
    return csv.encode()


def test_known_archive_schema_and_fraction_units():
    result=m.funding_schema(rows(None))
    assert result["status"]=="FUNDING_SCHEMA_AND_CHECKSUM_OK"
    assert result["timestamp_unit"]=="ms"
    assert result["count"]==2
    assert result["rate_field"]=="last_funding_rate"
    assert result["interval_field"]=="funding_interval_hours"


def test_rest_alias_is_supported():
    b=b"fundingTime,fundingRate\n1748736000000,0.0001\n1748764800000,-0.0002\n"
    result=m.funding_schema(b)
    assert result["rate_field"]=="fundingRate"
    assert result["count"]==2


def test_unknown_schema_requires_review():
    b=b"symbol,rate,timestamp\nBTCUSDT,0.0001,1748736000000\n"
    assert m.funding_schema(b)["status"]=="SCHEMA_REVIEW_REQUIRED"


@pytest.mark.parametrize("value",["100%", "NaN", "0.5", "xyz"])
def test_bad_rate_rejected(value):
    b=f"fundingTime,fundingRate\n1748736000000,{value}\n".encode()
    with pytest.raises(ValueError,match="rate"):
        m.funding_schema(b)


def test_non_monotonic_time_rejected():
    b=b"fundingTime,fundingRate\n1748764800000,0.0001\n1748736000000,0.0002\n"
    with pytest.raises(ValueError,match="non-monotonic"):
        m.funding_schema(b)


def test_verified_csv_refuses_zip_path_traversal():
    out=io.BytesIO()
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("../malicious.csv",rows(None))
    with pytest.raises(ValueError,match="Unsafe"):
        m.verified_csv(out.getvalue())


def test_archive_sha_mismatch_fails_before_schema(tmp_path,monkeypatch):
    reference={"protocol":"p4_v19_btc_june2025_archive_integrity_pilot_v1",
       "sample":"BTCUSDT June 2025",
       "files":{k:{"status":"CHECKSUM_AND_SCHEMA_OK","sha256_verified":True}
                for k in ("spot","perp","mark")},
       "spot_perp_mark_alignment":{"common_m5_opens":8640}}
    reference["files"]["funding"]={"status":"SCHEMA_REVIEW_REQUIRED","sha256_verified":True}
    path=tmp_path/"v19.json"
    path.write_text(json.dumps(reference),encoding="utf-8")
    monkeypatch.setattr(m,"fetch",lambda url,limit: b"wrong zip" if url.endswith(".zip") else b"0"*64)
    with pytest.raises(ValueError,match="checksum mismatch"):
        m.run(path)
