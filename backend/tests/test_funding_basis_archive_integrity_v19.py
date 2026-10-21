"""V19 wholly offline ZIP/sha/timestamp unit regression tests."""
import hashlib
import io
import zipfile
import pandas as pd
import pytest
from mlpipeline.funding_basis_archive_integrity_v19 import (
    checksum, classify_timestamp, inspect, ZIP_LIMIT,
)


def fixture_zip(unit="ms"):
    times=pd.date_range("2025-06-01T00:00:00Z",periods=3,freq="5min")
    rows=[]
    for t in times:
        value=int(t.timestamp()*({"s":1,"ms":1000,"us":1000000}[unit]))
        rows.append(f"{value},100,101,99,100.5,42\n")
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",compression=zipfile.ZIP_DEFLATED) as f:
        f.writestr("BTCUSDT-5m-2025-06.csv","".join(rows))
    return buf.getvalue()


@pytest.mark.parametrize("unit",["s","ms","us"])
def test_units_and_validated_bars(unit):
    result=inspect(fixture_zip(unit),"spot")
    assert result["status"]=="CHECKSUM_AND_SCHEMA_OK"
    assert result["timestamp_unit"]==unit
    assert result["count"]==3
    assert len(result["open_ms"])==3


def test_sha_format_is_strict():
    digest=hashlib.sha256(b"abc").hexdigest()
    assert checksum((digest+"  file.zip\n").encode())==digest
    with pytest.raises(ValueError,match="CHECKSUM"):
        checksum(b"not-a-sha")


def test_invalid_timestamp_month_rejected():
    with pytest.raises(ValueError,match="June 2025"):
        classify_timestamp(pd.Series([1743465600000]))


def test_zip_member_does_not_extract_arbitrary_path():
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w") as a:
        a.writestr("../evil.csv","1748736000000,100,101,99,100,42\n")
    with pytest.raises(ValueError,match="Unsafe"):
        inspect(buf.getvalue(),"spot")
