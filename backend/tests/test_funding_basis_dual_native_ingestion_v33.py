"""V33 synthetic no-network checks; intentionally no real source ingestion."""
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import pytest

from mlpipeline import funding_basis_dual_native_ingestion_v33 as m
from mlpipeline import funding_basis_daily_mark_reconciliation_v30 as v30
from test_funding_basis_daily_mark_reconciliation_v30 import klines, month_minus_day


def test_frozen_archive_pins_and_urls():
    assert len(m.TARGETS)==4
    assert set(m.MARK_PINN)==set(m.TARGETS)
    assert len(set(m.TARGETS))==4
    for symbol in m.TARGETS:
        assert m.archive_url(symbol,"mark_daily").endswith(f"{symbol}-5m-2026-06-29.zip")
        assert "/monthly/markPriceKlines/" in m.archive_url(symbol,"mark_monthly")


def test_reconstruct_only_native_missing_day():
    joined, mark, rec=m.reconstruct_mark(month_minus_day(),
                                        klines(v30.DAY_BEGIN,v30.DAY_END))
    assert rec["monthly_missing_count"]==288
    assert len(pd.read_csv(io.BytesIO(joined),header=None))==8640
    assert len(mark[0])==8640
    assert mark[1]=="ms"


def test_reconstruct_refuses_any_extra_gap():
    extra=klines(v30.BEGIN,v30.END,omit=[
        v30.DAY_BEGIN,v30.DAY_BEGIN+v30.M5])
    with pytest.raises(ValueError,match="refused"):
        m.reconstruct_mark(extra,klines(v30.DAY_BEGIN,v30.DAY_END))


def test_reconstruct_refuses_changed_day():
    with pytest.raises(ValueError,match="Incorrect M5"):
        m.reconstruct_mark(month_minus_day(),klines(v30.DAY_BEGIN,v30.DAY_END,
                                                   omit=[v30.DAY_BEGIN]))


def test_check_zip_rejects_checksum_and_wrong_pin():
    blob=b"immutable";proof=hashlib.sha256(blob).hexdigest().encode()
    assert m.checked_zip_blob(blob,proof)["zip_sha256"]==hashlib.sha256(blob).hexdigest()
    with pytest.raises(ValueError,match="CHECKSUM"):
        m.checked_zip_blob(blob,b"0"*64)
    with pytest.raises(ValueError,match="approved SHA"):
        m.checked_zip_blob(blob,proof,"0"*64)


def test_frozen_plan_validates_offline_metadata(tmp_path):
    btc={
        "protocol":"p4_v30_btc_june2026_daily_mark_reconciliation_v1",
        "symbol":"BTCUSDT","month":m.MONTH,"daily_source_date":m.MARK_DAY,
        "monthly_archive":{"url":m.archive_url("BTCUSDT","mark_monthly"),
                           "zip_sha256":m.MARK_PINNED["BTCUSDT"][0],"checksum_verified":True},
        "daily_archive":{"url":m.archive_url("BTCUSDT","mark_daily"),
                         "zip_sha256":m.MARK_PINNED["BTCUSDT"][1],"checksum_verified":True},
        "reconciliation":{"status":"SOURCE_RECONCILIATION_CANDIDATE_NOT_PUBLISHED",
          "timeline_reconciliation_eligible_for_review":True,"monthly_missing_count":288,
          "missing_exactly_2026_06_29":True,"daily_m5_count":288,
          "monthly_daily_overlap_count":0,"combined_timeline_complete":True},
    }
    others={"protocol":"p4_v32_remaining_june2026_daily_mark_reconciliation_v1",
            "all_source_candidates_eligible":True,"reports":[]}
    for sym in m.TARGETS[1:]:
        item=dict(btc)
        item["symbol"]=sym
        item["day"]=m.MARK_DAY
        item["monthly_archive"]={"url":m.archive_url(sym,"mark_monthly"),
                                 "zip_sha256":m.MARK_PINNED[sym][0],"checksum_verified":True}
        item["daily_archive"]={"url":m.archive_url(sym,"mark_daily"),
                               "zip_sha256":m.MARK_PINNED[sym][1],"checksum_verified":True}
        others["reports"].append(item)
    p1=tmp_path/"v30.json";p2=tmp_path/"v32.json"
    p1.write_text(json.dumps(btc));p2.write_text(json.dumps(others))
    assert m.check_reports(p1,p2)["v30_sha256"]==m.sha_file(p1)
    others["reports"][0]["daily_archive"]["zip_sha256"]="0"*64
    p2.write_text(json.dumps(others))
    with pytest.raises(ValueError,match="origin/SHA"):
        m.check_reports(p1,p2)


def test_audit_counts_are_explicit_v33():
    rows=[
      {"status":"VERIFIED_V21"},
      *[{"status":"VERIFIED_V23"} for _ in range(79)],
      *[{"status":"VERIFIED_V33"} for _ in range(4)],
    ]
    result=m.audit_counts(rows)
    assert result["total_verified"]==84
    assert result["counts"]["VERIFIED_V33"]==4
    assert result["counts"]["MISSING"]==0
