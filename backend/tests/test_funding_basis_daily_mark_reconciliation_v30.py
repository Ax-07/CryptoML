"""V30 synthetic-only source reconciliation tests. No live network."""
import hashlib
import io
import json

import pandas as pd
import pytest

from mlpipeline import funding_basis_daily_mark_reconciliation_v30 as m


def klines(start, end, omit=None):
    omit = set(omit) if omit is not None else set()
    dates = pd.date_range(start, end - m.M5, freq="5min")
    buf = io.StringIO()
    for time in dates:
        if time in omit:
            continue
        t = int(time.timestamp() * 1000)
        buf.write(f"{t},100,102,99,101,10,{t+300000-1},0,0,0,0,0\n")
    return buf.getvalue().encode()


def month_minus_day():
    missing = pd.date_range(m.DAY_BEGIN, m.DAY_END - m.M5, freq="5min")
    return klines(m.BEGIN, m.END, omit=missing)


def test_exact_one_day_reconciles_without_writes():
    d = m.reconcile(month_minus_day(), klines(m.DAY_BEGIN, m.DAY_END))
    assert d["monthly_missing_count"] == 288
    assert d["missing_exactly_2026_06_29"]
    assert d["monthly_daily_overlap_count"] == 0
    assert d["combined_timeline_complete"]
    assert d["present_monthly_days_checked"] == 29
    assert d["present_monthly_day_failures"] == []
    assert d["status"] == "SOURCE_RECONCILIATION_CANDIDATE_NOT_PUBLISHED"


def test_missing_other_day_disqualifies():
    data = klines(m.BEGIN, m.END, omit=[m.BEGIN])
    d = m.reconcile(data, klines(m.DAY_BEGIN, m.DAY_END))
    assert d["missing_exactly_2026_06_29"] is False
    assert not d["timeline_reconciliation_eligible_for_review"]


def test_overlap_disqualifies():
    data = klines(m.BEGIN, m.END)
    d = m.reconcile(data, klines(m.DAY_BEGIN, m.DAY_END))
    assert d["monthly_daily_overlap_count"] == 288
    assert not d["timeline_reconciliation_eligible_for_review"]


def test_bad_daily_count_rejected():
    with pytest.raises(ValueError, match="Incorrect M5"):
        m.reconcile(month_minus_day(), klines(m.DAY_BEGIN, m.DAY_END, omit=[m.DAY_BEGIN]))


def test_requires_explicit_permission(tmp_path):
    with pytest.raises(ValueError, match="permit-downloads"):
        m.run(tmp_path/"result.json")


def test_checksum_failure_refuses_report(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "fetch", lambda url, limit: b"wrong")
    with pytest.raises(ValueError, match="CHECKSUM"):
        m.run(tmp_path/"result.json", True)
    assert not (tmp_path/"result.json").exists()


def test_fake_verified_archives_report(tmp_path, monkeypatch):
    monthly = month_minus_day()
    daily = klines(m.DAY_BEGIN, m.DAY_END)
    blobs = {m.BASE+m.PATHS["mark"].format(s=m.SYMBOL,m=m.MONTH): monthly,
             m.BASE+m.DAILY_PATH: daily}
    def fake_fetch(url, limit):
        if url.endswith(".CHECKSUM"):
            return hashlib.sha256(blobs[url[:-9]]).hexdigest().encode()
        return blobs[url]
    monkeypatch.setattr(m, "fetch", fake_fetch)
    monkeypatch.setattr(m, "verified_csv", lambda b: b)
    output = tmp_path/"result.json"
    result = m.run(output, True)
    assert output.exists()
    assert result["reconciliation"]["timeline_reconciliation_eligible_for_review"]
    assert json.loads(output.read_text())["no_partition_written"]
