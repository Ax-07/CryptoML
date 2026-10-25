"""V35 synthetic-only methodology audit; no archive or holdout access."""
import pandas as pd
import pytest
from mlpipeline import funding_basis_walkforward_preflight_v35 as m


def test_calendar_is_frozen_and_finite():
    folds=m.make_calendar_folds()
    assert 5 < len(folds) < 100
    assert folds[0]["train_start_utc"]=="2024-10-01T00:00:00+00:00"
    assert all(pd.Timestamp(f["test_end_exclusive_utc"])<=m.INTERNAL_RESERVE_START for f in folds)
    assert all(pd.Timestamp(f["windows"]["test"]["last_label_maturity_utc"])
               < pd.Timestamp(f["test_end_exclusive_utc"]) for f in folds)


def test_fold_test_windows_do_not_overlap():
    folds=m.make_calendar_folds()
    for prev,cur in zip(folds,folds[1:]):
        assert pd.Timestamp(prev["test_end_exclusive_utc"])<=pd.Timestamp(cur["test_start_utc"])


def test_embargo_and_label_maturity_strict():
    folds=m.make_calendar_folds()
    f=folds[0]
    start=pd.Timestamp(f["test_start_utc"])
    w=f["windows"]["test"]
    assert pd.Timestamp(w["first_candidate_close_utc"])==start+6*m.M5
    assert pd.Timestamp(w["last_label_maturity_utc"])<pd.Timestamp(f["test_end_exclusive_utc"])


def test_short_window_fails_closed():
    t=m.BEGIN
    with pytest.raises(ValueError,match="Empty"):
        m._expected_closes(t,t+6*m.M5,embargo_bars=6)


def test_reject_naive_timestamps():
    with pytest.raises(ValueError,match="UTC"):
        m._expected_closes(pd.Timestamp("2025-01-01"),m.CUTOFF,embargo_bars=0)


def test_legacy_feature_and_funding_asof_are_blocked():
    r=m.compatibility_report()
    assert "BLOCKED" in r["feature_matrix_status"]
    assert "BLOCKED" in r["funding_asof_status"]
    assert "trade_count" in r["missing_legacy_feature_inputs"]


def test_no_overwrite_and_wrong_provenance(tmp_path):
    old=tmp_path/"existing.json"
    old.write_text("original")
    with pytest.raises(ValueError,match="distinct NEW"):
        m.run(tmp_path/"freeze","0"*64,old,tmp_path/"out.csv")
    assert old.read_text()=="original"
    with pytest.raises(ValueError,match="approved"):
        m.run(tmp_path/"freeze","0"*64,tmp_path/"out.json",tmp_path/"out.csv")


def test_schedule_has_no_prices_or_maturity_leak():
    rows=m.schedule_report(m.make_calendar_folds())
    assert len(rows)==len(m.make_calendar_folds())*3
    assert all(not any(x in row for x in ("target","feature","price","pnl","prediction"))
               for row in rows)
    for row in rows:
        assert pd.Timestamp(row["last_label_maturity_utc"])<pd.Timestamp(row["window_end_exclusive_utc"])


def test_holdout_and_june_are_both_sealed():
    assert m.INTERNAL_RESERVE_START<pd.Timestamp("2026-07-01T00:00:00Z")
    assert max(pd.Timestamp(f["test_end_exclusive_utc"]) for f in m.make_calendar_folds())<=m.INTERNAL_RESERVE_START
