"""V34 no-network regression checks, including fail-closed OOS access."""
import json

import pandas as pd
import pytest

from mlpipeline import funding_basis_unified_dev_reader_v34 as m


def test_frozen_calendar_and_schemas():
    assert len(m.EXPECTED_MONTHS) == 21
    assert len(m.SYMBOLS) == 4
    assert len(m.MARKET_FIELDS) == 16
    assert len(m.FUNDING_FIELDS) == 3
    assert len(m.KIND_FILES) == 14
    assert m.EXPECTED_MONTHS[-1] == "2026-06"
    assert m.CUTOFF == pd.Timestamp("2026-07-01T00:00:00Z")


@pytest.mark.parametrize("month",["2026-07","2026-08","2026-09","2024-09"])
def test_holdout_and_outside_calendar_refused_before_disk_access(month):
    reader=m.UnifiedDevelopmentReader(roots=(),identities={})
    with pytest.raises(ValueError,match="Outside approved"):
        reader.read_month("BTCUSDT",month)


def test_unapproved_symbol_refused_before_disk_access():
    reader=m.UnifiedDevelopmentReader(roots=(),identities={})
    with pytest.raises(ValueError,match="Outside approved"):
        reader.read_month("ADAUSDT","2025-01")


def test_unverified_month_refused():
    reader=m.UnifiedDevelopmentReader(roots=(),identities={})
    with pytest.raises(ValueError,match="lacks frozen verification"):
        reader.read_month("BTCUSDT","2026-06")


def test_hash_baseline_cannot_be_replaced(tmp_path):
    freeze=tmp_path/"freeze.json"
    freeze.write_text(json.dumps({"protocol":m.FREEZE_PROTOCOL}))
    with pytest.raises(ValueError,match="preregistered"):
        m._load_and_check_freeze(freeze,"0"*64)
    with pytest.raises(ValueError,match="SHA-256 mismatch"):
        m._load_and_check_freeze(freeze,m.APPROVED_FREEZE_SHA256)


def test_output_guard_fails_before_reading_protected_files(tmp_path):
    existing=tmp_path/"existing.json"
    existing.write_text("keep")
    with pytest.raises(ValueError,match="distinct NEW"):
        m.run(tmp_path/"freeze",m.APPROVED_FREEZE_SHA256,existing,tmp_path/"report.csv")
    assert existing.read_text()=="keep"


def test_frozen_protocol_counts():
    assert m.FROZEN_COUNTS == {
        "VERIFIED_V21":1,"VERIFIED_V23":79,"VERIFIED_V33":4,
        "MISSING":0,"FAILED_INTEGRITY":0,
    }


def test_partition_inventory_paths_are_unique():
    assert len(set(m.KIND_FILES))==14
    assert not any("2026-07" in item for item in m.KIND_FILES)
