"""Read-only P3 preflight guards. Never read real user P0 or ADA."""
from datetime import datetime, timezone

import pytest

from mlpipeline.prospective_preflight import (
    _assert_provenance, validate_packs, EXPECTED_PACK_SHA256,
)
from mlpipeline.prospective_protocol import register


def test_preflight_rejects_missing_real_model_packs_before_joblib(tmp_path):
    protocol=tmp_path/"p3"
    register(protocol,now=datetime(2026,10,14,tzinfo=timezone.utc))
    with pytest.raises(FileNotFoundError):
        validate_packs(tmp_path/"missing",protocol)


def test_preflight_detects_wrong_hash_before_deserialization(tmp_path, monkeypatch):
    protocol=tmp_path/"p3"
    register(protocol,now=datetime(2026,10,14,tzinfo=timezone.utc))
    packs=tmp_path/"packs"
    packs.mkdir()
    (packs/"h2_model.joblib").write_bytes(b"not a trusted joblib")
    (packs/"h6_model.joblib").write_bytes(b"not a trusted joblib")
    (packs/"threshold_policy.json").write_text("{}",encoding="utf-8")
    # No dangerous pickle deserialization should be reached.
    import mlpipeline.prospective_preflight as module
    monkeypatch.setattr(module,"_load_packs",
                        lambda x: pytest.fail("Must verify digest before joblib"))
    with pytest.raises(ValueError,match="operator-frozen SHA256"):
        validate_packs(packs,protocol)


def test_preflight_rejects_policy_after_frozen_start():
    base={
        "fitted_before_utc":"2026-10-17T00:00:00Z",
        "thresholds":{
            "H2":{"12-17_UTC":0.1,"all_day_UTC":0.2},
            "H6":{"12-17_UTC":0.1,"all_day_UTC":0.2}
        },
        "training_provenance":{},
    }
    with pytest.raises(ValueError,match="not fitted before"):
        _assert_provenance(
            base,
            start=datetime(2026,10,16,tzinfo=timezone.utc),
            now=datetime(2026,10,18,tzinfo=timezone.utc),
        )


def test_frozen_expected_hashes_are_full_sha256():
    assert set(EXPECTED_PACK_SHA256)=={"H2","H6"}
    assert all(len(v)==64 and int(v,16)>=0 for v in EXPECTED_PACK_SHA256.values())
