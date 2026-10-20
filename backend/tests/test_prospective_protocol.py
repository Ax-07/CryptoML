"""Synthetic-only prospective preregistration safety tests."""
from datetime import datetime, timedelta, timezone
import json

import pytest
from mlpipeline.prospective_protocol import (
    activate, make_registration, register, verify, verify_activation,
)


def now():
    return datetime(2026, 10, 14, 15, 4, tzinfo=timezone.utc)


def example_policy():
    return {
        "thresholds": {
            "H2": {"12-17_UTC": 0.12, "all_day_UTC": 0.18},
            "H6": {"12-17_UTC": 0.14, "all_day_UTC": 0.20},
        },
        "fitted_before_utc": now().isoformat(),
        "training_provenance": {
            f"H{h}": {
                "horizon_bars": h, "feature_schema": "features_v1",
                "feature_count": 30, "label_schema": "vol_scaled_v1",
                "alpha": 1.0, "train_prior_counts": [220, 480, 240],
                "training_data_last_close_utc": "2026-06-30T23:55:00Z",
            } for h in (2, 6)
        },
    }


def setup(tmp_path):
    root = tmp_path / "prospective"
    register(root, now=now())
    h2, h6 = tmp_path / "h2.joblib", tmp_path / "h6.joblib"
    h2.write_bytes(b"model-H2")
    h6.write_bytes(b"model-H6")
    tune = tmp_path / "policy.json"
    tune.write_text(json.dumps(example_policy()), encoding="utf-8")
    return root, h2, h6, tune


def test_registration_uses_only_future_dates_and_frozen_primary():
    reg = make_registration(now=now())
    assert reg["prospective_start_utc"] == "2026-10-16T00:00:00Z"
    assert reg["prospective_end_exclusive_utc"] == "2026-12-11T00:00:00Z"
    assert reg["evaluation_not_before_utc"] == "2026-12-11T01:00:00Z"
    assert reg["protocol"]["primary"]["cohort"] == "12_17_UTC_inclusive_six_hour_block"
    assert reg["status"] == "registered_only_not_active"


def test_registration_is_immutable(tmp_path):
    root = tmp_path / "reg"
    reg = register(root, now=now())
    assert verify(root) == reg
    with pytest.raises(FileExistsError):
        register(root, now=now())
    with (root / "protocol.json").open("ab") as f:
        f.write(b" ")
    with pytest.raises(ValueError, match="SHA256"):
        verify(root)


def test_models_frozen_before_start_and_verified(tmp_path):
    root, h2, h6, tune = setup(tmp_path)
    activated = activate(
        root, model_h2=h2, model_h6=h6,
        threshold_file=tune, now=now() + timedelta(minutes=4),
    )
    assert verify_activation(root) == activated
    with pytest.raises(FileExistsError):
        activate(root, model_h2=h2, model_h6=h6,
                 threshold_file=tune, now=now())
    h2.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="modified"):
        verify_activation(root)


def test_no_activation_after_collection_start(tmp_path):
    root, h2, h6, tune = setup(tmp_path)
    with pytest.raises(ValueError, match="Too late"):
        activate(root, model_h2=h2, model_h6=h6,
                 threshold_file=tune, now=now() + timedelta(days=2))


def test_previously_inspected_final_window_never_trains_models(tmp_path):
    root, h2, h6, tune = setup(tmp_path)
    policy = json.loads(tune.read_text())
    policy["training_provenance"]["H2"]["training_data_last_close_utc"] = "2026-09-29T12:00:00Z"
    tune.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="reserved final"):
        activate(root, model_h2=h2, model_h6=h6,
                 threshold_file=tune, now=now())


def test_invalid_threshold_policy_rejected(tmp_path):
    root, h2, h6, tune = setup(tmp_path)
    policy = json.loads(tune.read_text())
    policy["thresholds"]["H2"]["12-17_UTC"] = 1.2
    tune.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="Thresholds"):
        activate(root, model_h2=h2, model_h6=h6,
                 threshold_file=tune, now=now())


def test_naive_registration_clock_rejected():
    with pytest.raises(ValueError, match="UTC-aware"):
        make_registration(now=datetime(2026, 10, 14, 12))
