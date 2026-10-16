"""Protect final holdout boundaries, split purity, and frozen protocol audit."""
import copy

import numpy as np
import pandas as pd
import pytest

from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.final_holdout import (
    FINAL_SETTINGS, _partitions, audit_final_protocol,
    evaluate_frozen_final,
)


def synthetic_examples():
    times = pd.date_range("2025-01-01T00:05:00Z", periods=15 * 288, freq="5min")
    frames = []
    for k, symbol in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT")):
        rng = np.random.default_rng(100 + k)
        frame = pd.DataFrame({
            "symbol": symbol,
            "bar_close_at": times,
            "label_mature_at": times + pd.Timedelta(minutes=30),
            "target": rng.integers(0, 3, size=len(times)),
        })
        for feature in FEATURE_COLUMNS:
            frame[feature] = rng.normal(size=len(times)).astype("float32")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def settings():
    return {
        **FINAL_SETTINGS,
        "months": 1, "train_days": 4, "validation_days": 4,
        "final_holdout_days": 2, "n_estimators": 20,
    }


def test_audit_does_not_score_holdout_and_preserves_purging():
    examples = synthetic_examples()
    cfg = settings()
    audit = audit_final_protocol(examples, "fake_fingerprint", cfg)
    assert audit["status"] == "SEALED_AUDIT_ONLY"
    assert "final_metrics" not in audit
    bounds, parts = _partitions(examples, cfg)
    assert all(len(part) > 100 for part in parts.values())
    assert parts["ada_final"].symbol.eq("ADAUSDT").all()
    assert not parts["train"].symbol.eq("ADAUSDT").any()
    assert all(
        parts[name].label_mature_at.lt(bounds[boundary]).all()
        for name, boundary in [
            ("train", "validation_start"),
            ("early_stopping", "early_stop_end"),
            ("decision_tuning", "decision_tune_end"),
            ("validation_check", "final_start"),
            ("seen_final", "final_end"),
            ("ada_final", "final_end"),
        ]
    )
    assert audit["partition_counts"]["ada_final"] > 0


def test_unseal_requires_exact_prior_audit():
    examples = synthetic_examples()
    cfg = settings()
    audit = audit_final_protocol(examples, "fake_fingerprint", cfg)
    wrong = copy.deepcopy(audit)
    wrong["parameters"]["seed"] = 1
    with pytest.raises(ValueError, match="Audit differs"):
        evaluate_frozen_final(examples, wrong, "fake_fingerprint", cfg)


def test_frozen_final_produces_independent_seen_and_ada_scores():
    examples = synthetic_examples()
    cfg = settings()
    audit = audit_final_protocol(examples, "fake_fingerprint", cfg)
    report = evaluate_frozen_final(examples, audit, "fake_fingerprint", cfg)
    assert report["status"] == "FINAL_UNSEALED_EVALUATED"
    assert set(report["final_metrics"]) == {"seen_final", "ada_final"}
    for split in ("seen_final", "ada_final"):
        scored = report["final_metrics"][split]
        n = audit["partition_counts"][split]
        for candidate in ("lightgbm", "logistic"):
            result = scored[candidate]
            assert result["probability_metrics"]["n"] == n
            assert result["adjusted_decision"]["n"] == n
        assert scored["class_prior_baseline"]["probability_metrics"]["n"] == n
    assert set(report["final_metrics"]["ada_final"]["per_symbol"]) == {"ADAUSDT"}
