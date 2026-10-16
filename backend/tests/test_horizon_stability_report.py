"""Synthetic checks: posthoc horizon stability never trains or opens P0."""
import copy

import pytest

from mlpipeline.horizon_stability_report import (
    ASSETS, analyze_horizon_stability,
)


def directional(precision, uplift, total=100, selected=50):
    hits = int(round(precision * selected))
    return {
        "n_total": total,
        "selected": selected,
        "correct_direction_class": hits,
        "directional_precision": hits / selected,
        "uplift_points": uplift,
        "coverage": selected / total,
        "per_direction": {
            "DOWN": {"predicted": 20, "correct": 8},
            "UP": {"predicted": 30, "correct": hits - 8},
        },
    }


def fake_report():
    folds = []
    for index in range(6):
        models = {}
        for horizon in (1, 2, 3, 5, 6, 9, 12):
            p = (0.38 if horizon == 2 else 0.34) + (index % 2) * 0.02
            metrics = directional(p, 4.5 if horizon == 2 else 3.7)
            models[str(horizon)] = {
                "metrics": {"directional": metrics},
                "per_symbol_directional": {
                    asset: copy.deepcopy(metrics) for asset in ASSETS
                },
            }
        folds.append({
            "fold_id": index,
            "validation_start_utc": (
                f"2025-01-{index + 1:02d}T00:00:00+00:00"
            ),
            "horizons": models,
        })
    return {
        "protocol": "cpu_p2_7_horizons_shared_samples_h12_purge_v1",
        "ADA_P0_loaded_or_scored": False,
        "fold_oos_tests_scored": False,
        "final_90_days_scored": False,
        "horizons_predeclared": [1, 2, 3, 5, 6, 9, 12],
        "seen_symbols_only": list(ASSETS),
        "folds": folds,
    }


def test_positive_paired_h2_uplift_and_asset_coverage():
    report = analyze_horizon_stability(fake_report())
    assert report["development_folds"] == 6
    assert report["model_retrained"] is False
    assert report["independent_confirmation"] is False
    assert set(report["by_asset"]) == set(ASSETS)
    comparison = report["overall"]["comparisons"]["H2_vs_H6"]
    assert comparison["folds"] == 6
    assert comparison["wins_precision"] == 6
    assert comparison["wins_uplift"] == 6
    assert comparison["mean_uplift_delta_points"] == pytest.approx(0.8)
    assert report["by_time"]["earlier_half"]["folds"] == 3
    assert report["by_time"]["chronological_block_01"]["folds"] == 6


def test_horizons_must_be_same_sample_size_per_asset():
    source = fake_report()
    source["folds"][0]["horizons"]["2"]["per_symbol_directional"]["BTCUSDT"]["n_total"] = 99
    with pytest.raises(ValueError, match="Misaligned"):
        analyze_horizon_stability(source)


def test_rejects_reserved_scoring_or_malformed_horizons():
    source = fake_report()
    source["final_90_days_scored"] = True
    with pytest.raises(ValueError, match="protected"):
        analyze_horizon_stability(source)
    source = fake_report()
    source["horizons_predeclared"] = [1, 2, 3]
    with pytest.raises(ValueError, match="Unexpected horizon"):
        analyze_horizon_stability(source)
