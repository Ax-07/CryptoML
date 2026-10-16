"""Unit tests for no-fit diagnosis of prior P2 directional false positives."""
import copy

import pytest

from mlpipeline.detection_error_audit import _audit_metrics, audit
from mlpipeline.feature_reduction_diagnostics import PROTOCOL as SOURCE_PROTOCOL


def cell(selected=10, correct=5, neutral=3, total=25):
    random_ref=0.3
    return {
        "selected":selected,"n_total":total,
        "correct_direction_class":correct,
        "neutral_among_selected":neutral,
        "random_reference_precision":random_ref,
        "uplift_points":100*(correct/selected-random_ref),
        "coverage":selected/total,
        "per_direction":{
            "DOWN":{"predicted":6,"correct":3,"support":10},
            "UP":{"predicted":4,"correct":2,"support":8},
        },
    }


def test_errors_partition_into_neutral_and_opposite():
    result=_audit_metrics([cell(),cell()])
    assert result["selected"]==20
    assert result["selected_correct"]==10
    assert result["selected_truth_neutral"]==6
    assert result["selected_truth_opposite"]==4
    assert result["fraction_of_wrong_decisions_neutral"]==pytest.approx(0.6)
    assert result["uplift_selected_weighted_points"]==pytest.approx(20)
    assert result["directional"]["DOWN"]["precision"]==pytest.approx(0.5)


def test_reject_corrupted_counts():
    sample=cell()
    sample["neutral_among_selected"]=8
    with pytest.raises(ValueError,match="Inconsistent"):
        _audit_metrics([sample])


def full_report():
    base=cell()
    by_asset={s:copy.deepcopy(base) for s in
              ("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")}
    # Group totals = sum of four assets.
    pooled={
        "selected":40,"n_total":100,"correct_direction_class":20,
        "neutral_among_selected":12,"random_reference_precision":0.3,
        "uplift_points":20.0,"coverage":0.4,
        "per_direction":{
            "DOWN":{"predicted":24,"correct":12,"support":40},
            "UP":{"predicted":16,"correct":8,"support":32},
        }
    }
    return {
        "protocol":SOURCE_PROTOCOL,"mode":"individual",
        "research_status":"EXPLORATORY_PREVIOUSLY_INSPECTED_DEV_CHECKS",
        "ADA_loaded_or_scored":False,"fold_OOS_scored":False,
        "final_90_days_scored":False,"frozen_P3_models_modified":False,
        "folds_analyzed":36,
        "folds":[{
            "fold_id":idx,
            "horizons":{
                key:{
                    "baseline_30":{
                        "feasible":True,
                        "check_metrics":copy.deepcopy(pooled),
                        "by_asset":copy.deepcopy(by_asset),
                    }
                } for key in ("H2","H6")
            },
        } for idx in range(36)]
    }


def test_full_existing_baseline_audit_no_new_training():
    result=audit(full_report())
    assert result["no_new_training"] is True
    assert result["horizons"]["H2"]["overall"]["selected_truth_opposite"]==288
    assert result["horizons"]["H6"]["feasible_folds"]==36


def test_reject_sealed_or_partial_reports():
    source=full_report()
    source["ADA_loaded_or_scored"]=True
    with pytest.raises(ValueError,match="safety"):
        audit(source)
    source=full_report()
    source["folds"].pop()
    with pytest.raises(ValueError):
        audit(source)
