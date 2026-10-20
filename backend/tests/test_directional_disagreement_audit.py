"""Pure synthetic invariants; no raw candles, private assets or real folds."""
import numpy as np
import pytest

from mlpipeline.directional_disagreement_audit import (
    disagreement_counts, summarize_cells, crosscheck_counts,
    verify_baseline_report, PROTOCOL,
)
from mlpipeline.directional_rule_benchmark import PROTOCOL as BASELINE_PROTOCOL


def test_six_cases_exhaustive_with_neutral_disagreement():
    truth=np.array([0,1,2, 2,0,1],dtype=int)
    ml=np.array([0,0,0, 2,2,0],dtype=int)
    rev=np.array([0,0,0, 0,0,2],dtype=int)
    result=disagreement_counts(truth,ml,rev,np.ones(6,dtype=bool))
    assert result == {
        "agree_correct":1,
        "agree_wrong_neutral":1,
        "agree_wrong_opposite":1,
        "disagree_ml_correct":1,
        "disagree_reversion_correct":1,
        "disagree_both_wrong_neutral":1,
        "n":6,
    }
    summary=summarize_cells([result,None,result])
    assert summary["rows"]==12
    assert summary["agreement_rate"]==pytest.approx(0.5)
    assert summary["net_gain_ml_points"]==pytest.approx(0.0)
    assert summary["disagreement_neutral_rate"]==pytest.approx(1/3)
    assert summary["ml_win_fraction_among_DISAGREEMENTS_with_true_movement"]==0.5


def test_zero_selected_does_not_fabricate_counts():
    y=np.array([0,2])
    assert disagreement_counts(y,np.array([0,2]),np.array([2,0]),
                               np.array([False,False])) is None
    summary=summarize_cells([None])
    assert summary["rows"]==0
    assert summary["net_gain_ml_points"] is None
    assert summary["fold_records_with_rows"]==0
    with pytest.raises(ValueError):
        disagreement_counts(y,np.array([0,2]),np.array([2,0]),
                            np.array([1,0]))


def test_conservation_matching_baseline_and_drift_rejected():
    y=np.array([0,1,2,2])
    ml=np.array([0,0,2,0])
    rev=np.array([2,2,2,2])
    result=disagreement_counts(y,ml,rev,np.ones(4,dtype=bool))
    assert result["n"]==4
    ml_hits=result["agree_correct"]+result["disagree_ml_correct"]
    rev_hits=result["agree_correct"]+result["disagree_reversion_correct"]
    old={"rules":{
        "lightgbm_30":{
            "all_check":{"n":4,"correct":ml_hits},
            "shared_ml_selected":{"n":4,"correct":ml_hits},
        },
        "reversion_3":{
            "all_check":{"n":4,"correct":rev_hits},
            "shared_ml_selected":{"n":4,"correct":rev_hits},
        },
    }}
    crosscheck_counts(old,{
        "all_check":result,
        "shared_ml_selected":result,
    },"reversion_3")
    old["rules"]["lightgbm_30"]["shared_ml_selected"]["correct"]+=1
    with pytest.raises(ValueError,match="drift"):
        crosscheck_counts(old,{"all_check":result,"shared_ml_selected":result},
                          "reversion_3")


def test_baseline_gate_forbids_unsafe_flags():
    source={"dataset_fingerprint_sha256":"hash"}
    rows=[{"fold_id":i} for i in range(36)]
    old={
        "protocol":BASELINE_PROTOCOL,
        "folds_analyzed":36,
        "n_estimators":150,
        "seed":42,
        "ml_tuning_coverage_target":0.40,
        "source_dataset_fingerprint_sha256":"hash",
        "ADA_loaded_or_scored":False,
        "fold_OOS_test_labels_accessed":False,
        "final90_test_labels_accessed":False,
        "P3_or_forward_models_modified":False,
        "folds":rows,
    }
    assert len(verify_baseline_report(old,source))==36
    old["fold_OOS_test_labels_accessed"]=True
    with pytest.raises(ValueError,match="Unsafe"):
        verify_baseline_report(old,source)
