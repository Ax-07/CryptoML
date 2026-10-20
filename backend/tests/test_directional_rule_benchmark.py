"""Synthetic unit tests: no access to historical P0 or sealed OOS."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.directional_rule_benchmark import (
    RULES, build_causal_contexts, fixed_mask_metrics, majority_direction,
    prepare_fold_examples, rule_predictions, summarize,
)
from mlpipeline.features import FEATURE_COLUMNS


def _frame():
    n=5
    return pd.DataFrame({
        "ret_1":[0.01,-0.01,0.0,0.03,-0.02],
        "ret_3":[0.02,0.02,-0.02,0.,-0.01],
        "ret_6":[-0.01,0.02,0.02,-0.03,0.],
    })


def test_majority_from_training_only_and_zero_momentum_tie():
    check=_frame()
    probs=np.array([
        [0.4,0.2,0.4],[0.2,0.1,0.7],[0.7,0.2,0.1],
        [0.8,0.1,0.1],[0.1,0.7,0.2],
    ])
    ytrain=np.array([0,0,1,2,2])
    assert majority_direction(ytrain)==0
    pred=rule_predictions(check,ytrain,probs)
    assert set(pred)==set(RULES)
    assert pred["majority_train"].tolist()==[0]*5
    assert pred["momentum_1"].tolist()==[2,0,0,2,0]
    assert pred["reversion_1"].tolist()==[0,2,0,0,2]
    assert pred["lightgbm_30"].tolist()==[0,2,0,0,2]
    with pytest.raises(ValueError):
        majority_direction(np.array([1,1,1]))


def test_same_mask_and_neutral_count_incorrect():
    y=np.array([0,1,2,0,1])
    pred=np.array([0,2,2,2,0])
    ref=np.zeros(5,dtype=int)
    mask=np.array([True,True,True,False,False])
    out=fixed_mask_metrics(y,pred,mask,ref)
    assert out["n"]==3
    assert out["correct"]==2
    assert out["neutral_errors"]==1
    assert out["opposite_errors"]==0
    assert out["precision"]==pytest.approx(2/3)
    assert out["majority_precision_on_same_rows"]==pytest.approx(1/3)
    assert out["gain_vs_majority_points"]==pytest.approx(100/3)
    assert fixed_mask_metrics(y,pred,np.zeros(5,dtype=bool),ref) is None


def test_strictly_reject_ada_and_never_use_targets_in_rules():
    with pytest.raises(ValueError,match="four"):
        build_causal_contexts(
            {"ADAUSDT":pd.DataFrame()},pd.Timestamp("2025-03-01",tz="UTC"))
    ytrain=np.array([0,2,2])
    frame=_frame().copy()
    probs=np.tile([0.4,0.2,0.4],(5,1))
    baseline=rule_predictions(frame,ytrain,probs)
    frame["target_h6"]=[2,2,2,2,2]
    assert np.array_equal(
        rule_predictions(frame,ytrain,probs)["momentum_3"],
        baseline["momentum_3"])


def test_summary_is_aligned_and_skips_infeasible_selection():
    rec={"horizons":{}}
    truth=np.array([0,1,2])
    majority=np.array([0,0,0])
    candidate=np.array([0,2,2])
    all_rows=np.ones(len(truth),dtype=bool)
    oracle_movers=truth!=1
    for horizon in (2,6):
        # The majority_train rule MUST predict the identical train-derived
        # majority reference on every row. The old fixture incorrectly gave
        # it candidate predictions, creating a spurious +33.33 point gain.
        rules={}
        for name in RULES:
            pred=majority if name=="majority_train" else candidate
            rules[name]={
                "all_check":fixed_mask_metrics(truth,pred,all_rows,majority),
                "oracle_true_movers":fixed_mask_metrics(
                    truth,pred,oracle_movers,majority),
                "shared_ml_selected":None,
            }
        rec["horizons"][f"H{horizon}"]={
            "ml_tuning":{"feasible":False},
            "realized_check_coverage":None,
            "rules":rules,
            "by_asset_selected":{
                sym:{name:None for name in RULES}
                for sym in ("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")
            }
        }
    result=summarize([rec])
    assert result["H2"]["shared_ml_selected"]["momentum_1"]["comparable_folds"]==0
    assert result["H6"]["ml_mask"]["feasible_folds"]==0
    assert result["H2"]["all_check"]["majority_train"]["mean_gain_vs_train_majority_points"]==pytest.approx(0.0)
    assert result["H2"]["oracle_true_movers"]["majority_train"]["mean_gain_vs_train_majority_points"]==pytest.approx(0.0)
    assert result["H2"]["all_check"]["momentum_1"]["mean_gain_vs_train_majority_points"]==pytest.approx(100/3)
    assert result["H2"]["oracle_true_movers"]["momentum_1"]["mean_gain_vs_train_majority_points"]==pytest.approx(50.0)
    assert result["H2"]["oracle_true_movers"]["momentum_1"]["total_selected_rows_across_folds"]==2
