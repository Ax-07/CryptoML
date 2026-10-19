"""No-P0 synthetic tests for isolated H1 individual-feature exploration."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.h1_individual_feature_study import (
    select_features, compare, score_variant, summarize, align_check,
)


def test_fixed_catalog_selection():
    assert len(select_features(None))==30
    assert select_features(1)==(FEATURE_COLUMNS[0],)
    assert select_features(30)==tuple(FEATURE_COLUMNS)
    for v in (0,31,-1,1.2):
        with pytest.raises(ValueError):
            select_features(v)


def test_exact_paired_comparison_and_flat_mask():
    truth=np.array([2,0,2,0])
    pred=np.array([2,2,2,0])
    base=np.array([0,0,0,0])
    mask=np.array([True,True,True,False])
    x=compare(truth,pred,base,mask)
    assert x["n"]==3
    assert x["candidate_correct"]==2 and x["reference_correct"]==1
    assert x["candidate_only_correct"]==2 and x["reference_only_correct"]==1
    assert x["gain_points"]==pytest.approx(100/3)
    assert compare(truth,pred,base,np.zeros(4,dtype=bool)) is None


def check_rows():
    stamp=pd.date_range("2025-03-01",periods=4,freq="5min",tz="UTC")
    return pd.DataFrame({
        "symbol":["BTCUSDT"]*4,"bar_close_at":stamp,
        "future_return_h1":[.001,-.002,0.,.01],
        "target_h1":[2,0,1,2],
        "baseline_shared_mask":[True,False,True,False],
        "ema2_autonomous_mask":[False,True,True,True],
        "lightgbm_30_sign":[2,0,2,0],
        "lightgbm_ema2_sign":[0,0,2,2],
    })


def test_scores_keep_same_reference_rows_and_count_flats():
    rows=check_rows()
    prob=np.array([[.1,.2,.7],[.7,.2,.1],[.1,.8,.1],[.1,.1,.8]])
    from mlpipeline.h1_individual_feature_study import score_variant
    x=score_variant(rows,np.array([2,0,0,2]),prob)
    total=x["views"]["all_nonflat"]["lightgbm_ema2"]
    assert total["n"]==3 and total["candidate_correct"]==3
    assert total["reference_correct"]==2
    selected=x["views"]["shared30_nonflat"]["lightgbm_ema2"]
    assert selected["n"]==1
    assert x["canonical"]["all_check_n"]==4


def test_alignment_rejects_wrong_target_or_missing_row():
    check=check_rows()
    ref=check[["symbol","bar_close_at","baseline_shared_mask",
               "ema2_autonomous_mask","lightgbm_30_sign","lightgbm_ema2_sign"]].copy()
    ref["fold_id"]=0
    ref["target_3class"]=check.target_h1
    ref["next_bar_return"]=check.future_return_h1
    with pytest.raises(ValueError,match="target/future return drift"):
        bad=ref.copy()
        bad.loc[0,"next_bar_return"]+=.1
        align_check(check,bad,0)
    with pytest.raises(ValueError,match="count differs"):
        align_check(check,ref.iloc[:-1],0)


def test_summarize_variant_is_complete():
    x={"views":{"all_nonflat":{"lightgbm_ema2":{
        "n":2,"candidate_correct":2,"reference_correct":1,
        "gain_points":50}}},
       "by_asset":{s:{"n":1,"candidate_correct":1,"reference_correct":0}
                   for s in ("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")},
       "logloss":.8}
    record={"candidates":{"ret_1":{"solo":x,"ema2_plus":x}}}
    out=summarize([record],("ret_1",))
    assert out["ret_1"]["solo"]["pooled_delta_vs_ema2_points"]==50
    assert out["ret_1"]["ema2_plus"]["positive_folds"]==1
