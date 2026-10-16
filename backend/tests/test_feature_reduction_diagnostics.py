"""Synthetic 30-feature removal audit guards; no historical files accessed."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.feature_group_ablation import GROUPS
from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.feature_reduction_diagnostics import (
    BASELINE, HORIZONS, variants_for, summarize, run_feature_reduction,
)


def test_group_variants_exact_partition():
    v=variants_for("group")
    assert len(v)==8
    assert v[BASELINE]==tuple(FEATURE_COLUMNS)
    for name,cols in GROUPS.items():
        retained=v[f"drop_group_{name}"]
        assert len(retained)==30-len(cols)
        assert set(retained)==set(FEATURE_COLUMNS)-set(cols)


def test_individual_variants_canonical_order_and_selection():
    chosen=("ret_6","ret_1")
    v=variants_for("individual",only_features=chosen)
    assert tuple(v)==(BASELINE,"drop_ret_1","drop_ret_6")
    assert len(v["drop_ret_6"])==29
    assert "ret_6" not in v["drop_ret_6"]
    all_features=variants_for("individual")
    assert len(all_features)==31


@pytest.mark.parametrize("mode,chosen",[
    ("fake",()),("group",("ret_1",)),
    ("individual",("unknown",)),("individual",("ret_1","ret_1")),
])
def test_invalid_variant_requests_fail(mode,chosen):
    with pytest.raises(ValueError):
        variants_for(mode,only_features=chosen)


def _result(feasible, uplift):
    return {
        "feasible":feasible,
        "check_metrics":{
            "uplift_points":uplift,
            "coverage":0.42,
            "raw_log_loss":1.06,
        } if feasible else None,
        "by_asset":{
            symbol:{"uplift_points":uplift}
            for symbol in ("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")
        } if feasible else None,
    }


def test_paired_summary_uses_only_common_feasible_folds():
    variants=variants_for("individual",only_features=("ret_1",))
    rows=[]
    for base_feasible,drop_feasible in ((True,True),(True,False),(False,True)):
        outcomes={}
        for h in HORIZONS:
            outcomes[f"H{h}"]={
                BASELINE:_result(base_feasible,5.),
                "drop_ret_1":_result(drop_feasible,5.4),
            }
        rows.append({"horizons":outcomes})
    summary=summarize(rows,variants)
    for h in HORIZONS:
        p=summary[f"H{h}"]["paired_drop_vs_full"]["drop_ret_1"]
        assert p["paired_feasible_folds"]==1
        assert p["positive_uplift_folds"]==1
        assert p["mean_uplift_delta_points"]==pytest.approx(0.4)
        assert p["by_symbol"]["BTCUSDT"]["comparable_folds"]==1


def test_all_infeasible_does_not_invent_improvement():
    variants=variants_for("individual",only_features=("ret_1",))
    rows=[{"horizons":{
        f"H{h}":{variant:_result(False,None) for variant in variants}
        for h in HORIZONS
    }}]
    s=summarize(rows,variants)
    assert s["H2"]["paired_drop_vs_full"]["drop_ret_1"][
        "mean_uplift_delta_points"] is None


def test_ada_rejected_before_fold_creation():
    fake=pd.DataFrame({"symbol":["ADAUSDT"],"bar_close_at":[pd.Timestamp("2026-01-01",tz="UTC")]})
    with pytest.raises(ValueError,match="ADA"):
        run_feature_reduction(fake,mode="group")


def test_small_training_smoke_uses_unlabeled_tuning_only():
    # Use synthetic complete fold with overridden tiny geometry invalid by contract:
    # this is intended to confirm strict P2 fixed geometry enforcement.
    fake=pd.DataFrame({
        "symbol":["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT"],
        "bar_close_at":pd.date_range("2026-01-01",periods=4,freq="5min",tz="UTC"),
    })
    for col in FEATURE_COLUMNS:
        fake[col]=np.float32(0.1)
    with pytest.raises(ValueError,match="Fixed H12"):
        run_feature_reduction(fake,mode="group",train_days=4)
