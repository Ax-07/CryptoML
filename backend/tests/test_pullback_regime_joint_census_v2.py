"""Synthetic-only tests for preregistered cross-tab; no P0/outcomes."""
import pandas as pd
import pytest
from mlpipeline.pullback_regime_joint_census_v2 import cross_counts, joint_report
from mlpipeline.pullback_regime_v2 import FEATURE_COLUMNS
from mlpipeline.pullback_regime_census_v2 import SYMBOLS
from tests.test_pullback_regime_census_v2 import synthetic
from mlpipeline.pullback_regime_v2 import describe_regimes

def test_joint_reconciles_four_symbols_seven_months():
    features = {s:describe_regimes(synthetic(s)) for s in SYMBOLS}
    report,table = joint_report(features,"a"*64)
    assert len(table)==4*7*3*4
    assert report["total"]["events"] == sum(report["total"]["joint"][s][d] for s in ("low","mid","high") for d in ("no_undercut","shallow","medium","deep")) + report["total"]["unclassified"]
    assert not table.events.lt(0).any()

def test_fixed_bucket_cross_tab():
    base=pd.DataFrame({"pullback_signal":[True]*4,**{x:[1.]*4 for x in FEATURE_COLUMNS}})
    base["v2_trend_spread_atr"]=[0.2,1.,2.,2.]
    base["v2_pullback_depth_ema9_atr"]=[0.25,0.75,1.2,0.3]
    result=cross_counts(base)
    assert result["events"]==4
    assert result["joint"]["low"]["shallow"]==1
    assert result["joint"]["mid"]["medium"]==1
    assert result["joint"]["high"]["deep"]==1
    assert result["joint"]["high"]["shallow"]==1

def test_reject_outcome_columns():
    base=pd.DataFrame({"pullback_signal":[False],**{x:[0.] for x in FEATURE_COLUMNS},
                       "net_5bps":[-1.]})
    with pytest.raises(ValueError,match="forbidden"):
        cross_counts(base)

def test_zero_signals_explicit_zeros():
    base=pd.DataFrame({"pullback_signal":[False],**{x:[0.] for x in FEATURE_COLUMNS}})
    c=cross_counts(base)
    assert c["events"]==0
    assert len(c["joint"])==3
    assert all(v==0 for sub in c["joint"].values() for v in sub.values())
