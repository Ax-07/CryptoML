"""V40 synthetic-only target regression tests; no historical market reads."""
import numpy as np
import pandas as pd
import pytest
from mlpipeline import funding_basis_target_contract_v40 as m


def fixture(n=130, start="2025-01-01T00:00:00Z", trend=0.0):
    opens=pd.date_range(start,periods=n,freq="5min",tz="UTC")
    d={"bar_open_at":opens,"bar_close_at":opens+m.M5}
    for i,k in enumerate(("spot","perp","mark")):
        p=100+i*.2+np.arange(n)*trend
        d.update({k+"_open":p-.02,k+"_high":p+.04,
                  k+"_low":p-.04,k+"_close":p})
        if k!="mark": d[k+"_volume"]=np.full(n,3+i,dtype=float)
    return pd.DataFrame(d).loc[:,list(m.MARKET_FIELDS)]


def label(f, subset="train", end=None):
    return m.classify_in_memory_test_fixture(
        f,window_start=pd.Timestamp("2025-01-01T00:00:00Z"),
        window_end=end or pd.Timestamp("2025-01-02T00:00:00Z"),subset=subset)


def test_fixed_schema_and_approval_gate(tmp_path):
    assert (m.HORIZON_BARS,m.VOLATILITY_LOOKBACK_RETURNS,m.THRESHOLD_MULTIPLIER)==(6,48,.5)
    with pytest.raises(ValueError,match="Explicit"):
        m.audit_contract(tmp_path/"normative",tmp_path/"v38","0"*64,
                         tmp_path/"out.json",synthetic_only_approval=False)
    assert not (tmp_path/"out.json").exists()


def test_flat_prices_only_neutral_after_volatility_warmup():
    output=label(fixture())
    assert not output.empty
    assert set(output.target)=={m.NEUTRAL}
    assert output.bar_close_at.min()==pd.Timestamp("2025-01-01T04:05:00Z")
    assert (output.label_mature_at-output.bar_close_at).eq(pd.Timedelta(minutes=30)).all()


def test_up_and_down_strict_threshold():
    rising=fixture(trend=.2)
    falling=fixture(trend=-.2)
    assert m.UP in set(label(rising).target)
    assert m.DOWN in set(label(falling).target)


def test_future_suffix_changes_labels_not_prior_features():
    frame=fixture()
    adjusted=frame.copy()
    # Future prices may alter historical targets, NEVER the features X(t).
    for col in ("perp_open","perp_high","perp_low","perp_close"):
        adjusted.loc[85:,col]=adjusted.loc[85:,col]*1.08
    original=m.build_causal_features(frame)
    mutated=m.build_causal_features(adjusted)
    pd.testing.assert_frame_equal(original.iloc[:85],mutated.iloc[:85])
    a=label(frame)
    b=label(adjusted)
    assert (a.target!=b.target).any()


def test_strict_maturity_excludes_end_equal():
    end=pd.Timestamp("2025-01-01T10:00:00Z")
    result=label(fixture(),end=end)
    assert not result.empty
    assert result.label_mature_at.lt(end).all()
    assert not result.label_mature_at.eq(end).any()


def test_embargo_at_validation_start():
    market=fixture()
    train=label(market,"train")
    validation=label(market,"validation")
    assert validation.bar_close_at.min()==train.bar_close_at.min()
    # Beginning-of-window embargo must be tested with a warm-up before start.
    late_start=pd.Timestamp("2025-01-01T07:00:00Z")
    result=m.classify_in_memory_test_fixture(
        market,window_start=late_start,
        window_end=pd.Timestamp("2025-01-02T00:00:00Z"),subset="validation")
    assert result.bar_close_at.min()>=late_start+6*m.M5


def test_gap_and_reserved_month_fail_closed():
    with pytest.raises(ValueError,match="incomplete"):
        label(fixture().drop(index=15).reset_index(drop=True))
    with pytest.raises(ValueError,match="out-of-scope"):
        label(fixture(start="2026-06-01T00:00:00Z"))
    with pytest.raises(ValueError,match="Unauthorized"):
        m.classify_in_memory_test_fixture(
            fixture(),
            window_start=pd.Timestamp("2026-06-01T00:00:00Z"),
            window_end=pd.Timestamp("2026-07-01T00:00:00Z"),subset="test")


def test_no_missing_or_nonfinite_input_accepted():
    f=fixture()
    f.loc[60,"perp_close"]=np.nan
    with pytest.raises(ValueError,match="Nonfinite"):
        label(f)


def test_no_export_or_invented_data_when_evidence_absent(tmp_path):
    with pytest.raises(ValueError,match="normative document"):
        m.audit_contract(
            tmp_path/m.V39_DOCUMENT_NAME,tmp_path/"v38.json","0"*64,
            tmp_path/"result.json",synthetic_only_approval=True)
    assert list(tmp_path.iterdir())==[]
