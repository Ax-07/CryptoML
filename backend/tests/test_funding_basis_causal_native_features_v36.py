"""V36 synthetic no-network causal contract and future-tail invariance."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline import funding_basis_causal_native_features_v36 as m


def synthetic(n=180, start="2025-02-01T00:00:00Z"):
    opens=pd.date_range(start, periods=n, freq="5min", tz="UTC")
    data={"bar_open_at":opens,"bar_close_at":opens+m.M5}
    for i,kind in enumerate(("spot","perp","mark")):
        close=100.0+i*0.2+np.arange(n)*0.01
        data.update({f"{kind}_open":close-0.02,
                     f"{kind}_high":close+0.04,
                     f"{kind}_low":close-0.04,
                     f"{kind}_close":close})
        if kind!="mark":
            data[kind+"_volume"]=np.ones(n)*(2+i)
    return pd.DataFrame(data).loc[:,list(m.MARKET_FIELDS)]


def test_v36_feature_schema_distinct_from_legacy():
    assert m.FEATURE_SCHEMA_VERSION != "features_v1"
    assert len(m.FEATURE_COLUMNS)==19
    assert not any("funding" in x for x in m.FEATURE_COLUMNS)


def test_causal_prefix_invariant_when_future_changes():
    frame=synthetic()
    prefix=m.build_causal_features(frame.iloc[:125].copy())
    modified=frame.copy()
    for col in m.MARKET_FIELDS[2:]:
        modified.loc[125:,col]=modified.loc[125:,col]*1.02
    full=m.build_causal_features(modified)
    pd.testing.assert_frame_equal(prefix,full.iloc[:125].reset_index(drop=True))


def test_single_bar_future_does_not_revise_previous_features():
    frame=synthetic(80)
    before=m.build_causal_features(frame.iloc[:-1].copy())
    after=m.build_causal_features(frame)
    pd.testing.assert_frame_equal(before,after.iloc[:-1].reset_index(drop=True))


def test_no_lookahead_and_first_volatility_row_needs_49_closes():
    out=m.build_causal_features(synthetic(80))
    assert pd.isna(out.loc[47,"spot_volatility_48"])
    assert np.isfinite(out.loc[48,"spot_volatility_48"])
    assert pd.isna(out.loc[47,"spot_volume_rel_48"]) is False or np.isfinite(out.loc[47,"spot_volume_rel_48"])


def test_gap_rejected_not_interpolated():
    frame=synthetic(80).drop(index=12).reset_index(drop=True)
    with pytest.raises(ValueError,match="incomplete"):
        m.build_causal_features(frame)


def test_nan_or_negative_native_market_rejected():
    frame=synthetic(80)
    frame.loc[25,"mark_close"]=np.nan
    with pytest.raises(ValueError,match="Nonfinite"):
        m.build_causal_features(frame)
    frame=synthetic(80)
    frame.loc[25,"spot_volume"]=-1
    with pytest.raises(ValueError,match="Negative"):
        m.build_causal_features(frame)


def test_blind_and_external_holdout_rejected_before_features():
    with pytest.raises(ValueError,match="out-of-scope"):
        m.build_causal_features(synthetic(80,"2026-06-01T00:00:00Z"))
    with pytest.raises(ValueError,match="out-of-scope"):
        m.build_causal_features(synthetic(80,"2026-07-01T00:00:00Z"))


def test_missing_columns_rejected_no_fabrication():
    frame=synthetic(80).drop(columns=["mark_close"])
    with pytest.raises(ValueError,match="exact V34"):
        m.build_causal_features(frame)


def test_new_reports_required_and_v35_hash_bound(tmp_path):
    j=tmp_path/"old.json"
    j.write_text("do not replace")
    with pytest.raises(ValueError,match="distinct NEW"):
        m.run(tmp_path/"freeze",m.APPROVED_FREEZE_SHA256,
              tmp_path/"v35","0"*64,j,tmp_path/"report.csv")
    assert j.read_text()=="do not replace"
    with pytest.raises(ValueError,match="V35 audit SHA"):
        m.run(tmp_path/"freeze",m.APPROVED_FREEZE_SHA256,
              tmp_path/"v35","0"*64,tmp_path/"report.json",tmp_path/"report.csv")


def test_internal_boundary_can_be_filtered_before_features():
    frame=synthetic(70,"2026-05-31T18:10:00Z")
    assert frame.bar_close_at.iloc[-1]>=m.INTERNAL_RESERVE_START
    with pytest.raises(ValueError,match="out-of-scope"):
        m.build_causal_features(frame)
    trimmed=frame.loc[frame.bar_close_at.lt(m.INTERNAL_RESERVE_START)].reset_index(drop=True)
    assert m.build_causal_features(trimmed).bar_close_at.max()<m.INTERNAL_RESERVE_START
