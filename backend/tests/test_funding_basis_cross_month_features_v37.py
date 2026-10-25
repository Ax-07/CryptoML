"""V37 no-network tests: exact cross-month M5, causal prefixes and fail closed."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline import funding_basis_cross_month_features_v37 as m


def native(start,n):
    opens=pd.date_range(start,periods=n,freq="5min",tz="UTC")
    f={"bar_open_at":opens,"bar_close_at":opens+m.M5}
    for i,kind in enumerate(("spot","perp","mark")):
        px=np.arange(n)*0.01+100+i*0.1
        f.update({kind+"_open":px-0.01,kind+"_high":px+0.02,
                  kind+"_low":px-0.03,kind+"_close":px})
        if kind!="mark":
            f[kind+"_volume"]=np.ones(n)*3
    return pd.DataFrame(f).loc[:,list(m.MARKET_FIELDS)]


def two_months():
    # 60 bars ending on 31 Oct, followed by 70 bars opening 1 Nov UTC.
    a=native("2024-10-31T19:00:00Z",60)
    b=native("2024-11-01T00:00:00Z",70)
    return [("2024-10",a),("2024-11",b)]


def test_stitch_reclaims_month_warmup_without_fabrication():
    counts,rows=m.stitch_verified_months(two_months())
    assert counts["native_closed_m5"]==130
    assert counts["monthly_reset_complete"]==34
    assert counts["stitched_complete"]==82
    assert counts["newly_available_from_native_carryover"]==48
    assert rows[0]["newly_available_from_native_carryover"]==0
    assert rows[1]["newly_available_from_native_carryover"]==48


def test_cross_month_gap_fails_closed():
    months=two_months()
    b=months[1][1].copy()
    b.loc[:,"bar_open_at"]+=m.M5
    b.loc[:,"bar_close_at"]+=m.M5
    with pytest.raises(ValueError,match="gap/overlap"):
        m.stitch_verified_months([months[0],("2024-11",b)])


def test_overlapping_cross_month_fails_closed():
    months=two_months()
    b=months[1][1].copy()
    b.loc[:,"bar_open_at"]-=m.M5
    b.loc[:,"bar_close_at"]-=m.M5
    with pytest.raises(ValueError,match="gap/overlap"):
        m.stitch_verified_months([months[0],("2024-11",b)])


def test_future_month_mutation_cannot_change_previous_features():
    months=two_months()
    joined=pd.concat([v for _,v in months],ignore_index=True)
    head=m.build_causal_features(joined.iloc[:60])
    change=joined.copy()
    for kind in ("spot","perp","mark"):
        for field in ("open","high","low","close"):
            change.loc[60:,f"{kind}_{field}"]*=1.01
    after=m.build_causal_features(change)
    pd.testing.assert_frame_equal(head,after.iloc[:60].reset_index(drop=True),check_exact=True)


def test_blind_month_rejected():
    f=native("2026-06-01T00:00:00Z",60)
    with pytest.raises(ValueError,match="Invalid development month"):
        m.stitch_verified_months([("2026-06",f)])


def test_malformed_v36_digest_refused(tmp_path):
    with pytest.raises(ValueError,match="V36 report SHA"):
        m.verify_v36_report(tmp_path/"missing.json","not-a-digest",v35_sha="0"*64)


def test_report_nonoverwrite_guard(tmp_path):
    old=tmp_path/"old.json"
    old.write_text("untouched")
    with pytest.raises(ValueError,match="distinct NEW"):
        m.run(tmp_path/"freeze","0"*64,tmp_path/"v35","0"*64,
              tmp_path/"v36","0"*64,old,tmp_path/"out.csv")
    assert old.read_text()=="untouched"


def test_v37_contract_only_native_inputs():
    assert len(m.ELIGIBLE_MONTHS)==20
    assert m.ELIGIBLE_MONTHS[-1]=="2026-05"
    assert m.EXPECTED_GAIN==19*m.LOOKBACK_BARS
    assert m.EXPECTED_STITCHED_COMPLETE==m.SOURCE_BARS_PER_ASSET-m.LOOKBACK_BARS
    assert "funding" not in " ".join(m.FEATURE_COLUMNS)
