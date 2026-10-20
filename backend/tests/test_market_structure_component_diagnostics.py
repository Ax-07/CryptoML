"""Pure, synthetic component-ablations guards; no Binance/data access."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.features import FEATURE_COLUMNS
from mlpipeline.market_structure_features import STRUCTURE_COLUMNS
from mlpipeline.market_structure_component_diagnostics import (
    STATE, BOS, CHOCH, POSITION, VARIANTS, validate_groups,
    event_frequency, frequency_report, ablation_summary,
)


def test_component_partition_exact_and_reference_unchanged():
    validate_groups()
    assert len(STATE)==4 and len(BOS)==2 and len(CHOCH)==2 and len(POSITION)==4
    assert set((*STATE,*BOS,*CHOCH,*POSITION)) == set(STRUCTURE_COLUMNS)
    assert tuple(FEATURE_COLUMNS)==VARIANTS["baseline_30"]
    assert [len(v) for v in VARIANTS.values()]==[30,34,32,32,34,42]


def _data():
    times=pd.date_range("2026-01-02T12:00:00Z",periods=4,freq="5min")
    row=pd.DataFrame({"symbol":["BTCUSDT"]*4,"bar_close_at":times})
    for c in (*STATE,*BOS,*CHOCH):
        row[c]=np.float32(0)
    row.loc[0,"structure_hh"]=1
    row.loc[1,"structure_hh"]=1
    row.loc[1,"structure_bos_up"]=1
    row.loc[2,"structure_choch_down"]=1
    return row


def test_freq_distinguishes_state_from_pulse_and_avoids_double_count():
    a=_data()
    result=event_frequency(a)
    assert result["rows"]==4
    assert result["flags"]["structure_hh"]["active_bars"]==2
    assert result["bos_any"]==1
    assert result["choch_any"]==1
    assert result["break_any"]==2


def test_frequency_report_rejects_overlap():
    a=_data()
    with pytest.raises(ValueError,match="Overlapping"):
        frequency_report([(0,a),(1,a)])
    r=frequency_report([(0,a)])
    assert r["primary_12_17_UTC"]["rows"]==4
    assert r["by_symbol"]["BTCUSDT"]["bos_any"]==1


def test_summary_only_compares_feasible_paired_folds():
    fake={}
    for h in ("H2","H6"):
        fake[h]={}
        for name in VARIANTS:
            feasible=name!="plus_choch_32"
            fake[h][name]={
                "feasible":feasible,
                "check_metrics":{
                    "coverage":0.4,
                    "uplift_points":5.1 if name!="baseline_30" else 5.,
                    "raw_log_loss":1.02,
                } if feasible else None,
            }
    out=ablation_summary([{"horizons":fake}])
    assert out["H2"]["paired_vs_baseline"]["plus_state_34"]["paired_feasible_folds"]==1
    assert out["H2"]["paired_vs_baseline"]["plus_state_34"]["mean_uplift_delta_points"]==pytest.approx(0.1)
    assert out["H6"]["paired_vs_baseline"]["plus_choch_32"]["paired_feasible_folds"]==0
    assert out["H6"]["paired_vs_baseline"]["plus_choch_32"]["mean_uplift_delta_points"] is None
