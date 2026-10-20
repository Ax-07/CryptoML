"""Synthetic-only tests for historical H1 chronological, non-executable markout."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.h1_chronological_markout import (
    BAR, COST_ROUNDTRIP_BPS, MODEL_NAMES, calculate, chronological_gate,
    _score_scope, confidence_strata,
)
from mlpipeline.h1_binary_diagnostic import CANDIDATES


def sample():
    stamps=pd.date_range("2025-01-01",periods=4,freq="5min",tz="UTC")
    frame=pd.DataFrame({
        "fold_id":[0]*4,
        "symbol":["BTCUSDT"]*4,
        "bar_close_at":stamps,
        "actual_return_h1":[.001,-.002,0.,.0001],
    })
    for name in MODEL_NAMES:
        frame[f"{name}_sign"]=[1,0,1,0]
    frame["always_down_sign"]=0
    for name in CANDIDATES:
        frame[f"{name}_p_up"]=[.55,.45,.52,.49]
    return frame


def test_pooled_proxy_costs_use_all_bars_including_flat():
    x=_score_scope(sample())
    a=x["models"]["binary_30"]
    assert x["all_rows"]==4 and x["nonflat_rows"]==3 and x["flat_rows"]==1
    assert a["nonflat_accuracy"]==pytest.approx(2/3)
    # 10, +20, zero, -1 bps from long, short, long, short directions
    assert a["mean_gross_proxy_bps_per_signal"]==pytest.approx(29/4)
    assert a["cost_scenarios"]["5"]["mean_net_proxy_bps_per_signal"]==pytest.approx(29/4-5)
    assert COST_ROUNDTRIP_BPS==(0.,2.,5.,10.)


def test_chronological_gate_settles_after_each_prediction():
    result=chronological_gate(sample())
    assert result["forecast_events"]==4
    assert result["settlement_events"]==4
    assert result["strict_forecast_before_maturity"] is True
    assert result["live_timestamped_prediction_proof"] is False
    assert BAR==pd.Timedelta(minutes=5)


def test_chronological_gate_rejects_duplicate_rows():
    x=sample()
    with pytest.raises(ValueError,match="Duplicate"):
        chronological_gate(pd.concat([x,x.iloc[[0]]],ignore_index=True))


def test_confidence_bins_exhaustive_and_fixed():
    x=confidence_strata(sample())
    assert len(x)==4
    assert sum(b["n"] for b in x["binary_ema2"])==4
    assert [b["margin_low_inclusive"] for b in x["binary_30"]]==[0.,.01,.025,.05]


def test_calculate_outputs_not_execution_backtest():
    # Only synthetic fold 0 here; calculate has 36 slots by construction.
    result=calculate(sample())
    assert result["actual_trades"] is False
    assert result["fill_guarantee"] is False
    assert result["pooled"]["nonflat_rows"]==3
    assert len(result["by_fold"])==36
    assert result["no_P0_read_no_ADA_no_fold_OOS_no_final90_no_P3"] is True
