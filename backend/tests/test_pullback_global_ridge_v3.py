"""Synthetic-only V3 Ridge protocol tests (no P0, ADA, P3 or future market data)."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.pullback_global_ridge_v3 import (
    SYMBOLS, V1_FEATURES, REGIME_FEATURES, MODEL_SPECS,
    START, SHORT_START, TRAIN_END, APRIL_END, MAY_END, CUTOFF,
    _partition, _random_expectation, _validate_and_combine,
    annotate_confirmation_long, compare_fixed_models,
)
from mlpipeline.trend_pullback import M5


def fake_events():
    # Genuine chronological UTC sets covering both long and short trainings.
    train_long = pd.date_range("2024-11-10T00:00:00Z", periods=40, freq="1h")
    train_short = pd.date_range("2026-02-10T00:00:00Z", periods=40, freq="1h")
    april = pd.date_range("2026-04-10T00:00:00Z", periods=12, freq="1h")
    may = pd.date_range("2026-05-10T00:00:00Z", periods=6, freq="1h")
    june = pd.date_range("2026-06-10T00:00:00Z", periods=6, freq="1h")
    timestamps = train_long.append([train_short, april, may, june])
    result = {}
    for k, symbol in enumerate(SYMBOLS):
        f = pd.DataFrame({
            "symbol": symbol,
            "signal_close_at": timestamps,
            "entry_at": timestamps+pd.Timedelta(minutes=5),
            "mature_at": timestamps+pd.Timedelta(minutes=35),
            "baseline_b_confirmed": np.arange(len(timestamps)) % 5 != 0,
            "exit_reason": "timeout",
        })
        raw = (np.sin(np.arange(len(f))*0.3 + k)*10.0)
        for cost in (2, 5, 10):
            f[f"net_{cost}bps"] = raw - cost
        for j, col in enumerate(V1_FEATURES + REGIME_FEATURES):
            f[col] = np.sin(np.arange(len(f))*0.13 + j*0.17 + k*0.09)
        result[symbol] = f
    return result


def test_compare_smoke_three_fixed_models_and_eight_groups():
    events = fake_events()
    report, table = compare_fixed_models(events, "a"*64)
    assert len(report["models"]) == 3
    assert len(table) == 8 * 6
    assert set(report["diagnostics"]) == {"2026-05","2026-06"}
    for name in MODEL_SPECS:
        assert report["models"][name]["threshold_score_net_5bps"] >= 0
        assert report["combined_may_june"][name]["trades"] <= report["combined_may_june"]["B_confirmed"]["trades"]
    assert report["combined_may_june"]["D_no_trade"]["net_5bps_per_opportunity"] == 0
    assert report["status"].endswith("NOT_INDEPENDENT_OOS")


def test_mature_time_purge_is_strict():
    f = pd.DataFrame({
        "signal_close_at": [TRAIN_END-pd.Timedelta(minutes=5),
                            TRAIN_END-pd.Timedelta(minutes=40)],
        "mature_at": [TRAIN_END+pd.Timedelta(minutes=30),
                      TRAIN_END-pd.Timedelta(minutes=5)],
    })
    assert len(_partition(f,START,TRAIN_END)) == 1


def test_random_expectation_matches_count_and_B_mean():
    g = pd.DataFrame({"baseline_b_confirmed":[True,True,False],
                      "net_2bps":[4.,8.,100.],
                      "net_5bps":[1.,5.,100.],
                      "net_10bps":[-4.,0.,100.]})
    x = _random_expectation(g,1)
    assert x["expected_sum_net_5bps"] == 3.
    assert x["expected_sum_net_2bps"] == 6.
    with pytest.raises(ValueError,match="exceed"):
        _random_expectation(g,3)


def test_reject_unseen_or_out_of_period():
    f = fake_events()
    f["ADAUSDT"] = f.pop("BTCUSDT")
    with pytest.raises(ValueError,match="Four seen"):
        compare_fixed_models(f, "a"*64)
    # Move an early event to the end: the first guard correctly rejects
    # nonmonotonic events before it checks the temporal cutoff.
    f = fake_events()
    f["BTCUSDT"].loc[0,"signal_close_at"] = CUTOFF
    with pytest.raises(ValueError, match="invalid event frame"):
        compare_fixed_models(f,"a"*64)


def test_reject_reserved_date_even_when_chronology_and_clocks_are_valid():
    # Exercise the independent July cutoff guard, rather than accidentally
    # exercising only the earlier sortedness check.
    f = fake_events()
    last = f["BTCUSDT"].index[-1]
    f["BTCUSDT"].loc[last, "signal_close_at"] = CUTOFF
    f["BTCUSDT"].loc[last, "entry_at"] = CUTOFF + M5
    f["BTCUSDT"].loc[last, "mature_at"] = CUTOFF + pd.Timedelta(minutes=35)
    with pytest.raises(ValueError, match="out-of-scope"):
        compare_fixed_models(f, "a"*64)


def test_reject_invalid_features_and_model_cohort():
    f = fake_events()
    selected = f["ETHUSDT"].index[f["ETHUSDT"].baseline_b_confirmed][0]
    f["ETHUSDT"].loc[selected,REGIME_FEATURES[0]] = np.nan
    with pytest.raises(ValueError,match="incomplete common"):
        compare_fixed_models(f,"a"*64)


def test_confirmation_rule_causal_and_precedes_entry():
    dates = pd.date_range("2025-01-01T00:05:00Z",periods=14,freq="5min")
    candles = pd.DataFrame({
        "bar_open_at":dates-M5,"bar_close_at":dates,
        "open":np.full(14,100.),
        "close":np.full(14,100.),
    })
    candles.loc[5,"close"] = 103
    detected = pd.DataFrame({
        "bar_close_at":dates, "ema9":np.full(14,99.),
        "pullback_signal":[i==4 for i in range(14)],
    })
    t=dates[4]
    outcome = pd.DataFrame({
        "signal_close_at":[t],"entry_at":[t+M5],
        "mature_at":[t+pd.Timedelta(minutes=35)],
    })
    first=annotate_confirmation_long(candles,detected,outcome)
    assert bool(first.baseline_b_confirmed.iloc[0])
    changed=candles.copy()
    changed.loc[7:,"close"]=9999.
    later=annotate_confirmation_long(changed,detected,outcome)
    assert bool(later.baseline_b_confirmed.iloc[0])
    incorrect=outcome.copy()
    incorrect.loc[0,"entry_at"]+=M5
    with pytest.raises(ValueError,match="timing"):
        annotate_confirmation_long(candles,detected,incorrect)
