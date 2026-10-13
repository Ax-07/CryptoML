"""Offline paper replay guardrails (all data synthetic, no trades sent)."""
import numpy as np
import pandas as pd
import pytest

from mlpipeline.paper_replay import plan_positions, settle_positions


CUT = "2026-09-30T23:35:00Z"
TRAIN_CUT = "2026-07-02T23:35:00Z"


def signals():
    times = pd.date_range("2026-10-01T00:05:00Z", periods=12, freq="5min")
    p = np.tile([0.15, 0.20, 0.65], (len(times), 1))
    return pd.DataFrame({
        "symbol": "BTCUSDT", "bar_close_at": times,
        "recorded_at": times + pd.Timedelta(seconds=1),
        "model_trained_until": pd.Timestamp(TRAIN_CUT),
        "model_id": "frozen_cpu_v1", "model_family": "lightgbm",
        "p_down": p[:, 0], "p_neutral": p[:, 1], "p_up": p[:, 2],
        "prior_down": 0.30, "prior_neutral": 0.45, "prior_up": 0.25,
    })


def test_plan_never_reads_future_outcomes_and_respects_symbol_lock():
    plan, meta = plan_positions(
        signals(), alpha=0.75, final_end=CUT,
        training_end=TRAIN_CUT,
    )
    assert len(plan) == 2
    assert (plan.side == "LONG").all()
    assert meta["abstentions"]["same_symbol_busy"] == 10
    assert (plan.entry_at > plan.bar_close_at).all()
    assert (plan.exit_at - plan.entry_at == pd.Timedelta(minutes=30)).all()


def test_reject_unseen_information_in_signals():
    x = signals()
    x["future_return"] = 0.2
    with pytest.raises(ValueError, match="forbidden"):
        plan_positions(x, alpha=0.5, final_end=CUT, training_end=TRAIN_CUT)
    x = signals()
    x.loc[0, "recorded_at"] = x.loc[0, "bar_close_at"] + pd.Timedelta(minutes=6)
    with pytest.raises(ValueError, match="recorded"):
        plan_positions(x, alpha=0.5, final_end=CUT, training_end=TRAIN_CUT)
    x = signals()
    x.loc[0, "bar_close_at"] = pd.Timestamp("2026-09-30T23:30:00Z")
    with pytest.raises(ValueError, match="overlaps"):
        plan_positions(x, alpha=0.5, final_end=CUT, training_end=TRAIN_CUT)


def test_costs_and_causal_settlement():
    plan, _ = plan_positions(
        signals(), alpha=0.75, final_end=CUT,
        training_end=TRAIN_CUT,
    )
    outcomes = plan[["symbol", "entry_at", "exit_at"]].copy()
    outcomes["entry_mid"] = 100.0
    outcomes["exit_mid"] = 101.0
    outcomes["observed_at"] = outcomes.exit_at + pd.Timedelta(seconds=1)
    settled, summary = settle_positions(
        plan, outcomes, fee_bps=5, slippage_bps=5, notional_usdt=100
    )
    assert len(settled) == 2
    assert summary["trades"] == 2
    assert 0 < summary["net_pnl_usdt"] < 2
    outcomes.loc[0, "observed_at"] = outcomes.loc[0, "exit_at"] - pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="before"):
        settle_positions(plan, outcomes)


def test_short_synthetic_profit_and_missing_outcomes_fail_closed():
    x = signals()
    x.loc[:, "p_down"] = 0.7
    x.loc[:, "p_neutral"] = 0.2
    x.loc[:, "p_up"] = 0.1
    plan, _ = plan_positions(x, alpha=0.0, final_end=CUT, training_end=TRAIN_CUT)
    obs = plan[["symbol", "entry_at", "exit_at"]].copy()
    obs["entry_mid"] = 100.0
    obs["exit_mid"] = 99.0
    obs["observed_at"] = obs.exit_at
    result, summary = settle_positions(plan, obs, fee_bps=0, slippage_bps=0)
    assert (result.net_pnl_usdt > 0).all()
    assert summary["net_pnl_usdt"] > 0
    with pytest.raises(ValueError, match="Missing outcome"):
        settle_positions(plan, obs.iloc[:1])
