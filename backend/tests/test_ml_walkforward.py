"""Temporal leakage regression tests for the walk-forward CPU split."""
import pandas as pd
import pytest

from mlpipeline.walkforward import make_walk_forward_folds, split_walk_forward_fold


def samples():
    times = pd.date_range("2025-01-01T00:05:00Z", periods=14 * 288, freq="5min")
    return pd.concat([
        pd.DataFrame({
            "symbol": symbol, "bar_close_at": times,
            "label_mature_at": times + pd.Timedelta(minutes=30),
            "target": 1,
        })
        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT")
    ], ignore_index=True)


def params():
    return dict(holdout_symbol="ADAUSDT", train_days=4, validation_days=2,
                test_days=2, step_days=2, final_holdout_days=2)


def test_disjoint_oos_holdout_and_strict_purge():
    data = samples()
    folds = make_walk_forward_folds(data, **params())
    assert len(folds) == 3
    seen_keys = set()
    final_start = data.bar_close_at.max() + pd.Timedelta(minutes=5) - pd.Timedelta(days=2)
    for fold in folds:
        p = split_walk_forward_fold(data, fold=fold, holdout_symbol="ADAUSDT")
        assert all(not p[name].empty for name in p)
        assert not p["train"].symbol.eq("ADAUSDT").any()
        assert not p["validation"].symbol.eq("ADAUSDT").any()
        assert not p["seen_test"].symbol.eq("ADAUSDT").any()
        assert p["unseen_test"].symbol.eq("ADAUSDT").all()
        assert p["train"].label_mature_at.lt(fold.validation_start).all()
        assert p["validation"].label_mature_at.lt(fold.test_start).all()
        for name in ("seen_test", "unseen_test"):
            part = p[name]
            assert part.label_mature_at.lt(fold.test_end).all()
            assert part.bar_close_at.lt(final_start).all()
            keys = set(zip(part.symbol, part.bar_close_at))
            assert keys.isdisjoint(seen_keys)
            seen_keys.update(keys)
    fold = folds[0]
    p = split_walk_forward_fold(data, fold=fold, holdout_symbol="ADAUSDT")
    boundary_train = data.loc[data.label_mature_at.eq(fold.validation_start) & data.symbol.eq("BTCUSDT")]
    boundary_val = data.loc[data.label_mature_at.eq(fold.test_start) & data.symbol.eq("BTCUSDT")]
    assert not boundary_train.empty and not boundary_val.empty
    assert not p["train"].bar_close_at.isin(boundary_train.bar_close_at).any()
    assert not p["validation"].bar_close_at.isin(boundary_val.bar_close_at).any()


def test_reject_overlap_and_short_history():
    with pytest.raises(ValueError, match="overlap"):
        make_walk_forward_folds(samples(), **{**params(), "step_days": 1})
    with pytest.raises(ValueError, match="Insufficient history"):
        make_walk_forward_folds(samples(), holdout_symbol="ADAUSDT")


def test_holdout_dates_do_not_set_fold_boundaries():
    data = samples()
    original = make_walk_forward_folds(data, **params())
    extra = data.loc[data.symbol.eq("ADAUSDT")].copy()
    extra["bar_close_at"] += pd.Timedelta(days=200)
    extra["label_mature_at"] += pd.Timedelta(days=200)
    extended = make_walk_forward_folds(pd.concat([data, extra], ignore_index=True), **params())
    assert extended == original
