"""Calendar-based purged walk-forward folds for CPU research."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    validation_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # exclusive


def make_walk_forward_folds(
    data: pd.DataFrame,
    *,
    holdout_symbol: str,
    train_days: int = 120,
    validation_days: int = 14,
    test_days: int = 14,
    step_days: int = 14,
    final_holdout_days: int = 90,
) -> list[WalkForwardFold]:
    """Reserve final dates; never choose fold limits using unseen-symbol dates."""
    required = {"symbol", "bar_close_at", "label_mature_at"}
    if missing := required.difference(data.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if data.empty or holdout_symbol not in set(data.symbol):
        raise ValueError("Missing data or holdout symbol")
    if holdout_symbol in {"BTCUSDT", "ETHUSDT"}:
        raise ValueError("BTC/ETH are required market context")
    if any(type(d) is not int or d < 1 for d in
           (train_days, validation_days, test_days, step_days, final_holdout_days)):
        raise ValueError("All durations must be positive integer days")
    if step_days < test_days:
        raise ValueError("step_days must be >= test_days: OOS tests cannot overlap")
    for col in ("bar_close_at", "label_mature_at"):
        if not isinstance(data[col].dtype, pd.DatetimeTZDtype) or str(data[col].dt.tz) != "UTC":
            raise ValueError(f"{col} must be UTC-aware")
        if data[col].isna().any():
            raise ValueError(f"{col} has null timestamps")
    if data.duplicated(["symbol", "bar_close_at"]).any():
        raise ValueError("Duplicate sample key")
    seen = data.loc[data.symbol != holdout_symbol]
    if seen.symbol.nunique() < 2:
        raise ValueError("Need two or more seen assets")
    duration = lambda d: pd.Timedelta(days=d)
    first = seen.bar_close_at.min()
    # M5 close is exclusive; reserve last N full days.
    final_start = seen.bar_close_at.max() + pd.Timedelta(minutes=5) - duration(final_holdout_days)
    folds: list[WalkForwardFold] = []
    val_start = first + duration(train_days)
    while val_start + duration(validation_days + test_days) <= final_start:
        test_start = val_start + duration(validation_days)
        folds.append(WalkForwardFold(
            len(folds), val_start - duration(train_days),
            val_start, test_start, test_start + duration(test_days)
        ))
        val_start += duration(step_days)
    if not folds:
        raise ValueError("Insufficient history for folds and untouched final holdout")
    return folds


def split_walk_forward_fold(
    data: pd.DataFrame, *, fold: WalkForwardFold, holdout_symbol: str
) -> dict[str, pd.DataFrame]:
    """Strictly purge labels maturing on/after each next time boundary."""
    t, mature = data.bar_close_at, data.label_mature_at
    seen = data.symbol != holdout_symbol
    training = seen & t.ge(fold.train_start) & t.lt(fold.validation_start) & mature.lt(fold.validation_start)
    validation = seen & t.ge(fold.validation_start) & t.lt(fold.test_start) & mature.lt(fold.test_start)
    oos = t.ge(fold.test_start) & t.lt(fold.test_end) & mature.lt(fold.test_end)
    return {
        key: data.loc[mask].copy().reset_index(drop=True)
        for key, mask in {
            "train": training, "validation": validation,
            "seen_test": seen & oos, "unseen_test": ~seen & oos,
        }.items()
    }
