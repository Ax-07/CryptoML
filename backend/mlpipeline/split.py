"""Purged chronological split with a completely unseen symbol holdout."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeBoundaries:
    validation_start: pd.Timestamp
    test_start: pd.Timestamp
    first: pd.Timestamp
    last: pd.Timestamp

    def as_dict(self) -> dict[str, str]:
        return {k: str(getattr(self, k).isoformat().replace("+00:00", "Z"))
                for k in ("first", "validation_start", "test_start", "last")}


def boundaries(data: pd.DataFrame, symbols: list[str], *, train_ratio: float = 0.60,
               validation_ratio: float = 0.20) -> TimeBoundaries:
    """Cut by time, not by random rows or raw dataset size."""
    if not (0.1 < train_ratio < 0.8 and 0.05 < validation_ratio < 0.5
            and train_ratio + validation_ratio < 0.95):
        raise ValueError("Invalid chronological split ratios")
    times = pd.DatetimeIndex(data.loc[data["symbol"].isin(symbols), "bar_close_at"].unique()).sort_values()
    if len(times) < 100:
        raise ValueError("Too few distinct timestamps")
    i = int(len(times) * train_ratio)
    j = int(len(times) * (train_ratio+validation_ratio))
    return TimeBoundaries(times[i], times[j], times[0], times[-1])


def split_with_purge(data: pd.DataFrame, *, boundaries: TimeBoundaries,
                     holdout_symbol: str) -> dict[str, pd.DataFrame]:
    """Keep label maturation strictly BEFORE next segment boundary.

    Samples of held-out crypto appear only in `unseen_test`; its previous
    history remains allowed for trailing features but never used for fitting.
    """
    t = data["bar_close_at"]
    mature = data["label_mature_at"]
    seen = data["symbol"] != holdout_symbol
    b = boundaries
    selections = {
        "train": seen & (t < b.validation_start) & (mature < b.validation_start),
        "validation": seen & (t >= b.validation_start) & (t < b.test_start)
                      & (mature < b.test_start),
        "seen_test": seen & (t >= b.test_start),
        "unseen_test": (~seen) & (t >= b.test_start),
    }
    return {k: data.loc[v].copy().reset_index(drop=True) for k,v in selections.items()}
