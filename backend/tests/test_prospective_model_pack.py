"""Boundary tests: June open-time partition, exclusive July close cutoff."""
import pandas as pd
import pytest

from mlpipeline.prospective_model_pack import CUTOFF, exclude_july_boundary_close


def _frame(closes):
    closes = pd.to_datetime(closes, utc=True)
    return pd.DataFrame({
        "bar_open_at": closes - pd.Timedelta(minutes=5),
        "bar_close_at": closes,
    })


def test_expected_final_june_bar_excluded_without_changing_prior_bars():
    frame = _frame([
        "2026-06-30T23:50:00Z",
        "2026-06-30T23:55:00Z",
        "2026-07-01T00:00:00Z",
    ])
    result = exclude_july_boundary_close(frame, period="2026-06")
    assert len(result) == 2
    assert result.bar_close_at.max() == CUTOFF - pd.Timedelta(minutes=5)
    assert result.bar_close_at.lt(CUTOFF).all()
    assert len(frame) == 3  # do not alter the original source


def test_any_crossing_from_other_month_is_integrity_failure():
    with pytest.raises(AssertionError, match="crossover"):
        exclude_july_boundary_close(
            _frame(["2026-07-01T00:00:00Z"]), period="2026-05"
        )


def test_extra_post_cutoff_bars_not_silently_filtered():
    with pytest.raises(AssertionError, match="crossover"):
        exclude_july_boundary_close(
            _frame(["2026-07-01T00:00:00Z", "2026-07-01T00:05:00Z"]),
            period="2026-06",
        )


def test_safe_history_remains_unchanged():
    frame = _frame(["2026-06-30T23:50:00Z"])
    assert exclude_july_boundary_close(frame, period="2026-06") is frame
