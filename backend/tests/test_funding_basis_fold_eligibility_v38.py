"""V38 offline calendar-only tests; no source files or labels."""
import pandas as pd
import pytest
from mlpipeline import funding_basis_fold_eligibility_v38 as m

def test_fold_counts_and_maturity():
    rows=m.eligibility(m.make_calendar_folds())
    assert len(rows)==99
    counts={k:sum(x["theoretical_upper_bound_all_four_symbols"] for x in rows if x["subset"]==k) for k in ("train","validation","test")}
    assert counts=={"train":4560804,"validation":530640,"test":530640}
    assert all(pd.Timestamp(r["last_hypothetical_label_maturity_utc"])<m.INTERNAL_RESERVE_START for r in rows)
    assert all(r["theoretical_upper_bound_per_symbol"]>0 for r in rows)

def test_first_training_window_warmup():
    rows=m.eligibility(m.make_calendar_folds())
    assert rows[0]["subset"]=="train"
    assert rows[0]["theoretical_upper_bound_per_symbol"]==34505
    assert rows[1]["theoretical_upper_bound_per_symbol"]==4020
    assert rows[2]["theoretical_upper_bound_per_symbol"]==4020

def test_empty_wrong_inputs_rejected(tmp_path):
    with pytest.raises(ValueError,match="distinct NEW"):
        old=tmp_path/"old.json"
        old.write_text("keep")
        m.run(tmp_path/"missing","0"*64,tmp_path/"missing2","0"*64,old,tmp_path/"new.csv")
    assert old.read_text()=="keep"
    with pytest.raises(ValueError,match="Frozen report"):
        m.run(tmp_path/"missing","0"*64,tmp_path/"missing2","0"*64,
              tmp_path/"new.json",tmp_path/"new.csv")

def test_no_unsupported_features_or_trades_in_outputs():
    assert "funding" not in m.FEATURE_SCHEMA_VERSION
    assert m.FIRST_FULL_FEATURE_CLOSE==pd.Timestamp("2024-10-01T04:05:00Z")
    assert all(r["subset"] in ("train","validation","test") for r in m.eligibility(m.make_calendar_folds()))
