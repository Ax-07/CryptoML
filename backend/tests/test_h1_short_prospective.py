"""Pure synthetic safety checks; no REST, P0 or background execution."""
from datetime import timedelta
import numpy as np
import pandas as pd
import pytest
from mlpipeline.h1_short_prospective import valid_capture_clock, MODELS, PROTOCOL, _meta


def test_fixed_scope_three_models():
    assert list(MODELS)==["ema2","ema2_close_location","lightgbm_30"]
    assert len(MODELS["ema2"])==2
    assert len(MODELS["ema2_close_location"])==3
    assert len(MODELS["lightgbm_30"])==30


def test_time_gates_are_strict():
    start=pd.Timestamp("2026-11-01T00:00:00Z")
    end=start+pd.Timedelta(hours=72)
    meta={"start_utc":start.isoformat(),"end_exclusive_utc":end.isoformat()}
    t=start+pd.Timedelta(minutes=5)
    assert valid_capture_clock(meta,t+pd.Timedelta(seconds=20),t)
    assert not valid_capture_clock(meta,t+pd.Timedelta(seconds=1),t)
    assert not valid_capture_clock(meta,t+pd.Timedelta(minutes=5),t)
    assert not valid_capture_clock(meta,start-pd.Timedelta(minutes=5),start-pd.Timedelta(minutes=5))
    assert not valid_capture_clock(meta,end,end)


def test_meta_refuses_long_duration(tmp_path):
    import json
    base={"protocol":PROTOCOL,"duration_hours":72,
          "symbols":["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT"],
          "model_columns":MODELS,
          "start_utc":"2026-11-01T00:00:00+00:00",
          "end_exclusive_utc":"2026-11-04T00:00:00+00:00"}
    (tmp_path/"protocol.json").write_text(json.dumps(base))
    assert _meta(tmp_path)["duration_hours"]==72
    base["end_exclusive_utc"]="2026-11-05T00:00:00+00:00"
    (tmp_path/"protocol.json").write_text(json.dumps(base))
    with pytest.raises(ValueError,match="Invalid short"):
        _meta(tmp_path)
