"""H2/H3/H6 old P2 checks: offline no-fit, non-executable markouts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ingestion.storage import CandleStore
from mlpipeline.dataset import load_symbol,manifest_fingerprint
from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS
from mlpipeline.features import build_symbol_features
from mlpipeline.h1_individual_feature_study import load_reference
from mlpipeline.label_stability_audit import load_plan
from mlpipeline.horizon_markout_source import (
    HORIZONS,MODELS,KEY,rebuild,phase_rows,evaluate_rows)

PROTOCOL="p2_h2_h3_h6_markout_devchecks_v1"

def verify_grid(grid,report,plan):
    expected=report["predictions_rows"]
    if (len(grid)!=expected or grid.duplicated(KEY).any()
        or not grid.symbol.isin(SEEN_SYMBOLS).all()
        or not grid.horizon.isin(range(1,7)).all()
        or not grid.fold_id.isin(range(36)).all()
        or not isinstance(grid.bar_close_at.dtype,pd.DatetimeTZDtype)
        or str(grid.bar_close_at.dt.tz)!="UTC"):
        raise ValueError("Source prediction grid mismatch")
    h1=grid.loc[grid.horizon.eq(1),KEY[:3]]
    for h in range(2,7):
        sub=grid.loc[grid.horizon.eq(h),KEY[:3]]
        if len(sub)!=len(h1) or len(sub.merge(h1,on=KEY[:3],validate="one_to_one"))!=len(h1):
            raise ValueError("Missing cross-horizon prediction keys")
    if not grid.loc[grid.horizon.ne(1),"next_bar_return"].isna().all():
        raise ValueError("Original H>1 returns must be absent")
    for f in plan:
        sub=h1.loc[h1.fold_id.eq(f["fold_id"])]
        if (len(sub)!=f["expected_check_rows"]
            or not sub.bar_close_at.ge(f["check_start"]).all()
            or not (sub.bar_close_at+pd.Timedelta(minutes=60)).lt(f["check_end"]).all()):
            raise ValueError("Escaped original development folds")
    return grid.loc[grid.horizon.isin(HORIZONS)].copy()

def verify_rebuild(data,report):
    for fid in range(36):
        for h in HORIZONS:
            sub=data.loc[data.fold_id.eq(fid)&data.horizon.eq(h)]
            old=report["folds"][fid]["horizons"][f"H{h}"]["metrics"]["all_check"]
            for model in MODELS:
                if (old[model] is None or old[model]["n"]!=len(sub)
                    or old[model]["correct"]!=int(sub[model+"_sign"].eq(sub.target_3class).sum())):
                    raise ValueError("Original check score drift")
