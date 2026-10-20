"""Helpers for historical H2 H3 H6 diagnostics."""
import numpy as np
import pandas as pd
from mlpipeline.labels import add_labels
from mlpipeline.expanded_horizon_diagnostics import SEEN_SYMBOLS

HORIZONS=(2,3,6)
MODELS=("lightgbm_30","lightgbm_ema2","ema_cross_rule")
KEY=["fold_id","symbol","bar_close_at","horizon"]

def rebuild(pred,features,plan,report):
    if set(features)!=set(SEEN_SYMBOLS):
        raise ValueError("Seen symbols only")
    result=[]
    for fold in plan:
        fid=fold["fold_id"]
        end=fold["check_end"]
        selected=pred.loc[pred.fold_id.eq(fid)]
        if len(selected)!=3*fold["expected_check_rows"]:
            raise ValueError("Wrong fold size")
        for sym in SEEN_SYMBOLS:
            local=features[sym]
            local=local.loc[local.bar_close_at.ge(fold["check_start"]-pd.Timedelta(hours=6)) & local.bar_close_at.lt(end)]
            for h in HORIZONS:
                truth=add_labels(local,horizon=h,threshold_multiplier=.5)
                actual=truth[["bar_close_at","future_return","target","ret_1"]].rename(columns={
                    "future_return":"actual_return","target":"verified_class","ret_1":"previous_return"})
                frame=selected.loc[selected.symbol.eq(sym)&selected.horizon.eq(h)].merge(
                    actual,on="bar_close_at",how="left",validate="one_to_one")
                if (frame.empty or frame.actual_return.isna().any() or
                    frame.previous_return.isna().any() or
                    not np.array_equal(frame.target_3class.to_numpy(dtype=int),
                                       frame.verified_class.to_numpy(dtype=int)) or
                    not (frame.bar_close_at+pd.Timedelta(minutes=h*5)).lt(end).all()):
                    raise ValueError(f"Truth mismatch fold={fid} asset={sym} H{h}")
                result.append(frame)
    out=pd.concat(result,ignore_index=True)
    if len(out)!=len(pred) or out.duplicated(KEY).any():
        raise ValueError("Rebuilt keys drift")
    return out
