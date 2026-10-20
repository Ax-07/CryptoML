"""Hindsight-only, fixed barrier outcomes. Never creates signal-time features."""
from __future__ import annotations
import numpy as np
import pandas as pd
from mlpipeline.trend_pullback import M5, _validate_closed_m5, detect_pullbacks

ENTRY_DELAY_BARS=2
HOLD_BARS=6
COSTS_BPS=(2,5,10)

def research_outcomes(candles:pd.DataFrame,*,development_end:pd.Timestamp)->pd.DataFrame:
    """Evaluate only fully matured events strictly before a UTC cutoff.

    Hypothetical entry at OPEN(t+5min); stops use 1*ATR, targets 1.5*ATR.
    Intrabar stop/target tie -> stop; adverse gaps exit at actual worse open.
    Six M5 candles are inspected and absence of a barrier means timeout.
    These markouts do not establish executable fills.
    """
    if (not isinstance(development_end,pd.Timestamp) or development_end.tzinfo is None
            or str(development_end.tz)!="UTC"):
        raise ValueError("development_end must be a UTC pandas Timestamp")
    bars=_validate_closed_m5(candles)
    bars=bars.loc[bars.bar_close_at.lt(development_end)].reset_index(drop=True)
    columns=["symbol","signal_close_at","entry_at","mature_at","entry_price",
             "exit_price","atr_at_signal","exit_reason","gross_bps",
             "net_2bps","net_5bps","net_10bps"]
    if bars.empty:
        return pd.DataFrame(columns=columns)
    signals=detect_pullbacks(bars)
    result=[]
    for i in np.flatnonzero(signals.pullback_signal.to_numpy()):
        end=i+ENTRY_DELAY_BARS+HOLD_BARS-1
        if end>=len(bars):
            continue
        window=bars.iloc[i+ENTRY_DELAY_BARS:end+1]
        expected_open=signals.bar_close_at.iloc[i]+M5
        expected_mature=signals.bar_close_at.iloc[i]+(ENTRY_DELAY_BARS+HOLD_BARS-1)*M5
        if (len(window)!=HOLD_BARS or window.bar_open_at.iloc[0]!=expected_open
                or window.bar_close_at.iloc[-1]!=expected_mature
                or not window.bar_open_at.diff().iloc[1:].eq(M5).all()):
            continue
        entry=float(window.open.iloc[0])
        atr=float(signals.atr14.iloc[i])
        stop=entry-atr
        target=entry+1.5*atr
        if stop<=0:
            continue
        exit_price=float(window.close.iloc[-1])
        reason="timeout"
        for bar in window.itertuples(index=False):
            if bar.open<=stop:
                exit_price=float(bar.open);reason="gap_stop"
            elif bar.open>=target:
                exit_price=target;reason="gap_target_capped"
            elif bar.low<=stop and bar.high>=target:
                exit_price=stop;reason="both_hit_stop_first"
            elif bar.low<=stop:
                exit_price=stop;reason="stop"
            elif bar.high>=target:
                exit_price=target;reason="target"
            else:
                continue
            break
        gross=10000*(exit_price/entry-1)
        result.append({"symbol":str(signals.symbol.iloc[i]),
            "signal_close_at":signals.bar_close_at.iloc[i],
            "entry_at":window.bar_open_at.iloc[0],
            "mature_at":window.bar_close_at.iloc[-1],
            "entry_price":entry,"exit_price":exit_price,
            "atr_at_signal":atr,"exit_reason":reason,
            "gross_bps":gross,"net_2bps":gross-2,
            "net_5bps":gross-5,"net_10bps":gross-10})
    return pd.DataFrame.from_records(result,columns=columns)
