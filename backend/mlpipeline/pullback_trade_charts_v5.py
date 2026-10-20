"""Standalone HTML candlestick charts from actual archived Binance M5 candles.

Positions are HYPOTHETICAL BACKTEST selections, NOT exchange executions.
Shows model C0 Ridge9 short4m selected May/June 2026 trades on real
BTC/ETH/SOL/XRP candles, plus train-period B outcomes if desired.
Does not read ADA nor July-September reserved data. No matplotlib needed.
"""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import json

import numpy as np
import pandas as pd

from ingestion.storage import CandleStore
from mlpipeline.pullback_long_history_v3 import SYMBOLS, load_long_development_history
from mlpipeline.pullback_global_ridge_v3 import (
    START, SHORT_START, TRAIN_END, APRIL_END, MAY_END, CUTOFF,
    V1_FEATURES, _partition, _fit_fixed_ridge, prepare_symbol,
)

def select_c0(events):
    """Exact C0 decision: fit 4mo, calibrate April *scores* (not labels)."""
    all_events=pd.concat([events[s] for s in SYMBOLS],ignore_index=True)
    train=_partition(all_events, SHORT_START, TRAIN_END)
    train=train.loc[train.baseline_b_confirmed]
    april=_partition(all_events,TRAIN_END,APRIL_END)
    april=april.loc[april.baseline_b_confirmed]
    model=_fit_fixed_ridge(train,V1_FEATURES)
    threshold=max(0.,float(np.quantile(model.predict(
        april.loc[:,V1_FEATURES].to_numpy(dtype=float)),0.80)))
    chosen=[]
    for start,end in ((APRIL_END,MAY_END),(MAY_END,CUTOFF)):
        window=_partition(all_events,start,end)
        pool=window.loc[window.baseline_b_confirmed].copy()
        if len(pool):
            pool["model_score"]=model.predict(pool.loc[:,V1_FEATURES].to_numpy(dtype=float))
            chosen.append(pool.loc[pool.model_score>threshold])
    result=pd.concat(chosen,ignore_index=True) if chosen else all_events.iloc[:0]
    return result,threshold


def actual_exit_time(bars, event):
    """Replay fixed OHLC barrier to determine the observed exit candle.

    Outcome label only stores exit price/reason, not exit_at. Replicates the
    research_outcomes stop-first and six-bar convention exactly.
    """
    entry=event.entry_at
    atr=float(event.atr_at_signal)
    ep=float(event.entry_price)
    stop=ep-atr
    target=ep+1.5*atr
    window=bars.loc[(bars.bar_open_at>=entry)&
                    (bars.bar_open_at<entry+pd.Timedelta(minutes=30))]
    if len(window)!=6:
        raise ValueError("Expected six contiguous M5 candles")
    reason="timeout"
    exit_price=float(window.close.iloc[-1])
    exit_time=window.bar_close_at.iloc[-1]
    for row in window.itertuples(index=False):
        if row.open<=stop:
            reason,exit_price="gap_stop",float(row.open)
        elif row.open>=target:
            reason,exit_price="gap_target_capped",target
        elif row.low<=stop and row.high>=target:
            reason,exit_price="both_hit_stop_first",stop
        elif row.low<=stop:
            reason,exit_price="stop",stop
        elif row.high>=target:
            reason,exit_price="target",target
        else:
            continue
        exit_time=row.bar_close_at
        break
    if (reason!=event.exit_reason or
            not np.isclose(exit_price,float(event.exit_price),atol=1e-7,rtol=1e-9)):
        raise AssertionError("Replayed exit differs from frozen research outcome")
    return exit_time,stop,target


def _svg(bars,event,width=1050,height=420):
    exit_at,stop,target=actual_exit_time(bars,event)
    i=bars.index[bars.bar_close_at==event.signal_close_at]
    if len(i)!=1:
        raise ValueError("Signal candle missing in verified M5 data")
    loc=bars.index.get_loc(i[0])
    visible=bars.iloc[max(0,loc-45):min(len(bars),loc+29)].reset_index(drop=True)
    if visible.empty:
        raise ValueError("No visible candles")
    left,right,top,bottom=62,18,24,45
    plotw,ploth=width-left-right,height-top-bottom
    ylo=min(float(visible.low.min()),stop,float(event.entry_price),float(event.exit_price))
    yhi=max(float(visible.high.max()),target,float(event.entry_price),float(event.exit_price))
    pad=(yhi-ylo)*0.09 if yhi>ylo else 1
    ylo-=pad; yhi+=pad
    px=lambda index:left+(index+0.5)*plotw/len(visible)
    py=lambda value:top+(yhi-value)/(yhi-ylo)*ploth
    tickw=max(2.0,min(8.0,plotw/len(visible)*0.57))
    parts=[f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Bougies Binance M5 et position hypothétique">',
           f'<rect width="{width}" height="{height}" fill="#101820"/>']
    for j in range(6):
        price=ylo+(yhi-ylo)*j/5
        y=py(price)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#344352" stroke-width="0.6"/>')
        parts.append(f'<text x="8" y="{y+4:.2f}" font-size="11" fill="#a1abb6">{price:.3f}</text>')
    for idx,bar in enumerate(visible.itertuples(index=False)):
        x=px(idx)
        up=bar.close>=bar.open
        color="#4bc29d" if up else "#ee767a"
        parts.append(f'<line x1="{x:.2f}" y1="{py(bar.high):.2f}" x2="{x:.2f}" y2="{py(bar.low):.2f}" stroke="{color}" stroke-width="1.3"/>')
        y1,y2=sorted([py(bar.open),py(bar.close)])
        parts.append(f'<rect x="{x-tickw/2:.2f}" y="{y1:.2f}" width="{tickw:.2f}" height="{max(y2-y1,1):.2f}" fill="{color}"/>')
    for price,name,col in ((stop,"STOP","#ee767a"),(target,"TARGET","#4bc29d"),(float(event.entry_price),"ENTRÉE","#eab764")):
        y=py(price)
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.2f}" y2="{y:.2f}" stroke="{col}" stroke-dasharray="7,5" opacity="0.8"/>')
        parts.append(f'<text x="{width-right-5}" y="{y-5:.2f}" text-anchor="end" font-size="11" fill="{col}">{name} {price:.3f}</text>')
    for stamp,name,col in ((event.signal_close_at,"SIGNAL","#78aaff"),(event.entry_at,"ENTRÉE","#eab764"),(exit_at,"SORTIE","#f9fcff")):
        idxs=np.flatnonzero(visible.bar_close_at.eq(stamp).to_numpy())
        if not len(idxs) and name=="ENTRÉE":
            idxs=np.flatnonzero(visible.bar_open_at.eq(stamp).to_numpy())
        if len(idxs):
            x=px(int(idxs[0]))
            parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" stroke="{col}" stroke-dasharray="3,5"/>')
            parts.append(f'<text x="{x+4:.2f}" y="{top+12}" font-size="12" fill="{col}">{name}</text>')
    parts.append(f'<text x="{left}" y="{height-9}" font-size="12" fill="#abb4c2">{escape(str(visible.bar_close_at.iloc[0]))} — {escape(str(visible.bar_close_at.iloc[-1]))} UTC · M5</text>')
    parts.append('</svg>')
    return "".join(parts),exit_at


def generate_charts(data,events,mode):
    chosen,threshold=select_c0(events)
    if mode=="model":
        if len(chosen)!=84:
            raise ValueError(f"C0 frozen may-june expected 84 selections, observed {len(chosen)}; do not silently alter")
        examples=pd.concat([
            chosen.nlargest(2,"net_5bps"),chosen.nsmallest(2,"net_5bps"),
            chosen.loc[chosen.exit_reason.eq("timeout")].head(1),
            chosen.sort_values("net_5bps").iloc[[len(chosen)//2]],
        ]).drop_duplicates(subset=["symbol","signal_close_at"]).head(6)
    else:
        pool=pd.concat([events[s] for s in SYMBOLS],ignore_index=True)
        train=_partition(pool,SHORT_START,TRAIN_END)
        pool=train.loc[train.baseline_b_confirmed].copy()
        examples=pd.concat([
            pool.nlargest(2,"net_5bps"),pool.nsmallest(2,"net_5bps"),
            pool.loc[pool.exit_reason.eq("timeout")].head(1),
            pool.sort_values("net_5bps").iloc[[len(pool)//2]]
        ]).drop_duplicates(subset=["symbol","signal_close_at"]).head(6)
    cards=[]
    for _,item in examples.iterrows():
        bars=data[str(item.symbol)]
        svg,exit_at=_svg(bars,item)
        net=float(item.net_5bps)
        color="#4bc29d" if net>=0 else "#ee767a"
        details=f'<p>Signal : {escape(str(item.signal_close_at))} · Entrée : {escape(str(item.entry_at))} · Sortie : {escape(str(exit_at))} · Motif : <b>{escape(str(item.exit_reason))}</b></p>'
        details+=f'<p>Prix réel des bougies · entrée simulée {float(item.entry_price):.4f} · sortie simulée {float(item.exit_price):.4f} · brut {float(item.gross_bps):+.2f} bp · <strong style="color:{color}">net 5 bp : {net:+.2f} bp</strong></p>'
        if "model_score" in item:
            details+=f'<p>Score Ridge prédit : {float(item.model_score):+.3f} bp · seuil fixe : {threshold:+.3f} bp</p>'
        cards.append(f'<section><h2>{escape(str(item.symbol))} · {escape(str(item.signal_close_at))}</h2>{svg}{details}</section>')
    return cards,len(chosen),threshold


def main():
    p=argparse.ArgumentParser(description="Candle charts of frozen simulated P4 decisions")
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--output-html",type=Path,required=True)
    p.add_argument("--mode",choices=("model","training"),default="model")
    a=p.parse_args()
    root=a.data_root.resolve()
    output=a.output_html.resolve()
    if output.exists() or not output.parent.is_dir() or output.is_relative_to(root):
        p.error("Use a NEW html file outside original market data")
    candles,_=load_long_development_history(CandleStore(root))
    events={}
    for symbol in SYMBOLS:
        events[symbol],_=prepare_symbol(candles[symbol])
    cards,n,threshold=generate_charts(candles,events,a.mode)
    if not cards:
        raise ValueError("No examples")
    title="Ridge C0 : positions sélectionnées mai–juin 2026" if a.mode=="model" else "Exemples B utilisés pour entraînement (déc 2025–mars 2026)"
    doc='''<!doctype html><html lang="fr"><meta charset="utf-8"><title>CryptoML · Exemples</title>
    <style>body{font-family:system-ui,Arial;background:#0a1119;color:#e1ebf4;max-width:1150px;margin:32px auto;padding:0 20px}
    section{background:#16202b;margin:22px 0;padding:20px;border-radius:14px}section svg{width:100%;height:auto}
    p{color:#c3cbd6;line-height:1.5} h1{font-size:26px}h2{font-size:19px} .note{background:#263442;padding:14px;border-radius:10px}</style>
    <h1>'''+escape(title)+'''</h1><p class="note">BOUGIES HISTORIQUES BINANCE RÉELLES · POSITIONS DU BACKTEST SIMULÉES, JAMAIS EXÉCUTÉES.
    Les traits montrent signal, entrée OPEN(t+2), barrière stop 1 ATR, target 1.5 ATR, sortie retrouvée.
    Les prix OHLC sont réels, les fills au niveau barrière restent des hypothèses.</p>'''+''.join(cards)+'</html>'
    with output.open("x",encoding="utf-8") as handle:
        handle.write(doc)
    print("Mode:",a.mode,"C0 selected trades (May-Jun):",n,"threshold:",threshold)
    print("HTML charts:",output)


if __name__=="__main__":
    main()
