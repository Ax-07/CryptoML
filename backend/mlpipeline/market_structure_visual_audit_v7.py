"""P4 V7: audit visual causal du contexte M5/M15/H1, sans labels.

Observations sur vraies bougies historiques Binance, jamais des ordres.
Sélection fixe: cas ETH déjà connu + premiers LONG/SHORT UP/DOWN/RANGE
de mai 2026, indépendamment des rendements. Aucune modification V6.
"""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ingestion.storage import CandleStore
from mlpipeline.dataset import manifest_fingerprint
from mlpipeline.market_structure_v6 import (
    REGIMES, classify_context, classify_m5, closed_higher_candles,
)
from mlpipeline.pullback_long_history_v3 import (
    CUTOFF, SYMBOLS, load_long_development_history,
)
from mlpipeline.trend_pullback import M5, _segment_features

PROTOCOL = "p4_v7_causal_context_visual_audit_no_outcomes_v1"
KNOWN_CASE = ("ETHUSDT", pd.Timestamp("2026-06-06T17:00:00Z"))
MAY_START = pd.Timestamp("2026-05-01T00:00:00Z")
MAY_END = pd.Timestamp("2026-06-01T00:00:00Z")
# Not chosen by trades' results. These are *examples*, not performance samples.
PREDECLARED = (
    ("Premier LONG avec UP concordant", "long_signal", "UP"),
    ("Premier SHORT avec DOWN concordant", "short_signal", "DOWN"),
    ("Premier LONG classe RANGE", "long_signal", "RANGE"),
    ("Premier SHORT classe RANGE", "short_signal", "RANGE"),
)
VIEW_BARS = {"M5": 168, "M15": 112, "H1": 84}
BAR_FACTORS = {"M5": 1, "M15": 3, "H1": 12}


def select_cases(features: dict[str, pd.DataFrame]) -> list[dict]:
    """Predeclared case study and first chronological candidate per bucket."""
    if set(features) != set(SYMBOLS):
        raise ValueError("Expected four approved symbols, never ADA")
    eth = features[KNOWN_CASE[0]]
    eth_rows = eth.loc[eth.bar_close_at.eq(KNOWN_CASE[1])]
    if len(eth_rows) != 1:
        raise ValueError("Known ETH signal timestamp missing or duplicated")
    chosen = [{
        "selection": "ETH 6 juin — étude de cas déjà inspectée",
        "symbol": KNOWN_CASE[0],
        "signal_close_at": KNOWN_CASE[1],
        "expected_side": "LONG",
    }]
    combined = pd.concat([
        features[s].loc[
            features[s].bar_close_at.ge(MAY_START)
            & features[s].bar_close_at.lt(MAY_END),
            ["symbol", "bar_close_at", "market_regime", "long_signal", "short_signal"],
        ] for s in SYMBOLS
    ], ignore_index=True)
    combined = combined.sort_values(["bar_close_at", "symbol"]).reset_index(drop=True)
    for title, direction, regime in PREDECLARED:
        matches = combined.loc[
            combined[direction] & combined.market_regime.eq(regime)
        ]
        if matches.empty:
            # Preserve missing category; never replace with an attractive case.
            chosen.append({
                "selection": title, "symbol": None, "signal_close_at": None,
                "expected_side": "LONG" if direction=="long_signal" else "SHORT",
                "requested_regime": regime, "missing": True,
            })
            continue
        first = matches.iloc[0]
        chosen.append({
            "selection": title, "symbol": str(first.symbol),
            "signal_close_at": first.bar_close_at,
            "expected_side": "LONG" if direction=="long_signal" else "SHORT",
            "requested_regime": regime,
        })
    return chosen


def verify_reference(reference: dict, fingerprint: str,
                     features: dict[str, pd.DataFrame]) -> None:
    """All frozen V6 census figures must match before any examples are rendered."""
    if (reference.get("protocol") != "p4_v6_m15_h1_symmetric_long_short_unlabeled_v1"
            or reference.get("source_dataset_fingerprint_sha256") != fingerprint
            or tuple(reference.get("assets", ())) != SYMBOLS
            or set(features) != set(SYMBOLS)):
        raise ValueError("V6 reference protocol, dataset fingerprint or assets mismatch")
    required = (
        "long_signal", "short_signal", "long_with_trend", "short_with_trend",
        "regime_up", "regime_down", "regime_range", "regime_unknown",
        "closed_bars",
    )
    counts = {}
    for sym in SYMBOLS:
        f = features[sym]
        if not f.bar_close_at.lt(CUTOFF).all():
            raise ValueError("Protected period not permitted")
        counts[sym] = {
            "closed_bars":len(f),
            "long_signal":int(f.long_signal.sum()),
            "short_signal":int(f.short_signal.sum()),
            "long_with_trend":int(f.long_with_trend.sum()),
            "short_with_trend":int(f.short_with_trend.sum()),
            **{"regime_"+r.lower():int(f.market_regime.eq(r).sum()) for r in REGIMES},
        }
        recorded=reference["by_asset"][sym]["total"]
        for key in required:
            if counts[sym][key] != recorded[key]:
                raise AssertionError(f"V6 reference count mismatch: {sym}/{key}")
    for key in required:
        if sum(counts[s][key] for s in SYMBOLS) != reference["total"][key]:
            raise AssertionError(f"V6 total count mismatch: {key}")


def _indicators(bars: pd.DataFrame, factor: int) -> pd.DataFrame:
    """Recreate visual EMA9/26 using same reset-at-gap semantics as V6."""
    if bars.empty:
        return bars.copy()
    delta = factor*M5
    data = bars.copy().sort_values("bar_close_at").reset_index(drop=True)
    data["_seg"] = data.bar_open_at.diff().ne(delta).cumsum()
    return pd.concat([
        _segment_features(block.reset_index(drop=True))
        for _,block in data.groupby("_seg", sort=False)
    ],ignore_index=True)


def _html_svg(bars: pd.DataFrame, timeframe: str,
              asof: pd.Timestamp, last_closed: pd.Timestamp) -> str:
    """Static SVG: NO candle later than the signal is allowed into the view."""
    if timeframe not in VIEW_BARS:
        raise ValueError("Unsupported timeframe")
    if last_closed > asof:
        raise AssertionError("Chart includes future higher timeframe")
    cut=bars.loc[bars.bar_close_at.le(last_closed)].tail(VIEW_BARS[timeframe])
    if cut.empty or cut.bar_close_at.gt(asof).any():
        raise ValueError("Empty or future candle data in chart")
    width,height=1060,286
    l,r,t,b=78,22,24,48
    pw,ph=width-l-r,height-t-b
    hi=float(cut.high.max())
    lo=float(cut.low.min())
    for name in ("ema9","ema26"):
        finite=cut[name].dropna()
        if len(finite):
            hi=max(hi,float(finite.max()));lo=min(lo,float(finite.min()))
    margin=max((hi-lo)*0.07, max(hi,1.)*0.00001)
    lo-=margin;hi+=margin
    x=lambda i:l+(i+0.5)*pw/len(cut)
    y=lambda p:t+(hi-float(p))/(hi-lo)*ph
    w=max(1.5,min(8.,pw/len(cut)*0.65))
    out=[f'<svg viewBox="0 0 {width} {height}" role="img" '
         f'aria-label="Chandeliers {timeframe} historiques closes avant le signal">',
         f'<rect width="{width}" height="{height}" fill="#101b29"/>']
    for k in range(5):
        price=lo+(hi-lo)*k/4
        yp=y(price)
        out.append(f'<line x1="{l}" x2="{width-r}" y1="{yp:.1f}" y2="{yp:.1f}" stroke="#33465b"/>')
        out.append(f'<text x="6" y="{yp+4:.1f}" font-size="12" fill="#9aafc1">{price:.4f}</text>')
    for i,row in enumerate(cut.itertuples(index=False)):
        xx=x(i)
        color="#39ba94" if row.close>=row.open else "#f28283"
        out.append(f'<line x1="{xx:.2f}" x2="{xx:.2f}" y1="{y(row.high):.2f}" y2="{y(row.low):.2f}" '
                   f'stroke="{color}" stroke-width="1.4"/>')
        yo,yc=sorted((y(row.open),y(row.close)))
        out.append(f'<rect x="{xx-w/2:.2f}" y="{yo:.2f}" width="{w:.2f}" '
                   f'height="{max(1.,yc-yo):.2f}" fill="{color}"/>')
    for column,color in (("ema9","#f3bf5d"),("ema26","#6cb5fa")):
        prev=None
        for i,value in enumerate(cut[column].to_numpy(dtype=float)):
            if not np.isfinite(value):
                prev=None;continue
            point=(x(i),y(value))
            if prev:
                out.append(f'<line x1="{prev[0]:.2f}" y1="{prev[1]:.2f}" '
                           f'x2="{point[0]:.2f}" y2="{point[1]:.2f}" '
                           f'stroke="{color}" stroke-width="1.6" opacity="0.95"/>')
            prev=point
    last_x=x(len(cut)-1)
    out.append(f'<line x1="{last_x:.2f}" y1="{t}" x2="{last_x:.2f}" y2="{height-b}" '
               f'stroke="#eee" stroke-dasharray="3,3"/>')
    out.append(f'<text x="{l}" y="{height-13}" fill="#b9c7d6" font-size="12">'
               f'{escape(cut.bar_close_at.iloc[0].strftime("%Y-%m-%d %H:%M"))} — '
               f'{escape(last_closed.strftime("%Y-%m-%d %H:%M"))} UTC · dernière clôture connue</text>')
    out.append('</svg>')
    return "".join(out)


def _asof_context(frame: pd.DataFrame, latest_closed: pd.Timestamp,
                  at: pd.Timestamp, factor: int, expected_regime: str) -> dict:
    """Get the exact closed M15/H1 state that V6's backward asof should use."""
    matches=frame.loc[frame.bar_close_at.eq(latest_closed)]
    if len(matches)!=1:
        raise ValueError("Missing complete higher timeframe close")
    age=at-latest_closed
    if not (pd.Timedelta(0)<=age<M5*factor):
        raise AssertionError("Unavailable or stale higher timeframe context")
    row=matches.iloc[0]
    if row.regime!=expected_regime:
        raise AssertionError("Visual context differs from frozen V6 context")
    def finite_or_none(value):
        return float(value) if pd.notna(value) and np.isfinite(float(value)) else None
    return {
        "regime":str(row.regime),
        "closed_at":latest_closed.isoformat(),
        "efficiency_er12":finite_or_none(row.efficiency),
        "ema9_minus_ema26_div_atr14":finite_or_none(row.spread_atr),
        "age_minutes":float(age/pd.Timedelta(minutes=1)),
    }


def render_case(case: dict, bars: pd.DataFrame, features: pd.DataFrame,
                higher: dict[str,pd.DataFrame], contexts: dict[str,pd.DataFrame]) -> tuple[dict,str]:
    """Three prices-only charts and exact as-of V6 labels; no future candles."""
    symbol=case["symbol"]
    t=case["signal_close_at"]
    row=features.loc[features.bar_close_at.eq(t)]
    if len(row)!=1:
        raise ValueError("Signal absent du recensement V6")
    row=row.iloc[0]
    side=case["expected_side"]
    if ((side=="LONG" and not bool(row.long_signal))
            or (side=="SHORT" and not bool(row.short_signal))):
        raise AssertionError("Expected directional signal missing")
    summary={
        "selection":case["selection"],
        "symbol":symbol,
        "signal_close_at":t.isoformat(),
        "signal_side":side,
        "long_signal":bool(row.long_signal),
        "short_signal":bool(row.short_signal),
        "market_regime":str(row.market_regime),
        "long_with_trend":bool(row.long_with_trend),
        "short_with_trend":bool(row.short_with_trend),
        "timeframes":{},
    }
    if case.get("requested_regime") and row.market_regime!=case["requested_regime"]:
        raise AssertionError("Category changed since predeclared selection")
    grids=[]
    for tf,factor in BAR_FACTORS.items():
        if tf=="M5":
            last=t
            view=_indicators(bars.loc[bars.bar_close_at.le(t)],1)
            state=None
        else:
            key=tf.lower()
            expected=str(row["regime_"+key])
            known=row["context_closed_at_"+key]
            if pd.isna(known):
                if expected!="UNKNOWN":
                    raise AssertionError("No closed higher candle but known V6 regime")
                summary["timeframes"][tf]={"regime":"UNKNOWN","closed_at":None}
                grids.append(f'<div class="pane"><h3>{tf} · UNKNOWN</h3><p>Aucune bougie clôturée disponible.</p></div>')
                continue
            last=known
            if last>t:
                raise AssertionError("Future data in context timestamp")
            latest=contexts[tf].loc[contexts[tf].bar_close_at.eq(last)]
            if (expected=="UNKNOWN" and
                    (t-last>=M5*factor or latest.empty)):
                summary["timeframes"][tf]={"regime":"UNKNOWN","closed_at":None,
                                           "stale_context_closed_at":last.isoformat()}
            else:
                summary["timeframes"][tf]=_asof_context(
                    contexts[tf],last,t,factor,expected)
            view=_indicators(higher[tf].loc[higher[tf].bar_close_at.le(last)],factor)
            state=expected
        # Zero future bars admitted, even when the backing store contains more.
        if view.empty:
            raise AssertionError(f"No {tf} candles to display")
        svg=_html_svg(view,tf,t,last)
        metric=summary["timeframes"].get(tf,{})
        er=metric.get("efficiency_er12")
        spread=metric.get("ema9_minus_ema26_div_atr14")
        additional=""
        if tf!="M5":
            additional=(f"ER12={er:.3f}" if er is not None else "ER12=non disponible")
            additional+=" · "+(f"écart EMA/ATR={spread:.3f}" if spread is not None else "écart EMA/ATR=non disponible")
        grids.append(f'<div class="pane"><h3>{tf} · {escape(state or "SIGNAL M5")} · '
                     f'clôture {escape(last.strftime("%Y-%m-%d %H:%M"))} UTC</h3>'
                     f'{svg}<p>{escape(additional)}</p></div>')
    summary["timeframes"]["M5"]={
        "signal_close_at":t.isoformat(),
        "candles_after_signal_in_charts":0,
    }
    badge="aligné" if (row.long_with_trend or row.short_with_trend) else "non aligné"
    html=(f'<section><h2>{escape(case["selection"])}</h2>'
          f'<p><b>{escape(symbol)}</b> · signal {escape(t.isoformat())} '
          f'· {escape(side)} · contexte global <strong>{escape(str(row.market_regime))}</strong> '
          f'· {badge}</p>'
          + "".join(grids) + '</section>')
    return summary,html


def build_audit(candles: dict[str,pd.DataFrame],
                reference:dict,fingerprint:str) -> tuple[dict,str]:
    if set(candles)!=set(SYMBOLS):
        raise ValueError("Only four approved symbols")
    features={}
    for symbol in SYMBOLS:
        print("Context and candidates:",symbol,flush=True)
        features[symbol]=classify_m5(candles[symbol])
    verify_reference(reference,fingerprint,features)
    cases=select_cases(features)
    selected_symbols={c["symbol"] for c in cases if c.get("symbol")}
    selected_higher={}
    selected_context={}
    for sym in selected_symbols:
        selected_higher[sym]={}
        selected_context[sym]={}
        for tf,factor in BAR_FACTORS.items():
            if tf=="M5":continue
            h=closed_higher_candles(candles[sym],factor)
            selected_higher[sym][tf]=h
            selected_context[sym][tf]=classify_context(h,factor)
    result_cases=[]
    fragments=[]
    for c in cases:
        if c.get("missing"):
            result_cases.append({
                "selection":c["selection"],"missing":True,
                "reason":"Aucun candidat de cette catégorie en mai 2026",
            })
            fragments.append(f'<section><h2>{escape(c["selection"])}</h2>'
                             '<p>Aucun candidat dans la fenêtre fixée (mai 2026). '
                             'Aucune substitution.</p></section>')
            continue
        item,html=render_case(
            c,candles[c["symbol"]],features[c["symbol"]],
            selected_higher[c["symbol"]],selected_context[c["symbol"]])
        result_cases.append(item)
        fragments.append(html)
    report={
        "protocol":PROTOCOL,
        "source_dataset_fingerprint_sha256":fingerprint,
        "v6_reference_protocol":reference["protocol"],
        "unlabeled_no_returns_no_orders":True,
        "selection_rule":"ETH fixed known case + first chronological May-2026 candidate per fixed class",
        "warning":"RANGE includes transitions/disagreement. Real P0 OHLC, backtest-only signals, not live orders.",
        "cases":result_cases,
    }
    title="CryptoML — V7 Audit visuel du contexte M5 / M15 / H1"
    html='''<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>CryptoML V7 · Contexte MTF</title><style>
    body{font:15px/1.55 system-ui,sans-serif;background:#0d1420;color:#e6edf5;margin:0 auto;max-width:1180px;padding:22px}
    .note{padding:15px 18px;border:1px solid #41546a;border-radius:10px;background:#1b2b3b}
    section{margin:26px 0;padding:19px;background:#152334;border-radius:14px}
    .pane{padding:12px 0;border-top:1px solid #3b4c5d}.pane svg{width:100%;height:auto}
    h2{font-size:20px}h3{font-size:15px;color:#c9dcf0;margin-bottom:5px}
    p{color:#bfcddd}strong{color:#fff}code{color:#9ac2f1}
    </style><h1>'''+escape(title)+'''</h1>
    <div class="note"><b>Lecture descriptive, pas une exécution :</b> chandeliers Binance historiques réels.
    Aucun PnL, retour, label, frais ou ordre. Chaque graphique se termine à la dernière
    bougie effectivement <b>clôturée à l'heure du signal</b>. EMA9 jaune, EMA26 bleu.
    <b>RANGE</b> est une classe résiduelle (range, désaccord, transition ou tendance trop faible),
    et non une preuve de consolidation horizontale.</div>'''+''.join(fragments)+'</html>'
    return report,html


def main():
    p=argparse.ArgumentParser(description="P4 V7 causal M5/M15/H1 visual audit; no labels")
    p.add_argument("--data-root",required=True,type=Path)
    p.add_argument("--reference-json",required=True,type=Path,
                   help="Previously saved V6 unlabeled census JSON")
    p.add_argument("--output-html",required=True,type=Path)
    p.add_argument("--output-json",required=True,type=Path)
    a=p.parse_args()
    root=a.data_root.resolve()
    ref=a.reference_json.resolve()
    html_path,json_path=a.output_html.resolve(),a.output_json.resolve()
    if (not ref.is_file() or ref.is_relative_to(root)
            or html_path==json_path or html_path.parent!=json_path.parent
            or not html_path.parent.is_dir() or
            any(p.exists() or p.is_relative_to(root) or p==ref
                for p in (html_path,json_path))):
        p.error("Existing external reference JSON and two NEW outputs in same folder outside P0 required")
    with ref.open("r",encoding="utf-8") as f:
        reference=json.load(f)
    candles,manifests=load_long_development_history(CandleStore(root))
    fingerprint=manifest_fingerprint(manifests)
    report,html=build_audit(candles,reference,fingerprint)
    # exclusive new files; no overwrites, never inside protected P0
    with html_path.open("x",encoding="utf-8") as f:
        f.write(html)
    with json_path.open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    for c in report["cases"]:
        print(c["selection"],"MISSING" if c.get("missing") else
              f"{c['symbol']} {c['signal_close_at']} => {c['market_regime']}")
        if not c.get("missing"):
            for tf in ("M15","H1"):
                print(" ",tf,c["timeframes"].get(tf))
    print("NO RETURNS / NO MODEL TUNING / NO FUTURE CANDLES / NO ORDERS")
    print("Saved",html_path,json_path)


if __name__=="__main__":
    main()
