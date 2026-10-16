"""Forward-only, offline M5 paper replay. Research tool: never places orders.

Signals MUST have been recorded at prediction time. Outcomes are joined only
AFTER the full position plan is built, so future prices cannot choose trades.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mlpipeline.decision_diagnostics import decide

FIVE_MINUTES = pd.Timedelta(minutes=5)
HOLD = pd.Timedelta(minutes=30)
CLASSES = ("DOWN", "NEUTRAL", "UP")
SIGNAL_COLUMNS = {
    "symbol", "bar_close_at", "recorded_at", "model_trained_until",
    "model_id", "model_family", "p_down", "p_neutral", "p_up",
    "prior_down", "prior_neutral", "prior_up",
}
OUTCOME_COLUMNS = {
    "symbol", "entry_at", "exit_at", "entry_mid", "exit_mid", "observed_at",
}
FORBIDDEN_SIGNAL_COLUMNS = {
    "target", "actual_class", "future_return", "actual_return",
    "label_mature_at", "entry_mid", "exit_mid", "realized_pnl",
}


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Expected .csv or .parquet input")


def _utc(values: pd.Series, field: str) -> pd.Series:
    # Reject naive datetimes: utc=True alone would silently assume UTC.
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        out = values.dt.tz_convert("UTC")
    else:
        if not values.astype(str).str.contains(
            r"(?:Z|[+-]\d{2}:\d{2})$", case=False, regex=True
        ).all():
            raise ValueError(f"{field}: explicit UTC timezone offset required")
        out = pd.to_datetime(values, utc=True, errors="raise")
    if out.isna().any():
        raise ValueError(f"{field}: null timestamp")
    return out


def _probabilities(frame: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    values = frame.loc[:, list(names)].to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError(f"Invalid columns: {names}")
    if (values < 0).any() or (values > 1).any() or not np.allclose(
        values.sum(axis=1), 1.0, atol=1e-5
    ):
        raise ValueError(f"Invalid probability distribution: {names}")
    return values / values.sum(axis=1, keepdims=True)


def plan_positions(
    signals: pd.DataFrame, *, alpha: float, final_end: str,
    training_end: str, max_positions: int = 4,
    min_margin: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """Plan without seeing outcomes, using strictly timestamped predictions."""
    if not 0 <= alpha <= 1 or not 0 <= min_margin <= 1:
        raise ValueError("alpha and margin must be in [0,1]")
    if not isinstance(max_positions, int) or max_positions < 1:
        raise ValueError("max_positions must be positive")
    if not SIGNAL_COLUMNS.issubset(signals.columns):
        raise ValueError(f"Missing signal columns: {sorted(SIGNAL_COLUMNS - set(signals.columns))}")
    if forbidden := FORBIDDEN_SIGNAL_COLUMNS.intersection(signals.columns):
        raise ValueError(f"Future outcomes are forbidden in signal log: {sorted(forbidden)}")
    if signals.empty:
        raise ValueError("No signals to replay")
    cutoff = pd.Timestamp(final_end)
    training_cutoff = pd.Timestamp(training_end)
    if cutoff.tzinfo is None or training_cutoff.tzinfo is None:
        raise ValueError("Cutoffs must be timezone-aware UTC")
    cutoff = cutoff.tz_convert("UTC")
    training_cutoff = training_cutoff.tz_convert("UTC")
    df = signals.copy()
    for field in ("bar_close_at", "recorded_at", "model_trained_until"):
        df[field] = _utc(df[field], field)
    if df.duplicated(["symbol", "bar_close_at"]).any():
        raise ValueError("Duplicate symbol/time prediction")
    if df.symbol.isna().any() or not df.symbol.astype(str).str.fullmatch(r"[A-Z0-9]+USDT").all():
        raise ValueError("Expected uppercase USDT symbols")
    if df.model_id.isna().any() or df.model_id.nunique() != 1:
        raise ValueError("Exactly one frozen model_id required")
    if df.model_family.isna().any() or df.model_family.nunique() != 1:
        raise ValueError("Exactly one model_family required")
    if (df.bar_close_at < cutoff).any():
        raise ValueError("Signal overlaps the previously unsealed final test")
    if (df.model_trained_until > training_cutoff).any():
        raise ValueError("Model training boundary extends into final holdout")
    if (df.model_trained_until >= df.bar_close_at).any():
        raise ValueError("Model trained on or after the signal timestamp")
    entry = df.bar_close_at + FIVE_MINUTES
    if ((df.recorded_at < df.bar_close_at) | (df.recorded_at >= entry)).any():
        raise ValueError("Signal not recorded after close and before assumed entry")
    probs = _probabilities(df, ("p_down", "p_neutral", "p_up"))
    priors = _probabilities(df, ("prior_down", "prior_neutral", "prior_up"))
    if (priors <= 0).any() or not np.allclose(priors, priors[:1], atol=1e-10):
        raise ValueError("One strictly positive frozen training prior required")
    predictions = decide(probs, priors[0], alpha)
    scores = probs * np.power(priors, -alpha)
    scores = scores / scores.sum(axis=1, keepdims=True)
    sorted_scores = np.sort(scores, axis=1)
    margin = sorted_scores[:, -1] - sorted_scores[:, -2]

    df["entry_at"] = entry
    df["exit_at"] = entry + HOLD
    df["predicted_class"] = predictions
    df["adjusted_margin"] = margin
    df = df.sort_values(["entry_at", "symbol"], kind="stable").reset_index(drop=True)

    active: list[tuple[str, pd.Timestamp]] = []
    rows: list[dict] = []
    reasons: dict[str, int] = {}
    for row in df.itertuples(index=False):
        active = [(symbol, end) for symbol, end in active if end > row.entry_at]
        reason = None
        if row.predicted_class == 1:
            reason = "neutral"
        elif row.adjusted_margin < min_margin:
            reason = "low_margin"
        elif any(symbol == row.symbol for symbol, _ in active):
            reason = "same_symbol_busy"
        elif len(active) >= max_positions:
            reason = "portfolio_full"
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        active.append((row.symbol, row.exit_at))
        rows.append({
            "symbol": row.symbol, "bar_close_at": row.bar_close_at,
            "recorded_at": row.recorded_at, "entry_at": row.entry_at,
            "exit_at": row.exit_at,
            "model_id": row.model_id,
            "model_family": row.model_family,
            "predicted_class": int(row.predicted_class),
            "side": "LONG" if row.predicted_class == 2 else "SHORT",
            "adjusted_margin": float(row.adjusted_margin),
            "p_down": float(row.p_down), "p_neutral": float(row.p_neutral),
            "p_up": float(row.p_up),
        })
    if not rows:
        raise ValueError("No trade plan after abstention/capacity filters")
    return pd.DataFrame(rows), {
        "signals_total": len(df), "trades_planned": len(rows),
        "abstentions": reasons,
        "model_id": str(df.model_id.iloc[0]),
        "model_family": str(df.model_family.iloc[0]),
        "training_priors": dict(zip(CLASSES, [float(x) for x in priors[0]])),
        "alpha": float(alpha), "min_margin": float(min_margin),
        "max_positions": max_positions,
        "first_signal": df.bar_close_at.min().isoformat(),
        "last_signal": df.bar_close_at.max().isoformat(),
    }


def settle_positions(
    plan: pd.DataFrame, outcomes: pd.DataFrame, *,
    fee_bps: float = 5, slippage_bps: float = 5,
    notional_usdt: float = 100,
) -> tuple[pd.DataFrame, dict]:
    """Join outcome prices only AFTER signal-only position selection.

    Entry is next M5 close after prediction; exit six M5 bars after entry.
    This is a simplified midprice + fixed execution-cost approximation.
    """
    if not np.isfinite([fee_bps, slippage_bps, notional_usdt]).all():
        raise ValueError("Costs and notional must be finite")
    if fee_bps < 0 or slippage_bps < 0 or slippage_bps >= 10000 or notional_usdt <= 0:
        raise ValueError("Invalid costs or notional")
    if not OUTCOME_COLUMNS.issubset(outcomes.columns):
        raise ValueError(f"Missing outcome columns: {sorted(OUTCOME_COLUMNS - set(outcomes.columns))}")
    obs = outcomes.copy()
    for field in ("entry_at", "exit_at", "observed_at"):
        obs[field] = _utc(obs[field], field)
    if obs.duplicated(["symbol", "entry_at"]).any():
        raise ValueError("Duplicate observed symbol/entry")
    obs["entry_mid"] = pd.to_numeric(obs.entry_mid, errors="raise")
    obs["exit_mid"] = pd.to_numeric(obs.exit_mid, errors="raise")
    prices = obs[["entry_mid", "exit_mid"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Prices must be finite and strictly positive")
    if (obs.observed_at < obs.exit_at).any():
        raise ValueError("Outcome recorded before the exit price was known")
    merged = plan.merge(
        obs, how="left", on=["symbol", "entry_at"],
        suffixes=("", "_observed"), validate="one_to_one", indicator=True,
    )
    if (merged["_merge"] != "both").any():
        raise ValueError("Missing outcome for at least one planned trade")
    if (merged.exit_at != merged.exit_at_observed).any():
        raise ValueError("Outcome exit differs from the scheduled 30-minute exit")
    s = slippage_bps / 10000
    f = fee_bps / 10000
    direction = np.where(merged.side.eq("LONG"), 1.0, -1.0)
    # Long buys at ask and sells at bid; short sells at bid and covers at ask.
    entry_exec = merged.entry_mid.to_numpy(dtype=float) * (1 + direction * s)
    exit_exec = merged.exit_mid.to_numpy(dtype=float) * (1 - direction * s)
    quantity = notional_usdt / entry_exec
    gross = quantity * direction * (exit_exec - entry_exec)
    fees = f * (notional_usdt + quantity * exit_exec)
    merged["quantity"] = quantity
    merged["gross_usdt_after_slippage"] = gross
    merged["fees_usdt"] = fees
    merged["net_pnl_usdt"] = gross - fees
    merged["notional_usdt"] = notional_usdt
    merged = merged.drop(columns=["_merge", "exit_at_observed"])
    closed = merged.sort_values(["exit_at", "symbol"]).reset_index(drop=True)
    pnl = closed.net_pnl_usdt.to_numpy(dtype=float)
    equity = np.cumsum(pnl)
    peaks = np.maximum.accumulate(np.r_[0.0, equity])[1:]
    max_drawdown = float((peaks - equity).max(initial=0.0))
    return closed, {
        "trades": len(closed),
        "total_notional_usdt": float(notional_usdt * len(closed)),
        "net_pnl_usdt": float(pnl.sum()),
        "mean_net_return_pct_of_trade_notional": float(pnl.mean() / notional_usdt * 100),
        "win_fraction_after_costs": float((pnl > 0).mean()),
        "max_closed_trade_drawdown_usdt": max_drawdown,
        "fee_bps_each_side": fee_bps,
        "slippage_bps_each_side": slippage_bps,
        "notional_usdt_each_trade": notional_usdt,
        "per_symbol": {
            symbol: {
                "trades": len(part),
                "net_pnl_usdt": float(part.net_pnl_usdt.sum()),
                "win_fraction": float((part.net_pnl_usdt > 0).mean()),
            }
            for symbol, part in closed.groupby("symbol")
        },
        "limitations": (
            "Retrospective paper replay only; entry/exit at recorded M5 midprices "
            "with fixed spread/slippage approximations; no depth, latency, "
            "funding, borrow costs, liquidation, or mark-to-market drawdown."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline forward-only paper replay; no brokerage connection")
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--frozen-final-report", type=Path, required=True)
    parser.add_argument("--model-family", choices=("lightgbm", "logistic"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--notional-usdt", type=float, default=100.0)
    parser.add_argument("--max-positions", type=int, default=4)
    parser.add_argument("--min-margin", type=float, default=0.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory already exists; refusing to overwrite a replay")
    frozen = json.loads(args.frozen_final_report.read_text(encoding="utf-8"))
    if frozen.get("status") != "FINAL_UNSEALED_EVALUATED":
        parser.error("Expected completed and frozen final research protocol")
    audit = frozen["audit"]
    if audit.get("protocol") != "frozen_cpu_final_holdout_v1":
        parser.error("Unsupported frozen protocol")
    alpha = float(frozen["chosen_alphas_on_pre_final_tuning_only"][args.model_family]["alpha"])
    signals = _read_table(args.signals)
    if not signals.empty and not signals.model_family.eq(args.model_family).all():
        parser.error("Signal model_family differs from frozen candidate")
    plan, planning = plan_positions(
        signals, alpha=alpha,
        final_end=audit["boundaries_utc"]["final_end"],
        training_end=audit["boundaries_utc"]["final_start"],
        max_positions=args.max_positions, min_margin=args.min_margin,
    )
    trades, settled = settle_positions(
        plan, _read_table(args.outcomes),
        fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
        notional_usdt=args.notional_usdt,
    )
    report = {
        "status": "OFFLINE_PAPER_REPLAY_ONLY",
        "model_family": args.model_family,
        "post_final_cutoff_utc": audit["boundaries_utc"]["final_end"],
        "source_dataset_fingerprint_sha256": audit["dataset_fingerprint_sha256"],
        "planning": planning,
        "settlement": settled,
        "never_connected_to_exchange": True,
        "disclaimer": "Not forward-live verified unless signals were genuinely logged at prediction time.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    trades.to_parquet(args.output_dir / "paper_trades.parquet", index=False)
    print(f"Paper replay: {args.output_dir.resolve()}")
    print(f"Simulated trades: {settled['trades']}")
    print(f"Net PnL (USDT, illustrative only): {settled['net_pnl_usdt']:.4f}")


if __name__ == "__main__":
    main()
