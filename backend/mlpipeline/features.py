"""Leak-resistant relative OHLCV features for each contiguous 5m history.

Only CLOSED bars through t contribute to X(t). Missing bars reset windows.
No asset identity is included among model features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FIVE_MINUTES = pd.Timedelta(minutes=5)
FEATURE_SCHEMA_VERSION = "features_v1"
FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "ret_48",
    "volatility_12", "volatility_48", "ema_gap_12", "ema_gap_48",
    "atr_pct_14", "rsi_14", "volume_relative_12", "volume_relative_48",
    "quote_volume_relative_48", "trade_count_relative_48",
    "taker_buy_share", "body_relative", "upper_wick_relative",
    "lower_wick_relative", "close_location", "range_pct",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "btc_ret_3", "btc_volatility_12", "eth_ret_3", "eth_volatility_12",
]


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def _one_segment(segment: pd.DataFrame) -> pd.DataFrame:
    """Compute one symbol's features without crossing a missing-bar boundary."""
    segment = segment.copy()
    close = segment["close"].astype(float)
    high = segment["high"].astype(float)
    low = segment["low"].astype(float)
    opened = segment["open"].astype(float)
    vol = segment["volume_base"].astype(float)
    quote = segment["volume_quote"].astype(float)
    trades = segment["trade_count"].astype(float)

    for h in (1, 3, 6, 12, 24, 48):
        segment[f"ret_{h}"] = close.pct_change(h, fill_method=None)
    logret = np.log(close).diff()
    segment["volatility_12"] = logret.rolling(12, min_periods=12).std()
    segment["volatility_48"] = logret.rolling(48, min_periods=48).std()
    for n in (12, 48):
        ema = close.ewm(span=n, min_periods=n, adjust=False).mean()
        segment[f"ema_gap_{n}"] = close / ema - 1
    prior = close.shift(1)
    tr = pd.concat([(high-low), (high-prior).abs(), (low-prior).abs()], axis=1).max(axis=1)
    tr.iloc[0] = np.nan
    segment["atr_pct_14"] = tr.rolling(14, min_periods=14).mean() / close
    diff = close.diff()
    up = diff.clip(lower=0).rolling(14, min_periods=14).mean()
    down = (-diff.clip(upper=0)).rolling(14, min_periods=14).mean()
    segment["rsi_14"] = 100.0 * _safe_div(up, (up + down)) / 100.0  # normalized [0,1]
    segment["volume_relative_12"] = _safe_div(vol, vol.rolling(12, min_periods=12).mean())
    segment["volume_relative_48"] = _safe_div(vol, vol.rolling(48, min_periods=48).mean())
    segment["quote_volume_relative_48"] = _safe_div(quote, quote.rolling(48, min_periods=48).mean())
    segment["trade_count_relative_48"] = _safe_div(trades, trades.rolling(48, min_periods=48).mean())
    segment["taker_buy_share"] = _safe_div(segment["taker_buy_volume_base"], vol)
    span = (high-low).replace(0, np.nan)
    segment["body_relative"] = (close-opened)/span
    segment["upper_wick_relative"] = (high-pd.concat([opened, close], axis=1).max(axis=1))/span
    segment["lower_wick_relative"] = (pd.concat([opened, close], axis=1).min(axis=1)-low)/span
    segment["close_location"] = (close-low)/span
    segment["range_pct"] = (high-low)/close
    return segment


def build_symbol_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Features based on same-symbol contiguous observed M5 bars, at close t."""
    bars = candles.sort_values("bar_open_at").reset_index(drop=True).copy()
    if bars.empty:
        raise ValueError("No candles")
    if bars["symbol"].nunique() != 1:
        raise ValueError("build_symbol_features takes a single symbol")
    if bars["bar_open_at"].duplicated().any():
        raise ValueError("Duplicated candle timestamps")
    bars["segment_id"] = bars["bar_open_at"].diff().ne(FIVE_MINUTES).cumsum()
    chunks = [_one_segment(g) for _, g in bars.groupby("segment_id", sort=False)]
    result = pd.concat(chunks, ignore_index=True)
    hours = result["bar_close_at"].dt.hour + result["bar_close_at"].dt.minute/60
    dow = result["bar_close_at"].dt.dayofweek
    result["hour_sin"] = np.sin(2*np.pi*hours/24)
    result["hour_cos"] = np.cos(2*np.pi*hours/24)
    result["dow_sin"] = np.sin(2*np.pi*dow/7)
    result["dow_cos"] = np.cos(2*np.pi*dow/7)
    return result


def join_market_context(target: pd.DataFrame, btc: pd.DataFrame, eth: pd.DataFrame) -> pd.DataFrame:
    """Exact timestamp join: missing BTC/ETH context means sample is ineligible."""
    out = target.copy()
    for name, frame in (("btc", btc), ("eth", eth)):
        ctx = frame[["bar_close_at", "ret_3", "volatility_12"]].rename(
            columns={"ret_3": f"{name}_ret_3", "volatility_12": f"{name}_volatility_12"}
        )
        if ctx["bar_close_at"].duplicated().any():
            raise ValueError(f"Duplicate {name} context timestamps")
        out = out.merge(ctx, on="bar_close_at", how="left", sort=False, validate="many_to_one")
    return out


def feature_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    """Reject incomplete windows; preserve model feature ordering/dtypes."""
    return rows.loc[:, FEATURE_COLUMNS].astype("float32")
