"""Run: python -m mlpipeline.cli train --months 6 --horizon 6"""
from __future__ import annotations

import argparse

from app.core.config import get_settings
from ingestion.storage import CandleStore
from mlpipeline.trainer import DEFAULT_SYMBOLS, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="CryptoML directional LightGBM prototype")
    sub = parser.add_subparsers(dest="action", required=True)
    train = sub.add_parser("train", help="Train and evaluate one fixed horizon")
    train.add_argument("--months", type=int, default=None,
                       help="Most recent N complete monthly partitions; default all local history")
    train.add_argument("--horizon", type=int, default=6, help="M5 bars ahead, default 6=30 min")
    train.add_argument("--holdout-symbol", default="ADAUSDT", choices=DEFAULT_SYMBOLS)
    train.add_argument("--threshold-multiplier", type=float, default=0.5)
    train.add_argument("--n-estimators", type=int, default=350)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    if args.action == "train":
        if args.n_estimators < 20:
            parser.error("--n-estimators must be >= 20")
        store = CandleStore(get_settings().data_root)
        run_training(store, symbols=DEFAULT_SYMBOLS, holdout_symbol=args.holdout_symbol,
                     months=args.months, horizon=args.horizon,
                     threshold_multiplier=args.threshold_multiplier,
                     n_estimators=args.n_estimators, seed=args.seed,
                     verify_hashes=args.verify_hashes)


if __name__ == "__main__":
    main()
