"""Backtest saved model scores with a simple neutral allocation baseline."""

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from scipy.stats import rankdata

from src.backtest.engine import BacktestConfig, run_backtest
from src.models.ridge_baseline import DATA_DIR, HORIZONS, TRAIN_END, VALIDATION_END


def parse_utc_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.astimezone(timezone.utc).replace(tzinfo=None) if parsed.tzinfo else parsed


def predict_scores(
    features: pl.DataFrame, bundle: dict, model_name: str, start: datetime, end: datetime
) -> pl.DataFrame:
    """Infer without reading future labels or target-validity flags."""
    feature_cols = bundle["feature_cols"]
    rows = features.filter(
        pl.col("is_observed")
        & (pl.col("timestamp") >= start) & (pl.col("timestamp") < end)
        & pl.all_horizontal([pl.col(c).is_finite().fill_null(False) for c in feature_cols])
    )
    if rows.is_empty():
        raise ValueError("No rows with complete live features")
    if model_name == "best_single":
        scores = rows[bundle["best_single_feature"]].to_numpy() * bundle["best_single_sign"]
    elif model_name == "ridge":
        scores = bundle["model"].predict(rows.select(feature_cols).to_numpy())
    elif model_name == "lightgbm":
        scores = bundle["model"].predict(rows.select(feature_cols).to_pandas())
    else:
        raise ValueError(f"Unknown model: {model_name}")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite model scores")
    return rows.select("timestamp", "symbol").with_columns(pl.Series("score", scores))


def build_target_weights(
    scores: pl.DataFrame,
    features: pl.DataFrame,
    *,
    start: datetime,
    end: datetime,
    rebalance_hours: int,
    gross: float = 1.0,
    max_position: float = 0.20,
    min_symbols: int = 6,
) -> pl.DataFrame:
    """Project centered score ranks onto dollar/BTC-beta-neutral weights.

    A zero-score BTC hedge is allowed even when BTC has no model prediction.
    Scaling down enforces both limits; gross exposure may be below the budget.
    Missing scheduled scores/betas lead to an explicit flat target.
    This is a transparent allocation baseline, not a risk/turnover optimizer.
    """
    if rebalance_hours < 1 or not isinstance(rebalance_hours, int):
        raise ValueError("rebalance_hours must be a positive integer")
    if not start < end or gross <= 0 or max_position <= 0 or min_symbols < 3:
        raise ValueError("Invalid allocation settings")
    if scores.unique(["timestamp", "symbol"]).height != scores.height:
        raise ValueError("Duplicate score timestamp/symbol keys")
    symbols = features["symbol"].unique().sort().to_list()
    index = {symbol: i for i, symbol in enumerate(symbols)}
    by_time = {group["timestamp"][0]: group for group in scores.partition_by("timestamp")}
    beta_by_time = {
        group["timestamp"][0]: group for group in features.select(
            "timestamp", "symbol", "ex_ante_btc_beta"
        ).partition_by("timestamp")
    }
    records = []
    timestamp = start
    while timestamp < end:
        weights = np.zeros(len(symbols))
        beta = np.full(len(symbols), np.nan)
        if timestamp in beta_by_time:
            for row in beta_by_time[timestamp].iter_rows(named=True):
                value = row["ex_ante_btc_beta"]
                if value is not None and np.isfinite(value):
                    beta[index[row["symbol"]]] = value
        signal = np.zeros(len(symbols))
        available = []
        if timestamp in by_time:
            for row in by_time[timestamp].iter_rows(named=True):
                i = index[row["symbol"]]
                if row["score"] is not None and np.isfinite(row["score"]) and np.isfinite(beta[i]):
                    available.append((i, row["score"]))
        if len(available) >= min_symbols:
            indices, values = zip(*available)
            ranks = rankdata(values, method="average")
            signal[list(indices)] = ranks - ranks.mean()
            active = np.isfinite(beta)
            constraints = np.vstack([np.ones(active.sum()), beta[active]])
            projected = signal[active] - np.linalg.pinv(constraints) @ (constraints @ signal[active])
            size = np.abs(projected).sum()
            if size > 1e-10:
                scale = min(gross / size, max_position / np.abs(projected).max())
                weights[active] = projected * scale
        if abs(weights.sum()) > 1e-9 or abs(weights @ np.nan_to_num(beta)) > 1e-9:
            raise RuntimeError("Neutrality projection failed")
        for i, symbol in enumerate(symbols):
            records.append({
                "timestamp": timestamp, "symbol": symbol, "weight": weights[i],
                "signal_beta": beta[i] if np.isfinite(beta[i]) else None,
            })
        timestamp += timedelta(hours=rebalance_hours)
    return pl.DataFrame(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("ridge", "lightgbm", "best_single"),
                        default=["ridge", "lightgbm", "best_single"])
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument("--start", type=parse_utc_time, default=TRAIN_END)
    parser.add_argument("--end", type=parse_utc_time, default=VALIDATION_END)
    parser.add_argument("--rebalance-hours", type=int, default=None,
                        help="Default: rebalance every horizon hours")
    parser.add_argument("--fee-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--gross", type=float, default=1.0)
    parser.add_argument("--max-position", type=float, default=0.20)
    parser.add_argument("--min-symbols", type=int, default=6)
    parser.add_argument("--model-dir", type=Path, default=DATA_DIR / "baselines")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "backtests")
    args = parser.parse_args()
    if args.start < TRAIN_END or args.end > VALIDATION_END:
        parser.error("This baseline uses models trained through 2023; evaluate only January-June 2024")
    if (args.start >= args.end or args.start.minute or args.start.second or args.start.microsecond
            or args.end.minute or args.end.second or args.end.microsecond):
        parser.error("Use an increasing interval with whole-hour UTC boundaries")
    config = BacktestConfig(
        fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
        max_gross=args.gross, max_position=args.max_position,
    )
    # Endpoint OHLC/funding settle the last June bar; no July trading signal is used.
    features = pl.scan_parquet(DATA_DIR / "panel_1h_features.parquet").filter(
        (pl.col("timestamp") >= args.start) & (pl.col("timestamp") <= args.end)
    ).collect()
    market = features.filter(pl.col("timestamp") > args.start).select(
        "timestamp", "symbol", "perp_open", "perp_close",
        "funding_payment_rate", "ex_ante_btc_beta",
    )
    expected_hours = int((args.end - args.start).total_seconds() / 3600)
    if market["timestamp"].n_unique() != expected_hours:
        raise ValueError("Market data do not cover the requested evaluation interval")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for h in args.horizons:
        for name in args.models:
            source = "ridge" if name == "best_single" else name
            bundle = joblib.load(args.model_dir / f"{source}_{h}h.joblib")
            scores = predict_scores(features, bundle, name, args.start, args.end)
            rebalance = args.rebalance_hours if args.rebalance_hours is not None else h
            weights = build_target_weights(
                scores, features, start=args.start, end=args.end,
                rebalance_hours=rebalance, gross=args.gross,
                max_position=args.max_position, min_symbols=args.min_symbols,
            )
            result = run_backtest(market, weights, config)
            folder = args.output_dir / f"{name}_{h}h"
            folder.mkdir(parents=True, exist_ok=True)
            scores.write_parquet(folder / "scores.parquet")
            weights.write_parquet(folder / "target_weights.parquet")
            for field in ("equity", "positions", "trades"):
                result[field].write_parquet(folder / f"{field}.parquet")
            summary = result["summary"].with_columns(
                pl.lit(name).alias("model"), pl.lit(h).alias("horizon"),
                pl.lit(rebalance).alias("rebalance_hours"),
            )
            summary.write_csv(folder / "summary.csv")
            run_config = {
                **asdict(config), "model": name, "horizon": h,
                "start": args.start.isoformat(), "end": args.end.isoformat(),
                "rebalance_hours": rebalance, "min_symbols": args.min_symbols,
                "model_file": str((args.model_dir / f"{source}_{h}h.joblib").resolve()),
                "allocation": "centered ranks projected to dollar/beta neutrality; uniform scaling to caps",
                "execution": "next hourly bar open; funding at candle close before terminal liquidation",
                "funding_price": "perp_close proxy for mark price",
            }
            (folder / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
            summaries.append(summary)
            print(f"Finished {name} {h}h, rebalance={rebalance}h", flush=True)
    combined = pl.concat(summaries).sort(["horizon", "total_return"], descending=[False, True])
    combined.write_csv(args.output_dir / "summary.csv")
    with pl.Config(tbl_rows=20, tbl_cols=12, tbl_width_chars=200):
        print(combined.select(
            "horizon", "model", "rebalance_hours", "total_return", "sharpe",
            "max_drawdown", "price_pnl", "funding_pnl", "fees", "slippage",
        ))
    print(f"Saved backtest reports to: {args.output_dir}")
