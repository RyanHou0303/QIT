"""Evaluate cross-sectional Rank IC from a saved feature panel."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import polars as pl


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
HORIZONS = (1, 4, 12, 24)
SPLITS = {
    "train": (datetime(2023, 1, 1), datetime(2024, 1, 1)),
    "validation": (datetime(2024, 1, 1), datetime(2024, 7, 1)),
}


def summarize_ic(hourly: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    """Summarize valid ICs with equal weight per timestamp; do not annualize."""
    summary = hourly.group_by(group_cols).agg(
        pl.len().alias("candidate_timestamps"),
        pl.col("rank_ic").count().alias("valid_timestamps"),
        pl.col("rank_ic").mean().alias("mean_ic"),
        pl.col("rank_ic").std(ddof=1).alias("std_ic"),
        (pl.col("rank_ic") > 0).mean().alias("positive_ic_ratio"),
        pl.col("n_symbols")
        .filter(pl.col("rank_ic").is_not_null())
        .mean()
        .alias("mean_symbols"),
    )
    return summary.with_columns(
        (pl.col("valid_timestamps") / pl.col("candidate_timestamps"))
        .alias("coverage"),
        pl.when(pl.col("std_ic") > 1e-12)
        .then(pl.col("mean_ic") / pl.col("std_ic"))
        .otherwise(None)
        .alias("ic_ir"),
        pl.col("mean_ic").abs().alias("abs_mean_ic"),
    )


def evaluate_rank_ic(
    panel: pl.DataFrame,
    feature_cols: list[str],
    *,
    start: datetime,
    end: datetime,
    horizons: tuple[int, ...] = HORIZONS,
    min_symbols: int = 6,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return hourly, overall, and monthly cross-sectional Rank IC reports.

    Evaluate each feature/horizon on its own finite feature/label pairs.
    Ties receive average ranks. Constant cross-sections produce null ICs.
    The interval is [start, end); label endpoints must also be before end.
    Coverage is conditional on at least one valid label at that timestamp.
    Overlapping forward labels are retained; no IID t-tests are reported.
    """
    if start >= end:
        raise ValueError("start must precede end")
    if min_symbols < 3:
        raise ValueError("min_symbols must be at least 3")
    if not feature_cols or len(feature_cols) != len(set(feature_cols)):
        raise ValueError("feature_cols must be nonempty and unique")
    if not horizons or any(h not in HORIZONS for h in horizons):
        raise ValueError(f"horizons must be a nonempty subset of {HORIZONS}")
    if panel.unique(["timestamp", "symbol"]).height != panel.height:
        raise ValueError("Duplicate timestamp/symbol keys would distort IC")

    sample = panel.filter(
        pl.col("is_observed")
        & (pl.col("timestamp") >= start)
        & (pl.col("timestamp") < end)
    )
    results = []

    for h in horizons:
        target = f"fwd_res_ret_z_{h}h"
        base = sample.filter(
            pl.col(f"target_valid_{h}h")
            & pl.col(target).is_finite()
            # Purge labels whose future endpoint enters the next split.
            & ((pl.col("timestamp") + pl.duration(hours=h)) < end)
        )
        if base.is_empty():
            raise ValueError(f"No valid {h}h labels inside the requested interval")

        timestamps = base.select("timestamp").unique()

        for feature in feature_cols:
            pairs = base.filter(pl.col(feature).is_finite()).select(
                "timestamp", feature, target
            )
            by_time = pairs.group_by("timestamp").agg(
                pl.len().alias("n_symbols"),
                pl.corr(feature, target, method="spearman").alias("rank_ic"),
            )
            by_time = timestamps.join(by_time, on="timestamp", how="left")
            results.append(by_time.with_columns(
                pl.col("n_symbols").fill_null(0),
                pl.when(
                    (pl.col("n_symbols") >= min_symbols)
                    & pl.col("rank_ic").is_finite()
                )
                .then(pl.col("rank_ic"))
                .otherwise(None)
                .alias("rank_ic"),
                pl.lit(feature).alias("feature"),
                pl.lit(h).alias("horizon"),
            ))

    hourly = pl.concat(results).sort(["horizon", "feature", "timestamp"])
    summary = summarize_ic(hourly, ["feature", "horizon"]).sort(
        ["horizon", "mean_ic"], descending=[False, True], nulls_last=True
    )
    monthly = summarize_ic(
        hourly.with_columns(pl.col("timestamp").dt.truncate("1mo").alias("month")),
        ["feature", "horizon", "month"],
    ).sort(["horizon", "feature", "month"])
    return hourly, summary, monthly


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--min-symbols", type=int, default=6)
    args = parser.parse_args()

    input_path = DATA_DIR / "panel_1h_features.parquet"
    metadata = pl.read_parquet_metadata(input_path)
    if "feature_cols" not in metadata:
        raise ValueError("Run alphas.py to save the panel with feature metadata")
    feature_cols = json.loads(metadata["feature_cols"])
    panel = pl.read_parquet(input_path)
    start, end = SPLITS[args.split]

    hourly, summary, monthly = evaluate_rank_ic(
        panel, feature_cols, start=start, end=end, min_symbols=args.min_symbols
    )

    output_dir = DATA_DIR / "ic" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    hourly.write_parquet(output_dir / "rank_ic_hourly.parquet")
    summary.write_csv(output_dir / "rank_ic_summary.csv")
    monthly.write_csv(output_dir / "rank_ic_monthly.csv")

    print(f"Split: {args.split}; interval: [{start}, {end})")
    print(f"Features: {len(feature_cols)}; minimum paired symbols: {args.min_symbols}")
    for h in HORIZONS:
        print(f"\nTop 10 by mean Rank IC, highest first: {h}h")
        with pl.Config(tbl_cols=10, tbl_width_chars=180, fmt_str_lengths=40):
            print(summary.filter(pl.col("horizon") == h).select(
                "feature", "mean_ic", "ic_ir", "positive_ic_ratio",
                "valid_timestamps", "coverage", "mean_symbols",
            ).head(10))
    print(f"\nSaved reports to: {output_dir}")
    print("ICIR = mean(IC) / sample_std(IC), without annualization.")
