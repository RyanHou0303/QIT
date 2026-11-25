from pathlib import Path

import polars as pl

HORIZONS = [1, 4, 12, 24]
BETA_WINDOW = 168  # Seven days of hourly returns, including the current close.
BETA_VAR_EPS = 1e-12
LABEL_STD_EPS = 1e-12


def compute_targets(df: pl.DataFrame, *, beta_window: int = BETA_WINDOW) -> pl.DataFrame:
    """Build close-time targets on a complete hourly grid without filling prices.

    Beta at t uses returns available through t. Signals become available after
    that close. Keep the grid intact for rolling features; filter training last.
    Forward labels must never be used as features.
    """
    if beta_window < 2:
        raise ValueError("beta_window must be at least 2")
    if df.unique(["symbol", "timestamp"]).height != df.height:
        raise ValueError("Duplicate symbol/timestamp keys")
    if "BTCUSDT" not in df["symbol"].unique():
        raise ValueError("BTCUSDT prices are required for residual targets")

    df = (
        df.sort(["symbol", "timestamp"])
        .with_columns(pl.lit(True).alias("is_observed"))
        .upsample(
            time_column="timestamp", every="1h", group_by="symbol",
            maintain_order=True,
        )
        .with_columns(pl.col("is_observed").fill_null(False))
        .sort(["symbol", "timestamp"])
    )
    df = df.with_columns(
        (pl.col("perp_close") / pl.col("perp_close").shift(1).over("symbol") - 1.0)
        .alias("ret_1h"),
        *[
            (pl.col("perp_close").shift(-h).over("symbol") / pl.col("perp_close") - 1.0)
            .alias(f"fwd_ret_{h}h")
            for h in HORIZONS
        ],
    )

    btc_columns = ["ret_1h", *[f"fwd_ret_{h}h" for h in HORIZONS]]
    btc = (
        df.filter(pl.col("symbol") == "BTCUSDT")
        .select("timestamp", *btc_columns)
        .rename({c: f"btc_{c}" for c in btc_columns})
    )
    df = df.join(btc, on="timestamp", how="left").sort(["symbol", "timestamp"])

    # Covariance and variance must use the same observations.
    paired = pl.col("ret_1h").is_finite() & pl.col("btc_ret_1h").is_finite()
    df = df.with_columns(
        pl.when(paired).then(pl.col("ret_1h")).alias("_paired_ret"),
        pl.when(paired).then(pl.col("btc_ret_1h")).alias("_paired_btc_ret"),
    )
    covariance = pl.rolling_cov(
        "_paired_ret", "_paired_btc_ret",
        window_size=beta_window, min_samples=beta_window,
    ).over("symbol")
    variance = pl.col("_paired_btc_ret").rolling_var(
        window_size=beta_window, min_samples=beta_window,
    ).over("symbol")
    df = df.with_columns(
        pl.when(variance > BETA_VAR_EPS)
        .then(covariance / (variance + BETA_VAR_EPS))
        .otherwise(None)
        .clip(0.1, 3.0)
        .alias("ex_ante_btc_beta")
    ).drop("_paired_ret", "_paired_btc_ret")

    for h in HORIZONS:
        raw = f"fwd_res_ret_raw_{h}h"
        centered = f"fwd_res_ret_{h}h"
        normalized = f"fwd_res_ret_z_{h}h"
        df = df.with_columns(
            (pl.col(f"fwd_ret_{h}h")
             - pl.col("ex_ante_btc_beta") * pl.col(f"btc_fwd_ret_{h}h"))
            .alias(raw)
        )
        # Preserve centered-return columns; add z-scores separately.
        df = df.with_columns(
            (pl.col(raw) - pl.col(raw).mean().over("timestamp")).alias(centered)
        )
        std = pl.col(raw).std(ddof=1).over("timestamp")
        df = df.with_columns(
            pl.when(std > LABEL_STD_EPS)
            .then(pl.col(centered) / std)
            .otherwise(None)
            .alias(normalized)
        )
        df = df.with_columns(
            (pl.col("is_observed")
             & pl.col("ex_ante_btc_beta").is_finite()
             & pl.col(normalized).is_finite())
            .fill_null(False)
            .alias(f"target_valid_{h}h")
        )
    return df


def select_training_rows(
    df: pl.DataFrame, *, horizons: tuple[int, ...] = tuple(HORIZONS)
) -> pl.DataFrame:
    """Filter after rolling features; require all requested targets.

    horizons=(1,) selects a 1h model; the default selects a multi-horizon model.
    Feature validity and split-boundary purging are separate checks.
    """
    if not horizons or any(h not in HORIZONS for h in horizons):
        raise ValueError(f"horizons must be a nonempty subset of {HORIZONS}")
    return df.filter(pl.all_horizontal([pl.col(f"target_valid_{h}h") for h in horizons]))


def build_research_panel():
    df = compute_targets(pl.read_parquet("data/processed/panel_1h_raw.parquet"))
    out_path = Path("data/processed/panel_1h_targets.parquet")
    print(df.columns)
    df.write_parquet(out_path)
    print(f"Built research panel with residual targets: {df.shape} -> {out_path}")
    for h in HORIZONS:
        print(f"Valid {h}h target rows: {df[f'target_valid_{h}h'].sum()}")



if __name__ == "__main__":
    build_research_panel()
