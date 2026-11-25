
import numpy as np
import polars as pl


EPS = 1e-9
MOM_WINDOWS = [4, 12, 24, 72, 168]
FLOW_WINDOWS = [1, 4, 12, 24]


def cs_zscore(
    col_name: str,
    clip_val: float = 3.5,
) -> pl.Expr:
    """Standardize across symbols at each timestamp and clip extreme values."""
    c = pl.col(col_name)

    z = (
        (c - c.mean().over("timestamp"))
        / (c.std().over("timestamp") + EPS)
    )

    return z.clip(-clip_val, clip_val)


def add_alpha_families(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str]]:
    """
    Build momentum and trade-flow features.

    Returns the feature panel and a list of feature column names.
    The input must retain a complete hourly grid; missing values are not filled.

    Implemented:
    - Raw and residual momentum, including volatility-adjusted versions.
    - Spot and perpetual aggressive trade-flow imbalance.
    - Perpetual-minus-spot flow imbalance.

    Funding and basis features are not implemented yet.
    """
    df = df.sort(["symbol", "timestamp"])

    # Use the previous timestamp's beta to remove the BTC contribution
    # from the most recent hourly return. Missing beta remains null.
    expr_beta_residual = (
        pl.col("ret_1h")
        - pl.col("ex_ante_btc_beta").shift(1).over("symbol")
        * pl.col("btc_ret_1h")
    ).alias("res_ret_1h")

    df = df.with_columns(expr_beta_residual)

    # Momentum features
    expr_momentum = []
    expr_raw_momentum = []

    for w in MOM_WINDOWS:
        # Sum hourly residual returns over the lookback window.
        # This is not an exact compounded return.
        res_momentum = (
            pl.col("res_ret_1h")
            .rolling_sum(window_size=w, min_samples=w)
            .over("symbol")
        )

        res_volatility = (
            pl.col("res_ret_1h")
            .rolling_std(window_size=w, min_samples=w)
            .over("symbol")
        )

        expr_momentum.extend([
            res_momentum.alias(f"res_mom_{w}h"),

            # Approximate window volatility using hourly volatility * sqrt(w).
            pl.when(res_volatility > EPS)
            .then(res_momentum / (res_volatility * np.sqrt(w)))
            .otherwise(None)
            .alias(f"res_mom_vol_adj_{w}h"),
        ])

        # Calculate the cumulative raw return from window endpoint prices.
        raw_momentum = (
            pl.col("perp_close")
            / pl.col("perp_close").shift(w).over("symbol")
            - 1.0
        )

        raw_volatility = (
            pl.col("ret_1h")
            .rolling_std(window_size=w, min_samples=w)
            .over("symbol")
        )

        expr_raw_momentum.extend([
            raw_momentum.alias(f"raw_mom_{w}h"),

            pl.when(raw_volatility > EPS)
            .then(raw_momentum / (raw_volatility * np.sqrt(w)))
            .otherwise(None)
            .alias(f"raw_mom_vol_adj_{w}h"),
        ])

    df = df.with_columns(expr_momentum + expr_raw_momentum)

    # Trade-flow features
    expr_signed_flow = []

    for market in ["perp", "spot"]:
        # Aggressive buy notional minus aggressive sell notional.
        # Quote volume is measured in USDT.
        expr_signed_flow.append(
            (
                2.0 * pl.col(f"{market}_taker_buy_quote_vol")
                - pl.col(f"{market}_quote_vol")
            ).alias(f"{market}_signed_flow")
        )

    # Materialize intermediate columns before referencing them below.
    df = df.with_columns(expr_signed_flow)

    expr_flow = []

    for market in ["perp", "spot"]:
        for w in FLOW_WINDOWS:
            signed_flow = (
                pl.col(f"{market}_signed_flow")
                .rolling_sum(window_size=w, min_samples=w)
                .over("symbol")
            )

            total_volume = (
                pl.col(f"{market}_quote_vol")
                .rolling_sum(window_size=w, min_samples=w)
                .over("symbol")
            )

            # Aggregate notionals before dividing to preserve volume weighting.
            # Positive values indicate buy-side aggression.
            # Leave the feature null when total volume is negligible.
            expr_flow.append(
                pl.when(total_volume > EPS)
                .then(signed_flow / total_volume)
                .otherwise(None)
                .alias(f"{market}_flow_imb_{w}h")
            )

    df = df.with_columns(expr_flow)

    # Cross-market flow differences
    # Positive values indicate stronger buy-side imbalance in perpetuals.
    expr_flow_gap = [
        (
            pl.col(f"perp_flow_imb_{w}h")
            - pl.col(f"spot_flow_imb_{w}h")
        ).alias(f"perp_spot_flow_gap_{w}h")
        for w in FLOW_WINDOWS
    ]

    df = df.with_columns(expr_flow_gap)

    # Return final feature names; retain residual returns and signed flows
    # in the panel as intermediate columns.
    feature_exprs = (
        expr_momentum
        + expr_raw_momentum
        + expr_flow
        + expr_flow_gap
    )
    feature_cols = [
        expr.meta.output_name()
        for expr in feature_exprs
    ]

    return df, feature_cols
