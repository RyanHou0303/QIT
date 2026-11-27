from pathlib import Path

import polars as pl


IC_DIR = Path("data/processed/ic")
KEYS = ["feature", "horizon"]

METRICS = [
    "mean_ic",
    "ic_ir",
    "positive_ic_ratio",
    "coverage",
    "valid_timestamps",
    "mean_symbols",
]


def load_ic_report(split: str) -> pl.DataFrame:
    """Load overall and monthly statistics for one evaluation split."""
    folder = IC_DIR / split

    summary = (
        pl.read_csv(folder / "rank_ic_summary.csv")
        .select(KEYS + METRICS)
        .rename({
            name: f"{split}_{name}"
            for name in METRICS
        })
    )

    # Fraction of valid months with a positive mean IC.
    monthly = (
        pl.read_csv(folder / "rank_ic_monthly.csv")
        .group_by(KEYS)
        .agg(
            pl.col("mean_ic")
            .count()
            .alias(f"{split}_valid_months"),

            (pl.col("mean_ic") > 0)
            .mean()
            .alias(f"{split}_positive_month_ratio"),
        )
    )

    return summary.join(
        monthly,
        on=KEYS,
        how="left",
        validate="1:1",
    )


def compare_ic_results() -> pl.DataFrame:
    """Compare training and validation results without changing signal signs."""
    train = load_ic_report("train")
    validation = load_ic_report("validation")

    comparison = train.join(
        validation,
        on=KEYS,
        how="full",
        coalesce=True,
        validate="1:1",
    )

    comparison = comparison.with_columns(
        # Signed difference: validation mean IC minus training mean IC.
        (
            pl.col("validation_mean_ic")
            - pl.col("train_mean_ic")
        ).alias("mean_ic_change"),

        # Zero or missing mean IC has no defined directional agreement.
        pl.when(
            (pl.col("train_mean_ic") != 0)
            & (pl.col("validation_mean_ic") != 0)
        )
        .then(
            pl.col("train_mean_ic").sign()
            == pl.col("validation_mean_ic").sign()
        )
        .otherwise(None)
        .alias("same_direction"),
    )

    # Preserve ranking based on training results.
    return comparison.sort(
        ["horizon", "train_mean_ic"],
        descending=[False, True],
        nulls_last=True,
    )


if __name__ == "__main__":
    comparison = compare_ic_results()

    display_cols = [
        "feature",
        "train_mean_ic",
        "validation_mean_ic",
        "mean_ic_change",
        "same_direction",
        "train_ic_ir",
        "validation_ic_ir",
        "train_positive_month_ratio",
        "validation_positive_month_ratio",
    ]

    with pl.Config(
        tbl_rows=40,
        tbl_cols=10,
        tbl_width_chars=220,
        fmt_str_lengths=40,
    ):
        for h in [1, 4, 12, 24]:
            print(f"\nTrain vs validation: {h}h")

            print(
                comparison
                .filter(pl.col("horizon") == h)
                .select(display_cols)
            )