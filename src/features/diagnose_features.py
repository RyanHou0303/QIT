import json
from datetime import datetime
from pathlib import Path

import polars as pl


INPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data" / "processed" / "panel_1h_features.parquet"
)


def load_feature_panel(path: Path) -> tuple[pl.DataFrame, list[str]]:
    """Read the saved panel and its explicit feature list without recomputation."""
    metadata = pl.read_parquet_metadata(path)
    if "feature_cols" not in metadata:
        raise ValueError(
            "Feature metadata is missing. Run src/features/alphas.py "
            "to rebuild panel_1h_features.parquet."
        )

    feature_cols = json.loads(metadata["feature_cols"])
    return pl.read_parquet(path), feature_cols


def diagnose_features(
    panel: pl.DataFrame,
    feature_cols: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Profile features and identify highly correlated pairs."""
    if panel.is_empty():
        raise ValueError("The diagnostic sample is empty.")
    if not feature_cols:
        raise ValueError("The feature list is empty.")

    records = []

    for name in feature_cols:
        c = pl.col(name)
        valid = c.is_finite().fill_null(False)
        finite = c.filter(valid)

        stats = panel.select(
            valid.sum().alias("valid_count"),
            c.is_null().mean().alias("null_ratio"),
            c.is_nan().sum().alias("nan_count"),
            c.is_infinite().sum().alias("inf_count"),
            pl.col("timestamp")
            .filter(valid)
            .min()
            .alias("first_valid"),
            finite.n_unique().alias("n_unique"),
            finite.mean().alias("mean"),
            finite.std().alias("std"),
            finite.min().alias("min"),
            finite.quantile(0.01).alias("p01"),
            finite.median().alias("median"),
            finite.quantile(0.99).alias("p99"),
            finite.max().alias("max"),
        ).row(0, named=True)

        records.append({"feature": name, **stats})

    profile = (
        pl.DataFrame(records)
        .sort("null_ratio", descending=True)
    )

    # Convert non-finite values to null before pairwise correlation.
    clean = panel.select([
        pl.when(pl.col(name).is_finite())
        .then(pl.col(name))
        .otherwise(None)
        .alias(name)
        for name in feature_cols
    ])

    # Each correlation uses rows where both features are available.
    corr = clean.to_pandas().corr(min_periods=1000)

    pairs = []

    for i, left in enumerate(feature_cols):
        for j in range(i + 1, len(feature_cols)):
            right = feature_cols[j]
            value = float(corr.iloc[i, j])

            if abs(value) >= 0.95:
                pairs.append({
                    "feature_a": left,
                    "feature_b": right,
                    "correlation": value,
                    "abs_correlation": abs(value),
                })

    correlated = pl.DataFrame(
        pairs,
        schema={
            "feature_a": pl.String,
            "feature_b": pl.String,
            "correlation": pl.Float64,
            "abs_correlation": pl.Float64,
        },
    ).sort("abs_correlation", descending=True)

    return profile, correlated


if __name__ == "__main__":
    # Read the precomputed artifact; never regenerate features here.
    panel, feature_cols = load_feature_panel(INPUT_PATH)

    # Use 2023 as the diagnostic training period.
    train = panel.filter(
        pl.col("is_observed")
        & (pl.col("timestamp") >= datetime(2023, 1, 1))
        & (pl.col("timestamp") < datetime(2024, 1, 1))
    )

    profile, correlated = diagnose_features(train, feature_cols)

    with pl.Config(
        tbl_rows=100,
        tbl_cols=10,
        tbl_width_chars=200,
        fmt_str_lengths=40,
    ):
        print("Feature profiles:")
        print(profile.select(
            "feature",
            "valid_count",
            "null_ratio",
            "nan_count",
            "inf_count",
            "first_valid",
            "std",
            "min",
            "max",
        ))

        print("\nHighly correlated feature pairs:")
        print(correlated)

    all_features_valid = pl.all_horizontal([
        pl.col(name).is_finite().fill_null(False)
        for name in feature_cols
    ])

    print(f"\nFeature count: {len(feature_cols)}")
    print(f"Training rows: {train.height}")
    print(
        "Rows with all features valid:",
        train.filter(all_features_valid).height,
    )
