"""Compare LightGBM, Ridge, and a single signal under identical time splits.

Run from the repository root: python -m src.models.lightgbm_baseline
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import polars as pl
from lightgbm import LGBMRegressor

from src.models.ridge_baseline import (
    ALPHAS, CV_WINDOWS, DATA_DIR, HORIZONS, TRAIN_END, TRAIN_START,
    VALIDATION_END, model_rows, run_baseline, score_predictions,
)


# Choose tree complexity and boosting rounds using only the inner 2023 folds.
LIGHTGBM_GRID = tuple(
    {"num_leaves": leaves, "n_estimators": trees}
    for leaves in (7, 15, 31)
    for trees in (100, 300)
)
LIGHTGBM_DEFAULTS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "min_child_samples": 200,
    "reg_lambda": 10.0,
    "max_bin": 63,
    "random_state": 42,
    "n_jobs": 4,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}


def fit_lightgbm(
    rows: pl.DataFrame, feature_cols: list[str], horizon: int, config: dict
) -> LGBMRegressor:
    """Fit raw features on training rows; tree splits do not require scaling.

    Boosting rounds are a CV candidate parameter. No external validation set
    or early-stopping callback is used during final fitting.
    """
    model = LGBMRegressor(**{**LIGHTGBM_DEFAULTS, **config})
    model.fit(
        rows.select(feature_cols).to_pandas(),
        rows[f"fwd_res_ret_z_{horizon}h"].to_numpy(),
    )
    return model


def run_comparison(
    panel: pl.DataFrame,
    feature_cols: list[str],
    *,
    horizons: tuple[int, ...] = HORIZONS,
    alphas: tuple[float, ...] = ALPHAS,
    lightgbm_grid: tuple[dict, ...] = LIGHTGBM_GRID,
    train_start: datetime = TRAIN_START,
    train_end: datetime = TRAIN_END,
    validation_end: datetime = VALIDATION_END,
    cv_windows: tuple[tuple[datetime, datetime], ...] = CV_WINDOWS,
    min_symbols: int = 6,
) -> dict:
    """Extend the existing Ridge run with LightGBM on the same complete cases."""
    if not lightgbm_grid:
        raise ValueError("The LightGBM parameter grid must not be empty")

    # Reuse the unchanged Ridge baseline, including its feature/period checks.
    result = run_baseline(
        panel, feature_cols, horizons=horizons, alphas=alphas,
        train_start=train_start, train_end=train_end, validation_end=validation_end,
        cv_windows=cv_windows, min_symbols=min_symbols,
    )
    result["models"] = {
        f"ridge_{h}h": bundle for h, bundle in result["models"].items()
    }
    additions = {name: [] for name in (
        "cv_scores", "param_selection", "predictions", "ic_hourly",
        "ic_summary", "ic_monthly", "feature_importances",
    )}

    for h in horizons:
        cv_records = []
        for fold, (split, stop) in enumerate(cv_windows, start=1):
            fit_rows = model_rows(panel, feature_cols, h, train_start, split, min_symbols)
            tune_rows = model_rows(panel, feature_cols, h, split, stop, min_symbols)
            x_tune = tune_rows.select(feature_cols).to_pandas()

            for config_id, config in enumerate(lightgbm_grid):
                model = fit_lightgbm(fit_rows, feature_cols, h, config)
                predictions = tune_rows.select("timestamp", "symbol").with_columns(
                    pl.Series("lightgbm", model.predict(x_tune))
                )
                _, summary, _ = score_predictions(
                    panel, predictions, ["lightgbm"], h, split, stop, min_symbols
                )
                metrics = summary.row(0, named=True)
                cv_records.append({
                    "horizon": h, "fold": fold, "config_id": config_id,
                    "params": json.dumps(config, sort_keys=True),
                    "fit_end": split, "tune_end": stop,
                    "fit_rows": fit_rows.height, "tune_rows": tune_rows.height,
                    "mean_ic": metrics["mean_ic"],
                    "valid_timestamps": metrics["valid_timestamps"],
                })

        cv = pl.DataFrame(cv_records)
        selection = cv.group_by("horizon", "config_id", "params").agg(
            ((pl.col("mean_ic") * pl.col("valid_timestamps")).sum()
             / pl.col("valid_timestamps").sum()).alias("cv_mean_ic"),
            pl.col("valid_timestamps").sum().alias("valid_timestamps"),
        ).sort(["cv_mean_ic", "config_id"], descending=[True, False], nulls_last=True)
        candidates = selection.filter(pl.col("cv_mean_ic").is_finite())
        if candidates.is_empty():
            raise ValueError(f"No valid LightGBM CV IC for {h}h")
        best_id = candidates["config_id"][0]
        best_config = dict(lightgbm_grid[best_id])
        selection = selection.with_columns(
            (pl.col("config_id") == best_id).alias("selected")
        )
        additions["cv_scores"].append(cv)
        additions["param_selection"].append(selection)

        training = model_rows(panel, feature_cols, h, train_start, train_end, min_symbols)
        validation = model_rows(panel, feature_cols, h, train_end, validation_end, min_symbols)
        model = fit_lightgbm(training, feature_cols, h, best_config)
        result["models"][f"lightgbm_{h}h"] = {
            "model": model, "feature_cols": list(feature_cols), "horizon": h,
            "params": model.get_params(),
        }
        additions["feature_importances"].append(pl.DataFrame({
            "horizon": [h] * len(feature_cols), "feature": feature_cols,
            "gain": model.booster_.feature_importance(importance_type="gain"),
            "split_count": model.booster_.feature_importance(importance_type="split"),
        }))

        for split_name, rows, start, stop in (
            ("train", training, train_start, train_end),
            ("validation", validation, train_end, validation_end),
        ):
            predictions = rows.select("timestamp", "symbol").with_columns(
                pl.Series("lightgbm", model.predict(rows.select(feature_cols).to_pandas()))
            )
            reports = score_predictions(
                panel, predictions, ["lightgbm"], h, start, stop, min_symbols
            )
            for name, report in zip(("ic_hourly", "ic_summary", "ic_monthly"), reports):
                additions[name].append(
                    report.rename({"feature": "model"})
                    .with_columns(pl.lit(split_name).alias("split"))
                )
            additions["predictions"].append(predictions.with_columns(
                pl.lit(h).alias("horizon"), pl.lit(split_name).alias("split"),
            ))
        print(f"Finished LightGBM {h}h: {best_config}", flush=True)

    for name in ("ic_hourly", "ic_summary", "ic_monthly"):
        result[name] = pl.concat([result[name], *additions[name]])
    light_predictions = pl.concat(additions["predictions"])
    keys = ["timestamp", "symbol", "horizon", "split"]
    if result["predictions"].select(keys).sort(keys).equals(
        light_predictions.select(keys).sort(keys)
    ) is False:
        raise ValueError("Ridge and LightGBM prediction samples differ")
    result["predictions"] = result["predictions"].join(
        light_predictions, on=keys, how="left", validate="1:1", maintain_order="left"
    )
    if not result["predictions"]["lightgbm"].is_finite().all():
        raise ValueError("LightGBM produced missing or non-finite predictions")
    result["lightgbm_cv_scores"] = pl.concat(additions["cv_scores"])
    result["lightgbm_param_selection"] = pl.concat(additions["param_selection"])
    result["lightgbm_feature_importances"] = pl.concat(additions["feature_importances"])
    result["ic_summary"] = result["ic_summary"].sort(
        ["split", "horizon", "mean_ic"], descending=[False, False, True], nulls_last=True
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "panel_1h_features.parquet")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "baselines")
    parser.add_argument("--min-symbols", type=int, default=6)
    args = parser.parse_args()

    metadata = pl.read_parquet_metadata(args.input)
    if "feature_cols" not in metadata:
        raise ValueError("Run alphas.py to save the feature list in parquet metadata")
    feature_cols = json.loads(metadata["feature_cols"])
    # Do not collect or evaluate the final July-December 2024 holdout.
    panel = pl.scan_parquet(args.input).filter(
        (pl.col("timestamp") >= TRAIN_START) & (pl.col("timestamp") < VALIDATION_END)
    ).collect()
    result = run_comparison(panel, feature_cols, min_symbols=args.min_symbols)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in result.items():
        if name == "models":
            for model_name, bundle in table.items():
                joblib.dump(bundle, args.output_dir / f"{model_name}.joblib")
        elif name in ("predictions", "ic_hourly"):
            table.write_parquet(args.output_dir / f"{name}.parquet")
        else:
            table.write_csv(args.output_dir / f"{name}.csv")
    config = {
        "input": str(args.input.resolve()), "feature_cols": feature_cols,
        "train_start": TRAIN_START.isoformat(), "train_end": TRAIN_END.isoformat(),
        "validation_end": VALIDATION_END.isoformat(), "ridge_alphas": ALPHAS,
        "cv_windows": [[a.isoformat(), b.isoformat()] for a, b in CV_WINDOWS],
        "lightgbm_defaults": LIGHTGBM_DEFAULTS, "lightgbm_grid": LIGHTGBM_GRID,
        "min_symbols": args.min_symbols,
        "selection_metric": "timestamp-weighted CV mean Rank IC within 2023",
        "preprocessing": "same complete cases; Ridge uses a training-only scaler; LightGBM uses raw features",
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("\nValidation results (parameters and single-signal direction fixed on training):")
    with pl.Config(tbl_rows=20, tbl_cols=10, tbl_width_chars=180):
        print(result["ic_summary"].filter(pl.col("split") == "validation").select(
            "horizon", "model", "mean_ic", "ic_ir", "positive_ic_ratio", "coverage", "mean_symbols"
        ))
    print(f"\nSaved models and comparison reports to: {args.output_dir}")
