"""Fit purged walk-forward Ridge baselines and evaluate held-out Rank IC.

Run from the repository root: python -m src.models.ridge_baseline
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.evaluate_ic import HORIZONS, evaluate_rank_ic


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
TRAIN_START = datetime(2023, 1, 1)
TRAIN_END = datetime(2024, 1, 1)
VALIDATION_END = datetime(2024, 7, 1)
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
# Each pair is (expanding-training end / tuning start, tuning end).
CV_WINDOWS = (
    (datetime(2023, 7, 1), datetime(2023, 9, 1)),
    (datetime(2023, 9, 1), datetime(2023, 11, 1)),
    (datetime(2023, 11, 1), TRAIN_END),
)


def model_rows(
    panel: pl.DataFrame,
    feature_cols: list[str],
    horizon: int,
    start: datetime,
    end: datetime,
    min_symbols: int,
) -> pl.DataFrame:
    """Require complete feature vectors and purge labels crossing the interval."""
    target = f"fwd_res_ret_z_{horizon}h"
    valid_target = f"target_valid_{horizon}h"
    rows = panel.filter(
        pl.col("is_observed")
        & (pl.col("timestamp") >= start)
        & ((pl.col("timestamp") + pl.duration(hours=horizon)) < end)
        & pl.col(valid_target)
        & pl.col(target).is_finite()
        & pl.all_horizontal([
            pl.col(c).is_finite().fill_null(False) for c in feature_cols
        ])
    ).select(
        "timestamp", "symbol", "is_observed", valid_target, target, *feature_cols
    )
    rows = rows.filter(pl.len().over("timestamp") >= min_symbols)
    if rows.is_empty():
        raise ValueError(f"No complete {horizon}h model rows in [{start}, {end})")
    return rows.sort(["timestamp", "symbol"])


def fit_ridge(
    rows: pl.DataFrame, feature_cols: list[str], horizon: int, alpha: float
) -> Pipeline:
    """Fit both the scaler and Ridge exclusively on the supplied training rows."""
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="svd"))
    model.fit(
        rows.select(feature_cols).to_numpy(),
        rows[f"fwd_res_ret_z_{horizon}h"].to_numpy(),
    )
    return model


def score_predictions(
    panel: pl.DataFrame,
    predictions: pl.DataFrame,
    prediction_cols: list[str],
    horizon: int,
    start: datetime,
    end: datetime,
    min_symbols: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Evaluate common prediction rows while retaining label-eligible coverage."""
    scoring = panel.filter(
        (pl.col("timestamp") >= start) & (pl.col("timestamp") < end)
    ).select(
        "timestamp", "symbol", "is_observed",
        f"target_valid_{horizon}h", f"fwd_res_ret_z_{horizon}h",
    ).join(
        predictions.select("timestamp", "symbol", *prediction_cols),
        on=["timestamp", "symbol"], how="left", validate="1:1",
    )
    return evaluate_rank_ic(
        scoring, prediction_cols, start=start, end=end,
        horizons=(horizon,), min_symbols=min_symbols,
    )


def run_baseline(
    panel: pl.DataFrame,
    feature_cols: list[str],
    *,
    horizons: tuple[int, ...] = HORIZONS,
    alphas: tuple[float, ...] = ALPHAS,
    train_start: datetime = TRAIN_START,
    train_end: datetime = TRAIN_END,
    validation_end: datetime = VALIDATION_END,
    cv_windows: tuple[tuple[datetime, datetime], ...] = CV_WINDOWS,
    min_symbols: int = 6,
) -> dict:
    """Tune within training, refit on full training, and evaluate validation once.

    Targets are standardized residual returns, not expected percentage returns.
    The best-single comparator and its sign are selected on training data only.
    Complete-case rows are shared by Ridge and the single-feature comparator.
    """
    if not train_start < train_end < validation_end:
        raise ValueError("Require train_start < train_end < validation_end")
    if not horizons or len(set(horizons)) != len(horizons) or any(h not in HORIZONS for h in horizons):
        raise ValueError(f"horizons must be unique members of {HORIZONS}")
    if not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("alphas must contain positive finite values")
    if min_symbols < 3:
        raise ValueError("min_symbols must be at least 3")
    if not feature_cols or len(set(feature_cols)) != len(feature_cols):
        raise ValueError("Feature names must be nonempty and unique")
    if any(c.startswith(("fwd_", "btc_fwd_", "target_valid_"))
           or c in ("timestamp", "symbol", "is_observed") for c in feature_cols):
        raise ValueError("Keys and forward targets cannot be input features")
    if not cv_windows or any(
        not train_start < split < stop <= train_end for split, stop in cv_windows
    ):
        raise ValueError("Every CV training/tuning interval must lie inside training")
    if panel.unique(["timestamp", "symbol"]).height != panel.height:
        raise ValueError("Duplicate timestamp/symbol keys")

    outputs = {name: [] for name in (
        "cv_scores", "alpha_selection", "predictions", "ic_hourly",
        "ic_summary", "ic_monthly", "coefficients", "best_single",
    )}
    models = {}

    for h in horizons:
        cv_records = []
        for fold, (split, stop) in enumerate(cv_windows, start=1):
            fit_rows = model_rows(panel, feature_cols, h, train_start, split, min_symbols)
            tune_rows = model_rows(panel, feature_cols, h, split, stop, min_symbols)
            for alpha in alphas:
                model = fit_ridge(fit_rows, feature_cols, h, alpha)
                predictions = tune_rows.select("timestamp", "symbol").with_columns(
                    pl.Series("ridge", model.predict(tune_rows.select(feature_cols).to_numpy()))
                )
                _, summary, _ = score_predictions(
                    panel, predictions, ["ridge"], h, split, stop, min_symbols
                )
                metrics = summary.row(0, named=True)
                cv_records.append({
                    "horizon": h, "fold": fold, "alpha": float(alpha),
                    "fit_end": split, "tune_end": stop,
                    "fit_rows": fit_rows.height, "tune_rows": tune_rows.height,
                    "mean_ic": metrics["mean_ic"],
                    "valid_timestamps": metrics["valid_timestamps"],
                })

        cv = pl.DataFrame(cv_records)
        selection = cv.group_by("horizon", "alpha").agg(
            ((pl.col("mean_ic") * pl.col("valid_timestamps")).sum()
             / pl.col("valid_timestamps").sum()).alias("cv_mean_ic"),
            pl.col("valid_timestamps").sum().alias("valid_timestamps"),
        ).sort(["cv_mean_ic", "alpha"], descending=True, nulls_last=True)
        candidates = selection.filter(pl.col("cv_mean_ic").is_finite())
        if candidates.is_empty():
            raise ValueError(f"No valid cross-validation IC for {h}h")
        best_alpha = candidates["alpha"][0]
        selection = selection.with_columns(
            (pl.col("alpha") == best_alpha).alias("selected")
        )
        outputs["cv_scores"].append(cv)
        outputs["alpha_selection"].append(selection)

        training = model_rows(panel, feature_cols, h, train_start, train_end, min_symbols)
        validation = model_rows(panel, feature_cols, h, train_end, validation_end, min_symbols)
        model = fit_ridge(training, feature_cols, h, best_alpha)

        # Select the best single signal and its orientation using training only.
        _, single_scores, _ = evaluate_rank_ic(
            training, feature_cols, start=train_start, end=train_end,
            horizons=(h,), min_symbols=min_symbols,
        )
        best = single_scores.filter(pl.col("mean_ic").is_finite()).sort(
            ["abs_mean_ic", "feature"], descending=[True, False]
        ).row(0, named=True)
        single_name = best["feature"]
        single_sign = 1.0 if best["mean_ic"] >= 0 else -1.0
        outputs["best_single"].append(pl.DataFrame([{
            "horizon": h, "feature": single_name, "sign": single_sign,
            "training_raw_mean_ic": best["mean_ic"],
        }]))
        models[h] = {
            "model": model, "feature_cols": list(feature_cols), "horizon": h,
            "alpha": best_alpha, "best_single_feature": single_name,
            "best_single_sign": single_sign,
        }

        scaler = model.named_steps["standardscaler"]
        ridge = model.named_steps["ridge"]
        outputs["coefficients"].append(pl.DataFrame({
            "horizon": [h] * len(feature_cols), "feature": feature_cols,
            "coefficient_standardized": ridge.coef_,
            "coefficient_raw": ridge.coef_ / scaler.scale_,
            "training_mean": scaler.mean_, "training_scale": scaler.scale_,
        }))

        for split_name, rows, start, stop in (
            ("train", training, train_start, train_end),
            ("validation", validation, train_end, validation_end),
        ):
            predictions = rows.select("timestamp", "symbol").with_columns(
                pl.Series("ridge", model.predict(rows.select(feature_cols).to_numpy())),
                pl.Series("best_single", rows[single_name] * single_sign),
            )
            reports = score_predictions(
                panel, predictions, ["ridge", "best_single"], h, start, stop, min_symbols
            )
            for name, report in zip(("ic_hourly", "ic_summary", "ic_monthly"), reports):
                outputs[name].append(
                    report.rename({"feature": "model"}).with_columns(pl.lit(split_name).alias("split"))
                )
            outputs["predictions"].append(predictions.with_columns(
                pl.Series("target", rows[f"fwd_res_ret_z_{h}h"]),
                pl.lit(h).alias("horizon"), pl.lit(split_name).alias("split"),
            ))
        print(f"Finished {h}h: alpha={best_alpha:g}; best single={single_sign:+g} * {single_name}", flush=True)

    result = {name: pl.concat(frames) for name, frames in outputs.items()}
    result["models"] = models
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "panel_1h_features.parquet")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "ridge")
    parser.add_argument("--min-symbols", type=int, default=6)
    args = parser.parse_args()

    metadata = pl.read_parquet_metadata(args.input)
    if "feature_cols" not in metadata:
        raise ValueError("Run alphas.py to save the feature list in parquet metadata")
    feature_cols = json.loads(metadata["feature_cols"])
    # The final July-December 2024 holdout is not loaded or evaluated.
    panel = pl.scan_parquet(args.input).filter(
        (pl.col("timestamp") >= TRAIN_START) & (pl.col("timestamp") < VALIDATION_END)
    ).collect()
    result = run_baseline(panel, feature_cols, min_symbols=args.min_symbols)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in result.items():
        if name == "models":
            for horizon, bundle in table.items():
                joblib.dump(bundle, args.output_dir / f"ridge_{horizon}h.joblib")
        elif name in ("predictions", "ic_hourly"):
            table.write_parquet(args.output_dir / f"{name}.parquet")
        else:
            table.write_csv(args.output_dir / f"{name}.csv")
    config = {
        "input": str(args.input.resolve()), "feature_cols": feature_cols,
        "train_start": TRAIN_START.isoformat(), "train_end": TRAIN_END.isoformat(),
        "validation_end": VALIDATION_END.isoformat(), "alphas": ALPHAS,
        "cv_windows": [[a.isoformat(), b.isoformat()] for a, b in CV_WINDOWS],
        "min_symbols": args.min_symbols,
        "selection_metric": "timestamp-weighted CV mean Rank IC within 2023",
        "preprocessing": "complete cases; StandardScaler fitted only on each training window",
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("\nValidation results (single feature selected and oriented on training):")
    with pl.Config(tbl_rows=20, tbl_cols=10, tbl_width_chars=180):
        print(result["ic_summary"].filter(pl.col("split") == "validation").select(
            "horizon", "model", "mean_ic", "ic_ir", "positive_ic_ratio", "coverage", "mean_symbols"
        ).sort(["horizon", "mean_ic"], descending=[False, True]))
    print(f"\nSaved models and reports to: {args.output_dir}")
