import contextlib
import io
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from src.models.lightgbm_baseline import run_comparison


START = datetime(2023, 1, 1)
TRAIN_END = START + timedelta(hours=120)
END = START + timedelta(hours=180)
FEATURES = ["x1", "x2"]
CV = (
    (START + timedelta(hours=48), START + timedelta(hours=72)),
    (START + timedelta(hours=72), START + timedelta(hours=96)),
)
GRID = tuple({
    "num_leaves": leaves, "n_estimators": 80, "min_child_samples": 5,
    "learning_rate": 0.1, "reg_lambda": 1.0, "n_jobs": 1,
} for leaves in (7, 15))


def synthetic_panel():
    rng = np.random.default_rng(2025)
    records = []
    for hour in range(180):
        x = rng.uniform(-2, 2, size=(6, 2))
        y = x[:, 0] * x[:, 1] + rng.normal(scale=0.01, size=6)
        y = (y - y.mean()) / y.std(ddof=1)
        for symbol in range(6):
            records.append({
                "timestamp": START + timedelta(hours=hour), "symbol": f"S{symbol}",
                "is_observed": True, "target_valid_1h": True,
                "fwd_res_ret_z_1h": y[symbol], "x1": x[symbol, 0], "x2": x[symbol, 1],
            })
    return pl.DataFrame(records)


def run(panel):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_comparison(
            panel, FEATURES, horizons=(1,), alphas=(1.0,), lightgbm_grid=GRID,
            train_start=START, train_end=TRAIN_END, validation_end=END,
            cv_windows=CV, min_symbols=6,
        )


class LightGBMBaselineTests(unittest.TestCase):
    def test_comparison_has_three_models_on_identical_samples(self):
        result = run(synthetic_panel())
        scores = result["ic_summary"].filter(pl.col("split") == "validation")
        self.assertEqual(set(scores["model"]), {"ridge", "best_single", "lightgbm"})
        for column in ("valid_timestamps", "mean_symbols", "coverage"):
            self.assertEqual(scores[column].n_unique(), 1)
        predictions = result["predictions"]
        self.assertTrue(predictions["lightgbm"].is_finite().all())
        validation = predictions.filter(pl.col("split") == "validation")
        self.assertLess(validation["timestamp"].max() + timedelta(hours=1), END)

    def test_learns_nonlinear_signal_and_model_roundtrip(self):
        panel = synthetic_panel()
        result = run(panel)
        scores = result["ic_summary"].filter(pl.col("split") == "validation")
        light_ic = scores.filter(pl.col("model") == "lightgbm")["mean_ic"][0]
        ridge_ic = scores.filter(pl.col("model") == "ridge")["mean_ic"][0]
        self.assertGreater(light_ic, 0.6)
        self.assertGreater(light_ic, ridge_ic)
        model = result["models"]["lightgbm_1h"]["model"]
        x = panel.select(FEATURES).head(12).to_pandas()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
        np.testing.assert_allclose(model.predict(x), restored.predict(x))

    def test_validation_values_cannot_change_selected_parameters_or_trees(self):
        panel = synthetic_panel()
        original = run(panel)
        changed = panel.with_columns(*[
            pl.when(pl.col("timestamp") >= TRAIN_END)
            .then(pl.col(c) * -10 + 100).otherwise(pl.col(c)).alias(c)
            for c in [*FEATURES, "fwd_res_ret_z_1h"]
        ])
        modified = run(changed)
        self.assertTrue(original["lightgbm_param_selection"].equals(modified["lightgbm_param_selection"]))
        left = original["models"]["lightgbm_1h"]["model"].booster_.model_to_string()
        right = modified["models"]["lightgbm_1h"]["model"].booster_.model_to_string()
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
