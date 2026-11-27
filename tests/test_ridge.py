import contextlib
import io
import unittest
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from src.models.ridge_baseline import model_rows, run_baseline


START = datetime(2023, 1, 1)
TRAIN_END = START + timedelta(hours=120)
END = START + timedelta(hours=180)
FEATURES = ["x1", "x2"]
CV = (
    (START + timedelta(hours=48), START + timedelta(hours=72)),
    (START + timedelta(hours=72), START + timedelta(hours=96)),
)


def synthetic_panel():
    rng = np.random.default_rng(42)
    records = []
    for hour in range(180):
        x = rng.normal(size=(6, 2))
        y = x[:, 0] - x[:, 1] + rng.normal(scale=0.05, size=6)
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
        return run_baseline(
            panel, FEATURES, horizons=(1,), alphas=(1.0, 10.0),
            train_start=START, train_end=TRAIN_END, validation_end=END,
            cv_windows=CV, min_symbols=6,
        )


class RidgeBaselineTests(unittest.TestCase):
    def test_labels_crossing_training_boundary_are_purged(self):
        rows = model_rows(synthetic_panel(), FEATURES, 1, START, TRAIN_END, 6)
        self.assertLess(rows["timestamp"].max() + timedelta(hours=1), TRAIN_END)

    def test_validation_changes_cannot_affect_scaler_alpha_or_coefficients(self):
        panel = synthetic_panel()
        original = run(panel)
        changed = panel.with_columns(*[
            pl.when(pl.col("timestamp") >= TRAIN_END)
            .then(pl.col(c) * -100 + 500).otherwise(pl.col(c)).alias(c)
            for c in [*FEATURES, "fwd_res_ret_z_1h"]
        ])
        modified = run(changed)
        left, right = original["models"][1], modified["models"][1]
        self.assertEqual(left["alpha"], right["alpha"])
        self.assertTrue(original["alpha_selection"].equals(modified["alpha_selection"]))
        self.assertTrue(original["best_single"].equals(modified["best_single"]))
        for step, attr in (("standardscaler", "mean_"), ("standardscaler", "scale_"), ("ridge", "coef_")):
            np.testing.assert_allclose(
                getattr(left["model"].named_steps[step], attr),
                getattr(right["model"].named_steps[step], attr),
            )

    def test_learns_a_known_combination_and_compares_identical_samples(self):
        result = run(synthetic_panel())
        scores = result["ic_summary"].filter(pl.col("split") == "validation")
        ridge = scores.filter(pl.col("model") == "ridge").row(0, named=True)
        single = scores.filter(pl.col("model") == "best_single").row(0, named=True)
        self.assertGreater(ridge["mean_ic"], 0.9)
        self.assertGreater(ridge["mean_ic"], single["mean_ic"])
        self.assertEqual(ridge["valid_timestamps"], single["valid_timestamps"])
        self.assertEqual(ridge["mean_symbols"], single["mean_symbols"])
        coefficients = result["models"][1]["model"].named_steps["ridge"].coef_
        self.assertGreater(coefficients[0], 0)
        self.assertLess(coefficients[1], 0)

    def test_forward_label_cannot_be_a_predictor(self):
        with self.assertRaisesRegex(ValueError, "forward targets"):
            run_baseline(synthetic_panel(), ["fwd_res_ret_z_1h"])


if __name__ == "__main__":
    unittest.main()
