import unittest
from datetime import datetime, timedelta

import polars as pl
from scipy.stats import spearmanr

from src.features.evaluate_ic import evaluate_rank_ic


START = datetime(2023, 1, 1)
END = datetime(2023, 1, 2)


def panel_at(hour, values, labels, horizon=1):
    return pl.DataFrame({
        "timestamp": [START + timedelta(hours=hour)] * len(values),
        "symbol": [f"S{i}" for i in range(len(values))],
        "is_observed": [True] * len(values),
        "feature": values,
        f"fwd_res_ret_z_{horizon}h": labels,
        f"target_valid_{horizon}h": [True] * len(values),
    })


def evaluate(panel, horizon=1, min_symbols=3):
    return evaluate_rank_ic(
        panel, ["feature"], start=START, end=END,
        horizons=(horizon,), min_symbols=min_symbols,
    )


class RankICTests(unittest.TestCase):
    def test_ic_is_cross_sectional_instead_of_pooled_across_time(self):
        panel = pl.concat([
            panel_at(4, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
            panel_at(5, [101.0, 102.0, 103.0], [1003.0, 1002.0, 1001.0]),
        ])
        hourly, summary, monthly = evaluate(panel)
        self.assertEqual(hourly["rank_ic"].to_list(), [1.0, -1.0])
        self.assertEqual(summary["mean_ic"][0], 0.0)
        self.assertEqual(summary["positive_ic_ratio"][0], 0.5)
        self.assertEqual(monthly["mean_ic"][0], 0.0)

    def test_pairwise_finite_filtering_and_ties_match_scipy(self):
        panel = panel_at(
            4, [1.0, 1.0, 2.0, 4.0, None, 8.0],
            [3.0, 1.0, 2.0, 4.0, 5.0, float("inf")],
        )
        hourly, _, _ = evaluate(panel)
        expected = spearmanr([1.0, 1.0, 2.0, 4.0], [3.0, 1.0, 2.0, 4.0]).statistic
        self.assertEqual(hourly["n_symbols"][0], 4)
        self.assertAlmostEqual(hourly["rank_ic"][0], expected)

    def test_constant_and_empty_feature_cross_sections_are_not_zero_ic(self):
        panel = pl.concat([
            panel_at(4, [1.0, 1.0, 1.0], [1.0, 2.0, 3.0]),
            panel_at(5, [float("nan")] * 3, [1.0, 2.0, 3.0]),
        ])
        hourly, summary, _ = evaluate(panel)
        self.assertEqual(hourly["rank_ic"].null_count(), 2)
        self.assertEqual(hourly["n_symbols"].to_list(), [3, 0])
        self.assertEqual(summary["valid_timestamps"][0], 0)
        self.assertEqual(summary["coverage"][0], 0.0)
        self.assertIsNone(summary["mean_ic"][0])
        self.assertIsNone(summary["ic_ir"][0])

    def test_horizon_endpoint_cannot_cross_the_split_boundary(self):
        panel = pl.concat([
            panel_at(19, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], horizon=4),
            panel_at(20, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], horizon=4),
        ])
        hourly, _, _ = evaluate(panel, horizon=4)
        self.assertEqual(hourly.height, 1)
        self.assertEqual(hourly["timestamp"][0], START + timedelta(hours=19))

    def test_insufficient_cross_section_is_excluded(self):
        hourly, summary, _ = evaluate(
            panel_at(4, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), min_symbols=6
        )
        self.assertIsNone(hourly["rank_ic"][0])
        self.assertEqual(summary["coverage"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
