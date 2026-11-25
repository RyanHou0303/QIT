import math
import unittest
from datetime import datetime, timedelta

import polars as pl

from data.build_panel import compute_targets, select_training_rows


def price_panel(hours=60):
    rows = []
    for symbol, beta, noise in (("BTCUSDT", 1.0, 0.0), ("ETHUSDT", 1.4, 0.002),
                                ("SOLUSDT", 0.8, -0.003)):
        price = 100.0
        for hour in range(hours):
            btc_return = 0.003 * math.sin(hour * 0.8) + 0.001 * math.cos(hour * 0.3)
            price *= 1.0 + beta * btc_return + noise * math.cos(hour * 0.7)
            rows.append({"symbol": symbol, "timestamp": datetime(2023, 1, 1) + timedelta(hours=hour),
                         "perp_close": price})
    return pl.DataFrame(rows)


class TargetTests(unittest.TestCase):
    def test_gap_keeps_clock_horizons_and_beta_requires_recovery(self):
        raw = price_panel()
        gap = datetime(2023, 1, 1, 10)
        result = compute_targets(raw.filter(pl.col("timestamp") != gap), beta_window=4)
        eth = result.filter(pl.col("symbol") == "ETHUSDT").sort("timestamp")
        original = raw.filter(pl.col("symbol") == "ETHUSDT").sort("timestamp")
        self.assertFalse(eth["is_observed"][10])
        self.assertIsNone(eth["perp_close"][10])
        self.assertIsNone(eth["fwd_ret_1h"][9])
        self.assertIsNone(eth["ret_1h"][11])
        self.assertAlmostEqual(eth["fwd_ret_4h"][9], original["perp_close"][13] / original["perp_close"][9] - 1)
        self.assertEqual(eth["ex_ante_btc_beta"][10:15].null_count(), 5)
        self.assertIsNotNone(eth["ex_ante_btc_beta"][15])
        self.assertFalse(eth["target_valid_1h"][10])

    def test_beta_warmup_and_known_slope(self):
        raw = price_panel().filter(pl.col("symbol") == "BTCUSDT")
        prices = raw["perp_close"].to_list()
        eth_prices = [100.0]
        for previous, current in zip(prices, prices[1:]):
            eth_prices.append(eth_prices[-1] * (1 + 2 * (current / previous - 1)))
        eth = raw.with_columns(pl.lit("ETHUSDT").alias("symbol"), pl.Series("perp_close", eth_prices))
        result = compute_targets(pl.concat([raw, eth]), beta_window=4)
        eth = result.filter(pl.col("symbol") == "ETHUSDT").sort("timestamp")
        self.assertEqual(eth["ex_ante_btc_beta"][:4].null_count(), 4)
        self.assertAlmostEqual(eth["ex_ante_btc_beta"][4], 2.0, delta=1e-5)
        self.assertFalse(any(eth["target_valid_1h"][:4]))

    def test_flat_btc_produces_no_beta_or_training_rows(self):
        raw = price_panel().with_columns(pl.lit(100.0).alias("perp_close"))
        result = compute_targets(raw, beta_window=4)
        self.assertEqual(result["ex_ante_btc_beta"].null_count(), result.height)
        self.assertTrue(select_training_rows(result).is_empty())

    def test_standardized_labels_have_zero_mean_and_unit_sample_std(self):
        result = compute_targets(price_panel(), beta_window=4)
        for h in (1, 4, 12, 24):
            column = f"fwd_res_ret_z_{h}h"
            stats = result.group_by("timestamp").agg(
                pl.col(column).mean().alias("mean"), pl.col(column).std().alias("std")
            ).drop_nulls()
            self.assertGreater(stats.height, 0)
            self.assertLess(stats["mean"].abs().max(), 1e-10)
            self.assertLess((stats["std"] - 1).abs().max(), 1e-10)
            self.assertFalse(result[column].is_nan().any())

    def test_zero_cross_sectional_dispersion_has_no_zscore(self):
        btc = price_panel().filter(pl.col("symbol") == "BTCUSDT")
        identical = pl.concat([btc, btc.with_columns(pl.lit("ETHUSDT").alias("symbol"))])
        result = compute_targets(identical, beta_window=4)
        self.assertEqual(result["fwd_res_ret_z_1h"].null_count(), result.height)
        self.assertTrue(select_training_rows(result, horizons=(1,)).is_empty())

    def test_tail_filter_is_per_horizon(self):
        result = compute_targets(price_panel(), beta_window=4)
        for h in (1, 4, 12, 24):
            eth = result.filter(pl.col("symbol") == "ETHUSDT").sort("timestamp")
            self.assertEqual(eth[f"fwd_ret_{h}h"][-h:].null_count(), h)
            self.assertFalse(any(eth[f"target_valid_{h}h"][-h:]))
            valid = select_training_rows(result, horizons=(h,))
            self.assertLessEqual(valid["timestamp"].max(), datetime(2023, 1, 1) + timedelta(hours=59-h))
        self.assertGreater(select_training_rows(result, horizons=(1,)).height,
                           select_training_rows(result).height)

    def test_beta_does_not_use_future_prices(self):
        raw = price_panel()
        boundary = datetime(2023, 1, 1, 20)
        changed = raw.with_columns(
            pl.when(pl.col("timestamp") > boundary)
            .then(pl.col("perp_close") * 1.5).otherwise(pl.col("perp_close"))
            .alias("perp_close")
        )
        before = compute_targets(raw, beta_window=4).filter(pl.col("timestamp") <= boundary)
        after = compute_targets(changed, beta_window=4).filter(pl.col("timestamp") <= boundary)
        self.assertEqual(before["ex_ante_btc_beta"].to_list(), after["ex_ante_btc_beta"].to_list())


if __name__ == "__main__":
    unittest.main()
