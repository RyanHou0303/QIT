import unittest
from datetime import datetime

import polars as pl

from src.data.download import align_funding, parse_funding


class FundingAlignmentTests(unittest.TestCase):
    def test_settlement_timestamp_is_rounded_without_one_hour_shift(self):
        for scale in (1, 1000):
            with self.subTest(timestamp_scale=scale):
                raw = (
                    "calc_time,funding_interval_hours,last_funding_rate\n"
                    f"{1672560000008 * scale},8,0.0001\n"
                ).encode()
                funding = parse_funding(raw, "BTCUSDT")
                self.assertEqual(funding["timestamp"][0], datetime(2023, 1, 1, 8))
                self.assertEqual(funding["funding_rate"][0], 0.0001)

    def test_payment_is_sparse_and_last_known_rate_uses_prior_settlement(self):
        funding = parse_funding(
            b"calc_time,funding_interval_hours,last_funding_rate\n"
            b"1672531200000,8,0.0001\n"
            b"1672560000008,8,-0.0002\n",
            "BTCUSDT",
        )
        panel = pl.DataFrame({
            "timestamp": [datetime(2023, 1, 1, h) for h in (9, 1, 8)],
            "symbol": ["BTCUSDT"] * 3,
        })
        result = align_funding(panel, funding)
        self.assertEqual(result["funding_rate"].to_list(), [None, -0.0002, None])
        self.assertEqual(result["funding_payment_rate"].to_list(), [0.0, -0.0002, 0.0])
        self.assertEqual(result["funding_rate_ffill"].to_list(), [0.0001, -0.0002, -0.0002])

    def test_last_known_rate_does_not_leak_across_symbols_or_from_future(self):
        funding = pl.DataFrame({
            "timestamp": [datetime(2023, 1, 1, h) for h in (0, 8, 8)],
            "symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT"],
            "funding_rate": [0.0001, 0.0002, -0.0003],
        })
        panel = pl.DataFrame({
            "timestamp": [datetime(2023, 1, 1, h) for h in (1, 8, 1, 8)],
            "symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT"],
        })
        result = align_funding(panel, funding)
        self.assertEqual(result["funding_rate_ffill"].to_list(), [0.0001, 0.0002, None, -0.0003])

    def test_complete_join_failure_raises_instead_of_returning_zeros(self):
        panel = pl.DataFrame({
            "timestamp": [datetime(2023, 1, 1, 8)], "symbol": ["BTCUSDT"],
        })
        funding = panel.with_columns(
            (pl.col("timestamp") + pl.duration(milliseconds=8)).alias("timestamp"),
            pl.lit(0.0001).alias("funding_rate"),
        )
        with self.assertRaisesRegex(RuntimeError, "No funding timestamps matched"):
            align_funding(panel, funding)
        with self.assertRaisesRegex(RuntimeError, "Missing funding data"):
            align_funding(panel, funding.with_columns(pl.lit("ETHUSDT").alias("symbol")))


if __name__ == "__main__":
    unittest.main()
