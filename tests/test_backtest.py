import unittest
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.run_baselines import build_target_weights, predict_scores


START = datetime(2024, 1, 1)


def market(open_a, close_a, funding_a=None):
    funding_a = funding_a if funding_a is not None else [0.0] * len(open_a)
    rows = []
    for i, (opening, closing, rate) in enumerate(zip(open_a, close_a, funding_a)):
        for symbol, op, cl, funding in (("A", opening, closing, rate), ("B", 100.0, 100.0, 0.0)):
            rows.append({
                "timestamp": START + timedelta(hours=i + 1), "symbol": symbol,
                "perp_open": float(op), "perp_close": float(cl),
                "funding_payment_rate": float(funding),
            })
    return pl.DataFrame(rows)


def targets(hour=0, weight=0.5):
    return pl.DataFrame({
        "timestamp": [START + timedelta(hours=hour)] * 2,
        "symbol": ["A", "B"], "weight": [weight, -weight],
    })


def config(**overrides):
    return BacktestConfig(**{
        "initial_equity": 1000.0, "fee_bps": 0.0, "slippage_bps": 0.0,
        "max_position": 0.60, **overrides,
    })


class FirstFeatureModel:
    def predict(self, x):
        return x[:, 0]


class BacktestEngineTests(unittest.TestCase):
    def test_close_signal_cannot_earn_its_own_bar_return(self):
        result = run_backtest(
            market([100, 200, 200], [200, 200, 200]), targets(hour=1), config()
        )
        self.assertAlmostEqual(result["summary"]["final_equity"][0], 1000.0)
        first = result["positions"].filter(pl.col("timestamp") == START + timedelta(hours=1))
        self.assertEqual(first["held_units"].abs().sum(), 0.0)
        self.assertEqual(result["trades"]["execution_timestamp"][0], START + timedelta(hours=1))

    def test_execution_uses_next_open_not_a_previous_close(self):
        result = run_backtest(market([200], [200]), targets(), config())
        entry = result["trades"].filter((pl.col("symbol") == "A") & (pl.col("reason") == "rebalance"))
        self.assertEqual(entry["price"][0], 200.0)
        self.assertAlmostEqual(entry["delta_units"][0], 2.5)
        self.assertAlmostEqual(result["summary"]["final_equity"][0], 1000.0)

    def test_units_stay_fixed_and_weights_drift_between_orders(self):
        result = run_backtest(market([100, 110], [110, 121]), targets(), config())
        self.assertAlmostEqual(result["summary"]["final_equity"][0], 1105.0)
        a = result["positions"].filter(pl.col("symbol") == "A")
        np.testing.assert_allclose(a["held_units"].to_numpy(), [5.0, 5.0])
        self.assertGreater(a["end_weight"][0], 0.5)
        self.assertEqual(result["positions"]["end_units"][-2:].abs().sum(), 0.0)

    def test_opening_gap_belongs_to_old_units_before_rebalancing(self):
        orders = pl.concat([targets(), targets(hour=1, weight=0.0)])
        result = run_backtest(market([100, 120], [110, 125]), orders, config())
        # Existing A units earn 100->110, then the 110->120 opening gap.
        # They are sold at 120 and cannot earn the later 120->125 move.
        self.assertAlmostEqual(result["summary"]["final_equity"][0], 1100.0)

    def test_future_prices_cannot_change_prior_equity_or_entry_trades(self):
        before = run_backtest(market([100, 110, 120], [110, 120, 130]), targets(), config())
        after = run_backtest(market([100, 110, 120], [110, 120, 500]), targets(), config())
        cutoff = START + timedelta(hours=2)
        self.assertTrue(before["equity"].filter(pl.col("timestamp") <= cutoff).equals(
            after["equity"].filter(pl.col("timestamp") <= cutoff)
        ))
        self.assertTrue(before["trades"].filter(pl.col("reason") == "rebalance").equals(
            after["trades"].filter(pl.col("reason") == "rebalance")
        ))

    def test_entry_and_exit_costs_and_post_cost_target_weights(self):
        cfg = config(fee_bps=2.0, slippage_bps=3.0)
        result = run_backtest(market([100, 100], [100, 100]), targets(), cfg)
        rate = 5e-4
        expected = 1000 * (1 - rate) / (1 + rate)
        self.assertAlmostEqual(result["summary"]["final_equity"][0], expected)
        self.assertAlmostEqual(result["summary"]["fees"][0] / result["summary"]["slippage"][0], 2 / 3)
        first = result["positions"].filter(pl.col("timestamp") == START + timedelta(hours=1))
        np.testing.assert_allclose(first["end_weight"].to_numpy(), [0.5, -0.5])
        self.assertEqual(set(result["trades"]["reason"]), {"rebalance", "terminal_close"})

    def test_funding_sign_and_settlement_precede_next_open_exit(self):
        for weight, funding, expected in ((0.5, 0.01, 995), (-0.5, 0.01, 1005), (0.5, -0.01, 1005)):
            with self.subTest(weight=weight, funding=funding):
                orders = pl.concat([targets(weight=weight), targets(hour=1, weight=0.0)])
                result = run_backtest(market([100, 100], [100, 100], [funding, 0.0]), orders, config())
                self.assertAlmostEqual(result["summary"]["final_equity"][0], expected)
                self.assertAlmostEqual(result["equity"]["funding_pnl"][1], expected - 1000)

    def test_gap_and_unknown_funding_fail_instead_of_creating_free_returns(self):
        data = market([100, 100, 100], [100, 100, 100])
        gap = data.filter(pl.col("timestamp") != START + timedelta(hours=2))
        with self.assertRaisesRegex(ValueError, "complete hourly"):
            run_backtest(gap, targets(), config())
        unknown = data.with_columns(pl.lit(None).cast(pl.Float64).alias("funding_payment_rate"))
        with self.assertRaisesRegex(ValueError, "funding_payment_rate"):
            run_backtest(unknown, targets(), config())

    def test_flat_strategy_and_accounting_attribution(self):
        result = run_backtest(market([100, 110], [110, 120]), targets(weight=0.0), config())
        self.assertEqual(result["summary"]["total_return"][0], 0.0)
        self.assertTrue(result["trades"].is_empty())
        self.assertIsNone(result["summary"]["sharpe"][0])
        result = run_backtest(market([100, 110], [110, 120], [0.001, 0]), targets(), config(fee_bps=2))
        row = result["summary"].row(0, named=True)
        expected = row["price_pnl"] + row["funding_pnl"] - row["fees"] - row["slippage"]
        self.assertAlmostEqual(row["final_equity"] - row["initial_equity"], expected)

    def test_incomplete_targets_and_excessive_positions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "every symbol"):
            run_backtest(market([100], [100]), targets().head(1), config())
        with self.assertRaisesRegex(ValueError, "position limit"):
            run_backtest(market([100], [100]), targets(), config(max_position=0.4))


class BacktestSignalTests(unittest.TestCase):
    def test_neutral_projection_caps_and_missing_signal_flattening(self):
        records = []
        for hour in (0, 1):
            for symbol, beta in (("A", 0.7), ("B", 1.4), ("C", 0.9), ("BTCUSDT", 1.0)):
                records.append({"timestamp": START + timedelta(hours=hour), "symbol": symbol,
                                "ex_ante_btc_beta": beta})
        features = pl.DataFrame(records)
        scores = pl.DataFrame({
            "timestamp": [START] * 3, "symbol": ["A", "B", "C"], "score": [3.0, 1.0, 2.0],
        })
        weights = build_target_weights(
            scores, features, start=START, end=START + timedelta(hours=2),
            rebalance_hours=1, max_position=0.25, min_symbols=3,
        )
        first = weights.filter(pl.col("timestamp") == START)
        self.assertAlmostEqual(first["weight"].sum(), 0.0)
        self.assertAlmostEqual((first["weight"] * first["signal_beta"]).sum(), 0.0)
        self.assertLessEqual(first["weight"].abs().max(), 0.25 + 1e-12)
        self.assertLessEqual(first["weight"].abs().sum(), 1.0 + 1e-12)
        self.assertGreater(first["weight"].abs().sum(), 0.0)
        self.assertEqual(weights.filter(pl.col("timestamp") > START)["weight"].abs().sum(), 0.0)

    def test_inference_does_not_depend_on_forward_labels_or_validity_flags(self):
        features = pl.DataFrame({
            "timestamp": [START] * 2, "symbol": ["A", "B"], "is_observed": [True, True],
            "x": [1.0, 2.0], "fwd_res_ret_z_1h": [None, None], "target_valid_1h": [False, False],
        })
        bundle = {"feature_cols": ["x"], "model": FirstFeatureModel()}
        before = predict_scores(features, bundle, "ridge", START, START + timedelta(hours=1))
        changed = features.with_columns(
            pl.lit(999.0).alias("fwd_res_ret_z_1h"), pl.lit(True).alias("target_valid_1h")
        )
        after = predict_scores(changed, bundle, "ridge", START, START + timedelta(hours=1))
        self.assertTrue(before.equals(after))
        self.assertEqual(before.height, 2)


if __name__ == "__main__":
    unittest.main()
