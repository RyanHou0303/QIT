"""Hourly linear-perpetual accounting with next-bar-open execution."""

from dataclasses import asdict, dataclass
from datetime import timedelta

import numpy as np
import polars as pl
from scipy.optimize import brentq


HOURS_PER_YEAR = 365 * 24


@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float = 100_000.0
    fee_bps: float = 2.0
    slippage_bps: float = 3.0
    execution_lag_bars: int = 1
    max_gross: float = 1.0
    max_position: float = 0.20
    liquidate_at_end: bool = True

    def __post_init__(self):
        values = (self.initial_equity, self.fee_bps, self.slippage_bps,
                  self.max_gross, self.max_position)
        if not all(np.isfinite(v) for v in values):
            raise ValueError("Configuration values must be finite")
        if self.initial_equity <= 0 or self.max_gross <= 0 or self.max_position <= 0:
            raise ValueError("Capital and exposure limits must be positive")
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("Costs must not be negative")
        if not isinstance(self.execution_lag_bars, int) or self.execution_lag_bars < 1:
            raise ValueError("execution_lag_bars must be an integer of at least one")
        if (self.fee_bps + self.slippage_bps) * 1e-4 * self.max_gross >= 1:
            raise ValueError("Costs and leverage do not admit a stable rebalance")


def _rebalance(units, prices, equity, weights, config):
    """Solve for post-cost equity so requested weights include their own costs."""
    rate = (config.fee_bps + config.slippage_bps) * 1e-4
    old_notionals = units * prices
    if rate == 0:
        after = equity
    else:
        if rate * np.abs(old_notionals).sum() >= equity:
            raise ValueError("Equity cannot cover liquidation costs")
        after = brentq(
            lambda value: value + rate * np.abs(weights * value - old_notionals).sum() - equity,
            0.0, equity, xtol=1e-10,
        )
    new_units = weights * after / prices
    traded = np.abs(new_units - units) * prices
    fees = traded * config.fee_bps * 1e-4
    slippage = traded * config.slippage_bps * 1e-4
    return new_units, after, traded, fees, slippage


def run_backtest(
    market: pl.DataFrame,
    target_weights: pl.DataFrame,
    config: BacktestConfig = BacktestConfig(),
) -> dict:
    """Account for a fixed set of linear USDT perpetuals on complete hourly bars.

    Market timestamps are candle END times. A signal at t with lag=1 trades
    at the open of the candle ending t+1h, after the previous close/settlement.
    Thus it cannot earn the bar ending at its own signal timestamp.

    Units remain fixed between orders; weights drift with prices and equity.
    Funding uses units held through the current close, before any closing trade.
    Close prices proxy funding mark prices. Slippage is a linear notional cost.
    Unknown/missing prices or payment rates fail rather than creating zero PnL.
    Limits constrain target weights; drifted exposures are reported, not clamped.
    """
    required = ["timestamp", "symbol", "perp_open", "perp_close", "funding_payment_rate"]
    missing = set(required) - set(market.columns)
    if missing:
        raise ValueError(f"Missing market columns: {sorted(missing)}")
    if market.is_empty():
        raise ValueError("Market panel is empty")
    if market.unique(["timestamp", "symbol"]).height != market.height:
        raise ValueError("Duplicate market timestamp/symbol keys")
    if not {"timestamp", "symbol", "weight"}.issubset(target_weights.columns):
        raise ValueError("Targets need timestamp, symbol, and weight columns")
    if target_weights.unique(["timestamp", "symbol"]).height != target_weights.height:
        raise ValueError("Duplicate target timestamp/symbol keys")

    market = market.sort(["timestamp", "symbol"])
    times = market["timestamp"].unique().sort().to_list()
    time_set = set(times)
    symbols = market["symbol"].unique().sort().to_list()
    n_times, n_symbols = len(times), len(symbols)
    if market.height != n_times * n_symbols or any(
        right - left != timedelta(hours=1) for left, right in zip(times, times[1:])
    ):
        raise ValueError("Market must contain a complete hourly symbol/time grid")
    for column in ("perp_open", "perp_close", "funding_payment_rate"):
        if not market.select(pl.col(column).is_finite().fill_null(False).all()).item():
            raise ValueError(f"Missing or non-finite market values in {column}")
    if market.filter((pl.col("perp_open") <= 0) | (pl.col("perp_close") <= 0)).height:
        raise ValueError("Prices must be positive")
    if set(target_weights["symbol"].unique()) - set(symbols):
        raise ValueError("Targets contain symbols absent from the market")
    if not target_weights.select(pl.col("weight").is_finite().fill_null(False).all()).item():
        raise ValueError("Target weights must be finite")

    index = {symbol: i for i, symbol in enumerate(symbols)}
    scheduled = {}
    for group in target_weights.partition_by("timestamp", maintain_order=True):
        signal_time = group["timestamp"][0]
        if len(group) != n_symbols or set(group["symbol"]) != set(symbols):
            raise ValueError("Every target timestamp must explicitly include every symbol")
        weights = np.zeros(n_symbols)
        for row in group.iter_rows(named=True):
            weights[index[row["symbol"]]] = row["weight"]
        if np.abs(weights).sum() > config.max_gross + 1e-9:
            raise ValueError("Target exceeds gross exposure limit")
        if np.abs(weights).max() > config.max_position + 1e-9:
            raise ValueError("Target exceeds position limit")
        bar_end = signal_time + timedelta(hours=config.execution_lag_bars)
        if times[0] <= bar_end <= times[-1]:
            if bar_end not in time_set:
                raise ValueError("Target execution time is not on the market grid")
            scheduled[bar_end] = (signal_time, weights)

    opens = market["perp_open"].to_numpy().reshape(n_times, n_symbols)
    closes = market["perp_close"].to_numpy().reshape(n_times, n_symbols)
    funding = market["funding_payment_rate"].to_numpy().reshape(n_times, n_symbols)
    betas = (market["ex_ante_btc_beta"].to_numpy().reshape(n_times, n_symbols)
             if "ex_ante_btc_beta" in market.columns else None)
    units = np.zeros(n_symbols)
    equity = config.initial_equity
    peak = equity
    start = times[0] - timedelta(hours=1)
    ledger = [{
        "timestamp": start, "equity": equity, "net_return": 0.0,
        "price_pnl": 0.0, "funding_pnl": 0.0, "fees": 0.0, "slippage": 0.0,
        "traded_notional": 0.0, "turnover": 0.0, "gross_exposure": 0.0,
        "net_exposure": 0.0, "beta_exposure": 0.0 if betas is not None else None,
        "drawdown": 0.0,
    }]
    positions, trades = [], []

    for t, timestamp in enumerate(times):
        previous_equity = equity
        old_units = units.copy()
        gap_pnl = (old_units * (opens[t] - closes[t - 1]) if t else np.zeros(n_symbols))
        equity += gap_pnl.sum()
        if equity <= 0:
            raise ValueError(f"Equity exhausted at the open of {timestamp}")
        traded = np.zeros(n_symbols)
        fees = np.zeros(n_symbols)
        slippage = np.zeros(n_symbols)

        def execute(weights, prices, when, reason, signal_time):
            nonlocal units, equity, traded, fees, slippage
            before = units.copy()
            units, equity, turnover_cash, fee_cash, slip_cash = _rebalance(
                units, prices, equity, weights, config
            )
            traded += turnover_cash
            fees += fee_cash
            slippage += slip_cash
            for i, symbol in enumerate(symbols):
                if turnover_cash[i] > 1e-10:
                    trades.append({
                        "execution_timestamp": when, "signal_timestamp": signal_time,
                        "symbol": symbol, "reason": reason, "price": prices[i],
                        "delta_units": units[i] - before[i], "target_weight": weights[i],
                        "traded_notional": turnover_cash[i], "fees": fee_cash[i],
                        "slippage": slip_cash[i],
                    })

        if timestamp in scheduled:
            signal_time, weights = scheduled[timestamp]
            execute(weights, opens[t], timestamp - timedelta(hours=1), "rebalance", signal_time)

        held_units = units.copy()
        price_pnl = gap_pnl + held_units * (closes[t] - opens[t])
        # Positive settlement rates charge longs and credit shorts.
        funding_pnl = -held_units * closes[t] * funding[t]
        equity += (price_pnl - gap_pnl + funding_pnl).sum()
        if equity <= 0:
            raise ValueError(f"Equity exhausted at {timestamp}")
        if t == n_times - 1 and config.liquidate_at_end:
            execute(np.zeros(n_symbols), closes[t], timestamp, "terminal_close", None)

        weights_at_close = units * closes[t] / equity
        beta_exposure = (float(weights_at_close @ betas[t])
                         if betas is not None and np.isfinite(betas[t]).all() else None)
        peak = max(peak, equity)
        accounted = previous_equity + price_pnl.sum() + funding_pnl.sum() - fees.sum() - slippage.sum()
        if not np.isclose(equity, accounted, rtol=1e-10, atol=1e-7):
            raise RuntimeError("Portfolio accounting identity failed")
        ledger.append({
            "timestamp": timestamp, "equity": equity,
            "net_return": equity / previous_equity - 1.0,
            "price_pnl": price_pnl.sum(), "funding_pnl": funding_pnl.sum(),
            "fees": fees.sum(), "slippage": slippage.sum(),
            "traded_notional": traded.sum(), "turnover": traded.sum() / previous_equity,
            "gross_exposure": np.abs(weights_at_close).sum(),
            "net_exposure": weights_at_close.sum(), "beta_exposure": beta_exposure,
            "drawdown": equity / peak - 1.0,
        })
        for i, symbol in enumerate(symbols):
            positions.append({
                "timestamp": timestamp, "symbol": symbol, "held_units": held_units[i],
                "end_units": units[i], "end_weight": weights_at_close[i],
                "price_pnl": price_pnl[i], "funding_pnl": funding_pnl[i],
                "fees": fees[i], "slippage": slippage[i],
                "traded_notional": traded[i],
            })

    curve = pl.DataFrame(ledger)
    returns = curve["net_return"].to_numpy()[1:]
    volatility = returns.std(ddof=1) if len(returns) > 1 else 0.0
    years = (times[-1] - start).total_seconds() / (HOURS_PER_YEAR * 3600)
    annual_log_return = np.log(equity / config.initial_equity) / years
    annual_return = (float(np.expm1(annual_log_return))
                     if annual_log_return < np.log(np.finfo(float).max) else None)
    summary = pl.DataFrame([{
        "start": start, "end": times[-1], "hours": n_times,
        "initial_equity": config.initial_equity, "final_equity": equity,
        "total_return": equity / config.initial_equity - 1.0,
        "annualized_return": annual_return,
        "annualized_volatility": volatility * np.sqrt(HOURS_PER_YEAR),
        "sharpe": returns.mean() / volatility * np.sqrt(HOURS_PER_YEAR) if volatility > 1e-12 else None,
        "max_drawdown": -curve["drawdown"].min(),
        "price_pnl": curve["price_pnl"].sum(), "funding_pnl": curve["funding_pnl"].sum(),
        "fees": curve["fees"].sum(), "slippage": curve["slippage"].sum(),
        "total_traded_notional": curve["traded_notional"].sum(),
        "mean_hourly_turnover": float(curve["turnover"][1:].mean()),
        "max_gross_exposure": curve["gross_exposure"].max(),
        "max_abs_net_exposure": curve["net_exposure"].abs().max(),
    }], schema_overrides={"sharpe": pl.Float64, "annualized_return": pl.Float64})
    trade_schema = {
        "execution_timestamp": pl.Datetime("us"), "signal_timestamp": pl.Datetime("us"),
        "symbol": pl.String, "reason": pl.String, "price": pl.Float64,
        "delta_units": pl.Float64, "target_weight": pl.Float64,
        "traded_notional": pl.Float64, "fees": pl.Float64, "slippage": pl.Float64,
    }
    return {
        "equity": curve, "positions": pl.DataFrame(positions),
        "trades": pl.DataFrame(trades, schema=trade_schema), "summary": summary,
        "config": asdict(config),
    }
