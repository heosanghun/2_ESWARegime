"""
Backtesting engine – Clean Markowitz-style portfolio return calculation.

For each step i (hourly):
  - Agent holds weight w_i (fraction of portfolio in BTC)
  - Asset return: r_asset = (close[i+1] - close[i]) / close[i]
  - Step return:  r_step  = w_i * r_asset - tc_rate * |w_{i+1} - w_i|
  - Portfolio:    PV[i+1] = PV[i] * (1 + r_step)

No cash / holdings tracking needed – mathematically exact.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import logging

from .metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class Backtester:

    # Long-only weight map (same as trading_env)
    WEIGHT_MAP = {0: 0.0, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.0}

    def __init__(
        self,
        initial_capital: float = 10000.0,
        transaction_fee: float = 0.0005,
        slippage: float = 0.0002,
    ):
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        self.metrics_calculator = PerformanceMetrics()

    # ------------------------------------------------------------------
    def run_backtest(
        self,
        trading_history: List[Dict],
        ohlcv_data: pd.DataFrame,
        ohlcv_columns: Optional[dict] = None,
    ) -> Dict:
        logger.info(f"Running backtest on {len(trading_history)} trades")

        if ohlcv_columns is None:
            ohlcv_columns = {
                'open': 'open', 'high': 'high',
                'low': 'low', 'close': 'close', 'volume': 'volume',
            }

        # ── build aligned price series (effective_weight 지원) ──
        prices = []
        actions = []
        valid_ts = []
        eff_weights = []
        for trade in trading_history:
            ts = trade['timestamp']
            if ts in ohlcv_data.index:
                prices.append(float(ohlcv_data.loc[ts, ohlcv_columns['close']]))
                actions.append(trade['action'])
                valid_ts.append(ts)
                eff_weights.append(trade.get('effective_weight'))

        if len(prices) < 2:
            logger.error("Not enough price data for backtest")
            return self._empty_result(trading_history)

        prices = np.array(prices)
        n = len(prices)

        # ── convert actions → weights (effective_weight 우선) ──
        if any(w is not None for w in eff_weights):
            weights = np.array([
                w if w is not None else self.WEIGHT_MAP.get(a, 0.5)
                for w, a in zip(eff_weights, actions)
            ])
            weights = np.clip(weights, 0.0, 3.0)
        else:
            weights = np.array([self.WEIGHT_MAP.get(a, 0.5) for a in actions])

        # ── compute step returns ──
        #   r_asset[i]  = (price[i+1] - price[i]) / price[i]
        #   r_step[i]   = w[i] * r_asset[i] - tc * |w[i+1] - w[i]|  (slippage included in tc)
        asset_returns = np.diff(prices) / prices[:-1]          # length n-1
        tc_rate = self.transaction_fee + self.slippage

        # |w[i+1]-w[i]| for i=0..n-2 -> length n-1 (align with asset_returns)
        weight_changes = np.abs(np.diff(weights))

        step_returns = weights[:-1] * asset_returns - tc_rate * weight_changes

        # ── cumulative portfolio values ──
        pv = np.empty(n)
        pv[0] = self.initial_capital
        for i in range(n - 1):
            pv[i + 1] = pv[i] * (1 + step_returns[i])
            pv[i + 1] = max(pv[i + 1], 1.0)

        # ── metrics ──
        if len(valid_ts) > 1:
            days = (valid_ts[-1] - valid_ts[0]).days
            num_years = max(days / 365.25, 0.01)
        else:
            num_years = 1.0

        metrics = self.metrics_calculator.calculate_all_metrics(pv, step_returns, num_years)

        # ── trade log ──
        trades = []
        for i in range(1, n):
            if abs(weights[i] - weights[i - 1]) > 1e-6:
                trades.append({
                    'timestamp': valid_ts[i],
                    'action': actions[i],
                    'price': prices[i],
                    'weight_before': weights[i - 1],
                    'weight_after': weights[i],
                    'transaction_cost': tc_rate * abs(weights[i] - weights[i - 1]) * pv[i - 1],
                })

        logger.info("Backtest completed")
        logger.info(f"  Cumulative Return : {metrics['cumulative_return']*100:+.2f}%")
        logger.info(f"  CAGR              : {metrics['cagr']*100:+.2f}%")
        logger.info(f"  Sharpe Ratio      : {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown      : {metrics['max_drawdown']*100:.2f}%")
        logger.info(f"  Win Rate          : {metrics['win_rate']*100:.1f}%")
        logger.info(f"  Profit Factor     : {metrics['profit_factor']:.2f}")

        return {
            'portfolio_values': pv,
            'returns': step_returns,
            'trades': trades,
            'metrics': metrics,
            'num_trades': len(trades),
            'trading_history': trading_history,
        }

    # ------------------------------------------------------------------
    def calculate_buy_hold_benchmark(
        self,
        ohlcv_data: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        ohlcv_columns: Optional[dict] = None,
    ) -> Dict:
        if ohlcv_columns is None:
            ohlcv_columns = {'close': 'close'}

        mask = (ohlcv_data.index >= start_date) & (ohlcv_data.index <= end_date)
        data = ohlcv_data[mask]
        if len(data) == 0:
            return {'metrics': {}, 'portfolio_values': np.array([]), 'returns': np.array([])}

        prices = data[ohlcv_columns['close']].values
        pv = prices / prices[0] * self.initial_capital
        rets = np.diff(prices) / prices[:-1]
        days = (end_date - start_date).days
        num_years = max(days / 365.25, 0.01)
        metrics = self.metrics_calculator.calculate_all_metrics(pv, rets, num_years)
        return {'portfolio_values': pv, 'returns': rets, 'metrics': metrics}

    # ------------------------------------------------------------------
    def _empty_result(self, trading_history):
        return {
            'portfolio_values': np.array([self.initial_capital]),
            'returns': np.array([0.0]),
            'trades': [],
            'metrics': self.metrics_calculator.calculate_all_metrics(
                np.array([self.initial_capital]), np.array([0.0]), 1.0
            ),
            'num_trades': 0,
            'trading_history': trading_history,
        }
