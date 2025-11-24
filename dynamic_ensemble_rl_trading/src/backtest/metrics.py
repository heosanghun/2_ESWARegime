"""
Performance metrics calculation for backtesting.

This module calculates various performance metrics including:
- Cumulative Return, CAGR, Sharpe Ratio, MDD, Win Rate, Profit Factor
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Calculates performance metrics for trading strategies.
    
    Provides comprehensive metrics for evaluating trading system
    performance and comparing against benchmarks.
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize the metrics calculator.
        
        Parameters
        ----------
        risk_free_rate : float, default=0.0
            Risk-free rate for Sharpe Ratio calculation.
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_cumulative_return(
        self,
        portfolio_values: np.ndarray
    ) -> float:
        """
        Calculate cumulative return.
        
        Cumulative Return = (V_final - V_initial) / V_initial
        
        Parameters
        ----------
        portfolio_values : np.ndarray
            Array of portfolio values over time.
        
        Returns
        -------
        float
            Cumulative return as fraction.
        """
        if len(portfolio_values) < 2:
            return 0.0
        
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        
        if initial_value <= 0:
            return 0.0
        
        cumulative_return = (final_value - initial_value) / initial_value
        
        return cumulative_return
    
    def calculate_cagr(
        self,
        portfolio_values: np.ndarray,
        num_years: float
    ) -> float:
        """
        Calculate Compounded Annual Growth Rate (CAGR).
        
        CAGR = (V_final / V_initial)^(1/num_years) - 1
        
        Parameters
        ----------
        portfolio_values : np.ndarray
            Array of portfolio values over time.
        num_years : float
            Number of years in the period.
        
        Returns
        -------
        float
            CAGR as fraction.
        """
        if len(portfolio_values) < 2 or num_years <= 0:
            return 0.0
        
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        
        if initial_value <= 0:
            return 0.0
        
        cagr = (final_value / initial_value) ** (1.0 / num_years) - 1.0
        
        return cagr
    
    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe Ratio.
        
        Sharpe Ratio = (R_p - R_f) / sigma_p * sqrt(periods_per_year)
        
        Parameters
        ----------
        returns : np.ndarray
            Array of returns.
        periods_per_year : int, default=252
            Number of trading periods per year.
        
        Returns
        -------
        float
            Sharpe Ratio.
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe Ratio
        sharpe_ratio = (
            (mean_return - self.risk_free_rate / periods_per_year) / std_return
        ) * np.sqrt(periods_per_year)
        
        return sharpe_ratio
    
    def calculate_max_drawdown(
        self,
        portfolio_values: np.ndarray
    ) -> float:
        """
        Calculate Maximum Drawdown (MDD).
        
        MDD = max((peak - trough) / peak)
        
        Parameters
        ----------
        portfolio_values : np.ndarray
            Array of portfolio values over time.
        
        Returns
        -------
        float
            Maximum drawdown as fraction (negative value).
        """
        if len(portfolio_values) < 2:
            return 0.0
        
        # Calculate running maximum (peak)
        running_max = np.maximum.accumulate(portfolio_values)
        
        # Calculate drawdown
        drawdown = (portfolio_values - running_max) / running_max
        
        # Maximum drawdown (most negative)
        max_drawdown = np.min(drawdown)
        
        return max_drawdown
    
    def calculate_win_rate(
        self,
        returns: np.ndarray
    ) -> float:
        """
        Calculate win rate (percentage of profitable trades).
        
        Parameters
        ----------
        returns : np.ndarray
            Array of returns.
        
        Returns
        -------
        float
            Win rate as fraction (0 to 1).
        """
        if len(returns) == 0:
            return 0.0
        
        # Count positive returns
        winning_trades = np.sum(returns > 0)
        total_trades = len(returns)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        return win_rate
    
    def calculate_profit_factor(
        self,
        returns: np.ndarray
    ) -> float:
        """
        Calculate Profit Factor.
        
        Profit Factor = Total Profit / Total Loss
        
        Parameters
        ----------
        returns : np.ndarray
            Array of returns.
        
        Returns
        -------
        float
            Profit Factor.
        """
        if len(returns) == 0:
            return 0.0
        
        total_profit = np.sum(returns[returns > 0])
        total_loss = abs(np.sum(returns[returns < 0]))
        
        if total_loss == 0:
            if total_profit > 0:
                return np.inf
            else:
                return 0.0
        
        profit_factor = total_profit / total_loss
        
        return profit_factor
    
    def calculate_all_metrics(
        self,
        portfolio_values: np.ndarray,
        returns: Optional[np.ndarray] = None,
        num_years: Optional[float] = None
    ) -> dict:
        """
        Calculate all performance metrics.
        
        Parameters
        ----------
        portfolio_values : np.ndarray
            Array of portfolio values over time.
        returns : np.ndarray, optional
            Array of returns. If None, calculated from portfolio_values.
        num_years : float, optional
            Number of years. If None, estimated from data length.
        
        Returns
        -------
        dict
            Dictionary containing all metrics.
        """
        if returns is None:
            # Calculate returns from portfolio values
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        if num_years is None:
            # Estimate years (assuming daily data)
            num_years = len(portfolio_values) / 252.0
        
        metrics = {
            'cumulative_return': self.calculate_cumulative_return(portfolio_values),
            'cagr': self.calculate_cagr(portfolio_values, num_years),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(portfolio_values),
            'win_rate': self.calculate_win_rate(returns),
            'profit_factor': self.calculate_profit_factor(returns)
        }
        
        return metrics
    
    def compare_with_benchmark(
        self,
        strategy_metrics: dict,
        benchmark_metrics: dict
    ) -> dict:
        """
        Compare strategy metrics with benchmark.
        
        Parameters
        ----------
        strategy_metrics : dict
            Metrics for the trading strategy.
        benchmark_metrics : dict
            Metrics for the benchmark (e.g., Buy & Hold).
        
        Returns
        -------
        dict
            Comparison results.
        """
        comparison = {}
        
        for metric_name in strategy_metrics:
            if metric_name in benchmark_metrics:
                strategy_value = strategy_metrics[metric_name]
                benchmark_value = benchmark_metrics[metric_name]
                
                if benchmark_value != 0:
                    improvement = (
                        (strategy_value - benchmark_value) / abs(benchmark_value)
                    ) * 100
                else:
                    improvement = 0.0
                
                comparison[metric_name] = {
                    'strategy': strategy_value,
                    'benchmark': benchmark_value,
                    'improvement_pct': improvement
                }
        
        return comparison

