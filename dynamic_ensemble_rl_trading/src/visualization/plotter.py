"""
Visualization module for trading system results.

This module creates plots for portfolio performance, regime detection,
and dynamic weight allocation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class TradingPlotter:
    """
    Creates visualization plots for trading system analysis.
    """
    
    def __init__(self, figsize: tuple = (12, 6)):
        """
        Initialize the plotter.
        
        Parameters
        ----------
        figsize : tuple, default=(12, 6)
            Default figure size.
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_portfolio_vs_benchmark(
        self,
        portfolio_values: np.ndarray,
        benchmark_values: np.ndarray,
        timestamps: pd.DatetimeIndex,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot portfolio value vs benchmark.
        
        Parameters
        ----------
        portfolio_values : np.ndarray
            Portfolio values over time.
        benchmark_values : np.ndarray
            Benchmark (e.g., Buy & Hold) values over time.
        timestamps : pd.DatetimeIndex
            Timestamps for x-axis.
        save_path : str, optional
            Path to save the plot.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(timestamps, portfolio_values, label='Proposed System', linewidth=2)
        ax.plot(timestamps, benchmark_values, label='Buy & Hold', linewidth=2, linestyle='--')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value')
        ax.set_title('Portfolio Value vs Benchmark')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_regime_detection(
        self,
        timestamps: pd.DatetimeIndex,
        detected_regimes: List[str],
        regime_confidences: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot detected regimes and confidence over time.
        
        Parameters
        ----------
        timestamps : pd.DatetimeIndex
            Timestamps.
        detected_regimes : list of str
            Detected regime for each timestamp.
        regime_confidences : np.ndarray
            Confidence values for regime detection.
        save_path : str, optional
            Path to save the plot.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5))
        
        # Map regimes to numbers for plotting
        regime_map = {'Bear': 0, 'Sideways': 1, 'Bull': 2}
        regime_numbers = [regime_map.get(r, 1) for r in detected_regimes]
        
        # Plot regime
        ax1.plot(timestamps, regime_numbers, marker='o', markersize=3, linewidth=1)
        ax1.set_ylabel('Regime')
        ax1.set_title('Detected Market Regime Over Time')
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(['Bear', 'Sideways', 'Bull'])
        ax1.grid(True, alpha=0.3)
        
        # Plot confidence
        ax2.plot(timestamps, regime_confidences, marker='o', markersize=3, linewidth=1, color='green')
        ax2.axhline(y=0.6, color='r', linestyle='--', label='Confidence Threshold (0.6)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Regime Classification Confidence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_dynamic_weights(
        self,
        timestamps: pd.DatetimeIndex,
        weights_history: List[np.ndarray],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot dynamic weights of agents over time.
        
        Parameters
        ----------
        timestamps : pd.DatetimeIndex
            Timestamps.
        weights_history : list of np.ndarray
            Weight arrays for each timestamp.
        save_path : str, optional
            Path to save the plot.
        """
        if len(weights_history) == 0:
            logger.warning("No weights history to plot")
            return
        
        num_agents = len(weights_history[0])
        weights_matrix = np.array(weights_history)
        
        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1] * 0.8))
        
        # Stacked area chart
        ax.stackplot(
            timestamps,
            *[weights_matrix[:, i] for i in range(num_agents)],
            labels=[f'Agent {i+1}' for i in range(num_agents)],
            alpha=0.7
        )
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Weight')
        ax.set_title('Dynamic Agent Weights Over Time')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_performance_metrics(
        self,
        strategy_metrics: Dict,
        benchmark_metrics: Dict,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot performance metrics comparison.
        
        Parameters
        ----------
        strategy_metrics : dict
            Metrics for the trading strategy.
        benchmark_metrics : dict
            Metrics for the benchmark.
        save_path : str, optional
            Path to save the plot.
        """
        metric_names = ['cumulative_return', 'cagr', 'sharpe_ratio', 'max_drawdown']
        metric_labels = ['Cumulative Return', 'CAGR', 'Sharpe Ratio', 'Max Drawdown']
        
        strategy_values = [strategy_metrics.get(m, 0) for m in metric_names]
        benchmark_values = [benchmark_metrics.get(m, 0) for m in metric_names]
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        bars1 = ax.bar(x - width/2, strategy_values, width, label='Proposed System')
        bars2 = ax.bar(x + width/2, benchmark_values, width, label='Buy & Hold')
        
        ax.set_ylabel('Value')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

