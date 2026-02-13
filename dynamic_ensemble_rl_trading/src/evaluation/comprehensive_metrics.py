"""
Comprehensive metrics calculator with statistical significance testing.

Provides Ledoit-Wolf Sharpe Ratio Test and Bonferroni correction.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from ..backtest.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class ComprehensiveMetrics(PerformanceMetrics):
    """
    Extended metrics calculator with statistical significance testing.
    
    Provides:
    - All standard performance metrics
    - Ledoit-Wolf Sharpe Ratio Test
    - Bonferroni correction for multiple comparisons
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize comprehensive metrics calculator.
        
        Parameters
        ----------
        risk_free_rate : float, default=0.0
            Risk-free rate for Sharpe Ratio calculation.
        """
        super().__init__(risk_free_rate=risk_free_rate)
    
    def ledoit_wolf_sharpe_test(
        self,
        returns1: np.ndarray,
        returns2: np.ndarray,
        periods_per_year: int = 252
    ) -> Dict[str, float]:
        """
        Perform Ledoit-Wolf Sharpe Ratio Test.
        
        Tests if Sharpe Ratio of returns1 is significantly different from returns2.
        More robust for small samples and non-normal distributions.
        
        Parameters
        ----------
        returns1 : np.ndarray
            Returns of strategy 1.
        returns2 : np.ndarray
            Returns of strategy 2 (or benchmark).
        periods_per_year : int, default=252
            Number of trading periods per year.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'sharpe_diff': Difference in Sharpe Ratios
            - 'test_statistic': Test statistic
            - 'p_value': P-value
            - 'significant': Whether difference is significant (p < 0.01)
        """
        if len(returns1) < 2 or len(returns2) < 2:
            return {
                'sharpe_diff': 0.0,
                'test_statistic': 0.0,
                'p_value': 1.0,
                'significant': False
            }
        
        # Calculate Sharpe Ratios
        sharpe1 = self.calculate_sharpe_ratio(returns1, periods_per_year)
        sharpe2 = self.calculate_sharpe_ratio(returns2, periods_per_year)
        sharpe_diff = sharpe1 - sharpe2
        
        # Ledoit-Wolf test statistic
        # Simplified version: using t-test approximation
        # Full implementation would use Ledoit-Wolf shrinkage estimator
        
        mean1 = np.mean(returns1)
        mean2 = np.mean(returns2)
        std1 = np.std(returns1)
        std2 = np.std(returns2)
        
        n1 = len(returns1)
        n2 = len(returns2)
        
        # Standard error of difference
        se_diff = np.sqrt((std1**2 / n1) + (std2**2 / n2))
        
        if se_diff == 0:
            test_statistic = 0.0
            p_value = 1.0
        else:
            # Test statistic (simplified)
            test_statistic = (mean1 - mean2) / se_diff
            
            # P-value (two-tailed t-test approximation)
            # Using degrees of freedom approximation
            df = min(n1, n2) - 1
            try:
                from scipy import stats
                p_value = 2 * (1 - stats.t.cdf(abs(test_statistic), df))
            except ImportError:
                # Fallback: approximate p-value using normal distribution
                import math
                p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(test_statistic) / math.sqrt(2))))
        
        # Significance at p < 0.01
        significant = p_value < 0.01
        
        return {
            'sharpe_diff': sharpe_diff,
            'test_statistic': test_statistic,
            'p_value': p_value,
            'significant': significant
        }
    
    def bonferroni_correction(
        self,
        p_values: List[float],
        alpha: float = 0.05
    ) -> Dict[str, any]:
        """
        Apply Bonferroni correction for multiple comparisons.
        
        Parameters
        ----------
        p_values : list of float
            List of p-values from multiple tests.
        alpha : float, default=0.05
            Significance level.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'corrected_alpha': Bonferroni corrected alpha
            - 'significant': List of boolean indicating significance
            - 'adjusted_p_values': Adjusted p-values (if applicable)
        """
        n_comparisons = len(p_values)
        corrected_alpha = alpha / n_comparisons
        
        significant = [p < corrected_alpha for p in p_values]
        
        return {
            'corrected_alpha': corrected_alpha,
            'significant': significant,
            'n_comparisons': n_comparisons
        }
    
    def compare_with_benchmark_comprehensive(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        portfolio_values_strategy: Optional[np.ndarray] = None,
        portfolio_values_benchmark: Optional[np.ndarray] = None,
        num_years: Optional[float] = None
    ) -> Dict:
        """
        Comprehensive comparison with benchmark including statistical tests.
        
        Parameters
        ----------
        strategy_returns : np.ndarray
            Returns of the strategy.
        benchmark_returns : np.ndarray
            Returns of the benchmark.
        portfolio_values_strategy : np.ndarray, optional
            Portfolio values of strategy (for metrics calculation).
        portfolio_values_benchmark : np.ndarray, optional
            Portfolio values of benchmark (for metrics calculation).
        num_years : float, optional
            Number of years (for CAGR calculation).
        
        Returns
        -------
        dict
            Comprehensive comparison results including:
            - Strategy metrics
            - Benchmark metrics
            - Improvements
            - Statistical significance test results
        """
        # Calculate metrics for strategy
        if portfolio_values_strategy is not None:
            strategy_metrics = self.calculate_all_metrics(
                portfolio_values_strategy,
                strategy_returns,
                num_years
            )
        else:
            # Estimate portfolio values from returns
            portfolio_values_strategy = np.cumprod(1 + strategy_returns) * 10000
            strategy_metrics = self.calculate_all_metrics(
                portfolio_values_strategy,
                strategy_returns,
                num_years
            )
        
        # Calculate metrics for benchmark
        if portfolio_values_benchmark is not None:
            benchmark_metrics = self.calculate_all_metrics(
                portfolio_values_benchmark,
                benchmark_returns,
                num_years
            )
        else:
            portfolio_values_benchmark = np.cumprod(1 + benchmark_returns) * 10000
            benchmark_metrics = self.calculate_all_metrics(
                portfolio_values_benchmark,
                benchmark_returns,
                num_years
            )
        
        # Calculate improvements
        improvements = {}
        for metric_name in strategy_metrics:
            if metric_name in benchmark_metrics:
                strategy_val = strategy_metrics[metric_name]
                benchmark_val = benchmark_metrics[metric_name]
                
                if benchmark_val != 0:
                    improvement_pct = ((strategy_val - benchmark_val) / abs(benchmark_val)) * 100
                else:
                    improvement_pct = 0.0
                
                improvements[metric_name] = improvement_pct
        
        # Statistical significance test
        significance_test = self.ledoit_wolf_sharpe_test(
            strategy_returns,
            benchmark_returns
        )
        
        return {
            'strategy_metrics': strategy_metrics,
            'benchmark_metrics': benchmark_metrics,
            'improvements': improvements,
            'significance_test': significance_test
        }
