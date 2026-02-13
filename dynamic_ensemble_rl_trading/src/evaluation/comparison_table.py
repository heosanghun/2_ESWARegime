"""
Comparison table generator for Table 2 format.

Generates performance comparison tables in various formats (CSV, LaTeX, Markdown).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ComparisonTable:
    """
    Generate Table 2 format comparison tables.
    
    Methods compared:
    - Buy & Hold
    - PPO (Single Agent)
    - XGBoost
    - CNN
    - Ensemble (Simple)
    - Proposed Method (Dynamic Ensemble)
    - Ablation Models (Model 1-4)
    """
    
    def __init__(self):
        """Initialize comparison table generator."""
        self.results = {}
    
    def add_result(
        self,
        method_name: str,
        metrics: Dict[str, float]
    ) -> None:
        """
        Add method result to comparison table.
        
        Parameters
        ----------
        method_name : str
            Name of the method.
        metrics : dict
            Dictionary of performance metrics:
            - sharpe_ratio: float
            - cumulative_return: float (as fraction)
            - cagr: float (as fraction)
            - max_drawdown: float (as fraction, negative)
            - win_rate: float (as fraction)
            - profit_factor: float (optional)
        """
        self.results[method_name] = metrics
        logger.info(f"Added result for {method_name}")
    
    def generate_dataframe(self) -> pd.DataFrame:
        """
        Generate comparison table as pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Comparison table with methods as rows and metrics as columns.
        """
        if not self.results:
            raise ValueError("No results added. Call add_result() first.")
        
        # Prepare data
        data = []
        for method_name, metrics in self.results.items():
            row = {
                'Method': method_name,
                'Sharpe Ratio': metrics.get('sharpe_ratio', np.nan),
                'Cumulative Return (%)': metrics.get('cumulative_return', np.nan) * 100,
                'CAGR (%)': metrics.get('cagr', np.nan) * 100,
                'Max Drawdown (%)': metrics.get('max_drawdown', np.nan) * 100,
                'Win Rate (%)': metrics.get('win_rate', np.nan) * 100,
            }
            
            if 'profit_factor' in metrics:
                row['Profit Factor'] = metrics['profit_factor']
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Sort by Sharpe Ratio (descending)
        df = df.sort_values('Sharpe Ratio', ascending=False)
        
        return df
    
    def save_csv(self, filepath: str) -> None:
        """
        Save comparison table as CSV.
        
        Parameters
        ----------
        filepath : str
            Output file path.
        """
        df = self.generate_dataframe()
        df.to_csv(filepath, index=False, float_format='%.2f')
        logger.info(f"Saved comparison table to {filepath}")
    
    def save_markdown(self, filepath: str) -> None:
        """
        Save comparison table as Markdown.
        
        Parameters
        ----------
        filepath : str
            Output file path.
        """
        df = self.generate_dataframe()
        
        # Format DataFrame for Markdown
        markdown = "## Performance Comparison Table\n\n"
        markdown += df.to_markdown(index=False, floatfmt='.2f')
        markdown += "\n"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        logger.info(f"Saved comparison table to {filepath}")
    
    def save_latex(self, filepath: str, caption: str = "Performance Comparison") -> None:
        """
        Save comparison table as LaTeX.
        
        Parameters
        ----------
        filepath : str
            Output file path.
        caption : str, default="Performance Comparison"
            Table caption.
        """
        df = self.generate_dataframe()
        
        latex = df.to_latex(
            index=False,
            float_format='%.2f',
            caption=caption,
            label='tab:comparison'
        )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(latex)
        
        logger.info(f"Saved comparison table to {filepath}")
    
    def print_table(self) -> None:
        """Print comparison table to console."""
        df = self.generate_dataframe()
        print("\n" + "=" * 100)
        print("Performance Comparison Table")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100 + "\n")
